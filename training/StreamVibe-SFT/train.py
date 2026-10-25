#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Adaptive streaming SFT with global class-balanced decision loss and DeepSpeed."""

from __future__ import annotations

import argparse
import array
import contextlib
import datetime
import importlib.util
import json
import math
import os
import random
import shutil
import signal
import sys
import threading
import time
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import av
import deepspeed
import torch
import torch.distributed as dist
from PIL import Image
from qwen_vl_utils import process_vision_info
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoProcessor,
    get_cosine_schedule_with_warmup,
)

ROOT = Path(__file__).resolve().parents[1]
COLLATOR_PATH = Path(__file__).resolve().with_name("collator.py")
# Keep the streaming action prompt identical between training and inference.
STREAMING_SYSTEM_PROMPT = 'You are a real-time video streaming assistant observing a continuous camera feed frame by frame. The last frame represents the current moment.\n\n## Action Format\nAt every inference step you MUST choose exactly one of the following two actions:\n\n**Stay silent** — output ONLY:\n</silence>\nChoose this when nothing noteworthy has changed in the scene, no user query is pending, or there is nothing useful to say.\n\n**Speak** — output the token followed by a concise reply:\n</response> Your reply here.\nChoose this when you observe something worth reporting or a significant state change, or when you can answer a user question based on available evidence.'
TURN_BASED_SYSTEM_PROMPT = """You are a helpful video assistant.
Answer the user's questions using only the provided visual content.
Begin every answer with </response>.
Give a concise and directly relevant answer."""
# Kept for importers that use the historical constant.
SYSTEM_PROMPT = STREAMING_SYSTEM_PROMPT


def ensure_response_prefix(answer: str) -> str:
    """Add exactly one leading response token while preserving the answer text."""
    stripped = answer.lstrip()
    if stripped.startswith("</response>"):
        return stripped
    return "</response>" if not stripped else f"</response> {stripped}"


def load_collator():
    spec = importlib.util.spec_from_file_location(
        "streamvibe_collator", COLLATOR_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(COLLATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COLLATOR = load_collator()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def checkpoint_storage_guard(output_dir: Path, device: torch.device, rank: int) -> None:
    """Report NFS free space before ZeRO writes without stopping training."""
    del device
    if rank != 0:
        return
    try:
        free_bytes = shutil.disk_usage(output_dir).free
    except OSError as exc:
        print(
            json.dumps(
                {
                    "event": "checkpoint_storage_warning",
                    "free_bytes": None,
                    "error": str(exc),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return
    if free_bytes <= 1024**4:
        print(
            json.dumps(
                {
                    "event": "checkpoint_storage_warning",
                    "free_bytes": free_bytes,
                    "message": "free space is at or below 1 TiB; continuing by policy",
                },
                sort_keys=True,
            ),
            flush=True,
        )


def prune_step_checkpoints(checkpoint_root: Path, keep: int) -> None:
    """Keep only the newest step tags; never remove epoch checkpoints."""
    if keep < 1 or not checkpoint_root.exists():
        return
    step_dirs = sorted(
        (p for p in checkpoint_root.glob("step_*") if p.is_dir()),
        key=lambda p: p.name,
    )
    for old in step_dirs[:-keep]:
        shutil.rmtree(old, ignore_errors=False)


class ResumeSkipSampler(Sampler):
    """Fast-forward a resumed epoch without decoding what was already trained.

    Consuming and discarding batches costs a full video decode per skipped
    window, which grows with training progress -- a restart at update 1500
    would replay 4500 decodes per rank before the first step.  The epoch order
    is a seeded permutation and nothing in the data path draws from a global
    RNG, so dropping the first `skip` indices yields exactly the sample
    sequence the discarding loop would have left behind.
    """

    def __init__(self, base: DistributedSampler) -> None:
        self.base = base
        self.skip = 0

    def set_epoch(self, epoch: int) -> None:
        self.base.set_epoch(epoch)

    def __iter__(self):
        return iter(list(self.base)[self.skip :])

    def __len__(self) -> int:
        return max(0, len(self.base) - self.skip)


class ManifestDataset(Dataset):
    @staticmethod
    def index_paths(path: Path) -> tuple[Path, Path]:
        return (
            Path(str(path) + ".offsets.u64"),
            Path(str(path) + ".offsets.meta.json"),
        )

    @classmethod
    def ensure_index(cls, path: Path) -> tuple[Path, int]:
        index_path, meta_path = cls.index_paths(path)
        source_stat = path.stat()
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            rows = int(metadata["rows"])
            valid = (
                int(metadata["source_size"]) == source_stat.st_size
                and int(metadata["source_mtime_ns"]) == source_stat.st_mtime_ns
                and index_path.stat().st_size == rows * array.array("Q").itemsize
            )
            if valid:
                return index_path, rows
        except (
            FileNotFoundError,
            KeyError,
            ValueError,
            TypeError,
            json.JSONDecodeError,
        ):
            pass

        offsets = array.array("Q")
        with path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    offsets.append(offset)
        partial = Path(str(index_path) + f".{os.getpid()}.partial")
        with partial.open("wb") as handle:
            offsets.tofile(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, index_path)
        atomic_json(
            meta_path,
            {
                "schema": "streamvibe-jsonl-offsets",
                "source": str(path.resolve()),
                "source_size": source_stat.st_size,
                "source_mtime_ns": source_stat.st_mtime_ns,
                "rows": len(offsets),
            },
        )
        return index_path, len(offsets)

    def __init__(self, path: Path, max_samples: int = 0) -> None:
        self.path = path
        index_path, rows = self.ensure_index(path)
        self.offsets = array.array("Q")
        with index_path.open("rb") as handle:
            self.offsets.fromfile(handle, rows)
        if max_samples:
            self.offsets = self.offsets[:max_samples]
        self.handle = None

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.handle is None:
            self.handle = self.path.open("rb")
        self.handle.seek(self.offsets[index])
        return json.loads(self.handle.readline())


_UNDECODABLE_SEEN: set[str] = set()


def _record_undecodable(uri: str, exc: BaseException) -> None:
    """Append an undecodable medium to a side log, once per rank per uri.

    Written next to the checkpoints rather than raised, so the run survives;
    the file is the input to pruning these rows out of the manifest later.
    """
    if uri in _UNDECODABLE_SEEN:
        return
    _UNDECODABLE_SEEN.add(uri)
    target = os.environ.get("UNDECODABLE_LOG")
    if not target:
        return
    try:
        with open(target, "a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "uri": uri,
                        "error": f"{type(exc).__name__}: {exc}",
                        "rank": os.environ.get("RANK", "?"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    except OSError:
        pass


def decision_seconds(row: dict[str, Any]) -> list[int]:
    start = int(math.ceil(float(row["window_start_sec"])))
    end = int(math.floor(float(row["window_end_sec"])))
    if end < start:
        raise ValueError(f"empty decision range for {row['window_id']}")
    return list(range(start, end + 1))


@contextlib.contextmanager
def decode_deadline(seconds: int):
    """Bound a decode in wall-clock time, so a hang degrades into an error.

    `ignore_err` covers packets FFmpeg can skip and the FFmpegError path covers
    the ones it cannot, but neither covers a decode that simply never returns:
    the rank blocks inside libav, the other 159 sit in the collective waiting
    for it, no exception is raised and nothing crashes. That is what stalled the
    10% run at update 18 -- every rank alive, every GPU at 100%, zero progress
    until the supervisor's 900s watchdog tore it down.

    SIGALRM works here because the loader runs with num_workers=0, so decode
    happens on the main thread of the main process. A budget of a few minutes is
    far above a healthy decode (p95 is ~2s in the collator pre-flight) and far
    below the supervisor's stall threshold, so a poisoned row costs one grey
    micro-batch instead of the whole attempt.
    """
    if seconds <= 0 or threading.current_thread() is not threading.main_thread():
        yield
        return

    def on_alarm(signum, frame):  # noqa: ANN001 - signal handler signature
        raise TimeoutError(f"decode exceeded {seconds}s budget")

    previous = signal.signal(signal.SIGALRM, on_alarm)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


DECODE_BUDGET_S = int(os.environ.get("MIX_DECODE_BUDGET_S", "180"))


def decode_one_fps(row: dict[str, Any]) -> list[Any]:
    targets = decision_seconds(row)
    path = str(row["resolved_uri"])
    frames: list[Any] = []
    # A small number of otherwise ffprobe-valid public videos contain a
    # damaged packet late in the stream.  FFmpeg's ignore_err mode skips that
    # packet; the existing last-frame padding then preserves the 1-FPS timing
    # grid instead of aborting all distributed ranks.
    with av.open(path, mode="r", options={"err_detect": "ignore_err"}) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        start = max(0.0, targets[0] - 1.0)
        try:
            container.seek(
                int(start / float(stream.time_base)), stream=stream, backward=True
            )
        except (ValueError, OverflowError):
            container.seek(0, stream=stream, backward=True)
        target_index = 0
        last_image = None
        try:
            for frame in container.decode(stream):
                timestamp = float(frame.time or 0.0)
                if timestamp + 0.050 < targets[0]:
                    last_image = frame.to_image()
                    continue
                while (
                    target_index < len(targets)
                    and timestamp + 0.050 >= targets[target_index]
                ):
                    frames.append(frame.to_image())
                    last_image = frames[-1]
                    target_index += 1
                if target_index >= len(targets):
                    break
        except av.error.FFmpegError:
            if last_image is None and not frames:
                raise
        if last_image is None and frames:
            last_image = frames[-1]
        while target_index < len(targets) and last_image is not None:
            frames.append(last_image.copy())
            target_index += 1
    if len(frames) != len(targets):
        raise ValueError(
            f"decoded {len(frames)}/{len(targets)} 1-FPS frames for {row['window_id']}"
        )
    return frames


class VideoWindowCollator(COLLATOR.StreamVibeCollator):
    def __init__(
        self,
        *args: Any,
        max_total_pixels: int,
        turn_based_system_prompt: str | None = None,
        turn_response_prefix: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.max_total_pixels = max_total_pixels
        self.turn_based_system_prompt = (
            self.system_prompt
            if turn_based_system_prompt is None
            else turn_based_system_prompt
        )
        self.turn_response_prefix = turn_response_prefix

    # Streaming windows are kinds 0-2 and carry a per-second decision; a
    # turn-based row is kind 3 and carries an ordinary conversation. The two
    # share a model, a processor and a loss function, but nothing else: a
    # turn-based row has no window bounds, no 1-FPS grid and no control token,
    # so it must not be pushed through decode_one_fps or the class-balanced
    # weighting.
    TURN_BASED_KIND = 3

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if len(rows) != 1:
            raise ValueError("micro batch size must be 1")
        row = rows[0]
        if str(row.get("row_kind", "streaming")) == "turn_based":
            return self._turn_based_batch(row)
        start_time = time.monotonic()
        seconds = decision_seconds(row)
        try:
            with decode_deadline(DECODE_BUDGET_S):
                frames = decode_one_fps(row)
        except Exception as exc:  # noqa: BLE001 - see the turn-based path
            # Same contract as the turn-based branch: every rank must emit a
            # batch on every step, so an undecodable or hanging window degrades
            # into grey frames rather than taking the other 159 ranks with it.
            # The grid length is preserved so the per-second control tokens and
            # the class-balanced weighting still line up.
            _record_undecodable(str(row["resolved_uri"]), exc)
            frames = [
                Image.new("RGB", (112, 112), (128, 128, 128)) for _ in seconds
            ]
        decode_seconds = time.monotonic() - start_time
        questions = sorted(row.get("question", []), key=lambda x: float(x["time"]))
        responses = sorted(row.get("responses", []), key=lambda x: float(x["time"]))
        question_by_second: dict[int, list[str]] = {}
        for question in questions:
            timestamp = int(
                round(max(float(question["time"]), float(row["window_start_sec"])))
            )
            timestamp = min(max(timestamp, seconds[0]), seconds[-1])
            question_by_second.setdefault(timestamp, []).append(
                str(question["content"])
            )
        response_by_second: dict[int, list[str]] = {}
        for response in responses:
            timestamp = int(round(float(response["time"])))
            # Manifest construction already restricted responses to this
            # window. Clamp fractional boundary annotations to the nearest
            # available 1-FPS slot so a positive row cannot silently become
            # all-silence (common for a response at a fractional EOF).
            timestamp = min(max(timestamp, seconds[0]), seconds[-1])
            response_by_second.setdefault(timestamp, []).append(
                str(response["content"])
            )
        if (
            row["window_type"] == "response_centered_positive"
            and not response_by_second
        ):
            raise ValueError(
                f"positive window has no response slot: {row['window_id']}"
            )

        # A fixed per-frame ceiling makes activation memory grow linearly with
        # windows up to 200 seconds.  Keep short clips crisp while bounding the
        # aggregate visual token budget for medium/long streaming windows.
        per_frame_max_pixels = max(
            self.min_pixels,
            min(self.max_pixels, self.max_total_pixels // max(1, len(frames))),
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}
        ]
        for timestamp, image in zip(seconds, frames):
            prompt_lines = question_by_second.get(timestamp, [])
            text = (
                "\n".join(prompt_lines) + "\n" if prompt_lines else ""
            ) + f"<{timestamp:.1f} seconds>\n"
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {
                            "type": "image",
                            "image": image,
                            "max_pixels": per_frame_max_pixels,
                            "min_pixels": self.min_pixels,
                        },
                    ],
                }
            )
            answer = response_by_second.get(timestamp)
            messages.append(
                {
                    "role": "assistant",
                    "content": "</response> " + " ".join(answer)
                    if answer
                    else "</silence>",
                }
            )
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        image_inputs, video_inputs = process_vision_info(
            messages,
            image_patch_size=int(self.processor.image_processor.patch_size),
        )
        batch = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            do_resize=False,
        )
        if int(batch["input_ids"].shape[1]) > self.max_length:
            raise ValueError(
                f"sequence for {row['window_id']} exceeds max length: "
                f"{batch['input_ids'].shape[1]} > {self.max_length}"
            )
        labels, weights = self._labels_and_weights(batch["input_ids"][0])
        batch["labels"] = labels.unsqueeze(0)
        batch["loss_weights"] = weights.unsqueeze(0)
        batch["decode_seconds"] = torch.tensor([decode_seconds], dtype=torch.float64)
        batch["window_kind"] = torch.tensor(
            [
                {
                    "response_centered_positive": 0,
                    "near_trigger_hard_negative": 1,
                    "pure_silence": 2,
                }[row["window_type"]]
            ],
            dtype=torch.int64,
        )
        return dict(batch)

    def _turn_based_frames(self, path: str) -> list[Any]:
        """Uniformly sample frames from a turn-based clip.

        Streaming rows decode a 1-FPS grid because their supervision is
        per-second. A turn-based row asks one question about the whole clip, so
        the grid is meaningless and a fixed frame count is what the annotation
        assumes. The count is bounded by the same per-window pixel budget the
        streaming path uses, so both row kinds cost roughly the same activation
        memory and the batch topology stays predictable.
        """
        frame_cap = max(1, self.max_total_pixels // self.max_pixels)
        frames: list[Any] = []
        # A truncated or corrupt clip raises partway through the decode loop.
        # Keep the frames that came out before the bad packet instead of losing
        # the row: a clip that yields even a few frames is still a usable
        # sample, and the alternative is killing 144 ranks over one file.
        with av.open(path, mode="r", options={"err_detect": "ignore_err"}) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            total = stream.frames or 0
            try:
                if total > 0:
                    wanted = {
                        int(round(i * (total - 1) / max(1, frame_cap - 1)))
                        for i in range(min(frame_cap, total))
                    }
                    for index, frame in enumerate(container.decode(stream)):
                        if index in wanted:
                            frames.append(frame.to_image())
                        if len(frames) >= frame_cap:
                            break
                else:
                    # Some clips report no frame count; take the first frame_cap.
                    for frame in container.decode(stream):
                        frames.append(frame.to_image())
                        if len(frames) >= frame_cap:
                            break
            except Exception:
                if not frames:
                    raise
        if not frames:
            raise ValueError(f"decoded no frames for {path}")
        return frames

    def _turn_based_batch(self, row: dict[str, Any]) -> dict[str, torch.Tensor]:
        start_time = time.monotonic()
        media_type = str(row.get("media_type", "video"))
        uri = str(row["resolved_uri"])
        try:
            if media_type == "video":
                media = self._turn_based_frames(uri)
            else:
                media = [Image.open(uri).convert("RGB")]
        except Exception as exc:
            # Every rank must emit a batch on every step or the collective
            # desyncs, so an undecodable file cannot abort the sample -- it has
            # to degrade into one. The 800k verify pass probes media with
            # ffprobe, which accepts files that then fail a full decode here,
            # so this path is reachable no matter how clean the manifest looked.
            # One grey frame keeps the topology intact; the row contributes
            # noise for one micro-batch out of the global 576 and the uri is
            # recorded so it can be pruned from the manifest afterwards.
            _record_undecodable(uri, exc)
            media = [Image.new("RGB", (112, 112), (128, 128, 128))]
        decode_seconds = time.monotonic() - start_time

        per_frame_max_pixels = max(
            self.min_pixels,
            min(self.max_pixels, self.max_total_pixels // max(1, len(media))),
        )
        turns = list(row["conversations"])
        if not turns:
            raise ValueError(
                f"turn-based row has no conversation: {row.get('sample_id')}"
            )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.turn_based_system_prompt}
        ]
        attached = False
        for turn in turns:
            speaker = str(turn.get("from", "")).lower()
            text = str(turn.get("value", ""))
            if speaker in ("human", "user"):
                content: list[dict[str, Any]] = []
                if not attached:
                    # The annotation marks the media slot with <image>; the slot
                    # is rebuilt structurally here, so strip the literal token
                    # rather than leaving it to be tokenised as text.
                    for image in media:
                        content.append(
                            {
                                "type": "image",
                                "image": image,
                                "max_pixels": per_frame_max_pixels,
                                "min_pixels": self.min_pixels,
                            }
                        )
                    attached = True
                cleaned = text.replace("<image>", "").replace("<video>", "").strip()
                content.append({"type": "text", "text": cleaned})
                messages.append({"role": "user", "content": content})
            elif speaker in ("assistant", "gpt"):
                answer = ensure_response_prefix(text) if self.turn_response_prefix else text
                messages.append({"role": "assistant", "content": answer})
            else:
                raise ValueError(
                    f"unknown turn speaker {speaker!r}: {row.get('sample_id')}"
                )

        if messages[-1]["role"] != "assistant":
            raise ValueError(
                f"turn-based row does not end on an assistant turn: {row.get('sample_id')}"
            )
        if self.turn_response_prefix:
            rendered_answers = [
                str(message["content"])
                for message in messages
                if message["role"] == "assistant"
            ]
            if not rendered_answers or any(
                not answer.startswith("</response>") for answer in rendered_answers
            ):
                raise ValueError(
                    f"turn response prefix invariant failed: {row.get('sample_id')}"
                )
            if "</silence>" in self.turn_based_system_prompt or any(
                "</silence>" in answer for answer in rendered_answers
            ):
                raise ValueError(
                    f"turn protocol contains </silence>: {row.get('sample_id')}"
                )

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        image_inputs, video_inputs = process_vision_info(
            messages,
            image_patch_size=int(self.processor.image_processor.patch_size),
        )
        batch = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            do_resize=False,
        )
        if int(batch["input_ids"].shape[1]) > self.max_length:
            raise ValueError(
                f"sequence for {row.get('sample_id')} exceeds max length: "
                f"{batch['input_ids'].shape[1]} > {self.max_length}"
            )
        # Same assistant-span masking as the streaming path. The weights it
        # returns are ignored: global_cb_weights rebuilds them, and for a
        # turn-based row it assigns a flat weight because there is no control
        # token to balance.
        labels, weights = self._labels_and_weights(batch["input_ids"][0])
        batch["labels"] = labels.unsqueeze(0)
        batch["loss_weights"] = weights.unsqueeze(0)
        batch["decode_seconds"] = torch.tensor([decode_seconds], dtype=torch.float64)
        batch["window_kind"] = torch.tensor([self.TURN_BASED_KIND], dtype=torch.int64)
        return dict(batch)


def global_cb_weights(
    batches: list[dict[str, torch.Tensor]],
    kinds: list[int],
    silence_id: int,
    response_id: int,
    beta: float,
    lambda_text: float,
    device: torch.device,
    turn_based_kind: int = 3,
    turn_token_weight: float | None = None,
) -> tuple[list[torch.Tensor], dict[str, float | int]]:
    """Class-balanced weights over the streaming rows of a global batch.

    Turn-based rows are always excluded from the streaming class count. In the
    turn-based mode, turn_token_weight=1 gives the response prefix and answer
    text plain cross-entropy weights; None preserves the existing weighting.
    """
    streaming = [
        batch for batch, kind in zip(batches, kinds) if kind != turn_based_kind
    ]
    local_silence = 0
    local_response = 0
    for batch in streaming:
        labels = batch["labels"]
        local_silence += int(labels.eq(silence_id).sum())
        local_response += int(labels.eq(response_id).sum())
    counts = torch.tensor(
        [local_silence, local_response], device=device, dtype=torch.float64
    )
    local_streaming = torch.tensor([len(streaming)], device=device, dtype=torch.float64)
    dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_streaming, op=dist.ReduceOp.SUM)
    global_streaming = int(local_streaming.item())

    if global_streaming == 0:
        # An all-turn-based global batch is legitimate under a mixed manifest.
        # There is nothing to balance, so every supervised token is weighted 1.0
        # and the update behaves as plain cross-entropy.
        outputs = []
        for batch in batches:
            labels = batch["labels"]
            weights = torch.zeros_like(batch["loss_weights"], dtype=torch.float32)
            weights[labels.ne(-100)] = 1.0
            outputs.append(weights)
        return outputs, {
            "n_silence_global": 0,
            "n_response_global": 0,
            "w_silence": 0.0,
            "w_response": 0.0,
            "beta": beta,
            "lambda_text": lambda_text,
            "n_streaming_global": 0,
        }

    if torch.any(counts <= 0):
        raise RuntimeError(
            f"global optimizer batch lacks a decision class: silence={counts[0].item()} "
            f"response={counts[1].item()} over {global_streaming} streaming rows"
        )
    effective = (
        1.0 - torch.pow(torch.tensor(beta, device=device, dtype=torch.float64), counts)
    ) / (1.0 - beta)
    inverse = effective.reciprocal()
    class_weights = inverse / inverse.sum() * 2.0
    outputs: list[torch.Tensor] = []
    for batch, kind in zip(batches, kinds):
        labels = batch["labels"]
        weights = torch.zeros_like(batch["loss_weights"], dtype=torch.float32)
        valid = labels.ne(-100)
        if kind == turn_based_kind and turn_token_weight is not None:
            weights[valid] = float(turn_token_weight)
            outputs.append(weights)
            continue
        silence = labels.eq(silence_id)
        response = labels.eq(response_id)
        weights[silence] = class_weights[0].to(torch.float32)
        weights[response] = class_weights[1].to(torch.float32)
        weights[valid & ~silence & ~response] = float(lambda_text)
        outputs.append(weights)
    return outputs, {
        "n_silence_global": int(counts[0].item()),
        "n_response_global": int(counts[1].item()),
        "w_silence": float(class_weights[0].item()),
        "w_response": float(class_weights[1].item()),
        "beta": beta,
        "lambda_text": lambda_text,
        "n_streaming_global": global_streaming,
    }


def global_group_token_counts(
    batches: list[dict[str, torch.Tensor]],
    kinds: list[int],
    device: torch.device,
    turn_based_kind: int = 3,
) -> torch.Tensor:
    """Return turn/stream supervised-token counts for the optimizer batch."""
    local = torch.zeros(2, device=device, dtype=torch.float64)
    for batch, kind in zip(batches, kinds):
        # sparse_prediction_targets predicts labels[:, 1:], so the grouped
        # denominator must use exactly the same shifted target positions.
        count = batch["labels"][:, 1:].ne(-100).sum().to(torch.float64)
        local[0 if kind == turn_based_kind else 1] += count
    dist.all_reduce(local, op=dist.ReduceOp.SUM)
    return local


def freeze_vit_train_mergers(model: torch.nn.Module) -> dict[str, int]:
    model.requires_grad_(True)
    visual = model.model.visual
    for name, parameter in visual.named_parameters():
        parameter.requires_grad = name.startswith("merger.") or name.startswith(
            "deepstack_merger_list."
        )
    counts = Counter()
    for name, parameter in model.named_parameters():
        group = "trainable" if parameter.requires_grad else "frozen"
        counts[group] += parameter.numel()
        if parameter.requires_grad and name.startswith("model.visual"):
            counts["trainable_connector"] += parameter.numel()
        if parameter.requires_grad and (
            name.startswith("model.language_model") or name.startswith("lm_head")
        ):
            counts["trainable_llm"] += parameter.numel()
    return dict(counts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--deepspeed-config", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--lambda-text", type=float, default=2.0)
    parser.add_argument(
        "--turn-prompt-mode", choices=("shared", "qa_response"), default="shared"
    )
    parser.add_argument(
        "--loss-mode", choices=("shared", "turn_stream_balanced"), default="shared"
    )
    parser.add_argument("--turn-loss-ratio", type=float, default=0.5)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--max-pixels", type=int, default=65536)
    parser.add_argument("--max-total-pixels", type=int, default=2097152)
    parser.add_argument("--min-pixels", type=int, default=3136)
    parser.add_argument("--max-length", type=int, default=65536)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=3)
    parser.add_argument("--expected-world-size", type=int, default=160)
    parser.add_argument("--num-nodes", type=int, default=20)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--save-every-updates", type=int, default=200)
    parser.add_argument("--log-every-updates", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-updates", type=int, default=0)
    parser.add_argument("--scheduler-total-updates", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-step-checkpoints", type=int, default=2)
    parser.add_argument("--local_rank", type=int, default=-1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 < args.turn_loss_ratio < 1.0:
        raise ValueError("turn-loss-ratio must be strictly between 0 and 1")
    # These were originally locked to the first production recipe
    # (beta=0.9996, lr=1e-5). Experiments now pass both values explicitly, so
    # validate their mathematical domains here and record the exact values in
    # run_metadata instead of silently constraining every run to the baseline.
    if not 0.0 < args.beta < 1.0:
        raise ValueError(f"beta must be in (0, 1), got {args.beta}")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0.0:
        raise ValueError(f"learning rate must be finite and positive, got {args.learning_rate}")
    if not math.isfinite(args.lambda_text) or args.lambda_text <= 0.0:
        raise ValueError(f"lambda-text must be finite and positive, got {args.lambda_text}")
    if not args.train_manifest.is_file() or not args.eval_manifest.is_file():
        raise FileNotFoundError(
            f"missing train/eval manifest: {args.train_manifest} / {args.eval_manifest}"
        )


    dist_timeout = int(os.environ.get("MIX_DIST_TIMEOUT_S", "600"))
    deepspeed.init_distributed(
        dist_backend="nccl", timeout=datetime.timedelta(seconds=dist_timeout)
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if args.expected_world_size != args.num_nodes * args.gpus_per_node:
        raise ValueError("expected-world-size must equal num-nodes * gpus-per-node")
    if world_size != args.expected_world_size:
        raise ValueError(f"expected {args.expected_world_size} ranks, got {world_size}")
    if world_size == 160 and args.gradient_accumulation_steps != 3:
        raise ValueError(
            "20x8 final topology is locked to grad_accum=3 / global batch 480"
        )
    random.seed(20260816 + rank)
    torch.manual_seed(20260816 + rank)

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    processor.tokenizer.add_special_tokens(
        {"additional_special_tokens": ["</silence>", "</response>"]}
    )
    silence_id = processor.tokenizer.convert_tokens_to_ids("</silence>")
    response_id = processor.tokenizer.convert_tokens_to_ids("</response>")
    for token, token_id in (("</silence>", silence_id), ("</response>", response_id)):
        encoded = processor.tokenizer.encode(token, add_special_tokens=False)
        if encoded != [token_id]:
            raise ValueError(
                f"control token is not atomic: {token} -> {encoded}, id={token_id}"
            )

    if rank == 0:
        index_path, indexed_rows = ManifestDataset.ensure_index(args.train_manifest)
        print(
            json.dumps(
                {
                    "event": "manifest_index_ready",
                    "path": str(index_path),
                    "rows": indexed_rows,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.barrier()
    train_dataset = ManifestDataset(args.train_manifest, args.max_samples)
    sampler = ResumeSkipSampler(
        DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=20260816,
            drop_last=True,
        )
    )
    collator = VideoWindowCollator(
        processor,
        args.max_pixels,
        args.min_pixels,
        args.max_length,
        max_total_pixels=args.max_total_pixels,
        response_onset_weight=1.0,
        cb_stream_loss=True,
        cb_pure=True,
        system_prompt=STREAMING_SYSTEM_PROMPT,
        turn_based_system_prompt=(
            TURN_BASED_SYSTEM_PROMPT
            if args.turn_prompt_mode == "qa_response"
            else STREAMING_SYSTEM_PROMPT
        ),
        turn_response_prefix=args.turn_prompt_mode == "qa_response",
    )
    loader = DataLoader(
        train_dataset,
        batch_size=1,
        sampler=sampler,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    # Dispatch on the checkpoint's own architecture instead of hard-coding
    # Qwen3-VL. Pointing this script at Qwen3.5-9B with the Qwen3-VL class does
    # not fail loudly: transformers builds a Qwen3-VL skeleton, finds that
    # Qwen3.5's gated attention makes q_proj [8192, 4096] against the skeleton's
    # [4096, 4096], and reinitialises eight layers before raising on
    # ignore_mismatched_sizes. AutoModelForImageTextToText resolves qwen3_vl and
    # qwen3_5 alike, so the same script serves both base models.
    model_arch = (AutoConfig.from_pretrained(args.model).architectures or ["?"])[0]
    if rank == 0:
        print(f"loading {model_arch} from {args.model}", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        local_files_only=True,
    )
    required_vocab = max(silence_id, response_id) + 1
    if required_vocab > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(required_vocab)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    # from_pretrained leaves the model in eval mode (modeling_utils.py:4350) and
    # DeepSpeed's engine never flips it, so without this line the checkpointing
    # enabled just above is a no-op: transformers guards it with
    # `if self.gradient_checkpointing and self.training` (modeling_layers.py:80).
    # This loop is self-contained -- it borrows the collator and loss from
    # collator.py but not its train() call -- so the flag has to be
    # set here. Every run this script has driven so far paid full activation
    # memory while its config said gradient_checkpointing: true, which is the
    # likely reason the 5K smoke OOMed twice before micro-batch 1 stuck.
    model.train()
    parameter_counts = freeze_vit_train_mergers(model)
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = len(loader) // args.gradient_accumulation_steps
    natural_total_updates = max(1, updates_per_epoch * args.epochs)
    total_updates = args.scheduler_total_updates or natural_total_updates
    if total_updates < max(1, args.max_updates):
        raise ValueError(
            f"scheduler total {total_updates} is below max update {args.max_updates}"
        )
    warmup_updates = int(total_updates * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_updates, total_updates
    )
    ds_config = json.loads(args.deepspeed_config.read_text(encoding="utf-8"))
    expected_batch = world_size * args.gradient_accumulation_steps
    if int(ds_config["train_batch_size"]) != expected_batch:
        raise ValueError(
            f"DeepSpeed global batch mismatch: {ds_config['train_batch_size']} != {expected_batch}"
        )
    engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=trainable,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config=ds_config,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        processor.save_pretrained(args.output_dir / "tokenizer_with_control_tokens")
        metadata = {
            "beta": args.beta,
            "lambda_text": args.lambda_text,
            "turn_prompt_mode": args.turn_prompt_mode,
            "loss_mode": args.loss_mode,
            "turn_loss_ratio": args.turn_loss_ratio,
            "stream_loss_ratio": 1.0 - args.turn_loss_ratio,
            "max_updates": args.max_updates,
            "learning_rate": args.learning_rate,
            "num_nodes": args.num_nodes,
            "gpus_per_node": args.gpus_per_node,
            "world_size": world_size,
            "micro_batch_size_per_gpu": 1,
            "max_pixels_per_frame": args.max_pixels,
            "max_total_pixels_per_window": args.max_total_pixels,
            "num_workers_per_rank": args.num_workers,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_global_batch_size": expected_batch,
            "epochs": args.epochs,
            "warmup_updates": warmup_updates,
            "total_updates": total_updates,
            "natural_total_updates": natural_total_updates,
            "freeze_vision_encoder": True,
            "train_connector": True,
            "train_llm": True,
            "parameter_counts": parameter_counts,
            "train_manifest": str(args.train_manifest.resolve()),
            "eval_manifest": str(args.eval_manifest.resolve()),
        }
        atomic_json(args.output_dir / "run_metadata.json", metadata)
        print(
            json.dumps({"event": "run_start", **metadata}, sort_keys=True), flush=True
        )
    dist.barrier()

    client_state: dict[str, Any] = {}
    start_epoch = 0
    global_update = 0
    if args.resume:
        load_path, client_state = engine.load_checkpoint(
            str(args.output_dir / "checkpoints")
        )
        if load_path:
            start_epoch = int(client_state.get("epoch", 0))
            global_update = int(client_state.get("global_update", 0))
        # Do not let the uploader reclaim the transiently restored checkpoint
        # until all ranks have finished reading optimizer/model state.
        dist.barrier()
        if rank == 0:
            resume_lock = os.environ.get("STREAMVIBE_RESUME_LOCK")
            if resume_lock:
                Path(resume_lock).unlink(missing_ok=True)
            print(
                json.dumps(
                    {
                        "event": "resume_checkpoint_loaded",
                        "load_path": load_path,
                        "global_update": global_update,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        dist.barrier()
    metrics_path = args.output_dir / "training_metrics.jsonl"
    updates_per_epoch = max(1, len(loader) // args.gradient_accumulation_steps)
    stopped_at_update_limit = False
    current_epoch = start_epoch
    for epoch in range(start_epoch, args.epochs):
        current_epoch = epoch
        sampler.set_epoch(epoch)
        micro_group = 0
        # Resume from the exact data position represented by the checkpoint.
        # DeepSpeed restores optimizer/scheduler state, while the sampler is
        # reconstructed, so drop the already-trained windows from this epoch's
        # index order before the loader materialises them.  Slicing the sampler
        # is equivalent to pulling those batches and discarding them, but skips
        # the video decode each one would have cost -- a replay that grows with
        # progress and dominated the restart after the update-665 NCCL failure.
        skip_updates = (
            max(0, global_update - epoch * updates_per_epoch)
            if epoch == start_epoch
            else 0
        )
        sampler.skip = skip_updates * args.gradient_accumulation_steps
        iterator = iter(loader)
        while True:
            if args.max_updates and global_update >= args.max_updates:
                stopped_at_update_limit = True
                break
            cpu_batches: list[dict[str, torch.Tensor]] = []
            try:
                for _ in range(args.gradient_accumulation_steps):
                    cpu_batches.append(next(iterator))
            except StopIteration:
                break
            device_batches: list[dict[str, torch.Tensor]] = []
            batch_kinds: list[int] = []
            decode_wait = 0.0
            window_kinds = Counter()
            for batch in cpu_batches:
                decode_wait += float(batch.pop("decode_seconds").item())
                kind = int(batch.pop("window_kind").item())
                window_kinds[kind] += 1
                batch_kinds.append(kind)
                device_batches.append(
                    {
                        key: value.to(device, non_blocking=True)
                        for key, value in batch.items()
                    }
                )
            cb_weights, diagnostics = global_cb_weights(
                device_batches,
                batch_kinds,
                silence_id,
                response_id,
                args.beta,
                args.lambda_text,
                device,
                turn_based_kind=VideoWindowCollator.TURN_BASED_KIND,
                turn_token_weight=(
                    1.0 if args.turn_prompt_mode == "qa_response" else None
                ),
            )
            group_denominators = global_group_token_counts(
                device_batches,
                batch_kinds,
                device,
                turn_based_kind=VideoWindowCollator.TURN_BASED_KIND,
            )
            if args.loss_mode == "turn_stream_balanced" and torch.any(
                group_denominators <= 0
            ):
                raise RuntimeError(
                    "optimizer batch lacks a loss group: "
                    f"turn_tokens={group_denominators[0].item()} "
                    f"stream_tokens={group_denominators[1].item()}"
                )
            # turn numerator/denominator, stream numerator/denominator,
            # stream control/text tokens, turn prefix correct/total.
            local_stats = torch.zeros(8, device=device, dtype=torch.float64)
            update_start = time.monotonic()
            for batch, weights, kind in zip(device_batches, cb_weights, batch_kinds):
                labels = batch.pop("labels")
                batch.pop("loss_weights")
                prediction_indices, target_labels, target_weights = (
                    COLLATOR.sparse_prediction_targets(labels, weights)
                )
                outputs = engine(
                    **batch, use_cache=False, logits_to_keep=prediction_indices
                )
                is_turn = kind == VideoWindowCollator.TURN_BASED_KIND
                loss_kwargs: dict[str, Any] = {}
                if args.loss_mode == "turn_stream_balanced":
                    group_index = 0 if is_turn else 1
                    group_ratio = (
                        args.turn_loss_ratio
                        if is_turn
                        else 1.0 - args.turn_loss_ratio
                    )
                    loss_kwargs = {
                        "precomputed_global_denominator": group_denominators[
                            group_index
                        ],
                        # DeepSpeed divides each microstep loss by GAS. Undo
                        # that division so the accumulated gradient is exactly
                        # alpha * global_numerator / global_denominator.
                        "loss_multiplier": (
                            group_ratio * args.gradient_accumulation_steps
                        ),
                    }
                loss, numerator, denominator, _ = COLLATOR.weighted_sparse_loss(
                    outputs.logits,
                    target_labels,
                    target_weights,
                    world_size,
                    "token_count",
                    **loss_kwargs,
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite loss at rank={rank} update={global_update + 1}"
                    )
                engine.backward(loss)
                engine.step()
                stats_offset = 0 if is_turn else 2
                local_stats[stats_offset] += numerator.to(torch.float64)
                local_stats[stats_offset + 1] += denominator.to(torch.float64)
                if is_turn:
                    prefix_mask = target_labels.eq(response_id)
                    local_stats[7] += prefix_mask.sum().to(torch.float64)
                    if torch.any(prefix_mask):
                        predictions = outputs.logits.argmax(dim=-1)
                        local_stats[6] += predictions.masked_select(prefix_mask).eq(
                            response_id
                        ).sum().to(torch.float64)
                else:
                    control = target_labels.eq(silence_id) | target_labels.eq(
                        response_id
                    )
                    local_stats[4] += control.sum().to(torch.float64)
                    local_stats[5] += (~control).sum().to(torch.float64)
                del outputs
            global_update += 1
            micro_group += 1
            stats = local_stats
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            if not torch.all(torch.isfinite(stats)) or stats[1] + stats[3] <= 0:
                raise FloatingPointError(
                    f"invalid global loss statistics at update={global_update}: {stats.tolist()}"
                )
            elapsed = time.monotonic() - update_start
            if rank == 0 and global_update % args.log_every_updates == 0:
                turn_loss = stats[0] / stats[1].clamp_min(1.0)
                stream_loss = stats[2] / stats[3].clamp_min(1.0)
                if args.loss_mode == "turn_stream_balanced":
                    turn_contribution = args.turn_loss_ratio * turn_loss
                    stream_contribution = (1.0 - args.turn_loss_ratio) * stream_loss
                    total_loss = turn_contribution + stream_contribution
                    observed_turn = turn_contribution / total_loss.clamp_min(1e-12)
                else:
                    total_loss = (stats[0] + stats[2]) / (
                        stats[1] + stats[3]
                    ).clamp_min(1.0)
                    observed_turn = stats[0] / (stats[0] + stats[2]).clamp_min(1e-12)
                metric = {
                    "epoch": epoch,
                    "update": global_update,
                    "loss_total": float(total_loss.item()),
                    "loss_turn": float(turn_loss.item()) if stats[1] > 0 else None,
                    "loss_stream": float(stream_loss.item()) if stats[3] > 0 else None,
                    "turn_supervised_tokens": int(stats[1].item()),
                    "stream_supervised_tokens": int(stats[3].item()),
                    "stream_control_tokens": int(stats[4].item()),
                    "stream_text_tokens": int(stats[5].item()),
                    "target_loss_mix_turn": (
                        args.turn_loss_ratio
                        if args.loss_mode == "turn_stream_balanced"
                        else None
                    ),
                    "target_loss_mix_stream": (
                        1.0 - args.turn_loss_ratio
                        if args.loss_mode == "turn_stream_balanced"
                        else None
                    ),
                    "loss_contribution_turn": float(observed_turn.item()),
                    "loss_contribution_stream": float((1.0 - observed_turn).item()),
                    "turn_response_prefix_accuracy": (
                        float((stats[6] / stats[7]).item()) if stats[7] > 0 else None
                    ),
                    "loss_control": None,
                    "loss_response_text": None,
                    **diagnostics,
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                    "samples_per_second": expected_batch / max(elapsed, 1e-9),
                    "video_decode_wait_ratio": decode_wait
                    / max(decode_wait + elapsed, 1e-9),
                    "window_kind_local": dict(window_kinds),
                }
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(metric, sort_keys=True) + "\n")
                print(
                    json.dumps({"event": "train_update", **metric}, sort_keys=True),
                    flush=True,
                )
            if args.save_every_updates and global_update % args.save_every_updates == 0:
                checkpoint_storage_guard(args.output_dir, device, rank)
                engine.save_checkpoint(
                    str(args.output_dir / "checkpoints"),
                    tag=f"step_{global_update:06d}",
                    client_state={
                        "epoch": epoch,
                        "global_update": global_update,
                        "beta": args.beta,
                        "learning_rate": args.learning_rate,
                    },
                )
                dist.barrier()
                if rank == 0:
                    prune_step_checkpoints(
                        args.output_dir / "checkpoints", args.keep_step_checkpoints
                    )
                dist.barrier()
        if stopped_at_update_limit:
            break
        checkpoint_storage_guard(args.output_dir, device, rank)
        engine.save_checkpoint(
            str(args.output_dir / "checkpoints"),
            tag=f"epoch_{epoch:02d}",
            client_state={
                "epoch": epoch + 1,
                "global_update": global_update,
                "beta": args.beta,
                "learning_rate": args.learning_rate,
            },
        )
    if stopped_at_update_limit:
        step_tag = f"step_{global_update:06d}"
        if not (args.output_dir / "checkpoints" / step_tag).exists():
            checkpoint_storage_guard(args.output_dir, device, rank)
            engine.save_checkpoint(
                str(args.output_dir / "checkpoints"),
                tag=step_tag,
                client_state={
                    "epoch": current_epoch,
                    "global_update": global_update,
                    "beta": args.beta,
                    "learning_rate": args.learning_rate,
                },
            )
        dist.barrier()
        if rank == 0:
            # A max-updates cap equal to the natural one-epoch endpoint is a
            # completed run, not a pause.  The cap is still useful because it
            # guarantees a final step_N tag for downstream evaluation.
            if global_update >= natural_total_updates:
                atomic_json(
                    args.output_dir / "training_complete.json",
                    {
                        "status": "complete",
                        "global_update": global_update,
                        "beta": args.beta,
                        "learning_rate": args.learning_rate,
                    },
                )
                (args.output_dir / "training_paused.json").unlink(missing_ok=True)
            else:
                atomic_json(
                    args.output_dir / "training_paused.json",
                    {
                        "status": "paused_at_update_limit",
                        "global_update": global_update,
                        "max_updates": args.max_updates,
                    },
                )
    elif rank == 0:
        atomic_json(
            args.output_dir / "training_complete.json",
            {
                "status": "complete",
                "global_update": global_update,
                "beta": args.beta,
                "learning_rate": args.learning_rate,
            },
        )
    dist.barrier()
    return 0


if __name__ == "__main__":
    sys.exit(main())
