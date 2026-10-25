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

"""LoRA SFT utilities for per-second streaming video annotations.

Annotations become alternating user/assistant turns, with supervision on
assistant tokens. Default control-token weights are 1.0 for first silence,
0.4 for repeated silence, and 1.5 for response onset.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from qwen_vl_utils import process_vision_info
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, get_cosine_schedule_with_warmup


SYSTEM_PROMPT = """You are a real-time video streaming assistant observing a continuous camera feed frame by frame. The last frame represents the current moment.
## Action Format
At every inference step you MUST choose exactly one of the following three actions:
**Stay silent** — output ONLY:
</silence>
Choose this when nothing noteworthy has changed in the scene, no user query is pending, or there is nothing useful to say.
**Speak** — output the token followed by a concise reply:
</response> Your reply here.
Choose this when you observe something worth reporting or a significant state change, or when you can answer a user question based on available evidence.

**Delegate** — when a question is too hard or error-prone to answer reliably yourself, speak a brief note that you're delegating, then hand the question to the background solver:
</response> Brief note that you're delegating. </delegation> <the question>"""

# Two-action prompt for training data without delegation targets.
# Selected with --no-delegation-prompt.
SYSTEM_PROMPT_NO_DELEGATION = (
    SYSTEM_PROMPT.split("\n\n**Delegate**")[0]
    # ...and stop announcing three actions when only two remain.
    .replace("one of the following three actions", "one of the following two actions")
)

SYSTEM_PROMPT_CLEAR_USEFUL = """You are a proactive assistant. Respond only when there is a clear and useful opportunity to help; otherwise, remain silent.

Base your decision only on visual evidence available up to the current frame. Do not use future information or make unsupported guesses.

Follow this response protocol:

Output </silence> if the current visual evidence is irrelevant or insufficient to provide reliable help.
Output </response> only when the available evidence is sufficient to provide a clear, useful, and complete response.

When responding, provide a concise and actionable answer. Every valid response must begin with </response>."""


# Optional prefix mapping for annotation paths from another storage location.
IMAGE_PATH_REMAP = (
    os.environ.get("STREAMVIBE_IMAGE_SOURCE_PREFIX", ""),
    os.environ.get("STREAMVIBE_IMAGE_TARGET_PREFIX", ""),
)


def remap_image_path(path: str) -> str:
    old, new = IMAGE_PATH_REMAP
    # This migration is deterministic. Checking the stale path and replacement
    # with stat() here performed two NFS metadata operations for every frame
    # (hundreds of thousands per rank) before training could start. Lazy sample
    # validation in JsonDataset.__getitem__ verifies the remapped directory.
    if old and new and old in path:
        return path.replace(old, new)
    return path


class JsonDataset(Dataset):
    def __init__(self, path: str, max_samples: int, seed: int,
                 strict: bool = False) -> None:
        root = Path(path)
        if root.is_dir():
            files = sorted(root.rglob("*.jsonl"))
            if not files:
                files = sorted(root.rglob("*.json"))
        else:
            files = [root]
        if not files:
            raise FileNotFoundError(f"no JSON/JSONL training files under {root}")

        self.rows: list[dict] | None = []
        self.records: list[tuple[Path, int]] = []
        self._handles: dict[Path, object] = {}
        self._runtime_dir_cache: dict[str, set[str]] = {}
        self.strict = strict
        # Every DDP rank used to repeat the same 200 MB JSONL scan plus ~7,700
        # NFS directory listings. At 16 ranks that makes startup look hung for
        # many minutes and puts all workers in rpc_wait/d_alloc_parallel. Cache
        # the structurally valid (file, byte-offset) index beside each stable
        # split. Missing-frame validation remains lazy in __getitem__, so the
        # ranks share that work through DistributedSampler instead of all
        # validating the entire corpus before model load. Source size/mtime
        # signatures invalidate the cache whenever a shard changes.
        cache_name = ".streamvibe_structural_index_strict.json" if strict \
            else ".streamvibe_structural_index.json"
        cache_path = root / cache_name if root.is_dir() else None
        sources = [
            {
                "path": str(file_path.resolve()),
                "size": file_path.stat().st_size,
                "mtime_ns": file_path.stat().st_mtime_ns,
            }
            for file_path in files
        ]
        self._length_cache_path = root / ".streamvibe_sequence_lengths.json" \
            if root.is_dir() else None
        self._source_signature = sources
        if cache_path is not None and all(file_path.suffix == ".jsonl" for file_path in files):
            try:
                with cache_path.open("r", encoding="utf-8") as handle:
                    cached = json.load(handle)
                if cached.get("sources") == sources:
                    self.rows = None
                    self.records = [
                        (Path(file_path), int(offset))
                        for file_path, offset in cached["records"]
                    ]
                    population = len(self.records)
                    if max_samples and max_samples < population:
                        rng = random.Random(seed)
                        indices = list(range(population))
                        rng.shuffle(indices)
                        self.records = [self.records[i] for i in indices[:max_samples]]
                    return
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                pass

        for file_path in files:
            if file_path.suffix == ".jsonl":
                self.rows = None
                with file_path.open("rb") as handle:
                    while True:
                        offset = handle.tell()
                        line = handle.readline()
                        if not line:
                            break
                        stripped = line.strip()
                        if not stripped:
                            continue


                        try:
                            rec = json.loads(stripped)
                        except Exception as exc:
                            if self.strict:
                                raise ValueError(
                                    f"malformed JSONL record at {file_path}:{offset}"
                                ) from exc
                            continue
                        images = [remap_image_path(x) for x in rec.get("images", [])]
                        rec["images"] = images
                        if self.strict and any(key in rec for key in ("cut_frame", "kind")):
                            raise ValueError(
                                f"cut/prefix field present at {file_path}:{offset}"
                            )
                        n_images = len(images)
                        n_tags = sum(
                            m.get("content", "").count("<image>")
                            for m in rec.get("messages", [])
                            if isinstance(m.get("content"), str)
                        )
                        if n_tags != n_images:
                            if self.strict:
                                raise ValueError(
                                    f"image-tag mismatch at {file_path}:{offset}: "
                                    f"tags={n_tags}, images={n_images}"
                                )
                            if n_tags > n_images:
                                continue
                        self.records.append((file_path, offset))
            else:
                if self.rows is None:
                    raise ValueError("do not mix JSON arrays and JSONL files")
                with file_path.open("r", encoding="utf-8") as handle:
                    value = json.load(handle)
                self.rows.extend(value if isinstance(value, list) else [value])

        if cache_path is not None and self.rows is None:
            payload = {
                "sources": sources,
                "records": [[str(file_path.resolve()), offset] for file_path, offset in self.records],
            }
            temp_path = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
            try:
                with temp_path.open("w", encoding="utf-8") as handle:
                    json.dump(payload, handle, ensure_ascii=False)
                os.replace(temp_path, cache_path)
            finally:
                try:
                    temp_path.unlink()
                except FileNotFoundError:
                    pass

        population = len(self.records) if self.rows is None else len(self.rows)
        if max_samples and max_samples < population:
            rng = random.Random(seed)
            indices = list(range(population))
            rng.shuffle(indices)
            indices = indices[:max_samples]
            if self.rows is None:
                self.records = [self.records[i] for i in indices]
            else:
                self.rows = [self.rows[i] for i in indices]

    def __len__(self) -> int:
        return len(self.records) if self.rows is None else len(self.rows)

    def sequence_lengths(self) -> list[int]:
        """Return image counts without resolving or decoding frame files."""
        if self._length_cache_path is not None:
            try:
                payload = json.loads(self._length_cache_path.read_text(encoding="utf-8"))
                cached = [int(value) for value in payload["lengths"]]
                if (payload.get("sources") == self._source_signature
                        and len(cached) == len(self)):
                    return cached
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                pass
        if self.rows is not None:
            lengths = [len(row.get("images", [])) for row in self.rows]
        else:
            lengths = []
            handles: dict[Path, object] = {}
            try:
                for path, offset in self.records:
                    handle = handles.get(path)
                    if handle is None:
                        handle = path.open("rb")
                        handles[path] = handle
                    handle.seek(offset)
                    row = json.loads(handle.readline())
                    lengths.append(len(row.get("images", [])))
            finally:
                for handle in handles.values():
                    handle.close()
        if self._length_cache_path is not None:
            payload = {"sources": self._source_signature, "lengths": lengths}
            temp = self._length_cache_path.with_name(
                f".{self._length_cache_path.name}.{os.getpid()}.tmp")
            try:
                temp.write_text(json.dumps(payload), encoding="utf-8")
                os.replace(temp, self._length_cache_path)
            finally:
                try:
                    temp.unlink()
                except FileNotFoundError:
                    pass
        return lengths

    def __getitem__(self, index: int) -> dict:
        population = len(self.records) if self.rows is None else len(self.rows)
        attempts = 1 if self.strict else population
        for attempt in range(attempts):
            candidate = (index + attempt) % population
            if self.rows is not None:
                row = self.rows[candidate].copy()
            else:
                path, offset = self.records[candidate]
                handle = self._handles.get(path)
                if handle is None:
                    handle = path.open("rb")
                    self._handles[path] = handle
                handle.seek(offset)
                row = json.loads(handle.readline())

            # JSONL-backed datasets retain only byte offsets, so remap again at
            # access time. Validate one directory listing per frame directory;
            # if the converter referenced a missing final frame, substitute the
            # next valid record rather than crashing every DDP rank mid-step.
            images = [remap_image_path(image) for image in row.get("images", [])]
            missing = False
            for image in images:
                image_dir = os.path.dirname(image)
                listing = self._runtime_dir_cache.get(image_dir)
                if listing is None:
                    try:
                        listing = set(os.listdir(image_dir))
                    except OSError:
                        listing = set()
                    self._runtime_dir_cache[image_dir] = listing
                if os.path.basename(image) not in listing:
                    missing = True
                    break
            if not missing:
                row["images"] = images
                return row
            if self.strict:
                raise FileNotFoundError(
                    f"strict dataset row {candidate} has missing frame: {image}"
                )
        raise RuntimeError("dataset contains no samples with complete frame files")


class LengthBucketDistributedSampler(Sampler[int]):
    """Group comparable rows in each global batch, then shuffle batch order.

    Each rank processes one row. A fully random sampler therefore makes all
    ranks wait for the longest unrelated video in almost every step. This
    sampler keeps the exact one-epoch population while reducing that DDP
    straggler waste. Both global-batch order and rank assignment are seeded.
    """

    def __init__(self, dataset: JsonDataset, num_replicas: int, rank: int,
                 seed: int) -> None:
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        self.num_samples = math.ceil(len(dataset) / num_replicas)
        self.total_size = self.num_samples * num_replicas
        self.lengths = dataset.sequence_lengths()

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        ordered = sorted(range(len(self.dataset)), key=lambda idx: (self.lengths[idx], idx))
        if len(ordered) < self.total_size:
            ordered.extend(ordered[: self.total_size - len(ordered)])
        batches = [ordered[start:start + self.num_replicas]
                   for start in range(0, self.total_size, self.num_replicas)]
        batch_order = torch.randperm(len(batches), generator=generator).tolist()
        flattened = []
        for batch_index in batch_order:
            batch = batches[batch_index]
            within = torch.randperm(len(batch), generator=generator).tolist()
            flattened.extend(batch[index] for index in within)
        indices = flattened[self.rank:self.total_size:self.num_replicas]
        if len(indices) != self.num_samples:
            raise AssertionError((len(indices), self.num_samples))
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


def setup_distributed() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if world_size > 1:
        # Longer timeout so a slow first step / kernel autotune / straggler rank
        # does not trip the default 10-min NCCL watchdog and abort the job.
        from datetime import timedelta
        kwargs = {"timeout": timedelta(minutes=30)}


        if os.environ.get("STREAMVIBE_LAZY_NCCL_INIT", "0") != "1":
            kwargs["device_id"] = torch.device(f"cuda:{local_rank}")
        dist.init_process_group("nccl", **kwargs)
    return rank, local_rank, world_size


def is_main(rank: int) -> bool:
    return rank == 0


def build_messages(row: dict, max_pixels: int, min_pixels: int,
                  system_prompt: str = SYSTEM_PROMPT) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    image_paths = iter(row["images"])
    used_images = 0

    for message in row["messages"]:
        if message["role"] == "assistant":
            messages.append({"role": "assistant", "content": message["content"]})
            continue

        pieces = message["content"].split("<image>")
        content: list[dict] = []
        for index, piece in enumerate(pieces):
            if piece:
                content.append({"type": "text", "text": piece})
            if index < len(pieces) - 1:
                try:
                    image_path = Path(next(image_paths)).resolve()
                except StopIteration as exc:
                    raise ValueError("more <image> tags than image paths") from exc
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
                content.append(
                    {
                        "type": "image",
                        "image": str(image_path),
                        "max_pixels": max_pixels,
                        "min_pixels": min_pixels,
                    }
                )
                used_images += 1
        messages.append({"role": "user", "content": content})

    try:
        next(image_paths)
    except StopIteration:
        pass
    else:
        raise ValueError("more image paths than <image> tags")
    if used_images != len(row["images"]):
        raise ValueError(f"used {used_images} images, expected {len(row['images'])}")
    return messages


class StreamVibeCollator:
    def __init__(self, processor, max_pixels: int, min_pixels: int, max_length: int,
                 response_onset_weight: float = 1.5,
                 soft_slot_weight: float | None = None,
                 pre_optimal_silence_weight: float | None = None,
                 post_optimal_silence_weight: float | None = None,
                 first_response_multiplier: float = 1.0,
                 cb_stream_loss: bool = False,
                 cb_pure: bool = False,
                 system_prompt: str = SYSTEM_PROMPT) -> None:
        self.processor = processor
        self.system_prompt = system_prompt
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_length = max_length
        # Defaults reproduce the original hardcoded schedule exactly, so the
        # Narration weights are unchanged unless a flag is passed.
        self.response_onset_weight = response_onset_weight
        self.soft_slot_weight = soft_slot_weight
        self.pre_optimal_silence_weight = pre_optimal_silence_weight
        self.post_optimal_silence_weight = post_optimal_silence_weight
        if soft_slot_weight is not None and (
            pre_optimal_silence_weight is not None or post_optimal_silence_weight is not None
        ):
            raise ValueError(
                "--soft-slot-weight cannot be combined with pre/post-optimal silence weights"
            )
        for name, value in (
            ("pre-optimal", pre_optimal_silence_weight),
            ("post-optimal", post_optimal_silence_weight),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name} silence weight must be non-negative")
        
        self.first_response_multiplier = first_response_multiplier
        self.cb_stream_loss = cb_stream_loss
        self.cb_pure = cb_pure
        tok = processor.tokenizer
        self.im_start_id = tok.convert_tokens_to_ids("<|im_start|>")
        self.im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
        self.silence_id = tok.convert_tokens_to_ids("</silence>")
        self.response_id = tok.convert_tokens_to_ids("</response>")
        self.assistant_header = [self.im_start_id] + tok.encode("assistant\n", add_special_tokens=False)

    @staticmethod
    def _find_subsequence(values: list[int], pattern: list[int], start: int) -> int:
        last = len(values) - len(pattern) + 1
        for index in range(start, last):
            if values[index : index + len(pattern)] == pattern:
                return index
        return -1

    def _labels_and_weights(self, input_ids: torch.Tensor,
                            soft_slots: list[int] | None = None,
                            pre_optimal_soft_slots: list[int] | None = None,
                            post_optimal_soft_slots: list[int] | None = None,
                            masked_assistant_slots: list[int] | None = None,
                            hard_silence_slots: list[int] | None = None,
                            action_only_assistant_slots: list[int] | None = None,
                            ) -> tuple[torch.Tensor, torch.Tensor]:
        """Supervise assistant spans and weight their control tokens.

        ``soft_slots`` lists assistant-turn indices that fall inside a
        gold help window but are not its optimal moment. The benchmark scores a
        hit anywhere in that window, so training them as full-strength silence
        punishes exactly what evaluation rewards — and does so on the slots
        adjacent to the positive, right on the decision boundary. When
        ``--soft-slot-weight`` is set, those spans are down-weighted instead.
        """
        soft = set(soft_slots) if (soft_slots and self.soft_slot_weight is not None) else set()
        pre = set(pre_optimal_soft_slots or [])
        post = set(post_optimal_soft_slots or [])
        masked = set(masked_assistant_slots or [])
        hard_silence = set(hard_silence_slots or [])
        action_only = set(action_only_assistant_slots or [])
        if pre & post:
            raise ValueError(f"assistant slots occur in both pre and post sets: {sorted(pre & post)}")
        if hard_silence & (soft | pre | post | masked):
            raise ValueError(
                "hard-silence slots overlap soft/masked slots: "
                f"{sorted(hard_silence & (soft | pre | post | masked))}"
            )
        ids = input_ids.tolist()
        labels = torch.full_like(input_ids, -100)
        weights = torch.zeros_like(input_ids, dtype=torch.float32)
        cursor = 0
        previous_action = None
        seen_response = False
        spans = 0
        while True:
            header = self._find_subsequence(ids, self.assistant_header, cursor)
            if header < 0:
                break
            begin = header + len(self.assistant_header)
            try:
                end = ids.index(self.im_end_id, begin)
            except ValueError as exc:
                raise ValueError("assistant turn has no <|im_end|>") from exc
            labels[begin : end + 1] = input_ids[begin : end + 1]
            weights[begin : end + 1] = 1.0
            action = ids[begin] if begin < end else None
            if action == self.silence_id:
                if self.cb_pure:
                    # Strict class-balanced streaming loss: the collator contributes no temporal
                    # prior.  Every decision token enters cb_stream_weights()
                    # at unit weight; only the dynamic class-balanced factor is
                    # allowed to distinguish silence from response.
                    pass
                elif spans in hard_silence:
                    # State-conditional data can mark a small number of audited,
                    # outside-all-gold-window negatives. Give the complete silence
                    # span normal unit weight instead of the 0.4 repeated-silence
                    # onset. Rows without this optional field remain bit-identical.
                    weights[begin : end + 1] = 1.0
                elif spans in soft:
                    weights[begin : end + 1] = self.soft_slot_weight
                elif spans in pre:
                    if self.pre_optimal_silence_weight is None:
                        raise ValueError("row has pre-optimal slots but no configured weight")
                    weights[begin : end + 1] = self.pre_optimal_silence_weight
                elif spans in post:
                    if self.post_optimal_silence_weight is None:
                        raise ValueError("row has post-optimal slots but no configured weight")
                    weights[begin : end + 1] = self.post_optimal_silence_weight
                else:
                    weights[begin] = 0.4 if previous_action == self.silence_id else 1.0
            elif action == self.response_id:
                if spans in soft or spans in pre or spans in post:
                    raise ValueError(f"response assistant slot {spans} is marked as silence-softened")
                # The first time a task speaks is much harder than the rest: on
                # the held-out 400, the first gold probe in a task is hit 13.0%
                # of the time versus 41.3% for the third, and the gap survives
                # bucketing by gold history (i.e. it is not an artefact of the
                # model's own output trajectory). A global onset weight lifts
                # both buckets proportionally and cannot close that gap, so the
                # onset token gets an extra multiplier until the task has spoken.
                if not self.cb_pure:
                    weight = self.response_onset_weight
                    if not seen_response:
                        weight *= self.first_response_multiplier
                    weights[begin] = weight
                seen_response = True
            previous_action = action
            # State-aware timeline chunks keep selected earlier turns as model
            # context. They must update previous_action / seen_response above,
            # but must never be supervised again: every original streaming frame is
            # assigned to exactly one chunk's unmasked region.
            if spans in masked:
                labels[begin : end + 1] = -100
                weights[begin : end + 1] = 0.0
            elif spans in action_only:
                # Calibration rows teach only the state-conditional action
                # boundary.  Keep the first control token as the sole target;
                # answer text and <|im_end|> remain context but carry no loss.
                labels[begin + 1 : end + 1] = -100
                weights[begin + 1 : end + 1] = 0.0
                # Equalise response/silence calibration cells independently of
                # the legacy onset schedule (1.5 response, 0.4 repeated silence).
                weights[begin] = 1.0
            cursor = end + 1
            spans += 1
        invalid_slots = (soft | pre | post | masked | hard_silence | action_only) - set(range(spans))
        if invalid_slots:
            raise ValueError(f"soft-slot indices exceed assistant span count: {sorted(invalid_slots)}")
        if spans == 0 or not torch.any(labels != -100):
            raise ValueError("failed to locate assistant spans")
        return labels, weights

    def __call__(self, rows: list[dict]) -> dict[str, torch.Tensor]:
        if len(rows) != 1:
            raise ValueError("this collator currently requires micro batch size 1")
        messages = build_messages(rows[0], self.max_pixels, self.min_pixels,
                                  self.system_prompt)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        image_inputs, video_inputs = process_vision_info(messages)
        batch = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        sequence_length = int(batch["input_ids"].shape[1])
        if sequence_length > self.max_length:
            raise ValueError(
                f"sequence length {sequence_length} exceeds --max-length {self.max_length}; "
                "reduce --max-pixels or increase the context limit"
            )
        row = rows[0]
        if self.cb_stream_loss and "row_loss_mass_target" in row:
            raise ValueError(
                "CB-Stream loss is incompatible with row_loss_mass_target; "
                "row-level mass normalisation would erase its cross-row class weights"
            )
        has_old_field = "soft_slots" in row
        has_new_field = "pre_optimal_soft_slots" in row or "post_optimal_soft_slots" in row
        new_configured = self.pre_optimal_silence_weight is not None \
            or self.post_optimal_silence_weight is not None
        if (has_old_field and (has_new_field or new_configured)) or \
                (has_new_field and self.soft_slot_weight is not None):
            raise ValueError("legacy soft_slots cannot be mixed with pre/post-window soft slots")
        labels, loss_weights = self._labels_and_weights(
            batch["input_ids"][0], row.get("soft_slots"),
            row.get("pre_optimal_soft_slots"), row.get("post_optimal_soft_slots"),
            row.get("masked_assistant_slots"), row.get("hard_silence_slots"),
            row.get("action_only_assistant_slots"))
        if "row_loss_mass_target" in row:
            target = float(row["row_loss_mass_target"])
            if not math.isfinite(target) or target <= 0:
                raise ValueError(f"invalid row_loss_mass_target: {target!r}")
            current = float(loss_weights.sum())
            if current <= 0:
                raise ValueError("cannot normalise a row with zero loss mass")
            loss_weights.mul_(target / current)
        batch["labels"] = labels.unsqueeze(0)
        batch["loss_weights"] = loss_weights.unsqueeze(0)
        # Optional validation-only routing for the state-balanced action
        # calibration pilot.  Ordinary rows carry -1 and remain bit-identical.
        # Keeping the id out of the text avoids changing the model input while
        # allowing evaluate() to report each action cell independently.
        calibration_groups = {
            "pre_first_inside_response": 0,
            "pre_first_outside_silence": 1,
            "post_first_inside_response": 2,
            "post_first_outside_silence": 3,
            "mixed_full_token_rehearsal": 4,
        }
        batch["calibration_group"] = torch.tensor(
            [calibration_groups.get(row.get("calibration_cell"), -1)],
            dtype=torch.int64,
        )
        return dict(batch)


CB_STREAM_HISTOGRAM_BOUNDS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)


def cb_stream_weights(labels: torch.Tensor, temporal_weights: torch.Tensor,
                      silence_id: int, response_id: int, beta: float,
                      lambda_text: float, world_size: int,
                      include_histogram: bool = False,
                      ) -> tuple[torch.Tensor, dict]:
    """Build global-batch class-balanced streaming loss token weights.

    Class-balanced weights apply only to the two decision tokens.  The existing
    pre/post-window and repeated-silence weights are retained as an independent
    temporal factor and globally mean-normalised over decision tokens.  Every
    other supervised token receives ``lambda_text``. Counts and temporal
    normalisation are global across DDP ranks because
    each rank holds only one timeline.
    """
    if labels.shape != temporal_weights.shape:
        raise ValueError(
            f"CB-Stream label/weight shape mismatch: {tuple(labels.shape)} "
            f"vs {tuple(temporal_weights.shape)}"
        )
    if not 0.0 <= beta < 1.0:
        raise ValueError(f"CB-Stream beta must be in [0, 1), got {beta!r}")
    if not math.isfinite(lambda_text) or lambda_text <= 0:
        raise ValueError(
            f"CB-Stream lambda_text must be finite and positive, got {lambda_text!r}"
        )

    silence = labels.eq(silence_id)
    response = labels.eq(response_id)
    action = silence | response
    valid = labels.ne(-100)
    local = torch.stack((
        silence.sum(),
        response.sum(),
        temporal_weights.masked_select(action).to(torch.float64).sum(),
        action.sum(),
    )).to(device=labels.device, dtype=torch.float64)
    global_stats = local.clone()
    if world_size > 1:
        dist.all_reduce(global_stats, op=dist.ReduceOp.SUM)

    raw_counts = global_stats[:2]
    safe_counts = raw_counts.clamp_min(1.0)
    beta_tensor = torch.tensor(beta, device=labels.device, dtype=torch.float64)
    if beta == 0.0:
        effective = torch.ones_like(safe_counts)
    else:
        effective = (1.0 - torch.pow(beta_tensor, safe_counts)) / (1.0 - beta_tensor)
    inverse_effective = effective.reciprocal()
    class_weights = inverse_effective / inverse_effective.sum() * 2.0

    action_count = global_stats[3]
    temporal_mean = global_stats[2] / action_count.clamp_min(1.0)
    if action_count.item() == 0:
        temporal_mean = torch.ones_like(temporal_mean)

    weights = torch.zeros_like(temporal_weights, dtype=torch.float32)
    text = valid & ~action
    weights[text] = float(lambda_text)
    weights[silence] = (
        temporal_weights[silence].to(torch.float64) / temporal_mean * class_weights[0]
    ).to(torch.float32)
    weights[response] = (
        temporal_weights[response].to(torch.float64) / temporal_mean * class_weights[1]
    ).to(torch.float32)

    diagnostics = {
        "global_n_silence": int(raw_counts[0].item()),
        "global_n_response": int(raw_counts[1].item()),
        "class_weight_silence": float(class_weights[0].item()),
        "class_weight_response": float(class_weights[1].item()),
        "class_weight_ratio_response_to_silence": float(
            (class_weights[1] / class_weights[0]).item()
        ),
        "temporal_factor_global_mean_before_normalization": float(temporal_mean.item()),
        "lambda_text": float(lambda_text),
    }
    if include_histogram:
        supervised = weights.masked_select(valid).to(torch.float64)
        boundaries = torch.tensor(
            CB_STREAM_HISTOGRAM_BOUNDS, device=labels.device, dtype=torch.float64
        )
        buckets = torch.bincount(
            torch.bucketize(supervised, boundaries), minlength=len(boundaries) + 1
        ).to(torch.float64)
        if world_size > 1:
            dist.all_reduce(buckets, op=dist.ReduceOp.SUM)
        diagnostics["final_weight_histogram"] = {
            "bounds": list(CB_STREAM_HISTOGRAM_BOUNDS),
            "counts": [int(value) for value in buckets.cpu().tolist()],
        }
    return weights, diagnostics


LOSS_CHUNK_TOKENS = int(os.environ.get("STREAMVIBE_LOSS_CHUNK_TOKENS", "8192"))


def _chunk_weighted_nll(chunk_logits, chunk_labels, chunk_weights):
    """Weighted NLL summed over one token chunk, computed in FP32.

    Kept as a standalone function so torch.utils.checkpoint can rerun it during
    backward: the FP32 upcast and the [chunk, vocab] softmax intermediates are
    freed after the forward pass and rebuilt one chunk at a time, so peak memory
    scales with LOSS_CHUNK_TOKENS instead of the full sequence length.
    """
    token_loss = F.cross_entropy(
        chunk_logits.float(),
        chunk_labels,
        reduction="none",
        ignore_index=-100,
    )
    # cross_entropy already returns 0 for ignored positions; multiplying by the
    # validity mask keeps that explicit and guards the weight product.
    return (token_loss * chunk_weights * chunk_labels.ne(-100)).sum()


def weighted_loss(logits, labels, weights, world_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Shift via narrow views. Calling .contiguous() on the full [1, T, vocab]
    # logits would itself copy a multi-GiB tensor before any loss math runs.
    shift_logits = logits[:, :-1]
    shift_labels = labels[:, 1:]
    shift_weights = weights[:, 1:]
    valid = shift_labels.ne(-100)

    flat_logits = shift_logits.reshape(-1, shift_logits.shape[-1])
    flat_labels = shift_labels.reshape(-1)
    flat_weights = shift_weights.reshape(-1).to(torch.float32)

    total_tokens = flat_labels.shape[0]
    numerator = None
    for start in range(0, total_tokens, LOSS_CHUNK_TOKENS):
        stop = min(start + LOSS_CHUNK_TOKENS, total_tokens)
        chunk_logits = flat_logits[start:stop]
        chunk_labels = flat_labels[start:stop]
        chunk_weights = flat_weights[start:stop]
        if torch.is_grad_enabled() and chunk_logits.requires_grad:
            chunk_sum = torch.utils.checkpoint.checkpoint(
                _chunk_weighted_nll,
                chunk_logits,
                chunk_labels,
                chunk_weights,
                use_reentrant=False,
            )
        else:
            # Inference/eval path: nothing to recompute, so skip checkpointing.
            chunk_sum = _chunk_weighted_nll(chunk_logits, chunk_labels, chunk_weights)
        numerator = chunk_sum if numerator is None else numerator + chunk_sum

    if numerator is None:
        numerator = flat_logits.sum() * 0.0
    denominator = valid.sum().to(numerator.dtype)
    global_denominator = denominator.detach().clone()
    if world_size > 1:
        dist.all_reduce(global_denominator, op=dist.ReduceOp.SUM)
    # DDP averages gradients across ranks, hence the world_size multiplier.
    loss = numerator * world_size / global_denominator.clamp_min(1.0)
    return loss, numerator.detach(), denominator.detach()


def sparse_prediction_targets(labels: torch.Tensor, weights: torch.Tensor):
    """Select logit positions whose next token is supervised.

    Qwen3-VL accepts a tensor for ``logits_to_keep`` and applies the LM head only
    at those sequence positions.  Timeline rows supervise a few thousand
    assistant tokens inside sequences that can exceed 100k tokens, so selecting
    before the LM head avoids materialising a full [sequence, vocabulary] tensor
    while preserving the exact shifted causal-LM objective.
    """
    if labels.ndim != 2 or labels.shape[0] != 1 or weights.shape != labels.shape:
        raise ValueError("sparse logits currently require matching [1, sequence] labels/weights")
    shifted_labels = labels[:, 1:]
    shifted_weights = weights[:, 1:]
    prediction_indices = shifted_labels[0].ne(-100).nonzero(as_tuple=False).flatten()
    if prediction_indices.numel() == 0:
        raise ValueError("sample has no supervised next-token targets")
    return (
        prediction_indices,
        shifted_labels.index_select(1, prediction_indices),
        shifted_weights.index_select(1, prediction_indices),
    )


def weighted_sparse_loss(logits, target_labels, target_weights,
                         world_size: int, denominator_mode: str = "token_count",
                         precomputed_global_denominator=None,
                         loss_multiplier: float = 1.0,
                         ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Weighted NLL for logits already selected at supervised positions."""
    if logits.shape[:-1] != target_labels.shape or target_weights.shape != target_labels.shape:
        raise ValueError(
            f"sparse logit/target shape mismatch: {tuple(logits.shape)} "
            f"vs {tuple(target_labels.shape)} / {tuple(target_weights.shape)}"
        )
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = target_labels.reshape(-1)
    flat_weights = target_weights.reshape(-1).to(torch.float32)
    numerator = None
    for start in range(0, flat_labels.numel(), LOSS_CHUNK_TOKENS):
        stop = min(start + LOSS_CHUNK_TOKENS, flat_labels.numel())
        chunk_logits = flat_logits[start:stop]
        chunk_labels = flat_labels[start:stop]
        chunk_weights = flat_weights[start:stop]
        if torch.is_grad_enabled() and chunk_logits.requires_grad:
            chunk_sum = torch.utils.checkpoint.checkpoint(
                _chunk_weighted_nll,
                chunk_logits,
                chunk_labels,
                chunk_weights,
                use_reentrant=False,
            )
        else:
            chunk_sum = _chunk_weighted_nll(chunk_logits, chunk_labels, chunk_weights)
        numerator = chunk_sum if numerator is None else numerator + chunk_sum
    if numerator is None:
        numerator = flat_logits.sum() * 0.0
    valid_count = flat_labels.ne(-100).sum().to(numerator.dtype)
    if denominator_mode == "weight_mass":
        denominator = (flat_weights * flat_labels.ne(-100)).sum().to(numerator.dtype)
    elif denominator_mode == "token_count":
        denominator = valid_count
    else:
        raise ValueError(f"unknown loss denominator mode: {denominator_mode!r}")
    if precomputed_global_denominator is None:
        global_denominator = denominator.detach().clone()
        if world_size > 1:
            dist.all_reduce(global_denominator, op=dist.ReduceOp.SUM)
    else:
        global_denominator = torch.as_tensor(
            precomputed_global_denominator,
            device=numerator.device,
            dtype=numerator.dtype,
        ).detach()
    loss = (
        numerator
        * world_size
        * float(loss_multiplier)
        / global_denominator.clamp_min(1.0)
    )
    return loss, numerator.detach(), denominator.detach(), valid_count.detach()


def _unwrap(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def save_step_checkpoint(path: Path, model, optimizer, scheduler, processor, args,
                         epoch: int, step: int, update: int, keep_last: int = 3) -> None:
    """Write a fully resumable checkpoint atomically.
    
    Save adapter, optimizer, scheduler, and data-loader position in a temporary
    sibling directory, then replace the destination. An interrupted write leaves
    the previous checkpoint intact.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if staging.exists():
        import shutil
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        _unwrap(model).save_pretrained(staging)
        processor.save_pretrained(staging)
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "step": step,
                "update": update,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
                "python_rng": random.getstate(),
                "args": vars(args),
            },
            staging / "trainer_state.pt",
        )
        if path.exists():
            import shutil
            shutil.rmtree(path, ignore_errors=True)
        os.replace(staging, path)
    finally:
        if staging.exists():
            import shutil
            shutil.rmtree(staging, ignore_errors=True)

    # Retain only the most recent checkpoints; adapters are small but the
    # optimizer state is not.
    siblings = sorted(
        (p for p in path.parent.glob("step_*") if p.is_dir()),
        key=lambda p: int(p.name.split("_")[-1]),
    )
    import shutil
    for stale in siblings[:-keep_last]:
        shutil.rmtree(stale, ignore_errors=True)


def find_latest_checkpoint(output_dir: Path) -> Path | None:
    candidates = [
        p for p in output_dir.glob("step_*")
        if p.is_dir() and (p / "trainer_state.pt").is_file()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: int(p.name.split("_")[-1]))


@torch.no_grad()
def evaluate(model, loader, local_rank: int, world_size: int,
             denominator_mode: str = "token_count", *,
             cb_stream_loss: bool = False, silence_id: int | None = None,
             response_id: int | None = None, cb_beta: float = 0.9999,
             cb_lambda_text: float = 2.0) -> tuple[float, int, dict]:
    """Weighted loss over a held-out split, same numerator/denominator as training.

    ``weighted_loss`` is called with world_size=1 so it skips its internal
    all_reduce (that one only rescales the DDP training loss); the sums are
    reduced once here instead. Note DistributedSampler pads the split so every
    rank sees the same count, which repeats a handful of rows when the split
    does not divide evenly — immaterial at 250 rows over 16 ranks.
    """
    model.eval()
    # NLL numerator, selected loss denominator, and the true supervised-token
    # count.  For ordinary training the latter two are identical.  Calibration
    # rows can instead use weight-mass normalisation while retaining honest
    # token-count logging.
    totals = torch.zeros(3, device=torch.device(f"cuda:{local_rank}"), dtype=torch.float64)
    # For each optional calibration group: NLL numerator, weighted denominator,
    # exact action-token hits, and examples.  The four calibration rows contain
    # exactly one supervised token; the rehearsal group contains full-token loss
    # and intentionally has no action-accuracy statistic.
    group_totals = torch.zeros(
        (5, 4), device=torch.device(f"cuda:{local_rank}"), dtype=torch.float64
    )
    for batch in loader:
        group_id = int(batch.pop("calibration_group").item())
        labels = batch.pop("labels").to(local_rank, non_blocking=True)
        weights = batch.pop("loss_weights").to(local_rank, non_blocking=True)
        if cb_stream_loss:
            if silence_id is None or response_id is None:
                raise ValueError("CB-Stream validation requires both control-token ids")
            weights, _ = cb_stream_weights(
                labels, weights, silence_id, response_id, cb_beta,
                cb_lambda_text, world_size,
            )
        inputs = {key: value.to(local_rank, non_blocking=True) for key, value in batch.items()}
        prediction_indices, target_labels, target_weights = sparse_prediction_targets(
            labels, weights
        )
        outputs = model(
            **inputs, use_cache=False, logits_to_keep=prediction_indices
        )
        _, numerator, denominator, valid_count = weighted_sparse_loss(
            outputs.logits, target_labels, target_weights, 1, denominator_mode
        )
        totals += torch.stack([numerator, denominator, valid_count]).to(dtype=torch.float64)
        if 0 <= group_id < 5:
            group_totals[group_id, 0] += numerator.to(dtype=torch.float64)
            group_totals[group_id, 1] += target_weights.sum().to(dtype=torch.float64)
            group_totals[group_id, 3] += 1
            if group_id < 4:
                if target_labels.numel() != 1:
                    raise ValueError(
                        "action calibration validation row must have exactly one target, "
                        f"got {target_labels.numel()}"
                    )
                predicted = outputs.logits.reshape(-1, outputs.logits.shape[-1]).argmax(dim=-1)
                group_totals[group_id, 2] += predicted.eq(target_labels.reshape(-1)).sum()
        del outputs
    if world_size > 1:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        dist.all_reduce(group_totals, op=dist.ReduceOp.SUM)
    model.train()
    names = (
        "pre_first_inside_response",
        "pre_first_outside_silence",
        "post_first_inside_response",
        "post_first_outside_silence",
        "mixed_full_token_rehearsal",
    )
    groups = {}
    for index, name in enumerate(names):
        examples = int(group_totals[index, 3].cpu())
        if examples == 0:
            continue
        row = {
            "loss": float(
                (group_totals[index, 0] / group_totals[index, 1].clamp_min(1)).cpu()
            ),
            "loss_mass": float(group_totals[index, 1].cpu()),
            "examples": examples,
        }
        if index < 4:
            row["action_accuracy"] = float(
                (group_totals[index, 2] / group_totals[index, 3].clamp_min(1)).cpu()
            )
        groups[name] = row
    return (
        float((totals[0] / totals[1].clamp_min(1)).cpu()),
        int(totals[2].cpu()),
        groups,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--output-dir", required=True)
    # Continue training an existing adapter instead of initialising a fresh
    # LoRA. The saved adapter carries trainable_tokens_delta for </silence> and
    # </response>, and PEFT writes those rows by index_copy (absolute values,
    # not additive), so the trained control-token embeddings are restored too.
    parser.add_argument("--init-adapter", default=None)
    # Optional held-out split; evaluated at the end of every epoch with the
    # same weighted_loss() used for training, so the number is comparable to
    # train/loss and across runs.
    parser.add_argument("--val-data", default=None)
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.add_argument(
        "--save-every-updates", type=int, default=0,
        help="Write a fully resumable checkpoint every N optimizer updates; 0 disables it.")
    parser.add_argument(
        "--resume-from-checkpoint", default=None,
        help="Path to a step_* checkpoint, or 'auto' to pick the newest one in --output-dir.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--stop-after-epoch", type=int, default=None,
        help="Stop cleanly after saving this zero-based epoch while retaining the "
             "LR schedule implied by --epochs. Used for controlled early-checkpoint pilots.")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--freeze-control-tokens", action="store_true",
        help="Do not train the </silence>/</response> embedding + lm_head rows. "
             "Newly added tokens retain their initial values when frozen; use "
             "this with an --init-adapter that already trained them.")
    parser.add_argument("--max-pixels", type=int, default=65536)
    parser.add_argument("--min-pixels", type=int, default=3136)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument(
        "--attn-implementation",
        choices=("flash_attention_2", "sdpa"),
        default="flash_attention_2",
        help="Exact attention kernel. FlashAttention preserves the objective while reducing activation memory.",
    )
    parser.add_argument(
        "--activation-offload-image-threshold",
        type=int,
        default=0,
        help="CPU-offload saved activations for samples with at least this many images; 0 disables it.",
    )
    # Control-token loss weights; defaults preserve the fixed weighting schedule.
    parser.add_argument("--response-onset-weight", type=float, default=1.5)
    # Off by default. Set (e.g. 0.1) to down-weight in-window silence slots
    # listed by each row's "soft_slots" field. Rows without the field are
    # unaffected, allowing mixed annotation formats.
    parser.add_argument("--soft-slot-weight", type=float, default=None)
    parser.add_argument("--pre-optimal-silence-weight", type=float, default=None)
    parser.add_argument("--post-optimal-silence-weight", type=float, default=None)
    # Extra multiplier on the FIRST </response> onset of each sample; see the
    # comment in _labels_and_weights. 1.0 (default) disables it, keeping the
    # weight schedule identical to runs that predate the flag.
    parser.add_argument("--first-response-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--cb-stream-loss", action="store_true",
        help="Replace the static decision-token weights with global-batch "
             "class-balanced decision-token weights. Requires token_count denominator "
             "and unit response/first-response onset multipliers.")
    parser.add_argument(
        "--cb-pure", action="store_true",
        help='Use strict class-balanced streaming loss without legacy temporal weighting. Every silence/response decision token has unit temporal weight before the dynamic class-balanced factor is applied. Requires --cb-stream-loss and forbids soft/pre/post temporal weights.')
    parser.add_argument("--cb-beta", type=float, default=0.9999)
    parser.add_argument("--cb-lambda-text", type=float, default=2.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--length-bucketed-sampler", action="store_true",
        help="Group similar image-count rows per DDP global batch to reduce straggler waits.")
    parser.add_argument(
        "--sampler-num-replicas", type=int, default=0,
        help="Optional virtual replica count for the training sampler. This is only for "
             "world-size recovery: keep the original sample order and updates-per-epoch "
             "when fewer DDP ranks resume a checkpoint.")
    parser.add_argument(
        "--loss-denominator", choices=("token_count", "weight_mass"),
        default="token_count",
        help="Normalise weighted NLL by supervised-token count (legacy default) or by "
             "the sum of loss weights. Use weight_mass when row_loss_mass_target is "
             "present so equal-mass rows also retain a normal gradient scale.")
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument(
        "--no-delegation-prompt", action="store_true",
        help="Drop the Delegate action from the system prompt. Use when the "
             "delegation task type has been excluded from the training data.")
    parser.add_argument(
        "--clear-useful-system-prompt", action="store_true",
        help="Use the clear-and-useful two-action system prompt for the prompt-alignment arm.")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--wandb-project", default="streamvibe-sft")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    parser.add_argument("--wandb-tags", nargs="*", default=[])
    parser.add_argument("--dry-run-collator", action="store_true")
    parser.add_argument(
        "--strict-data", action="store_true",
        help="Fail on malformed rows, image-tag mismatches, or missing frames; "
             "never substitute a later sample. Required for audited timeline runs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.no_delegation_prompt and args.clear_useful_system_prompt:
        raise ValueError(
            "--no-delegation-prompt and --clear-useful-system-prompt are mutually exclusive"
        )
    if args.stop_after_epoch is not None and not 0 <= args.stop_after_epoch < args.epochs:
        raise ValueError(
            f"--stop-after-epoch must be in [0, {args.epochs - 1}], "
            f"got {args.stop_after_epoch}"
        )
    if args.cb_stream_loss:
        if args.loss_denominator != "token_count":
            raise ValueError("--cb-stream-loss requires --loss-denominator token_count")
        if args.response_onset_weight != 1.0:
            raise ValueError(
                "--cb-stream-loss requires --response-onset-weight 1.0 to avoid "
                "double-weighting the response class"
            )
        if args.first_response_multiplier != 1.0:
            raise ValueError(
                "--cb-stream-loss requires --first-response-multiplier 1.0 to avoid "
                "double-weighting first responses"
            )
        if not 0.0 <= args.cb_beta < 1.0:
            raise ValueError(f"--cb-beta must be in [0, 1), got {args.cb_beta!r}")
        if not math.isfinite(args.cb_lambda_text) or args.cb_lambda_text <= 0:
            raise ValueError(
                "--cb-lambda-text must be finite and positive, "
                f"got {args.cb_lambda_text!r}"
            )
    if args.cb_pure:
        if not args.cb_stream_loss:
            raise ValueError("--cb-pure requires --cb-stream-loss")
        if any(value is not None for value in (
            args.soft_slot_weight,
            args.pre_optimal_silence_weight,
            args.post_optimal_silence_weight,
        )):
            raise ValueError(
                "--cb-pure forbids soft/pre/post temporal weights; strict CB "
                "must receive unit temporal weights"
            )
    rank, local_rank, world_size = setup_distributed()
    sampler_num_replicas = args.sampler_num_replicas or world_size
    if sampler_num_replicas < world_size:
        raise ValueError(
            "--sampler-num-replicas must be at least the actual DDP world size "
            f"({world_size}), got {sampler_num_replicas}"
        )
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    added_control_tokens = processor.tokenizer.add_special_tokens(
        {"additional_special_tokens": ["</silence>", "</response>"]}
    )
    control_token_ids = [
        processor.tokenizer.convert_tokens_to_ids("</silence>"),
        processor.tokenizer.convert_tokens_to_ids("</response>"),
    ]
    dataset = JsonDataset(args.train_data, args.max_samples, args.seed,
                          strict=args.strict_data)
    if world_size > 1 and args.length_bucketed_sampler:
        sampler = LengthBucketDistributedSampler(
            dataset, sampler_num_replicas, rank, args.seed
        )
    else:
        sampler = DistributedSampler(
            dataset, num_replicas=sampler_num_replicas, rank=rank,
            shuffle=True, seed=args.seed
        ) \
            if world_size > 1 else None
    collator = StreamVibeCollator(processor, args.max_pixels, args.min_pixels, args.max_length,
                             response_onset_weight=args.response_onset_weight,
                             soft_slot_weight=args.soft_slot_weight,
                             pre_optimal_silence_weight=args.pre_optimal_silence_weight,
                             post_optimal_silence_weight=args.post_optimal_silence_weight,
                             first_response_multiplier=args.first_response_multiplier,
                             cb_stream_loss=args.cb_stream_loss,
                             cb_pure=args.cb_pure,
                             system_prompt=(
                                 SYSTEM_PROMPT_CLEAR_USEFUL
                                 if args.clear_useful_system_prompt
                                 else SYSTEM_PROMPT_NO_DELEGATION
                                 if args.no_delegation_prompt
                                 else SYSTEM_PROMPT
                             ))
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        shuffle=sampler is None,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = None
    if args.val_data:
        val_dataset = JsonDataset(args.val_data, 0, args.seed,
                                  strict=args.strict_data)
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if world_size > 1 else None
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            sampler=val_sampler,
            shuffle=False,
            collate_fn=collator,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    if args.dry_run_collator:
        batch = next(iter(loader))
        if is_main(rank):
            print(
                json.dumps(
                    {
                        "input_shape": list(batch["input_ids"].shape),
                        "supervised_tokens": int(batch["labels"].ne(-100).sum()),
                        "images": int(batch.get("image_grid_thw", torch.empty(0, 3)).shape[0]),
                    }
                ),
                flush=True,
            )
        if world_size > 1:
            dist.destroy_process_group()
        return

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        local_files_only=True,
    )
    # Qwen3-VL reserves padded vocabulary rows beyond the tokenizer length. The
    # two control tokens normally fit in those rows; resize only if necessary.
    required_vocab = max(control_token_ids) + 1
    current_vocab = model.get_input_embeddings().weight.shape[0]
    if required_vocab > current_vocab:
        model.resize_token_embeddings(required_vocab)
    if added_control_tokens:
        # Give newly named reserved rows a semantic starting point instead of
        # relying on arbitrary padded-vocabulary values.
        with torch.no_grad():
            input_embeddings = model.get_input_embeddings().weight
            output_embeddings = model.get_output_embeddings().weight
            for token, token_id in zip(("silence", "response"), control_token_ids):
                source_ids = processor.tokenizer.encode(token, add_special_tokens=False)
                input_embeddings[token_id].copy_(input_embeddings[source_ids].mean(dim=0))
                output_embeddings[token_id].copy_(output_embeddings[source_ids].mean(dim=0))
    # PEFT's TrainableTokensLayer already implements nn.Linear forward/merge
    # for an untied LM head, but peft 0.17.1 still reads ``embedding_dim``
    # unconditionally during initialization. Expose the Linear's equivalent
    # dimension so the two lm_head rows can be trained and saved alongside the
    # embedding rows without expanding modules_to_save to the full vocabulary.
    output_embeddings = model.get_output_embeddings()
    if isinstance(output_embeddings, torch.nn.Linear) and not hasattr(output_embeddings, "embedding_dim"):
        output_embeddings.embedding_dim = output_embeddings.in_features
    model.config.use_cache = False
    model.requires_grad_(False)
    # Non-reentrant checkpointing works even when a checkpointed vision block's
    # positional inputs do not require gradients. The reentrant default warned
    # "None of the inputs have requires_grad=True", retained those activations,
    # and OOMed on a long smoke sample at ~80 GiB with r=64.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    # Resolve a step checkpoint to resume from before deciding how to build the
    # PEFT model: its adapter directory replaces --init-adapter as the source of
    # LoRA weights, and its trainer_state.pt restores optimizer/scheduler/data
    # position further below.
    resume_dir = None
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint == "auto":
            resume_dir = find_latest_checkpoint(Path(args.output_dir))
            if resume_dir is None and is_main(rank):
                print("resume=auto found no step_* checkpoint; starting fresh", flush=True)
        else:
            resume_dir = Path(args.resume_from_checkpoint)
            if not (resume_dir / "trainer_state.pt").is_file():
                raise FileNotFoundError(f"no trainer_state.pt under {resume_dir}")

    adapter_source = str(resume_dir) if resume_dir is not None else args.init_adapter
    if adapter_source:
        # Load an existing LoRA + control-token rows and keep training them. The
        # mean-init above is harmless: trainable_tokens_delta holds absolute
        # values and overwrites those rows on every forward.
        #
        # peft 0.19.1 calls _maybe_shard_state_dict_for_tp() from
        # set_peft_model_state_dict() whenever torch.distributed is initialised,
        # and that function unconditionally imports EmbeddingParallel, which
        # transformers 4.57.1 does not define -> ImportError on every rank.
        # We train with DDP, not tensor parallelism, so there is nothing to
        # shard; neutralise it for the duration of the load. Assert the model
        # really has no TP plan first, so this stays a no-op rather than
        # silently skipping work that mattered.
        import peft.utils.save_and_load as _peft_sl

        tp_plan = getattr(getattr(model, "config", None), "base_model_tp_plan", None) \
            or getattr(model, "_tp_plan", None)
        if tp_plan:
            raise RuntimeError(f"model declares a tensor-parallel plan ({tp_plan}); "
                               "the TP-sharding bypass below would be wrong")
        # The hook exists in peft >= 0.19; patch only when available so
        # adapter loading also works with earlier versions.
        _shard_name = "_maybe_shard_state_dict_for_tp"
        _patched = hasattr(_peft_sl, _shard_name)
        original_shard = getattr(_peft_sl, _shard_name, None)
        if _patched:
            setattr(_peft_sl, _shard_name, lambda *a, **k: None)
        try:
            model = PeftModel.from_pretrained(model, adapter_source, is_trainable=True)
        finally:
            if _patched:
                setattr(_peft_sl, _shard_name, original_shard)
        model = model.to(local_rank)
        if is_main(rank):
            print(f"resumed_adapter={Path(adapter_source).resolve()}", flush=True)
    else:
        lora = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            # The pretrained backbone may lack the two action control tokens. Train
            # only their input/output rows alongside LoRA, not the full 152k vocab.
            **({} if args.freeze_control_tokens else {
                "trainable_token_indices": {
                    "embed_tokens": control_token_ids,
                    "lm_head": control_token_ids,
                }}),
        )
        model = get_peft_model(model, lora).to(local_rank)
    if is_main(rank):
        model.print_trainable_parameters()
    if world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            broadcast_buffers=False,
        )

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    updates_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_updates = max(updates_per_epoch * args.epochs, 1)
    warmup_steps = int(total_updates * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_updates)

    wandb_run = None
    if is_main(rank) and args.wandb_mode != "disabled":
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            mode=args.wandb_mode,
            tags=args.wandb_tags,
            config={
                **vars(args),
                "world_size": world_size,
                "sampler_num_replicas": sampler_num_replicas,
                "train_samples": len(dataset),
                "updates_per_epoch": updates_per_epoch,
                "total_updates": total_updates,
                "effective_batch_size": world_size * args.grad_accum,
            },
        )
        wandb.define_metric("train/update")
        wandb.define_metric("train/*", step_metric="train/update")

    # Restore optimizer/scheduler/RNG and the data position. The LoRA weights
    # themselves came from the checkpoint's adapter directory above.
    start_epoch = 0
    resume_step = 0
    update = 0
    if resume_dir is not None:
        state = torch.load(resume_dir / "trainer_state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"]
        # step is the index of the last COMPLETED micro-batch in that epoch, so
        # resume at the following one.
        resume_step = state["step"] + 1
        update = state["update"]
        try:
            torch.set_rng_state(state["torch_rng"])
            random.setstate(state["python_rng"])
            cuda_states = state.get("cuda_rng")
            if cuda_states and len(cuda_states) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all(cuda_states)
        except (RuntimeError, TypeError, ValueError) as exc:
            # A different GPU count / driver can reject a saved RNG blob. Data
            # order comes from the sampler seed, not this, so warn and continue.
            if is_main(rank):
                print(f"[warn] could not restore RNG state ({exc}); continuing", flush=True)
        if resume_step >= len(loader):
            start_epoch += 1
            resume_step = 0
        if is_main(rank):
            print(
                json.dumps({
                    "event": "resumed",
                    "checkpoint": str(resume_dir.resolve()),
                    "epoch": start_epoch,
                    "skip_micro_batches": resume_step,
                    "update": update,
                }),
                flush=True,
            )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    accumulated_stats = torch.zeros(
        3, device=torch.device(f"cuda:{local_rank}"), dtype=torch.float64
    )
    training_started = time.monotonic()
    cb_diagnostics_emitted = False
    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for step, batch in enumerate(loader):
            # Fast-forward to the resume point. The batches still have to be
            # produced so the sampler/dataloader order matches the original run,
            # but nothing is moved to GPU and no forward/backward runs.
            if epoch == start_epoch and step < resume_step:
                continue
            # Metadata used only by validation accounting must never be passed
            # to the model forward.
            batch.pop("calibration_group", None)
            labels = batch.pop("labels").to(local_rank, non_blocking=True)
            weights = batch.pop("loss_weights").to(local_rank, non_blocking=True)
            if args.cb_stream_loss:
                weights, cb_diagnostics = cb_stream_weights(
                    labels, weights, control_token_ids[0], control_token_ids[1],
                    args.cb_beta, args.cb_lambda_text, world_size,
                    include_histogram=not cb_diagnostics_emitted,
                )
                if args.cb_pure and abs(
                    cb_diagnostics["temporal_factor_global_mean_before_normalization"] - 1.0
                ) > 1e-12:
                    raise RuntimeError(
                        "strict CB isolation failed: action temporal mean is not 1.0: "
                        f"{cb_diagnostics}"
                    )
                if not cb_diagnostics_emitted:
                    if is_main(rank):
                        print(
                            json.dumps({
                                "event": "cb_stream_first_batch",
                                "loss_denominator": args.loss_denominator,
                                "beta": args.cb_beta,
                                "pure_cb": args.cb_pure,
                                **cb_diagnostics,
                            }),
                            flush=True,
                        )
                    cb_diagnostics_emitted = True
            inputs = {key: value.to(local_rank, non_blocking=True) for key, value in batch.items()}
            image_count = int(inputs.get("image_grid_thw", torch.empty(0)).shape[0])
            sequence_length = int(inputs["input_ids"].shape[1])
            should_offload = (
                args.activation_offload_image_threshold > 0
                and image_count >= args.activation_offload_image_threshold
            )
            activation_context = (
                torch.autograd.graph.save_on_cpu(pin_memory=True)
                if should_offload
                else contextlib.nullcontext()
            )
            try:
                with activation_context:
                    prediction_indices, target_labels, target_weights = sparse_prediction_targets(
                        labels, weights
                    )
                    outputs = model(
                        **inputs, use_cache=False, logits_to_keep=prediction_indices
                    )
                    loss, numerator, denominator, valid_count = weighted_sparse_loss(
                        outputs.logits, target_labels, target_weights, world_size,
                        args.loss_denominator,
                    )
                    (loss / args.grad_accum).backward()
            except torch.OutOfMemoryError:
                # Every rank sees different samples under DistributedSampler.
                # Emit the local shape that actually failed; rank-0-only
                # aggregate metrics cannot identify a long-tail offender.
                print(
                    json.dumps(
                        {
                            "event": "cuda_oom",
                            "rank": rank,
                            "epoch": epoch,
                            "step": step,
                            "images": image_count,
                            "sequence_length": sequence_length,
                            "activation_offload": should_offload,
                            "allocated_gib": torch.cuda.memory_allocated(local_rank) / 2**30,
                            "reserved_gib": torch.cuda.memory_reserved(local_rank) / 2**30,
                        }
                    ),
                    flush=True,
                )
                raise
            accumulated_stats += torch.stack([numerator, denominator, valid_count]).to(
                device=local_rank, dtype=torch.float64
            )
            del outputs

            should_update = (step + 1) % args.grad_accum == 0 or step + 1 == len(loader)
            if should_update:
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update += 1

                stats = accumulated_stats.clone()
                accumulated_stats.zero_()
                if world_size > 1:
                    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                if is_main(rank) and update % args.log_every == 0:
                    metrics = {
                        "epoch": epoch,
                        "update": update,
                        "total_updates": total_updates,
                        "loss": float((stats[0] / stats[1].clamp_min(1)).cpu()),
                        "lr": scheduler.get_last_lr()[0],
                        "grad_norm": float(grad_norm.detach().cpu()),
                        "loss_denominator": float(stats[1].cpu()),
                        "supervised_tokens": int(stats[2].cpu()),
                        "elapsed_seconds": time.monotonic() - training_started,
                        "max_memory_gib": torch.cuda.max_memory_allocated(local_rank) / 2**30,
                    }
                    print(
                        json.dumps(metrics),
                        flush=True,
                    )
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "train/update": update,
                                "train/epoch": epoch,
                                "train/loss": metrics["loss"],
                                "train/lr": metrics["lr"],
                                "train/grad_norm": metrics["grad_norm"],
                                "train/supervised_tokens": metrics["supervised_tokens"],
                                "train/elapsed_seconds": metrics["elapsed_seconds"],
                                "train/max_memory_gib": metrics["max_memory_gib"],
                            }
                        )

                if args.save_every_updates > 0 and update % args.save_every_updates == 0:
                    # Barriers on both sides: rank 0 writes while the others wait,
                    # so no rank races ahead and desynchronises the saved state.
                    if world_size > 1:
                        dist.barrier()
                    if is_main(rank):
                        checkpoint = Path(args.output_dir) / f"step_{update}"
                        save_step_checkpoint(
                            checkpoint, model, optimizer, scheduler, processor,
                            args, epoch, step, update,
                        )
                        print(f"saved_step_checkpoint={checkpoint.resolve()}", flush=True)
                    if world_size > 1:
                        dist.barrier()

        if val_loader is not None:
            val_loss, val_tokens, val_groups = evaluate(
                model, val_loader, local_rank, world_size, args.loss_denominator,
                cb_stream_loss=args.cb_stream_loss,
                silence_id=control_token_ids[0], response_id=control_token_ids[1],
                cb_beta=args.cb_beta, cb_lambda_text=args.cb_lambda_text,
            )
            if is_main(rank):
                print(json.dumps({"epoch": epoch, "val_loss": val_loss,
                                  "val_supervised_tokens": val_tokens,
                                  "val_groups": val_groups}), flush=True)
                if wandb_run is not None:
                    wandb_run.log({"train/update": update, "val/loss": val_loss,
                                   "val/supervised_tokens": val_tokens, "val/epoch": epoch})

        # Always make the exact epoch boundary resumable.  A periodic interval
        # need not divide updates_per_epoch (for example, mixed 1:1 has 506
        # updates/epoch), and resuming from the preceding periodic checkpoint
        # would silently replay the tail of the epoch.
        if args.save_every_updates > 0:
            if world_size > 1:
                dist.barrier()
            if is_main(rank):
                boundary_checkpoint = Path(args.output_dir) / f"step_{update}"
                if not boundary_checkpoint.exists():
                    save_step_checkpoint(
                        boundary_checkpoint, model, optimizer, scheduler, processor,
                        args, epoch, step, update,
                    )
                    print(
                        f"saved_epoch_boundary_step_checkpoint="
                        f"{boundary_checkpoint.resolve()}",
                        flush=True,
                    )
            if world_size > 1:
                dist.barrier()

        if args.save_every_epoch:
            if world_size > 1:
                dist.barrier()
            if is_main(rank):
                checkpoint = Path(args.output_dir) / f"epoch_{epoch}"
                checkpoint.mkdir(parents=True, exist_ok=True)
                snapshot = model.module if isinstance(model, DistributedDataParallel) else model
                snapshot.save_pretrained(checkpoint)
                processor.save_pretrained(checkpoint)
                print(f"saved_epoch_checkpoint={checkpoint.resolve()}", flush=True)
            if world_size > 1:
                dist.barrier()

        if args.stop_after_epoch is not None and epoch >= args.stop_after_epoch:
            if is_main(rank):
                print(
                    f"controlled_stop_after_epoch={epoch} "
                    f"planned_total_updates={total_updates} completed_updates={update}",
                    flush=True,
                )
            break

    if world_size > 1:
        dist.barrier()
    if is_main(rank):
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
        unwrapped.save_pretrained(output)
        processor.save_pretrained(output)
        with (output / "training_args.json").open("w", encoding="utf-8") as handle:
            json.dump(vars(args), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(f"saved_adapter={output.resolve()}", flush=True)
        if wandb_run is not None:
            wandb_run.summary["saved_adapter"] = str(output.resolve())
            wandb_run.finish()
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
