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

"""Causal, round-synchronous streaming rollout against a pool of generation workers.

Each lane advances one second per round and conditions on its own earlier
responses. Images use the same frame materialization and resizing as training.
Action tokens are preserved in generated text."""

from __future__ import annotations

import base64
import io
import json
import threading
import random
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from frames import materialize_video_frame_window, frame_path, process_image


MARGIN_CAPTURE = (
    __import__("os").environ.get("CT_MARGIN_ALPHA", "0") not in ("", "0", "0.0")
    or __import__("os").environ.get("CT_CAPTURE_P", "0") not in ("", "0")
)

from prompting import (
    RESPONSE_TOKEN_ID,
    SILENCE_TOKEN_ID,
    build_messages_native,
    classify_action,
    gt_causal_memory_frames,
    gt_prefix,
)

_SEED_MOD = 2**31 - 1


def lane_seed(base: int, gen_call: int, row: int, g: int, r: int) -> int:
    """Identical construction to the original per-lane seed."""
    return (base * 1_000_003 + gen_call * 100_003 + row * 1_009 + g * 31 + r) % _SEED_MOD


@dataclass
class RolloutConfig:
    engine_urls: list[str]
    model: str = "policy"
    recent_s: int = 5
    anchor_s: int = 10
    max_images: int = 64
    max_new_tokens: int = 48
    temperature: float = 1.0
    top_p: float = 1.0
    min_pixels: int = 3136
    max_pixels: int = 65536
    seed_base: int = 1
    request_timeout: float = 60.0   # hung engines answer /health but never a completion; fail fast and fail over
    request_tries: int = 4          # a transient timeout must not end the run
    max_workers: int = 256
    frame_cache_root: str = "outputs/cache/video_frames"
    jpeg_quality: int = 95


# --------------------------------------------------------------------------- #
# Sample preparation
# --------------------------------------------------------------------------- #
def prepare_sample(row: dict, cfg: RolloutConfig) -> dict:
    """Parse one dataset row and materialize its frames node-locally."""
    gt = json.loads(row["answer"]) if isinstance(row.get("answer"), str) else dict(row["answer"])
    for key in ("trajectory_start_s", "trajectory_end_s", "help_points"):
        if key not in gt:
            raise KeyError(f"row {row.get('id')} missing answer field {key!r}")
    start, end = int(gt["trajectory_start_s"]), int(gt["trajectory_end_s"])
    n_frames = int(row["n_frames"])
    if not (0 <= start <= end < n_frames):
        raise ValueError(f"invalid interval [{start}, {end}] for n_frames={n_frames} ({row.get('id')})")
    videos = row["videos"]
    video_path = videos if isinstance(videos, str) else list(videos)[0]
    prefix_frames, prefix_actions = gt_prefix(gt, start, cfg.recent_s, cfg.anchor_s)
    required = prefix_frames + list(range(start, end + 1))
    frames_dir = materialize_video_frame_window(
        str(video_path), required, cache_root=cfg.frame_cache_root
    )
    return {
        "id": str(row.get("id")),
        
        "source": str(row.get("source_dataset") or row.get("rl_source") or "unknown"),
        "problem": row["problem_reserved_text"],
        "gt": gt,
        "start": start,
        "end": end,
        "frames_dir": frames_dir,
        "prefix_actions": prefix_actions,
    }


# --------------------------------------------------------------------------- #
# HTTP plumbing
# --------------------------------------------------------------------------- #
def _post_chat(url: str, payload: dict, timeout: float, tries: int) -> dict:
    body = json.dumps(payload).encode()
    last: Exception | None = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(
                f"{url}/v1/chat/completions",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:400]
            if exc.code == 400 and ("maximum" in detail or "context length" in detail or "too long" in detail):
                raise OverLengthError(detail) from exc
            last = RuntimeError(f"HTTP {exc.code}: {detail}")
        except Exception as exc:  # noqa: BLE001
            last = exc
        time.sleep(min(2.0 * (attempt + 1), 10.0))
    raise RuntimeError(f"chat request failed after {tries} tries against {url}: {last}")


_BAD_ENGINES: dict[str, float] = {}
_BAD_LOCK = threading.Lock()
_BAD_COOLDOWN_S = 600.0


def _mark_engine_bad(url: str) -> None:
    with _BAD_LOCK:
        _BAD_ENGINES[url] = time.time()


def _healthy_alternatives(urls: list[str], failed: str) -> list[str]:
    now = time.time()
    with _BAD_LOCK:
        bad = {u for u, t in _BAD_ENGINES.items() if now - t < _BAD_COOLDOWN_S}
    alts = [u for u in urls if u != failed and u not in bad]
    random.shuffle(alts)
    return alts[:4]


class OverLengthError(RuntimeError):
    """Prompt exceeded the engine's max model length — stop this lane, keep the rest."""


class _FrameEncoder:
    """path -> base64 JPEG data URL, resized exactly like the trainer."""

    def __init__(self, cfg: RolloutConfig):
        self.cfg = cfg
        self._cache: dict[str, str] = {}

    def data_url(self, path: str) -> str:
        cached = self._cache.get(path)
        if cached is None:
            image = process_image(path, self.cfg.min_pixels, self.cfg.max_pixels)
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=self.cfg.jpeg_quality)
            cached = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
            self._cache[path] = cached
        return cached


def messages_to_openai(messages: list[dict], image_urls: list[str]) -> list[dict]:
    """Convert build_messages_native output (literal <image> per user turn) to
    OpenAI chat format, consuming ``image_urls`` in order."""
    out: list[dict] = []
    cursor = 0
    for msg in messages:
        if msg["role"] != "user":
            out.append({"role": msg["role"], "content": msg["content"]})
            continue
        parts: list[dict] = []
        chunks = msg["content"].split("<image>")
        for i, chunk in enumerate(chunks):
            if i != 0:
                parts.append({"type": "image_url", "image_url": {"url": image_urls[cursor]}})
                cursor += 1
            if chunk:
                parts.append({"type": "text", "text": chunk})
        out.append({"role": "user", "content": parts})
    if cursor != len(image_urls):
        raise ValueError(f"image count mismatch: consumed {cursor} of {len(image_urls)}")
    return out


# --------------------------------------------------------------------------- #
# Rollout
# --------------------------------------------------------------------------- #
@dataclass
class Lane:
    sample_index: int
    g: int
    sample: dict
    engine_url: str
    actions: dict[int, str] = field(default_factory=dict)  # prefix + own, frame -> canonical
    turns: list[tuple] = field(default_factory=list)       # (t, kind, action)
    margins: dict[float, float] = field(default_factory=dict)  
    steps: list[dict] = field(default_factory=list)        # {"at", "frames"}
    done: bool = False
    done_reason: str = "completed"


def rollout_batch(
    samples: list[dict],
    G: int,
    cfg: RolloutConfig,
    gen_call: int,
) -> list[Lane]:
    """Roll out G trajectories per prepared sample, lock-step across rounds."""
    if not cfg.engine_urls:
        raise ValueError("no engine urls")
    lanes: list[Lane] = []
    for i, sample in enumerate(samples):
        for g in range(G):
            lane_index = i * G + g
            lanes.append(
                Lane(
                    sample_index=i,
                    g=g,
                    sample=sample,
                    engine_url=cfg.engine_urls[lane_index % len(cfg.engine_urls)],
                    actions=dict(sample["prefix_actions"]),
                )
            )

    encoder = _FrameEncoder(cfg)

    def run_one(lane: Lane, at: int) -> None:
        sample = lane.sample
        win = gt_causal_memory_frames(
            at=at,
            actions=lane.actions,
            recent_s=cfg.recent_s,
            anchor_s=cfg.anchor_s,
            max_images=cfg.max_images,
        )
        messages = build_messages_native(sample["problem"], at, win, lane.actions, interval_s=1.0)
        urls = [encoder.data_url(frame_path(sample["frames_dir"], f)) for f in win]
        payload = {
            "model": cfg.model,
            "messages": messages_to_openai(messages, urls),
            "max_tokens": cfg.max_new_tokens,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "seed": lane_seed(cfg.seed_base, gen_call, lane.sample_index, lane.g, at),
            "skip_special_tokens": False,
        }
        if MARGIN_CAPTURE:
            payload["logprobs"] = True
            # 点名要这两个 token, 不能指望它们落在 top-5 里
            payload["lp_token_ids"] = [RESPONSE_TOKEN_ID, SILENCE_TOKEN_ID]
        try:
            # A hung vLLM engine still answers /health but never returns a
            # completion (seen repeatedly on one node).  Fail over to other
            # engines rather than killing the whole step.
            try:
                result = _post_chat(lane.engine_url, payload, cfg.request_timeout, cfg.request_tries)
            except OverLengthError:
                raise
            except Exception as first_exc:  # noqa: BLE001
                _mark_engine_bad(lane.engine_url)
                result = None
                for alt in _healthy_alternatives(cfg.engine_urls, lane.engine_url):
                    try:
                        result = _post_chat(alt, payload, cfg.request_timeout, 2)
                        lane.engine_url = alt
                        break
                    except OverLengthError:
                        raise
                    except Exception:  # noqa: BLE001
                        continue
                if result is None:
                    raise first_exc
        except OverLengthError:
            lane.done = True
            lane.done_reason = f"over_length_at_{at}"
            return
        text = (result["choices"][0]["message"].get("content") or "")
        kind, action = classify_action(text)
        ftl = result.get("first_token_logprobs")
        if ftl:
            # JSON 的对象 key 一律是字符串, 而下面用 int token id 去取。
            # 不归一化的话 ftl.get(248078) 永远是 None -> 两个都落到默认 -20.0
            # -> p(response) 恒等于 0.5。这个退化值看起来像数据, 不像故障。
            ftl = {int(k): v for k, v in ftl.items()}
            # normalized p(response) over the two decision tokens; either may
            # fall out of the top-5 -- treat missing mass as exp(-20)
            import math as _m
            pr = _m.exp(ftl.get(RESPONSE_TOKEN_ID, -20.0))
            ps = _m.exp(ftl.get(SILENCE_TOKEN_ID, -20.0))
            if pr + ps > 0:
                lane.margins[float(at)] = pr / (pr + ps)
        lane.turns.append((float(at), kind, action))
        lane.steps.append({"at": at, "frames": list(win)})
        lane.actions[at] = action

    # 原实现是逐秒 lock-step: 每一轮把所有 lane 在该秒的请求并发发出, 然后
    # 用 pool.map 的隐式栅栏等最慢的那个回来才进下一秒。lane 之间本来是独立的
    # (run_one 只读写 lane 自己的 actions/turns/steps/margins, seed 由
    # (sample_index, g, at) 决定, 与执行顺序无关), 所以这个栅栏纯属实现产物:
    # 60 轮 x 等 p99 而不是 60 轮 x 均值。
    #
    # 改成每个 lane 独立跑完自己的整段时间轴, lane 内部仍严格按秒递增(因果性
    # 由此保证: 第 t 秒的 prompt 要读 lane.actions 里 <t 的动作)。快的 lane
    # 不再被慢的拖住, 消掉 60 次全局同步。
    def run_lane(lane: Lane) -> None:
        start, end = lane.sample["start"], lane.sample["end"]
        for at in range(start, end + 1):
            if lane.done:
                return
            run_one(lane, at)
        lane.done = True

    # 每个 lane 要独占一个线程才能独立推进; 上限仍受 cfg.max_workers 约束,
    # 但不再因为 max_workers < len(lanes) 而退化成分波执行。
    workers = min(len(lanes), max(cfg.max_workers, len(lanes)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # list() forces exceptions to surface here rather than being dropped
        list(pool.map(run_lane, lanes))
    return lanes
