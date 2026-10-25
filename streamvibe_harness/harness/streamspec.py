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

"""Lazily expand compact video specifications into per-second streaming rows.

The runner expands only the current batch of lanes, bounding memory usage for
long videos. `spec_turns` determines sequence length without full expansion.
"""
from __future__ import annotations


def spec_turns(row: dict) -> int | None:
    spec = row.get("stream_spec")
    if not spec:
        return None
    return int(spec["turns"])


def expand(row: dict) -> dict:
    """Materialise `messages` + `images` for a row carrying a `stream_spec`.

    Rows without a spec pass through untouched, so the two kinds of suite can
    sit side by side in one run.
    """
    spec = row.get("stream_spec")
    if not spec:
        return row
    if row.get("messages"):
        return row

    turns = int(spec["turns"])
    fps = float(spec.get("fps", 1.0))
    video = spec["video_path"]
    query = spec.get("query") or ""
    query_turn = int(spec.get("query_turn", 0))
    gold_turn = int(spec.get("gold_turn", turns - 1))
    gold_text = spec.get("gold_text") or ""

    turn_seconds = spec.get("turn_seconds")
    if turn_seconds is None:
        turn_seconds = [t / fps for t in range(turns)]
    if len(turn_seconds) != turns:
        raise ValueError(
            f"{row.get('eval_id')}: turn_seconds has {len(turn_seconds)} "
            f"entries for {turns} turns")

    frame_times_by_turn = spec.get("frame_times_by_turn")
    if frame_times_by_turn is None:
        frame_times_by_turn = [[second] for second in turn_seconds]
    if len(frame_times_by_turn) != turns:
        raise ValueError(
            f"{row.get('eval_id')}: frame_times_by_turn has "
            f"{len(frame_times_by_turn)} entries for {turns} turns")

    gold_raw_by_turn = spec.get("gold_raw_by_turn")
    if gold_raw_by_turn is not None and len(gold_raw_by_turn) != turns:
        raise ValueError(
            f"{row.get('eval_id')}: gold_raw_by_turn has "
            f"{len(gold_raw_by_turn)} entries for {turns} turns")

    # One decimal is the streaming convention and is right at 1 FPS. Above 1 FPS it
    # is NOT: at 4 FPS, seconds 0.25 and 0.75 both round to different values
    # than they are ("0.2", "0.8"), and two distinct steps can print the same
    # timestamp -- which the model reads as time standing still.
    places = 1 if fps <= 1.0 else 2
    messages, images = [], []
    for t in range(turns):
        second = float(turn_seconds[t])
        frame_times = [float(value) for value in frame_times_by_turn[t]]
        if not frame_times:
            raise ValueError(f"{row.get('eval_id')}: turn {t} has no frames")
        if any(value > second + 1e-6 for value in frame_times):
            raise ValueError(
                f"{row.get('eval_id')}: turn {t} contains a future frame")
        parts = [f"<{second:.{places}f} seconds>"]
        parts.extend("<image>" for _ in frame_times)
        if t == query_turn and query:
            parts.insert(0, query)
        messages.append({"role": "user", "content": "\n".join(parts)})
        # The frame is a virtual URI; the runner's ffmpeg step materialises it
        # on demand, so nothing has to be pre-extracted for a 283 GB suite.
        images.extend(
            f"video-frame::{frame_second:.3f}::{video}"
            for frame_second in frame_times)
        if gold_raw_by_turn is not None:
            gold_raw = str(gold_raw_by_turn[t])
        else:
            gold_raw = f"</response> {gold_text}" if t == gold_turn else "</silence>"
        messages.append({
            "role": "assistant",
            "content": gold_raw,
        })

    out = dict(row)
    out["messages"] = messages
    out["images"] = images
    return out
