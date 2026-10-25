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

"""Shared inference helpers for StreamVibe Harness."""
from __future__ import annotations
import json
import re
from pathlib import Path


MAX_PIXELS = 65536

MIN_PIXELS = 3136

TIME_RE = re.compile(r"<([0-9.]+) seconds>")

DEFAULT_MAX_CONTEXT_STEPS = 16

HELP_ACTIONS = {"response"}

ACTION_MARKERS = (
    ("response", ("</response>",)),
    ("silence", ("</silence>", "<|silent|>")),
)

def action_of(text: str) -> str:
    """Parse the model action from its generated text.
    
    The first action marker determines silence or response. If no marker is
    present, non-empty text counts as a response. Format compliance is recorded
    separately by `is_wrapped`.
    """
    value = text or ""
    best_at, best_action = len(value), None
    for action, markers in ACTION_MARKERS:
        for marker in markers:
            at = value.find(marker)
            if at != -1 and at < best_at:
                best_at, best_action = at, action
    if best_action is not None:
        return best_action
    return "response" if value.strip() else "silence"

def is_wrapped(text: str) -> bool:
    """Whether the generation carried an action wrapper at all.

    Reported alongside the metrics rather than folded into them: an arm with a
    high unwrapped rate is not answering the protocol as posed, and its numbers
    should be read knowing that, but the unwrapped turns still get scored on
    what the model actually did.
    """
    value = text or ""
    return any(marker in value for _, markers in ACTION_MARKERS for marker in markers)

def clean_text(text: str) -> str:
    value = text or ""
    for token in ("</response>", "</silence>"):
        value = value.replace(token, "")
    return " ".join(value.split())

def turn_table(row: dict) -> list[dict]:
    turns = []
    pending_user = ""
    index = 0
    for message in row.get("messages", []):
        if message.get("role") == "user":
            pending_user = str(message.get("content", ""))
        elif message.get("role") == "assistant":
            match = TIME_RE.search(pending_user)
            turns.append(
                {
                    "turn": index,
                    "second": float(match.group(1)) if match else float(index),
                    "user": pending_user,
                    "gold_raw": str(message.get("content", "")),
                    "gold_action": action_of(str(message.get("content", ""))),
                    "has_query": bool(
                        re.sub(r"<[^>]+>|\s+", "", pending_user.replace("<image>", ""))
                    ),
                }
            )
            index += 1
    return turns

def task_probes(row: dict, task: str) -> list[dict]:
    turns = turn_table(row)
    positives = []
    if task == "background":
        positives = [turn for turn in turns if turn["gold_action"] == "delegate"]
    else:
        positives = [turn for turn in turns if turn["gold_action"] == "response"]
    silences = [turn for turn in turns if turn["gold_action"] == "silence"]
    if not positives:
        return []

    # Narration can have many legitimate response moments; two deterministic
    # positives capture early/late content without letting long clips dominate.
    chosen_pos = [positives[0]]
    if task == "narration" and len(positives) > 1:
        chosen_pos.append(positives[-1])
    probes = []
    for position, turn in enumerate(chosen_pos):
        probe = dict(turn)
        probe["probe_label"] = "positive" if position == 0 else "positive_late"
        probes.append(probe)

    first_turn = positives[0]["turn"]
    earlier = [turn for turn in silences if turn["turn"] < first_turn]
    if task == "narration" and silences:
        positive_turns = [turn["turn"] for turn in positives]
        ranked = sorted(
            silences,
            key=lambda turn: min(abs(turn["turn"] - value) for value in positive_turns),
        )
        near = dict(ranked[0])
        near["probe_label"] = "negative_near"
        probes.append(near)
        if len(ranked) > 1:
            far = dict(ranked[-1])
            far["probe_label"] = "negative_far"
            probes.append(far)
        return sorted(probes, key=lambda probe: probe["turn"])
    if earlier:
        near = dict(earlier[-1])
        near["probe_label"] = "negative_near"
        probes.append(near)

    if task in ("event_grounding", "narration"):
        query_turns = [
            turn["turn"]
            for turn in turns
            if turn["has_query"] and turn["turn"] < first_turn
        ]
        lower = query_turns[-1] if query_turns else 0
        far_candidates = [
            turn
            for turn in silences
            if lower <= turn["turn"] < max(lower, first_turn - 1)
            and turn["turn"] != (earlier[-1]["turn"] if earlier else -1)
        ]
        if far_candidates:
            far = dict(far_candidates[len(far_candidates) // 2])
            far["probe_label"] = "negative_far"
            probes.append(far)

    # Stable order is chronological, which also makes timing audits readable.
    return sorted(probes, key=lambda probe: probe["turn"])

def dense_probes(row: dict) -> list[dict]:
    """Evaluate every annotated timestamp with a causal visual window.

    ``build_prefix`` keeps the most recent visual steps and teacher-forces only
    earlier gold help actions in chronological order. Any non-silence gold action
    is a help point; this matters for background, where the source contains both
    foreground responses and background delegations.
    """
    probes = []
    for turn in turn_table(row):
        probe = dict(turn)
        probe["probe_label"] = (
            "positive_dense"
            if turn["gold_action"] == "response"
            else "negative_dense"
        )
        probes.append(probe)
    return probes

def _turn_records(row: dict) -> list[dict] | None:
    """Align each user timestamp with its images and following gold action."""
    image_paths = iter(row.get("images", []))
    pending = None
    records = []
    for message in row.get("messages", []):
        role = message.get("role")
        if role == "user":
            raw = str(message.get("content", ""))
            paths = []
            for _ in range(raw.count("<image>")):
                try:
                    paths.append(Path(next(image_paths)))
                except StopIteration:
                    return None
            pending = {"user_raw": raw, "image_paths": paths}
        elif role == "assistant":
            if pending is None:
                return None
            pending["assistant_raw"] = str(message.get("content", ""))
            records.append(pending)
            pending = None
    return records

def _user_content(
    record: dict, *, include_images: bool, images_per_step: int = 0
) -> list[dict]:
    """Render one user turn, optionally removing only its visual payload.

    ``images_per_step`` caps how many of the turn's frames are attached; 0 keeps
    all of them, which is the historical behaviour. The cap exists because this
    dataset stores every native frame that falls inside a one-second slot -- 1, 2
    or 4 of them depending on the source video -- while both the trainer and
    streaming inference use 1 FPS with one frame per second. Left
    uncapped, a 32-step window carries 76-93 images instead of 32, so the model
    sees a visual density it was never trained on.

    The kept frames are the earliest in the slot: the trainer's ``decode_one_fps``
    takes the first frame at or after each whole second, so frame 0 of the slot is
    the one it would have chosen. Whitespace-only separators belonging to dropped
    images are dropped with them; any real text (the timestamp line, a user query)
    is always kept.
    """
    pieces = record["user_raw"].split("<image>")
    paths = iter(record["image_paths"])
    content = []
    kept = 0
    for index, piece in enumerate(pieces):
        capped = images_per_step > 0 and kept >= images_per_step
        if piece and not (capped and not piece.strip()):
            content.append({"type": "text", "text": piece})
        if index < len(pieces) - 1:
            raw_image_path = str(next(paths))
            if include_images and not capped:
                kept += 1
                # ``video-frame::<second>::<video>`` is a virtual frame URI used
                # by the streaming suite when a canonical pre-extracted frame is not
                # present.  The streaming worker materialises it under node-local
                # the local frame cache; treating it as a Path would corrupt the URI by
                # prepending cwd and resolving it on NFS.
                if raw_image_path.startswith("video-frame::"):
                    rendered_image_path = raw_image_path
                else:
                    image_path = Path(raw_image_path)
                    if not image_path.is_absolute():
                        image_path = Path.cwd() / image_path
                    rendered_image_path = str(image_path.resolve())
                content.append(
                    {
                        "type": "image",
                        "image": rendered_image_path,
                        "max_pixels": MAX_PIXELS,
                        "min_pixels": MIN_PIXELS,
                    }
                )
    return content

def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows
