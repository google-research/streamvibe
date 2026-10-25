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

"""Causal streaming prompts and supervised target selection for StreamVibe-RL.

Includes message construction, causal memory-frame selection, ground-truth
prefix assembly, and critical-step selection.
"""

from __future__ import annotations

import os

# --------------------------------------------------------------------------- #
# Streaming system prompt and control tokens
# --------------------------------------------------------------------------- #
NATIVE_SYSTEM = (
    "You are a real-time video streaming assistant observing a continuous camera feed "
    "frame by frame. The last frame represents the current moment.\n"
    "## Action Format\n"
    "At every inference step you MUST choose exactly one of the following two actions:\n"
    "**Stay silent** — output ONLY:\n"
    "</silence>\n"
    "Choose this when nothing noteworthy has changed in the scene, no user query is "
    "pending, or there is nothing useful to say.\n"
    "**Speak** — output the token followed by a concise reply:\n"
    "</response> Your reply here.\n"
    "Choose this when you observe something worth reporting or a significant state "
    "change, or when you can answer a user question based on available evidence."
)

# </silence> and </response> are added SPECIAL tokens on this checkpoint family
# (verified against the CT base checkpoint's tokenizer_config.json).
# Decision-token ids are model-family specific (Qwen3-VL: 151669/151670,
# Qwen3.5: 248077/248078).  Override via env for non-default backbones.
SILENCE_TOKEN_ID = int(os.environ.get("CT_SILENCE_TOKEN_ID", "151669"))
RESPONSE_TOKEN_ID = int(os.environ.get("CT_RESPONSE_TOKEN_ID", "151670"))

SILENCE_ACTION = "</silence>"
RESPONSE_MARKER = "</response>"
SILENCE_MARKER = "</silence>"


def classify_action(text: str) -> tuple[str, str]:
    """Classify one generated turn and return (kind, canonical_action).

    The original classifies by FIRST generated token id.  Over the OpenAI chat
    API we only get text back; because the two control tokens are added special
    tokens, "first token is the control token" is equivalent to "decoded text
    starts with the control string with no leading characters".
    """
    stripped = text.strip() if text else ""
    raw = text if text else ""
    if raw.startswith(SILENCE_ACTION):
        return "silence", SILENCE_ACTION
    if raw.startswith(RESPONSE_MARKER):
        body = raw[len(RESPONSE_MARKER):].strip()
        return "response", f"{RESPONSE_MARKER} {body}".rstrip()
    return "malformed", stripped


HISTORY_MODE = os.environ.get("CT_HISTORY_MODE", "anchored")
HISTORY_WINDOW = int(os.environ.get("CT_HISTORY_WINDOW", "32"))
HISTORY_SILENCE = os.environ.get("CT_HISTORY_SILENCE", "keep")


def gt_causal_memory_frames(
    at: int,
    actions: dict[int, str],
    recent_s: int,
    anchor_s: int,
    max_images: int | None = None,
) -> list[int]:
    """Return past responses, sparse old anchors, and a fixed dense recent window.

    The selection depends only on actions already taken before ``at``. It never
    grows from the artificial short-trajectory boundary, and it never includes
    a future frame or a current/future action.
    """
    if recent_s < 0 or anchor_s <= 0:
        raise ValueError("require recent_s >= 0 and anchor_s > 0")

    if HISTORY_MODE == "none":
        return [int(at)]

    if HISTORY_MODE == "contiguous":
        # Images: the contiguous window ending at `at`.  Out-of-window help
        # points are added by out_of_window_helps() as TEXT-ONLY turns.
        window = max(1, HISTORY_WINDOW)
        start = max(0, int(at) - window + 1)
        ordered = list(range(start, int(at) + 1))
        if max_images and len(ordered) > int(max_images):
            ordered = ordered[-int(max_images):]
        return ordered

    recent_start = max(0, int(at) - int(recent_s))
    recent = set(range(recent_start, int(at) + 1))
    anchors = set(range(0, recent_start, int(anchor_s)))
    responses = {
        int(frame)
        for frame, action in actions.items()
        if int(frame) < int(at) and str(action).lstrip().startswith(RESPONSE_MARKER)
    }
    frames = sorted(frame for frame in recent | anchors | responses if 0 <= frame <= int(at))
    if not max_images or len(frames) <= int(max_images):
        return frames

    max_images = int(max_images)
    if max_images <= 0:
        raise ValueError("max_images must be positive when specified")

    # The multimodal limit is hard. Preserve the decision-critical dense recent
    # context first, then every feasible past response frame; only silence
    # anchors are thinned in normal operation.
    keep = set(sorted(recent)[-max_images:])
    remaining = max_images - len(keep)
    if remaining > 0:
        response_candidates = sorted(responses - keep, reverse=True)
        keep.update(response_candidates[:remaining])
    remaining = max_images - len(keep)
    if remaining > 0:
        anchor_candidates = sorted(anchors - keep, reverse=True)
        keep.update(anchor_candidates[:remaining])
    return sorted(keep)


def out_of_window_helps(at: int, actions: dict[int, str], window_frames: list[int]) -> list[int]:
    """Past help turns older than the visual window (kept text-only, as eval does)."""
    if HISTORY_MODE not in ("contiguous", "none"):
        return []
    visible = set(window_frames)
    return sorted(
        int(frame)
        for frame, action in (actions or {}).items()
        if int(frame) < int(at)
        and int(frame) not in visible
        and str(action).lstrip().startswith(RESPONSE_MARKER)
    )


def hp_time(hp: dict) -> float:
    value = hp.get("optimal_s", hp.get("start_s"))
    if value is None:
        raise ValueError(f"help point has no usable time: {hp}")
    return float(value)


def gt_prefix(
    gt: dict, start: int, recent_s: int, anchor_s: int
) -> tuple[list[int], dict[int, str]]:
    """Sparse causal teacher-forced history strictly before a short trajectory."""
    if start <= 0:
        return [], {}
    historical = gt.get("history_help_points") or []
    response_at: dict[int, list[str]] = {}
    for hp in historical:
        frame = int(round(hp_time(hp)))
        if not (0 <= frame < start):
            continue
        text = (hp.get("reference") or "").strip()
        if text:
            response_at.setdefault(frame, []).append(text)

    recent_start = max(0, start - recent_s)
    frames = set(range(recent_start, start))
    frames.update(range(0, recent_start, anchor_s))
    # Every earlier gold response is retained even when it is not an anchor.
    frames.update(response_at)
    ordered = sorted(f for f in frames if f < start)
    actions = {}
    for frame in ordered:
        texts = response_at.get(frame)
        actions[frame] = (
            f"{RESPONSE_MARKER} {' '.join(texts)}" if texts else SILENCE_ACTION
        )
    return ordered, actions


# --------------------------------------------------------------------------- #
# Chat layout (port of build_messages_native)
# --------------------------------------------------------------------------- #
def build_messages_native(
    request: str,
    at_frame: int,
    window_frames: list[int],
    actions: dict[int, str] | None = None,
    interval_s: float = 1.0,
    textonly_frames: list[int] | None = None,
) -> list[dict]:
    """Cumulative realtime chat: one (user frame, assistant action) pair per second.

        system      NATIVE_SYSTEM
        user        {request}\\n<0.0 seconds>\\n<image>      <- request rides the FIRST
        assistant   </silence>                                  visible frame's turn
        ...
        user        <t.0 seconds>\\n<image>                  <- current frame, no action

    Invariants (raise rather than degrade): every visible frame BEFORE the
    current one carries exactly one assistant action (silence included); the
    current frame carries none.
    """
    actions = actions or {}
    textonly = list(textonly_frames or [])
    if not window_frames:
        raise ValueError("at least one visible frame is required")
    if list(window_frames) != sorted(window_frames) or len(set(window_frames)) != len(window_frames):
        raise ValueError("window_frames must be unique and increasing")
    if window_frames[-1] != at_frame:
        raise ValueError(f"window must end at the frame being decided: {window_frames[-1]} != {at_frame}")

    msgs: list[dict] = [{"role": "system", "content": NATIVE_SYSTEM}]
    ordered = sorted({*window_frames, *textonly} | {int(at_frame)})
    # The instruction rides the first turn that actually survives; dropping
    # silence turns must never drop the user's request with them.
    head_pending = True
    for f in ordered:
        head = f'The user asked you for: "{request}"\n' if head_pending else ""
        img = "\n<image>" if f in set(window_frames) else ""
        if f == at_frame:
            if f in actions:
                raise ValueError(f"the frame being decided already has an action at f={f}")
            msgs.append({"role": "user", "content": f"{head}<{f * interval_s:.1f} seconds>{img}"})
            head_pending = False
            continue
        action = actions.get(f)
        if action is None:
            if HISTORY_MODE in ("contiguous", "none"):
                # A contiguous window reaches back before the trajectory start,
                # where no action was ever taken.  Those seconds are silent by
                # definition -- the eval harness likewise shows the frame with
                # no assistant reply.  Only the legacy anchored mode, which
                # selects exactly the frames that carry actions, treats a gap
                # as a bug.
                action = SILENCE_MARKER
            else:
                raise ValueError(
                    f"missing historical assistant action at f={f}; every visible frame "
                    "before the current one must carry one (silence included)"
                )
        is_silence = not str(action).lstrip().startswith(RESPONSE_MARKER)
        # Eval keeps the framed user turn for every second in the window and
        # appends an assistant message ONLY at help points -- consecutive user
        # turns with no assistant between them are expected, not a violation.
        msgs.append({"role": "user", "content": f"{head}<{f * interval_s:.1f} seconds>{img}"})
        head_pending = False
        if is_silence and HISTORY_SILENCE == "drop":
            continue
        msgs.append({"role": "assistant", "content": str(action)})
    return msgs


def split_images(text: str) -> list[dict]:
    """Split on the literal <image> placeholder (port of _split_images)."""
    content: list[dict] = []
    for i, chunk in enumerate(text.split("<image>")):
        if i != 0:
            content.append({"type": "image"})
        if chunk:
            content.append({"type": "text", "text": chunk})
    return content


# --------------------------------------------------------------------------- #
# Trainable-turn selection (port of _pick_gt_critical_steps)
# --------------------------------------------------------------------------- #
def pick_gt_critical_steps(
    steps: list[dict],
    turns: list[tuple],
    target_help_points: list[dict],
    count: int,
    rng,
) -> list[dict]:
    """Keep response-causing and GT-critical decisions among the trained rows.

    ``steps`` and ``turns`` are parallel arrays (one entry per decided second).
    Order of retention: exact target instants, then sampled responses (nearest
    to a target first), then the +/-2s opportunities, then shuffled remote
    silence, then anything left. Returns steps sorted by original index.
    """
    if not steps:
        return []
    if len(steps) != len(turns):
        raise ValueError(f"steps/turns must be parallel: {len(steps)} != {len(turns)}")
    target_times = [int(round(hp_time(h))) for h in target_help_points]
    picked: list[int] = []

    def add(index: int) -> None:
        if 0 <= index < len(steps) and index not in picked and len(picked) < count:
            picked.append(index)

    # Always retain the exact target decision.
    for t in target_times:
        candidates = [i for i, step in enumerate(steps) if step["at"] == t]
        if candidates:
            add(candidates[0])

    use_turn_rewards = os.environ.get("STREAMVIBE_REWARD_MODE") == "turn_local"
    asym = os.environ.get("CT_WINDOW_ASYM") == "1"
    early = float(os.environ.get("CT_WINDOW_EARLY_S", "4.0")) if asym else 2.0
    late = 2.0

    response_indices = [i for i, turn in enumerate(turns) if turn[1] == "response"]
    response_indices.sort(
        key=lambda i: min((abs(steps[i]["at"] - t) for t in target_times), default=0)
    )
    for i in response_indices:
        add(i)


    if use_turn_rewards:
        deltas = sorted(
            [d for d in range(-int(early), int(late) + 1) if d != 0], key=abs
        )
    else:
        deltas = (-2, -1, 1, 2)
    for delta in deltas:
        for t in target_times:
            candidates = [i for i, step in enumerate(steps) if step["at"] == t + delta]
            if candidates:
                add(candidates[0])

    # use_turn_rewards: guarantee far-silence rows a floor AFTER the window is covered.
    # Under the legacy order far silence ranked last and usually got ~2 of 11
    # slots; the use_turn_rewards reward finally gives those rows gradient, so starving them
    # would waste the change.  CT_FAR_SILENCE_MIN=0 keeps legacy behaviour.
    far_quota = int(os.environ.get("CT_FAR_SILENCE_MIN", "0"))
    if far_quota > 0 and target_times:
        far_silence = [
            i for i, turn in enumerate(turns)
            if turn[1] != "response"
            and all(not (-early <= steps[i]["at"] - t <= late) for t in target_times)
        ]
        rng.shuffle(far_silence)
        for i in far_silence[:far_quota]:
            add(i)

    negative = [
        i
        for i, step in enumerate(steps)
        if all(abs(step["at"] - t) > 2 for t in target_times)
    ]
    rng.shuffle(negative)
    for i in negative:
        add(i)
    remaining = [i for i in range(len(steps)) if i not in picked]
    rng.shuffle(remaining)
    for i in remaining:
        add(i)
    return [steps[i] for i in sorted(picked)]
