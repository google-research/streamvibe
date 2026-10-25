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

"""Chunk-scoped, prefix-aware prompt assembly for StreamVibe Harness.

    [ system ]
    [ chunk header: Memory Text + carried active query ]   <- constant per chunk
    [ turn s   : frame (+ any query typed on it) ]         <- append-only
    [ turn s   : the model's own action              ]
      ...
    [ turn t   : frame                               ]     <- the step being decided

The engine prefills the Memory Text once at the
start of each chunk, and every later step in that chunk only has to compute the
KV of the new frame and the previous response, because everything before it is
a byte-identical prefix that vLLM's prefix cache already holds.

That only works if the rendering is PREFIX-MONOTONE inside a chunk: the string
built for turn t must be a strict prefix of the string built for turn t+1.  Two
rules follow, and both are load-bearing:

  * the header is snapshotted at the boundary and never recomputed inside the
    chunk (see ThreeTierMemory.memory_text);
  * a turn renders identically whether it is the current turn or a past one, so
    nothing may be appended to "the turn being decided".  In particular the
    active query is NOT re-stated on the current frame -- it is either already
    in the chunk (typed on an earlier turn of this chunk) or in the header
    (carried in from an earlier chunk).

The prefix-invariance contract applies directly to rendered
strings; if a change here breaks it, prefix caching silently stops working and
the only symptom is that the latency numbers get four times worse.
"""
from __future__ import annotations

from .architecture_runtime import transform_epoch_header

import re

MEMORY_HEADER = "[VIDEO MEMORY]"
QUERY_HEADER = "[ACTIVE QUERY]"
STANDING_REQUEST_HEADER = "[STANDING REQUEST]"
CONTROL_SENTINEL = "__STREAMING_HARNESS_PRIVATE_CONTROL__\n"


def _with_turn_extra(content, extra: str):
    if not extra:
        return content
    if isinstance(content, list):
        return [*content, {"type": "text", "text": extra}]
    return f"{content}\n{extra}" if content else extra


def _append_control(messages: list[dict], content, extra: str,
                    control_role: str) -> None:
    """Append private harness state separately from the real user utterance.

    ``user`` preserves the historical layout for ablations. ``system`` uses a
    sentinel user transport because the interaction model's Qwen chat template silently drops
    later system messages; :func:`rewrite_control_roles` changes the serialized
    role marker to system and removes the sentinel before inference. The real
    timestamped user content always remains a separate, untouched message.
    """
    if control_role == "system":
        if extra:
            messages.append({"role": "user", "content": CONTROL_SENTINEL + extra})
        if content:
            messages.append({"role": "user", "content": content})
        return
    if control_role == "assistant":
        if extra:
            messages.append({"role": "assistant", "content": extra})
        if content:
            messages.append({"role": "user", "content": content})
        return
    if control_role != "user":
        raise ValueError(f"unsupported control_role {control_role!r}")
    content = _with_turn_extra(content, extra)
    if content:
        messages.append({"role": "user", "content": content})


def rewrite_control_roles(chat: str) -> str:
    """Turn sentinel transports into real system-role segments on the wire."""
    needle = "<|im_start|>user\n" + CONTROL_SENTINEL
    return (chat or "").replace(needle, "<|im_start|>system\n")


def build_chunk_header(memory_text: str, carried_query, card: str = "",
                       query_projection: str = "waiting") -> str:
    """The constant block for one chunk. Empty string means no header turn.

    ``card`` is the harness-state card (harness/card.py).  It is rendered once
    per cache epoch exactly like the memory text, so adding it keeps the
    prefix-monotone property; see EpochGuard for the runtime check.
    """
    parts: list[str] = []
    if card:
        parts.append(card)
    if memory_text:
        parts.append(f"{MEMORY_HEADER}\nThe frames below are only the most recent "
                     f"part of the stream. Earlier content is available only as "
                     f"this memory:\n{memory_text}")
    if carried_query is not None:
        if query_projection == "waiting":
            parts.append(f"{QUERY_HEADER}\nThe user asked at {carried_query.second:.0f}s "
                         f"and is still waiting: {carried_query.text}")
        elif query_projection == "standing":
            parts.append(
                f"{STANDING_REQUEST_HEADER}\nOriginal request at "
                f"{carried_query.second:.0f}s; it remains in force only for "
                f"future distinct evidence: {carried_query.text}")
        else:
            raise ValueError(f"unknown query projection {query_projection!r}")
    return "\n\n".join(parts)


def response_epoch_start(base_start: int, last_response_turn: int | None,
                         enabled: bool, lookback_turns: int = 0) -> int:
    """Causal replay start for a response-segmented cache epoch.

    A raw response at turn ``r`` may only affect prompts from ``r + 1`` onward.
    Starting at ``r`` retains both the evidence frame that produced the reply
    and the model's exact committed reply, while older reply bodies and visual
    clutter no longer accumulate.  This is context projection before the next
    the interaction model decision; it never changes the response at ``r``.
    """
    if not enabled or last_response_turn is None:
        return base_start
    return max(base_start, int(last_response_turn) - max(0, int(lookback_turns)))


def response_similarity(left: str, right: str) -> float:
    """Multilingual lexical overlap for already committed response receipts."""
    def tokens(text: str) -> set[str]:
        value = (text or "").lower()
        words = set(re.findall(r"[a-z0-9]+", value))
        han = "".join(re.findall(r"[\u4e00-\u9fff]", value))
        return words | {han[i:i + 2] for i in range(max(0, len(han) - 1))}

    a, b = tokens(left), tokens(right)
    return len(a & b) / max(1, len(a | b))


def response_epoch_due(spoke_turns: list[int], anchor: int | None,
                       enabled: bool, trigger: str = "every_response",
                       burst_gap_turns: int = 1,
                       spoke_texts: list[str] | None = None,
                       repeat_threshold: float = 0.9,
                       repeat_history: int = 8,
                       repeat_gap_turns: int = 16) -> bool:
    """Whether past raw replies should open the *next* context epoch.

    ``burst`` and ``burst_then_every`` are intentionally not response gates:
    the first and second raw replies are already committed unchanged.  They
    only prevent that reply run from becoming an ever-growing demonstration in
    later the interaction model prompts.
    """
    if not enabled or not spoke_turns or spoke_turns[-1] == anchor:
        return False
    if trigger == "every_response":
        return True
    if trigger == "burst":
        return (len(spoke_turns) >= 2
                and spoke_turns[-1] - spoke_turns[-2] <= int(burst_gap_turns))
    if trigger == "repeated_response":
        texts = spoke_texts or []
        if len(spoke_turns) < 2 or len(texts) != len(spoke_turns):
            return False
        latest_turn, latest_text = spoke_turns[-1], texts[-1]
        start = max(0, len(spoke_turns) - 1 - max(1, int(repeat_history)))
        return any(
            latest_turn - spoke_turns[index] <= int(repeat_gap_turns)
            and response_similarity(latest_text, texts[index])
            >= float(repeat_threshold)
            for index in range(start, len(spoke_turns) - 1)
        )
    if trigger == "loop_then_every":
        # Before the first loop boundary this is exactly repeated_response.
        # Once that boundary has armed reply-body compression, every later raw
        # reply opens its own causal epoch so changing which body is "latest"
        # can never mutate a cached prefix in place.
        if anchor is not None:
            return True
        return response_epoch_due(
            spoke_turns, anchor, enabled, "repeated_response",
            burst_gap_turns, spoke_texts, repeat_threshold,
            repeat_history, repeat_gap_turns)
    if trigger == "burst_then_every":
        # Before the first burst boundary this is exactly burst: a single
        # committed response or two separated responses retain continuous
        # causal visual context. Once an adjacent committed response pair has
        # armed segmentation, every later raw reply opens its own causal epoch.
        if anchor is not None:
            return True
        return response_epoch_due(
            spoke_turns, anchor, enabled, "burst", burst_gap_turns,
            spoke_texts, repeat_threshold, repeat_history, repeat_gap_turns)
    raise ValueError(f"unknown response epoch trigger {trigger!r}")


def is_silence(action: str) -> bool:
    a = (action or "").lstrip()
    return a.startswith("</silence>") or a.startswith("<|silent|>")


def history_response(action: str, mode: str = "full", *, latest: bool = False) -> str:
    """Project a committed response into a later pre-decision prompt.

    The raw response remains the committed/scored action and is retained in the
    trace and external memory.  This function only controls its bounded replay
    inside the rolling dialogue window.  ``action_marker`` preserves the model's
    own control token without copyable prose; ``receipt`` also states that the
    prior turn was delivered. ``state_only`` omits the assistant turn entirely:
    delivery is represented by the epoch-frozen Current State instead. Neither
    mode changes or filters an output.
    """
    if mode == "full":
        return action
    if mode == "action_marker":
        return "</response>"
    if mode == "receipt":
        return "</response> Prior response delivered."
    if mode == "active_receipt":
        return "</response> One event covered; standing request remains active."
    if mode == "latest_full":
        return action if latest else "</response>"
    if mode == "state_only":
        return ""
    raise ValueError(f"unsupported response history mode {mode!r}")


def latest_response_turn(own_actions: dict, turn: int) -> int | None:
    """Newest already-committed response strictly before ``turn``."""
    hits = [t for t, action in own_actions.items()
            if t < turn and not is_silence(action)]
    return max(hits) if hits else None


def build_harness_messages(
    records,
    turn: int,
    own_actions: dict,
    short_start: int,
    chunk_header: str,
    system_prompt: str,
    user_content,
    images_per_step: int = 1,
    silence_in_history: bool = False,
    turn_extras: dict[int, str] | None = None,
    control_role: str = "user",
    current_system_extra: str = "",
    response_history_mode: str = "full",
    silence_history_mode: str | None = None,
    silence_history_start: int | None = None,
    evidence_start: int | None = None,
):
    """Messages for the decision at `turn`.

    `user_content` is `run_inference._user_content`, injected rather than
    imported so this module stays free of torch/transformers and can be unit
    tested on a login node.
    """


    system_content = system_prompt
    if chunk_header:
        system_content += "\n\n" + transform_epoch_header(chunk_header)
    if current_system_extra:
        system_content += "\n\n" + current_system_extra
    messages = [{"role": "system", "content": system_content}]
    # A response epoch normally starts at the most recent committed reply.  A
    # bounded evidence tail may begin earlier, but it is deliberately image
    # only: replaying the old user text, private deltas, or assistant bodies is
    # precisely the self-demonstration path that response segmentation removes.
    # The evidence prefix is fixed for the whole epoch, so later turns remain
    # append-only and can still reuse their prefill cache.
    visual_start = int(short_start if evidence_start is None else evidence_start)
    if visual_start < 0 or visual_start > int(short_start):
        raise ValueError(
            "evidence_start must be non-negative and no later than short_start")
    for t in range(visual_start, int(short_start)):
        raw_content = user_content(
            records[t], include_images=True,
            images_per_step=images_per_step)
        if not isinstance(raw_content, list):
            continue
        visual_content = [
            part for part in raw_content
            if isinstance(part, dict) and part.get("type") == "image"
        ]
        if visual_content:
            messages.append({"role": "user", "content": visual_content})
    latest_response = latest_response_turn(own_actions, turn)
    silence_mode = (silence_history_mode
                    or ("all" if silence_in_history else "drop"))
    if silence_mode not in {"drop", "all", "first_per_epoch"}:
        raise ValueError(f"unsupported silence history mode {silence_mode!r}")
    receipt_start = max(
        int(short_start),
        int(short_start if silence_history_start is None
            else silence_history_start),
    )
    first_silence = None
    if silence_mode == "first_per_epoch":
        first_silence = next((t for t in range(receipt_start, turn)
                              if is_silence(own_actions[t])), None)
    for t in range(short_start, turn + 1):
        content = user_content(records[t], include_images=True, images_per_step=images_per_step)
        _append_control(messages, content, (turn_extras or {}).get(t, ""),
                        control_role)
        if t < turn:
            action = own_actions[t]
            if (is_silence(action)
                    and (t < receipt_start
                         or silence_mode == "drop"
                         or (silence_mode == "first_per_epoch"
                             and t != first_silence))):
                # Omit this silent assistant turn while retaining its frame.
                # Existing turns stay unchanged, preserving prefix-cache reuse.
                continue
            if (not is_silence(action)
                    and response_history_mode == "state_only"):
                # A natural-language receipt such as "Prior response
                # delivered" is a copyable assistant demonstration.  the interaction model
                # learned to emit that receipt as a fresh answer, creating a
                # self-response loop.  The committed delivery remains in
                # StateRuntime and the immutable trace; only its replay in the
                # next pre-decision prompt is removed.
                continue
            messages.append({
                "role": "assistant",
                "content": history_response(
                    action, response_history_mode, latest=t == latest_response),
            })
    return messages


def build_original32_messages(
    records,
    turn: int,
    own_actions: dict,
    memory_header: str,
    system_prompt: str,
    user_content,
    images_per_step: int = 1,
    window: int = 32,
    turn_extras: dict[int, str] | None = None,
    stride: int = 1,
    anchors: tuple = (),
    control_role: str = "user",
    current_system_extra: str = "",
    response_history_mode: str = "full",
):
    """The historical anchor-self context, optionally plus harness memory.

    This deliberately mirrors ``build_anchor_self_messages`` in the canonical
    streaming streaming evaluator: keep the rolling visual tail, every user query,
    and every prior non-empty self response; omit silent assistant turns.  The
    paired harness arm calls this exact function with a non-empty
    ``memory_header``, making that block the only prompt-level difference.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    # As above, past memory is system context. A separate user message is a
    # learned trigger for an assistant response and confounds the paired arm.
    system_content = system_prompt
    if memory_header:
        system_content += "\n\n" + memory_header
    if current_system_extra:
        system_content += "\n\n" + current_system_extra
    messages = [{"role": "system", "content": system_content}]
    latest_response = latest_response_turn(own_actions, turn)
    visual_start = max(0, int(turn) - int(window) + 1)
    # Pre-decision frame selection. `stride` thins the tail (the current turn is
    # always kept); `anchors` are extra turns to keep -- the frames of the last
    # committed responses, so the model can compare "then" with "now".
    if stride > 1 or anchors:
        keep = {t for t in range(visual_start, turn + 1)
                if (turn - t) % max(1, stride) == 0}
        keep.add(turn)
        keep |= {int(a) for a in anchors if visual_start <= int(a) <= turn}
    else:
        keep = None
    for t, record in enumerate(records[: turn + 1]):
        own = own_actions.get(t, "")
        prior_help = (
            t < turn
            and own.lstrip().startswith("</response>")
            and bool(own.split("</response>", 1)[-1].strip())
        )
        raw_without_markup = str(record.get("user_raw") or "").replace("<image>", "")
        has_query = any(
            ch not in " <>.0123456789seconds\n\t" for ch in raw_without_markup
        )
        if t < visual_start and not prior_help and not has_query:
            continue
        show_image = t >= visual_start and (keep is None or t in keep)
        content = user_content(
            record,
            include_images=show_image,
            images_per_step=images_per_step,
        )
        _append_control(messages, content, (turn_extras or {}).get(t, ""),
                        control_role)
        if prior_help:
            messages.append({
                "role": "assistant",
                "content": history_response(
                    own, response_history_mode, latest=t == latest_response),
            })
    return messages


def images_of(messages) -> list[str]:
    """Frame paths in the order the chat template will consume them."""
    paths = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "image":
                    paths.append(part["image"])
    return paths
