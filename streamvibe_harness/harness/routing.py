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

"""Request-only routing for cache-stable pre-decision contracts.

The router never sees dataset/family labels, frames, gold actions, or the interaction model
outputs. It is a small serving projection of P.request + recalled E.
"""
from __future__ import annotations

import re

from .criteria import (
    EVENT,
    EXPLAIN,
    HAZARD,
    MCQ,
    MILESTONE,
    MISTAKE,
    NARRATE,
    NEXT_STEP,
    SUMMARY,
    TYPES,
    is_question,
    parse_instruction,
)


# A route controls only how the already-independent criteria are projected.
# It must never delete or replace a criterion. The order resolves which
# presentation contract to use when several active criteria coexist.
CRITERION_PROFILE_PRIORITY = (
    MCQ, HAZARD, MISTAKE, EVENT, NARRATE, EXPLAIN, NEXT_STEP,
    MILESTONE, SUMMARY,
)


DIGITAL_TASK = re.compile(
    r"\b(browser|web|website|slide|slides|presentation|powerpoint|excel|"
    r"spreadsheet|chart|axis|axes|pdf|menu|dialog|software|app|desktop)\b",
    re.I,
)

DIGITAL_GUIDANCE_INTENT = re.compile(
    r"\b(help me|show me|walk me through|guide me|navigate|locate|find|open|"
    r"select|click|convert|embed|configure|set up)\b",
    re.I,
)

MCQ_OPTIONS = re.compile(r"\boptions?\s*:", re.I)
GENERIC_EVIDENCE_WORDS = {
    "the", "and", "for", "with", "after", "before", "into", "from",
    "question", "what", "which", "who", "where", "when", "why", "how",
    "person", "people", "someone", "they", "them", "their", "she", "her",
    "he", "him", "his", "doing", "does", "did", "action", "activity",
    "perform", "performing", "happen", "happening", "beginning", "start",
    "video", "clip", "scene", "visual", "evidence", "sufficient",
}


def request_evidence_focus(request: str, mode: str = "full_request") -> str:
    """Return request text safe for evidence retrieval, never answer choice.

    In ``question_stem`` mode an MCQ's option block and response-format
    instruction are excluded.  Distractor wording may be shown to the interaction model as part
    of the real request, but it must not decide which external-memory facts B
    retrieves and puts before the model.
    """
    value = " ".join((request or "").split())
    if mode == "full_request":
        return value
    if mode != "question_stem":
        raise ValueError(f"unknown evidence focus mode {mode!r}")
    match = MCQ_OPTIONS.search(value)
    if match:
        value = value[:match.start()]
    value = re.sub(r"^\s*question\s*:\s*", "", value, flags=re.I)
    return value.strip()


def has_specific_evidence_focus(request: str) -> bool:
    """Whether the question stem has a concrete, retrievable video anchor."""
    stem = request_evidence_focus(request, "question_stem")
    words = {
        word for word in re.findall(r"[a-z0-9]+", stem.lower())
        if len(word) > 2 and word not in GENERIC_EVIDENCE_WORDS
    }
    if words:
        return True
    han = re.sub(
        r"(什么|哪个|哪一个|谁|哪里|哪儿|何时|为什么|怎么|如何|正在|做什么|"
        r"动作|行为|视频|画面|片段|开始|开头|证据|足够|这个|那个|他们|她|他)",
        "", "".join(re.findall(r"[\u4e00-\u9fff]+", stem)))
    return bool(han.strip())


def memory_ranges_redundant(
        memory_dump: dict,
        visible_start_second: float,
        visible_end_second: float,
) -> bool:
    """Whether every rendered video summary is already in the visual window.

    This is a provenance check, not a semantic judgment.  An empty summary set
    is redundant by definition; request/procedure state must not manufacture a
    claim that older visual evidence exists.  If any mid/long summary reaches
    outside the interaction model's current causal frame interval, the full
    Memory projection is retained.
    """
    ranges = [
        (float(item["t0"]), float(item["t1"]))
        for tier in ("long", "mid")
        for item in (memory_dump.get(tier) or [])
        if item.get("t0") is not None and item.get("t1") is not None
    ]
    epsilon = 1e-6
    return all(
        t0 + epsilon >= float(visible_start_second)
        and t1 <= float(visible_end_second) + epsilon
        for t0, t1 in ranges
    )


def request_profile(request: str) -> str:
    """Return a semantic profile using only request text and criteria."""
    if not (request or "").strip():
        return "passive_observation"
    active = {
        op.target for op in parse_instruction(request or "", first=True)
        if op.kind == "add"
    }
    if HAZARD in active or (MISTAKE in active and NEXT_STEP not in active):
        return "monitoring"
    if EVENT in active:
        return "event_watch"
    if NARRATE in active:
        return "narration"
    if EXPLAIN in active:
        return "explanation"
    if NEXT_STEP in active:
        return "digital_guidance" if DIGITAL_TASK.search(request or "") else "physical_guidance"
    if is_question(request):
        return "question_answering"
    if DIGITAL_TASK.search(request or "") and DIGITAL_GUIDANCE_INTENT.search(request or ""):
        return "digital_guidance"
    return "fallback"


def routed_prompt_profile(request: str, router: dict, fallback: str) -> tuple[str, str]:
    """Return ``(semantic_profile, system_prompt_profile)`` for one request."""
    semantic = request_profile(request)
    criterion_profiles = router.get("criteria_profiles") or {}
    if criterion_profiles:
        active = {
            op.target for op in parse_instruction(request or "", first=True)
            if op.kind == "add"
        }
        semantic = next(
            (str(profile) for criterion, profile in criterion_profiles.items()
             if criterion in active), semantic)
    mapping = router.get("profiles") or {}
    return semantic, str(mapping.get(semantic) or mapping.get("fallback") or fallback)


def routed_state_profile(
        request: str,
        active_criteria,
        question_pending: bool,
        router: dict,
        fallback: str,
        *,
        temporary_question_profile: bool = False,
) -> tuple[str, str]:
    """Route from the complete causal request state, never the latest clause.

    The criterion set remains owned by CriterionSet; this function selects only
    a prompt projection. Additive instructions therefore cannot retire or hide
    an earlier standing criterion.
    """
    active = set(active_criteria)
    if question_pending or MCQ in active:
        semantic = "question_answering"
    elif HAZARD in active or MISTAKE in active:
        semantic = "monitoring"
    elif EVENT in active:
        semantic = "event_watch"
    elif NARRATE in active:
        semantic = "narration"
    elif EXPLAIN in active:
        semantic = "explanation"
    elif NEXT_STEP in active:
        semantic = (
            "digital_guidance"
            if DIGITAL_TASK.search(request or "") else "physical_guidance")
    elif MILESTONE in active or SUMMARY in active:
        semantic = "monitoring"
    else:
        semantic = request_profile(request)

    configured = router.get("criteria_profiles") or {}
    # A one-shot scene question is a temporary projection over the standing
    # criteria, not another standing criterion.  With the opt-in lifecycle,
    # it must outrank the standing profile until a response is committed.  The
    # old priority behavior remains byte-for-byte selectable for existing
    # policies by leaving the flag disabled.
    if question_pending and temporary_question_profile:
        semantic = str(configured.get("question") or "question_answering")
    else:
        for criterion in CRITERION_PROFILE_PRIORITY:
            if criterion in active and criterion in configured:
                semantic = str(configured[criterion])
                break
    mapping = router.get("profiles") or {}
    return semantic, str(mapping.get(semantic) or mapping.get("fallback") or fallback)


def standing_route_after_question(
        current_profile: str,
        standing_request: str,
        active_criteria,
        question_pending: bool,
        router: dict,
        fallback: str,
) -> tuple[str, str] | None:
    """Return the standing route after a temporary question is answered.

    This is a pre-decision state transition.  It reads only the committed
    request state from the previous turn; it neither inspects nor changes the
    current model proposal.  A standing MCQ (or another criterion explicitly
    routed to question_answering) naturally remains on the question profile.
    """
    if (not bool(router.get("temporary_question_profile", False))
            or current_profile != "question_answering"
            or question_pending):
        return None
    restored = routed_state_profile(
        standing_request, active_criteria, False, router, fallback,
        temporary_question_profile=True)
    return None if restored[0] == current_profile else restored


def scoped_silence_history(
        default: str,
        by_profile: dict,
        by_criterion: dict,
        profile: str,
        active_criteria: set[str] | dict[str, float],
        focus_gated_criteria=(),
        request: str = "",
        initial_presence_gated_criteria=(),
        initial_criteria=(),
) -> tuple[str, str]:
    """Resolve bounded silence history from causal request state.

    Criterion scope takes precedence over the coarser semantic profile.  The
    canonical criterion order makes multi-intent requests deterministic and
    independent of JSON key order.  The returned source is recorded in traces
    so a history change can be attributed without inspecting model output.
    """
    active = set(active_criteria)
    focus_gated = set(focus_gated_criteria)
    initial_gated = set(initial_presence_gated_criteria)
    initial = set(initial_criteria)
    for criterion in TYPES:
        if criterion in active and criterion in by_criterion:
            if criterion in initial_gated and criterion not in initial:
                continue
            if (criterion in focus_gated
                    and not has_specific_evidence_focus(request)):
                continue
            return str(by_criterion[criterion]), f"criterion:{criterion}"
    if profile in by_profile:
        return str(by_profile[profile]), f"profile:{profile}"
    return str(default), "default"


def response_phase_profile(now: float, last_response: float | None,
                           base_profile: str, cooldown: dict) -> tuple[str, dict]:
    """Select a pre-decision contract from past committed-response state.

    The selector never sees the current the interaction model proposal. A response committed at
    ``last_response`` opens a short P.said phase on later decisions; the interaction model still
    receives the raw current frame and owns the response/silence token. The
    returned metadata is recorded per step so the phase cannot be mistaken for
    a post-generation gate during audit.
    """
    enabled = bool(cooldown.get("enabled", False))
    duration = max(0.0, float(cooldown.get("seconds", 0)))
    profile = str(cooldown.get("profile") or base_profile)
    age = None if last_response is None else float(now) - float(last_response)
    active = bool(enabled and age is not None and 0.0 < age <= duration)
    selected = profile if active else base_profile
    return selected, {
        "active": active,
        "age_s": age,
        "duration_s": duration,
        "base_profile": base_profile,
        "selected_profile": selected,
    }
