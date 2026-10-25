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

"""Parse and project private current-frame evidence before the online model.

The 4B tracker may run every second, but its prose is not automatically exposed
to the interaction model.  This module converts versioned structured results into a bounded status. Only
criterion matches carry natural-language evidence; ordinary progress is reduced
to enum state so it cannot become accidental commentary material.

Nothing here maps evidence to an interactive action.  The resulting status is
inserted before the interaction model, which still sees the raw frames and produces the sole
``silence``/``response`` token.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


NOVELTY = frozenset({"NONE", "CONTINUING", "NEW", "UNCERTAIN"})
TASK_EVIDENCE = frozenset({"NONE", "CONTEXT_ONLY", "CRITERION_MATCH", "UNCERTAIN"})
MATCH_KIND = frozenset({"TARGET", "COMPLETION", "ERROR", "TRANSITION",
                        "ANSWER_FACT", "NONE"})
GROUNDING = frozenset({"DIRECT", "PARTIAL", "NONE"})
LEGACY_FIELD_NAMES = (
    "CURRENT STATE", "VISUAL NOVELTY", "TASK EVIDENCE", "MATCH KIND", "EVIDENCE")
GROUNDED_FIELD_NAMES = (
    "CURRENT STATE", "VISUAL NOVELTY", "TASK EVIDENCE", "MATCH KIND",
    "GROUNDING", "EVIDENCE")
BLIND_ALIGNED_FIELD_NAMES = (
    "CURRENT STATE", "VISUAL NOVELTY", "TASK EVIDENCE", "MATCH KIND",
    "GROUNDING", "ANCHOR", "EVIDENCE")
FIELD_NAMES = BLIND_ALIGNED_FIELD_NAMES
REQUIRED_FIELD_NAMES = LEGACY_FIELD_NAMES
BLIND_MARKER = "[REQUEST-BLIND OBSERVATION]"
ALIGNMENT_MARKER = "[REQUEST ALIGNMENT]"


def event_profile_enabled(profile: str, profiles) -> bool:
    """Whether current-evidence normalisation is active for a request profile.

    An empty profile set preserves the historical all-profile behaviour.  A
    non-empty set is a causal request-semantic projection: it cannot inspect a
    dataset name, future frame, gold event, or interaction-model output.
    """
    allowed = set(profiles or ())
    return not allowed or profile in allowed


def _field(text: str, name: str) -> str:
    # Small aligned observers often serialize all fields on one line even when
    # asked for five lines. Stop at the next known label, not only a newline.
    next_label = "|".join(re.escape(value) for value in FIELD_NAMES)
    standalone = r"(?<!TASK\s)" if name == "EVIDENCE" else ""
    match = re.search(
        rf"(?:^|\s){standalone}{re.escape(name)}\s*:\s*(.*?)"
        rf"(?=\s+(?:{next_label})\s*:|$)",
        text or "", flags=re.I | re.S)
    return " ".join(match.group(1).split()) if match else ""


def _has_field(text: str, name: str) -> bool:
    standalone = r"(?<!TASK\s)" if name == "EVIDENCE" else ""
    return bool(re.search(
        rf"(?:^|\s){standalone}{re.escape(name)}\s*:", text or "", flags=re.I))


def _enum(text: str, name: str, allowed: frozenset[str], default: str) -> str:
    value = re.sub(r"[^A-Z_]+", "_", _field(text, name).upper()).strip("_")
    return value if value in allowed else default


def _normalise_bare_fields(text: str) -> str:
    """Accept compact legacy, grounded, and anchored observer schemas."""
    value = (text or "").strip()
    if any(re.search(rf"\b{re.escape(name)}\s*:", value, re.I)
           for name in FIELD_NAMES):
        return value
    lines = [" ".join(line.split()) for line in value.splitlines() if line.strip()]
    if len(lines) == 4:
        lines.append("NONE")
    if len(lines) not in {5, 6, 7}:
        return value

    def enum_value(item: str) -> str:
        return re.sub(r"[^A-Z_]+", "_", item.upper()).strip("_")

    if (enum_value(lines[1]) not in NOVELTY
            or enum_value(lines[2]) not in TASK_EVIDENCE
            or enum_value(lines[3]) not in MATCH_KIND):
        return value
    names = {5: LEGACY_FIELD_NAMES, 6: GROUNDED_FIELD_NAMES,
             7: BLIND_ALIGNED_FIELD_NAMES}[len(lines)]
    if len(lines) >= 6 and enum_value(lines[4]) not in GROUNDING:
        return value
    return "\n".join(f"{name}: {item}" for name, item in zip(names, lines))


def _blind_observation(text: str) -> str:
    """Return only the image-grounded stage-one block from a combined result."""
    value = text or ""
    if BLIND_MARKER not in value or ALIGNMENT_MARKER not in value:
        return ""
    return value.split(BLIND_MARKER, 1)[1].split(ALIGNMENT_MARKER, 1)[0].strip()


def _normalised_span(value: str) -> str:
    return " ".join((value or "").casefold().split())


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _jaccard(a: str, b: str) -> float:
    left, right = _tokens(a), _tokens(b)
    if not left and not right:
        return 1.0
    return len(left & right) / max(1, len(left | right))


def sample_turn_indices(lo: int, hi: int, count: int) -> tuple[int, ...]:
    """Uniform causal samples over a temporal span, always including `hi`."""
    lo, hi, count = int(lo), int(hi), max(1, int(count))
    if hi <= lo or count == 1:
        return (hi,)
    if hi - lo + 1 <= count:
        return tuple(range(lo, hi + 1))
    points = {round(lo + i * (hi - lo) / (count - 1))
              for i in range(count)}
    points.add(hi)
    return tuple(sorted(points))


@dataclass(frozen=True)
class CurrentEvidence:
    state: str
    novelty: str
    task_evidence: str
    match_kind: str
    grounding: str
    anchor: str
    evidence: str
    valid: bool

    @classmethod
    def parse(cls, text: str) -> "CurrentEvidence":
        text = _normalise_bare_fields(text)
        state = _field(text, "CURRENT STATE")[:240]
        novelty = _enum(text, "VISUAL NOVELTY", NOVELTY, "UNCERTAIN")
        task = _enum(text, "TASK EVIDENCE", TASK_EVIDENCE, "UNCERTAIN")
        kind = _enum(text, "MATCH KIND", MATCH_KIND, "NONE")
        has_grounding = _has_field(text, "GROUNDING")
        grounding = (_enum(text, "GROUNDING", GROUNDING, "NONE")
                     if has_grounding else "UNSPECIFIED")
        has_anchor = _has_field(text, "ANCHOR")
        anchor = _field(text, "ANCHOR")[:320]
        blind_observation = _blind_observation(text)
        blind_schema = bool(blind_observation)
        usable_anchor = bool(anchor and anchor.casefold() not in {
            "none", "unknown", "uncertain", "n/a"})
        anchor_supported = bool(
            usable_anchor
            and _normalised_span(anchor) in _normalised_span(blind_observation))
        evidence = _field(text, "EVIDENCE")[:320]
        # Small observers sometimes leave the EVIDENCE value blank for NONE.
        # The five-field schema is still structurally valid in that case; an
        # empty positive match remains invalid and is downgraded below.
        valid = (all(_has_field(text, key) for key in REQUIRED_FIELD_NAMES)
                 and bool(state)
                 and bool(_field(text, "VISUAL NOVELTY"))
                 and bool(_field(text, "TASK EVIDENCE"))
                 and bool(_field(text, "MATCH KIND"))
                 and (not has_grounding or bool(_field(text, "GROUNDING")))
                 and (not has_anchor or bool(_field(text, "ANCHOR")))
                 and (not blind_schema or (has_grounding and has_anchor))
                 and (bool(evidence) or task != "CRITERION_MATCH"))
        if task != "CRITERION_MATCH":
            kind, grounding, anchor, evidence = "NONE", (
                grounding if has_grounding else "UNSPECIFIED"), "", ""
        elif kind == "NONE" or not evidence or evidence.lower() in {
                "none", "unknown", "uncertain", "n/a"}:
            task, kind, grounding, anchor, evidence = "UNCERTAIN", "NONE", (
                grounding if has_grounding else "UNSPECIFIED"), "", ""
        elif has_grounding and grounding != "DIRECT":
            # A calibrated observer may report a plausible category cue, but
            # only direct current-frame grounding can become confirmed state.
            task, kind, anchor, evidence = "UNCERTAIN", "NONE", "", ""
        elif blind_schema and not anchor_supported:
            # The request-aware stage cannot introduce an observation.  A
            # positive fact without an exact stage-one span is malformed and
            # cannot advance the external state.
            valid = False
            task, kind, anchor, evidence = "UNCERTAIN", "NONE", "", ""
        elif task == "CRITERION_MATCH" and not has_anchor:
            anchor = ""
        return cls(state, novelty, task, kind, grounding, anchor, evidence, valid)


class CurrentEvidenceTracker:
    """Deduplicate repeated match evidence and render a non-action status."""

    def __init__(self, *, repeat_gap: int = 8, repeat_jaccard: float = 0.55,
                 render_mode: str = "status", end_confirmations: int = 2,
                 instance_input_gating: bool = False):
        self.repeat_gap = max(0, int(repeat_gap))
        self.repeat_jaccard = max(0.0, min(1.0, float(repeat_jaccard)))
        self.render_mode = render_mode
        self.end_confirmations = max(1, int(end_confirmations))
        # Opt-in pre-decision lifecycle.  It never changes an action: it only
        # controls whether a confirmed event is still projected into a later
        # model input.  Legacy policies retain the byte-compatible behaviour.
        self.instance_input_gating = bool(instance_input_gating)
        self.last_raw = ""
        self.last_match_second: float | None = None
        self.last_match_kind = "NONE"
        self.last_match_evidence = ""
        self.instance_open = False
        self.nonmatch_confirmations = 0
        self.instances = 0
        self.active_instance_id: int | None = None
        self.reported_instance_id: int | None = None
        self.last_update_instance_id: int | None = None
        self.last_update_opened_instance = False
        self.closed_instances = 0
        self.parsed = 0
        self.invalid = 0
        self.matches = 0
        self.repeated = 0
        self.visible = 0
        self.grounded = 0
        self.grounding_direct = 0
        self.grounding_partial = 0
        self.blind_observations = 0
        self.anchored_matches = 0

    def update(self, text: str, second: float) -> tuple[CurrentEvidence, str]:
        item = CurrentEvidence.parse(text)
        self.last_raw = (text or "")[:1200]
        self.parsed += 1
        if not item.valid:
            self.invalid += 1
        if item.grounding != "UNSPECIFIED":
            self.grounded += 1
            self.grounding_direct += int(item.grounding == "DIRECT")
            self.grounding_partial += int(item.grounding == "PARTIAL")
        self.blind_observations += int(bool(_blind_observation(text)))
        self.anchored_matches += int(
            item.task_evidence == "CRITERION_MATCH" and bool(item.anchor))

        repeated = False
        self.last_update_instance_id = None
        self.last_update_opened_instance = False
        if item.task_evidence == "CRITERION_MATCH":
            # A time gap alone cannot turn one continuously visible event into
            # a new instance.  The instance remains open until grounded
            # non-match observations close it, or a materially different
            # match arrives (for example the next procedural completion).
            if (self.instance_open
                    and item.match_kind == self.last_match_kind
                    and _jaccard(item.evidence, self.last_match_evidence)
                    >= self.repeat_jaccard):
                repeated = True
                self.repeated += 1
                self.last_update_instance_id = self.active_instance_id
                if (not self.instance_input_gating
                        or self.reported_instance_id == self.active_instance_id):
                    # Legacy deduplication and the opt-in post-commit state both
                    # suppress the old positive edge.  In the latter case this
                    # is the input-side close: the raw model output is untouched.
                    item = CurrentEvidence(
                        item.state, "CONTINUING", "CONTEXT_ONLY", "NONE",
                        item.grounding, "", "", item.valid)
                else:
                    # Until the model actually produces a non-empty response,
                    # keep presenting the same grounded instance.  This avoids
                    # trading duplicate suppression for a one-frame recall loss.
                    item = CurrentEvidence(
                        item.state, "CONTINUING", "CRITERION_MATCH",
                        item.match_kind, item.grounding, item.anchor,
                        item.evidence, item.valid)
            else:
                self.matches += 1
                self.instances += 1
                self.instance_open = True
                self.active_instance_id = self.instances
                self.last_update_instance_id = self.active_instance_id
                self.last_update_opened_instance = True
                self.nonmatch_confirmations = 0
                self.last_match_second = float(second)
                self.last_match_kind = item.match_kind
                self.last_match_evidence = item.evidence
        elif item.valid and item.task_evidence in {"NONE", "CONTEXT_ONLY"}:
            # Require repeated observer evidence before closing an instance;
            # one uncertain/occluded frame must not re-arm an ongoing event.
            if self.instance_open:
                self.nonmatch_confirmations += 1
                if self.nonmatch_confirmations >= self.end_confirmations:
                    self.instance_open = False
                    self.active_instance_id = None
                    self.nonmatch_confirmations = 0
                    self.closed_instances += 1
            else:
                self.nonmatch_confirmations = 0

        if self.render_mode == "edge" and item.task_evidence != "CRITERION_MATCH":
            return item, ""
        status = (f"task_evidence={item.task_evidence}; "
                  f"visual_novelty={item.novelty}; match_kind={item.match_kind}")
        if item.grounding != "UNSPECIFIED":
            status += f"; grounding={item.grounding}"
        if item.task_evidence == "CRITERION_MATCH":
            if item.anchor:
                status += f"; anchor={item.anchor}"
            status += f"; support={item.evidence}"
        if repeated:
            status += "; repeated_edge=true"
        self.visible += 1
        return item, status

    def mark_reported(self) -> int | None:
        """Close projection for the active instance after a real response.

        The method is deliberately separate from ``update``: the observer does
        not choose whether the assistant speaks.  The runner calls this only
        after the interaction model committed a non-empty response.
        """
        if not self.instance_input_gating or not self.instance_open:
            return None
        self.reported_instance_id = self.active_instance_id
        return self.reported_instance_id

    def dump(self) -> dict:
        return {
            "render_mode": self.render_mode,
            "repeat_gap": self.repeat_gap,
            "repeat_jaccard": self.repeat_jaccard,
            "end_confirmations": self.end_confirmations,
            "instance_input_gating": self.instance_input_gating,
            "instance_open": self.instance_open,
            "active_instance_id": self.active_instance_id,
            "reported_instance_id": self.reported_instance_id,
            "instances": self.instances,
            "closed_instances": self.closed_instances,
            "parsed": self.parsed,
            "invalid": self.invalid,
            "matches": self.matches,
            "repeated": self.repeated,
            "visible": self.visible,
            "grounded": self.grounded,
            "grounding_direct": self.grounding_direct,
            "grounding_partial": self.grounding_partial,
            "blind_observations": self.blind_observations,
            "anchored_matches": self.anchored_matches,
        }
