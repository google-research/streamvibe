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

"""Typed causal state projected before the interactive streaming model.

The state deliberately keeps *provenance* separate from semantics:

* a user request is a condition to watch, never an observation;
* raw-frame difference is motion/change evidence, never proof of a task event;
* only a parsed criterion match or evidence-grounded procedure update advances
  the normalised evidence version;
* committed replies are receipts tied to the evidence version they used.

Nothing in this module returns an action or edits model output.
"""
from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass, field

from .context import response_similarity


def should_commit_response_state(action: str, text: str, *,
                                 nonempty_only: bool = False) -> bool:
    """Whether a raw model action may mutate causal response state.

    This is not an output filter: the raw action remains the prediction seen by
    scorers.  Opt-in hygiene only prevents a bare ``</response>`` marker from
    becoming an answer receipt, a SAID item, or a response-history epoch.
    """
    return bool(action == "response"
                and (not nonempty_only or (text or "").strip()))


@dataclass(frozen=True)
class VisualTransition:
    second: float
    transition_id: int
    kind: str
    score: float
    edge: bool

    def render_delta(self) -> str:
        return (
            f"[Δ {self.second:.0f}s] OBS.transition: id={self.transition_id}; "
            f"source=RAW_FRAME_DIFFERENCE; motion={self.kind}; "
            f"magnitude={self.score:.3f}; semantic=UNRESOLVED"
        )


class FrameTransitionTracker:
    """Cheap model-free motion boundary tracker over causal current frames.

    It emits only category edges. Continuous camera or object motion therefore
    stays one bounded state instead of creating a new prompt event every second.
    """

    def __init__(self, *, minor_threshold: float = 0.025,
                 major_threshold: float = 0.14, edge_gap: int = 3):
        if not 0 <= minor_threshold < major_threshold <= 1:
            raise ValueError("transition thresholds must satisfy 0 <= minor < major <= 1")
        self.minor_threshold = float(minor_threshold)
        self.major_threshold = float(major_threshold)
        self.edge_gap = max(0, int(edge_gap))
        self.previous: bytes | None = None
        self.kind = "UNKNOWN"
        self.transition_id = 0
        self.last_edge_second: float | None = None
        self.calls = 0
        self.edges = 0
        self.unsupported = 0

    @staticmethod
    def _thumbnail(image) -> bytes | None:
        try:
            return image.convert("L").resize((32, 32)).tobytes()
        except (AttributeError, TypeError, ValueError):
            return None

    def update(self, image, second: float) -> VisualTransition | None:
        current = self._thumbnail(image)
        self.calls += 1
        if current is None:
            self.unsupported += 1
            return None
        if self.previous is None:
            self.previous = current
            self.kind = "INITIAL"
            return VisualTransition(float(second), 0, "INITIAL", 0.0, False)
        score = sum(abs(a - b) for a, b in zip(current, self.previous)) \
            / (255.0 * len(current))
        self.previous = current
        if score >= self.major_threshold:
            kind = "MAJOR_CHANGE"
        elif score >= self.minor_threshold:
            kind = "MOTION"
        else:
            kind = "STABLE"
        category_edge = kind != self.kind
        major_refresh = (
            kind == "MAJOR_CHANGE"
            and self.last_edge_second is not None
            and float(second) - self.last_edge_second >= self.edge_gap
        )
        edge = category_edge or major_refresh
        if edge:
            self.transition_id += 1
            self.last_edge_second = float(second)
            self.edges += 1
        self.kind = kind
        return VisualTransition(float(second), self.transition_id, kind, score, edge)

    def dump(self) -> dict:
        return {
            "minor_threshold": self.minor_threshold,
            "major_threshold": self.major_threshold,
            "edge_gap": self.edge_gap,
            "calls": self.calls,
            "edges": self.edges,
            "unsupported": self.unsupported,
            "transition_id": self.transition_id,
            "motion": self.kind,
        }


@dataclass
class StreamDecisionState:
    """Bounded P/OBS/SAID lifecycle with explicit causal versions."""

    profile: str = "passive_observation"
    request_version: int = 0
    request_status: str = "NONE"
    request_second: float | None = None
    normalised_evidence_version: int = 0
    procedure_version: int = 0
    ordered_observation_version: int = 0
    visual_transition_id: int = 0
    visual_motion: str = "UNKNOWN"
    event_instance_id: int = 0
    event_status: str = "WAITING_FOR_MATCH"
    event_match_kind: str = "NONE"
    event_opened_second: float | None = None
    last_reported_event_instance: int | None = None
    last_response_turn: int | None = None
    last_response_evidence_version: int | None = None
    last_response_transition_id: int | None = None
    last_response_ordered_observation_version: int | None = None
    responses_without_new_evidence: int = 0
    responses_without_new_context: int = 0
    response_pattern: str = "NONE"
    _procedure_digest: str = ""
    _responses: deque = field(default_factory=lambda: deque(maxlen=8))

    def on_request(self, second: float, profile: str, *, question: bool = False) -> str:
        self.profile = profile
        self.request_version += 1
        self.request_second = float(second)
        self.request_status = "QUESTION_PENDING" if question else "ARMED_UNCONFIRMED"
        # A request mutation changes the condition being watched. Existing raw
        # motion is not evidence for the new condition; only a later confirmed
        # normaliser match can open an event instance.
        self.event_status = "WAITING_FOR_MATCH"
        self.event_match_kind = "NONE"
        return (
            f"[Δ {second:.0f}s] P.lifecycle: request_v={self.request_version}; "
            f"intent={profile}; source=USER_CONDITION; "
            f"observation=UNCONFIRMED; status={self.request_status}"
        )

    def on_visual_transition(self, transition: VisualTransition) -> str:
        changed = transition.transition_id != self.visual_transition_id
        self.visual_transition_id = transition.transition_id
        self.visual_motion = transition.kind
        if changed:
            self.responses_without_new_context = 0
            self.response_pattern = "NONE"
        return transition.render_delta() if transition.edge else ""

    def on_confirmed_evidence(self, second: float, kind: str,
                              source: str = "VISUAL_NORMALISER") -> str:
        self.normalised_evidence_version += 1
        self.event_instance_id += 1
        self.event_status = "OPEN"
        self.event_match_kind = str(kind)
        self.event_opened_second = float(second)
        self.request_status = "EVIDENCE_CONFIRMED"
        self.responses_without_new_evidence = 0
        self.responses_without_new_context = 0
        self.response_pattern = "NONE"
        return (
            f"[Δ {second:.0f}s] OBS.lifecycle: evidence_v="
            f"{self.normalised_evidence_version}; source={source}; "
            f"match={kind}; event_instance={self.event_instance_id}; "
            "event_status=OPEN; status=CONFIRMED"
        )

    def sync_ordered_observations(self, event_version: int,
                                  second: float) -> str:
        """Expose new visual-memory evidence without asserting a task match.

        The ordered ledger is produced from past video frames, so it is valid
        observed context.  It is still not proof that the user's condition was
        satisfied.  Keep that distinction explicit: this version resets only
        the no-new-context receipt and never advances normalised_evidence_v or
        request_status.
        """
        version = max(0, int(event_version))
        if version <= self.ordered_observation_version:
            return ""
        self.ordered_observation_version = version
        self.responses_without_new_context = 0
        self.response_pattern = "NONE"
        return (
            f"[Δ {second:.0f}s] OBS.memory: ordered_observation_v={version}; "
            "source=VISUAL_SUMMARY; task_match=UNRESOLVED; status=OBSERVED"
        )

    def sync_procedure(self, procedure_dump: dict, second: float) -> str:
        observable = {
            key: procedure_dump.get(key)
            for key in ("current", "completed", "deviation", "anchors", "updates")
        }
        digest = hashlib.sha1(repr(observable).encode("utf-8")).hexdigest()
        if not procedure_dump.get("active") or not procedure_dump.get("updates"):
            return ""
        if digest == self._procedure_digest:
            return ""
        self._procedure_digest = digest
        self.procedure_version += 1
        return self.on_confirmed_evidence(
            second, f"PROCEDURE_TRANSITION_V{self.procedure_version}",
            source="EVIDENCE_GROUNDED_PROCEDURE")

    def on_response(self, turn: int, second: float, text: str) -> str:
        evidence_version = self.normalised_evidence_version
        if self.last_response_evidence_version == evidence_version:
            self.responses_without_new_evidence += 1
        else:
            self.responses_without_new_evidence = 1
        if (self.last_response_evidence_version == evidence_version
                and self.last_response_transition_id == self.visual_transition_id
                and self.last_response_ordered_observation_version
                == self.ordered_observation_version):
            self.responses_without_new_context += 1
        else:
            self.responses_without_new_context = 1
        prior = list(self._responses)
        pattern = "DISTINCT"
        if prior and response_similarity(text, prior[-1][1]) >= 0.82:
            pattern = "REPEAT_1"
        elif len(prior) >= 2 and response_similarity(text, prior[-2][1]) >= 0.82:
            pattern = "ALTERNATING_2"
        elif len(prior) >= 3 and response_similarity(text, prior[-3][1]) >= 0.82:
            pattern = "CYCLE_3"
        self._responses.append((int(turn), text))
        self.last_response_turn = int(turn)
        self.last_response_evidence_version = evidence_version
        self.last_response_transition_id = self.visual_transition_id
        self.last_response_ordered_observation_version = (
            self.ordered_observation_version)
        self.response_pattern = pattern
        if self.event_status == "OPEN":
            self.event_status = "REPORTED"
            self.last_reported_event_instance = self.event_instance_id
        elif self.request_status == "QUESTION_PENDING":
            self.event_status = "ANSWERED"
        else:
            self.event_status = "WAITING_FOR_NEW_EVIDENCE"
        self.request_status = (
            "ANSWERED" if self.request_status == "QUESTION_PENDING"
            else "ARMED_WAITING_FOR_NEW_EVIDENCE")
        return (
            f"[Δ {second:.0f}s] SAID.lifecycle: committed=true; "
            f"based_on_evidence_v={evidence_version}; "
            f"responses_without_new_normalised_evidence="
            f"{self.responses_without_new_evidence}; visual_transition_v="
            f"{self.visual_transition_id}; ordered_observation_v="
            f"{self.ordered_observation_version}; responses_without_new_context="
            f"{self.responses_without_new_context}; pattern={pattern}; "
            f"event_instance={self.event_instance_id}; "
            f"event_status={self.event_status}; "
            "action_authority=StreamVibe-VL"
        )

    def render_line(self) -> str:
        last = "none" if self.last_response_turn is None else str(self.last_response_turn)
        based = ("none" if self.last_response_evidence_version is None
                 else str(self.last_response_evidence_version))
        return (
            "P.lifecycle: "
            f"intent={self.profile}; request_v={self.request_version}; "
            f"request_status={self.request_status}; request_source=USER_CONDITION; "
            f"normalised_evidence_v={self.normalised_evidence_version}; "
            f"procedure_v={self.procedure_version}; ordered_observation_v="
            f"{self.ordered_observation_version}; visual_transition_v="
            f"{self.visual_transition_id}; motion={self.visual_motion}; "
            f"event_instance={self.event_instance_id}; "
            f"event_status={self.event_status}; "
            f"event_match={self.event_match_kind}; "
            f"last_response_turn={last}; response_evidence_v={based}; "
            f"responses_without_new_normalised_evidence="
            f"{self.responses_without_new_evidence}; responses_without_new_context="
            f"{self.responses_without_new_context}; pattern={self.response_pattern}"
        )

    def dump(self) -> dict:
        return {
            "profile": self.profile,
            "request_version": self.request_version,
            "request_status": self.request_status,
            "request_second": self.request_second,
            "normalised_evidence_version": self.normalised_evidence_version,
            "procedure_version": self.procedure_version,
            "ordered_observation_version": self.ordered_observation_version,
            "visual_transition_id": self.visual_transition_id,
            "visual_motion": self.visual_motion,
            "event_instance_id": self.event_instance_id,
            "event_status": self.event_status,
            "event_match_kind": self.event_match_kind,
            "event_opened_second": self.event_opened_second,
            "last_reported_event_instance": self.last_reported_event_instance,
            "last_response_turn": self.last_response_turn,
            "last_response_evidence_version": self.last_response_evidence_version,
            "last_response_transition_id": self.last_response_transition_id,
            "last_response_ordered_observation_version": (
                self.last_response_ordered_observation_version),
            "responses_without_new_evidence": self.responses_without_new_evidence,
            "responses_without_new_context": self.responses_without_new_context,
            "response_pattern": self.response_pattern,
        }
