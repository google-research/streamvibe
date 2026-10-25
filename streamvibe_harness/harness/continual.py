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

"""Bounded continual refinement for the streaming harness.

The refiner deliberately separates *proposal* from *promotion*.  Online traces
are useful for detecting failure signatures, but they do not provide a trusted
task reward.  ``shadow`` mode therefore records versioned CRUD proposals while
leaving serving behaviour unchanged.  ``adaptive`` mode may apply only the
small numeric novelty-policy edits defined here; prompt, specialist, skill and
memory-rule edits still require replay/canary validation before promotion.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field

from .novelty import NovelResponseGate, token_jaccard


@dataclass(frozen=True)
class DecisionEvent:
    turn: int
    proposal: str = ""
    verifier_response: bool = False
    committed: str = ""
    rescued: bool = False
    verifier_failed: bool = False
    query_arrived: bool = False
    past_evidence_items: int = 0
    retrieval_used: bool = False


@dataclass(frozen=True)
class FailureSignature:
    kind: str
    severity: float
    evidence: dict


@dataclass(frozen=True)
class HarnessPatch:
    operation: str
    component: str
    key: str
    value: object
    reason: str


@dataclass
class HarnessState:
    revision: int = 0
    novelty_threshold: float = 0.6
    novelty_min_gap: int = 8
    novelty_history: int = 8
    response_rules: dict[str, str] = field(default_factory=dict)
    evidence_rules: dict[str, str] = field(default_factory=dict)

    def dump(self) -> dict:
        return asdict(self)


class FailureMonitor:
    """Detect reward-free failure proxies from information available online."""

    def __init__(self, window: int = 64):
        if window < 8:
            raise ValueError("window must be >= 8")
        self.window = window
        self.events: deque[DecisionEvent] = deque(maxlen=window)

    def observe(self, event: DecisionEvent) -> None:
        self.events.append(event)

    def signatures(self) -> list[FailureSignature]:
        events = list(self.events)
        if len(events) < self.window:
            return []
        proposed = [e for e in events if e.proposal]
        committed = [e for e in events if e.committed]
        rejected = [e for e in proposed if not e.verifier_response]
        failures = [e for e in events if e.verifier_failed]
        late_unretrieved = [e for e in events
                            if e.query_arrived and e.turn >= 32
                            and e.past_evidence_items > 0 and not e.retrieval_used]
        repeats = 0
        for left, right in zip(committed, committed[1:]):
            if token_jaccard(left.committed, right.committed) >= 0.8:
                repeats += 1

        out: list[FailureSignature] = []
        proposal_rate = len(proposed) / len(events)
        rejection_rate = len(rejected) / max(1, len(proposed))
        repeat_rate = repeats / max(1, len(committed) - 1)
        if proposal_rate >= 0.30 and rejection_rate >= 0.60:
            out.append(FailureSignature(
                "ungrounded_speech_burst",
                min(1.0, (proposal_rate + rejection_rate) / 2),
                {"proposal_rate": round(proposal_rate, 4),
                 "rejection_rate": round(rejection_rate, 4)},
            ))
        if repeat_rate >= 0.20:
            out.append(FailureSignature(
                "repeated_response",
                min(1.0, repeat_rate),
                {"repeat_rate": round(repeat_rate, 4),
                 "committed": len(committed)},
            ))
        if failures:
            out.append(FailureSignature(
                "verifier_instability",
                min(1.0, len(failures) / max(1, len(proposed))),
                {"failures": len(failures), "proposals": len(proposed)},
            ))
        if late_unretrieved:
            out.append(FailureSignature(
                "late_query_evidence_access_gap", 1.0,
                {"queries": len(late_unretrieved),
                 "max_past_items": max(e.past_evidence_items
                                       for e in late_unretrieved)},
            ))
        return out


class BoundedRefiner:
    """Turn failure signatures into auditable, bounded CRUD proposals."""

    def propose(self, state: HarnessState,
                signatures: list[FailureSignature]) -> list[HarnessPatch]:
        patches: list[HarnessPatch] = []
        kinds = {s.kind for s in signatures}
        if "ungrounded_speech_burst" in kinds:
            patches.extend([
                HarnessPatch(
                    "update", "intervention", "novelty_threshold",
                    round(max(0.40, state.novelty_threshold - 0.05), 2),
                    "Verifier rejects most proposals in a high-speech window.",
                ),
                HarnessPatch(
                    "update", "intervention", "novelty_min_gap",
                    min(14, state.novelty_min_gap + 2),
                    "Increase the quiet interval before rescuing rejected speech.",
                ),
                HarnessPatch(
                    "upsert", "response_policy", "current_evidence_only",
                    "Do not speak from past memory alone; require a current visual event.",
                    "Suppress memory-triggered narration bursts.",
                ),
            ])
        if "repeated_response" in kinds:
            patches.extend([
                HarnessPatch(
                    "upsert", "intervention", "semantic_dedup",
                    {"similarity": 0.8, "history": state.novelty_history},
                    "Collapse semantically repeated committed responses.",
                ),
                HarnessPatch(
                    "upsert", "response_policy", "repetition_critic",
                    "Compare a candidate with recent committed answers before speaking.",
                    "Repeated responses need a decision rule, not more memory.",
                ),
            ])
        if "verifier_instability" in kinds:
            patches.append(HarnessPatch(
                "upsert", "intervention", "verifier_fail_open",
                {"mode": "fail_open", "log_for_replay": True},
                "Preserve availability while collecting failures for repair.",
            ))
        if "late_query_evidence_access_gap" in kinds:
            patches.extend([
                HarnessPatch(
                    "upsert", "evidence_policy", "future_query_evidence",
                    "When no request is active, retain entities and attributes, object interactions, action order, counts, visible text, and locations for future queries.",
                    "A late query had past evidence but no query-time retrieval.",
                ),
                HarnessPatch(
                    "update", "evidence_config", "skills",
                    {"enabled": True, "retrieval_top_k": 2},
                    "Enable bounded query-time retrieval over existing memory blocks.",
                ),
            ])
        return patches

    def apply_numeric(self, state: HarnessState, gate: NovelResponseGate,
                      patches: list[HarnessPatch]) -> list[HarnessPatch]:
        """Apply only reversible bounded controller edits in adaptive mode."""
        applied = []
        for patch in patches:
            if patch.component != "intervention" or patch.operation != "update":
                continue
            if patch.key == "novelty_threshold":
                state.novelty_threshold = float(patch.value)
                gate.threshold = state.novelty_threshold
                applied.append(patch)
            elif patch.key == "novelty_min_gap":
                state.novelty_min_gap = int(patch.value)
                gate.min_gap = state.novelty_min_gap
                applied.append(patch)
        if applied:
            state.revision += 1
        return applied


class ContinualController:
    """Online monitor plus versioned shadow/adaptive refinement journal."""

    def __init__(self, mode: str, gate: NovelResponseGate, *, every: int = 64,
                 warmup: int = 64):
        if mode not in {"off", "shadow", "adaptive"}:
            raise ValueError(f"unknown continual mode: {mode}")
        if every < 8 or warmup < 0:
            raise ValueError("every must be >= 8 and warmup must be >= 0")
        self.mode = mode
        self.gate = gate
        self.every = every
        self.warmup = warmup
        self.monitor = FailureMonitor(every)
        self.refiner = BoundedRefiner()
        self.state = HarnessState(
            novelty_threshold=gate.threshold,
            novelty_min_gap=gate.min_gap,
            novelty_history=gate.committed.maxlen or 8,
        )
        self.journal: list[dict] = []
        self._last_cycle_turn: int | None = None

    def observe(self, event: DecisionEvent) -> None:
        if self.mode == "off":
            return
        self.monitor.observe(event)
        boundary = event.turn + 1
        if boundary < self.warmup or boundary % self.every:
            return
        self._refine(event.turn)

    def _refine(self, turn: int) -> None:
        if self._last_cycle_turn == turn:
            return
        signatures = self.monitor.signatures()
        patches = self.refiner.propose(self.state, signatures)
        applied = (self.refiner.apply_numeric(self.state, self.gate, patches)
                   if self.mode == "adaptive" else [])
        self.journal.append({
            "turn": turn,
            "revision_before": self.state.revision - int(bool(applied)),
            "revision_after": self.state.revision,
            "signatures": [asdict(s) for s in signatures],
            "proposals": [asdict(p) for p in patches],
            "applied": [asdict(p) for p in applied],
        })
        self._last_cycle_turn = turn

    def finalize(self, turn: int) -> None:
        """Flush a partial final refinement window without duplicating a boundary."""
        if self.mode != "off" and self.monitor.events:
            self._refine(turn)

    def dump(self) -> dict:
        return {"mode": self.mode, "every": self.every, "warmup": self.warmup,
                "state": self.state.dump(), "cycles": len(self.journal),
                "journal": self.journal}
