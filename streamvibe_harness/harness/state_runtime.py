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

"""Canonical typed-event state for the state-centric streaming Harness.

The runtime is the only owner of online operational state.  Parsers and the
commit path submit typed events; they cannot directly edit the rendered state.
Memory and Skills remain read-only evidence/reference stores.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


SOURCE_PERMISSIONS = {
    "REQUEST_ADDED": {"USER"},
    "REQUEST_RETIRED": {"USER"},
    "QUESTION_OPENED": {"USER"},
    "QUESTION_ANSWERED": {"COMMIT"},
    "VISUAL_OBSERVATION": {"VISUAL"},
    "EVENT_CLOSED": {"VISUAL"},
    "PROCEDURE_OBSERVED": {"VISUAL", "USER_CONFIRMATION"},
    "RESPONSE_DELIVERED": {"COMMIT"},
    "UNCERTAINTY_SET": {"USER", "VISUAL", "SYSTEM"},
    "UNCERTAINTY_CLEARED": {"USER", "VISUAL", "SYSTEM"},
}


@dataclass(frozen=True)
class CanonicalEvent:
    event_id: str
    type: str
    second: float
    source: str
    payload: dict[str, Any] = field(default_factory=dict)
    evidence_ref: str | None = None


@dataclass
class CanonicalState:
    active_obligations: dict[str, dict[str, Any]] = field(default_factory=dict)
    retired_obligations: dict[str, dict[str, Any]] = field(default_factory=dict)
    pending_question: dict[str, Any] | None = None
    procedure: dict[str, Any] = field(default_factory=lambda: {
        "goal": None, "observed_current": None, "last_completed": None,
        "deviation": None, "reference_next": None,
    })
    active_events: dict[str, dict[str, Any]] = field(default_factory=dict)
    deliveries: dict[str, dict[str, Any]] = field(default_factory=dict)
    skill_instances: dict[str, dict[str, Any]] = field(default_factory=dict)
    unresolved_fields: dict[str, str] = field(default_factory=dict)
    version: int = 0


class StateRuntime:
    """Append-only event log plus deterministic, provenance-checked reducers."""

    def __init__(self) -> None:
        self.state = CanonicalState()
        self.events: list[CanonicalEvent] = []
        self._seen: set[str] = set()
        self._duplicates = 0
        self._counter = 0
        self._criterion_ids: dict[str, str] = {}
        self._question_counter = 0
        self._event_counter = 0
        self._delivery_counter = 0
        self._pending_deltas: list[str] = []

    def _id(self, prefix: str, second: float, key: str = "") -> str:
        self._counter += 1
        digest = hashlib.sha1(
            f"{prefix}|{float(second):.3f}|{key}|{self._counter}".encode("utf-8")
        ).hexdigest()[:8]
        return f"{prefix}{self._counter:04d}_{digest}"

    def append(self, event: CanonicalEvent) -> bool:
        allowed = SOURCE_PERMISSIONS.get(event.type)
        if allowed is None:
            raise ValueError(f"unsupported canonical event type {event.type!r}")
        if event.source not in allowed:
            raise ValueError(
                f"{event.source} cannot produce {event.type}; allowed={sorted(allowed)}")
        if event.event_id in self._seen:
            self._duplicates += 1
            return False
        self._seen.add(event.event_id)
        self.events.append(event)
        self._reduce(event)
        self.state.version += 1
        self._pending_deltas.append(self._render_delta(event))
        return True

    def apply_instruction(self, ops: Iterable[Any], text: str, second: float) -> None:
        for op in ops:
            if op.kind == "add":
                criterion_id = self._criterion_ids.get(op.target)
                if criterion_id is None:
                    criterion_id = self._id("c", second, op.target)
                    self._criterion_ids[op.target] = criterion_id
                self.append(CanonicalEvent(
                    self._id("u", second, f"add:{op.target}"), "REQUEST_ADDED",
                    second, "USER", {"criterion_id": criterion_id,
                                     "criterion": op.target, "text": text}))
            elif op.kind == "retire":
                criterion_id = self._criterion_ids.get(op.target)
                if criterion_id:
                    self.append(CanonicalEvent(
                        self._id("u", second, f"retire:{op.target}"),
                        "REQUEST_RETIRED", second, "USER",
                        {"criterion_id": criterion_id,
                         "criterion": op.target, "text": text}))
            elif op.kind == "retire_others":
                keep = set(op.target.split(","))
                for criterion_id, obligation in list(
                        self.state.active_obligations.items()):
                    if obligation["criterion"] not in keep:
                        self.append(CanonicalEvent(
                            self._id("u", second,
                                     f"retire:{obligation['criterion']}"),
                            "REQUEST_RETIRED", second, "USER",
                            {"criterion_id": criterion_id,
                             "criterion": obligation["criterion"], "text": text}))
            elif op.kind == "question":
                self._question_counter += 1
                question_id = f"q{self._question_counter:04d}"
                self.append(CanonicalEvent(
                    self._id("u", second, question_id), "QUESTION_OPENED",
                    second, "USER", {"question_id": question_id, "text": op.target}))

    def observe(self, second: float, match_kind: str, evidence: str,
                *, evidence_ref: str | None = None,
                instance_id: int | str | None = None,
                obligation_ids: Iterable[str] = ()) -> str:
        if instance_id is None:
            self._event_counter += 1
            instance_id = self._event_counter
        event_instance_id = f"e{instance_id}"
        ref = evidence_ref or self._id("visual_obs_", second, match_kind)
        self.append(CanonicalEvent(
            self._id("v", second, f"{event_instance_id}:{match_kind}"),
            "VISUAL_OBSERVATION", second, "VISUAL",
            {"event_instance_id": event_instance_id, "match_kind": match_kind,
             "evidence": " ".join((evidence or "").split())[:240],
             "obligation_ids": list(obligation_ids)}, ref))
        return event_instance_id

    def sync_skill_instances(self, available_skills: Iterable[str]) -> None:
        """Maintain one private Skill execution instance per obligation.

        This is state bookkeeping, not recall competition.  The projection
        method below independently selects at most two relevant instances.
        """
        available = set(available_skills)
        active_ids = set(self.state.active_obligations)
        for criterion_id, obligation in self.state.active_obligations.items():
            skill = obligation["criterion"]
            if skill not in available:
                continue
            instance = self.state.skill_instances.setdefault(criterion_id, {
                "criterion_id": criterion_id, "skill": skill,
                "status": "armed", "event_instance_id": None,
                "match_status": "unconfirmed", "evidence_ref": None,
                "evidence_version": 0, "delivered": False,
            })
            if instance.get("status") == "retired":
                instance.update(status="armed", delivered=False)
        for criterion_id, instance in self.state.skill_instances.items():
            if criterion_id not in active_ids:
                instance["status"] = "retired"

    def render_skill_projection(self, skills: dict[str, Any],
                                max_visible: int = 2) -> str:
        """Render one compact, non-copyable execution-state guard.

        No trigger/speak/silent/answer examples are exposed.  The cadence is a
        short state-transition boundary, not prose that can be copied as a
        response.  The interaction model still owns the action decision.
        """
        priority = {"matched": 0, "armed": 1, "delivered": 2, "retired": 3}
        instances = sorted(
            (item for item in self.state.skill_instances.values()
             if item.get("status") in {"matched", "armed"}),
            key=lambda item: (priority.get(item.get("status"), 9),
                              item["criterion_id"]),
        )[:max(0, int(max_visible))]
        lines = []
        for item in instances:
            skill = skills.get(item["skill"])
            cadence = ""
            if skill is not None:
                cadence = skill.fields.get("cadence", "").split("|", 1)[0].strip()
            evidence = item.get("evidence_ref") or "unconfirmed"
            if item["status"] == "matched":
                phase = "matched"
                response_basis = "confirmed_event"
            elif self.state.pending_question is not None:
                phase = "question"
                response_basis = "open_question"
            else:
                phase = "waiting"
                response_basis = "none_until_new_evidence"
            lines.append(
                f"SKILL STATE: {item['skill']}; phase={phase}; "
                f"evidence={evidence}; response_basis={response_basis}"
                + (f"; cadence={cadence}" if cadence else ""))
        return "\n".join(lines)

    def has_meaningful_state(self) -> bool:
        """Whether a compact state card carries information beyond an empty shell."""
        p = self.state.procedure
        return bool(
            self.state.active_obligations
            or self.state.pending_question
            or self.state.active_events
            or self.state.deliveries
            or self.state.unresolved_fields
            or p.get("observed_current")
            or p.get("last_completed")
        )

    def close_event(self, second: float, event_instance_id: str) -> None:
        self.append(CanonicalEvent(
            self._id("v", second, f"close:{event_instance_id}"), "EVENT_CLOSED",
            second, "VISUAL", {"event_instance_id": event_instance_id}))

    def commit_response(self, second: float, text: str,
                        *, answer_question: bool = True) -> str:
        self._delivery_counter += 1
        delivery_id = f"d{self._delivery_counter:04d}"
        question_id = (self.state.pending_question or {}).get("question_id")
        active_event = next(reversed(self.state.active_events), None)
        obligation_ids = (
            list(self.state.active_events[active_event].get("obligation_ids", []))
            if active_event else list(self.state.active_obligations))
        self.append(CanonicalEvent(
            self._id("commit_", second, delivery_id), "RESPONSE_DELIVERED",
            second, "COMMIT", {
                "delivery_id": delivery_id,
                "question_id": question_id if answer_question else None,
                "event_instance_id": active_event,
                "obligation_ids": obligation_ids,
                "content_digest": hashlib.sha1(
                    (text or "").encode("utf-8")).hexdigest()[:12],
            }))
        if question_id and answer_question:
            self.append(CanonicalEvent(
                self._id("commit_", second, f"answered:{question_id}"),
                "QUESTION_ANSWERED", second, "COMMIT",
                {"question_id": question_id, "delivery_id": delivery_id}))
        return delivery_id

    def _reduce(self, event: CanonicalEvent) -> None:
        p = event.payload
        if event.type == "REQUEST_ADDED":
            self.state.active_obligations[p["criterion_id"]] = {
                "criterion": p["criterion"], "since": event.second,
                "source_event": event.event_id,
            }
            self.state.retired_obligations.pop(p["criterion_id"], None)
        elif event.type == "REQUEST_RETIRED":
            prior = self.state.active_obligations.pop(p["criterion_id"], None)
            if prior:
                self.state.retired_obligations[p["criterion_id"]] = {
                    **prior, "retired_at": event.second,
                }
        elif event.type == "QUESTION_OPENED":
            self.state.pending_question = {
                "question_id": p["question_id"], "text": p["text"],
                "since": event.second,
            }
        elif event.type == "QUESTION_ANSWERED":
            if ((self.state.pending_question or {}).get("question_id")
                    == p["question_id"]):
                self.state.pending_question = None
        elif event.type == "VISUAL_OBSERVATION":
            eid = p["event_instance_id"]
            previous = self.state.active_events.get(eid, {})
            self.state.active_events[eid] = {
                **previous, "match_kind": p["match_kind"],
                "status": "confirmed_open", "evidence": p.get("evidence", ""),
                "evidence_ref": event.evidence_ref,
                "obligation_ids": list(p.get("obligation_ids", [])),
                "evidence_version": int(previous.get("evidence_version", 0)) + 1,
                "observed_at": event.second,
            }
            for criterion_id in p.get("obligation_ids", []):
                instance = self.state.skill_instances.get(criterion_id)
                if instance is not None:
                    instance.update(
                        status="matched", match_status="confirmed",
                        event_instance_id=eid, evidence_ref=event.evidence_ref,
                        evidence_version=self.state.active_events[eid][
                            "evidence_version"], delivered=False)
        elif event.type == "EVENT_CLOSED":
            if p["event_instance_id"] in self.state.active_events:
                self.state.active_events[p["event_instance_id"]]["status"] = "closed"
            for instance in self.state.skill_instances.values():
                if instance.get("event_instance_id") == p["event_instance_id"]:
                    instance.update(
                        status="armed", match_status="unconfirmed",
                        event_instance_id=None, evidence_ref=None,
                        delivered=False)
        elif event.type == "PROCEDURE_OBSERVED":
            self.state.procedure.update(p)
        elif event.type == "RESPONSE_DELIVERED":
            self.state.deliveries[p["delivery_id"]] = {
                **p, "second": event.second,
            }
            eid = p.get("event_instance_id")
            if eid in self.state.active_events:
                self.state.active_events[eid]["status"] = "reported"
                self.state.active_events[eid]["delivery_id"] = p["delivery_id"]
            for criterion_id in p.get("obligation_ids", []):
                instance = self.state.skill_instances.get(criterion_id)
                if instance is not None:
                    instance.update(status="delivered", delivered=True)
        elif event.type == "UNCERTAINTY_SET":
            self.state.unresolved_fields[p["field"]] = p["reason"]
        elif event.type == "UNCERTAINTY_CLEARED":
            self.state.unresolved_fields.pop(p["field"], None)

    @staticmethod
    def _render_delta(event: CanonicalEvent) -> str:
        p = event.payload
        if event.type == "REQUEST_ADDED":
            body = f"obligation {p['criterion_id']} +{p['criterion']}"
        elif event.type == "REQUEST_RETIRED":
            body = f"obligation {p['criterion_id']} retired"
        elif event.type == "QUESTION_OPENED":
            body = f"question {p['question_id']} opened"
        elif event.type == "QUESTION_ANSWERED":
            body = f"question {p['question_id']} answered by {p['delivery_id']}"
        elif event.type == "VISUAL_OBSERVATION":
            body = (f"event {p['event_instance_id']} confirmed {p['match_kind']}; "
                    f"evidence={event.evidence_ref}")
        elif event.type == "EVENT_CLOSED":
            body = f"event {p['event_instance_id']} closed"
        elif event.type == "RESPONSE_DELIVERED":
            target = p.get("event_instance_id") or p.get("question_id") or "request"
            body = f"delivery {p['delivery_id']} committed for {target}"
        else:
            body = event.type.lower()
        return f"[Delta {event.second:.0f}s] {body}"

    def consume_deltas(self, *, folded: bool = False) -> str:
        text = "\n".join(self._pending_deltas)
        self._pending_deltas.clear()
        return "" if folded else text

    def render(self, evidence_text: str = "", *, evidence_chars: int = 180) -> str:
        """Project only non-empty decision state plus one bounded evidence fact."""
        obligations = ", ".join(
            f"{cid}:{item['criterion']}" for cid, item in
            self.state.active_obligations.items()) or "none"
        question = self.state.pending_question
        question_text = (f"{question['question_id']}:{question['text'][:120]}"
                         if question else "none")
        open_events = [
            f"{eid}:{item['match_kind']}:{item['status']}:v{item['evidence_version']}"
            for eid, item in self.state.active_events.items()
            if item.get("status") != "closed"
        ]
        event_text = ", ".join(open_events[-2:]) or "none"
        latest_delivery = next(reversed(self.state.deliveries.values()), None)
        delivery_text = (
            f"{latest_delivery['delivery_id']}@{latest_delivery['second']:.0f}s"
            if latest_delivery else "none")
        procedure = self.state.procedure
        procedure_text = "/".join(str(value) for value in (
            procedure.get("last_completed"), procedure.get("observed_current"))
            if value) or "unknown"
        uncertainty = ", ".join(
            f"{key}:{value}" for key, value in self.state.unresolved_fields.items()) or "none"
        # Internally B may contain several retrieved lines.  Current State
        # exposes only the highest-ranked causal fact; the original frames stay
        # available to the interaction model.
        evidence_lines = [
            " ".join(line.split())
            for line in str(evidence_text or "").splitlines()
            if line.strip()
        ]
        evidence = evidence_lines[0] if evidence_lines else ""
        if evidence.lower() == "none":
            evidence = ""
        evidence = evidence[:max(0, int(evidence_chars))]
        parts: list[str] = []
        # Active obligations are routing state, not new evidence.  Repeating
        # them in every epoch made the interaction model treat a standing request as a fresh
        # trigger.  They remain internal and continue to bind Skill instances.
        if question_text != "none":
            parts.append(f"question={question_text}")
        if procedure_text != "unknown":
            parts.append(f"procedure={procedure_text}")
        if event_text != "none":
            parts.append(f"event={event_text}")
        if delivery_text != "none":
            parts.append(f"delivered={delivery_text}")
            if question_text == "none":
                parts.append("pending=none")
        if uncertainty != "none":
            parts.append(f"uncertainty={uncertainty}")
        if evidence:
            parts.append(f"evidence={evidence}")
        return "CURRENT STATE: " + ("; ".join(parts) if parts else "idle")

    def dump(self) -> dict[str, Any]:
        return {
            "state": asdict(self.state),
            "events": [asdict(event) for event in self.events],
            "event_count": len(self.events),
            "duplicate_count": self._duplicates,
        }
