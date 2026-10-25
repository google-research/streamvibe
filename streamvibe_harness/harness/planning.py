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

"""Cached procedural state for proactive help, without a planner agent.

The base streaming model still owns the speak/silence decision.  This module is
only a structured part of memory: it turns explicit fields emitted by the
existing memory summariser into a small, persistent help plan.  Ordinary QA and
narration never activate it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


DEFAULT_PROCEDURAL_WORDS = (
    "help", "guide", "show me", "how to", "make", "build", "assemble",
    "install", "repair", "fix", "cook", "prepare", "procedure", "step",
    "walk me", "monitor", "review", "watch", "track", "explain",
    "keep an eye", "stay on top", "scan",
)


def _clean(value: str) -> str:
    return " ".join((value or "").strip().strip("-* ").split())[:500]


def _usable(value: str) -> bool:
    normalized = re.sub(r"[^a-z0-9 ]+", "", (value or "").lower()).strip()
    return bool(normalized and normalized not in {
        "none", "na", "unknown", "not observed", "not established",
    } and not normalized.startswith(("none ", "no explicit ", "not visible ",
                                      "not yet ", "unclear ")))


def _cue_seen(value: str) -> str:
    """Return positive cue evidence; normalise explicit negative placeholders."""
    cleaned = _clean(value)
    normalized = re.sub(r"[^a-z0-9 ]+", " ", cleaned.lower()).strip()
    if (not _usable(cleaned) or normalized in {"no", "false", "not seen"}
            or normalized.startswith(("no ", "false ", "not seen ",
                                      "not observed ", "not visible "))):
        return ""
    return cleaned


def _labelled_fields(text: str, labels: tuple[str, ...]) -> dict[str, str]:
    """Parse labelled fields even when a small model puts them on one line.

    The planner occasionally emits ``... | RESPONSE-RELEVANT IF: ...`` instead
    of the requested newline.  Splitting at every recognised label preserves
    ordinary ``|`` separators inside a field while recovering later labels.
    """
    if not text:
        return {}
    names = "|".join(re.escape(label) for label in labels)
    matches = list(re.finditer(
        rf"(?i)(?:\*\*)?({names})(?:\*\*)?\s*:\s*", text))
    fields: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        value = text[match.end():end]
        value = re.sub(r"^[\s|*`-]+|[\s|*`-]+$", "", value)
        fields[match.group(1).upper()] = _clean(value)
    return fields


@dataclass
class ProceduralPlanMemory:
    """Evidence-grounded goal/progress/deviation ledger.

    It deliberately does not invent an ideal recipe.  A field changes only when
    the memory summariser emits the corresponding explicit ``PLAN ...`` line.
    This keeps an uncertain plan from becoming false evidence downstream.
    """

    enabled: bool = False
    activation_words: tuple[str, ...] = DEFAULT_PROCEDURAL_WORDS
    max_completed: int = 8
    max_anchors: int = 12
    cue_planner_enabled: bool = False
    active: bool = False
    goal: str = ""
    current: str = ""
    next_step: str = ""
    watch_progress: str = ""
    watch_complete: str = ""
    watch_deviation: str = ""
    deviation: str = ""
    recovery: str = ""
    reference_stages: list[str] = field(default_factory=list)
    continue_if: str = ""
    response_if: str = ""
    deviation_if: str = ""
    recovery_reference: str = ""
    cue_continue_seen: str = ""
    cue_response_seen: str = ""
    cue_deviation_seen: str = ""
    cue_t0: float | None = None
    cue_t1: float | None = None
    reference_updates: int = 0
    reference_raw: str = ""
    reference_fields: list[str] = field(default_factory=list)
    reference_parse_failures: int = 0
    completed: list[str] = field(default_factory=list)
    anchors: list[dict] = field(default_factory=list)
    updates: int = 0

    @classmethod
    def from_config(cls, config: dict | None) -> "ProceduralPlanMemory":
        cfg = config or {}
        words = tuple(str(v).lower() for v in
                      cfg.get("activation_words", DEFAULT_PROCEDURAL_WORDS))
        return cls(enabled=bool(cfg.get("enabled", False)),
                   cue_planner_enabled=bool(cfg.get("cue_planner_enabled", False)),
                   activation_words=words,
                   max_completed=max(1, min(16, int(cfg.get("max_completed", 8)))),
                   max_anchors=max(1, min(24, int(cfg.get("max_anchors", 12)))))

    def set_goal(self, request: str) -> bool:
        request = _clean(request)
        if not self.enabled or not request:
            return False
        lowered = request.lower()
        if not any(word in lowered for word in self.activation_words):
            return False
        changed = request != self.goal
        self.active = True
        self.goal = request
        return changed

    def update_from_evidence(self, text: str, t0: float, t1: float) -> bool:
        if not self.active or not text:
            return False
        fields: dict[str, str] = {}
        for raw in text.splitlines():
            match = re.match(
                r"\s*(PLAN CURRENT|PLAN COMPLETED|PLAN NEXT|WATCH PROGRESS|"
                r"WATCH COMPLETE|WATCH DEVIATION|CUE CONTINUE SEEN|"
                r"CUE RESPONSE SEEN|CUE DEVIATION SEEN|DEVIATION|RECOVERY)\s*:\s*(.+)",
                raw, re.IGNORECASE)
            if match:
                fields[match.group(1).upper()] = _clean(match.group(2))
        changed = False
        mapping = {
            "PLAN CURRENT": "current", "PLAN NEXT": "next_step",
            "WATCH PROGRESS": "watch_progress",
            "WATCH COMPLETE": "watch_complete",
            "WATCH DEVIATION": "watch_deviation",
            "DEVIATION": "deviation", "RECOVERY": "recovery",
        }
        for key, attr in mapping.items():
            value = fields.get(key, "")
            if _usable(value) and value != getattr(self, attr):
                setattr(self, attr, value)
                changed = True
        completed = fields.get("PLAN COMPLETED", "")
        new_completed: list[str] = []
        if _usable(completed):
            parts = [_clean(v) for v in re.split(r"\s*[;|]\s*", completed)]
            for part in parts:
                if part and part.lower() not in {v.lower() for v in self.completed}:
                    self.completed.append(part)
                    new_completed.append(part)
                    changed = True
            self.completed = self.completed[-self.max_completed:]
        # Decision cues describe only the chunk that was just inspected.  When
        # the tracker explicitly writes unknown/none, clear the old cue instead
        # of letting a past event masquerade as current evidence.
        cue_mapping = {
            "CUE CONTINUE SEEN": "cue_continue_seen",
            "CUE RESPONSE SEEN": "cue_response_seen",
            "CUE DEVIATION SEEN": "cue_deviation_seen",
        }
        cue_present = False
        for key, attr in cue_mapping.items():
            if key not in fields:
                continue
            cue_present = True
            value = _cue_seen(fields[key])
            if value != getattr(self, attr):
                setattr(self, attr, value)
                changed = True
        # Keep the structured fields internally consistent.  A small visual
        # tracker sometimes emits PLAN COMPLETED / DEVIATION but then writes
        # CUE RESPONSE SEEN: none.  A newly added observed completion or
        # deviation is itself positive decision evidence; compiling that fact
        # does not choose the interaction model's action.
        if new_completed and not self.cue_response_seen:
            self.cue_response_seen = "newly completed stage: " + "; ".join(new_completed)
            cue_present = True
            changed = True
        if ("DEVIATION" in fields and _usable(fields["DEVIATION"])
                and not self.cue_deviation_seen):
            self.cue_deviation_seen = fields["DEVIATION"]
            cue_present = True
            changed = True
        if cue_present:
            self.cue_t0, self.cue_t1 = float(t0), float(t1)
        if changed:
            self.updates += 1
            self.anchors.append({"t0": float(t0), "t1": float(t1)})
            self.anchors = self.anchors[-self.max_anchors:]
        return changed

    def update_reference(self, text: str) -> bool:
        """Parse request-only reference knowledge, never observed progress."""
        if not self.active or not text:
            return False
        self.reference_raw = " ".join(text.split())[:1200]
        fields = _labelled_fields(text, (
            "REFERENCE STAGES", "CONTINUE IF", "RESPONSE-RELEVANT IF",
            "DEVIATION IF", "RECOVERY REFERENCE"))
        self.reference_fields = sorted(fields)
        if not fields:
            self.reference_parse_failures += 1
        changed = False
        stages = fields.get("REFERENCE STAGES", "")
        if _usable(stages):
            values = [_clean(v) for v in re.split(r"\s*[|;]\s*", stages)
                      if _usable(v)][:8]
            if values and values != self.reference_stages:
                self.reference_stages = values
                changed = True
        for key, attr in {
            "CONTINUE IF": "continue_if",
            "RESPONSE-RELEVANT IF": "response_if",
            "DEVIATION IF": "deviation_if",
            "RECOVERY REFERENCE": "recovery_reference",
        }.items():
            value = fields.get(key, "")
            if _usable(value) and value != getattr(self, attr):
                setattr(self, attr, value)
                changed = True
        if changed:
            self.reference_updates += 1
        return changed

    def render_reference(self) -> str:
        if not self.active:
            return ""
        lines = []
        if self.reference_stages:
            lines.append("REFERENCE PLAN: " + " -> ".join(self.reference_stages))
        if self.continue_if:
            lines.append("WATCH continuing if: " + self.continue_if)
        response_parts = [v for v in (self.response_if, self.deviation_if) if v]
        if response_parts:
            lines.append("WATCH response-relevant if: " + " | ".join(response_parts))
        if self.recovery_reference:
            lines.append("RECOVERY REFERENCE: " + self.recovery_reference)
        return "\n".join(lines)

    def render_cue_evidence(self) -> str:
        """Lagging observations from the last completed visual chunk."""
        if not self.active:
            return ""
        if self.cue_t1 is None:
            return ""
        triggers = []
        if self.cue_response_seen:
            triggers.append("response-relevant: " + self.cue_response_seen)
        if self.cue_deviation_seen:
            triggers.append("deviation: " + self.cue_deviation_seen)
        stamp = f"[{self.cue_t0:.0f}-{self.cue_t1:.0f}s; past]"
        if triggers:
            return (f"P.past decision evidence {stamp}: PAST_TRIGGER — "
                    + " | ".join(triggers)
                    + ". This is past evidence; current frames and SAID decide whether it is new now.")
        # Absence of a past trigger is not evidence about the current frame.
        # Rendering NO_HISTORY/NO_TRIGGER here caused the interaction model to miss current events
        # (Jobs 933/934).  Current continuation is handled by the event parser.
        return ""

    def render(self) -> str:
        if not self.active:
            return ""
        lines = ["## Cached active-help plan (past state, not a reason to speak)",
                 f"GOAL: {self.goal}"]
        if self.completed:
            lines.append("COMPLETED: " + "; ".join(self.completed))
        for label, value in (
            ("CURRENT", self.current), ("NEXT", self.next_step),
            ("WATCH PROGRESS", self.watch_progress),
            ("WATCH COMPLETE", self.watch_complete),
            ("WATCH DEVIATION", self.watch_deviation),
            ("DEVIATION", self.deviation), ("RECOVERY", self.recovery),
        ):
            if value:
                lines.append(f"{label}: {value}")
        return "\n".join(lines)

    def render_line(self) -> str:
        """One-line P.procedure slot for the card: only evidence-backed fields."""
        if not self.active:
            return ""
        parts = []
        if self.completed:
            parts.append("done: " + "; ".join(self.completed[-4:]))
        if self.current:
            parts.append(f"current: {self.current}")
        if self.next_step:
            parts.append(f"next: {self.next_step}")
        if self.deviation:
            parts.append(f"deviation: {self.deviation}")
        if not parts:
            return "P.procedure observed progress: unknown (no visible step evidence yet)"
        anchor = self.anchors[-1] if self.anchors else None
        stamp = f" [evidence {anchor['t0']:.0f}-{anchor['t1']:.0f}s]" if anchor else ""
        return "P.procedure observed progress: " + " | ".join(parts) + stamp

    def dump(self) -> dict:
        return {
            "enabled": self.enabled, "active": self.active, "goal": self.goal,
            "current": self.current, "completed": list(self.completed),
            "next": self.next_step, "watch_progress": self.watch_progress,
            "watch_complete": self.watch_complete,
            "watch_deviation": self.watch_deviation,
            "deviation": self.deviation, "recovery": self.recovery,
            "reference_stages": list(self.reference_stages),
            "continue_if": self.continue_if, "response_if": self.response_if,
            "deviation_if": self.deviation_if,
            "recovery_reference": self.recovery_reference,
            "cue_continue_seen": self.cue_continue_seen,
            "cue_response_seen": self.cue_response_seen,
            "cue_deviation_seen": self.cue_deviation_seen,
            "cue_t0": self.cue_t0, "cue_t1": self.cue_t1,
            "reference_updates": self.reference_updates,
            "reference_raw": self.reference_raw,
            "reference_fields": list(self.reference_fields),
            "reference_parse_failures": self.reference_parse_failures,
            "anchors": list(self.anchors), "updates": self.updates,
        }
