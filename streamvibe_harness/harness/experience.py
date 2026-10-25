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

"""Experience store: static skills, keyword recall, and bounded counters.

Skills are retrieved by keyword overlap and gated by active request criteria.
An empty retrieval is valid. Per-entry counters track retrieval and outcomes.

SKILL.md contract (six fixed lines after the header):

    # skill: <name>  status: active|candidate|deprecated  family: ...
    trigger: a | b | c
    watch:   ...
    speak:   ...
    silent:  ...
    cadence: ...
    answer:  ...
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent.parent / "skills"
FIELDS = ("trigger", "watch", "speak", "silent", "cadence", "answer")
GENERAL_FAMILIES = frozenset({
    "evidence_selection", "visual_question_answering", "event_detection",
    "causal_explanation", "anomaly_monitoring", "procedural_milestone",
    "procedural_error_detection", "temporal_narration",
    "procedural_guidance", "procedural_summarization",
})
PROMPT_PROJECTIONS = frozenset({"legacy", "answer_only", "execution_state"})
STOP = {"the", "a", "an", "me", "i", "if", "when", "whenever", "to", "and", "or", "of",
        "in", "on", "it", "is", "this", "that", "my", "you", "your", "with", "for", "as",
        "let", "know", "tell", "please", "just", "also", "oh", "any", "what", "do"}

# These families lost useful events when compact E exposed only negative
# conditions.  Their decision state needs both sides of the gate.  Sparse
# detection/narration/MCQ skills retain the conservative two-line projection
# that reduced repeated or premature responses in the first fast gate.
BALANCED_COMPACT = frozenset({
    "answer_question", "explain", "hazard_monitor", "milestone",
    "mistake_watch", "next_step", "stage_summary",
})


SKILL_CODES = {
    "answer_mcq": ("ANSWER_SUPPORTED|STREAM_END", "EVIDENCE_INSUFFICIENT|ALREADY_ANSWERED"),
    "answer_question": ("ANSWER_SUPPORTED", "EVIDENCE_INSUFFICIENT|ALREADY_ANSWERED"),
    "event_watch": ("TARGET_FIRST_VISIBLE", "TARGET_ABSENT|ALREADY_REPORTED"),
    "explain": ("EXPLAINABLE_TRANSITION|VISIBLE_CAUSE", "SAME_TECHNIQUE|ALREADY_EXPLAINED"),
    "hazard_monitor": ("TARGET_EVENT|EVENT_ESCALATION", "NORMAL_ACTIVITY|UNCHANGED_EVENT"),
    "milestone": ("STAGE_COMPLETE|FINAL_COMPLETE", "SUBSTEP_ONLY|IN_PROGRESS"),
    "mistake_watch": ("VISIBLE_ERROR|WRONG_ORDER|WRONG_SETTING", "NO_ERROR|CORRECTED|IN_PROGRESS"),
    "narrate": ("SCENE_CHANGE|ACTION_CHANGE", "STATIC_SCENE|ALREADY_DESCRIBED"),
    "next_step": ("STEP_COMPLETE|HELP_NEEDED|ORDER_ERROR", "IN_PROGRESS|UNCHANGED|NEXT_ALREADY_GIVEN"),
    "stage_summary": ("STAGE_COMPLETE|BETWEEN_STAGES", "MID_STAGE|ALREADY_SUMMARISED"),
}


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z一-鿿]+", text.lower()) if w not in STOP}


@dataclass
class Skill:
    name: str
    status: str
    family: str
    fields: dict[str, str]
    retrieved: int = 0
    helpful: int = 0
    harm: int = 0

    @property
    def trigger_tokens(self) -> set[str]:
        return _tokens(self.fields.get("trigger", ""))

    def render(self, lines: tuple[str, ...] = ("watch", "speak", "silent", "cadence", "answer")) -> str:
        out = [f"skill {self.name}:"]
        for key in lines:
            value = self.fields.get(key, "").strip()
            if value:
                out.append(f"  {key}: {value}")
        return "\n".join(out)


def parse_skill(text: str, name_hint: str = "") -> Skill:
    head = re.search(r"#\s*skill:\s*(\S+).*?status:\s*(\S+)(?:.*?family:\s*(\S+))?", text)
    if not head:
        raise ValueError(f"bad SKILL.md header in {name_hint or text[:40]!r}")
    fields: dict[str, str] = {}
    for key in FIELDS:
        m = re.search(rf"^{key}:\s*(.+?)\s*$", text, re.M)
        if m:
            fields[key] = " ".join(m.group(1).split())
    missing = [k for k in FIELDS if k not in fields]
    if missing:
        raise ValueError(f"{head.group(1)}: missing fields {missing}")
    return Skill(head.group(1), head.group(2), head.group(3) or "", fields)


@dataclass
class ExperienceStore:
    skills: dict[str, Skill] = field(default_factory=dict)
    max_render: int = 2
    grounded_only: bool = False
    min_grounding_overlap: int = 0
    profile_compatibility: dict[str, tuple[str, ...]] = field(default_factory=dict)
    last_recall: dict = field(default_factory=dict)

    @classmethod
    def load(cls, directory: Path = SKILL_DIR, *,
             require_general_taxonomy: bool = False, **kw) -> "ExperienceStore":
        store = cls(**kw)
        for path in sorted(Path(directory).glob("*.md")):
            skill = parse_skill(path.read_text(encoding="utf-8"), path.name)
            if skill.status == "active":
                if (require_general_taxonomy
                        and skill.family not in GENERAL_FAMILIES):
                    raise ValueError(
                        f"{path.name}: non-general skill family {skill.family!r}")
                store.skills[skill.name] = skill
        if require_general_taxonomy and not store.skills:
            raise ValueError(f"no active general skills found in {directory}")
        return store

    def recall(self, active_types: list[str], request_text: str = "",
               question_pending: bool = False, *, belief_text: str = "",
               procedure_text: str = "", event_text: str = "",
               semantic_profile: str = "") -> list[Skill]:
        """Recall against request state plus relevant B/procedure/event context.

        A pending scene question recalls `answer_question` first: answering it
        outranks every standing criterion while it is open. Direct criterion
        matches remain deterministic; contextual trigger overlap supplies a
        bounded fallback when no typed criterion maps to a skill.
        """
        active_types = list(active_types)
        context = " ".join(v for v in (
            belief_text, procedure_text, event_text) if v).strip()
        context_tokens = _tokens(context)
        request_tokens = _tokens(request_text)
        allowed = set(self.profile_compatibility.get(semantic_profile, ()))

        def compatible(skill: Skill) -> bool:
            return not self.profile_compatibility or skill.name in allowed

        def grounded(skill: Skill) -> bool:
            if not self.grounded_only:
                return True
            if not context_tokens:
                return False
            if self.min_grounding_overlap <= 0:
                return True
            focus = request_tokens | skill.trigger_tokens
            return len(focus & context_tokens) >= self.min_grounding_overlap

        chosen: list[Skill] = []
        if question_pending and "answer_question" in self.skills:
            skill = self.skills["answer_question"]
            if compatible(skill) and grounded(skill):
                chosen.append(skill)
        for kind in active_types:
            if kind in self.skills:
                skill = self.skills[kind]
                if (skill not in chosen and compatible(skill) and grounded(skill)):
                    chosen.append(skill)
        retrieval_context = " ".join(v for v in (
            request_text, belief_text, procedure_text, event_text) if v)
        if not chosen and context:
            words = _tokens(retrieval_context)
            scored = []
            for skill in self.skills.values():
                if not compatible(skill) or not grounded(skill):
                    continue
                overlap = len(words & skill.trigger_tokens)
                if overlap >= 2:
                    scored.append((overlap, skill.name, skill))
            for _, _, skill in sorted(scored, reverse=True)[: self.max_render]:
                chosen.append(skill)
        chosen = chosen[: self.max_render]
        for skill in chosen:
            skill.retrieved += 1
        self.last_recall = {
            "profile": semantic_profile,
            "active_types": active_types,
            "question_pending": bool(question_pending),
            "grounded_context": bool(context_tokens),
            "selected": [skill.name for skill in chosen],
        }
        return chosen

    def render(self, skills: list[Skill], compact: bool = False,
               symbolic: bool = False, *, projection: str = "legacy") -> str:
        """The E slot text (epoch-constant); "" when nothing was recalled.

        compact: one skill and a task-conditioned decision projection. Sparse
        event/narration/MCQ tasks retain silent + cadence; guidance, monitoring,
        lifecycle and question tasks receive speak + silent + cadence so E does
        not become a global suppression prior.

        ``answer_only`` is a content-form projection, not a response gate.  It
        exposes only the Skill's answer field and deliberately omits trigger,
        watch, speak, silent and cadence.  The interaction model therefore
        retains its learned response/silence decision while receiving a compact
        hint about how to realise text if it independently chooses response.
        """
        if projection not in PROMPT_PROJECTIONS:
            raise ValueError(f"unknown Experience prompt projection {projection!r}")
        if projection == "execution_state":
            raise ValueError(
                "execution_state projection is rendered by StateRuntime")
        if projection == "answer_only":
            rendered = []
            for skill in skills[: self.max_render]:
                answer = skill.fields.get("answer", "").strip()
                if answer:
                    rendered.append(
                        f"format-only {skill.name} (not a reason to speak): {answer}")
            return "\n".join(rendered)
        if symbolic:
            rendered = []
            for skill in skills[: self.max_render]:
                positive, negative = SKILL_CODES.get(
                    skill.name, ("CRITERION_MATCH", "NO_CRITERION_MATCH"))
                cadence = skill.fields.get("cadence", "").strip()
                rendered.append(
                    f"policy={skill.name}; positive={positive}; "
                    f"negative={negative}; cadence={cadence}")
            return "\n".join(rendered)
        if compact:
            rendered = []
            for skill in skills[:1]:
                lines = (("speak", "silent", "cadence")
                         if skill.name in BALANCED_COMPACT
                         else ("silent", "cadence"))
                rendered.append(skill.render(lines=lines))
            return "\n".join(rendered)
        return "\n".join(s.render() for s in skills)

    def stats(self) -> dict:
        return {name: {"retrieved": s.retrieved,
                       "helpful": s.helpful, "harm": s.harm}
                for name, s in self.skills.items() if s.retrieved}

    def receipt(self) -> dict:
        return {
            "enabled": True,
            "loaded": sorted(self.skills),
            "max_render": self.max_render,
            "grounded_only": self.grounded_only,
            "min_grounding_overlap": self.min_grounding_overlap,
            "profile_compatibility": {
                key: list(value)
                for key, value in sorted(self.profile_compatibility.items())
            },
            "last_recall": dict(self.last_recall),
            "stats": self.stats(),
        }
