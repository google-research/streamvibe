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

"""Environment adapter: event points -> harness meta-actions -> deltas.

The base model emits ``{response, silence}``. The adapter performs tracking,
commit, and recall operations at event points:

    instruction arrives   commit(request_state)  ->  "[Δ ts] criteria: ..."
    question arrives      track[object] on B     ->  "[Δ ts] B: fact; fact"
    model speaks          commit(said/question)  ->  next-turn delta, or boundary fold
    epoch boundary        render(B, P, E)        ->  new header

Everything it writes inside an epoch goes to ``turn_extras[t]`` for the turn
the event happened on, once, and is never edited afterwards, so the rendered
request stays a byte prefix of the next one.  The header is only ever built
at a boundary (``render_header``).
"""
from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field

from .card import render_card
from .context import build_chunk_header
from .criteria import TYPES, CriterionSet, parse_instruction
from .experience import ExperienceStore
from .routing import request_evidence_focus
from .evolved.evidence_honest_state_runtime import StateRuntime
from .evolved.guidance_procedure_bridge import sync_guidance_procedure

STOP = {"the", "a", "an", "i", "did", "do", "does", "is", "are", "was", "were", "how",
        "many", "much", "what", "which", "where", "when", "of", "on", "in", "to", "my",
        "me", "it", "this", "that", "there", "have", "has", "had", "yet", "already", "right",
        "now", "left", "should", "am", "using", "put", "add", "added"}

# Function words and pronouns are deliberately excluded from entity hints.  The
# CJK list is small rather than pretending to be a segmenter: `_terms` also emits
# characters and bigrams, which is sufficient for deterministic matching of
# concrete strings such as 刀 / 烤箱 / 砧板 without adding a model call.
CJK_STOP = ("它", "这个", "那个", "刚才", "现在", "后来", "哪里", "哪儿", "什么",
            "多少", "怎么", "了吗", "是不是", "被", "把", "放在", "放到", "放", "的",
            "了", "吗", "呢", "上", "下", "里")


def _nouns(text: str) -> set[str]:
    """Cheap multilingual lexical terms used for retrieval, not linguistic POS tags."""
    value = text or ""
    words = {w for w in re.findall(r"[a-z0-9]+", value.lower())
             if len(w) > 2 and w not in STOP}
    # crude singularisation so "screws" hits "screw"
    words |= {w[:-1] for w in words if w.endswith("s")}
    for span in re.findall(r"[\u4e00-\u9fff]+", value):
        for stop in CJK_STOP:
            span = span.replace(stop, " ")
        for part in span.split():
            words.update(part)
            words.update(part[i:i + 2] for i in range(max(0, len(part) - 1)))
    return words


def _focus_terms(text: str) -> set[str]:
    terms = _nouns(text)
    low = (text or "").lower()
    groups = {
        "temperature": {"temperature", "temp", "degree", "target", "display", "setting"},
        "quantity": {"count", "amount", "quantity", "remain", "remaining", "left"},
        "location": {"location", "place", "placed", "position"},
    }
    if any(v in low for v in ("temperature", "temp", "degree", "温度")):
        terms |= groups["temperature"]
    if any(v in low for v in ("how many", "how much", "count", "amount", "多少")):
        terms |= groups["quantity"]
    if any(v in low for v in ("where", "location", "哪里", "哪儿")):
        terms |= groups["location"]
    return terms


PLAN_BLOCK_HEAD = "## Cached active-help plan"


def strip_plan_block(memory_text: str) -> str:
    """Remove ThreeTierMemory's plan block (it is rendered as the procedure line instead)."""
    if not memory_text or PLAN_BLOCK_HEAD not in memory_text:
        return memory_text
    out, skipping = [], False
    for line in memory_text.splitlines():
        if line.startswith(PLAN_BLOCK_HEAD):
            skipping = True
            continue
        if skipping and line.startswith("## "):
            skipping = False
        if not skipping:
            out.append(line)
    return "\n".join(out).strip()


def observed_procedure(line: str) -> bool:
    """Whether a compact procedure line contains actual visual progress."""
    value = " ".join((line or "").lower().split())
    return bool(value and "unknown" not in value
                and "no visible step evidence" not in value)


def track_object(belief_text: str, question: str, k: int = 3,
                 entity_hints=()) -> list[str]:
    """Return newest belief lines matching explicit terms or prior entity hints.

    This is intentionally a deterministic resolver.  It does not claim general
    coreference: a pronoun-only question can resolve only when an earlier event
    established an entity hint.  Ambiguous questions therefore return no fact
    instead of guessing.
    """
    if not belief_text or not question:
        return []
    want = _nouns(question)
    if not want:
        want = set(entity_hints)
    if not want:
        return []
    hits = []
    for order, line in enumerate(belief_text.splitlines()):
        body = line.strip().lstrip("-• ").strip()
        if (not body
                or body.startswith(("REQUEST STATE", "VISUAL MEMORY FOCUS",
                                    "EVENTS:", "CHUNK END"))):
            continue
        toks = _nouns(body)
        score = len(want & toks)
        if score:
            stamps = [float(v) for v in re.findall(r"([0-9]+(?:\.[0-9]+)?)s", body)]
            newest = max(stamps) if stamps else float(order)
            hits.append((score, newest, order, body))
    hits.sort(key=lambda x: (-x[0], -x[1], -x[2]))
    return [body for _, _, _, body in hits[:k]]


def select_belief(belief_text: str, focus: str, k: int = 3,
                  anchors=()) -> str:
    """Compile rich B into at most ``k`` decision-relevant timestamped facts.

    Relevance is lexical and deterministic.  When no focus term matches, retain
    the newest facts rather than dumping the whole memory hierarchy into the interaction model.
    Section headings and procedural-plan fields are control text, not B facts.
    """
    if not belief_text or k <= 0:
        return ""
    want = _focus_terms(focus)
    facts, events = [], []
    for order, raw in enumerate(belief_text.splitlines()):
        body = raw.strip().lstrip("-• ").strip()
        if not body or body.startswith(("## ", "GOAL:", "COMPLETED:", "CURRENT:",
                                        "NEXT:", "WATCH ", "CUE ", "DEVIATION:", "RECOVERY:")):
            continue
        toks = _nouns(body)
        overlap = len(want & toks)
        fact_body = re.sub(r"^\[[^]]+\]\s*", "", body)
        if any(v in (focus or "").lower() for v in ("temperature", "temp", "degree", "温度")) \
                and re.search(r"\d|°|\b[cf]\b", fact_body.lower()):
            overlap += 1
        stamps = [float(v) for v in re.findall(r"([0-9]+(?:\.[0-9]+)?)s", body)]
        newest = max(stamps) if stamps else float(order)
        if stamps and any(min(stamps) <= float(a.get("t1", -1)) and
                          max(stamps) >= float(a.get("t0", 1e30))
                          for a in anchors):
            overlap += 2
        candidate = (overlap, newest, order, body)
        if re.match(r"^E\d{4}\s+\[", body):
            events.append(candidate)
        else:
            facts.append(candidate)
    if not facts and not events:
        return ""

    def choose(rows, *, chronological: bool = False):
        matched = [row for row in rows if row[0] > 0]
        pool = matched or rows
        pool = sorted(pool, key=lambda row: (-row[0], -row[1], -row[2]))[:k]
        if chronological:
            pool.sort(key=lambda row: row[2])
        return pool

    selected_facts = choose(facts)
    selected_events = choose(events, chronological=True)
    parts = []
    if selected_facts:
        parts.append(" · ".join(row[3] for row in selected_facts))
    if selected_events:
        parts.append("## Ordered observed transitions\n" +
                     "\n".join(row[3] for row in selected_events))
    return "\n".join(parts)


@dataclass
class RequestLog:
    """Bounded `said` record for request_state (ablation slot; off by default)."""
    keep: int = 4
    items: deque = field(default_factory=lambda: deque(maxlen=4))

    def __post_init__(self):
        self.items = deque(maxlen=self.keep)

    def add(self, second: float, text: str) -> None:
        if text:
            self.items.append((second, " ".join(text.split())[:120]))

    def render(self) -> str:
        if not self.items:
            return "P.request said: none"
        return "P.request said: " + " · ".join(f'"{t}" @{s:.0f}s' for s, t in self.items)


@dataclass
class HarnessAdapter:
    """Per-lane state and the four meta-actions, driven by the runner."""
    slots: set[str]
    enabled: bool = True
    criteria: CriterionSet = field(default_factory=CriterionSet)
    said: RequestLog = field(default_factory=RequestLog)
    experience: ExperienceStore | None = None
    experience_projection: str = "legacy"
    semantic_profile: str = ""
    belief_query_mode_by_criterion: dict[str, str] = field(default_factory=dict)
    show_single_request_for_criteria: set[str] = field(default_factory=set)
    track_k: int = 3
    recalled: list = field(default_factory=list)
    request_text: str = ""
    entity_hints: deque = field(default_factory=lambda: deque(maxlen=12))
    pending_deltas: deque = field(default_factory=deque)
    events: list[dict] = field(default_factory=list)
    rendered_slots: list[str] = field(default_factory=list)
    state_runtime: StateRuntime = field(default_factory=StateRuntime)

    def _render_experience(self, *, compact: bool, symbolic: bool) -> str:
        if self.experience is None:
            return ""
        if self.experience_projection == "execution_state":
            # Keep every obligation in private state, but expose only Skills
            # selected by evidence-grounded recall.  An active request alone is
            # not evidence that its response condition currently holds.
            self.state_runtime.sync_skill_instances(self.experience.skills)
            recalled = {skill.name: skill for skill in self.recalled}
            return self.state_runtime.render_skill_projection(
                recalled, max_visible=self.experience.max_render)
        return self.experience.render(
            self.recalled, compact=compact, symbolic=symbolic,
            projection=self.experience_projection)

    def render_response_support(self, projection: str = "answer_only") -> str:
        """Render E only for an already-chosen the interaction model response continuation.

        This method is read-only and does not recall, route, or select an
        action.  ``self.recalled`` was produced causally from B/P at the request
        event or epoch boundary.  The runner may expose this text only after
        the interaction model itself emitted ``</response>``.
        """
        if self.experience is None or not self.recalled:
            return ""
        experience_projection = (
            "answer_only" if projection == "answer_schema" else projection)
        return self.experience.render(
            self.recalled, compact=True, symbolic=False,
            projection=experience_projection)

    def _remember_entities(self, text: str, belief_text: str = "") -> None:
        terms = _nouns(text)
        if belief_text:
            belief_terms = _nouns(belief_text)
            terms = {term for term in terms if term in belief_terms}
        for term in sorted(terms, key=lambda v: (len(v), v)):
            if term in self.entity_hints:
                self.entity_hints.remove(term)
            self.entity_hints.append(term)

    def _record(self, kind: str, second: float, changes: list[str], **extra) -> None:
        self.events.append({"kind": kind, "second": float(second),
                            "changes": list(changes), **extra})

    # -- event: instruction / question arrives --------------------------------
    def on_instruction(self, text: str, second: float, belief_text: str) -> str:
        """commit(request_state) and, for a question, track[object]. Returns the delta ("" if none)."""
        first = not self.criteria.active and not self.criteria.log
        ops = parse_instruction(text, first=first)
        self.state_runtime.apply_instruction(ops, text, second)
        frags = self.criteria.apply(ops, second)
        if self.experience is not None:
            self.state_runtime.sync_skill_instances(self.experience.skills)
        # Keep all additive standing criteria as retrieval focus. A pending
        # scene question remains separate and is ranked first by Experience.
        self.request_text = self.criteria.focus_text() or text
        self._remember_entities(text, belief_text)
        self._record("instruction", second, frags, text=text)
        parts = []
        if not self.card_wanted():
            return ""
        if frags and "request" in self.slots:
            parts.append(f"[Δ {second:.0f}s] criteria: " + ", ".join(frags))
        if self.criteria.question and "belief" in self.slots:
            facts = track_object(belief_text, self.criteria.question, self.track_k,
                                 self.entity_hints)
            if facts:
                parts.append(f"[Δ {second:.0f}s] B (earlier facts, may be stale): " + " ; ".join(facts))
        if frags and self.experience is not None:
            previous = [s.name for s in self.recalled]
            self.recalled = self.experience.recall(
                list(self.criteria.active), self.request_text,
                question_pending=self.criteria.question is not None,
                belief_text=belief_text,
                semantic_profile=self.semantic_profile)
            current = [s.name for s in self.recalled]
            if current != previous and "experience" in self.slots:
                rendered = self._render_experience(
                    compact="compact" in self.slots,
                    symbolic="symbolic" in self.slots)
                parts.append(f"[Δ {second:.0f}s] E recall: " + (rendered or "none"))
        return "\n".join(parts)

    # -- event: model speaks --------------------------------------------------
    def on_response(self, second: float, text: str, *,
                    answer_question: bool = True) -> None:
        answered = self.criteria.question
        self.state_runtime.commit_response(
            second, text, answer_question=answer_question)
        if answer_question:
            self.criteria.answer_question()
        self.said.add(second, text)
        changes = []
        if answered and answer_question and "request" in self.slots:
            changes.append(f"question: -{answered[:120]}")
        if text and "said" in self.slots:
            changes.append(f'said: +"{" ".join(text.split())[:120]}"')
        if changes:
            self.pending_deltas.append(f"[Δ {second:.0f}s] " + " | ".join(changes))
        self._remember_entities(text)
        self._record(
            "response", second, changes, text=" ".join(text.split())[:200],
            question_answered=bool(answered and answer_question))

    def consume_pending_delta(self, *, folded: bool = False) -> str:
        """Return response-side mutations for the next turn, or fold at a boundary.

        A response is known only after its turn's prompt was rendered, so its
        state mutation cannot be attached retroactively.  The next turn carries
        it as an append-only delta.  When that turn starts a new epoch, the new
        header already contains the mutation and the queued delta is discarded.
        """
        pending = "\n".join(
            value for value in (
                "\n".join(self.pending_deltas),
                self.state_runtime.consume_deltas(folded=folded),
            ) if value)
        self.pending_deltas.clear()
        return "" if folded else pending

    def on_confirmed_observation(self, second: float, match_kind: str,
                                 evidence: str, *, evidence_ref: str = "",
                                 instance_id=None) -> str:
        """Convert parser output to a typed visual event before projection."""
        obligation_ids = [
            criterion_id
            for criterion_id, obligation in
            self.state_runtime.state.active_obligations.items()
            if (obligation["criterion"] == self.semantic_profile
                or (self.semantic_profile == "monitoring"
                    and obligation["criterion"] in {
                        "hazard_monitor", "mistake_watch", "milestone",
                        "stage_summary"}))
        ]
        self.state_runtime.observe(
            second, match_kind, evidence, evidence_ref=evidence_ref or None,
            instance_id=instance_id, obligation_ids=obligation_ids)
        return self.state_runtime.consume_deltas(folded=False)

    def card_wanted(self) -> bool:
        if not self.enabled:
            return False
        if "adaptive" not in self.slots:
            return True
        if self.criteria.question:
            return True
        if self.state_runtime.has_meaningful_state():
            return True


        return bool(self.criteria.active)

    def compile_slots(self, procedure_line: str = "") -> set[str]:
        """Project rich internal B/P/E into decision-relevant visible slots.

        `adaptive,compact` is the brief-state compiler. The initial request is
        already in the user turn, so a single unchanged criterion is omitted.
        Request state becomes visible after a lifecycle change or when several
        criteria must be coordinated. B normally keeps the established
        [VIDEO MEMORY] representation; the labelled, bounded B card is used
        only for an actual pending scene question.

        This never discards internal state; it controls only its visible epoch
        projection before the interaction model runs.
        """
        visible = set(self.slots)
        # A receipt exists only after something was actually committed.  An
        # empty ``P.request said: none`` changed the model's first-answer prior
        # even though there was no response-side state to report.  Hide the
        # slot until the next decision after the first raw response; later
        # receipts remain ordinary epoch-frozen pre-decision context.
        if "said" in visible and not self.said.items:
            visible.discard("said")
        # Procedure is evidence-bearing in every projection mode.  The empty
        # shell may still say ``unknown`` for format compatibility, but the
        # procedure slot itself is not marked visible until observed progress
        # exists.  Previously this guard ran only in compact/adaptive mode,
        # making trace provenance claim an unknown procedure was projected.
        if "procedure" in visible and not observed_procedure(procedure_line):
            visible.discard("procedure")
        if not {"adaptive", "compact"}.issubset(visible):
            return visible
        if not self.criteria.question:
            if "anchored" not in visible:
                visible.discard("belief")
        keep_single_request = bool(
            set(self.criteria.active) & self.show_single_request_for_criteria)
        if (not keep_single_request and not self.criteria.retired
                and len(self.criteria.active) <= 1):
            visible.discard("request")
        return visible

    def render_procedure_delta(self, second: float, procedure_line: str) -> str:
        """Project only evidence-grounded procedure progress into the prompt.

        ``StreamDecisionState.sync_procedure`` maintains the richer internal
        lifecycle counters.  Those counters are useful for tracing but are a
        harmful substitute for semantic evidence when no event normaliser is
        running.  A visible delta therefore contains the same observed
        procedure line that would be folded into the next epoch header, and is
        empty for profiles without the procedure slot or before real progress.
        """
        if (not observed_procedure(procedure_line)
                or "procedure" not in self.compile_slots(procedure_line)):
            return ""
        return f"[Δ {second:.0f}s] {procedure_line.strip()}"

    # -- event: epoch boundary ------------------------------------------------
    def render_header(self, mode: str, epoch: int, memory_text: str, carried_query,
                      procedure_line: str = "", belief_text: str | None = None,
                      lifecycle_line: str = "",
                      reference_line: str = "", cue_evidence: str = "",
                      plan_anchors=(), memory_projection: str = "all",
                      query_projection: str = "waiting") -> str:


        if not self.enabled:
            projected_memory = memory_text
            if memory_projection == "none":
                projected_memory = ""
            elif memory_projection == "on_demand" and not self.criteria.question:
                projected_memory = ""
            elif memory_projection not in {"all", "on_demand"}:
                raise ValueError(f"unknown memory projection {memory_projection!r}")
            self.recalled = []
            self.rendered_slots = []
            return build_chunk_header(
                projected_memory, carried_query,
                query_projection=query_projection)
        sync_guidance_procedure(self, procedure_line, epoch)
        if not self.card_wanted():
            self.recalled = []
            self.rendered_slots = []
            return build_chunk_header(
                memory_text, carried_query, query_projection=query_projection)
        visible_slots = self.compile_slots(procedure_line)
        self.rendered_slots = sorted(visible_slots)
        visible_mode = ",".join(
            name.strip() for name in mode.split(",") if name.strip() in visible_slots)
        if "procedure" in visible_slots:
            # the plan block otherwise sits inside the memory text; with the
            # procedure slot on it is rendered as its own line instead
            memory_text = strip_plan_block(memory_text)
        compact = "compact" in visible_slots
        # ``anchored`` requests bounded, query/plan-focused retrieval even when
        # the surrounding card keeps its full neutral shell.  This lets an A/B
        # replace opaque memory with B facts without changing empty-card format
        # before the first evidence exists.
        bounded_belief = compact or "anchored" in visible_slots
        indexed_belief = memory_text if belief_text is None else belief_text
        request_focus = self.request_text
        for criterion in TYPES:
            mode_for_criterion = self.belief_query_mode_by_criterion.get(criterion)
            if criterion in self.criteria.active and mode_for_criterion:
                request_focus = request_evidence_focus(
                    self.request_text, mode_for_criterion)
                break
        focus = " ".join(v for v in (
            self.criteria.question or "", request_focus, procedure_line,
            reference_line) if v)
        visible_belief = (select_belief(indexed_belief, focus, self.track_k,
                                        anchors=plan_anchors)
                          if bounded_belief else indexed_belief)
        experience_text = ""
        if self.experience is not None:
            self.recalled = self.experience.recall(
                list(self.criteria.active), self.request_text,
                question_pending=self.criteria.question is not None,
                belief_text=indexed_belief,
                procedure_text=(procedure_line
                                if observed_procedure(procedure_line) else ""),
                semantic_profile=self.semantic_profile)
            self.state_runtime.sync_skill_instances(self.experience.skills)
            if "experience" in visible_slots:
                experience_text = self._render_experience(
                    compact=compact, symbolic="symbolic" in visible_slots)
        # B remains private and is never replayed as a full memory block.  One
        # request-focused established fact is folded into the canonical state
        # so the interaction model retains a causal silence/speak basis.
        canonical_state = self.state_runtime.render(visible_belief)
        card = render_card(visible_mode, epoch, belief=visible_belief,
                           request_line=(self.criteria.render_compact() if compact
                                         else self.criteria.render()),
                           said_line=self.said.render() if "said" in visible_slots else "",
                           experience=experience_text,
                           procedure_line=procedure_line if "procedure" in visible_slots else "",
                           lifecycle_line=(canonical_state
                                           if "lifecycle" in visible_slots else ""),
                           reference_line=reference_line if "cues" in visible_slots else "",
                           cue_evidence=cue_evidence if "cues" in visible_slots else "")
        projected_memory = "" if "belief" in visible_slots else memory_text
        if memory_projection == "none":
            projected_memory = ""
        elif memory_projection == "on_demand" and not self.criteria.question:
            projected_memory = ""
        elif memory_projection != "all":
            if memory_projection != "on_demand":
                raise ValueError(f"unknown memory projection {memory_projection!r}")
        return build_chunk_header(
            projected_memory, carried_query, card=card,
            query_projection=query_projection)

    def observations(self) -> str:
        """`watch` lines of the recalled skills, for the summariser focus (F5)."""
        if not self.recalled:
            return ""
        return " ; ".join(s.fields.get("watch", "") for s in self.recalled if s.fields.get("watch"))

    def decision_reference(self, *, detailed: bool = True) -> str:
        """Request/Skill-side contrastive boundary, never an observed event."""
        if not self.recalled:
            return ""
        if not detailed:
            names = ", ".join(skill.name for skill in self.recalled[:2])
            if "symbolic" in self.slots:
                return ("P.reference: E policy codes are unobserved conditions "
                        f"({names}); no current match asserted.")
            return ("P.reference decision boundary: apply E skill speak as "
                    f"positive and silent as continuation/negative ({names}); not observed.")
        lines = ["P.reference decision boundary (request/skill; not observed):"]
        for skill in self.recalled[:2]:
            speak = skill.fields.get("speak", "").strip()
            silent = skill.fields.get("silent", "").strip()
            if speak:
                lines.append(f"- {skill.name} RESPONSE-RELEVANT IF: {speak}")
            if silent:
                lines.append(f"- {skill.name} CONTINUE/SILENT IF: {silent}")
        return "\n".join(lines) if len(lines) > 1 else ""

    def perception_reference(self) -> str:
        """Private natural-language criteria for the event parser only.

        Unlike ``decision_reference``, this text is never rendered into the interaction model's
        prompt.  Keeping the rich condition definitions private lets the 4B
        front-end assess relevance while the online model sees only bounded
        symbolic policy state.
        """
        if not self.recalled:
            return "Use the active request literally; no typed Skill was recalled."
        lines = []
        for skill in self.recalled[:2]:
            watch = skill.fields.get("watch", "").strip()
            speak = skill.fields.get("speak", "").strip()
            silent = skill.fields.get("silent", "").strip()
            lines.append(f"CRITERION {skill.name}")
            if watch:
                lines.append(f"WATCH: {watch}")
            if speak:
                lines.append(f"MATCH ONLY IF: {speak}")
            if silent:
                lines.append(f"NOT A MATCH: {silent}")
        return "\n".join(lines)

    def decision_cadence(self) -> dict:
        """Return bounded cadence evidence for the interaction model's pre-decision clock.

        This is not an output gate.  It exposes the primary recalled Skill's
        configured warmup/min-gap as numeric state; the online model can still
        override it when the current frames establish a genuinely new event.
        """
        if not self.recalled:
            return {"policy": "none", "warmup": 0, "min_gap": 0}
        skill = self.recalled[0]
        text = skill.fields.get("cadence", "")
        warmup = re.search(r"warmup\s+(\d+)s", text, flags=re.I)
        gap = re.search(r"min\s+gap\s+(\d+)s", text, flags=re.I)
        return {
            "policy": skill.name,
            "warmup": int(warmup.group(1)) if warmup else 0,
            "min_gap": int(gap.group(1)) if gap else 0,
        }

    def verifier_context(self) -> str:
        """Compact request state for the response verifier (F4); "" when no card."""
        if not self.slots or not self.card_wanted():
            return ""
        return "\n" + self.criteria.render_compact()

    def dump(self) -> dict:
        return {"criteria": self.criteria.dump(),
                "said": list(self.said.items),
                "entity_hints": list(self.entity_hints),
                "events": list(self.events),
                "rendered_slots": list(self.rendered_slots),
                "recalled": [s.name for s in self.recalled],
                "experience_projection": self.experience_projection,
                "experience": self.experience.stats() if self.experience else {}}
