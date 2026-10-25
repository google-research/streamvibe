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

"""Track the active help criteria requested by the user.

Deterministic instruction rules add or retire criteria and track pending
questions. Unrecognized instructions leave the criterion set unchanged and
still reach the interaction model in the original user turn.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Criterion types, in the order they are rendered.
NEXT_STEP, MISTAKE, MILESTONE, SUMMARY = "next_step", "mistake_watch", "milestone", "stage_summary"
EXPLAIN, HAZARD, EVENT, NARRATE, MCQ = "explain", "hazard_monitor", "event_watch", "narrate", "answer_mcq"
TYPES = (NEXT_STEP, MISTAKE, MILESTONE, SUMMARY, EXPLAIN, HAZARD, EVENT, NARRATE, MCQ)

# Phrases that identify a criterion type inside one clause.  Order matters only
# for readability; every matching type is collected.
TYPE_PATTERNS: dict[str, tuple[str, ...]] = {
    NEXT_STEP: (
        r"\bnext step", r"what (to do|i should do|comes) next", r"what step",
        r"walk me through", r"guide me", r"talk me through", r"step[- ]by[- ]step",
        r"the basics", r"what i should be doing", r"what should i do next",
        r"tell me what to do", r"what to do next", r"what .{0,30} next\b", r"\bnext\b.{0,40}\b(when|as|once) i finish",
        r"what (action|piece|step|item)s? .{0,20}(take|try|check|pick|do) next",
    ),
    MISTAKE: (
        r"\bmistake", r"\bmisstep", r"\berror", r"\bwrong\b", r"out of order",
        r"mess(ing)? up", r"\bstop me\b", r"\bcatch (me|any)", r"\bwarn me\b",
        r"interrupt me if", r"\bcorrect me\b", r"\bincorrect", r"\bmisalign",
        r"\bslip[- ]?ups?\b", r"screw(ing)? up", r"mess (anything|something|it|things) up", r"\bmiss (a|any) step",
        r"\bupside down\b", r"\bmiss (a|any) screw", r"\bif i (do|put) .{0,30}(wrong|out of order)",
    ),
    MILESTONE: (
        r"\bmilestone", r"major (step|stage)", r"main (step|stage)", r"key (stage|step)s? (is|are)? ?(done|complete)",
        r"complete[sd]? (a|each|key|major)", r"finish(ed)? (a|each|key|major) (stage|milestone|prep step)",
        r"wrap(ped)? up a (stage|major)", r"call out when a major step",
        r"(when|whenever|once) .*(stage|part|section) (is )?(done|complete|finished)",
        r"(stage|step|section|part) (finishes|is done|is complete|is finished)",
        r"(notify|update) me (only )?(when|once|whenever)",
    ),
    SUMMARY: (
        r"stage summar", r"\brecap\b", r"\bsummary\b", r"\bsummari[sz]e\b", r"sum up my progress",
        r"progress update", r"quick update", r"brief update", r"my progress", r"keep me posted",
        r"where i'?m at",
    ),
    EXPLAIN: (r"\bexplain", r"\bexplaining", r"\bwhy\b.*\bhow\b|\bhow\b.*\bwhy\b", r"\bcommentat", r"\bcoach me"),
    HAZARD: (r"\bmonitor (this|the|these|that)( \w+){0,3} (feed|footage|stream|camera|game|broadcast|video|clip|dashcam|compilation|site|area)\b",
             r"\bsurveillance", r"\bhazard", r"\baltercation", r"\bviolen", r"\bsafety",
             r"\bsuspicious", r"\bdefect", r"\bincident", r"\breview this .*footage",
             r"\breview this .*walkthrough", r"\bproperty damage", r"\bfixture issues?",
             r"\bareas? needing maintenance", r"\bmaintenance (issue|concern|need)s?"),
    EVENT: (r"tell me when .*(appear|happen|show up|shows up)",
            r"remind me (when|whenever)\b", r"flag it",
            r"alert me", r"watch for",
            r"let me know (when|if) .*(appear|happen|show up|shows up|see)",
            r"ping me .{0,24}(when|if|instant|moment)",
            r"帮我看看有没有出现", r"跟我说一声", r"喊我", r"提醒我",
            r"首次(检测|看到|发现|出现)"),
    NARRATE: (r"\bnarrat", r"real-time narration", r"\bbriefly describe\b",
              r"\bdescribe (what|whatever|as) (is|was|you see) happening\b",
              r"\bdescribe (what you see|the visible actions)\b",
              r"\bdescribe .{0,24}\b(video|scene|footage)\b",
              r"实时(解说|讲解|说明|讲讲)", r"简要描述", r"描述",
              r"边看边讲", r"边看.{0,8}讲", r"画面解说", r"比赛解说", r"讲解画面"),
    MCQ: (r"^question:", r"\boptions?:", r"respond only with the letter", r"answer with the letter"),
}

# A criterion word can occur while the user is explicitly denying that
# capability (for example, "do not assume access to audio or narration").
# Treating that token as a positive request adds a phantom active criterion and
# changes routing even though the raw request says the opposite.  The scope is
# deliberately clause-local: a later contrast such as "..., but narrate the
# visible actions" is still a positive request.
NEGATION_SCOPE = re.compile(
    r"(?:\bdo\s+not\b|\bdon['’]?t\b|\bdoes\s+not\b|\bdoesn['’]?t\b|"
    r"\bdid\s+not\b|\bdidn['’]?t\b|\bnot\b|\bnever\b|\bwithout\b|"
    r"\bno\s+need\b|\bcannot\b|\bcan['’]?t\b|\bshouldn['’]?t\b|"
    r"\bmustn['’]?t\b)[^.;:!?]{0,80}$",
    re.I,
)
CONTRAST_BOUNDARY = re.compile(r"[.;:!?]|\b(?:but|however|instead)\b", re.I)


def _positive_pattern_match(text: str, pattern: str) -> bool:
    """Return true when ``pattern`` has a non-negated clause-local match."""
    for match in re.finditer(pattern, text):
        prefix = text[max(0, match.start() - 120):match.start()]
        clause = CONTRAST_BOUNDARY.split(prefix)[-1]
        # "not only X, but also Y" is additive rather than a negation of X.
        if re.search(r"\bnot\s+only\b[^.;:!?]{0,80}$", clause, re.I):
            return True
        if NEGATION_SCOPE.search(clause):
            continue
        return True
    return False

QUESTION_HELP_VERBS = re.compile(
    r"\b(let me know|tell me|walk me|guide me|warn me|stop me|remind me|speak up|"
    r"call out|flag|alert me|notify me|catch|give me|recap|heads-up|summari[sz]e|update me|"
    r"keep me posted|could you|can you|would you|please)\b|^\s*oh,?\s*and\b", re.I)
DIRECT_QUESTION = re.compile(
    r"^\s*(?:please\s+)?(?:answer|solve|calculate|compute|work out)\b|"
    r"^\s*(?:please\s+)?give me (?:the )?(?:answer|solution|result)\b|"
    r"^\s*(?:what|where|who|which|how many|how much)\b|"
    r"(?:请问|请回答|计算|算一下|求解).*(?:多少|答案|结果)?", re.I)

RETIRE_CUES = (
    r"don'?t need to remind me about (?P<what>.+?) anymore",
    r"no need to (remind me about|walk me through|tell me about|mention) (?P<what>.+?)( anymore|\.|,|—|-|$)",
    r"never mind (the )?(?P<what>.+?)( anymore|\.|,|—|-|$)",
    r"forget (about )?(the )?(?P<what>.+?)( anymore|\.|,|—|-|$)",
    r"skip (the )?(?P<what>.+?)( from now on| anymore|\.|,|—|-|$)",
    r"stop (telling|reminding|walking) me (about |through )?(the )?(?P<what>.+?)( anymore|\.|,|—|-|$)",
)
EXCLUSIVE_CUES = (r"\bnow just\b", r"\bjust\b", r"\bonly\b", r"\binstead\b", r"\bfrom here( on)?\b",
                  r"\bfrom now on\b", r"\bfor the rest\b", r"\bswitch (focus|to)\b", r"\bmoving into\b",
                  r"more than i need", r"\bthat'?s enough\b", r"\brather than\b", r"\bi'?d rather have\b",
                  r"\bdial it back\b", r"\bless useful\b", r"\bgot the hang of\b", r"\bcomfortable with\b",
                  r"\bcovered now\b", r"\bdifferent, so\b", r"\bnever mind\b", r"\bno need\b", r"\bdon'?t need\b")
ADD_CUES = (r"^\s*(oh,?\s*)?and\b", r"\balso\b", r"\bin addition\b", r"\bas well\b", r"\bcan you also\b", r"\bplus\b")


# "Let me know when I finish / reach / complete ..." is a milestone request on
# its own, but only a TRIGGER when it qualifies another criterion ("tell me what
# comes next as I finish each step").  So these count as MILESTONE only when no
# other type matched.
WEAK_MILESTONE_PATTERNS = (
    r"(when|whenever|once|as) i('ve| have)? (successfully )?(finish|complete|wrap|fill|manage|reach|hit)",
    r"finish(ed|ing)? (a|each|one of the|the|all the|filling|placing|applying|finding|looking) ",
    r"let me know (when|once|as|as soon as) (i|all|the|it|every|each)\b", r"tell me when (i|all|the|it)\b",
    r"heads-up (when|once|whenever)",
)


def _types_in(text: str) -> list[str]:
    low = text.lower()
    found = []
    for kind in TYPES:
        if any(_positive_pattern_match(low, p) for p in TYPE_PATTERNS[kind]):
            found.append(kind)
    if not found and any(re.search(p, low) for p in WEAK_MILESTONE_PATTERNS):
        found.append(MILESTONE)
    return found


def is_question(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    typed = _types_in(stripped)
    if MCQ in typed:
        return False  # MCQ prompts are standing criteria, not one-shot questions
    if stripped.endswith("?") or stripped.endswith("？"):
        # "Can you also let me know whenever I hit a milestone?" is a request,
        # not a question about the scene.
        return not (QUESTION_HELP_VERBS.search(stripped) and typed)
    # Commands such as "calculate ..." or "give me the answer" are
    # one-shot questions even without terminal punctuation.  Standing help
    # requests still win whenever a typed criterion was recognised.
    return not typed and bool(DIRECT_QUESTION.search(stripped))


@dataclass(frozen=True)
class Op:
    kind: str          # add | retire | question
    target: str        # criterion type, or the question text
    source: str = ""   # the instruction it came from (for the delta line)


def parse_instruction(text: str, *, first: bool = False) -> list[Op]:
    """Turn one user instruction into criterion-set operations.

    * first instruction: add every recognised type;
    * later "oh, and also ..." : add every recognised type;
    * later "don't need X anymore -- now just Y": retire the types named in X,
      add the types named in Y, and if Y is exclusive ("just", "only",
      "instead", "from here on") retire every other active type as well;
    * a scene question: push a pending question.
    Unrecognised text yields no ops.
    """
    raw = " ".join(text.split())
    if not raw:
        return []
    low = raw.lower()
    if is_question(raw):
        return [Op("question", raw, raw)]

    ops: list[Op] = []
    retire_types: list[str] = []
    retire_end = -1
    for cue in RETIRE_CUES:
        m = re.search(cue, low)
        if m:
            retire_types += _types_in(m.group("what"))
            retire_end = max(retire_end, m.end())
    # Split the instruction into a retire clause and an add clause.  A later
    # instruction that carries an exclusive cue ("now just", "instead", "from
    # here on", "what I'd rather have") is a rewrite: types named BEFORE the cue
    # are being given up, types named AFTER it are the new set.  Without such a
    # cue ("oh, and also ...") the whole text is additive.
    add_clause, pre_clause, exclusive = low, "", False
    if not first:
        cuts = [m.start() for c in EXCLUSIVE_CUES for m in [re.search(c, low)] if m]
        if cuts:
            cut = min(cuts)
            # skip the cue itself and any punctuation/dash right after it
            m = re.search(r"(—|--| - |:|;|,|\banymore\b[.,]?|\. |\bjust\b|\bonly\b|\binstead,?\b|\bis this:)", low[cut:])
            tail_start = cut + (m.end() if m else 0)
            pre_clause, add_clause, exclusive = low[:cut], low[tail_start:], True
            retire_types += [k for k in _types_in(pre_clause)]
    if retire_end >= 0 and not first:
        # an explicit "don't need X anymore" / "never mind X": everything that
        # names the new set comes after it
        add_clause, exclusive = low[retire_end:], True
    add_types = [k for k in _types_in(add_clause) if k not in retire_types]
    if exclusive and not add_types:
        # the cue split may have swallowed the verb; fall back to the whole tail
        tail = low[retire_end:] if retire_end >= 0 else low[min(cuts):]
        add_types = [k for k in _types_in(tail) if k not in retire_types]

    for kind in retire_types:
        if kind not in add_types:
            ops.append(Op("retire", kind, raw))
    if exclusive and add_types:
        ops.append(Op("retire_others", ",".join(add_types), raw))
    for kind in add_types:
        ops.append(Op("add", kind, raw))
    if not ops and first:
        # A first instruction the rules cannot classify: keep it as a standing
        # request without a typed criterion; the raw text is in the prompt.
        return []
    return ops


@dataclass
class CriterionSet:
    """Active help criteria plus the pending scene question."""

    active: dict[str, float] = field(default_factory=dict)      # type -> since (s)
    sources: dict[str, str] = field(default_factory=dict)       # active type -> request clause
    retired: dict[str, float] = field(default_factory=dict)     # type -> at (s)
    question: str | None = None
    question_since: float | None = None
    log: list[dict] = field(default_factory=list)               # append-only delta log

    def apply(self, ops: list[Op], second: float) -> list[str]:
        """Apply ops; return the human-readable delta fragments."""
        frags: list[str] = []
        for op in ops:
            if op.kind == "add":
                if op.target not in self.active:
                    self.active[op.target] = second
                    self.retired.pop(op.target, None)
                    frags.append(f"+{op.target}")
                self.sources[op.target] = op.source
            elif op.kind == "retire":
                if op.target in self.active:
                    del self.active[op.target]
                    self.sources.pop(op.target, None)
                    self.retired[op.target] = second
                    frags.append(f"-{op.target}")
            elif op.kind == "retire_others":
                keep = set(op.target.split(","))
                for kind in list(self.active):
                    if kind not in keep:
                        del self.active[kind]
                        self.sources.pop(kind, None)
                        self.retired[kind] = second
                        frags.append(f"-{kind}")
            elif op.kind == "question":
                self.question, self.question_since = op.target, second
                frags.append(f"question: {op.target}")
        if frags:
            self.log.append({"second": second, "delta": frags})
        return frags

    def answer_question(self) -> None:
        self.question, self.question_since = None, None

    def focus_text(self) -> str:
        """All active request clauses, deduplicated in criterion order."""
        values = []
        for kind in TYPES:
            value = self.sources.get(kind, "")
            if kind in self.active and value and value not in values:
                values.append(value)
        return " | ".join(values)

    def render_compact(self) -> str:
        """Short form: active types, retired types in parentheses, pending question."""
        active = ", ".join(k for k in TYPES if k in self.active) or "none"
        retired = ", ".join(f"{k} retired {at:.0f}s" for k, at in sorted(self.retired.items(), key=lambda kv: kv[1]))
        q = f" | question: {self.question}" if self.question else ""
        return f"P.request criteria: {active}" + (f" ({retired})" if retired else "") + q

    def render(self) -> str:
        """The P.request criteria line for the card (epoch-constant)."""
        parts = []
        for kind in TYPES:
            if kind in self.active:
                parts.append(f"{kind} (since {self.active[kind]:.0f}s)")
        for kind, at in sorted(self.retired.items(), key=lambda kv: kv[1]):
            parts.append(f"{kind} retired at {at:.0f}s")
        crit = "; ".join(parts) if parts else "none"
        q = f"{self.question} (asked {self.question_since:.0f}s)" if self.question else "none"
        return f"P.request criteria: {crit} | pending question: {q} | style: none"

    def dump(self) -> dict:
        return {"active": dict(self.active), "sources": dict(self.sources),
                "retired": dict(self.retired),
                "question": self.question, "log": list(self.log)}
