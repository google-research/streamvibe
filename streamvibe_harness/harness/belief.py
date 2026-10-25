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

"""Bounded structured index over evidence produced by the three-tier memory.

The summariser remains the visual producer.  This module gives its text stable
fact IDs, evidence intervals and deterministic retrieval so B can be managed as
state instead of repeatedly dumping an opaque prose block into the prompt.
"""
from __future__ import annotations

import hashlib
import re
from collections import OrderedDict
from dataclasses import asdict, dataclass

CONTROL_PREFIXES = ("## ", "PLAN ", "WATCH ", "CUE ", "GOAL:", "COMPLETED:", "CURRENT:",
                    "NEXT:", "DEVIATION:", "RECOVERY:", "REQUEST STATE:",
                    "VISUAL MEMORY FOCUS:",
                    "EVENTS:", "CHUNK END")
STOP = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "of",
        "for", "and", "or", "what", "which", "where", "when", "how", "this", "that",
        "it", "now", "current", "user"}
CJK_STOP = ("它", "这个", "那个", "刚才", "现在", "哪里", "哪儿", "什么", "怎么", "多少",
            "被", "把", "放在", "放到", "的", "了", "吗", "呢")


def terms(text: str) -> set[str]:
    value = text or ""
    out = {w for w in re.findall(r"[a-z0-9]+", value.lower())
           if len(w) > 2 and w not in STOP}
    out |= {w[:-1] for w in out if w.endswith("s")}
    for span in re.findall(r"[\u4e00-\u9fff]+", value):
        for stop in CJK_STOP:
            span = span.replace(stop, " ")
        for part in span.split():
            out.update(part)
            out.update(part[i:i + 2] for i in range(max(0, len(part) - 1)))
    return out


def normalize(text: str) -> str:
    value = re.sub(r"^[-•\s]+", "", text or "")
    value = re.sub(r"^\[[^]]+\]\s*", "", value)
    return " ".join(value.lower().split()).strip(" .;|")


def fact_lines(text: str, *, exclude_event_section: bool = False) -> list[str]:
    out = []
    inside_events = False
    for raw in (text or "").splitlines():
        line = raw.strip().lstrip("-• ").strip()
        upper = line.upper()
        if upper == "EVENTS:":
            inside_events = True
            continue
        if inside_events and upper.startswith((
                "CHUNK END", "PLAN ", "WATCH ", "CUE ", "DEVIATION:",
                "RECOVERY:")):
            inside_events = False
        if exclude_event_section and inside_events:
            continue
        if not line or line.startswith(CONTROL_PREFIXES):
            continue
        if normalize(line):
            out.append(line)
    return out


def event_lines(text: str, *, strict_typed: bool = False) -> list[str]:
    """Extract the ordered EVENTS section without treating request prose as fact."""
    out, inside = [], False
    for raw in (text or "").splitlines():
        line = raw.strip()
        if line.upper() == "EVENTS:":
            inside = True
            continue
        if inside and line.upper().startswith((
                "CHUNK END", "PLAN ", "WATCH ", "CUE ", "DEVIATION:",
                "RECOVERY:")):
            inside = False
        if inside and line:
            body = line.lstrip("-• ").strip()
            if strict_typed and not (
                    re.match(r"(?i)^EVENT\s*\[[^]]+\]\s*:", body)
                    and all(re.search(rf"(?i)\b{label}\s*=", body)
                            for label in ("actor", "action", "object", "from",
                                          "to", "relation", "confidence"))):
                continue
            if body and normalize(body) not in {"none", "no request-relevant event"}:
                out.append(body)
    return out


def _event_interval(text: str, fallback_t0: float,
                    fallback_t1: float) -> tuple[float, float]:
    match = re.search(
        r"\[\s*([0-9]+(?:\.[0-9]+)?)s?\s*(?:-|–|to)\s*"
        r"([0-9]+(?:\.[0-9]+)?)s?\s*\]", text, re.I)
    if match:
        return float(match.group(1)), float(match.group(2))
    single = re.search(r"\[\s*([0-9]+(?:\.[0-9]+)?)s?\s*\]", text, re.I)
    if single:
        value = float(single.group(1))
        return value, value
    return float(fallback_t0), float(fallback_t1)


def _event_fields(text: str) -> dict[str, str]:
    labels = "actor|action|object|from|to|relation|confidence"
    matches = list(re.finditer(rf"(?i)\b({labels})\s*=\s*", text or ""))
    fields = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        value = text[match.end():end].strip(" ;|,.")
        fields[match.group(1).lower()] = " ".join(value.split())[:160]
    return fields


@dataclass
class FactRecord:
    fact_id: str
    text: str
    t0: float
    t1: float
    source: str
    chunk: int
    entities: tuple[str, ...]
    observations: int = 1

    def render(self) -> str:
        return f"[{self.t0:.0f}s-{self.t1:.0f}s] {self.text}"


@dataclass
class OrderedEventRecord:
    event_id: str
    sequence: int
    text: str
    t0: float
    t1: float
    source: str
    chunk: int
    previous_event_id: str | None
    actor: str = ""
    action: str = ""
    object: str = ""
    from_state: str = ""
    to_state: str = ""
    relation: str = ""
    confidence: str = ""

    def render(self) -> str:
        relation = f"; after={self.previous_event_id}" if self.previous_event_id else ""
        return (f"E{self.sequence:04d} [{self.t0:.0f}s-{self.t1:.0f}s] "
                f"{self.text}{relation}")


class BeliefIndex:
    def __init__(self, max_facts: int = 256):
        self.max_facts = max(1, int(max_facts))
        self._facts: OrderedDict[str, FactRecord] = OrderedDict()
        self._events: OrderedDict[str, OrderedEventRecord] = OrderedDict()
        self._event_sequence = 0

    def update(self, text: str, t0: float, t1: float, *, source: str, chunk: int,
               strict_ordered_events: bool = False) -> list[FactRecord]:
        changed = []
        for line in fact_lines(text, exclude_event_section=strict_ordered_events):
            key = normalize(line)
            fact_id = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
            if fact_id in self._facts:
                fact = self._facts.pop(fact_id)
                fact.t0 = min(fact.t0, float(t0))
                fact.t1 = max(fact.t1, float(t1))
                fact.observations += 1
                fact.source, fact.chunk = source, int(chunk)
            else:
                fact = FactRecord(fact_id, line, float(t0), float(t1), source, int(chunk),
                                  tuple(sorted(terms(line))))
            self._facts[fact_id] = fact
            changed.append(fact)
        while len(self._facts) > self.max_facts:
            self._facts.popitem(last=False)
        previous = next(reversed(self._events), None) if self._events else None
        for order, line in enumerate(
                event_lines(text, strict_typed=strict_ordered_events)):
            et0, et1 = _event_interval(line, t0, t1)
            key_text = f"{source}:{chunk}:{order}:{et0:.3f}:{et1:.3f}:{normalize(line)}"
            event_id = hashlib.sha1(key_text.encode("utf-8")).hexdigest()[:12]
            if event_id in self._events:
                previous = event_id
                continue
            fields = _event_fields(line)
            self._event_sequence += 1
            event = OrderedEventRecord(
                event_id, self._event_sequence, line, et0, et1, source,
                int(chunk), previous,
                actor=fields.get("actor", ""), action=fields.get("action", ""),
                object=fields.get("object", ""),
                from_state=fields.get("from", ""), to_state=fields.get("to", ""),
                relation=fields.get("relation", ""),
                confidence=fields.get("confidence", ""))
            self._events[event_id] = event
            previous = event_id
        while len(self._events) > self.max_facts:
            self._events.popitem(last=False)
        return changed

    def drop_through(self, second: float) -> int:
        doomed = [key for key, fact in self._facts.items() if fact.t1 <= float(second)]
        for key in doomed:
            del self._facts[key]
        event_doomed = [key for key, event in self._events.items()
                        if event.t1 <= float(second)]
        for key in event_doomed:
            del self._events[key]
        return len(doomed) + len(event_doomed)

    def retrieve(self, focus: str, k: int = 3) -> list[FactRecord]:
        want = terms(focus)
        ranked = []
        for order, fact in enumerate(self._facts.values()):
            overlap = len(want & set(fact.entities))
            ranked.append((overlap, fact.t1, order, fact))
        matched = [row for row in ranked if row[0] > 0]
        pool = matched or ranked
        pool.sort(key=lambda row: (-row[0], -row[1], -row[2]))
        return [row[3] for row in pool[:max(0, int(k))]]

    def render(self, focus: str = "", k: int | None = None) -> str:
        facts = (self.retrieve(focus, k) if k is not None
                 else list(self._facts.values()))
        parts = [fact.render() for fact in facts]
        events = list(self._events.values())
        if focus:
            want = terms(focus)
            matched = [event for event in events if want & terms(event.text)]
            events = matched or events
        if k is not None:
            events = events[-max(0, int(k)):]
        if events:
            parts.append("## Ordered observed transitions")
            parts.extend(event.render() for event in events)
        return "\n".join(parts)

    def dump(self) -> list[dict]:
        return [asdict(fact) for fact in self._facts.values()]

    def timeline_dump(self) -> list[dict]:
        return [asdict(event) for event in self._events.values()]

    @property
    def event_count(self) -> int:
        return len(self._events)

    @property
    def event_version(self) -> int:
        """Monotonic observed-event version, independent of bounded eviction."""
        return self._event_sequence

    def __len__(self) -> int:
        return len(self._facts)
