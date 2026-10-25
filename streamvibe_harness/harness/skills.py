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

"""Minimal skills that are specific to proactive video assistance.

These are deterministic evidence operations, not extra agents. The base model
retains the speak/silence decision. Skills only make relevant past evidence and
task state easier to use when a request arrives.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

STOP = {"the", "a", "an", "is", "are", "was", "were", "to", "of", "in",
        "on", "at", "and", "or", "what", "who", "when", "where", "how",
        "did", "does", "do", "this", "that", "with", "for", "it", "i"}
ERROR_WORDS = {"wrong", "error", "mistake", "miss", "missing", "incorrect",
               "forgot", "fail", "failed", "not", "without", "instead"}


def tokens(text: str) -> set[str]:
    return {v for v in re.findall(r"[a-z0-9]+", (text or "").lower())
            if v not in STOP and len(v) > 1}


@dataclass(frozen=True)
class EvidenceItem:
    t0: float
    t1: float
    text: str
    tier: str


class ActiveHelpSkills:
    """Evidence retrieval, progress state and mistake-evidence prioritisation."""

    def __init__(self, config: dict | None = None):
        cfg = config or {}


        self.enabled = bool(cfg.get("visible_retrieval",
                                    cfg.get("enabled", False)))
        self.top_k = max(1, min(4, int(cfg.get("retrieval_top_k", 2))))
        self.calls = 0
        self.hits = 0

    def retrieve(self, query: str, items: list[EvidenceItem]) -> str:
        if not self.enabled or not query or not items:
            return ""
        self.calls += 1
        qtokens = tokens(query)
        wants_error = bool(qtokens & ERROR_WORDS)
        ranked = []
        for index, item in enumerate(items):
            itokens = tokens(item.text)
            lexical = len(qtokens & itokens) / max(1, len(qtokens))
            error_boost = 0.25 if wants_error and itokens & ERROR_WORDS else 0.0
            recency = 0.05 * (index + 1) / len(items)
            ranked.append((lexical + error_boost + recency, index, item))
        ranked.sort(reverse=True)
        selected = [item for score, _, item in ranked[:self.top_k]
                    if score > 0.05]
        if not selected:
            selected = [items[-1]]
        self.hits += int(bool(selected))
        lines = ["[ACTIVE-HELP SKILL: RETRIEVED PAST EVIDENCE]",
                 "Observations only; the current frame and user request remain authoritative."]
        for item in sorted(selected, key=lambda value: value.t0):
            lines.append(f"[{item.t0:.0f}s-{item.t1:.0f}s; {item.tier}] {item.text}")
        return "\n".join(lines)

    def stats(self) -> dict:
        return {"enabled": self.enabled, "retrieval_top_k": self.top_k,
                "calls": self.calls, "hits": self.hits}
