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

"""Track active user requests across streaming frames and memory boundaries.

A new request becomes active when it arrives. Responses enter the interaction
history. Standing requests remain active; one-shot requests close after an answer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

TIME_RE = re.compile(r"<([0-9.]+) seconds>")


def extract_query(user_raw: str) -> str:
    """The user's words on a turn, with the frame and timestamp markup removed."""
    text = TIME_RE.sub("", (user_raw or "").replace("<image>", ""))
    return " ".join(text.split())


@dataclass
class ActiveQuery:
    turn: int
    second: float
    text: str


class QueryTracker:
    """Which query, if any, the model is currently expected to answer."""

    def __init__(self, records: list[dict], seconds: list[float], persist: bool = True,
                 mode: str | None = None):
        self.persist = persist
        self.mode = mode or ("standing" if persist else "oneshot")
        if self.mode not in {"standing", "oneshot", "auto"}:
            raise ValueError(f"invalid query mode: {self.mode}")
        self.queries: dict[int, ActiveQuery] = {}
        for turn, record in enumerate(records):
            text = extract_query(record.get("user_raw", ""))
            if text:
                self.queries[turn] = ActiveQuery(turn, seconds[turn], text)
        self.active: ActiveQuery | None = None
        self.answered: list[tuple[ActiveQuery, float, str]] = []

    def on_turn(self, turn: int) -> None:
        """Stage 1: a query arriving on this turn becomes the active one.

        A new query supersedes an unanswered older one rather than queueing.
        Queueing would let the model answer a question the user has already
        moved on from, and nothing in the data ever asks two at once.
        """
        q = self.queries.get(turn)
        if q is not None:
            self.active = q

    def on_action(self, turn, second, action, text):
        """Record a response and update the active query.
        
        Standing requests remain active. One-shot requests close after a response;
        auto mode uses the request text to select that behavior.
        """
        if action != "response":
            return None
        answered = self.active
        if self.mode == "oneshot" or (
                self.mode == "auto" and answered is not None
                and not _is_standing_request(answered.text)):
            self.active = None
        self.answered.append((answered, second, text))
        return answered

    def carry(self) -> ActiveQuery | None:
        """The query that must be restated in the next chunk's header."""
        return self.active


STANDING_PATTERNS = (
    "walk me", "guide me", "help me", "keep me", "monitor", "watch for",
    "let me know if", "tell me when", "remind me when", "remind me whenever",
    "warn me", "catch any", "step by step",
    "commentate", "narrate", "when the visual evidence is sufficient",
    "as soon as you have seen enough", "when you have seen enough",
    "wait until", "what happened next", "about to do",
)


def _is_standing_request(text: str) -> bool:
    """Conservative request-lifecycle skill for continuous proactive help."""
    value = " ".join((text or "").lower().split())
    return any(pattern in value for pattern in STANDING_PATTERNS)
