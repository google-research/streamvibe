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

"""Conservative recovery of novel responses rejected by the visual verifier.

The verifier is intentionally strict about repeated narration. On held-out
pilots that improves precision strongly but can suppress a genuinely new event.
This gate restores a rejected proposal only after a minimum quiet interval and
only when its lexical content is dissimilar to the recent committed replies.
It uses no gold labels and only information available online.
"""
from __future__ import annotations

import re
from collections import deque


def token_jaccard(left: str, right: str) -> float:
    a = set(re.findall(r"[a-z0-9]+", (left or "").lower()))
    b = set(re.findall(r"[a-z0-9]+", (right or "").lower()))
    return len(a & b) / len(a | b) if a and b else 0.0


class NovelResponseGate:
    def __init__(self, similarity_threshold: float = 0.6,
                 min_gap_turns: int = 8, history: int = 8):
        if not 0 <= similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be in [0, 1]")
        if min_gap_turns < 0 or history < 1:
            raise ValueError("min_gap_turns must be >= 0 and history >= 1")
        self.threshold = similarity_threshold
        self.min_gap = min_gap_turns
        self.committed: deque[tuple[int, str]] = deque(maxlen=history)

    def record(self, turn: int, text: str) -> None:
        if text:
            self.committed.append((int(turn), text))

    def should_rescue(self, turn: int, candidate: str) -> bool:
        if not candidate:
            return False
        if not self.committed:
            return True
        if int(turn) - self.committed[-1][0] < self.min_gap:
            return False
        similarity = max(token_jaccard(candidate, text)
                         for _, text in self.committed)
        return similarity < self.threshold
