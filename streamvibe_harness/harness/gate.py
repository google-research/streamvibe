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

"""Optional response-cadence gate for ablation configurations.

Checks warmup time, the gap since the last response, and lexical duplication.
Pending scene questions bypass the gap check. The released predecision-only
launcher disables this gate; the interaction model chooses its action.
"""
from __future__ import annotations

import json
import re
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

CADENCE_PATH = Path(__file__).resolve().parent.parent / "policies" / "cadence.json"
DEFAULT = {"warmup_s": 0, "min_gap_s": 0, "dedup_jaccard": 1.1}


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    return len(ta & tb) / max(1, len(ta | tb))


def load_cadence(path: Path | None = None) -> dict:
    path = path or CADENCE_PATH
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class CadenceGate:
    """Per-lane commit gate. `decide` returns (commit, reason)."""

    table: dict = field(default_factory=load_cadence)
    history: int = 8
    enabled: bool = True
    committed: deque = field(default_factory=lambda: deque(maxlen=8))
    counts: dict = field(default_factory=lambda: {"suppressed_warmup": 0,
                                                  "suppressed_gap": 0,
                                                  "suppressed_dedup": 0,
                                                  "committed": 0})

    def __post_init__(self):
        self.committed = deque(maxlen=max(1, self.history))

    def params(self, active_types) -> dict:
        """Most restrictive parameters over the active help types."""
        rows = [self.table[k] for k in active_types if k in self.table]
        if not rows:
            return dict(DEFAULT)
        return {"warmup_s": max(r.get("warmup_s", 0) for r in rows),
                "min_gap_s": max(r.get("min_gap_s", 0) for r in rows),
                # the strictest dedup is the SMALLEST threshold
                "dedup_jaccard": min(r.get("dedup_jaccard", 1.1) for r in rows)}

    def decide(self, second: float, text: str, active_types,
               question_pending: bool = False) -> tuple[bool, str]:
        if not self.enabled:
            return True, "off"
        p = self.params(active_types)
        if second < p["warmup_s"] and not question_pending:
            self.counts["suppressed_warmup"] += 1
            return False, f"warmup<{p['warmup_s']}s"
        if self.committed and not question_pending:
            if second - self.committed[-1][0] < p["min_gap_s"]:
                self.counts["suppressed_gap"] += 1
                return False, f"gap<{p['min_gap_s']}s"
        thr = p["dedup_jaccard"]
        if thr <= 1.0 and any(jaccard(text, t) >= thr for _s, t in self.committed):
            self.counts["suppressed_dedup"] += 1
            return False, f"dedup>={thr}"
        return True, "commit"

    def record(self, second: float, text: str) -> None:
        self.committed.append((float(second), text or ""))
        self.counts["committed"] += 1

    def dump(self) -> dict:
        return {"enabled": self.enabled, **self.counts}
