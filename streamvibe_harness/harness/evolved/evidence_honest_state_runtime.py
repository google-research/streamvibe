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

"""Evidence-honest projection of compact Skill execution state."""
from __future__ import annotations

from typing import Any

from ..state_runtime import StateRuntime as BaseStateRuntime


class StateRuntime(BaseStateRuntime):
    """Expose only Skill instances selected by grounded recall."""

    def render_skill_projection(self, skills: dict[str, Any],
                                max_visible: int = 2) -> str:
        priority = {"matched": 0, "armed": 1, "delivered": 2, "retired": 3}
        instances = sorted(
            (item for item in self.state.skill_instances.values()
             if (item.get("status") in {"matched", "armed"}
                 and item.get("skill") in skills)),
            key=lambda item: (priority.get(item.get("status"), 9),
                              item["criterion_id"]),
        )[:max(0, int(max_visible))]
        lines = []
        for item in instances:
            skill = skills[item["skill"]]
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
