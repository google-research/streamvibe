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

"""Causal evidence alignment and stream timing hook for epoch headers."""
from __future__ import annotations


def align_evidence_header(header: str, config: dict) -> str:
    """Append balanced causal evidence-grounding guidance to the epoch header."""
    if not header:
        return header

    guidance = (
        "\n\n[STREAM EVIDENCE & RESPONSE TIMING]\n"
        "- Base all responses directly on confirmed visual evidence from the stream.\n"
        "- When a question or request is pending, respond promptly when the relevant visual evidence or state transition occurs.\n"
        "- For proactive monitoring and guidance, emit timely updates upon new physical or procedural transitions without waiting for explicit prompts.\n"
        "- Maintain silence only when no new visual transition has occurred and prior observations remain unchanged.\n"
        "- Do not guess or extrapolate beyond observed visual facts."
    )
    return header + guidance
