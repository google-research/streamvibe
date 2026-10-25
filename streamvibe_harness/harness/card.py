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

"""The policy-facing harness card (StreamVibe Harness).

The card is the part of the system context that renders harness state
``H_t = (B_t, P_t, E_t)`` for the base model.  Two rules make it safe under
Prefix-cache reuse:

* the card is snapshotted once per cache epoch (== one memory chunk) and is
  byte-identical for every turn inside that epoch;
* anything that changes inside the epoch is appended to the turn it happened
  on as a delta (``turn_extras``), never written back into the card.

``EpochGuard`` makes the first rule observable: it counts every time the
rendered header changes while the chunk did not, which is exactly the event
that would invalidate the whole cached prefix.  The counter must stay at zero.

The shell mode renders empty slots for ablation.
"""
from __future__ import annotations

from dataclasses import dataclass

# A mode is "none", "shell", or a comma-separated set of filled slots drawn
# from SLOTS (e.g. "belief", "belief,request").  "shell" == every slot empty.
SLOTS = ("belief", "request", "said", "experience", "procedure", "lifecycle", "cues",
         "anchored", "compact", "symbolic", "silence_prior", "adaptive")
CARD_MODES = ("none", "shell", "belief", "request", "belief,request",
              "belief,request,experience", "belief,request,said",
              "belief,request,experience,said", "belief,request,experience,procedure",
              "belief,request,experience,said,procedure")


def parse_mode(mode: str) -> set[str] | None:
    """None when disabled, otherwise the set of filled slots ("shell" -> set())."""
    if mode == "none":
        return None
    if mode == "shell":
        return set()
    slots = {m.strip() for m in mode.split(",") if m.strip()}
    unknown = slots - set(SLOTS)
    if unknown:
        raise ValueError(f"unknown card slots {sorted(unknown)}")
    return slots

CARD_TAG = "[HARNESS STATE]"

# Slot labels are fixed strings so that later stages only change the values.
SHELL_LINES = (
    "B stabilized facts (may lag; the current frames take precedence): none",
    "P.request criteria: none | pending question: none | style: none",
    "P.request said: none",
    "P.procedure observed progress: unknown",
    "E experience: none",
)


BELIEF_LABEL = "B stabilized facts (may lag; the current frames take precedence):"
REQUEST_LABEL = "P.request criteria:"
SAID_LABEL = "P.request said:"
PROCEDURE_LABEL = "P.procedure observed progress:"
EXPERIENCE_LABEL = "E experience:"
LIFECYCLE_LABEL = "P.lifecycle:"
SILENCE_PRIOR = ("Default action: </silence>. Speak only when the CURRENT frames satisfy an active "
                 "criterion with something new; the state below is context, not a reason to speak.")


def render_card(mode: str, epoch: int, belief: str = "", request_line: str = "",
                said_line: str = "", experience: str = "", procedure_line: str = "",
                lifecycle_line: str = "", reference_line: str = "",
                cue_evidence: str = "") -> str:
    """Return the card text for one cache epoch, or "" when disabled.

    ``belief`` is the memory text of the chunk that just closed; with the
    ``belief`` slot filled it replaces the B line (and the caller drops the
    separate [VIDEO MEMORY] block).  ``request_line`` is CriterionSet.render();
    with the ``request`` slot filled it replaces the P.request criteria line.
    Slots that are not filled keep their empty shell line, so the shell stays
    content-free and every later stage changes exactly one line group.
    """
    slots = parse_mode(mode)
    if slots is None:
        return ""
    compact = "compact" in slots
    lines = [f"{CARD_TAG} epoch {int(epoch)}"]
    if "silence_prior" in slots:
        lines.append(SILENCE_PRIOR)
    for line in SHELL_LINES:
        filled = None
        if "belief" in slots and line.startswith(BELIEF_LABEL):
            body = belief.strip()
            filled = f"{BELIEF_LABEL}\n{body}" if body else None
        elif "request" in slots and line.startswith(REQUEST_LABEL):
            filled = request_line.strip() or None
        elif "said" in slots and line.startswith(SAID_LABEL):
            filled = said_line.strip() or None
        elif "procedure" in slots and line.startswith(PROCEDURE_LABEL):
            filled = procedure_line.strip() or None
        elif "experience" in slots and line.startswith(EXPERIENCE_LABEL):
            body = experience.strip()
            filled = f"{EXPERIENCE_LABEL}\n{body}" if body else None
        if filled is not None:
            lines.append(filled)
        elif not compact:
            lines.append(line)   # empty shell line (D0-style); dropped in compact mode
    if "lifecycle" in slots:
        body = lifecycle_line.strip()
        if body:
            lines.append(body)
        elif not compact:
            lines.append("P.lifecycle: none")
    if "cues" in slots:
        if reference_line.strip():
            lines.append(reference_line.strip())
        if cue_evidence.strip():
            lines.append(cue_evidence.strip())
    return "\n".join(lines)


@dataclass
class EpochGuard:
    """Counts header changes that are NOT aligned with a chunk boundary."""

    chunk: int = -1
    header: str = ""
    epochs: int = 0
    resets: int = 0

    def observe(self, chunk: int, header: str) -> None:
        if chunk != self.chunk:
            self.chunk = chunk
            self.header = header
            self.epochs += 1
            return
        if header != self.header:
            self.resets += 1
            self.header = header

    def dump(self) -> dict:
        return {"epochs": self.epochs, "epoch_resets": self.resets}
