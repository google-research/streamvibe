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

"""Shared inference helpers for StreamVibe Harness."""
from __future__ import annotations
import json
import re
from pathlib import Path
from run_inference import turn_table, task_probes


TRAINING_NATIVE_SYSTEM_PROMPT = (
    "You are a real-time video streaming assistant observing a continuous camera feed "
    "frame by frame. The last frame represents the current moment.\n"
    "## Action Format\n"
    "At every inference step you MUST choose exactly one of the following two actions:\n"
    "**Stay silent** — output ONLY:\n"
    "</silence>\n"
    "Choose this when nothing noteworthy has changed in the scene, no user query is "
    "pending, or there is nothing useful to say.\n"
    "**Speak** — output the token followed by a concise reply:\n"
    "</response> Your reply here.\n"
    "Choose this when you observe something worth reporting or a significant state "
    "change, or when you can answer a user question based on available evidence."
)

STREAMVIBE_TRAINING_SYSTEM_PROMPT = (
    "You are a proactive real-time video assistant observing a continuous stream.\n"
    "At every timestamp choose exactly one action using only frames visible up to that timestamp.\n"
    "Output </silence> when evidence is irrelevant or insufficient.\n"
    "Output </response> followed by a concise, grounded answer only when the evidence is sufficient.\n"
    "Never use future information or make unsupported guesses."
)

def anchor_probes(row: dict, task: str) -> list[dict]:
    """Return the frozen suite probes, enriched with the canonical gold turn."""
    golds = turn_table(row)
    raw = row.get("anchor_probes")
    if raw is None:
        return task_probes(row, task)
    probes = []
    for item in raw:
        turn = int(item["turn"])
        if not 0 <= turn < len(golds):
            raise ValueError(f"{row.get('eval_id')}: invalid anchor turn {turn}")
        probe = dict(golds[turn])
        probe["probe_label"] = str(item.get("probe_label") or "anchor")
        probes.append(probe)
    return sorted(probes, key=lambda p: p["turn"])

def generate_guarded(llm, reqs, sp):
    """Generate a batch while isolating requests that exceed context limits.

    If batch validation fails, retry each request independently. Return None
    for any request that cannot be admitted, preserving the remaining lanes.
    """
    try:
        return llm.generate(prompts=reqs, sampling_params=sp, use_tqdm=False)
    except ValueError:
        outs = []
        for req in reqs:
            try:
                outs.append(llm.generate(prompts=[req], sampling_params=sp, use_tqdm=False)[0])
            except ValueError as exc:
                print(f"[stream-eval] OVER-LENGTH: lane stopped -- {exc}", flush=True)
                outs.append(None)
        return outs
