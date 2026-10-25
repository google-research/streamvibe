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

"""Causal procedure-to-state bridge for guidance semantics."""
from __future__ import annotations

import hashlib
import re

from ..state_runtime import CanonicalEvent


def _observed(line: str) -> bool:
    value = " ".join((line or "").lower().split())
    return bool(value and "unknown" not in value
                and "no visible step evidence" not in value)


def sync_guidance_procedure(adapter, procedure_line: str, epoch: int) -> None:
    if (adapter.semantic_profile not in {"physical_guidance", "digital_guidance"}
            or not _observed(procedure_line)):
        return
    clean = re.sub(r"\s*\[evidence[^]]*]\s*$", "", procedure_line.strip())
    body = clean.split(":", 1)[1].strip() if ":" in clean else clean
    payload = {}
    for part in body.split(" | "):
        if ":" not in part:
            continue
        label, value = part.split(":", 1)
        label, value = label.strip().lower(), value.strip()
        if label == "done" and value:
            payload["last_completed"] = value.split(";")[-1].strip()[:240]
        elif label == "current" and value:
            payload["observed_current"] = value[:240]
        elif label == "deviation" and value:
            payload["deviation"] = value[:240]
    if not payload:
        return
    digest = hashlib.sha1(repr(sorted(payload.items())).encode("utf-8")).hexdigest()
    if digest == getattr(adapter, "_guidance_procedure_digest", ""):
        return
    adapter._guidance_procedure_digest = digest
    stamp = re.search(r"\[evidence\s+[0-9.]+-([0-9.]+)s]\s*$", procedure_line)
    second = float(stamp.group(1)) if stamp else float(epoch)
    adapter.state_runtime.append(CanonicalEvent(
        f"procedure_header_{digest[:12]}", "PROCEDURE_OBSERVED",
        second, "VISUAL", payload, f"procedure_header_{digest[:12]}"))
    adapter.state_runtime.consume_deltas(folded=True)
