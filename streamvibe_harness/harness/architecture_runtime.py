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

"""Runtime bridge for validated, Genome-declared predecision modules.

Only an epoch Header hook is exposed initially.  It is intentionally narrow:
an evolved module may append deterministic evidence/state to the Header before
the interaction model sees it, but cannot remove existing context, alter frames, inspect model
output, or choose an action.  More hook types can be added only with equally
strong causal contracts.
"""
from __future__ import annotations

import importlib
import json
import os
from functools import lru_cache
from pathlib import Path


GENOME_ENV = "STREAMVIBE_HARNESS_GENOME"


@lru_cache(maxsize=8)
def _hook_specs(genome_path: str) -> tuple[tuple[str, str, dict], ...]:
    path = Path(genome_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "streamvibe-harness-genome":
        raise ValueError("runtime received an unsupported Harness Genome")
    contracts = value.get("contracts") or {}
    required = {
        "decision_owner": "StreamVibe-VL",
        "mutation_boundary": "between_episodes",
        "postdecision_filter": False,
        "output_rewrite": False,
        "future_or_gold_access": False,
        "epoch_header_mutation": False,
    }
    if any(contracts.get(key) != expected for key, expected in required.items()):
        raise ValueError("runtime Harness Genome violates serving contracts")
    hooks = []
    for node in value.get("nodes") or []:
        config = node.get("config") or {}
        if not config.get("enabled", True):
            continue
        if config.get("runtime_hook") != "epoch_header":
            continue
        if config.get("epoch_stable") is not True:
            raise ValueError("epoch_header hooks must declare epoch_stable=true")
        implementation = str(node.get("implementation") or "")
        if ":" not in implementation:
            raise ValueError("Genome hook implementation must use module:symbol")
        hooks.append((str(node.get("node_id") or ""), implementation, config))
    hooks.sort(key=lambda item: (int(item[2].get("priority", 100)), item[0]))
    return tuple(hooks)


@lru_cache(maxsize=128)
def _transform_cached(genome_path: str, header: str) -> str:
    current = header
    for node_id, implementation, config in _hook_specs(genome_path):
        module_name, symbol = implementation.split(":", 1)
        if not module_name.startswith("harness."):
            raise ValueError("evolved hooks must live inside the Harness package")
        module = importlib.import_module(module_name)
        hook = getattr(module, symbol)
        updated = hook(current, dict(config))
        if not isinstance(updated, str):
            raise TypeError(f"Genome hook {node_id} did not return text")
        if not updated.startswith(current):
            raise ValueError(
                f"Genome hook {node_id} mutated or removed the frozen Header")
        if len(updated) - len(current) > int(config.get("max_append_chars", 8000)):
            raise ValueError(f"Genome hook {node_id} exceeded its append budget")
        current = updated
    return current


def transform_epoch_header(header: str) -> str:
    """Apply the active candidate's deterministic pre-decision Header modules."""
    genome_path = os.environ.get(GENOME_ENV, "").strip()
    if not genome_path or not header:
        return header
    return _transform_cached(str(Path(genome_path).resolve()), header)
