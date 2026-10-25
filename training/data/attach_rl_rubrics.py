#!/usr/bin/env python3
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

"""Attach generated binary rubrics to short-window rows; omit rows missing a rubric."""
import json, os, sys

SRC = os.environ.get("SRC", "data/StreamVibe-RL/windows.jsonl")
RUB = os.environ.get("RUB", "data/StreamVibe-RL/rubrics.jsonl")
OUT = os.environ.get("OUT", "data/StreamVibe-RL/StreamVibe-RL-train.jsonl")

rub = {}
with open(RUB) as f:
    for line in f:
        if not line.strip():
            continue
        d = json.loads(line)
        cps = d.get("checkpoints") or []
        if cps:
            rub[d["id"]] = cps
print(f"rubric 条数: {len(rub)}", flush=True)

n_in = n_out = n_skip = 0
ncp = 0
with open(SRC) as f, open(OUT, "w") as o:
    for line in f:
        if not line.strip():
            continue
        n_in += 1
        row = json.loads(line)
        a = json.loads(row["answer"]) if isinstance(row["answer"], str) else row["answer"]
        cps = rub.get(a["id"])
        if not cps:
            n_skip += 1
            continue
        a["rubric"] = cps
        row["answer"] = json.dumps(a, ensure_ascii=False)
        o.write(json.dumps(row, ensure_ascii=False) + "\n")
        n_out += 1
        ncp += len(cps)
print(f"输入 {n_in} -> 输出 {n_out} (跳过无 rubric {n_skip}), 平均 {ncp/max(n_out,1):.1f} 检查项")
