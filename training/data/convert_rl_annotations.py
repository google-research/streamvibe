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

"""StreamVibe-RL source annotations -> the intermediate row format build_gt_centered.py eats.

Every annotation source shares one schema: question[{content,time}] and
responses[{content,time}] (all timed).  Each timed response becomes one
required help point whose reference text is the response content.
"""
import json, math, sys, os
from pathlib import Path

SRC = Path(os.environ["RL_SOURCE_MANIFEST"])
OUT = Path(os.environ["RL_INTERMEDIATE_MANIFEST"])
OUT.parent.mkdir(parents=True, exist_ok=True)

n_in = n_out = n_skip = 0
with OUT.open("w") as sink:
    for line in SRC.open():
        if not line.strip():
            continue
        r = json.loads(line); n_in += 1
        fam = r.get("task_family") or r.get("primary_task") or "?"
        q = r.get("question") or []
        problem = (q[0].get("content") if isinstance(q, list) and q else str(q or "")).strip()
        dur = float(r.get("duration_sec") or 0)
        vid = r.get("resolved_uri") or (r.get("videos") or [None])[0]
        hps = []
        for resp in r.get("responses") or []:
            t = resp.get("time")
            c = (resp.get("content") or "").strip()
            if t is None or not c:
                continue
            t = float(t)
            if not (0 <= t <= dur + 1):
                continue
            hps.append({
                "optimal_s": t,
                "start_s": max(0.0, t - 2.0),
                "end_s": min(dur, t + 2.0),
                "necessity": "required",
                "reference": c,
            })
        n_frames = int(math.floor(dur))
        if not problem or not vid or not hps or n_frames < 5:
            n_skip += 1
            continue
        hps.sort(key=lambda h: h["optimal_s"])
        rid = f"{fam}_{r.get('annotation_id') or r.get('id')}"
        sink.write(json.dumps({
            "id": rid,
            "task_family": fam,
            "source_dataset": "StreamVibe-RL/" + str(r.get("rl_source") or r.get("source_canonical") or fam),
            "problem_reserved_text": problem,
            "videos": [vid],
            "n_frames": n_frames,
            "duration_s": dur,
            # build_gt_centered.py reads help_points from the answer blob
            "answer": json.dumps({
                "id": rid, "task_family": fam, "duration_s": dur,
                "help_points": hps,
            }, ensure_ascii=False),
        }, ensure_ascii=False) + "\n")
        n_out += 1
print(f"in={n_in} out={n_out} skip={n_skip}")
