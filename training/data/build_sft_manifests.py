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

"""Repartition CT/PS RL to 3K each and assemble StreamVibe-SFT manifests."""
from __future__ import annotations

import collections
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(os.environ["STREAMVIBE_SOURCE_ROOT"])
POOL_ROOT = Path(os.environ["STREAMVIBE_POOL_ROOT"])
OLD = POOL_ROOT / "StreamVibe-RL/source_manifests"
OUT = POOL_ROOT / "StreamVibe-SFT/manifests"
MISSING_PS_GROUPS = {"478a0eafb6cd1822dea278466375b5eeb8088be820255e8b94e2429101b1f0b9"}


def rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip(): yield json.loads(line)


def atomic(path, iterable):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial"); n = 0
    with tmp.open("w", encoding="utf-8") as handle:
        for row in iterable:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"); n += 1
        handle.flush(); os.fsync(handle.fileno())
    os.replace(tmp, path); return n


def rank(seed, value):
    return hashlib.sha256(seed + b"\0" + str(value).encode()).hexdigest()


def as_sft(row):
    result = dict(row); result.pop("rl_source", None); result["row_kind"] = "streaming"
    return result


def stream_files(paths, *, drop_missing_ps=False):
    for path in paths:
        for row in rows(path):
            if drop_missing_ps and str(row.get("split_group")) in MISSING_PS_GROUPS: continue
            row["row_kind"] = "streaming"
            yield row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    original = list(rows(OLD / "StreamVibe-RL-source.jsonl"))
    ct = [r for r in original if r.get("rl_source") == "criterion_timing"]
    ps = [r for r in original if r.get("rl_source") == "streaming40k"]
    timing = [r for r in original if r.get("rl_source") == "timing_extra"]

    retained_ct = []
    for family in ("G", "L", "M", "T"):
        pool = sorted((r for r in ct if r.get("task_family") == family),
                      key=lambda r: rank(bytes.fromhex("76382d6374336b"), r["annotation_id"]))
        retained_ct.extend(pool[:750])
    retained_ct_ids = {r["annotation_id"] for r in retained_ct}
    released_ct = [r for r in ct if r["annotation_id"] not in retained_ct_ids]

    groups = collections.defaultdict(list)
    for row in ps: groups[str(row["split_group"])].append(row)
    ordered = sorted(groups, key=lambda g: rank(bytes.fromhex("76382d7073336b2d67726f7570"), g))
    retained_groups, total = set(), 0
    # The pool contains thousands of singleton groups, so deterministic greedy
    # selection reaches exactly 3,000 while preserving full-video boundaries.
    for group in ordered:
        size = len(groups[group])
        if total + size <= 3000:
            retained_groups.add(group); total += size
        if total == 3000: break
    if total != 3000: raise RuntimeError(f"PS group selection reached {total}, not 3000")
    retained_ps = [r for r in ps if str(r["split_group"]) in retained_groups]
    released_ps = [r for r in ps if str(r["split_group"]) not in retained_groups]
    if list(map(len, (retained_ct, released_ct, retained_ps, released_ps, timing))) != [3000,2000,3000,2000,5000]:
        raise RuntimeError("unexpected repartition counts")

    rl11 = retained_ct + retained_ps + timing
    atomic(OUT / "StreamVibe-RL-reserved.jsonl", rl11)
    atomic(OUT / "StreamVibe-RL-criterion.jsonl", retained_ct)
    atomic(OUT / "StreamVibe-RL-streaming.jsonl", retained_ps)

    ct_sft = list(rows(OLD / "criterion_timing_train.jsonl")) + [as_sft(r) for r in released_ct]
    ps_base = [r for r in rows(OLD / "streaming_train.jsonl")
               if str(r.get("split_group")) not in MISSING_PS_GROUPS]
    ps_sft = ps_base + [as_sft(r) for r in released_ps]
    ct_count = atomic(OUT / "StreamVibe-SFT-criterion.jsonl", (as_sft(r) for r in ct_sft))
    ps_count = atomic(OUT / "StreamVibe-SFT-streaming.jsonl", (as_sft(r) for r in ps_sft))

    timing_train = OUT / "StreamVibe-SFT-video.jsonl"
    timing_count = sum(1 for _ in rows(timing_train))
    timing_report = json.loads((OUT / "StreamVibe-SFT-video-report.json").read_text())
    if timing_count != int(timing_report["rows"]) or not timing_report.get("accepted_available_maximum"):
        raise RuntimeError(f"Timing manifest/report mismatch: rows={timing_count}, report={timing_report}")
    train_count = atomic(OUT / "StreamVibe-SFT-train.jsonl", stream_files([
        timing_train, OUT / "StreamVibe-SFT-criterion.jsonl", OUT / "StreamVibe-SFT-streaming.jsonl"
    ]))
    dev_count = atomic(OUT / "StreamVibe-SFT-dev.jsonl", stream_files([
        ROOT / "manifests/dev.jsonl",
        OLD / "criterion_timing_dev.jsonl", OLD / "streaming_dev.jsonl",
    ], drop_missing_ps=True))

    def audit_keys(path):
        ids, groups = set(), set(); bad_bounds = 0; count = 0
        for row in rows(path):
            count += 1
            ids.add(str(row.get("window_id") or row.get("annotation_id") or ""))
            if row.get("split_group"): groups.add(str(row["split_group"]))
            duration = row.get("duration_sec")
            if (duration is None or abs(float(row.get("window_start_sec") or 0.0)) > 1e-6
                    or abs(float(row.get("window_end_sec") or 0.0) - float(duration)) > 1.01):
                bad_bounds += 1
        return count, ids, groups, bad_bounds

    train_audit = audit_keys(OUT / "StreamVibe-SFT-train.jsonl")
    rl_audit = audit_keys(OUT / "StreamVibe-RL-reserved.jsonl")
    dev_audit = audit_keys(OUT / "StreamVibe-SFT-dev.jsonl")
    contamination = {
        "train_duplicate_ids": train_audit[0] - len(train_audit[1]),
        "train_bad_fullvideo_bounds": train_audit[3],
        "train_rl_id_overlap": len(train_audit[1] & rl_audit[1]),
        "train_rl_group_overlap": len(train_audit[2] & rl_audit[2]),
        "train_dev_id_overlap": len(train_audit[1] & dev_audit[1]),
        "train_dev_group_overlap": len(train_audit[2] & dev_audit[2]),
    }
    hard_failures = {key: contamination[key] for key in (
        "train_duplicate_ids", "train_bad_fullvideo_bounds", "train_rl_id_overlap", "train_dev_id_overlap"
    )}
    if any(hard_failures.values()):
        raise RuntimeError(f"StreamVibe-SFT annotation/full-video audit failed: {contamination}")

    report = {
        "schema": "StreamVibe-SFT", "train_rows": train_count,
        "train_components": {"timing": timing_count, "ct": ct_count, "ps40k": ps_count},
        "dev_rows": dev_count, "rl_rows": len(rl11),
        "rl_components": {"criterion_timing": 3000, "streaming40k": 3000, "timing_extra": 5000},
        "ct_rl_families": dict(collections.Counter(r["task_family"] for r in retained_ct)),
        "ps_rl_task_names": dict(collections.Counter(r.get("task_name") for r in retained_ps)),
        "ps_rl_unique_videos": len(retained_groups),
        "dropped_missing_ps_rows": 2, "dropped_missing_ps_groups": sorted(MISSING_PS_GROUPS),
        "lambda_text": 2.5, "row_kind": "streaming", "contamination_audit": contamination,
        "video_group_overlap_allowed": True,
    }
    (OUT / "StreamVibe-SFT-report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__": main()
