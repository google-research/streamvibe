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

"""Convert normalized help-point annotations into bounded parent trajectories."""
import argparse, json, statistics as st
from collections import Counter
from pathlib import Path


def hp_time(h):
    v = h.get("optimal_s", h.get("start_s"))
    return float(v) if v is not None else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--cap", type=int, default=60)
    ap.add_argument("--lead", type=int, default=10, help="首个帮助点之前留多少秒")
    ap.add_argument("--segment", action="store_true", help="切成多段(否则只取第一段)")
    ap.add_argument("--quality-from", type=Path, default=None)
    ap.add_argument("--balance-sources", action="store_true",
                    help="按 source_dataset 下采样到最少的那一源, 恢复 StreamVibe-RL 的 1:1:1 配平。"
                         "长视频源(criterion_timing 中位 125s)每片切出的段更多, 不加这个开关"
                         "行级占比会变成 49/26/25, 而 StreamVibe-RL 是刻意做成视频级 5000/5000/5000 的。")
    ap.add_argument("--balance-seed", type=int, default=0)
    ap.add_argument("--early", type=int, default=4, help="仅用于统计: 非对称窗口的早侧")
    ap.add_argument("--late", type=int, default=2, help="仅用于统计: 非对称窗口的晚侧")
    a = ap.parse_args()

    good = None
    if a.quality_from:
        good = {json.loads(l)["source_id"] for l in a.quality_from.open()}

    out, n_in, n_hp_total, n_hp_kept = [], 0, 0, 0
    for line in a.rows.open():
        if not line.strip():
            continue
        r = json.loads(line)
        n_in += 1
        if good is not None and r["id"] not in good:
            continue
        gt = json.loads(r["answer"]) if isinstance(r["answer"], str) else dict(r["answer"])
        hps = gt.get("help_points") or []
        times = sorted(t for t in (hp_time(h) for h in hps) if t is not None)
        n = int(r.get("n_frames") or 0) or int(float(r.get("duration_s") or 0))
        if not times or n <= 1:
            continue
        n_hp_total += len(times)

        cur = max(0, int(times[0] - a.lead))
        seg_i = 0
        while cur < n - 1:
            we = int(min(n - 1, cur + a.cap))
            if we <= cur:
                break
            inside = [h for h in hps if hp_time(h) is not None and cur <= hp_time(h) <= we]
            if inside:
                sid = f"{r['id']}::sg{seg_i:02d}"
                answer = {
                    "id": sid, "source_id": r["id"],
                    "task_family": gt.get("task_family", r.get("task_family")),
                    "duration_s": gt.get("duration_s", r.get("duration_s")),
                    "trajectory_start_s": cur, "trajectory_end_s": we,
                    "help_points": inside, "protected_help_points": [],
                    "history_help_points": [
                        h for h in hps if hp_time(h) is not None and hp_time(h) < cur
                    ],
                }
                item = dict(r)
                item["id"] = sid
                item["source_id"] = r["id"]
                item["trajectory_start_s"] = cur
                item["trajectory_end_s"] = we
                item["problem_reserved_text"] = r.get("problem_reserved_text") or r.get("problem")
                item["answer"] = json.dumps(answer, ensure_ascii=False)
                out.append(item)
                n_hp_kept += len(inside)
                seg_i += 1
            if not a.segment:
                break
            cur = we + 1

    if a.balance_sources:
        import random
        bysrc = {}
        for it in out:
            bysrc.setdefault(it.get("source_dataset") or "?", []).append(it)
        keep = min(len(v) for v in bysrc.values())
        rng = random.Random(a.balance_seed)
        balanced = []
        for k in sorted(bysrc):
            v = bysrc[k][:]
            rng.shuffle(v)
            balanced.extend(v[:keep])
        print(f"[balance] 每源保留 {keep} 行; " +
              ", ".join(f"{k}:{len(v)}->{keep}" for k, v in sorted(bysrc.items())))
        # 重算帮助点保留率(下采样后)
        n_hp_kept = sum(len(json.loads(it["answer"])["help_points"]) for it in balanced)
        out = balanced

    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w") as f:
        for item in out:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # ---- 分布报告 ----
    spans, nhp, pfs, srcs, fams = [], [], [], Counter(), Counter()
    for it in out:
        ans = json.loads(it["answer"])
        ws, we = int(ans["trajectory_start_s"]), int(ans["trajectory_end_s"])
        ts = [hp_time(h) for h in ans["help_points"] if hp_time(h) is not None]
        spans.append(we - ws); nhp.append(len(ts))
        pos = sum(1 for s in range(ws, we + 1)
                  if any(-a.early <= s - t <= a.late for t in ts))
        pfs.append(pos / max(1, we - ws + 1))
        srcs[it.get("source_dataset") or "?"] += 1
        fams[ans.get("task_family") or "?"] += 1
    q = lambda v, p: sorted(v)[min(len(v) - 1, int(p / 100 * len(v)))]
    print(json.dumps({
        "输入视频": n_in, "输出行数": len(out),
        "帮助点保留": f"{n_hp_kept}/{n_hp_total} = {n_hp_kept/max(1,n_hp_total):.1%}",
        "span均值": round(st.mean(spans), 1), "span中位": st.median(spans),
        "span_p90": q(spans, 90), "span_max": max(spans),
        "HP中位": st.median(nhp), "HP均值": round(st.mean(nhp), 2),
        "多HP占比": f"{sum(1 for k in nhp if k >= 2)/len(nhp):.1%}",
        "该开口秒_中位": f"{st.median(pfs):.1%}", "该开口秒_均值": f"{st.mean(pfs):.1%}",
        "三源": dict(srcs), "家族": dict(fams.most_common()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
