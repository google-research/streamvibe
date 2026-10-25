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

"""Construct 10–15 second windows containing exactly one scored help point.

Exclude adjacent gold events by shrinking bounds; discard windows that become
too short or place the target outside the permitted relative position range.
Remove inherited rubrics so each short window receives a fresh rubric."""
import json
import os
import sys

SRC = os.environ.get("CT_SHORT_SRC", "data/StreamVibe-RL/segments.jsonl")
OUT = os.environ.get("CT_SHORT_OUT", "data/StreamVibe-RL/windows.jsonl")
WIN_MIN = float(os.environ.get("CT_WIN_MIN", "10"))
WIN_MAX = float(os.environ.get("CT_WIN_MAX", "15"))
MAX_HP = 1   # answer-centered: 每条 trajectory 只对准一个 gold response
PRE_FRAC = float(os.environ.get("CT_PRE_FRAC", "0.65"))   # 中心点落在窗口 65% 处
POS_MIN = float(os.environ.get("CT_POS_MIN", "0.20"))     # gold 点相对位置下限
POS_MAX = float(os.environ.get("CT_POS_MAX", "0.85"))     # 上限


def hp_t(h):
    v = h.get("optimal_s", h.get("start_s"))
    return None if v is None else float(v)


def main():
    n_in = n_out = skip_short = skip_dense = skip_pos = 0
    seen = set()
    with open(SRC) as f, open(OUT, "w") as o:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            a = json.loads(row["answer"]) if isinstance(row["answer"], str) else row["answer"]
            hps = [h for h in (a.get("help_points") or []) if hp_t(h) is not None]
            t0 = float(row.get("trajectory_start_s", 0))
            t1 = float(row.get("trajectory_end_s", 0))
            n_in += 1

            for k, hp in enumerate(hps):
                c = hp_t(hp)
                # 先按目标窗口宽度取, 中心点偏后(留出观察时间)
                span = WIN_MAX
                lo = max(t0, c - span * PRE_FRAC)
                hi = min(t1, lo + span)
                lo = max(t0, hi - span)

                # 窗口内除中心点外的 gold 点: 收缩边界排除掉
                others = [hp_t(h) for h in hps if lo <= hp_t(h) <= hi and hp_t(h) != c]
                if others:
                    lo = max(lo, max([t for t in others if t < c], default=lo - 1) + 1)
                    hi = min(hi, min([t for t in others if t > c], default=hi + 1) - 1)
                    if hi - lo < WIN_MIN or not (lo <= c <= hi):
                        skip_dense += 1
                        continue

                if hi - lo < WIN_MIN:
                    skip_short += 1
                    continue

                # 位置过滤: gold 点撞到视频头尾时会被裁到窗口极端位置。
                # 落在 0% 处的样本没有任何观察时间就要决定开口; 落在 100% 处
                # 的晚说 1 秒就出了匹配窗口。两者都不是有效的训练信号。
                rel = (c - lo) / (hi - lo)
                if not (POS_MIN <= rel <= POS_MAX):
                    skip_pos += 1
                    continue
                inside = [hp]                    # 严格一个 gold response

                key = (a.get("source_id", a["id"]), round(lo), round(hi))
                if key in seen:
                    continue
                seen.add(key)

                na = dict(a)
                na["id"] = f'{a["id"]}::w{k}'
                na["trajectory_start_s"] = lo
                na["trajectory_end_s"] = hi
                na["help_points"] = inside
                # 被排除的邻近 gold 点 + 原有 protected, 一并转入 protected
                na["protected_help_points"] = [
                    h for h in (a.get("protected_help_points") or [])
                    if hp_t(h) is not None and lo <= hp_t(h) <= hi
                ] + [h for h in hps if hp_t(h) != c and lo <= hp_t(h) <= hi]
                na["history_help_points"] = [
                    h for h in (a.get("history_help_points") or [])
                    if hp_t(h) is not None and lo <= hp_t(h) <= hi]
                na.pop("rubric", None)          # 旧 rubric 对短窗口无效, 待重新生成

                nr = dict(row)
                nr["id"] = na["id"]
                nr["trajectory_start_s"] = lo
                nr["trajectory_end_s"] = hi
                nr["answer"] = json.dumps(na, ensure_ascii=False)
                o.write(json.dumps(nr, ensure_ascii=False) + "\n")
                n_out += 1
    print(f"输入 {n_in} 条(52s) -> 输出 {n_out} 条")
    print(f"  跳过: 窗口太短 {skip_short}, gold点太密 {skip_dense}, 位置极端 {skip_pos}")


if __name__ == "__main__":
    sys.exit(main())
