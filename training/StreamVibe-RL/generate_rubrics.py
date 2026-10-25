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

"""Generate window-specific binary rubrics for single-help-point trajectories.

Rubrics cover intervention density, redundancy, coverage, and consistency.
They exclude timestamp scoring and verbatim-reference requirements, which are
handled by the alignment reward. Completed rows are skipped when resumed."""
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reward as R  # noqa: E402

SRC = os.environ.get("CT_RUBRIC_SRC", "data/StreamVibe-RL/windows.jsonl")
OUT = os.environ.get("CT_RUBRIC_OUT", "data/StreamVibe-RL/rubrics.jsonl")
CONCURRENCY = int(os.environ.get("CT_RUBRIC_CONCURRENCY", "48"))

PROMPT = """你是评测设计师。下面是一个流式视频助手的训练样本：用户任务，以及这段 {dur} 秒片段里唯一一个需要开口提醒的时刻（帮助点）。模型会在片段播放过程中逐秒决定是否开口。你要为「模型这一整条开口序列」设计 3-5 条二元检查项（每条只能判 true/false）。

必须覆盖以下两个维度，每个维度至少一条：

A. 开口的数量与分布
   - 开口是否集中在帮助点附近，没有在远离帮助点（相隔 5 秒以上）的时段反复插话
   - 开口总次数是否不超过 3 次
   - 是否没有出现连续多秒不间断的刷屏式重复开口

判 A 维度时请按"这条检查项是否描述了一个明显的失误"来判：模型在帮助点前后
1-2 秒内多说一两句属于正常，不算失误；只有明显跑题、刷屏、或在片段两端
无关时段插话才算 false。

B. 内容的覆盖与真实性
   - 开口内容是否提到了该帮助点参考话术的主要意图（不要求逐条覆盖全部细节）
   - 是否没有出现与任务明显矛盾或凭空编造的关键信息
   - 该提醒是否大体上能让用户知道下一步该做什么

严禁出现的检查项（已在别处计分，重复会导致双重计价）：
- 任何关于「第几秒开口 / 是否及时 / 早了晚了」的检查（由 S_time 计分）
- 任何要求「逐字复述参考话术」的检查（由 S_acc 计分）
注意：「开口了几次、有没有在无关时段插话」属于 A 维度，是允许的。

本片段只有一个帮助点，因此不要设计任何关于「多次提醒之间的先后顺序」的检查项。

用户任务: {problem}
帮助点: (t={t}s，片段范围 {lo}-{hi}s) {ref}

输出 JSON: {{"checkpoints": ["...", ...]}}，每条中文、可独立判真假。"""


def one(row):
    a = json.loads(row["answer"]) if isinstance(row["answer"], str) else row["answer"]
    hps = a.get("help_points") or []
    if not hps:
        return None
    h = hps[0]
    lo = float(row.get("trajectory_start_s", 0))
    hi = float(row.get("trajectory_end_s", 0))
    prompt = PROMPT.format(
        dur=int(hi - lo),
        problem=row.get("problem_reserved_text") or row.get("problem") or a.get("id", ""),
        t=h.get("optimal_s"), lo=int(lo), hi=int(hi),
        ref=h.get("reference", ""),
    )
    for attempt in range(6):
        try:
            resp = R._openai_judge_post(prompt)
            data = json.loads(resp["choices"][0]["message"]["content"])
            cps = [str(c).strip() for c in data.get("checkpoints", []) if str(c).strip()]
            if 2 <= len(cps) <= 6:
                return {"id": a["id"], "n_hp": 1, "checkpoints": cps}
            raise ValueError(f"bad checkpoint count {len(cps)}")
        except Exception as exc:  # noqa: BLE001
            code = getattr(exc, "code", None)
            time.sleep(min(4 * 2 ** min(attempt, 4), 60) if code in (429, 500, 502, 503, 504) else 2)
    return None


def main():
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            if line.strip():
                done.add(json.loads(line)["id"])
    todo = []
    for line in open(SRC):
        if not line.strip():
            continue
        row = json.loads(line)
        a = json.loads(row["answer"]) if isinstance(row["answer"], str) else row["answer"]
        if (a.get("help_points") or []) and a["id"] not in done:
            todo.append(row)
    print(f"todo={len(todo)} (already done={len(done)})", flush=True)
    lock = threading.Lock()
    ok = fail = 0
    total = len(todo)
    with open(OUT, "a") as sink, ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        for res in pool.map(one, todo):
            with lock:
                if res:
                    sink.write(json.dumps(res, ensure_ascii=False) + "\n")
                    ok += 1
                else:
                    fail += 1
                if (ok + fail) % 200 == 0:
                    sink.flush()
                    print(f"progress {ok+fail}/{total} ok={ok} fail={fail}", flush=True)
    print(f"DONE ok={ok} fail={fail}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
