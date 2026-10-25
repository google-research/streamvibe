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

"""StreamVibe alignment and trajectory rewards.

The StreamVibe-RL branch implements timing-times-content one-to-one matching and
R_align = (1 + alpha) * matched_score / (N_pred + alpha * N_gt). The trainer
combines 0.7 R_align + 0.3 R_glob; format validity is diagnostic only.
Earlier reward branches remain compatibility utilities, not the selected recipe."""

from __future__ import annotations

import itertools
import json
import math
import os
import random
import re
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from functools import lru_cache


USE_INITIAL_REWARD_WEIGHTS = os.environ.get("STREAMVIBE_INITIAL_REWARD_WEIGHTS") == "1"
# Select the reward formulation. The exponential variant provides continuous
# timing credit within the matching window.
REWARD_MODE = os.environ.get("STREAMVIBE_REWARD_MODE", "StreamVibe-RL")


TURN_ALPHA = float(os.environ.get("CT_TURN_ALPHA", "1.0"))
TIMING_TAU = float(os.environ.get("CT_TIMING_TAU", "1.5"))


TIMING_REF_S = float(os.environ.get("CT_TIMING_REF_S", "0.66"))


# Linear timing credit: max(0, 1 - |dt| / tau), with configurable scale.
TIMING_TAU_ALIGN = float(os.environ.get("STREAMVIBE_TIMING_TAU", "8.0"))


def s_time_score(delta: float) -> float:
    """Timing credit for a matched response |Delta t| seconds off the GT."""
    if REWARD_MODE == "StreamVibe-RL":
        return max(0.0, 1.0 - abs(float(delta)) / TIMING_TAU_ALIGN)
    if REWARD_MODE in ("exponential", "normalized_exponential", "turn_local", "trajectory_soft_f1"):
        raw = math.exp(-abs(float(delta)) / TIMING_TAU)
        if REWARD_MODE in ("normalized_exponential", "turn_local", "trajectory_soft_f1"):
            raw /= math.exp(-TIMING_REF_S / TIMING_TAU)
            return min(2.0, raw)          # cap: a bullseye is worth <= 2 hits
        return raw
    return 1.0 if abs(float(delta)) < 0.5 else 0.5

W_TIMING = 0.50
W_CONTENT = 0.30
W_SILENCE = 0.10 if USE_INITIAL_REWARD_WEIGHTS else 0.05
W_MISTIMING = float(os.environ.get("CT_W_MISTIMING", "0.10" if USE_INITIAL_REWARD_WEIGHTS else "0.20"))
W_FALSE = float(os.environ.get("CT_W_FALSE", "0.25" if USE_INITIAL_REWARD_WEIGHTS else "0.35"))
# Silence has to cost something.  With miss 0.25 < false 0.35, dropping a help
# point is cheaper than speaking at the wrong second, so "say less" is the
# optimum -- exactly what every run converged to (step-10 eval: FPR_near fell
# to 0.2523 but F1 fell with it and J_near hit 0.0514, its worst ever).
W_MISS = float(os.environ.get("CT_W_MISS", "0.25"))
W_DUPLICATE = 0.15 if USE_INITIAL_REWARD_WEIGHTS else 0.25
POSITIVE_WINDOW_S = float(os.environ.get("CT_POSITIVE_WINDOW_S", "2.0"))
MISS_SPREAD_SECONDS = 5  # ±2s window holds 5 integer seconds


TURN_FALSE_WEIGHT = float(os.environ.get("STREAMVIBE_TURN_FALSE_WEIGHT", "0.35"))
TURN_MISS_WEIGHT = float(os.environ.get("STREAMVIBE_TURN_MISS_WEIGHT", "0.35"))


TURN_ACCURACY_GATE = float(os.environ.get("CT_ACC_GATE", "0.0"))


TURN_SOFT_WEIGHT = float(os.environ.get("STREAMVIBE_TURN_SOFT_WEIGHT", "0.15"))


MARGIN_ALPHA = float(os.environ.get("CT_MARGIN_ALPHA", "0"))


RUBRIC_W = float(os.environ.get("CT_RUBRIC_W", "0.0"))


def window_seconds() -> int:
    """Integer seconds inside the positive window (asym 4/2 -> 7, sym +-2 -> 5)."""
    if WINDOW_ASYM:
        return int(WINDOW_EARLY_S) + int(WINDOW_LATE_S) + 1
    return 2 * int(POSITIVE_WINDOW_S) + 1


def group_turn_signal(turn_scores_list: "list[dict[float, float]]") -> bool:
    """True iff some second's r_t differs across the group's lanes.

    The turn-local training signal is A_t = r_t - same-second mean, so the
    degenerate-group check and the zero-variance sentinel must look at per-second
    variance -- trajectory-R variance can be nonzero while every retained A_t
    is zero, and vice versa.  A lane missing a second contributes 0.0 (its far
    silence is worth exactly that)."""
    seconds: set = set()
    for ts in turn_scores_list:
        seconds.update(ts)
    for t in seconds:
        vals = {round(ts.get(t, 0.0), 12) for ts in turn_scores_list}
        if len(vals) > 1:
            return True
    return False

# Asymmetric acceptance window (use_soft_f1).  Helping early is useful -- the user has
# not made the mistake yet -- while helping late is worthless, so a symmetric
# +-2s window mis-states the task.  With EARLY=4 / LATE=2 the model gets 7
# candidate seconds per help point instead of 5, and the extra room is all on
# the side that actually helps.  Set CT_WINDOW_ASYM=1 to enable.
WINDOW_ASYM = os.environ.get("CT_WINDOW_ASYM", "0") == "1"
WINDOW_EARLY_S = float(os.environ.get("CT_WINDOW_EARLY_S", "4.0"))
WINDOW_LATE_S = float(os.environ.get("CT_WINDOW_LATE_S", "2.0"))


def signed_offset(t: float, gt_t: float) -> float:
    """Seconds the response came AFTER the help point (negative = early)."""
    return float(t) - float(gt_t)


def in_positive_window(t: float, gt_t: float, positive_window_s: float = POSITIVE_WINDOW_S) -> bool:
    d = signed_offset(t, gt_t)
    if WINDOW_ASYM:
        return -WINDOW_EARLY_S <= d <= WINDOW_LATE_S
    return abs(d) <= positive_window_s


def window_distance(t: float, gt_t: float) -> float:
    """Distance used for timing credit: early seconds are discounted so that
    speaking 3s early is not punished as hard as speaking 3s late."""
    d = signed_offset(t, gt_t)
    if WINDOW_ASYM and d < 0:
        # compress the early side onto the late scale
        return abs(d) * (WINDOW_LATE_S / WINDOW_EARLY_S)
    return abs(d)

SILENCE_RE = re.compile(r"</silence>", re.IGNORECASE)
RESPONSE_RE = re.compile(r"</response>", re.IGNORECASE)
STEP_RE = re.compile(
    r"(?:^|\n)\s*(?:\[|<)?\s*t\s*[=:]?\s*(\d+(?:\.\d+)?)\s*s?\s*(?:\]|>)?\s*[:\-]?\s*"
    r"(.*?)(?=(?:\n\s*(?:\[|<)?\s*t\s*[=:])|\Z)",
    re.IGNORECASE | re.DOTALL,
)


@dataclass
class Step:
    t: float | None
    kind: str
    text: str
    valid: bool
    margin: "float | None" = None   


def parse_trajectory(text: str) -> list[Step]:
    """Parse a rendered trajectory string for offline replay."""
    result = []
    for match in STEP_RE.finditer(text or ""):
        chunk = match.group(2)
        has_response = bool(RESPONSE_RE.search(chunk))
        has_silence = bool(SILENCE_RE.search(chunk))
        body = RESPONSE_RE.split(chunk, maxsplit=1)[-1].strip() if has_response else ""
        valid = has_response ^ has_silence
        if has_response and not body:
            valid = False
        kind = (
            "response" if has_response and not has_silence
            else "silence" if has_silence and not has_response
            else "malformed"
        )
        result.append(Step(float(match.group(1)), kind, body, valid))
    return result


def steps_from_turns(turns: list[tuple], margins: "dict[float, float] | None" = None) -> list[Step]:
    """Build Steps straight from rollout turns [(t, kind, canonical_action)].

    Mirrors parse_trajectory's validity rules: a response with an empty body is
    invalid, a malformed turn is invalid, and either poisons the trajectory.
    """
    steps = []
    margins = margins or {}
    for t, kind, action in turns:
        m = margins.get(float(t))
        if kind == "silence":
            steps.append(Step(float(t), "silence", "", True, m))
        elif kind == "response":
            body = action.split("</response>", 1)[-1].strip() if "</response>" in action else str(action).strip()
            steps.append(Step(float(t), "response", body, bool(body), m))
        else:
            steps.append(Step(float(t), "malformed", str(action), False, m))
    return steps


def hp_time(hp: dict) -> float:
    value = hp.get("optimal_s", hp.get("start_s"))
    if value is None:
        raise ValueError(f"help point has no usable time: {hp}")
    return float(value)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


_STOP = _tokens(
    "the a an and or of to in on at is are was were be been being it its this that "
    "these those for with as by from you your i we they he she him her his their "
    "will would can could should may might must do does did have has had not no yes"
)


def lexical_content(text: str, hp: dict) -> float:
    """Token-overlap content score. Offline analysis only — never a judge fallback."""
    pred = _tokens(text) - _STOP
    if not pred:
        return 0.0
    facts = hp.get("key_facts") or []
    if facts:
        values = []
        for fact in facts:
            expected = _tokens(str(fact)) - _STOP
            if expected:
                values.append(len(expected & pred) / len(expected))
        return sum(values) / len(values) if values else 0.0
    expected = _tokens(hp.get("reference") or "") - _STOP
    return len(expected & pred) / len(expected) if expected else 0.0


# --------------------------------------------------------------------------- #
# Global one-to-one matching (identical port)
# --------------------------------------------------------------------------- #
def optimal_one_to_one_matches(
    targets: list[dict],
    candidates: list[int],
    steps: list[Step],
    positive_window_s: float = POSITIVE_WINDOW_S,
) -> dict[int, int]:
    """Exact non-crossing 1-D assignment, lexicographic cost: far pairs, then
    mistimed pairs, then non-exact hits, then total distance."""
    if not targets or not candidates:
        return {}
    ordered_gt = sorted(range(len(targets)), key=lambda gi: (hp_time(targets[gi]), gi))
    ordered_response = sorted(candidates, key=lambda ri: (float(steps[ri].t), ri))
    pair_count = min(len(ordered_gt), len(ordered_response))

    def pair_cost(gi: int, ri: int) -> tuple[int, int, int, float]:
        t, gt_t = float(steps[ri].t), hp_time(targets[gi])
        delta = abs(t - gt_t)
        hit = in_positive_window(t, gt_t, positive_window_s)
        near = (not hit) and delta <= 3.0
        return (
            int(not hit and not near),
            int(near),
            int(hit and window_distance(t, gt_t) >= 0.5),
            window_distance(t, gt_t),
        )

    def add(a, b):
        return tuple(x + y for x, y in zip(a, b))

    @lru_cache(maxsize=None)
    def solve(i: int, j: int, remaining: int):
        if remaining == 0:
            return (0, 0, 0, 0.0), ()
        if len(ordered_gt) - i < remaining or len(ordered_response) - j < remaining:
            return None
        options = []
        gi, ri = ordered_gt[i], ordered_response[j]
        tail = solve(i + 1, j + 1, remaining - 1)
        if tail is not None:
            options.append((add(pair_cost(gi, ri), tail[0]), ((gi, ri),) + tail[1]))
        if len(ordered_gt) - (i + 1) >= remaining:
            skipped_gt = solve(i + 1, j, remaining)
            if skipped_gt is not None:
                options.append(skipped_gt)
        if len(ordered_response) - (j + 1) >= remaining:
            skipped_response = solve(i, j + 1, remaining)
            if skipped_response is not None:
                options.append(skipped_response)
        return min(options, key=lambda item: (item[0], item[1])) if options else None

    solution = solve(0, 0, pair_count)
    if solution is None:
        raise RuntimeError("failed to construct one-to-one GT/response assignment")
    return dict(solution[1])


# --------------------------------------------------------------------------- #
# Trajectory analysis (aggregates identical to the original _prepare)
# --------------------------------------------------------------------------- #
def product_one_to_one_matches(targets, candidates, steps, acc_by_pair, positive_window_s):
    """Maximum-weight one-to-one matching using tau(delta) * rho(g, p)."""
    if not targets or not candidates:
        return {}
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    weights = np.zeros((len(targets), len(candidates)), dtype=float)
    for gi, hp in enumerate(targets):
        for cj, ri in enumerate(candidates):
            delta = float(steps[ri].t) - hp_time(hp)
            if in_positive_window(float(steps[ri].t), hp_time(hp), positive_window_s):
                content = min(1.0, max(0.0, float(acc_by_pair.get((gi, ri), 0.0))))
                weights[gi, cj] = s_time_score(delta) * content
    gold, pred = linear_sum_assignment(weights, maximize=True)
    return {int(gi): candidates[int(cj)] for gi, cj in zip(gold, pred)
            if weights[gi, cj] > 0.0}


def analyze(steps: list[Step], gt: dict, positive_window_s: float = POSITIVE_WINDOW_S,
            acc_by_pair: "dict[tuple[int, int], float] | None" = None) -> dict:
    """Match using manuscript product scores; content is judged before matching."""
    targets = gt.get("help_points") or []
    protected = gt.get("protected_help_points") or []
    responses = [i for i, step in enumerate(steps) if step.valid and step.kind == "response" and step.t is not None]

    # use_streamvibe_reward (论文对齐): 论文没有 neutral 免责区 —— protected_help_points 附近的
    # 回答既不算命中也不受罚, 等于给过度发言开了个后门。use_streamvibe_reward 取消该类别, 这些
    # 回答按普通误报进入 n_pred 分母。
    if REWARD_MODE == "StreamVibe-RL":
        neutral = set()
    else:
        neutral = {
            i
            for i in responses
            if any(in_positive_window(float(steps[i].t), hp_time(hp), positive_window_s) for hp in protected)
            and all(not in_positive_window(float(steps[i].t), hp_time(hp), positive_window_s) for hp in targets)
        }
    candidates = [i for i in responses if i not in neutral]

    if REWARD_MODE == "StreamVibe-RL":
        matched_gt = product_one_to_one_matches(
            targets, candidates, steps, acc_by_pair or {}, positive_window_s)
    elif REWARD_MODE in ("turn_local", "trajectory_soft_f1"):


        used: set = set()
        matched_gt = {}
        for gi in sorted(range(len(targets)), key=lambda g: (hp_time(targets[g]), g)):
            gt_t = hp_time(targets[gi])
            pool = [
                i for i in candidates
                if i not in used and in_positive_window(float(steps[i].t), gt_t, positive_window_s)
            ]
            if pool:
                ri = min(pool, key=lambda i: (float(steps[i].t), i))
                matched_gt[gi] = ri
                used.add(ri)
    else:
        matched_gt = optimal_one_to_one_matches(targets, candidates, steps, positive_window_s)
    used_response = set(matched_gt.values())

    timing_sum = 0.0
    mistiming = false = miss = 0
    content_pairs = []
    hit_training_count = 0
    response_class: dict[int, tuple] = {i: ("neutral",) for i in neutral}
    missed_targets: list[int] = []
    for gi, hp in enumerate(targets):
        ri = matched_gt.get(gi)
        if ri is None:
            miss += 1
            missed_targets.append(gi)
            continue
        t_resp, gt_t = float(steps[ri].t), hp_time(hp)
        delta = abs(t_resp - gt_t)
        if in_positive_window(t_resp, gt_t, positive_window_s):
            delta = window_distance(t_resp, gt_t)
            timing_sum += s_time_score(delta)
            hit_training_count += 1
            content_pairs.append((gi, ri, hp, steps[ri].text))
            response_class[ri] = ("hit", gi, delta)
        elif delta <= 3.0:
            mistiming += 1
            response_class[ri] = ("mistimed", gi, delta)
        else:
            false += 1
            response_class[ri] = ("far", gi, delta)

    duplicate = 0
    for ri in candidates:
        if ri in used_response:
            continue
        distance = min((abs(float(steps[ri].t) - hp_time(hp)) for hp in targets), default=float("inf"))
        if distance <= 3.0:
            duplicate += 1
            response_class[ri] = ("duplicate", distance)
        else:
            false += 1
            response_class[ri] = ("false", distance)

    negative_steps = [
        step
        for step in steps
        if step.t is not None
        and all(abs(float(step.t) - hp_time(hp)) > positive_window_s for hp in targets)
        and all(abs(float(step.t) - hp_time(hp)) > positive_window_s for hp in protected)
    ]
    silence = (
        sum(step.valid and step.kind == "silence" for step in negative_steps) / len(negative_steps)
        if negative_steps
        else 1.0
    )
    n_gt = max(1, len(targets))
    return {
        "malformed": any(not step.valid for step in steps) or not steps,


        "fmt_score": (sum(1 for step in steps if step.valid) / len(steps)) if steps else 0.0,
        "timing": timing_sum / n_gt,
        "silence": silence,
        "mistiming": min(1.0, mistiming / n_gt),
        "false_response": false / n_gt,  # deliberately uncapped
        "miss": min(1.0, miss / n_gt),
        "duplicate": min(1.0, duplicate / 2.0),
        "content_pairs": content_pairs,
        "n_response": float(len(responses)),
        "n_steps": float(len(steps)),
        "n_targets": float(len(targets)),
        "hit_training_recall": hit_training_count / n_gt,
        # per-index bookkeeping for r_t
        "response_class": response_class,
        "missed_targets": missed_targets,
        "targets": targets,
        "protected": protected,
    }


def overall_reward(analysis: dict, content_scores: list[float]) -> float:
    if analysis["malformed"]:
        return -1.0
    content = sum(content_scores) / max(1, int(analysis["n_targets"]))
    return (
        W_TIMING * analysis["timing"]
        + W_CONTENT * content
        + W_SILENCE * analysis["silence"]
        - W_MISTIMING * analysis["mistiming"]
        - W_FALSE * analysis["false_response"]
        - W_MISS * analysis["miss"]
        - W_DUPLICATE * analysis["duplicate"]
    )


def per_turn_scores(
    steps: list[Step],
    analysis: dict,
    judge_by_pair: dict[int, float],
    positive_window_s: float = POSITIVE_WINDOW_S,
) -> dict[float, float]:
    """Map decision second -> r_t.  ``judge_by_pair`` maps matched response
    index ri -> S_acc in [0, 1]."""
    if analysis["malformed"]:
        return {float(step.t): 0.0 for step in steps if step.t is not None}

    targets = analysis["targets"]
    missed_times = [hp_time(targets[gi]) for gi in analysis["missed_targets"]]
    klass = analysis["response_class"]
    use_turn_rewards = REWARD_MODE == "turn_local"
    scores: dict[float, float] = {}
    for i, step in enumerate(steps):
        if step.t is None:
            continue
        t = float(step.t)
        if step.kind == "response":
            tag = klass.get(i, ("neutral",))
            if tag[0] == "hit":
                delta = tag[2]
                s_time = s_time_score(delta)
                acc = float(judge_by_pair.get(i, 0.0))
                if use_turn_rewards and TURN_ACCURACY_GATE > 0.0 and acc < TURN_ACCURACY_GATE:
                    
                    
                    scores[t] = -TURN_SOFT_WEIGHT
                else:
                    scores[t] = s_time + acc
            elif tag[0] == "neutral":  # protected help point
                scores[t] = 0.0
            elif use_turn_rewards:
                
                
                in_win = any(
                    in_positive_window(t, hp_time(hp), positive_window_s)
                    for hp in targets
                )
                scores[t] = -(TURN_SOFT_WEIGHT if in_win else TURN_FALSE_WEIGHT)
            elif tag[0] == "mistimed":
                scores[t] = -W_MISTIMING
            elif tag[0] in ("far", "false"):
                scores[t] = -W_FALSE
            elif tag[0] == "duplicate":
                scores[t] = -W_DUPLICATE
            else:
                scores[t] = 0.0
        else:  # silence (valid by construction here)
            if use_turn_rewards:
                # spread the miss penalty over the (possibly asymmetric)
                # acceptance window so the total equals one false alarm
                penalty = sum(
                    TURN_MISS_WEIGHT / window_seconds()
                    for mt in missed_times
                    if in_positive_window(t, mt, positive_window_s)
                )
                if (MARGIN_ALPHA > 0.0 and penalty == 0.0
                        and step.margin is not None
                        and not any(in_positive_window(t, hp_time(hp), positive_window_s)
                                    for hp in targets)):
                    # far silence: decisive quiet earns more than hesitant quiet
                    scores[t] = MARGIN_ALPHA * (1.0 - float(step.margin))
                    continue
            else:
                penalty = sum(
                    W_MISS / MISS_SPREAD_SECONDS
                    for mt in missed_times
                    if abs(t - mt) <= positive_window_s
                )
            scores[t] = -penalty
    return scores


# --------------------------------------------------------------------------- #
# Content judge on Vertex with retries and explicit failures
# --------------------------------------------------------------------------- #
PROJECT = os.environ.get("CT_VERTEX_PROJECT", "")
LOCATION = "global"
JUDGE_MODEL = os.environ.get("CT_JUDGE_MODEL", "gemini-3.7-flash")
JUDGE_URL = (
    f"https://aiplatform.googleapis.com/v1/projects/{PROJECT}"
    f"/locations/{LOCATION}/publishers/google/models/{JUDGE_MODEL}:generateContent"
)
JUDGE_TIMEOUT = int(os.environ.get("CT_JUDGE_TIMEOUT", "180"))
JUDGE_MAX_TRIES = int(os.environ.get("CT_JUDGE_MAX_TRIES", "12"))

CONTENT_PROMPT = """Grade the factual correctness and usefulness of proactive replies.
Each item gives a user task, visible-event ground truth, and one candidate reply.
Score ONLY whether the reply correctly conveys useful help grounded in that event.
Use a continuous score from 0.0 to 1.0. Unsupported or irrelevant claims score 0.
All items below come from candidates in the SAME GRPO group. Apply one consistent
scoring scale across the entire list so their relative ordering is meaningful.

Items:
{items}

Return ONLY JSON: {{"scores": [{{"index": 0, "score": 0.0}}, ...]}}
"""


class _Token:
    def __init__(self, ttl: float = 2400.0) -> None:
        self._tok: str | None = None
        self._exp = 0.0
        self._ttl = ttl
        self._lock = threading.Lock()

    def get(self, force: bool = False) -> str:
        with self._lock:
            if force or self._tok is None or time.time() >= self._exp:
                out = subprocess.run(
                    ["gcloud", "auth", "application-default", "print-access-token"],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if out.returncode != 0:
                    raise RuntimeError(f"gcloud token failed: {out.stderr[-300:]}")
                self._tok = out.stdout.strip()
                self._exp = time.time() + self._ttl
            return self._tok


_TOKEN = _Token()


def _load_api_keys() -> list[str]:
    """Read judge credentials only from the runtime environment."""
    raw = os.environ.get("CT_JUDGE_API_KEYS", "")
    return [key.strip() for key in raw.split(",") if key.strip()]


def _load_env_file_value(name: str) -> str:
    # Historical function name retained for compatibility; no file is read.
    return os.environ.get(name, "")


API_KEYS = _load_api_keys()
# Optional runtime key pool; extra keys are used on retries.
PRIMARY_KEYS = int(os.environ.get("CT_JUDGE_PRIMARY_KEYS", "3"))
JUDGE_URL_APIKEY = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{JUDGE_MODEL}:generateContent"
)
_KEY_CURSOR = itertools.count()

# Judge provider: "openai" routes content judging to the OpenAI chat API
# (single key, no rotation); anything else keeps the Gemini API-key path.
JUDGE_PROVIDER = os.environ.get("CT_JUDGE_PROVIDER", "gemini")
OPENAI_API_KEY = _load_env_file_value("CT_OPENAI_API_KEY")
OPENAI_JUDGE_MODEL = os.environ.get("CT_OPENAI_JUDGE_MODEL", "gpt-5.4-nano")
OPENAI_URL = "https://api.openai.com/v1/chat/completions"


def _openai_judge_post(prompt_text: str) -> dict:
    """One OpenAI chat call returning the parsed JSON body (retry as caller)."""
    if not OPENAI_API_KEY:
        raise RuntimeError("Set CT_OPENAI_API_KEY in the runtime environment")
    body = json.dumps(
        {
            "model": OPENAI_JUDGE_MODEL,
            "messages": [{"role": "user", "content": prompt_text}],
            "response_format": {"type": "json_object"},
            "seed": int(os.environ.get("CT_JUDGE_SEED", "20260820")),
        }
    ).encode()
    req = urllib.request.Request(
        OPENAI_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        },
    )
    with urllib.request.urlopen(req, timeout=JUDGE_TIMEOUT) as r:
        return json.loads(r.read().decode())


def _judge_request(payload: dict, attempt: int) -> urllib.request.Request:
    body = json.dumps(payload).encode()
    if API_KEYS:
        # Round-robin the primary pool; retries widen to the full key list so
        # a rate-limited primary does not stall the group.
        pool = API_KEYS[: PRIMARY_KEYS or None] if attempt < 2 else API_KEYS
        key = pool[next(_KEY_CURSOR) % len(pool)]
        return urllib.request.Request(
            JUDGE_URL_APIKEY,
            data=body,
            headers={"Content-Type": "application/json", "x-goog-api-key": key},
        )
    return urllib.request.Request(
        JUDGE_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {_TOKEN.get(force=attempt > 0)}",
            "Content-Type": "application/json",
        },
    )


def _judge_post(payload: dict) -> dict:
    last: Exception | None = None
    for attempt in range(JUDGE_MAX_TRIES):
        try:
            with urllib.request.urlopen(
                _judge_request(payload, attempt), timeout=JUDGE_TIMEOUT
            ) as r:
                return json.loads(r.read().decode())
        except Exception as exc:  # noqa: BLE001 — every failure retries then raises
            last = exc
            # 429/503 are transient account-level pressure (quota is shared
            # across the whole account): back off exponentially with jitter
            # and ride it out rather than killing a 50-step run.
            delay = _retry_delay(exc, attempt)
            time.sleep(delay + random.uniform(0.0, 0.25 * delay))
    raise RuntimeError(f"judge request failed after {JUDGE_MAX_TRIES} tries: {last}")


def _retry_delay(exc: Exception, attempt: int) -> float:
    """Use exponential backoff for timeouts, rate limits, and server errors."""
    code = getattr(exc, "code", None)
    retryable = code in (429, 500, 502, 503, 504) or isinstance(
        exc, (socket.timeout, TimeoutError)) or "timed out" in str(exc).lower()
    base = min(4.0 * (2 ** min(attempt, 4)), 90.0) if retryable else min(2.0 * (attempt + 1), 8.0)
    return base + random.uniform(0.0, 0.25 * base)


JUDGE_BATCH = int(os.environ.get("CT_JUDGE_BATCH", "24"))


def judge_content(items: list[dict]) -> list[float]:
    """Score matched (problem, help point, reply) items.

    Raises on unparseable or incomplete results — the training loop must never
    silently train on a different content signal than it claims to use.
    """
    if not items:
        return []
    if len(items) > JUDGE_BATCH:
        out: list[float] = []
        for i in range(0, len(items), JUDGE_BATCH):
            out.extend(judge_content(items[i:i + JUDGE_BATCH]))
        return out
    blocks = []
    for i, item in enumerate(items):
        hp = item["hp"]
        blocks.append(
            f"Item {i}:\n"
            f"  User task: {item['problem']}\n"
            f"  Reference help: {hp.get('reference', '')}\n"
            f"  Visible evidence: {hp.get('evidence', '')}\n"
            f"  Key facts: {json.dumps(hp.get('key_facts') or [], ensure_ascii=False)}\n"
            f"  Candidate reply: {item['text']}"
        )
    prompt_text = CONTENT_PROMPT.format(items='\n\n'.join(blocks))
    if JUDGE_PROVIDER == "openai":
        last: Exception | None = None
        response = None
        for attempt in range(JUDGE_MAX_TRIES):
            try:
                response = _openai_judge_post(prompt_text)
                break
            except Exception as exc:  # noqa: BLE001 — retry then raise
                last = exc
                code = getattr(exc, "code", None)
                time.sleep(_retry_delay(exc, attempt))
        if response is None:
            raise RuntimeError(f"openai judge failed after {JUDGE_MAX_TRIES} tries: {last}")
        try:
            text = response["choices"][0]["message"]["content"]
            data = json.loads(text)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"openai judge returned unparseable content: {exc}") from exc
        return _extract_scores(data, len(items))
    response = _judge_post(
        {
            "contents": [{"role": "user", "parts": [{"text": prompt_text}]}],
            "generationConfig": {
                "temperature": 0.0,
                "seed": int(os.environ.get("CT_JUDGE_SEED", "20260820")),
                "maxOutputTokens": 4096,
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "OBJECT",
                    "properties": {
                        "scores": {
                            "type": "ARRAY",
                            "items": {
                                "type": "OBJECT",
                                "properties": {
                                    "index": {"type": "INTEGER"},
                                    "score": {"type": "NUMBER"},
                                },
                                "required": ["index", "score"],
                            },
                        }
                    },
                    "required": ["scores"],
                },
            },
        }
    )
    try:
        parts = response["candidates"][0]["content"]["parts"]
        text = "".join(part.get("text", "") for part in parts)
        data = json.loads(text)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"judge returned unparseable content: {exc}") from exc
    return _extract_scores(data, len(items))


# 严格模式只在重试循环内使用: 缺项先当作可重试的失败, 重试耗尽后才降级补 0。
_EXTRACT_STRICT = False


def _extract_scores(data: dict, n_items: int) -> list[float]:
    scores: list[float | None] = [None] * n_items
    for value in data.get("scores", []):
        if not isinstance(value, dict) or not isinstance(value.get("index"), int):
            continue
        index = value["index"]
        if not 0 <= index < len(scores):
            continue
        try:
            scores[index] = min(1.0, max(0.0, float(value.get("score"))))
        except (TypeError, ValueError):
            continue
    if any(v is None for v in scores):
        if not _EXTRACT_STRICT:
            # 降级: 一次 judge 返回体缺项不该打死一个 10 小时的 run。
            # (attempt 1 就是这么死的: 11:54 抛出, 监督 17:51 才发现, 空转 6 小时)
            missing = sum(1 for v in scores if v is None)
            print(f"[judge] 返回体缺 {missing}/{n_items} 项, 按 0.0 补齐并继续", flush=True)
            return [0.0 if v is None else v for v in scores]
        raise RuntimeError("judge result is missing item scores")
    return [float(v) for v in scores]


# --------------------------------------------------------------------------- #
# Group scoring entry point
# --------------------------------------------------------------------------- #
RUBRIC_PROMPT = """你是评测员。下面是一个用户任务、一组二元检查项，以及 {n} 个模型在同一段视频里的完整开口序列（每条按时间排序）。对每个模型，逐条判断检查项是否满足（true/false）。检查项只涉及多次开口的组合性质（顺序/矛盾/冗余/缺步/可完成性），不要评单句对错或时机。

用户任务: {problem}
检查项:
{checkpoints}

各模型的开口序列:
{transcripts}

输出 JSON: {{"lanes": [{{"index": 0, "passed": [true, ...]}}, ...]}}，passed 长度必须等于检查项数，每个模型一项。"""


def judge_rubric(problem: str, checkpoints: list[str], transcripts: "list[list[tuple]]") -> list:
    """Score each lane's full response sequence against the sample's rubric.

    Return the fraction of binary rubric items passed for every lane,
    including an empty response sequence. Invalid judge output is retried.
    """
    idx = list(range(len(transcripts)))
    if not idx:
        return [None] * len(transcripts)
    blocks = []
    for k, i in enumerate(idx):
        lines = "\n".join(f"    t={t:.0f}s: {text}" for t, text in transcripts[i]) or "    (no responses)"
        blocks.append(f"  模型 {k}:\n{lines}")
    prompt = RUBRIC_PROMPT.format(
        n=len(idx), problem=problem,
        checkpoints="\n".join(f"  {j+1}. {c}" for j, c in enumerate(checkpoints)),
        transcripts="\n".join(blocks),
    )
    last = None
    for attempt in range(JUDGE_MAX_TRIES):
        try:
            resp = _openai_judge_post(prompt)
            data = json.loads(resp["choices"][0]["message"]["content"])
            lanes = {int(l["index"]): l["passed"] for l in data["lanes"]}
            out: list = [None] * len(transcripts)
            for k, i in enumerate(idx):
                passed = lanes.get(k)
                if (not isinstance(passed, list) or len(passed) != len(checkpoints)
                        or any(type(value) is not bool for value in passed)):
                    raise ValueError("judge must return one boolean per rubric item for every lane")
                out[i] = sum(passed) / len(checkpoints)
            return out
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(_retry_delay(exc, attempt))
    raise RuntimeError(f"rubric judge failed after {JUDGE_MAX_TRIES} tries: {last}")


def score_group(
    problem: str,
    gt: dict,
    group_steps: list[list[Step]],
    judge_mode: str = "gemini",
) -> list[dict]:
    """Score the G trajectories of one GRPO group.

    All matched pairs across the group are judged in ONE call so the scale is
    consistent within the group (the ordering is what GRPO consumes).  Returns
    one dict per trajectory: {"R", "turn_scores", "metrics"}.
    """
    if judge_mode not in ("gemini", "lexical", "none"):
        raise ValueError(f"unknown judge_mode {judge_mode!r}")

    acc_maps: list[dict] = [{} for _ in group_steps]
    if REWARD_MODE == "StreamVibe-RL":
        # Judge every in-window candidate before product-score assignment.
        _targets = gt.get("help_points") or []
        pair_items: list[dict] = []
        for li, steps in enumerate(group_steps):
            resp = [i for i, s in enumerate(steps)
                    if s.valid and s.kind == "response" and s.t is not None]
            for gi, hp in enumerate(_targets):
                gt_t = hp_time(hp)
                for ri in resp:
                    if in_positive_window(float(steps[ri].t), gt_t):
                        pair_items.append({"problem": problem, "hp": hp, "text": steps[ri].text,
                                           "gi": gi, "ri": ri, "lane": li})
        if judge_mode == "gemini":
            # 去重: 组内 8 个 lane 常出现同一 (帮助点, 回答文本) 组合, 判一次即可。
            uniq: dict = {}
            for it in pair_items:
                uniq.setdefault((it["gi"], it["text"]), len(uniq))
            keys = sorted(uniq, key=uniq.get)
            u_judged = judge_content([
                {"problem": problem, "hp": _targets[k[0]], "text": k[1]} for k in keys])
            lut = {k: u_judged[i] for i, k in enumerate(keys)}
            pair_judged = [lut[(it["gi"], it["text"])] for it in pair_items]
        elif judge_mode == "lexical":
            pair_judged = [lexical_content(it["text"], it["hp"]) for it in pair_items]
        else:
            pair_judged = [0.0] * len(pair_items)
        for it, v in zip(pair_items, pair_judged):
            acc_maps[it["lane"]][(it["gi"], it["ri"])] = float(v)
        analyses = [analyze(steps, gt, acc_by_pair=acc_maps[li])
                    for li, steps in enumerate(group_steps)]
    else:
        analyses = [analyze(steps, gt) for steps in group_steps]

    all_items: list[dict] = []
    slices: list[tuple[int, int]] = []
    for li, analysis in enumerate(analyses):
        start = len(all_items)
        for gi, ri, hp, text in analysis["content_pairs"]:
            all_items.append({"problem": problem, "hp": hp, "text": text, "ri": ri,
                              "_acc": acc_maps[li].get((gi, ri), 0.0)})
        slices.append((start, len(all_items)))

    if REWARD_MODE == "StreamVibe-RL":
        judged = [it["_acc"] for it in all_items]        # 复用上面的判分, 不重复调用
    elif judge_mode == "gemini":
        judged = judge_content([{k: item[k] for k in ("problem", "hp", "text")} for item in all_items])
    elif judge_mode == "lexical":
        judged = [lexical_content(item["text"], item["hp"]) for item in all_items]
    else:
        judged = [0.0] * len(all_items)

    results = []
    for analysis, steps, (start, end) in zip(analyses, group_steps, slices):
        content_scores = judged[start:end]
        judge_by_pair = {all_items[start + k]["ri"]: judged[start + k] for k in range(end - start)}
        if REWARD_MODE == "StreamVibe-RL":
            # R_turn is the manuscript R_align. Format is logged, not rewarded.
            s_prime = 0.0
            for ri, cls in analysis["response_class"].items():
                if cls[0] == "hit":
                    timing = s_time_score(cls[2])
                    content = min(1.0, max(0.0, float(judge_by_pair.get(ri, 0.0))))
                    s_prime += timing * content
            n_pred = sum(1 for cls in analysis["response_class"].values() if cls[0] != "neutral")
            if int(analysis["n_targets"]) == 0:
                
                
                r_turn = 0.0
            else:
                n_gt = int(analysis["n_targets"])
                scale = 1.0 + TURN_ALPHA
                r_turn = (scale * s_prime) / (n_pred + TURN_ALPHA * n_gt)
            r_fmt = float(analysis.get("fmt_score", 0.0))
            reward = r_turn          # R_traj 由 train.py 补上
        elif REWARD_MODE == "trajectory_soft_f1":
            # Trajectory soft F1; malformed output receives a structural penalty of -1.
            # 分母对每一次多余开口计价, 话痨不需要逐秒罚款就会自我收税。
            s_prime = 0.0
            for ri, cls in analysis["response_class"].items():
                if cls[0] == "hit":
                    s_prime += s_time_score(cls[2]) + float(judge_by_pair.get(ri, 0.0))
            n_pred = sum(1 for cls in analysis["response_class"].values() if cls[0] != "neutral")
            n_gt = max(1, int(analysis["n_targets"]))
            reward = -1.0 if analysis["malformed"] else (2.0 * s_prime) / (n_pred + n_gt)
        else:
            reward = overall_reward(analysis, content_scores)
        turn_scores = per_turn_scores(steps, analysis, judge_by_pair)
        n_gt = max(1, int(analysis["n_targets"]))
        results.append(
            {
                "R": float(reward),
                "R_fmt": float(r_fmt) if REWARD_MODE == "StreamVibe-RL" else None,
                "R_turn": float(r_turn) if REWARD_MODE == "StreamVibe-RL" else None,
                "turn_scores": turn_scores,
                "metrics": {
                    "timing": analysis["timing"],
                    "content": sum(content_scores) / n_gt,
                    "appropriate_silence": analysis["silence"],
                    "mistiming": analysis["mistiming"],
                    "false_response": analysis["false_response"],
                    "miss": analysis["miss"],
                    "duplicate": analysis["duplicate"],
                    "malformed": float(analysis["malformed"]),
                    "fmt_score": float(analysis.get("fmt_score", 0.0)),
                    "response_count": analysis["n_response"],
                    "n_steps": analysis["n_steps"],
                    "hit_training_recall": analysis["hit_training_recall"],
                },
            }
        )


    checkpoints = list(gt.get("rubric") or [])
    if RUBRIC_W > 0.0 and checkpoints and judge_mode != "none":
        transcripts = []
        for analysis, steps in zip(analyses, group_steps):
            if REWARD_MODE != "StreamVibe-RL" and analysis["malformed"]:
                transcripts.append([])          # garbage transcript: exclude
                continue
            transcripts.append([
                (float(step.t), step.text) for step in steps
                if step.valid and step.kind == "response" and step.t is not None
            ])
        cs = judge_rubric(problem, checkpoints, transcripts)
        for result, c in zip(results, cs):
            result["rubric_c"] = c
    return results
