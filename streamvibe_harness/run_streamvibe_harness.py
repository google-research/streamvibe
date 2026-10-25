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

"""StreamVibe Harness: causal streaming inference with persistent intent state.

The interaction model chooses silence or response. The harness maintains
intent lifecycles, observed evidence, memory and delivered response history.
Interaction model: StreamVibe-VL. Benchmark: StreamVibe-Bench.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
from collections import OrderedDict
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("VLLM_ENABLEUSE_INITIAL_REWARD_WEIGHTS_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

WORKSPACE_ARTIFACT_SHA256 = os.environ.get(
    "STREAMING_HARNESS_WORKSPACE_ARTIFACT_SHA256", "").strip()
if (WORKSPACE_ARTIFACT_SHA256
        and (len(WORKSPACE_ARTIFACT_SHA256) != 64
             or any(ch not in "0123456789abcdef"
                    for ch in WORKSPACE_ARTIFACT_SHA256.lower()))):
    raise ValueError("invalid Harness workspace artifact identity")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from harness.agents import MemoryTranscriptAgent, build_agent  # noqa: E402
from harness.adapter import HarnessAdapter  # noqa: E402
from harness.card import EpochGuard, parse_mode  # noqa: E402
from harness.experience import ExperienceStore  # noqa: E402
from harness.gate import CadenceGate, load_cadence  # noqa: E402
from harness.context import (  # noqa: E402
    build_chunk_header,
    build_harness_messages,
    build_original32_messages,
    images_of,
    response_epoch_due,
    response_epoch_start,
    rewrite_control_roles,
)
from harness.continual import ContinualController, DecisionEvent  # noqa: E402
from harness.criteria import parse_instruction  # noqa: E402
from harness.memory import MemoryConfig, ThreeTierMemory  # noqa: E402
from harness.novelty import NovelResponseGate  # noqa: E402
from harness.policy import HarnessPolicy, closed_choice_request  # noqa: E402
from harness.perception import (  # noqa: E402
    CurrentEvidenceTracker,
    event_profile_enabled,
    sample_turn_indices,
)
from harness.state import (  # noqa: E402
    FrameTransitionTracker,
    StreamDecisionState,
    should_commit_response_state,
)
from harness.routing import (  # noqa: E402
    memory_ranges_redundant,
    request_profile,
    response_phase_profile,
    routed_prompt_profile,
    routed_state_profile,
    scoped_silence_history,
    standing_route_after_question,
)
from harness.response_generation import (  # noqa: E402
    ACTION_TOKENS,
    committed_history_action,
    compose_response,
    leading_action_token,
    response_support_messages,
    strict_action_token,
)
from harness.skills import ActiveHelpSkills  # noqa: E402
from harness.prompts import (  # noqa: E402
    HARNESS_GROUNDED_MEMORY_SUFFIX,
    HARNESS_GROUNDED_STATE_SUFFIX,
    STREAMVIBE_COMPACT_CONTEXT_SUFFIX,
    STREAMVIBE_CONTEXT_ROLES_SUFFIX,
    STREAMVIBE_CONTEXT_SUFFIX,
    STREAMVIBE_SKILL_CONDITIONS_SUFFIX,
    STREAMVIBE_CURRENT_OBSERVATION_SUFFIX,
    STREAMVIBE_PRIVATE_EVIDENCE_SUFFIX,
    STREAMVIBE_DECISION_CLOCK_SUFFIX,
    STREAMVIBE_SKILL_ROUTER_SUFFIX,
    STREAMVIBE_DISTINCT_EVENT_SUFFIX,
    STREAMVIBE_RESPONSE_CONTINUATION_SUFFIX,
    STREAMVIBE_RESPONSE_NOVELTY_SUFFIX,
    STREAMVIBE_NARRATION_SUFFIX,
    STREAMVIBE_PASSIVE_OBSERVATION_SUFFIX,
    STREAMVIBE_EVENT_WATCH_SUFFIX,
    STREAMVIBE_TYPED_EVIDENCE_SUFFIX,
    STREAMVIBE_INACTIVE_REQUEST_SUFFIX,
    current_event_prompt,
    HARNESS_NATIVE_SUFFIX,
    HARNESS_STANDING_REQUEST_SUFFIX,
    MEMORY_STYLES,
    STREAMVIBE_SYSTEM_PROMPT,
)
from harness.streamspec import expand, spec_turns  # noqa: E402
from harness.triggering import QueryTracker  # noqa: E402

from run_inference import (  # noqa: E402
    MAX_PIXELS,
    MIN_PIXELS,
    _turn_records,
    _user_content,
    action_of,
    clean_text,
    dense_probes,
    is_wrapped,
    load_rows,
    turn_table,
)
from run_inference_streaming import (  # noqa: E402
    STREAMVIBE_TRAINING_SYSTEM_PROMPT,
    TRAINING_NATIVE_SYSTEM_PROMPT,
    anchor_probes,
    generate_guarded,
)

SYSTEM_PROMPTS = {
    # Fixed streaming system prompt.
    "streamvibe": STREAMVIBE_SYSTEM_PROMPT,
    # What the interaction model was trained under. Use to isolate the prompt from the context.
    "streamvibe_training": STREAMVIBE_TRAINING_SYSTEM_PROMPT,
    "training_native": TRAINING_NATIVE_SYSTEM_PROMPT,
    # the interaction model's contract, unchanged, plus the harness facts. See prompts.py.
    "streamvibe_native": STREAMVIBE_TRAINING_SYSTEM_PROMPT + HARNESS_NATIVE_SUFFIX,
    "native_plus": TRAINING_NATIVE_SYSTEM_PROMPT + HARNESS_NATIVE_SUFFIX,
    "memory_standing_request": TRAINING_NATIVE_SYSTEM_PROMPT + HARNESS_STANDING_REQUEST_SUFFIX,
    "memory_grounded": TRAINING_NATIVE_SYSTEM_PROMPT + HARNESS_GROUNDED_MEMORY_SUFFIX,
    
    "memory_grounded_state": TRAINING_NATIVE_SYSTEM_PROMPT + HARNESS_GROUNDED_STATE_SUFFIX,
    "streamvibe_compact_context": TRAINING_NATIVE_SYSTEM_PROMPT + STREAMVIBE_COMPACT_CONTEXT_SUFFIX,
    "streamvibe_context_roles": TRAINING_NATIVE_SYSTEM_PROMPT + STREAMVIBE_CONTEXT_ROLES_SUFFIX,
    "streamvibe_context": TRAINING_NATIVE_SYSTEM_PROMPT + STREAMVIBE_CONTEXT_SUFFIX,
    "streamvibe_skill_conditions": TRAINING_NATIVE_SYSTEM_PROMPT + STREAMVIBE_SKILL_CONDITIONS_SUFFIX,
    "streamvibe_current_observation": TRAINING_NATIVE_SYSTEM_PROMPT + STREAMVIBE_CURRENT_OBSERVATION_SUFFIX,
    "streamvibe_private_evidence": TRAINING_NATIVE_SYSTEM_PROMPT + STREAMVIBE_PRIVATE_EVIDENCE_SUFFIX,
    "streamvibe_decision_clock": TRAINING_NATIVE_SYSTEM_PROMPT + STREAMVIBE_DECISION_CLOCK_SUFFIX,
    "streamvibe_skill_router": TRAINING_NATIVE_SYSTEM_PROMPT + STREAMVIBE_SKILL_ROUTER_SUFFIX,
    "streamvibe_distinct_event": TRAINING_NATIVE_SYSTEM_PROMPT + STREAMVIBE_DISTINCT_EVENT_SUFFIX,
    "streamvibe_response_continuation": TRAINING_NATIVE_SYSTEM_PROMPT + STREAMVIBE_RESPONSE_CONTINUATION_SUFFIX,
    "streamvibe_response_novelty": TRAINING_NATIVE_SYSTEM_PROMPT + STREAMVIBE_RESPONSE_NOVELTY_SUFFIX,
    "streamvibe_narration": (
        TRAINING_NATIVE_SYSTEM_PROMPT + STREAMVIBE_NARRATION_SUFFIX),
    "streamvibe_passive_observation": (
        TRAINING_NATIVE_SYSTEM_PROMPT + STREAMVIBE_PASSIVE_OBSERVATION_SUFFIX),
    "streamvibe_event_watch": (
        TRAINING_NATIVE_SYSTEM_PROMPT + STREAMVIBE_EVENT_WATCH_SUFFIX),
    "streamvibe_typed_evidence": (
        TRAINING_NATIVE_SYSTEM_PROMPT + STREAMVIBE_TYPED_EVIDENCE_SUFFIX),
    "streamvibe_inactive_request": (
        TRAINING_NATIVE_SYSTEM_PROMPT + STREAMVIBE_INACTIVE_REQUEST_SUFFIX),
}

# Increment when prompt/context semantics or metric-bearing runtime behaviour
# changes, so pre-fix traces cannot be silently pooled with corrected runs.
HARNESS_NAME = "StreamVibe Harness"


def window_start(arm: str, memory: ThreeTierMemory, turn: int, Ts: int) -> int:
    if arm == "cumulative":
        return 0
    if arm == "sliding":
        return max(0, turn - Ts + 1)
    if arm in ("original_32", "harness_32"):
        return max(0, turn - 31)   # bookkeeping only; the visual window is --visual-window
    return memory.short_start(turn)


def accept_verified_action(original: str, reviewed: str) -> str:
    """Accept only the verifier's strict action protocol; fail open otherwise."""
    value = (reviewed or "").strip()
    if value == "</silence>":
        return value
    if value.startswith("</response>") and clean_text(value):
        return value
    return original


def decision_input_fingerprint(chat: str, image_paths: list[str]) -> dict:
    """Stable trace-only identity for the complete pre-decision decision input.

    The prompt text and ordered frame identifiers are hashed separately and
    together.  Nothing is added to the model request.  Contemporary A/B runs
    can therefore distinguish a real Harness intervention from greedy-kernel
    or runtime drift when the first different output had identical inputs.
    """
    text_sha = hashlib.sha256(chat.encode("utf-8")).hexdigest()
    image_manifest = "\n".join(str(path) for path in image_paths)
    image_sha = hashlib.sha256(image_manifest.encode("utf-8")).hexdigest()
    full_sha = hashlib.sha256(
        (text_sha + "\n" + image_sha).encode("ascii")).hexdigest()
    return {
        "input_sha256": full_sha,
        "text_sha256": text_sha,
        "images_sha256": image_sha,
        "prompt_chars": len(chat),
        "image_count": len(image_paths),
    }


def remap_video_path(path: str, *, legacy_local_root: str = "",
                     source_root: str = "", local_root: str = "") -> str:
    """Resolve an optional byte-identical node-local video copy.

    ``source_root``/``local_root`` is a general prefix map for evaluation
    datasets outside ``data/videos``. The historical streaming remap remains the
    fallback. A missing local file always returns the original path.
    """
    if source_root and local_root:
        try:
            relative = Path(path).resolve().relative_to(Path(source_root).resolve())
        except ValueError:
            pass
        else:
            candidate = Path(local_root).resolve() / relative
            if candidate.is_file():
                return str(candidate)
    if legacy_local_root:
        marker = "media/videos/"
        at = path.find(marker)
        if at != -1:
            candidate = os.path.join(
                legacy_local_root, path[at + len(marker):])
            if os.path.exists(candidate):
                return candidate
    return path


def load_decision_replay(paths: list[Path]) -> dict[str, str]:
    """Load exact pre-decision input -> frozen interaction-model output pairs from trace files.

    This is an evaluation-only variance control.  A candidate turn is replayed
    only when its complete text+ordered-image fingerprint is byte-identical to
    a contemporary control turn.  Any Harness intervention changes that hash
    and therefore still invokes the interaction model normally.  Conflicting outputs for the
    same input are rejected instead of choosing one silently.
    """
    replay: dict[str, str] = {}
    origins: dict[str, str] = {}
    for path in paths:
        for line_number, raw in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            for step in row.get("steps") or []:
                fingerprint = ((step.get("decision_input") or {})
                               .get("input_sha256"))
                if not fingerprint or "pred_raw" not in step:
                    continue
                output = str(step.get("pred_raw") or "")
                if len(output) >= 1000:
                    raise ValueError(
                        f"{path}:{line_number}: replay output may be trace-truncated")
                origin = f"{path}:{line_number}:{step.get('turn')}"
                previous = replay.get(fingerprint)
                if previous is not None and previous != output:
                    raise ValueError(
                        "conflicting frozen outputs for decision input "
                        f"{fingerprint}: {origins[fingerprint]} vs {origin}")
                replay[fingerprint] = output
                origins[fingerprint] = origin
    return replay


def resolve_state_slots(default: set[str], by_profile: dict[str, set[str]],
                        by_criterion: dict[str, set[str]], profile: str,
                        active_criteria=(), *, question_pending: bool = False,
                        initial_presence_gated_criteria=(),
                        initial_criteria=(), criterion_profiles=None,
                        compose_active_criteria: bool = False) -> set[str]:
    """Resolve prompt-visible slots from causal request semantics only.

    Profile slots are the base projection.  Typed-criterion slots are additive
    and let a narrowly applicable B/P/E representation follow an MCQ, event
    watch, or later scene question without broadening every fallback request.
    """
    slots = set(by_profile.get(profile, default))
    criteria = set(active_criteria)
    if question_pending:
        criteria.add("question")
    initial_gated = set(initial_presence_gated_criteria)
    initial = set(initial_criteria)
    for criterion in sorted(criteria):
        if criterion in initial_gated and criterion not in initial:
            continue
        if compose_active_criteria:
            criterion_profile = (criterion_profiles or {}).get(criterion)
            if criterion_profile is not None:
                slots.update(by_profile.get(str(criterion_profile), ()))
        slots.update(by_criterion.get(criterion, ()))
    return slots


def request_criteria(text: str) -> tuple[set[str], bool]:
    """Typed criteria visible at the current causal instruction event."""
    operations = parse_instruction(text or "", first=True)
    return ({op.target for op in operations if op.kind == "add"},
            any(op.kind == "question" for op in operations))


def active_criteria_within_limit(active_count: int, maximum: int) -> bool:
    """Apply the generic observer budget; zero means no configured limit."""
    return maximum <= 0 or active_count <= maximum


def criterion_adaptive_experience_projection(
        profile: str, active_count: int, default: str,
        by_profile: dict[str, str], compact_when_multiple_active: bool) -> str:
    """Choose pre-decision Experience detail from causal criterion count."""
    if compact_when_multiple_active and active_count > 1:
        return "answer_only"
    return by_profile.get(profile, default)


def question_facet(request: str) -> str:
    """Benchmark-blind semantic evidence target for an MCQ request."""
    value = re.sub(r"\s+", " ", str(request or "")).lower()
    value = re.split(r"\boptions?\s*:", value, maxsplit=1)[0]
    if "question:" in value:
        value = value.rsplit("question:", 1)[1]
    if re.search(
            r"\b(help|assist(?:ance)?|advice|interven(?:e|tion)|support)\b|"
            r"\bwhat should (?:i|we|you) (?:say|tell|suggest|recommend)\b",
            value):
        return "assistance"
    if re.search(
            r"\b(intent|goal|purpose|plan(?:ning)?|next action|next step)\b|"
            r"\b(?:likely|going|about) to (?:do|try|make|change|use)\b|"
            r"\bimmediate(?:ly)?\b.{0,24}\b(?:goal|action|intent)\b|"
            r"\bshort[- ]term\b",
            value):
        return "immediate_intent"
    if re.search(
            r"\b(current|currently|present|visible|observed)\b.{0,40}"
            r"\b(state|status|behavio(?:u)?r|action|activity|doing|happening)\b|"
            r"\b(state|status|behavio(?:u)?r|action|activity)\b.{0,40}"
            r"\b(current|currently|present|visible|observed)\b|"
            r"\bwhat (?:is|are) .{0,32}\bdoing\b|\bwhat is happening\b",
            value):
        return "observed_state"
    return "general"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-path", type=Path, default=None)
    ap.add_argument("--task", required=True)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--trace-output", type=Path, default=None)
    ap.add_argument("--latency-output", type=Path, default=None)
    ap.add_argument("--arm", choices=("harness", "short_only", "sliding", "cumulative",
                                       "original_32", "harness_32"),
                    default="harness")
    ap.add_argument("--system-prompt-profile", choices=tuple(SYSTEM_PROMPTS),
                    default="streamvibe")
    ap.add_argument("--print-system-prompt", action="store_true")
    ap.add_argument("--Ts", type=int, default=200)
    ap.add_argument("--M", type=int, default=5)
    ap.add_argument("--L", type=int, default=42)
    ap.add_argument("--lookahead", type=int, default=None)
    ap.add_argument("--memory-agent", default="vllm",
                    choices=("vllm", "fake", "none"))
    ap.add_argument("--memory-url", default=None)
    ap.add_argument("--memory-model", default=None)
    ap.add_argument("--memory-max-frames", type=int, default=64)
    ap.add_argument("--mid-max-tokens", type=int, default=512)
    ap.add_argument("--long-max-tokens", type=int, default=512)
    ap.add_argument("--query-oneshot", action="store_true",
                    help="Deactivate a query as soon as it is answered once. "
                         "Ablation; the default keeps it active, because these "
                         "suites' queries are standing instructions.")
    ap.add_argument("--query-mode", choices=("auto", "standing", "oneshot"),
                    default="standing", help="Request lifecycle; auto keeps guidance "
                    "requests active and closes ordinary QA after one answer.")
    ap.add_argument("--memory-style", choices=tuple(MEMORY_STYLES), default="evidence",
                    help='Mid-term summary format: evidence text or typed fact-table slots.')
    ap.add_argument("--skill-dir", type=Path, default=None,
                    help="Directory of SKILL.md files for the experience slot "
                         "(default: streamvibe_harness/skills). E-stage candidates run "
                         "from a copy so the active set is untouched.")
    ap.add_argument("--visual-window", type=int, default=32,
                    help="Seconds of visual tail the model sees each step (harness_32 "
                         "/ original_32 arms). Pre-decision lever.")
    ap.add_argument("--visual-stride", type=int, default=1,
                    help="Keep every S-th frame of the tail; the current frame is "
                         "always kept. 1 = every frame.")
    ap.add_argument("--visual-anchor", choices=("on", "off"), default="off",
                    help="Also keep the frames of the last committed responses, so "
                         "the model can compare the prior state with the current state.")
    ap.add_argument("--cadence-gate", choices=("on", "off"), default="off",
                    help="Deterministic commit gate driven by the active help type's "
                         "calibrated cadence (policies/cadence.json). Suppresses a "
                         "proposed response that is too early, too soon after the last "
                         "one, or near-duplicate. No model call.")
    ap.add_argument("--card", type=str, default="none",
                    help='Harness-state card in the chunk header. `shell` renders empty slots for ablation; epoch_resets should remain zero.')
    ap.add_argument("--force-answer-at", choices=("none", "last"), default="none",
                    help="Force `</response>` at the final step for evaluation "
                         "with a fixed answer position.")
    ap.add_argument("--history-silence", choices=("drop", "keep"), default="drop",
                    help="Whether to replay silent assistant turns. `drop` keeps "
                         "frames and spoken responses; `keep` also retains "
                         "silence tokens in the conversation history.")
    ap.add_argument("--video-local-root", default=os.environ.get("STREAMVIBE_VIDEO_LOCAL"),
                    help="Node-local copy of data/videos. Paths under "
                         "'media/videos/' are remapped into it when the file is "
                         "present, so ffmpeg seeks hit local disk instead of NFS.")
    ap.add_argument(
        "--video-source-root", default=os.environ.get("STREAM_VIDEO_SOURCE_ROOT"),
        help="Optional source prefix for a byte-identical node-local evaluation "
             "video tree; use together with --video-remap-root.")
    ap.add_argument(
        "--video-remap-root", default=os.environ.get("STREAM_VIDEO_LOCAL_ROOT"),
        help="Node-local replacement prefix for --video-source-root. Missing "
             "local files safely fall back to their original paths.")
    ap.add_argument("--max-tokens", type=int, default=128,
                    help='Maximum generated tokens per decision step. Longer outputs can increase batch latency.')
    ap.add_argument("--memory-parts", choices=("all", "summaries", "said"),
                    default="all",
                    help='Select visual summaries, delivered-response history, or both for the memory context.')
    ap.add_argument("--said-keep", type=int, default=8)
    ap.add_argument("--response-verifier", choices=("none", "agent"), default="none",
                    help="Optionally ground each proposed response with the "
                         "separate memory VLM before committing it. Silent "
                         "steps are never sent to the verifier.")
    ap.add_argument("--verify-frames", type=int, default=4,
                    help="Most recent chronological frames shown to the optional "
                         "response verifier.")
    ap.add_argument("--verifier-action-policy",
                    choices=("preserve", "allow_silence"), default="allow_silence",
                    help="preserve keeps the base model's speak decision and lets "
                         "the verifier rewrite content only; allow_silence also "
                         "permits the verifier to suppress an ungrounded response.")
    ap.add_argument("--novelty-rescue", action="store_true",
                    help="Restore a verifier-rejected proposal only when it "
                         "follows a quiet interval and is lexically novel versus "
                         "recent committed responses.")
    ap.add_argument("--novelty-threshold", type=float, default=0.6)
    ap.add_argument("--novelty-min-gap", type=int, default=8)
    ap.add_argument("--novelty-history", type=int, default=8)
    ap.add_argument("--continual-refiner", choices=("off", "shadow", "adaptive"),
                    default="off",
                    help="Detect trajectory failure signatures and emit versioned "
                         "harness CRUD proposals. shadow records proposals without "
                         "changing serving; adaptive may apply only bounded novelty "
                         "controller edits.")
    ap.add_argument("--refine-every", type=int, default=64)
    ap.add_argument("--refine-warmup", type=int, default=64)
    ap.add_argument("--harness-policy", type=Path, default=None,
                    help="StreamVibe Harness evidence/intervention/response policy JSON. "
                         "Model weights and memory geometry are never changed by it.")
    ap.add_argument("--predecision-only", action="store_true",
                    help="Hard serving invariant: all harness influence must be in "
                         "the context before the online model call. Disables response "
                         "verifier/rewrite, novelty rescue, cadence suppression and "
                         "online continual adaptation even if a policy enables them.")
    ap.add_argument("--decision-replay-trace", type=Path, action="append", default=[],
                    help='Evaluation-only exact-input replay source. Turns whose complete pre-decision text+image fingerprint matches a control trace reuse that frozen interaction-model output; changed inputs still invoke the interaction model. Repeat for multiple shard traces.')
    ap.add_argument("--memory-replay-input", type=Path, default=None,
                    help='Evaluation-only frozen pre-decision cognitive-agent transcript. Every mid/long/plan/event input signature must match.')
    ap.add_argument("--memory-replay-output", type=Path, default=None,
                    help='Record pre-decision cognitive-agent products for a later paired candidate. Mutually exclusive with --memory-replay-input.')
    ap.add_argument("--frame-cache-size", type=int, default=None,
                    help='Maximum decoded frames cached in memory. Zero is unbounded; long-video defaults depend on chunk duration.')
    ap.add_argument("--frame-disk-purge", type=lambda v: v.lower() == "true",
                    default=None,
                    help="true/false; delete extracted JPEGs after each batch. "
                         "Default: on for streamified suites, off otherwise.")
    ap.add_argument("--images-per-step", type=int, default=1)
    ap.add_argument("--lanes", type=int, default=8)
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.72)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--eval-id", action="append", default=[],
                    help="Restrict to explicit trajectory IDs before sharding; "
                    "used for matched short smoke cohorts.")
    ap.add_argument("--sample-offset", type=int, default=0,
                    help="Skip this many post-shard rows before max-samples; used "
                         "for disjoint held-out canaries.")
    ap.add_argument("--max-turns", type=int, default=0,
                    help="Stop every lane after this many seconds. Smoke tests only.")
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--no-prefix-caching", action="store_true",
                    help="Disable prefix caching for StreamVibe Harness.")
    args = ap.parse_args()

    from config import MODEL_PATHS, REPO  # noqa: E402

    os.chdir(REPO)
    model_path = args.model_path or MODEL_PATHS[args.model]
    policy = HarnessPolicy.load(args.harness_policy)
    policy.apply_runtime(args)
    decision_replay = load_decision_replay(args.decision_replay_trace)
    if decision_replay:
        print(f"[harness] exact decision replay entries={len(decision_replay)} "
              f"sources={len(args.decision_replay_trace)}", flush=True)
    if args.predecision_only:
        args.response_verifier = "none"
        args.verifier_action_policy = "preserve"
        args.novelty_rescue = False
        args.cadence_gate = "off"
        args.continual_refiner = "off"
    system_prompt = SYSTEM_PROMPTS[args.system_prompt_profile] + policy.response_suffix()
    parsed_card = parse_mode(args.card)  # raises on unknown slots
    card_enabled = parsed_card is not None
    card_slots = parsed_card or set()
    cadence_table = load_cadence() if args.cadence_gate == "on" else {}
    if cadence_table:
        print(f"[harness] cadence gate on: {sorted(cadence_table)}", flush=True)
    skill_config = ((policy.evidence_policy or {}).get("skills") or {})
    skill_top_k = int(skill_config.get("retrieval_top_k", 2))
    skill_grounded_only = bool(skill_config.get("grounded_only", False))
    skill_min_grounding_overlap = int(
        skill_config.get("min_grounding_overlap", 0))
    skill_profile_compatibility = {
        str(profile): tuple(str(name) for name in names)
        for profile, names in (
            skill_config.get("profile_compatibility") or {}).items()
    }
    require_general_skills = bool(
        skill_config.get("general_taxonomy", False))
    experience_projection_default = str(
        skill_config.get("prompt_projection", "legacy"))
    experience_projection_by_profile = {
        str(profile): str(projection)
        for profile, projection in (
            skill_config.get("prompt_projection_by_profile") or {}).items()
    }
    experience_projection_by_question_facet = {
        str(facet): str(projection)
        for facet, projection in (
            skill_config.get("prompt_projection_by_question_facet") or {}).items()
    }
    typed_question_projection = bool(
        skill_config.get("typed_question_projection", False))
    execution_state_projection_excluded_profiles = {
        str(profile) for profile in (
            skill_config.get("execution_state_projection_excluded_profiles")
            or [])
    }
    # Keep ``answer_only`` in the public JSON for compatibility with legacy
    # bundle validators, while recording the effective state-centric
    # projection in runtime traces and audits.
    if bool(skill_config.get("execution_state_projection", False)):
        experience_projection_default = "execution_state"
        experience_projection_by_profile = {
            profile: (projection
                      if profile in execution_state_projection_excluded_profiles
                      else ("execution_state"
                            if profile != "question_answering"
                            or typed_question_projection else "answer_only"))
            for profile, projection in experience_projection_by_profile.items()
        }
        # Question facets select answer realization only after the user has
        # explicitly opened a question.  They are not evidence to speak and do
        # not replace the interaction model's action decision.
        experience_projection_by_question_facet = {
            facet: "answer_only"
            for facet in experience_projection_by_question_facet
        }
    compact_experience_when_multiple_active = bool(
        skill_config.get("compact_when_multiple_active", False))

    def experience_projection_for(
            profile: str, active_count: int, request_text: str = "",
            active_criteria: set[str] | frozenset[str] = frozenset()) -> str:
        projection = criterion_adaptive_experience_projection(
            profile, active_count, experience_projection_default,
            experience_projection_by_profile,
            compact_experience_when_multiple_active)
        if compact_experience_when_multiple_active and active_count > 1:
            return projection
        # Typed question routing is opt-in.  When enabled, a closed-choice
        # request gets the concise answer-realisation schema while open QA
        # keeps state evidence.  Route by criterion, never benchmark identity.
        if (typed_question_projection and profile == "question_answering"
                and closed_choice_request(active_criteria)):
            projection = experience_projection_by_question_facet.get(
                question_facet(request_text), projection)
        return projection
    response_generation_config = dict(
        (policy.response_policy or {}).get("generation") or {})
    response_generation_default = str(
        response_generation_config.get("mode", "joint"))
    response_generation_by_profile = {
        str(profile): str(mode)
        for profile, mode in (
            response_generation_config.get("mode_by_profile") or {}).items()
    }
    response_content_projection = str(
        response_generation_config.get("content_projection", "answer_only"))

    def response_generation_for(profile: str) -> str:
        return response_generation_by_profile.get(
            profile, response_generation_default)
    event_config = ((policy.evidence_policy or {}).get("event_parser") or {})
    event_enabled = bool(event_config.get("enabled", False))
    event_backend = str(event_config.get("backend", "agent"))
    event_prompt_profile = str(event_config.get("prompt_profile", "base"))
    event_profiles = set(event_config.get("profiles") or [])
    # Missing means unlimited.  Policy validation permits configured limits
    # from 1 through 9; zero remains the internal backwards-compatible
    # sentinel for an omitted limit.
    event_max_active_criteria = int(
        event_config.get("max_active_criteria", 0))
    event_stride = max(1, min(8, int(event_config.get("stride", 1))))
    event_frames = max(1, min(4, int(event_config.get("frames", 2))))
    event_window = max(1, min(32, int(event_config.get("window", event_frames))))
    event_window_by_skill = {
        str(name): max(1, min(32, int(value)))
        for name, value in (event_config.get("window_by_skill") or {}).items()
    }
    event_render_mode = str(event_config.get("render_mode", "raw"))
    event_repeat_gap = max(0, min(60, int(event_config.get("repeat_gap", 8))))
    event_repeat_jaccard = max(
        0.0, min(1.0, float(event_config.get("repeat_jaccard", 0.55))))
    event_end_confirmations = max(
        1, int(event_config.get("end_confirmations", 2)))


    event_instance_input_gating = bool(event_config.get(
        "instance_input_gating",
        event_config.get("rearm_on_reappearance_only", False)))
    commit_nonempty_only = bool(
        (policy.response_policy or {}).get("commit_nonempty_only", False))
    decision_clock_enabled = bool(
        ((policy.evidence_policy or {}).get("decision_clock") or {})
        .get("enabled", False))
    transition_config = ((policy.evidence_policy or {}).get("transition_state") or {})
    transition_enabled = bool(transition_config.get("enabled", False))
    transition_profiles = set(transition_config.get("profiles") or [])
    transition_minor = float(transition_config.get("minor_threshold", 0.025))
    transition_major = float(transition_config.get("major_threshold", 0.14))
    transition_edge_gap = int(transition_config.get("edge_gap", 3))
    render_visual_transition_deltas = bool(
        transition_config.get("render_visual_deltas", True))
    render_ordered_observation_deltas = bool(
        transition_config.get("render_ordered_observation_deltas", True))
    render_request_lifecycle_deltas = bool(
        transition_config.get("render_request_deltas", True))
    render_confirmed_evidence_deltas = bool(
        transition_config.get("render_confirmed_evidence_deltas", True))
    render_procedure_deltas = bool(
        transition_config.get("render_procedure_deltas", True))
    procedure_delta_mode = str(
        transition_config.get("procedure_delta_mode", "lifecycle"))
    render_response_lifecycle_deltas = bool(
        transition_config.get("render_response_deltas", True))
    context_config = ((policy.evidence_policy or {}).get("context_projection") or {})
    state_slots_default = set(context_config.get("state_slots") or card_slots)
    state_slots_by_profile = {
        str(profile): set(slots)
        for profile, slots in (context_config.get("state_slots_by_profile") or {}).items()
    }
    state_slots_by_criterion = {
        str(criterion): set(slots)
        for criterion, slots in (
            context_config.get("state_slots_by_criterion") or {}).items()
    }
    belief_query_mode_by_criterion = dict(
        context_config.get("belief_query_mode_by_criterion") or {})
    criterion_projection_initial_gate = set(
        context_config.get(
            "criterion_projection_requires_initial_presence") or [])
    control_role = str(
        context_config.get("control_role", "user"))
    header_memory_projection = str(
        (((policy.evidence_policy or {}).get("context_projection") or {})
         .get("header_memory", "all")))
    header_memory_by_profile = dict(
        (((policy.evidence_policy or {}).get("context_projection") or {})
         .get("header_memory_by_profile") or {}))
    header_memory_by_question_facet = dict(
        context_config.get("header_memory_by_question_facet") or {})
    minimal_state_if_redundant_by_question_facet = {
        str(facet): str(cue)
        for facet, cue in (
            context_config.get(
                "minimal_state_if_redundant_by_question_facet") or {}).items()
        if str(cue).strip()
    }
    event_lifetime = str(
        (((policy.evidence_policy or {}).get("context_projection") or {})
         .get("event_lifetime", "append")))
    decision_contract = str(
        (((policy.evidence_policy or {}).get("context_projection") or {})
         .get("decision_contract", "none")))
    response_history_mode = str(
        (((policy.evidence_policy or {}).get("context_projection") or {})
         .get("response_history", "full")))
    response_history_by_profile = dict(
        (((policy.evidence_policy or {}).get("context_projection") or {})
         .get("response_history_by_profile") or {}))
    silence_history_mode = str(
        (((policy.evidence_policy or {}).get("context_projection") or {})
         .get("silence_history",
              "all" if args.history_silence == "keep" else "drop")))
    silence_history_by_profile = dict(
        (((policy.evidence_policy or {}).get("context_projection") or {})
         .get("silence_history_by_profile") or {}))
    silence_history_by_criterion = dict(
        (((policy.evidence_policy or {}).get("context_projection") or {})
         .get("silence_history_by_criterion") or {}))
    silence_history_focus_gate = set(
        (((policy.evidence_policy or {}).get("context_projection") or {})
         .get("silence_history_requires_specific_focus") or []))
    prompt_router = dict(
        (((policy.evidence_policy or {}).get("context_projection") or {})
         .get("prompt_router") or {}))
    compose_active_criteria = bool(
        prompt_router.get("compose_active_criteria", False))
    temporary_question_profile = bool(
        prompt_router.get("temporary_question_profile", False))
    response_cooldown = dict(
        (((policy.evidence_policy or {}).get("context_projection") or {})
         .get("response_cooldown") or {}))
    response_epoch = dict(
        (((policy.evidence_policy or {}).get("context_projection") or {})
         .get("response_epoch") or {}))
    response_epoch_enabled = bool(response_epoch.get("enabled", False))
    response_epoch_profiles = set(response_epoch.get("profiles") or [])
    response_epoch_lookback = int(response_epoch.get("lookback_turns", 0))
    response_epoch_lookback_by_profile = dict(
        response_epoch.get("lookback_by_profile") or {})
    response_epoch_evidence_lookback = int(
        response_epoch.get("evidence_lookback_turns", 0))
    response_epoch_evidence_lookback_by_profile = dict(
        response_epoch.get("evidence_lookback_by_profile") or {})
    response_epoch_trigger = str(response_epoch.get("trigger", "every_response"))
    response_epoch_burst_gap = int(response_epoch.get("burst_gap_turns", 1))
    response_epoch_repeat_threshold = float(
        response_epoch.get("repeat_threshold", 0.9))
    response_epoch_repeat_history = int(
        response_epoch.get("repeat_history", 8))
    response_epoch_repeat_gap = int(
        response_epoch.get("repeat_gap_turns", 16))
    response_epoch_trigger_by_profile = dict(
        response_epoch.get("trigger_by_profile") or {})
    response_epoch_burst_gap_by_profile = dict(
        response_epoch.get("burst_gap_by_profile") or {})
    for routed_name in (prompt_router.get("profiles") or {}).values():
        if routed_name not in SYSTEM_PROMPTS:
            raise ValueError(f"unknown routed system prompt profile {routed_name!r}")
    cooldown_profile = str(response_cooldown.get("profile") or "")
    if cooldown_profile and cooldown_profile not in SYSTEM_PROMPTS:
        raise ValueError(f"unknown response cooldown prompt profile {cooldown_profile!r}")
    skills_enabled = bool(skill_config.get("enabled", True))
    # E_t is external state, not merely prompt prose.  Keep it active whenever
    # the policy enables Skills even when the projection is an empty shell:
    # request routing and memory focus may use the recalled Skill privately,
    # while the compiler remains free to expose no copyable Skill text to the interaction model.
    experience_store = ((ExperienceStore.load(
                            args.skill_dir, max_render=skill_top_k,
                            require_general_taxonomy=require_general_skills,
                            grounded_only=skill_grounded_only,
                            min_grounding_overlap=skill_min_grounding_overlap,
                            profile_compatibility=skill_profile_compatibility)
                         if args.skill_dir
                         else ExperienceStore.load(
                             max_render=skill_top_k,
                             require_general_taxonomy=require_general_skills,
                             grounded_only=skill_grounded_only,
                             min_grounding_overlap=skill_min_grounding_overlap,
                             profile_compatibility=skill_profile_compatibility))
                        if skills_enabled else None)
    if experience_store is not None:
        print(f"[harness] experience skills={sorted(experience_store.skills)} "
              f"dir={args.skill_dir or 'default'}", flush=True)

    if args.print_system_prompt:
        print(system_prompt)

    cfg = MemoryConfig(Ts=args.Ts, M=args.M, L=args.L, lookahead=args.lookahead,
                       said_keep=args.said_keep)
    agent_kind = (args.memory_agent
                  if args.arm in ("harness", "harness_32")
                  or args.response_verifier == "agent" else "none")
    agent_kw = {}
    if agent_kind == "vllm":
        agent_kw = dict(base_url=args.memory_url, model=args.memory_model,
                        mid_max_tokens=args.mid_max_tokens,
                        long_max_tokens=args.long_max_tokens,
                        max_frames=args.memory_max_frames,
                        workers=max(4, args.lanes),
                        memory_style=args.memory_style)
    agent = build_agent(agent_kind, **agent_kw)
    if args.memory_replay_input is not None or args.memory_replay_output is not None:
        agent = MemoryTranscriptAgent(
            agent, replay_input=args.memory_replay_input,
            record_output=args.memory_replay_output)
    if hasattr(agent, "health"):
        print(f"[harness] memory agent OK: {agent.health()}", flush=True)

    rows = load_rows(args.input)
    if args.eval_id:
        selected = set(args.eval_id)
        rows = [row for row in rows if str(row.get("eval_id")) in selected]
    if args.num_shards > 1:
        rows = rows[args.shard_id :: args.num_shards]
    if args.sample_offset:
        rows = rows[args.sample_offset:]
    if args.max_samples:
        rows = rows[: args.max_samples]

    print(f"[harness] model={args.model} arm={args.arm} prompt={args.system_prompt_profile} "
          f"Ts={cfg.Ts} M={cfg.M} L={cfg.L} lookahead={cfg.effective_lookahead} "
          f"coverage={cfg.coverage_s}s agent={agent_kind} rows={len(rows)} "
          f"memory_transcript={getattr(agent, 'mode', 'off')} "
          f"silence_in_history={args.history_silence} "
          f"verifier={args.response_verifier} novelty_rescue={args.novelty_rescue} "
          f"continual_refiner={args.continual_refiner} policy={policy.name}",
          flush=True)

    inference_backend = os.environ.get("STREAMVIBE_INFERENCE_BACKEND", "vllm").strip().lower()
    from transformers import AutoProcessor  # noqa: E402

    # Load RGB frames and apply the selected backend's image preprocessing.
    # The Hugging Face backend uses the checkpoint processor's image geometry.
    if inference_backend == "vllm":
        # Install the optional verl dependency into the active environment.
        from verl.utils.dataset import process_image  # noqa: E402
    else:
        from PIL import Image
        from qwen_vl_utils import smart_resize

        def process_image(path, min_pixels, max_pixels):
            with Image.open(path) as source:
                image = source.convert("RGB")
                width, height = image.size
                resized_height, resized_width = smart_resize(
                    height, width, factor=28, min_pixels=min_pixels,
                    max_pixels=max_pixels)
                if (resized_width, resized_height) != (width, height):
                    image = image.resize((resized_width, resized_height))
                return image.copy()

    # A streamified row is a whole video at 1 FPS -- up to 3,600 frames. The
    # existing streaming rollout keeps every decoded frame in an unbounded dict,
    # which is fine for a 100-400 frame clip and is roughly 700 MB PER LANE at
    # 3,600. So the cache is bounded whenever the suite is one of the long ones,
    # and left exactly as it was otherwise: a decoded frame is only ever needed
    # again while it is inside short-term memory or inside the chunk the
    # mid-term agent is summarising, both of which are Ts-bounded.
    streamified = any(r.get("stream_spec") for r in rows)
    if args.frame_cache_size is not None:
        cache_size = args.frame_cache_size
    elif args.arm == "cumulative" or not streamified:
        cache_size = 0  # unbounded: historical behaviour for the streaming suites
    else:
        cache_size = max(
            1024, 3 * (cfg.Ts + cfg.effective_lookahead) * args.lanes
            * max(1, args.images_per_step))

    frame_cache: "OrderedDict[str, object]" = OrderedDict()
    frame_evictions = 0
    local_frame_root = Path(os.environ.get("STREAMVIBE_FRAME_CACHE", "outputs/cache/harness_frames"))
    # Shared across every process in the job, NOT per-PID. A sweep runs the same
    # rows through a dozen configurations; keyed by PID each one re-extracts the
    # identical 64,000 JPEGs from scratch. The write is tmp+atomic-rename with a
    # PID-unique tmp name, so concurrent extractors of the same frame are safe.
    local_frame_root = local_frame_root / str(os.environ.get("SLURM_JOB_ID", "local"))

    # ffmpeg seeking into a 1080p mp4 over NFS costs ~0.9 s per frame and is the
    # rollout's actual bottleneck (measured: 7 s per 8-lane step at turn 25, where
    # the GPU work is one cached-prefix frame). The streaming video trees are 11.6 GB
    # total, so the job copies them to node-local disk and paths are remapped here.
    video_local_root = args.video_local_root
    video_source_root = args.video_source_root
    video_remap_root = args.video_remap_root
    def local_video(path: str) -> str:
        return remap_video_path(
            path, legacy_local_root=video_local_root or "",
            source_root=video_source_root or "", local_root=video_remap_root or "")
    purge_disk = (args.frame_disk_purge if args.frame_disk_purge is not None
                  else streamified)
    print(f"[harness] frame cache: in-process max={cache_size or 'unbounded'} "
          f"disk={local_frame_root} purge_per_batch={purge_disk}", flush=True)

    def materialize_frame(path: str) -> str:
        if not path.startswith("video-frame::"):
            return path
        _, second, video_path = path.split("::", 2)
        video_path = local_video(video_path)
        key = hashlib.sha256(path.encode()).hexdigest()
        target = local_frame_root / f"{key}.jpg"
        if target.exists():
            return str(target)
        local_frame_root.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(f".{os.getpid()}.tmp.jpg")
        first = subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-ss", second, "-i", video_path,
             "-frames:v", "1", "-q:v", "2", "-y", str(tmp)], check=False)
        if first.returncode != 0 or not tmp.is_file():
            subprocess.run(
                ["ffmpeg", "-nostdin", "-loglevel", "error", "-sseof", "-1",
                 "-i", video_path, "-frames:v", "1", "-q:v", "2", "-y", str(tmp)],
                check=True)
        if not tmp.is_file():
            raise RuntimeError(f"ffmpeg produced no frame for {path}")
        tmp.replace(target)
        return str(target)

    def load_frame(path: str):
        nonlocal frame_evictions
        hit = frame_cache.get(path)
        if hit is not None:
            frame_cache.move_to_end(path)
            return hit
        local_path = materialize_frame(path)
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"frame {local_path} missing; cwd={os.getcwd()}")
        image = process_image(local_path, MIN_PIXELS, MAX_PIXELS)
        frame_cache[path] = image
        if cache_size and len(frame_cache) > cache_size:
            frame_cache.popitem(last=False)
            frame_evictions += 1
        return image

    def purge_frame_disk() -> None:
        """Delete extracted frames after their batch completes.
        
        Cleanup bounds local disk use during long-video runs. Cleanup failures are
        non-fatal so completed trajectories remain available.
        """
        if not purge_disk or not local_frame_root.is_dir():
            return
        freed = 0
        try:
            for f in local_frame_root.iterdir():
                if f.suffix in (".jpg", ".tmp"):
                    freed += f.stat().st_size
                    f.unlink(missing_ok=True)
        except OSError as exc:
            print(f"[harness] frame purge failed (continuing): {exc}", flush=True)
        if freed:
            print(f"[harness] purged {freed / 1e9:.2f} GB of extracted frames",
                  flush=True)

    proc = AutoProcessor.from_pretrained(str(model_path))
    template_options = ({"enable_thinking": False}
                        if inference_backend == "hf" else {})

    inference_backend = os.environ.get("STREAMVIBE_INFERENCE_BACKEND", "vllm").strip().lower()
    if inference_backend == "vllm":
        from vllm import LLM, SamplingParams  # noqa: E402
    elif inference_backend == "hf":
        import torch
        from transformers import AutoModelForImageTextToText

        class SamplingParams:  # minimal interface consumed below
            def __init__(self, *, n=1, temperature=0.0, max_tokens=128,
                         detokenize=True, skip_special_tokens=False):
                self.n = n
                self.temperature = temperature
                self.max_tokens = max_tokens
                self.detokenize = detokenize
                self.skip_special_tokens = skip_special_tokens

        class HFLLM:
            """Inference backend adapter for the shared harness runtime."""

            def __init__(self, model_name: str):
                attn = os.environ.get("STREAMVIBE_HF_ATTN", "sdpa")
                self.model = AutoModelForImageTextToText.from_pretrained(
                    model_name, dtype=torch.bfloat16, device_map="cuda:0",
                    attn_implementation=attn, local_files_only=True).eval()
                # Qwen3.5 otherwise treats only end-of-text as EOS and can
                # continue by inventing another user/assistant turn after the
                # chat-template end marker.  The legacy the interaction model runtime stopped at
                # this boundary; restore that decoding contract before any
                # Harness state sees the output.
                stop_ids = self.model.generation_config.eos_token_id
                stop_ids = (list(stop_ids) if isinstance(stop_ids, (list, tuple))
                            else ([stop_ids] if stop_ids is not None else []))
                for token in (proc.tokenizer.eos_token_id,
                              proc.tokenizer.convert_tokens_to_ids("<|im_end|>")):
                    if token is not None and token != proc.tokenizer.unk_token_id and token not in stop_ids:
                        stop_ids.append(token)
                self.model.generation_config.eos_token_id = stop_ids

            @staticmethod
            def _truncate_chat_tail(value: str) -> str:
                cuts = [value.find(token) for token in
                        ("<|im_end|>", "<|endoftext|>", "<|im_start|>")]
                cuts = [cut for cut in cuts if cut >= 0]
                return value[:min(cuts)].strip() if cuts else value.strip()

            def generate(self, requests=None, sampling_params=None, **kwargs):
                requests = requests if requests is not None else kwargs.get("prompts", [])
                results = []
                for request in requests:
                    images = (request.get("multi_modal_data") or {}).get("image") or None
                    batch = proc(
                        text=[request["prompt"]], images=images, padding=True,
                        return_tensors="pt").to("cuda:0")
                    prompt_ids = batch["input_ids"][0].tolist()
                    with torch.inference_mode():
                        generated = self.model.generate(
                            **batch,
                            max_new_tokens=max(1, int(sampling_params.max_tokens)),
                            do_sample=False)
                    continuation = generated[0][batch["input_ids"].shape[1]:]
                    value = proc.decode(
                        continuation,
                        skip_special_tokens=bool(sampling_params.skip_special_tokens))
                    value = self._truncate_chat_tail(value)
                    results.append(SimpleNamespace(
                        outputs=[SimpleNamespace(text=value)],
                        prompt_token_ids=prompt_ids,
                        num_cached_tokens=0))
                    del batch, generated, continuation
                return results
    else:
        raise ValueError(f"unsupported STREAMVIBE_INFERENCE_BACKEND={inference_backend!r}")

    # The staged contract depends on one greedy decode step being exactly the
    # model's learned action token. Refuse to run a checkpoint whose tokenizer
    # does not carry both controls as single vocabulary items.
    control_token_ids = {}
    for control_token in sorted(ACTION_TOKENS):
        token_ids = proc.tokenizer.encode(
            control_token, add_special_tokens=False)
        if len(token_ids) != 1:
            raise RuntimeError(
                f"{control_token} must be one tokenizer token, got {token_ids}")
        control_token_ids[control_token] = int(token_ids[0])

    # Chunked context is bounded, so
    # the window no longer has to be sized for the longest clip in the shard.
    # Sized from the configuration instead: short-term frames at the pixel
    # budget actually in use, plus the memory text, plus the chunk's dialogue.
    tokens_per_frame = max(1, MAX_PIXELS // 1024)  # patch 16, spatial merge 2
    short_frames = (cfg.Ts + cfg.effective_lookahead) * max(1, args.images_per_step)
    if args.arm == "cumulative":
        default_len = 24576 + 128
    else:
        default_len = int(
            short_frames * tokens_per_frame
            + cfg.M * args.mid_max_tokens
            + cfg.L * args.long_max_tokens
            + short_frames * 24      # timestamps + the model's own replies
            + 2048)
        default_len = max(4096, min(131072, ((default_len + 1023) // 1024) * 1024))
    max_model_len = args.max_model_len or int(os.environ.get("CT_STREAM_MAX_LEN", 0)) or default_len

    shard_max_turns = max(
        (spec_turns(row) or len(_turn_records(row) or []) for row in rows), default=0)
    image_limit = (short_frames + 8 if args.arm != "cumulative"
                   else max(256, shard_max_turns + 8))
    print(f"[harness] tokens/frame={tokens_per_frame} short_frames={short_frames} "
          f"max_model_len={max_model_len} limit_mm.image={image_limit}", flush=True)

    if inference_backend == "vllm":
        llm = LLM(
            model=str(model_path), dtype="bfloat16", seed=1,
            max_model_len=max_model_len, tensor_parallel_size=1,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_num_batched_tokens=min(24672, max_model_len),
            enforce_eager=False,
            enable_prefix_caching=not args.no_prefix_caching,
            mm_processor_cache_gb=8,
            limit_mm_per_prompt={"image": image_limit, "video": 1},
            disable_custom_all_reduce=True,
        )
    else:
        llm = HFLLM(str(model_path))
    print(f"[harness] inference_backend={inference_backend}", flush=True)
    sp = SamplingParams(n=1, temperature=0.0, max_tokens=args.max_tokens,
                        detokenize=True, skip_special_tokens=False)
    action_sp = SamplingParams(
        n=1, temperature=0.0, max_tokens=1,
        detokenize=True, skip_special_tokens=False)
    content_sp = SamplingParams(
        n=1, temperature=0.0, max_tokens=max(1, args.max_tokens - 1),
        detokenize=True, skip_special_tokens=False)
    observer_sp = SamplingParams(
        n=1, temperature=0.0, max_tokens=96,
        detokenize=True, skip_special_tokens=False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    committed: set[str] = set()
    if args.trace_output and args.trace_output.is_file():
        for line in args.trace_output.read_text(encoding="utf-8").splitlines():
            try:
                committed.add(str(json.loads(line)["eval_id"]))
            except Exception:
                continue
        rows = [r for r in rows if str(r["eval_id"]) not in committed]
        print(f"[harness] resuming: {len(committed)} committed, {len(rows)} left", flush=True)

    # Length without expanding: a streamified row's length is in its spec, and
    # expanding 300 hour-long rows just to sort them would defeat the point of
    # the spec (see harness/streamspec.py).
    rows.sort(key=lambda r: spec_turns(r) or len(_turn_records(r) or []), reverse=True)

    latency: list[dict] = []
    written = 0
    trace_handle = (args.trace_output.open("a", encoding="utf-8")
                    if args.trace_output else None)
    with args.output.open("a", encoding="utf-8") as out:
        for start in range(0, len(rows), args.lanes):
            batch = rows[start : start + args.lanes]
            lanes = []
            for row in batch:
                row = expand(row)
                records = _turn_records(row)
                if not records:
                    continue
                golds = turn_table(row)
                if len(golds) != len(records):
                    raise ValueError(
                        f"{row.get('eval_id')}: turn_table {len(golds)} != records {len(records)}")
                probes = dense_probes(row)
                seconds = [g["second"] for g in golds]
                tracker = QueryTracker(
                    records, seconds, persist=not args.query_oneshot,
                    mode=("oneshot" if args.query_oneshot else args.query_mode))


                initial_query = tracker.queries.get(0)
                initial_request = initial_query.text if initial_query else ""
                initial_criteria, initial_question = request_criteria(initial_request)
                if prompt_router:
                    semantic_profile, routed_name = routed_state_profile(
                        initial_request, initial_criteria, initial_question,
                        prompt_router, args.system_prompt_profile,
                        temporary_question_profile=temporary_question_profile)
                else:
                    semantic_profile = (request_profile(initial_request)
                                        if response_epoch_profiles else "disabled")
                    routed_name = args.system_prompt_profile
                initial_silence_mode, initial_silence_scope = scoped_silence_history(
                    silence_history_mode, silence_history_by_profile,
                    silence_history_by_criterion, semantic_profile,
                    initial_criteria, silence_history_focus_gate,
                    initial_request,
                    criterion_projection_initial_gate, initial_criteria)
                lane_system_prompt = SYSTEM_PROMPTS[routed_name] + policy.response_suffix()
                novelty = NovelResponseGate(
                    args.novelty_threshold, args.novelty_min_gap,
                    args.novelty_history)
                lanes.append({
                    "row": row, "records": records, "golds": golds, "probes": probes,
                    "seconds": seconds,
                    "memory": ThreeTierMemory(
                        cfg, agent if args.arm in ("harness", "harness_32") else None,
                        lane_id=str(row["eval_id"]), parts_mode=args.memory_parts,
                        plan_config=(policy.evidence_policy or {}).get("plan_memory"),
                        strict_ordered_events=args.memory_style == "decision_ordered"),
                    "tracker": tracker,
                    "system_prompt": lane_system_prompt,
                    "system_prompt_profile": routed_name,
                    "system_prompt_sha256": hashlib.sha256(
                        lane_system_prompt.encode("utf-8")).hexdigest(),
                    "decision_prompt_profiles": {},
                    "decision_prompt_sha256": {},
                    "decision_input_fingerprints": {},
                    "decision_replayed": set(),
                    "response_generation_mode": response_generation_for(
                        semantic_profile),
                    "response_generation_steps": {},
                    "response_content_calls": 0,
                    "response_content_failures": 0,
                    "response_phases": {},
                    "prompt_route": {
                        "semantic_profile": semantic_profile,
                        "request": initial_request[:500],
                        "active_criteria": sorted(initial_criteria),
                        "question_pending": bool(initial_question),
                        "refresh": prompt_router.get("refresh", "initial_request"),
                        "events": [{
                            "turn": 0,
                            "second": float(seconds[0]),
                            "from": None,
                            "to": semantic_profile,
                            "request": initial_request[:500],
                        }],
                    },
                    "header_memory_projection": str(
                        header_memory_by_profile.get(
                            semantic_profile, header_memory_projection)),
                    "question_facet": question_facet(initial_request),
                    "memory_redundant_with_raw": False,
                    "minimal_state_projection": False,
                    "effective_header_memory_projection": str(
                        header_memory_by_profile.get(
                            semantic_profile, header_memory_projection)),
                    "response_history_mode": str(
                        response_history_by_profile.get(
                            semantic_profile, response_history_mode)),
                    "silence_history_mode": initial_silence_mode,
                    "silence_history_scope": initial_silence_scope,
                    "transition_state_effective": bool(
                        transition_enabled and (not transition_profiles
                                                or semantic_profile in transition_profiles)),
                    "response_epoch_enabled": bool(
                        response_epoch_enabled
                        and (not response_epoch_profiles
                             or semantic_profile in response_epoch_profiles)),
                    "response_epoch_lookback": int(
                        response_epoch_lookback_by_profile.get(
                            semantic_profile, response_epoch_lookback)),
                    "response_epoch_evidence_lookback": int(
                        response_epoch_evidence_lookback_by_profile.get(
                            semantic_profile,
                            response_epoch_evidence_lookback)),
                    "response_epoch_trigger": str(
                        response_epoch_trigger_by_profile.get(
                            semantic_profile, response_epoch_trigger)),
                    "response_epoch_burst_gap": int(
                        response_epoch_burst_gap_by_profile.get(
                            semantic_profile, response_epoch_burst_gap)),
                    "skills": ActiveHelpSkills(
                        (policy.evidence_policy or {}).get("skills")),
                    "turn_extras": {},
                    "current_system_extras": {},
                    "turn_silence_history_mode": {},
                    "turn_silence_history_start": {},
                    "silence_profile_start": 0,
                    "epoch_guard": EpochGuard(),
                    "epoch_log": [], "turn_epoch": {}, "turn_context_epoch": {},
                    "turn_context_start": {}, "turn_evidence_start": {},
                    "spoke_turns": [], "spoke_texts": [],
                    "adapter": HarnessAdapter(
                        slots=resolve_state_slots(
                            state_slots_default, state_slots_by_profile,
                            state_slots_by_criterion, semantic_profile,
                            initial_criteria,
                            question_pending=initial_question,
                            initial_presence_gated_criteria=(
                                criterion_projection_initial_gate),
                            initial_criteria=initial_criteria,
                            criterion_profiles=prompt_router.get(
                                "criteria_profiles") or {},
                            compose_active_criteria=compose_active_criteria),
                        enabled=card_enabled,
                        experience=experience_store,
                        experience_projection=experience_projection_for(
                            semantic_profile, len(initial_criteria), initial_request,
                            frozenset(initial_criteria)),
                        semantic_profile=semantic_profile,
                        belief_query_mode_by_criterion=(
                            belief_query_mode_by_criterion)),
                    "gate": CadenceGate(table=dict(cadence_table),
                                        enabled=args.cadence_gate == "on"),
                    "query_arrivals": set(), "retrieval_turns": set(),
                    "plan_reference_changed_turns": set(),
                    "past_evidence_items": {},
                    "event_calls": 0, "event_failed": 0, "event_nonempty": 0,
                    "event_now": {}, "event_visible": {}, "event_spans": {},
                    "last_event": "",
                    "event_tracker": CurrentEvidenceTracker(
                        repeat_gap=event_repeat_gap,
                        repeat_jaccard=event_repeat_jaccard,
                        render_mode=event_render_mode,
                        end_confirmations=event_end_confirmations,
                        instance_input_gating=event_instance_input_gating),
                    "stream_state": StreamDecisionState(profile=semantic_profile),
                    "frame_transition": FrameTransitionTracker(
                        minor_threshold=transition_minor,
                        major_threshold=transition_major,
                        edge_gap=transition_edge_gap),
                    "state_request_turns": set(),
                    "initial_criteria": set(initial_criteria),
                    "decision_clock": {},
                    "actions": {}, "raws": {}, "preds": {},
                    "header": "", "chunk": -1, "context_epoch": -1,
                    "context_start": 0, "evidence_start": 0,
                    "response_epoch_anchor": None,
                    "response_profile_start": 0,
                    "forced": None, "force_now": False,
                    "prompt_tokens": 0, "prompt_growth": 0, "over_length_at": None,
                    "preverify": {}, "verified": {}, "reviewed": {},
                    "verify_fail_turns": set(),
                    "verifier_calls": 0, "verifier_changed": 0,
                    "verifier_silenced": 0, "verifier_failed": 0,
                    "verifier_recommended_silence": 0,
                    "novelty": novelty,
                    "novelty_rescue_turns": set(),
                    "continual": ContinualController(
                        args.continual_refiner, novelty,
                        every=args.refine_every, warmup=args.refine_warmup),
                    "novelty_rescued": 0,
                    "last": (min(len(records), args.max_turns) - 1 if args.max_turns
                             else len(records) - 1),
                })
            if not lanes:
                continue

            reserve = int(sp.max_tokens or 0) + 8
            for turn in range(max(l["last"] for l in lanes) + 1):
                step_wall_t0 = time.perf_counter()
                live = [l for l in lanes
                        if turn <= l["last"] and l["over_length_at"] is None]
                if not live:
                    break
                for l in live:
                    if l["prompt_tokens"] and (
                            l["prompt_tokens"] + l["prompt_growth"] + reserve > max_model_len):
                        if args.force_answer_at == "last" and l.get("forced") is None:
                            # Give the lane ONE forced answer with whatever it did
                            # see before running out of context, then retire it.
                            # Scoring the unbounded-context arm as a flat zero on
                            # a 55-minute clip would be measuring the context
                            # limit twice; this measures what it could answer from
                            # the first few minutes, which is the honest baseline.
                            l["force_now"] = True
                            print(f"[harness] lane out of context at turn {turn}: "
                                  f"forcing a final answer then retiring", flush=True)
                            continue
                        l["over_length_at"] = turn
                        print(f"[harness] lane retired at turn {turn}: "
                              f"{l['prompt_tokens']}+{l['prompt_growth']}+{reserve} "
                              f"> {max_model_len}", flush=True)
                live = [l for l in live if l["over_length_at"] is None]
                if not live:
                    break

                # Prefetch this round's frames; ffmpeg releases the GIL.
                prefetch = []
                for lane in live:
                    prefetch.extend(
                        str(p.get("image") or "")
                        for p in _user_content(lane["records"][turn], include_images=True,
                                               images_per_step=args.images_per_step)
                        if isinstance(p, dict) and p.get("image"))
                fresh = [q for q in dict.fromkeys(prefetch) if q not in frame_cache]
                if fresh:
                    with ThreadPoolExecutor(max_workers=min(32, len(fresh))) as pool:
                        list(pool.map(load_frame, fresh))

                # Compile all new request-side references in parallel on the
                # separate harness model. They are joined before the interaction model runs and
                # never inspect or alter its output.
                for lane in live:
                    query = lane["tracker"].queries.get(turn)
                    if (args.arm in ("harness", "harness_32") and query is not None
                            and lane["memory"].plan.cue_planner_enabled):
                        future = lane["memory"].submit_plan_reference(query.text)
                        if future is not None:
                            lane["plan_future"] = future

                # A request-side reference is an INPUT to current-event
                # normalisation, not merely a later header decoration. Join all
                # already-submitted plan jobs before submitting perception jobs
                # so the latter can distinguish a step/checkpoint from generic
                # motion. The futures above were launched in parallel, and no
                # the interaction model proposal exists yet.
                for lane in live:
                    plan_future = lane.pop("plan_future", None)
                    if plan_future is not None:
                        if lane["memory"].join_plan_reference(plan_future):
                            lane["plan_reference_changed_turns"].add(turn)

                # Current-frame perception/event normalisation.  All lane jobs
                # are submitted before any result is joined, allowing the
                # independent 4B server to batch them.  Inputs end at `turn`;
                # the result is appended before the interaction model and never maps to an action.
                if (event_enabled and event_backend == "agent"
                        and args.arm in ("harness", "harness_32")
                        and turn % event_stride == 0):
                    for lane in live:
                        current_instruction = lane["tracker"].queries.get(turn)
                        active = current_instruction or lane["tracker"].carry()
                        if active is None:
                            continue
                        event_criteria = deepcopy(lane["adapter"].criteria)
                        if current_instruction is not None:
                            first = (not event_criteria.active
                                     and not event_criteria.log)
                            event_criteria.apply(
                                parse_instruction(
                                    current_instruction.text, first=first),
                                current_instruction.second)
                        active_count = len(event_criteria.active)
                        if (not event_profile_enabled(
                                lane["prompt_route"]["semantic_profile"],
                                event_profiles)
                                or not active_criteria_within_limit(
                                    active_count,
                                    event_max_active_criteria)):
                            continue
                        primary_skill = (lane["adapter"].recalled[0].name
                                         if lane["adapter"].recalled else "")
                        lane_event_window = event_window_by_skill.get(
                            primary_skill, event_window)
                        lo = max(0, turn - lane_event_window + 1)
                        event_turns = sample_turn_indices(lo, turn, event_frames)
                        current_frames = []
                        for et in event_turns:
                            for part in _user_content(
                                    lane["records"][et], include_images=True,
                                    images_per_step=args.images_per_step):
                                if isinstance(part, dict) and part.get("image"):
                                    current_frames.append(load_frame(part["image"]))
                        lane["event_calls"] += 1
                        event_reference = "\n".join(value for value in (
                            lane["adapter"].perception_reference(),
                            lane["memory"].plan.render_reference(),
                        ) if value)
                        # The observer follows the complete causal Request
                        # State, not merely QueryTracker's latest clause.  This
                        # preserves additive criteria and excludes retired
                        # ones without consulting task IDs, labels, or future
                        # turns.  A pending one-shot question temporarily takes
                        # precedence over the standing criteria.
                        event_request = (
                            event_criteria.question
                            or event_criteria.focus_text()
                            or active.text)
                        lane["event_future"] = agent.submit_event(
                            current_frames, lane["seconds"][event_turns[0]],
                            lane["seconds"][turn], request=event_request,
                            reference=event_reference,
                            previous=lane["last_event"],
                            lane=str(lane["row"]["eval_id"]),
                            prompt_profile=event_prompt_profile)
                        lane["event_spans"][turn] = event_turns

                # Optional aligned observer: use the same the interaction model weights for a
                # private visual-evidence pass before the interactive call.
                # This pass has no action protocol and its output is never
                # committed or scored; the final the interaction model call still sees the raw
                # frames and exclusively owns </silence>/</response>.
                if (event_enabled and event_backend == "decision_model"
                        and args.arm in ("harness", "harness_32")
                        and turn % event_stride == 0):
                    observer_reqs, observer_lanes = [], []
                    for lane in live:
                        current_instruction = lane["tracker"].queries.get(turn)
                        active = current_instruction or lane["tracker"].carry()
                        if active is None:
                            continue
                        event_criteria = deepcopy(lane["adapter"].criteria)
                        if current_instruction is not None:
                            first = (not event_criteria.active
                                     and not event_criteria.log)
                            event_criteria.apply(
                                parse_instruction(
                                    current_instruction.text, first=first),
                                current_instruction.second)
                        active_count = len(event_criteria.active)
                        if (not event_profile_enabled(
                                lane["prompt_route"]["semantic_profile"],
                                event_profiles)
                                or not active_criteria_within_limit(
                                    active_count,
                                    event_max_active_criteria)):
                            continue
                        primary_skill = (lane["adapter"].recalled[0].name
                                         if lane["adapter"].recalled else "")
                        lane_event_window = event_window_by_skill.get(
                            primary_skill, event_window)
                        lo = max(0, turn - lane_event_window + 1)
                        event_turns = sample_turn_indices(lo, turn, event_frames)
                        reference = "\n".join(value for value in (
                            lane["adapter"].perception_reference(),
                            lane["memory"].plan.render_reference(),
                        ) if value) or "Use the ACTIVE REQUEST literally."
                        event_request = (
                            event_criteria.question
                            or event_criteria.focus_text()
                            or active.text)
                        observer_text = current_event_prompt(event_prompt_profile).format(
                            n=len(event_turns),
                            t0=lane["seconds"][event_turns[0]],
                            t1=lane["seconds"][turn],
                            request=event_request,
                            reference=reference,
                            previous=lane["last_event"] or "none",
                        )
                        content = []
                        for et in event_turns:
                            content.append({
                                "type": "text",
                                "text": f"[CAUSAL FRAME {lane['seconds'][et]:.0f}s]",
                            })
                            for part in _user_content(
                                    lane["records"][et], include_images=True,
                                    images_per_step=args.images_per_step):
                                if isinstance(part, dict) and part.get("image"):
                                    content.append(part)
                        content.append({"type": "text", "text": observer_text})
                        observer_messages = [
                            {"role": "system", "content": (
                                "Private Harness perception pass. Do not answer the user "
                                "and do not emit action-control tokens. Output only the "
                                "requested structured evidence fields.")},
                            {"role": "user", "content": content},
                        ]
                        observer_chat = proc.apply_chat_template(
                            observer_messages, tokenize=False,
                            add_generation_prompt=True, **template_options)
                        observer_paths = images_of(observer_messages)
                        observer_reqs.append({
                            "prompt": observer_chat,
                            "multi_modal_data": {
                                "image": [load_frame(path) for path in observer_paths]},
                            "multi_modal_uuids": {
                                "image": [f"{path}:{MIN_PIXELS}:{MAX_PIXELS}"
                                          for path in observer_paths]},
                        })
                        observer_lanes.append(lane)
                        lane["event_calls"] += 1
                        lane["event_spans"][turn] = event_turns
                    if observer_reqs:
                        observer_outs = generate_guarded(
                            llm, observer_reqs, observer_sp)
                        for lane, observer_out in zip(observer_lanes, observer_outs):
                            if observer_out is None:
                                lane["event_failed"] += 1
                                lane.setdefault("decision_event_text", {})[turn] = ""
                            else:
                                lane.setdefault("decision_event_text", {})[turn] = (
                                    observer_out.outputs[0].text or "").strip()

                reqs = []
                decision_messages = []
                response_supports = []
                staged_effective = []
                for lane in live:
                    records, mem, tracker = lane["records"], lane["memory"], lane["tracker"]

                    active_before_turn = tracker.carry()
                    focus_text = policy.evidence_focus(
                        active_before_turn.text if active_before_turn else "")
                    obs = lane["adapter"].observations()
                    if obs:
                        focus_text += "\nOBSERVATIONS TO KEEP (from the active help skill): " + obs
                    skill_reference = lane["adapter"].decision_reference()
                    if skill_reference:
                        focus_text += ("\nREQUEST/SKILL DECISION BOUNDARY "
                                       "(not observed state):\n" + skill_reference)
                    plan_reference = mem.plan.render_reference()
                    if plan_reference:
                        focus_text += ("\nREQUEST-SIDE REFERENCE (not observed state):\n"
                                       + plan_reference)
                    mem.set_focus(focus_text)

                    def frames_for(lo, hi, _lane=lane):
                        images = []
                        for t in range(max(0, lo), min(hi, len(_lane["records"]) - 1) + 1):
                            for p in _user_content(_lane["records"][t], include_images=True,
                                                   images_per_step=args.images_per_step):
                                if isinstance(p, dict) and p.get("image"):
                                    images.append(load_frame(p["image"]))
                        return images

                    if args.arm in ("harness", "short_only", "harness_32"):
                        mem.before_step(turn, frames_for, lambda i, _l=lane: _l["seconds"][i])
                    memory_boundary = mem.chunk != lane["chunk"] or turn == 0
                    # Model-free visual motion state.  This is deliberately a
                    # provenance-only transition: it never claims the active
                    # request occurred and never maps motion to an action.
                    if lane["transition_state_effective"]:
                        memory_observation_delta = (
                            lane["stream_state"].sync_ordered_observations(
                                mem.belief.event_version,
                                lane["seconds"][turn]))
                        if (memory_observation_delta
                                and render_ordered_observation_deltas):
                            if not memory_boundary:
                                prior = lane["turn_extras"].get(turn, "")
                                lane["turn_extras"][turn] = (
                                    prior + "\n" + memory_observation_delta
                                    if prior else memory_observation_delta)
                        current_paths = [
                            part.get("image") for part in _user_content(
                                records[turn], include_images=True,
                                images_per_step=args.images_per_step)
                            if isinstance(part, dict) and part.get("image")]
                        if current_paths:
                            visual_transition = lane["frame_transition"].update(
                                load_frame(current_paths[-1]), lane["seconds"][turn])
                            if visual_transition is not None:
                                transition_delta = lane["stream_state"].on_visual_transition(
                                    visual_transition)
                                if (transition_delta
                                        and render_visual_transition_deltas
                                        and not memory_boundary):
                                    prior = lane["turn_extras"].get(turn, "")
                                    lane["turn_extras"][turn] = (
                                        prior + "\n" + transition_delta
                                        if prior else transition_delta)
                        procedure_delta = lane["stream_state"].sync_procedure(
                            mem.plan.dump(), lane["seconds"][turn])
                        if procedure_delta:
                            if procedure_delta_mode == "observed_line":
                                procedure_delta = lane["adapter"].render_procedure_delta(
                                    lane["seconds"][turn], mem.plan.render_line())
                            if (procedure_delta and render_procedure_deltas
                                    and not memory_boundary):
                                prior = lane["turn_extras"].get(turn, "")
                                lane["turn_extras"][turn] = (
                                    prior + "\n" + procedure_delta
                                    if prior else procedure_delta)
                    # Header BEFORE tracker.on_turn: it must carry only queries
                    # that were already pending when the chunk opened. A query
                    # typed on this very turn is already in this turn's own user
                    # content, and restating it in the header would duplicate it
                    # in the prompt.
                    route_change = None
                    route_request = tracker.queries.get(turn)
                    projected_criteria = set(lane["adapter"].criteria.active)
                    projected_question = (
                        lane["adapter"].criteria.question is not None)
                    if route_request is not None:
                        projected = deepcopy(lane["adapter"].criteria)
                        projected_first = not projected.active and not projected.log
                        projected.apply(
                            parse_instruction(route_request.text,
                                              first=projected_first),
                            route_request.second)
                        projected_criteria = set(projected.active)
                        projected_question = projected.question is not None
                    if (turn > 0 and route_request is not None
                            and lane["prompt_route"]["refresh"]
                            == "causal_request_events"):
                        if prompt_router:
                            desired_semantic, desired_prompt = routed_state_profile(
                                route_request.text, projected_criteria,
                                projected_question, prompt_router,
                                args.system_prompt_profile,
                                temporary_question_profile=(
                                    temporary_question_profile))
                        else:
                            desired_semantic = request_profile(route_request.text)
                            desired_prompt = args.system_prompt_profile
                        if (desired_semantic
                                != lane["prompt_route"]["semantic_profile"]
                                or desired_prompt != lane["system_prompt_profile"]):
                            route_change = {
                                "semantic_profile": desired_semantic,
                                "system_prompt_profile": desired_prompt,
                                "request": route_request.text,
                                "reason": "causal_request",
                            }
                    elif turn > 0 and prompt_router:
                        restored = standing_route_after_question(
                            lane["prompt_route"]["semantic_profile"],
                            lane["adapter"].criteria.focus_text(),
                            lane["adapter"].criteria.active,
                            lane["adapter"].criteria.question is not None,
                            prompt_router, args.system_prompt_profile)
                        if restored is not None:
                            route_change = {
                                "semantic_profile": restored[0],
                                "system_prompt_profile": restored[1],
                                "request": lane["adapter"].criteria.focus_text(),
                                "reason": "question_answered_restore",
                            }
                    route_boundary = route_change is not None
                    projected_profile = (
                        route_change["semantic_profile"]
                        if route_change is not None
                        else lane["prompt_route"]["semantic_profile"])
                    projected_silence_mode, projected_silence_scope = (
                        scoped_silence_history(
                            silence_history_mode, silence_history_by_profile,
                            silence_history_by_criterion, projected_profile,
                            projected_criteria, silence_history_focus_gate,
                            (route_request.text if route_request is not None
                             else ((route_change or {}).get("request")
                                   or lane["adapter"].request_text)),
                            criterion_projection_initial_gate,
                            lane["initial_criteria"]))
                    criterion_receipt_reset = bool(
                        turn > 0 and route_request is not None
                        and projected_silence_scope.startswith("criterion:")
                        and projected_silence_mode == "first_per_epoch")
                    silence_policy_boundary = bool(
                        projected_silence_mode != lane["silence_history_mode"]
                        or projected_silence_scope != lane["silence_history_scope"]
                        or criterion_receipt_reset)
                    latest_response_turn = (lane["spoke_turns"][-1]
                                            if lane["spoke_turns"] else None)
                    response_boundary = response_epoch_due(
                        lane["spoke_turns"][lane["response_profile_start"]:],
                        lane["response_epoch_anchor"],
                        lane["response_epoch_enabled"],
                        lane["response_epoch_trigger"],
                        lane["response_epoch_burst_gap"],
                        lane["spoke_texts"][lane["response_profile_start"]:],
                        response_epoch_repeat_threshold,
                        response_epoch_repeat_history,
                        response_epoch_repeat_gap)
                    boundary = (memory_boundary or response_boundary
                                or route_boundary or silence_policy_boundary)
                    # Response-side state changes become visible on the next
                    # turn.  At an epoch boundary they are already folded into
                    # the freshly rendered header; otherwise append them to the
                    # current user turn so the existing prefix remains intact.
                    response_delta = lane["adapter"].consume_pending_delta(
                        folded=boundary)
                    if response_delta:
                        prior = lane["turn_extras"].get(turn, "")
                        lane["turn_extras"][turn] = (
                            prior + "\n" + response_delta if prior else response_delta)
                    # F2: an instruction that arrives ON a boundary turn (turn 0
                    # for every streaming row) is committed before the header is
                    # rendered, so epoch 0 already carries the criteria / skills.
                    # The query text itself is NOT restated in the header (it
                    # sits in this turn's own user content); only derived state
                    # is. tracker.on_turn stays after the header for the carried
                    # query, exactly as before.
                    pre_query = tracker.queries.get(turn) if boundary else None
                    memory_snapshot = (mem.memory_text()
                                       if args.arm in ("harness", "harness_32") else "")
                    belief_snapshot = (mem.belief_text()
                                       if args.arm in ("harness", "harness_32") else "")
                    early_delta = ""
                    if pre_query is not None:
                        incoming_criteria, incoming_question = request_criteria(
                            pre_query.text)
                        incoming_profile = (route_change["semantic_profile"]
                                            if route_change is not None
                                            else lane["prompt_route"]["semantic_profile"])
                        lane["adapter"].slots = resolve_state_slots(
                            state_slots_default, state_slots_by_profile,
                            state_slots_by_criterion, incoming_profile,
                            set(lane["adapter"].criteria.active) | incoming_criteria,
                            question_pending=(
                                lane["adapter"].criteria.question is not None
                                or incoming_question),
                            criterion_profiles=prompt_router.get(
                                "criteria_profiles") or {},
                            compose_active_criteria=compose_active_criteria)
                        lane["adapter"].experience_projection = (
                            experience_projection_for(
                                incoming_profile, len(projected_criteria),
                                pre_query.text, frozenset(projected_criteria)))
                        early_delta = lane["adapter"].on_instruction(
                            pre_query.text, pre_query.second, belief_snapshot)
                        lane["prompt_route"]["active_criteria"] = sorted(
                            lane["adapter"].criteria.active)
                        lane["prompt_route"]["question_pending"] = bool(
                            lane["adapter"].criteria.question)
                        if args.arm in ("harness", "harness_32"):
                            if not mem.plan.goal:
                                mem.set_plan_goal(pre_query.text)
                            mem.join_plan_reference(lane.pop("plan_future", None))
                    if route_change is not None:
                        previous_semantic = lane["prompt_route"]["semantic_profile"]
                        semantic_profile = route_change["semantic_profile"]
                        routed_name = route_change["system_prompt_profile"]
                        lane["prompt_route"].update({
                            "semantic_profile": semantic_profile,
                            "request": route_change["request"][:500],
                            "active_criteria": sorted(
                                lane["adapter"].criteria.active),
                            "question_pending": bool(
                                lane["adapter"].criteria.question),
                        })
                        lane["prompt_route"]["events"].append({
                            "turn": int(turn),
                            "second": float(lane["seconds"][turn]),
                            "from": previous_semantic,
                            "to": semantic_profile,
                            "request": route_change["request"][:500],
                            "reason": route_change.get("reason", "causal_request"),
                        })
                        lane["header_memory_projection"] = str(
                            header_memory_by_profile.get(
                                semantic_profile, header_memory_projection))
                        lane["response_history_mode"] = str(
                            response_history_by_profile.get(
                                semantic_profile, response_history_mode))
                        lane["transition_state_effective"] = bool(
                            transition_enabled and (not transition_profiles
                                                    or semantic_profile
                                                    in transition_profiles))
                        lane["stream_state"].profile = semantic_profile
                        lane["adapter"].slots = resolve_state_slots(
                            state_slots_default, state_slots_by_profile,
                            state_slots_by_criterion, semantic_profile,
                            lane["adapter"].criteria.active,
                            question_pending=(
                                lane["adapter"].criteria.question is not None),
                            initial_presence_gated_criteria=(
                                criterion_projection_initial_gate),
                            initial_criteria=lane["initial_criteria"],
                            criterion_profiles=prompt_router.get(
                                "criteria_profiles") or {},
                            compose_active_criteria=compose_active_criteria)
                        lane["adapter"].experience_projection = (
                            experience_projection_for(
                                semantic_profile,
                                len(lane["adapter"].criteria.active),
                                route_change.get("request", ""),
                                frozenset(lane["adapter"].criteria.active)))
                        lane["adapter"].semantic_profile = semantic_profile
                        lane["response_generation_mode"] = (
                            response_generation_for(semantic_profile))
                        # Reply-loop receipts belong to the causal intent that
                        # produced them. A retired/replaced request cannot arm
                        # compression for the next profile.
                        lane["response_profile_start"] = len(lane["spoke_turns"])
                        lane["response_epoch_anchor"] = None
                        lane["response_epoch_enabled"] = bool(
                            response_epoch_enabled
                            and (not response_epoch_profiles
                                 or semantic_profile in response_epoch_profiles))
                        lane["response_epoch_trigger"] = str(
                            response_epoch_trigger_by_profile.get(
                                semantic_profile, response_epoch_trigger))
                        lane["response_epoch_lookback"] = int(
                            response_epoch_lookback_by_profile.get(
                                semantic_profile, response_epoch_lookback))
                        lane["response_epoch_evidence_lookback"] = int(
                            response_epoch_evidence_lookback_by_profile.get(
                                semantic_profile,
                                response_epoch_evidence_lookback))
                        lane["response_epoch_burst_gap"] = int(
                            response_epoch_burst_gap_by_profile.get(
                                semantic_profile, response_epoch_burst_gap))
                        lane["system_prompt_profile"] = routed_name
                        lane["system_prompt"] = (
                            SYSTEM_PROMPTS[routed_name] + policy.response_suffix())
                        lane["system_prompt_sha256"] = hashlib.sha256(
                            lane["system_prompt"].encode("utf-8")).hexdigest()
                    if route_change is not None or silence_policy_boundary:
                        lane["silence_history_mode"], lane["silence_history_scope"] = (
                            scoped_silence_history(
                                silence_history_mode, silence_history_by_profile,
                                silence_history_by_criterion,
                                lane["prompt_route"]["semantic_profile"],
                                lane["adapter"].criteria.active,
                                silence_history_focus_gate,
                                lane["adapter"].request_text,
                                criterion_projection_initial_gate,
                                lane["initial_criteria"]))
                        lane["silence_profile_start"] = int(turn)
                    if (pre_query is not None
                            and turn not in lane["state_request_turns"]):
                        state_delta = lane["stream_state"].on_request(
                            pre_query.second,
                            lane["prompt_route"]["semantic_profile"],
                            question=lane["adapter"].criteria.question is not None)
                        lane["state_request_turns"].add(turn)
                        # At a boundary the fresh Header below already folds
                        # this lifecycle mutation. Otherwise it is appended at
                        # the causal request event and never edited later.
                        if (not boundary and lane["transition_state_effective"]
                                and render_request_lifecycle_deltas
                                and "lifecycle" in lane["adapter"].slots):
                            prior = lane["turn_extras"].get(turn, "")
                            lane["turn_extras"][turn] = (
                                prior + "\n" + state_delta if prior else state_delta)
                    if boundary:
                        lane["chunk"] = mem.chunk
                        lane["context_epoch"] += 1
                        if response_boundary and not route_boundary:
                            lane["response_epoch_anchor"] = latest_response_turn
                        lane["context_start"] = response_epoch_start(
                            window_start(args.arm, mem, turn, cfg.Ts),
                            latest_response_turn, lane["response_epoch_enabled"],
                            lane["response_epoch_lookback"])
                        lane["evidence_start"] = min(
                            lane["context_start"],
                            response_epoch_start(
                                window_start(args.arm, mem, turn, cfg.Ts),
                                latest_response_turn,
                                lane["response_epoch_enabled"],
                                lane["response_epoch_evidence_lookback"]))
                        reference_snapshot = "\n".join(v for v in (
                            mem.plan.render_reference(),
                            lane["adapter"].decision_reference(detailed=False)) if v)
                        cue_snapshot = mem.plan.render_cue_evidence()
                        active_facet = question_facet(
                            lane["adapter"].request_text
                            or (pre_query.text if pre_query is not None else ""))
                        lane["question_facet"] = active_facet
                        if args.arm == "harness_32":
                            visible_start_turn = max(
                                0, turn - max(1, args.visual_window) + 1)
                        else:
                            visible_start_turn = lane["evidence_start"]
                        memory_redundant = memory_ranges_redundant(
                            mem.dump(), lane["seconds"][visible_start_turn],
                            lane["seconds"][turn])
                        lane["memory_redundant_with_raw"] = memory_redundant
                        requested_memory_projection = str(
                            header_memory_by_question_facet.get(
                                active_facet, lane["header_memory_projection"]))
                        if requested_memory_projection == "nonredundant":
                            effective_memory_projection = (
                                "none" if memory_redundant else "on_demand")
                        else:
                            effective_memory_projection = (
                                requested_memory_projection)
                        lane["effective_header_memory_projection"] = (
                            effective_memory_projection)
                        lane["header"] = lane["adapter"].render_header(
                            args.card, lane["context_epoch"], memory_snapshot,
                            tracker.carry() if args.arm in ("harness", "short_only") else None,
                            procedure_line=mem.plan.render_line(),
                            lifecycle_line=(lane["stream_state"].render_line()
                                            if lane["transition_state_effective"] else ""),
                            belief_text=belief_snapshot,
                            reference_line=reference_snapshot,
                            cue_evidence=cue_snapshot,
                            plan_anchors=mem.plan.anchors,
                            memory_projection=effective_memory_projection,
                            query_projection=(
                                "standing" if lane["response_epoch_enabled"]
                                and latest_response_turn is not None else "waiting"))
                        minimal_cue = (
                            minimal_state_if_redundant_by_question_facet.get(
                                active_facet, "") if memory_redundant else "")
                        lane["minimal_state_projection"] = bool(minimal_cue)
                        if minimal_cue:
                            # B/P/E continue updating privately.  When every B
                            # source is already among the interaction model's
                            # causal frames, avoid a lossy duplicate summary and
                            # project only one answer-free evidence-selection cue.
                            lane["header"] = (
                                "[HARNESS STATE]\n"
                                "[EVIDENCE SELECTION]\n"
                                f"- {minimal_cue}")
                            lane["adapter"].rendered_slots = ["experience"]
                        ad = lane["adapter"]
                        lane["epoch_log"].append({
                            "epoch": mem.chunk, "turn": turn, "second": lane["seconds"][turn],
                            "context_epoch": lane["context_epoch"],
                            "context_start": lane["context_start"],
                            "evidence_start": lane["evidence_start"],
                            "boundary_reasons": [
                                name for name, active in (
                                    ("memory", memory_boundary),
                                    ("response", response_boundary),
                                    ("request_route", route_boundary),
                                    ("silence_policy", silence_policy_boundary))
                                if active],
                            "card_on": ad.card_wanted() and bool(card_slots),
                            "criteria_active": sorted(ad.criteria.active),
                            "criteria_retired": sorted(ad.criteria.retired),
                            "question": ad.criteria.question,
                            "procedure": mem.plan.render_line(),
                            "lifecycle": (lane["stream_state"].render_line()
                                          if lane["transition_state_effective"] else ""),
                            "reference": reference_snapshot,
                            "cue_evidence": cue_snapshot,
                            "recalled": [sk.name for sk in ad.recalled],
                            "rendered_slots": list(ad.rendered_slots),
                            "experience_prompt_projection": (
                                ad.experience_projection),
                            "question_facet": lane["question_facet"],
                            "memory_redundant_with_raw": lane[
                                "memory_redundant_with_raw"],
                            "minimal_state_projection": lane[
                                "minimal_state_projection"],
                            "effective_header_memory_projection": lane[
                                "effective_header_memory_projection"],
                            "silence_history_mode": lane["silence_history_mode"],
                            "silence_history_scope": lane["silence_history_scope"],
                            "header_chars": len(lane["header"]),
                            "header": lane["header"][:6000],
                        })
                    lane["turn_epoch"][turn] = mem.chunk
                    lane["turn_context_epoch"][turn] = lane["context_epoch"]
                    lane["turn_context_start"][turn] = lane["context_start"]
                    lane["turn_evidence_start"][turn] = lane["evidence_start"]
                    lane["turn_silence_history_mode"][turn] = (
                        lane["silence_history_mode"])
                    lane["turn_silence_history_start"][turn] = (
                        lane["silence_profile_start"])
                    # The guard is fed EVERY turn, not only at boundaries: a
                    # header that changed mid-chunk is the one failure this
                    # design must never have, and the count goes into the trace.
                    lane["epoch_guard"].observe(
                        lane["context_epoch"], lane["header"])
                    tracker.on_turn(turn)
                    arriving_query = tracker.queries.get(turn)
                    if arriving_query is not None and pre_query is None:
                        # Event point inside an epoch: commit(request_state) and,
                        # for a scene question, track[object]. The card is not
                        # touched; the change is an append-only delta on this turn.
                        incoming_criteria, incoming_question = request_criteria(
                            arriving_query.text)
                        lane["adapter"].slots = resolve_state_slots(
                            state_slots_default, state_slots_by_profile,
                            state_slots_by_criterion,
                            lane["prompt_route"]["semantic_profile"],
                            set(lane["adapter"].criteria.active) | incoming_criteria,
                            question_pending=(
                                lane["adapter"].criteria.question is not None
                                or incoming_question),
                            initial_presence_gated_criteria=(
                                criterion_projection_initial_gate),
                            initial_criteria=lane["initial_criteria"],
                            criterion_profiles=prompt_router.get(
                                "criteria_profiles") or {},
                            compose_active_criteria=compose_active_criteria)
                        lane["adapter"].experience_projection = (
                            experience_projection_for(
                                lane["prompt_route"]["semantic_profile"],
                                len(projected_criteria), arriving_query.text,
                                frozenset(projected_criteria)))
                        delta = lane["adapter"].on_instruction(
                            arriving_query.text, arriving_query.second, belief_snapshot)
                        lane["prompt_route"]["active_criteria"] = sorted(
                            lane["adapter"].criteria.active)
                        lane["prompt_route"]["question_pending"] = bool(
                            lane["adapter"].criteria.question)
                        if delta:
                            prior = lane["turn_extras"].get(turn, "")
                            lane["turn_extras"][turn] = (prior + "\n" + delta) if prior else delta
                        if turn not in lane["state_request_turns"]:
                            state_delta = lane["stream_state"].on_request(
                                arriving_query.second,
                                lane["prompt_route"]["semantic_profile"],
                                question=lane["adapter"].criteria.question is not None)
                            lane["state_request_turns"].add(turn)
                            if (lane["transition_state_effective"]
                                    and render_request_lifecycle_deltas
                                    and "lifecycle" in lane["adapter"].slots):
                                prior = lane["turn_extras"].get(turn, "")
                                lane["turn_extras"][turn] = (
                                    prior + "\n" + state_delta
                                    if prior else state_delta)
                    elif early_delta:
                        # Header already carries the criteria; keep only the
                        # tracked belief facts (they are not in the header).
                        facts = "\n".join(l for l in early_delta.splitlines() if "] B (" in l)
                        if facts:
                            lane["turn_extras"][turn] = facts
                    if arriving_query is not None and args.arm in ("harness", "harness_32"):
                        if pre_query is None:
                            if not mem.plan.goal:
                                mem.set_plan_goal(arriving_query.text)
                            reference_changed = (
                                turn in lane["plan_reference_changed_turns"]
                                or mem.join_plan_reference(lane.pop("plan_future", None)))
                            if reference_changed:
                                reference = mem.plan.render_reference()
                                if reference:
                                    prior = lane["turn_extras"].get(turn, "")
                                    tagged = (f"[Δ {arriving_query.second:.0f}s] "
                                              f"request-side reference:\n{reference}")
                                    lane["turn_extras"][turn] = (
                                        prior + "\n" + tagged if prior else tagged)
                        lane["query_arrivals"].add(turn)
                        evidence_items = mem.evidence_items()
                        lane["past_evidence_items"][turn] = len(evidence_items)
                        retrieved = lane["skills"].retrieve(
                            arriving_query.text, evidence_items)
                        if retrieved:
                            # F1: append -- this used to overwrite the adapter delta
                            prior = lane["turn_extras"].get(turn, "")
                            lane["turn_extras"][turn] = (prior + "\n" + retrieved) if prior else retrieved
                            lane["retrieval_turns"].add(turn)
                    event_future = lane.pop("event_future", None)
                    decision_event = lane.get("decision_event_text", {}).pop(turn, None)
                    if event_future is not None or decision_event is not None:
                        try:
                            event_text = ((event_future.result() if event_future is not None
                                           else decision_event) or "").strip()
                        except Exception as exc:  # noqa: BLE001
                            lane["event_failed"] += 1
                            event_text = ""
                            print(f"[harness] current event parser failed: {exc}",
                                  flush=True)
                        if event_text:
                            lane["event_nonempty"] += 1
                            lane["last_event"] = event_text[:1200]
                            lane["event_now"][turn] = event_text[:1200]
                            if event_render_mode == "raw":
                                visible_event = event_text[:1200]
                                parsed_event = None
                            else:
                                parsed_event, visible_event = lane["event_tracker"].update(
                                    event_text, lane["seconds"][turn])
                            if (parsed_event is not None
                                    and parsed_event.task_evidence == "CRITERION_MATCH"
                                    and (not event_instance_input_gating
                                         or lane["event_tracker"]
                                         .last_update_opened_instance)):
                                canonical_delta = lane["adapter"].on_confirmed_observation(
                                    lane["seconds"][turn], parsed_event.match_kind,
                                    parsed_event.evidence,
                                    evidence_ref=(
                                        f"event_parser:{lane['row']['eval_id']}:"
                                        f"{turn}"),
                                    instance_id=lane["event_tracker"].last_update_instance_id)
                                state_delta = lane["stream_state"].on_confirmed_evidence(
                                    lane["seconds"][turn], parsed_event.match_kind)
                                if canonical_delta and not memory_boundary:
                                    prior = lane["turn_extras"].get(turn, "")
                                    lane["turn_extras"][turn] = (
                                        prior + "\n" + canonical_delta
                                        if prior else canonical_delta)
                                if (lane["transition_state_effective"]
                                        and render_confirmed_evidence_deltas
                                        and "lifecycle" in lane["adapter"].slots):
                                    prior = lane["turn_extras"].get(turn, "")
                                    lane["turn_extras"][turn] = (
                                        prior + "\n" + state_delta
                                        if prior else state_delta)
                            if visible_event:
                                lane["event_visible"][turn] = visible_event[:600]
                                tagged = (
                                    f"[CURRENT EVIDENCE {lane['seconds'][turn]:.0f}s; "
                                    "private parser, raw frames override] "
                                    + visible_event[:600])
                                if (event_lifetime == "current_system"
                                        or event_instance_input_gating):
                                    lane["current_system_extras"][turn] = tagged
                                else:
                                    prior = lane["turn_extras"].get(turn, "")
                                    lane["turn_extras"][turn] = (
                                        prior + "\n" + tagged if prior else tagged)
                    if (decision_clock_enabled
                            and args.arm in ("harness", "harness_32")):
                        cadence = lane["adapter"].decision_cadence()
                        last_turn = (lane["spoke_turns"][-1]
                                     if lane["spoke_turns"] else None)
                        age = (lane["seconds"][turn] - lane["seconds"][last_turn]
                               if last_turn is not None else None)
                        within = bool(age is not None and cadence["min_gap"] > 0
                                      and age < cadence["min_gap"])
                        clock = {
                            **cadence,
                            "last_response_turn": last_turn,
                            "elapsed": age,
                            "within_gap": within,
                        }
                        lane["decision_clock"][turn] = clock
                        elapsed = "none" if age is None else f"{age:.0f}s"
                        tagged = (
                            f"[DECISION CLOCK {lane['seconds'][turn]:.0f}s] "
                            f"policy={cadence['policy']}; elapsed={elapsed}; "
                            f"min_gap={cadence['min_gap']}s; "
                            f"within_gap={'true' if within else 'false'}")
                        prior = lane["turn_extras"].get(turn, "")
                        lane["turn_extras"][turn] = (
                            prior + "\n" + tagged if prior else tagged)
                    ws = (lane["context_start"] if lane["response_epoch_enabled"]
                          and args.arm in ("harness", "short_only")
                          else window_start(args.arm, mem, turn, cfg.Ts))
                    last_response_second = (
                        lane["seconds"][lane["spoke_turns"][-1]]
                        if lane["spoke_turns"] else None)
                    decision_prompt_profile, response_phase = response_phase_profile(
                        lane["seconds"][turn], last_response_second,
                        lane["system_prompt_profile"], response_cooldown)
                    decision_system_prompt = (
                        SYSTEM_PROMPTS[decision_prompt_profile]
                        + policy.response_suffix())
                    lane["decision_prompt_profiles"][turn] = decision_prompt_profile
                    lane["decision_prompt_sha256"][turn] = hashlib.sha256(
                        decision_system_prompt.encode("utf-8")).hexdigest()
                    lane["response_phases"][turn] = response_phase
                    if args.arm in ("original_32", "harness_32"):
                        anchors = tuple(sorted(lane["spoke_turns"])[-2:]) if args.visual_anchor == "on" else ()
                        msgs = build_original32_messages(
                            records, turn, lane["actions"], lane["header"],
                            decision_system_prompt, _user_content, args.images_per_step,
                            window=args.visual_window, turn_extras=lane["turn_extras"],
                            stride=args.visual_stride, anchors=anchors,
                            control_role=control_role,
                            current_system_extra=lane["current_system_extras"].get(turn, ""),
                            response_history_mode=(
                                "latest_full" if lane["response_history_mode"]
                                == "loop_latest_full"
                                and lane["response_epoch_anchor"] is not None
                                else "full" if lane["response_history_mode"]
                                == "loop_latest_full"
                                else lane["response_history_mode"]))
                    else:
                        msgs = build_harness_messages(
                            records, turn, lane["actions"], ws, lane["header"],
                            decision_system_prompt, _user_content, args.images_per_step,
                            silence_in_history=(args.history_silence == "keep"),
                            turn_extras=lane["turn_extras"],
                            control_role=control_role,
                            current_system_extra=lane["current_system_extras"].get(turn, ""),
                            response_history_mode=(
                                "latest_full" if lane["response_history_mode"]
                                == "loop_latest_full"
                                and lane["response_epoch_anchor"] is not None
                                else "full" if lane["response_history_mode"]
                                == "loop_latest_full"
                                else lane["response_history_mode"]),
                            silence_history_mode=lane["silence_history_mode"],
                            silence_history_start=lane["silence_profile_start"],
                            evidence_start=lane["evidence_start"])
                    chat = proc.apply_chat_template(msgs, tokenize=False,
                                                    add_generation_prompt=True,
                                                    **template_options)
                    chat = rewrite_control_roles(chat)
                    if args.force_answer_at == "last" and (
                            turn == lane["last"] or lane.get("force_now")):
                        # Appended AFTER the generation prompt, so what follows is
                        # the reply the model would have written had it chosen to
                        # speak. Same device as four_task_eval's streaming_FORCE_ACTION.
                        chat = chat + "</response>"
                        lane["forced"] = turn
                    paths = images_of(msgs)
                    support = ""
                    stage_this_turn = bool(
                        lane["response_generation_mode"] == "action_then_content"
                        and lane.get("forced") != turn)
                    if stage_this_turn:
                        support = lane["adapter"].render_response_support(
                            response_content_projection)
                        # With no causally recalled Skill there is no Harness
                        # content intervention to test. Retain the byte-identical
                        # joint decoder instead of adding a generic prompt prior.
                        stage_this_turn = bool(support)
                    lane["response_generation_steps"][turn] = {
                        "configured_mode": lane["response_generation_mode"],
                        "effective_mode": (
                            "action_then_content" if stage_this_turn else "joint"),
                        "content_projection": response_content_projection,
                        "support": support[:1200],
                        "action_model": args.model,
                        "content_model": args.model,
                        "action_preserved": True,
                        "postdecision_filter": False,
                    }
                    lane["decision_input_fingerprints"][turn] = (
                        decision_input_fingerprint(chat, paths))
                    decision_messages.append(msgs)
                    response_supports.append(support)
                    staged_effective.append(stage_this_turn)
                    reqs.append({
                        "prompt": chat,
                        "multi_modal_data": {
                            "image": [load_frame(path) for path in paths]},
                        "multi_modal_uuids": {
                            "image": [f"{path}:{MIN_PIXELS}:{MAX_PIXELS}"
                                      for path in paths]},
                    })

                # Paired evaluation may freeze outputs for byte-identical
                # control inputs.  Generate only misses; the first actual
                # Harness input change necessarily misses and reaches the interaction model.
                outs = [None] * len(reqs)
                joint_indices, joint_reqs = [], []
                action_indices, action_reqs = [], []
                for i, (lane, req) in enumerate(zip(live, reqs)):
                    fingerprint = lane["decision_input_fingerprints"][turn][
                        "input_sha256"]
                    replayed_text = decision_replay.get(fingerprint)
                    if replayed_text is None:
                        if staged_effective[i]:
                            action_indices.append(i)
                            action_reqs.append(req)
                        else:
                            joint_indices.append(i)
                            joint_reqs.append(req)
                        continue
                    lane["decision_replayed"].add(turn)
                    # Preserve the downstream token/cache accounting interface;
                    # replayed requests did no prefill or decode in this run.
                    outs[i] = SimpleNamespace(
                        outputs=[SimpleNamespace(text=replayed_text)],
                        prompt_token_ids=(), num_cached_tokens=0)
                generate_t0 = time.perf_counter()
                if joint_reqs:
                    generated = generate_guarded(llm, joint_reqs, sp)
                    for i, output in zip(joint_indices, generated):
                        outs[i] = output
                if action_reqs:
                    generated = generate_guarded(llm, action_reqs, action_sp)
                    for i, output in zip(action_indices, generated):
                        outs[i] = output

                # A staged lane exposes no content support until the interaction model's first
                # token is known. Silence is committed immediately. A response
                # token opens a second continuation from the SAME model with
                # that token already in the assistant prefix. Malformed action
                # tokens fall back to ordinary joint decoding from the original
                # prompt; the Harness never maps or repairs them.
                decoded = [None] * len(live)
                content_indices, content_reqs = [], []
                fallback_indices, fallback_reqs = [], []
                for i, (lane, o) in enumerate(zip(live, outs)):
                    if o is None:
                        continue
                    text = (o.outputs[0].text or "").strip()
                    if (lane.get("forced") == turn
                            and not text.startswith("</response>")):
                        # Put the marker back so action_of(), the history replay
                        # and every scorer see the ordinary wrapped form.
                        text = f"</response> {text}".strip()
                    if not staged_effective[i]:
                        decoded[i] = [o, text]
                        continue
                    generation_step = lane["response_generation_steps"][turn]
                    replayed = turn in lane["decision_replayed"]
                    action_token = (leading_action_token(text) if replayed
                                    else strict_action_token(text))
                    generation_step.update({
                        "action_source": (
                            "exact_input_replay" if replayed else "streamvibe_first_token"),
                        "action_phase_raw": action_token or text[:200],
                    })
                    if action_token is None:
                        generation_step["fallback_joint"] = True
                        if replayed:
                            # The frozen joint generation is already the exact
                            # fail-open result; do not invoke the model again.
                            decoded[i] = [o, text]
                        else:
                            fallback_indices.append(i)
                            fallback_reqs.append(reqs[i])
                        continue
                    generation_step["fallback_joint"] = False
                    if action_token == "</silence>":
                        generation_step["content_called"] = False
                        decoded[i] = [o, action_token]
                        continue

                    content_messages = response_support_messages(
                        decision_messages[i], response_supports[i],
                        response_content_projection)
                    content_chat = proc.apply_chat_template(
                        content_messages, tokenize=False,
                        add_generation_prompt=True, **template_options)
                    content_chat = rewrite_control_roles(content_chat) + action_token
                    content_paths = images_of(content_messages)
                    content_input = decision_input_fingerprint(
                        content_chat, content_paths)
                    generation_step.update({
                        "content_called": True,
                        "content_input": content_input,
                    })
                    lane["response_content_calls"] += 1
                    content_indices.append(i)
                    content_reqs.append({
                        "prompt": content_chat,
                        "multi_modal_data": {
                            "image": [load_frame(path) for path in content_paths]},
                        "multi_modal_uuids": {
                            "image": [f"{path}:{MIN_PIXELS}:{MAX_PIXELS}"
                                      for path in content_paths]},
                    })

                if fallback_reqs:
                    generated = generate_guarded(llm, fallback_reqs, sp)
                    for i, output in zip(fallback_indices, generated):
                        if output is not None:
                            fallback_text = (
                                output.outputs[0].text or "").strip()
                            decoded[i] = [output, fallback_text]
                if content_reqs:
                    generated = generate_guarded(llm, content_reqs, content_sp)
                    for i, output in zip(content_indices, generated):
                        lane = live[i]
                        generation_step = lane["response_generation_steps"][turn]
                        if output is None:
                            # Keep the action chosen by the interaction model. A failed content
                            # continuation is never allowed to suppress it.
                            lane["response_content_failures"] += 1
                            generation_step.update({
                                "content_failed": True,
                                "content_continuation_raw": "",
                            })
                            decoded[i] = [outs[i], "</response>"]
                            continue
                        continuation = output.outputs[0].text or ""
                        generation_step.update({
                            "content_failed": False,
                            "content_continuation_raw": continuation[:1000],
                        })
                        decoded[i] = [
                            outs[i], compose_response("</response>", continuation)]
                generate_s = time.perf_counter() - generate_t0

                # Decode first, then submit all sparse response checks together
                # so the separate agent GPU can batch lanes. A verifier error or
                # malformed action fails open to the online model's proposal.

                verify_jobs = {}
                if args.response_verifier == "agent":
                    for i, (lane, item) in enumerate(zip(live, decoded)):
                        if item is None or lane.get("forced") == turn:
                            continue
                        candidate = item[1]
                        if action_of(candidate) != "response":
                            continue
                        recent = []
                        lo = max(0, turn - max(1, args.verify_frames) + 1)
                        for vt in range(lo, turn + 1):
                            for part in _user_content(
                                    lane["records"][vt], include_images=True,
                                    images_per_step=args.images_per_step):
                                if isinstance(part, dict) and part.get("image"):
                                    recent.append(load_frame(part["image"]))
                        active = lane["tracker"].carry()
                        recent_answers = [
                            f"[{lane['seconds'][at]:.0f}s] {clean_text(action)}"
                            for at, action in sorted(lane["actions"].items())
                            if action_of(action) == "response"
                        ][-8:]
                        lane["verifier_calls"] += 1
                        lane["preverify"][turn] = candidate
                        verify_jobs[i] = agent.submit_verify(
                            recent[-max(1, args.verify_frames):],
                            focus=policy.response_focus(
                                (active.text if active is not None else "")
                                + lane["adapter"].verifier_context()),
                            memory=lane["memory"].memory_text(),
                            candidate=clean_text(candidate),
                            recent_answers="\n".join(recent_answers),
                            lane=str(lane["row"]["eval_id"]),
                        )
                    for i, fut in verify_jobs.items():
                        lane = live[i]
                        before = decoded[i][1]
                        try:
                            after = accept_verified_action(before, fut.result())
                        except Exception as exc:
                            lane["verifier_failed"] += 1
                            lane["verify_fail_turns"].add(turn)
                            print(f"[harness] response verifier failed open: {exc}",
                                  flush=True)
                            after = before
                        lane["reviewed"][turn] = after
                        lane["verifier_recommended_silence"] += int(
                            action_of(before) == "response"
                            and action_of(after) == "silence")
                        if (args.verifier_action_policy == "preserve"
                                and action_of(before) == "response"
                                and action_of(after) == "silence"):
                            after = before
                        lane["verified"][turn] = after
                        if (args.novelty_rescue
                                and action_of(before) == "response"
                                and action_of(after) == "silence"
                                and lane["novelty"].should_rescue(
                                    turn, clean_text(before))):
                            after = before
                            lane["novelty_rescued"] += 1
                            lane["novelty_rescue_turns"].add(turn)
                        lane["verifier_changed"] += int(after != before)
                        lane["verifier_silenced"] += int(
                            action_of(before) == "response"
                            and action_of(after) == "silence")
                        decoded[i][1] = after

                cached = prompt_tok = 0
                for lane, item in zip(live, decoded):
                    if item is None:
                        lane["over_length_at"] = turn
                        continue
                    o, text = item
                    n_prompt = len(o.prompt_token_ids)
                    lane["prompt_growth"] = max(0, n_prompt - lane["prompt_tokens"])
                    lane["prompt_tokens"] = n_prompt
                    prompt_tok += n_prompt
                    cached += int(getattr(o, "num_cached_tokens", 0) or 0)
                    act = action_of(text)
                    # Deterministic commit gate (harness/gate.py): the active help
                    # type's calibrated cadence decides whether this proposal is
                    # committed. Runs after the model and after any verifier, and
                    # only ever turns a response into silence.
                    if act == "response" and lane["gate"].enabled:
                        ad = lane["adapter"]
                        active = sorted(ad.criteria.active) or lane.get("fallback_types", [])
                        ok, why = lane["gate"].decide(
                            lane["seconds"][turn], clean_text(text), active,
                            question_pending=ad.criteria.question is not None)
                        if ok:
                            lane["gate"].record(lane["seconds"][turn], clean_text(text))
                        else:
                            text = "</silence>"
                            act = "silence"
                            lane["gate_suppressed"] = lane.get("gate_suppressed", []) + [(turn, why)]
                    committed_text = clean_text(text) if act == "response" else ""
                    response_state_commit = should_commit_response_state(
                        act, committed_text,
                        nonempty_only=commit_nonempty_only)
                    lane["raws"][turn] = text
                    # Keep raw prediction/scoring unchanged.  With commit
                    # hygiene enabled, an empty response marker is simply not
                    # replayed as a response and cannot open a response epoch.
                    lane["actions"][turn] = committed_history_action(
                        text,
                        model_action=act,
                        response_committed=(
                            act == "response" and response_state_commit),
                        silence_token="</silence>",
                    )
                    lane["preds"][turn] = text
                    proposal_raw = lane["preverify"].get(turn, text)
                    verified_raw = lane["reviewed"].get(
                        turn, lane["verified"].get(turn, text))
                    lane["continual"].observe(DecisionEvent(
                        turn=turn,
                        proposal=(clean_text(proposal_raw)
                                  if action_of(proposal_raw) == "response" else ""),
                        verifier_response=(action_of(verified_raw) == "response"),
                        committed=(clean_text(text) if act == "response" else ""),
                        rescued=turn in lane["novelty_rescue_turns"],
                        verifier_failed=turn in lane["verify_fail_turns"],
                        query_arrived=turn in lane["query_arrivals"],
                        past_evidence_items=lane["past_evidence_items"].get(turn, 0),
                        retrieval_used=turn in lane["retrieval_turns"],
                    ))
                    if response_state_commit:
                        lane["spoke_turns"].append(turn)
                        lane["spoke_texts"].append(committed_text)
                        lane["novelty"].record(turn, committed_text)
                        answered = lane["tracker"].on_action(
                            turn, lane["seconds"][turn], act, committed_text)
                        # A committed response answers the pending scene
                        # question (request_state pop). No delta is written:
                        # the answer itself is already in the history, and the
                        # card line folds at the next epoch boundary.
                        lane["adapter"].on_response(
                            lane["seconds"][turn], committed_text,
                            # Under the opt-in temporary-question lifecycle an
                            # empty response marker is not an answer receipt.
                            # Legacy policies retain their historical pop.
                            answer_question=(
                                not temporary_question_profile
                                or bool(committed_text)))
                        lane["prompt_route"]["question_pending"] = bool(
                            lane["adapter"].criteria.question)
                        lifecycle_delta = lane["stream_state"].on_response(
                            turn, lane["seconds"][turn], committed_text)
                        if event_instance_input_gating:
                            lane["event_tracker"].mark_reported()
                        if (lane["transition_state_effective"]
                                and render_response_lifecycle_deltas
                                and "lifecycle" in lane["adapter"].slots):
                            lane["adapter"].pending_deltas.append(lifecycle_delta)
                        if args.arm in ("harness", "harness_32"):
                            lane["memory"].record_answer(
                                lane["seconds"][turn],
                                answered.text if answered is not None else "",
                                committed_text)

                for lane in live:
                    if lane.pop("force_now", False):
                        lane["over_length_at"] = turn + 1

                step_s = time.perf_counter() - step_wall_t0
                latency.append({
                    "turn": turn, "lanes": len(live), "step_s": round(step_s, 4),
                    # Every stream waits for the complete batch, so its latency
                    # is the wall-clock step time. Dividing by lane count is a
                    # throughput/amortisation metric, not latency.
                    "per_lane_s": round(step_s, 4),
                    "generate_s": round(generate_s, 4),
                    "amortized_s_per_lane": round(step_s / max(1, len(live)), 4),
                    "prompt_tokens": prompt_tok, "cached_tokens": cached,
                    "cache_hit": round(cached / prompt_tok, 4) if prompt_tok else None,
                })
                if turn % 25 == 0 or turn == 0:
                    print(f"[harness] batch {start // args.lanes} turn {turn} "
                          f"live={len(live)} step={step_s:.2f}s "
                          f"hit={latency[-1]['cache_hit']} "
                          f"prompt={prompt_tok // max(1, len(live))}", flush=True)

            for lane in lanes:
                row = lane["row"]
                mem_stats = lane["memory"].stats()
                if lane["actions"]:
                    lane["continual"].finalize(max(lane["actions"]))
                if trace_handle is not None:
                    trace_handle.write(json.dumps({
                        "model": args.model, "arm": args.arm,
                        "harness_impl": HARNESS_NAME,
                        "workspace_artifact_sha256": (
                            WORKSPACE_ARTIFACT_SHA256 or None),
                        "decision_replay": {
                            "enabled": bool(decision_replay),
                            "source_files": len(args.decision_replay_trace),
                            "hits": len(lane["decision_replayed"]),
                            "misses": len(lane["actions"])
                                      - len(lane["decision_replayed"]),
                        },
                        "memory_transcript": (
                            agent.transcript_state(str(row["eval_id"]))
                            if hasattr(agent, "transcript_state") else {
                                "mode": "off", "entries_loaded": 0, "calls": 0}),
                        "system_prompt_profile": lane["system_prompt_profile"],
                        "system_prompt_sha256": lane["system_prompt_sha256"],
                        "prompt_route": lane["prompt_route"],
                        "task": row.get("task") or args.task,
                        "dataset": row.get("dataset") or args.task,
                        "eval_id": row["eval_id"], "video_name": row.get("video_name"),
                        "source": row.get("source"),
                        # long-video's short/medium/long split. The harness can
                        # only matter where the clip does not fit without it, so
                        # a single pooled number would average the effect away.
                        "bucket": (row.get("metadata") or {}).get("bucket"),
                        "duration_s": (row.get("metadata") or {}).get("duration_s"),
                        "turns_total": len(lane["records"]),
                        "turns_run": len(lane["actions"]),
                        "full_trajectory": (lane["over_length_at"] is None
                                            and len(lane["actions"]) == len(lane["records"])),
                        "over_length_at": lane["over_length_at"],
                        "memory": mem_stats,
                        "response_generation": {
                            "default_mode": response_generation_default,
                            "mode_by_profile": response_generation_by_profile,
                            "effective_mode": lane["response_generation_mode"],
                            "content_projection": response_content_projection,
                            "action_max_tokens": 1,
                            "content_max_tokens": max(1, args.max_tokens - 1),
                            "control_token_ids": control_token_ids,
                            "action_model": args.model,
                            "content_model": args.model,
                            "same_model": True,
                            "postdecision_filter": False,
                            "content_calls": lane["response_content_calls"],
                            "content_failures": lane["response_content_failures"],
                        },
                        "verifier": {
                            "mode": args.response_verifier,
                            "action_policy": args.verifier_action_policy,
                            "frames": args.verify_frames,
                            "calls": lane["verifier_calls"],
                            "changed": lane["verifier_changed"],
                            "silenced": lane["verifier_silenced"],
                            "recommended_silence": lane["verifier_recommended_silence"],
                            "failed": lane["verifier_failed"],
                            "novelty_rescue": args.novelty_rescue,
                            "novelty_threshold": args.novelty_threshold,
                            "novelty_min_gap": args.novelty_min_gap,
                            "novelty_history": args.novelty_history,
                            "novelty_rescued": lane["novelty_rescued"],
                        },
                        "continual": lane["continual"].dump(),
                        "harness_policy": policy.dump(),
                        "predecision_only": args.predecision_only,
                        "query_mode": ("oneshot" if args.query_oneshot
                                       else args.query_mode),
                        "skills": {
                            "legacy_evidence_retrieval": lane["skills"].stats(),
                            "experience": (
                                experience_store.receipt()
                                if experience_store is not None else {
                                    "enabled": False,
                                    "loaded": [],
                                }),
                        },
                        "canonical_state": lane["adapter"].state_runtime.dump(),
                        "card": {"mode": args.card, **lane["epoch_guard"].dump()},
                        "perception": {
                            "enabled": event_enabled,
                            "backend": event_backend,
                            "prompt_profile": event_prompt_profile,
                            "profiles": sorted(event_profiles),
                            "max_active_criteria": event_max_active_criteria,
                            "active_criteria_count": len(
                                lane["adapter"].criteria.active),
                            "effective": bool(
                                event_enabled
                                and event_profile_enabled(
                                    lane["prompt_route"]["semantic_profile"],
                                    event_profiles)
                                and active_criteria_within_limit(
                                    len(lane["adapter"].criteria.active),
                                    event_max_active_criteria)),
                            "stride": event_stride,
                            "frames": event_frames,
                            "window": event_window,
                            "window_by_skill": event_window_by_skill,
                            "calls": lane["event_calls"],
                            "nonempty": lane["event_nonempty"],
                            "failed": lane["event_failed"],
                            "projection": lane["event_tracker"].dump(),
                            "instance_input_gating": event_instance_input_gating,
                            "decision_clock_enabled": decision_clock_enabled,
                        },
                        "transition_state": {
                            "enabled": transition_enabled,
                            "profiles": sorted(transition_profiles),
                            "render_visual_deltas": render_visual_transition_deltas,
                            "render_ordered_observation_deltas": (
                                render_ordered_observation_deltas),
                            "render_request_deltas": render_request_lifecycle_deltas,
                            "render_confirmed_evidence_deltas": (
                                render_confirmed_evidence_deltas),
                            "render_procedure_deltas": render_procedure_deltas,
                            "procedure_delta_mode": procedure_delta_mode,
                            "render_response_deltas": render_response_lifecycle_deltas,
                            "effective": lane["transition_state_effective"],
                            "state": lane["stream_state"].dump(),
                            "visual": lane["frame_transition"].dump(),
                        },
                        "context_projection": {
                            "control_role": control_role,
                            "header_memory": lane["header_memory_projection"],
                            "effective_header_memory": lane[
                                "effective_header_memory_projection"],
                            "header_memory_default": header_memory_projection,
                            "header_memory_by_profile": header_memory_by_profile,
                            "header_memory_by_question_facet": (
                                header_memory_by_question_facet),
                            "question_facet": lane["question_facet"],
                            "memory_redundant_with_raw": lane[
                                "memory_redundant_with_raw"],
                            "minimal_state_projection": lane[
                                "minimal_state_projection"],
                            "state_slots": sorted(state_slots_default),
                            "state_slots_by_profile": {
                                profile: sorted(slots)
                                for profile, slots in state_slots_by_profile.items()},
                            "state_slots_by_criterion": {
                                criterion: sorted(slots)
                                for criterion, slots in state_slots_by_criterion.items()},
                            "belief_query_mode_by_criterion": (
                                belief_query_mode_by_criterion),
                            "criterion_projection_requires_initial_presence": sorted(
                                criterion_projection_initial_gate),
                            "initial_criteria": sorted(lane["initial_criteria"]),
                            "effective_state_slots": sorted(lane["adapter"].slots),
                            "experience_prompt_projection": {
                                "default": experience_projection_default,
                                "by_profile": experience_projection_by_profile,
                                "compact_when_multiple_active": (
                                    compact_experience_when_multiple_active),
                                "active_criteria_count": len(
                                    lane["adapter"].criteria.active),
                                "effective": lane[
                                    "adapter"].experience_projection,
                            },
                            "event_lifetime": event_lifetime,
                            "effective_event_lifetime": (
                                "current_system"
                                if event_instance_input_gating else event_lifetime),
                            "commit_nonempty_only": commit_nonempty_only,
                            "decision_contract": decision_contract,
                            "response_history": response_history_mode,
                            "response_history_by_profile": response_history_by_profile,
                            "effective_response_history": lane["response_history_mode"],
                            "silence_history": silence_history_mode,
                            "silence_history_by_profile": silence_history_by_profile,
                            "silence_history_by_criterion": silence_history_by_criterion,
                            "silence_history_requires_specific_focus": sorted(
                                silence_history_focus_gate),
                            "effective_silence_history": lane["silence_history_mode"],
                            "effective_silence_history_scope": lane[
                                "silence_history_scope"],
                            "prompt_router": prompt_router,
                            "response_cooldown": response_cooldown,
                            "response_epoch": {
                                **response_epoch,
                                "evidence_replay_mode": "image_only",
                                "effective": lane["response_epoch_enabled"],
                                "effective_trigger": lane["response_epoch_trigger"],
                                "effective_lookback_turns": lane[
                                    "response_epoch_lookback"],
                                "effective_evidence_lookback_turns": lane[
                                    "response_epoch_evidence_lookback"],
                                "effective_burst_gap_turns": lane[
                                    "response_epoch_burst_gap"],
                                "semantic_profile": lane["prompt_route"]["semantic_profile"],
                            },
                        },
                        "visual": {"window": args.visual_window, "stride": args.visual_stride,
                                   "anchor": args.visual_anchor},
                        "cadence_gate": {**lane["gate"].dump(),
                                         "suppressed": lane.get("gate_suppressed", [])[:64]},
                        "adapter": lane["adapter"].dump(),
                        "epochs": lane["epoch_log"],
                        **({"memory_dump": lane["memory"].dump()}
                           if args.arm in ("harness", "harness_32") else {}),
                        "steps": [{
                            "turn": t, "second": lane["golds"][t]["second"],
                            "gold_action": lane["golds"][t]["gold_action"],
                            **({"window": row["answer_window"]}
                               if row.get("answer_window")
                               and lane["golds"][t]["gold_action"] == "response" else {}),
                            "pred_action": action_of(lane["raws"][t]),
                            "history_action": action_of(lane["actions"][t]),
                            "epoch": lane["turn_epoch"].get(t),
                            "context_epoch": lane["turn_context_epoch"].get(t),
                            "context_start": lane["turn_context_start"].get(t),
                            "evidence_start": lane["turn_evidence_start"].get(t),
                            "decision_prompt_profile": lane["decision_prompt_profiles"].get(t),
                            "decision_prompt_sha256": lane["decision_prompt_sha256"].get(t),
                            "decision_input": lane[
                                "decision_input_fingerprints"].get(t),
                            "response_generation": lane[
                                "response_generation_steps"].get(t),
                            **({"decision_replayed": True}
                               if t in lane["decision_replayed"] else {}),
                            **({"harness_delta": lane["turn_extras"][t][:2000]}
                               if lane["turn_extras"].get(t) else {}),
                            "response_phase": lane["response_phases"].get(t),
                            "silence_history_mode": lane[
                                "turn_silence_history_mode"].get(t),
                            "silence_history_start": lane[
                                "turn_silence_history_start"].get(t),
                            "verifier": (
                                "none" if t not in lane["preverify"] else
                                "rescued" if t in lane["novelty_rescue_turns"] else
                                "silenced" if action_of(lane["reviewed"].get(t, "")) == "silence" else
                                "pass"),
                            "gold_text": clean_text(lane["golds"][t]["gold_raw"]),
                            "pred_text": clean_text(lane["raws"][t]),
                            "pred_raw": lane["raws"][t][:1000],
                            **({"perception": {
                                "input_turns": list(lane["event_spans"][t]),
                                "text": lane["event_now"].get(t, "")[:1200],
                                "visible": lane["event_visible"].get(t, "")[:600],
                            }} if t in lane["event_spans"] else {}),
                            **({"decision_clock": lane["decision_clock"][t]}
                               if t in lane["decision_clock"] else {}),
                            **({"preverify_raw": lane["preverify"][t][:1000]}
                               if t in lane["preverify"] else {}),
                            **({"forced": True} if lane.get("forced") == t else {}),
                        } for t in sorted(lane["actions"])],
                    }, ensure_ascii=False) + "\n")
                for probe in lane["probes"]:
                    if probe["turn"] not in lane["preds"]:
                        continue
                    prediction = lane["preds"][probe["turn"]]
                    out.write(json.dumps({
                        "model": args.model, "arm": args.arm,
                        "task": row.get("task") or args.task,
                        "dataset": row.get("dataset") or args.task,
                        "context_regime": f"harness_{args.arm}",
                        "harness": {"name": HARNESS_NAME,
                                    "Ts": cfg.Ts, "M": cfg.M, "L": cfg.L,
                                    "history_silence": args.history_silence,
                                    "lookahead": cfg.effective_lookahead,
                                    "agent": agent_kind,
                                    "memory_parts": args.memory_parts,
                                    "response_verifier": args.response_verifier,
                                    "verifier_action_policy": args.verifier_action_policy,
                                    "verify_frames": args.verify_frames,
                                    "novelty_rescue": args.novelty_rescue,
                                    "novelty_threshold": args.novelty_threshold,
                                    "novelty_min_gap": args.novelty_min_gap,
                                    "continual_refiner": args.continual_refiner,
                                    "refine_every": args.refine_every,
                                    "refine_warmup": args.refine_warmup,
                                    "policy_name": policy.name,
                                    "response_generation": {
                                        "default_mode": response_generation_default,
                                        "mode_by_profile": (
                                            response_generation_by_profile),
                                        "effective_mode": lane[
                                            "response_generation_mode"],
                                        "content_projection": (
                                            response_content_projection),
                                        "action_max_tokens": 1,
                                        "content_max_tokens": max(
                                            1, args.max_tokens - 1),
                                        "same_model": True,
                                        "postdecision_filter": False,
                                    },
                                    "event_parser": {
                                        "enabled": event_enabled,
                                        "backend": event_backend,
                                        "prompt_profile": event_prompt_profile,
                                        "profiles": sorted(event_profiles),
                                        "max_active_criteria": (
                                            event_max_active_criteria),
                                        "active_criteria_count": len(
                                            lane["adapter"].criteria.active),
                                        "effective": bool(
                                            event_enabled
                                            and event_profile_enabled(
                                                lane["prompt_route"]["semantic_profile"],
                                                event_profiles)
                                            and active_criteria_within_limit(
                                                len(lane["adapter"].criteria.active),
                                                event_max_active_criteria)),
                                        "stride": event_stride,
                                        "frames": event_frames,
                                        "window": event_window,
                                        "window_by_skill": event_window_by_skill,
                                        "calls": lane["event_calls"],
                                        "nonempty": lane["event_nonempty"],
                                        "failed": lane["event_failed"],
                                        "projection": lane["event_tracker"].dump(),
                                        "instance_input_gating": (
                                            event_instance_input_gating),
                                        "decision_clock_enabled": decision_clock_enabled,
                                    },
                                    "transition_state": {
                                        "enabled": transition_enabled,
                                        "profiles": sorted(transition_profiles),
                                        "render_visual_deltas": (
                                            render_visual_transition_deltas),
                                        "render_ordered_observation_deltas": (
                                            render_ordered_observation_deltas),
                                        "render_request_deltas": (
                                            render_request_lifecycle_deltas),
                                        "render_confirmed_evidence_deltas": (
                                            render_confirmed_evidence_deltas),
                                        "render_procedure_deltas": (
                                            render_procedure_deltas),
                                        "procedure_delta_mode": procedure_delta_mode,
                                        "render_response_deltas": (
                                            render_response_lifecycle_deltas),
                                        "effective": lane[
                                            "transition_state_effective"],
                                        "state": lane["stream_state"].dump(),
                                        "visual": lane["frame_transition"].dump(),
                                    },
                                    "context_projection": {
                                        "control_role": control_role,
                                        "header_memory": header_memory_projection,
                                        "state_slots": sorted(state_slots_default),
                                        "state_slots_by_profile": {
                                            profile: sorted(slots)
                                            for profile, slots
                                            in state_slots_by_profile.items()},
                                        "state_slots_by_criterion": {
                                            criterion: sorted(slots)
                                            for criterion, slots
                                            in state_slots_by_criterion.items()},
                                        "effective_state_slots": sorted(
                                            lane["adapter"].slots),
                                        "experience_prompt_projection": {
                                            "default": experience_projection_default,
                                            "by_profile": (
                                                experience_projection_by_profile),
                                            "compact_when_multiple_active": (
                                                compact_experience_when_multiple_active),
                                            "active_criteria_count": len(
                                                lane["adapter"].criteria.active),
                                            "effective": lane[
                                                "adapter"].experience_projection,
                                        },
                                        "event_lifetime": event_lifetime,
                                        "effective_event_lifetime": (
                                            "current_system"
                                            if event_instance_input_gating
                                            else event_lifetime),
                                        "commit_nonempty_only": (
                                            commit_nonempty_only),
                                        "decision_contract": decision_contract,
                                        "response_history": response_history_mode,
                                        "response_history_by_profile": response_history_by_profile,
                                        "effective_response_history": lane[
                                            "response_history_mode"],
                                        "prompt_router": prompt_router,
                                        "response_cooldown": response_cooldown,
                                        "response_epoch": {
                                            **response_epoch,
                                            "evidence_replay_mode": "image_only",
                                            "effective": lane["response_epoch_enabled"],
                                            "effective_trigger": lane[
                                                "response_epoch_trigger"],
                                            "effective_lookback_turns": lane[
                                                "response_epoch_lookback"],
                                            "effective_evidence_lookback_turns": lane[
                                                "response_epoch_evidence_lookback"],
                                            "effective_burst_gap_turns": lane[
                                                "response_epoch_burst_gap"],
                                            "semantic_profile": lane["prompt_route"]["semantic_profile"],
                                        },
                                    },
                                    "query_mode": ("oneshot" if args.query_oneshot
                                                   else args.query_mode),
                                    **mem_stats},
                        "system_prompt_sha256": lane["decision_prompt_sha256"].get(
                            probe["turn"], lane["system_prompt_sha256"]),
                        "decision_input": lane[
                            "decision_input_fingerprints"].get(probe["turn"]),
                        "temperature": 0.0, "max_tokens": args.max_tokens,
                        "images_per_step": args.images_per_step,
                        "probe_id": f"{row['eval_id']}:{probe['turn']}",
                        "eval_id": row["eval_id"], "video_name": row.get("video_name"),
                        "video_path": row.get("video_path"), "source": row.get("source"),
                        "turn": probe["turn"], "second": probe["second"],
                        "probe_label": probe["probe_label"],
                        "gold_action": probe["gold_action"],
                        "pred_action": action_of(prediction),
                        "gold_raw": probe["gold_raw"], "pred_raw": prediction[:1000],
                        "gold_text": clean_text(probe["gold_raw"]),
                        "pred_text": clean_text(prediction),
                        "context_exhausted": (lane["over_length_at"] is not None
                                              and probe["turn"] >= lane["over_length_at"]),
                        "traj_turns": len(lane["actions"]),
                        "traj_responses": sum(1 for a in lane["raws"].values()
                                              if action_of(a) == "response"),
                        "traj_unwrapped": sum(1 for a in lane["raws"].values()
                                              if not is_wrapped(a)),
                    }, ensure_ascii=False) + "\n")
                    written += 1
            out.flush()
            if trace_handle is not None:
                trace_handle.flush()
            frame_cache.clear()
            purge_frame_disk()

    if trace_handle is not None:
        trace_handle.close()
    agent.close()

    if latency:
        hits = [x["cache_hit"] for x in latency if x["cache_hit"] is not None]
        per_lane = [x["step_s"] for x in latency]
        print(f"[harness] steps={len(latency)} "
              f"median stream latency={statistics.median(per_lane):.3f}s "
              f"p90={sorted(per_lane)[int(0.9 * len(per_lane))]:.3f}s "
              f"median prefix-cache hit={statistics.median(hits) if hits else None}",
              flush=True)
        if args.latency_output:
            args.latency_output.parent.mkdir(parents=True, exist_ok=True)
            args.latency_output.write_text(
                "\n".join(json.dumps(x) for x in latency), encoding="utf-8")
    print(f"[harness] wrote {written} rows -> {args.output} "
          f"(frame evictions {frame_evictions})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
