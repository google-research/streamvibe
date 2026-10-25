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

"""Serving configuration for StreamVibe Harness.

Only three serving policies are mutable.  Geometry (Ts/M/L), model weights and
arbitrary executable tools are intentionally outside this manifest.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .card import SLOTS
from .criteria import TYPES
from .experience import PROMPT_PROJECTIONS
from .response_generation import CONTENT_PROJECTIONS, GENERATION_MODES


HARNESS_ROOT = Path(__file__).resolve().parent.parent
MEMORY_STYLE_NAMES = {"evidence", "facts", "decision", "decision_ordered"}
VALID_REQUEST_PROFILES = {
    "physical_guidance", "digital_guidance", "explanation", "monitoring",
    "event_watch", "narration", "question_answering", "passive_observation",
    "fallback",
}


@dataclass
class HarnessPolicy:
    name: str = "StreamVibe Harness"
    status: str = "candidate"
    evidence_policy: dict = field(default_factory=dict)
    intervention_policy: dict = field(default_factory=dict)
    response_policy: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | None) -> "HarnessPolicy":
        if path is None:
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema") != "streamvibe-harness-policy":
            raise ValueError(f"unsupported harness policy schema in {path}")
        policy = cls(**{k: data.get(k) for k in (
            "name", "status", "evidence_policy",
            "intervention_policy", "response_policy", "provenance")})
        policy.validate()
        return policy

    def validate(self) -> None:
        intervention = self.intervention_policy or {}
        threshold = float(intervention.get("novelty_threshold", 0.6))
        gap = int(intervention.get("novelty_min_gap", 8))
        history = int(intervention.get("novelty_history", 8))
        if not 0.0 <= threshold <= 1.0 or not 0 <= gap <= 60 or not 1 <= history <= 64:
            raise ValueError("harness intervention policy is outside safe bounds")
        response = self.response_policy or {}
        if response.get("verifier", "agent") not in {"none", "agent"}:
            raise ValueError("response_policy.verifier must be none or agent")
        if response.get("action_policy", "allow_silence") not in {
                "preserve", "allow_silence"}:
            raise ValueError("response_policy.action_policy is invalid")
        if ("commit_nonempty_only" in response
                and not isinstance(response["commit_nonempty_only"], bool)):
            raise ValueError(
                "response_policy.commit_nonempty_only must be boolean")
        if not 1 <= int(response.get("verify_frames", 4)) <= 16:
            raise ValueError("response_policy.verify_frames must be in [1,16]")
        generation = response.get("generation", {})
        if not isinstance(generation, dict):
            raise ValueError("response_policy.generation must be an object")
        generation_mode = generation.get("mode", "joint")
        generation_by_profile = generation.get("mode_by_profile", {})
        if generation_mode not in GENERATION_MODES:
            raise ValueError(
                "response_policy.generation.mode must be joint or "
                "action_then_content")
        if (not isinstance(generation_by_profile, dict)
                or any(profile not in VALID_REQUEST_PROFILES
                       or mode not in GENERATION_MODES
                       for profile, mode in generation_by_profile.items())):
            raise ValueError(
                "response_policy.generation.mode_by_profile must map causal "
                "request profiles to joint or action_then_content")
        content_projection = generation.get(
            "content_projection", "answer_only")
        if content_projection not in CONTENT_PROJECTIONS:
            raise ValueError(
                "response_policy.generation.content_projection must be "
                "answer_only or answer_schema")
        staged_profiles = {
            profile for profile in VALID_REQUEST_PROFILES
            if generation_by_profile.get(profile, generation_mode)
            == "action_then_content"
        }
        if staged_profiles and (
                response.get("verifier", "agent") != "none"
                or response.get("action_policy", "allow_silence") != "preserve"):
            raise ValueError(
                "action_then_content requires verifier=none and "
                "action_policy=preserve")
        for component in (self.evidence_policy, self.response_policy):
            rules = component.get("rules", {}) if component else {}
            if not isinstance(rules, dict) or any(not isinstance(v, str)
                                                  for v in rules.values()):
                raise ValueError("policy rules must be a string dictionary")
        skills = (self.evidence_policy or {}).get("skills", {})
        if not isinstance(skills, dict):
            raise ValueError("evidence_policy.skills must be an object")
        if ("visible_retrieval" in skills
                and not isinstance(skills["visible_retrieval"], bool)):
            raise ValueError("skills.visible_retrieval must be boolean")
        if not 1 <= int(skills.get("retrieval_top_k", 2)) <= 4:
            raise ValueError("retrieval_top_k must be in [1,4]")
        if ("grounded_only" in skills
                and not isinstance(skills["grounded_only"], bool)):
            raise ValueError("skills.grounded_only must be boolean")
        if not 0 <= int(skills.get("min_grounding_overlap", 0)) <= 8:
            raise ValueError("skills.min_grounding_overlap must be in [0,8]")
        compatibility = skills.get("profile_compatibility", {})
        valid_skill_names = set(TYPES) | {"answer_question"}
        if (not isinstance(compatibility, dict)
                or any(profile not in VALID_REQUEST_PROFILES
                       or not isinstance(names, list)
                       or len(names) != len(set(names))
                       or any(name not in valid_skill_names for name in names)
                       for profile, names in compatibility.items())):
            raise ValueError(
                "skills.profile_compatibility must map request profiles to "
                "unique known Skill names")
        if ("general_taxonomy" in skills
                and not isinstance(skills["general_taxonomy"], bool)):
            raise ValueError("skills.general_taxonomy must be boolean")
        if ("compact_when_multiple_active" in skills
                and not isinstance(
                    skills["compact_when_multiple_active"], bool)):
            raise ValueError(
                "skills.compact_when_multiple_active must be boolean")
        prompt_projection = skills.get("prompt_projection", "legacy")
        if prompt_projection not in PROMPT_PROJECTIONS:
            raise ValueError(
                "skills.prompt_projection is unsupported")
        prompt_projection_by_profile = skills.get(
            "prompt_projection_by_profile", {})
        if (not isinstance(prompt_projection_by_profile, dict)
                or any(profile not in VALID_REQUEST_PROFILES
                       or value not in PROMPT_PROJECTIONS
                       for profile, value
                       in prompt_projection_by_profile.items())):
            raise ValueError(
                "skills.prompt_projection_by_profile must map causal request "
                "profiles to supported projections")
        prompt_projection_by_question_facet = skills.get(
            "prompt_projection_by_question_facet", {})
        valid_question_facets = {
            "observed_state", "immediate_intent", "assistance", "general"
        }
        if (not isinstance(prompt_projection_by_question_facet, dict)
                or any(facet not in valid_question_facets
                       or value not in PROMPT_PROJECTIONS
                       for facet, value
                       in prompt_projection_by_question_facet.items())):
            raise ValueError(
                "skills.prompt_projection_by_question_facet must map generic "
                "question facets to supported projections")
        skill_directory = skills.get("directory")
        if skill_directory is not None:
            if not isinstance(skill_directory, str) or not skill_directory.strip():
                raise ValueError("skills.directory must be a non-empty relative path")
            skill_path = Path(skill_directory)
            if skill_path.is_absolute() or ".." in skill_path.parts:
                raise ValueError("skills.directory must stay inside the harness root")
            if not (HARNESS_ROOT / skill_path).is_dir():
                raise ValueError(f"skills.directory does not exist: {skill_directory}")
        memory_style = (self.evidence_policy or {}).get("memory_style")
        if memory_style is not None and memory_style not in MEMORY_STYLE_NAMES:
            raise ValueError(
                f"evidence_policy.memory_style must be one of {sorted(MEMORY_STYLE_NAMES)}")
        plan = (self.evidence_policy or {}).get("plan_memory", {})
        if not isinstance(plan, dict):
            raise ValueError("evidence_policy.plan_memory must be an object")
        if not 1 <= int(plan.get("max_completed", 8)) <= 16:
            raise ValueError("plan_memory.max_completed must be in [1,16]")
        if not 1 <= int(plan.get("max_anchors", 12)) <= 24:
            raise ValueError("plan_memory.max_anchors must be in [1,24]")
        event_parser = (self.evidence_policy or {}).get("event_parser", {})
        if not isinstance(event_parser, dict):
            raise ValueError("evidence_policy.event_parser must be an object")
        if not 1 <= int(event_parser.get("stride", 1)) <= 8:
            raise ValueError("event_parser.stride must be in [1,8]")
        if event_parser.get("backend", "agent") not in {"agent", "decision_model"}:
            raise ValueError("event_parser.backend must be agent or decision_model")
        event_prompt_profile = event_parser.get("prompt_profile", "base")
        if event_prompt_profile not in {"base", "grounded_evidence", "causal_alignment"}:
            raise ValueError(
                "event_parser.prompt_profile must be base, grounded_evidence, or causal_alignment")
        if (event_prompt_profile == "causal_alignment"
                and event_parser.get("backend", "agent") != "agent"):
            raise ValueError(
                "event_parser causal_alignment requires the non-action agent backend")
        event_profiles = event_parser.get("profiles", [])
        if (not isinstance(event_profiles, list)
                or any(profile not in VALID_REQUEST_PROFILES
                       for profile in event_profiles)):
            raise ValueError(
                "event_parser.profiles contains an unknown request profile")
        if ("max_active_criteria" in event_parser
                and not 1 <= int(event_parser["max_active_criteria"]) <= 9):
            raise ValueError(
                "event_parser.max_active_criteria must be in [1,9]")
        if not 1 <= int(event_parser.get("frames", 2)) <= 4:
            raise ValueError("event_parser.frames must be in [1,4]")
        if not 1 <= int(event_parser.get("window", event_parser.get("frames", 2))) <= 32:
            raise ValueError("event_parser.window must be in [1,32]")
        if event_parser.get("render_mode", "raw") not in {"raw", "status", "edge"}:
            raise ValueError("event_parser.render_mode must be raw, status, or edge")
        if not 0 <= int(event_parser.get("repeat_gap", 8)) <= 60:
            raise ValueError("event_parser.repeat_gap must be in [0,60]")
        if not 0.0 <= float(event_parser.get("repeat_jaccard", 0.55)) <= 1.0:
            raise ValueError("event_parser.repeat_jaccard must be in [0,1]")
        for key in ("instance_input_gating", "rearm_on_reappearance_only"):
            if key in event_parser and not isinstance(event_parser[key], bool):
                raise ValueError(f"event_parser.{key} must be boolean")
        decision_clock = (self.evidence_policy or {}).get("decision_clock", {})
        if not isinstance(decision_clock, dict):
            raise ValueError("evidence_policy.decision_clock must be an object")
        transition_state = (self.evidence_policy or {}).get("transition_state", {})
        if not isinstance(transition_state, dict):
            raise ValueError("evidence_policy.transition_state must be an object")
        if "enabled" in transition_state and not isinstance(
                transition_state["enabled"], bool):
            raise ValueError("transition_state.enabled must be boolean")
        for key in ("render_visual_deltas", "render_ordered_observation_deltas",
                    "render_request_deltas",
                    "render_confirmed_evidence_deltas",
                    "render_procedure_deltas", "render_response_deltas"):
            if key in transition_state and not isinstance(
                    transition_state[key], bool):
                raise ValueError(f"transition_state.{key} must be boolean")
        if transition_state.get("procedure_delta_mode", "lifecycle") not in {
                "lifecycle", "observed_line"}:
            raise ValueError(
                "transition_state.procedure_delta_mode must be lifecycle or observed_line")
        minor = float(transition_state.get("minor_threshold", 0.025))
        major = float(transition_state.get("major_threshold", 0.14))
        if not 0 <= minor < major <= 1:
            raise ValueError("transition_state thresholds must satisfy 0 <= minor < major <= 1")
        if not 0 <= int(transition_state.get("edge_gap", 3)) <= 60:
            raise ValueError("transition_state.edge_gap must be in [0,60]")
        projection = (self.evidence_policy or {}).get("context_projection", {})
        if not isinstance(projection, dict):
            raise ValueError("evidence_policy.context_projection must be an object")
        state_slots = projection.get("state_slots")
        state_slots_by_profile = projection.get("state_slots_by_profile", {})
        state_slots_by_criterion = projection.get("state_slots_by_criterion", {})

        def valid_slots(slots, *, allow_empty: bool = False):
            return (isinstance(slots, list)
                    and (allow_empty or bool(slots))
                    and all(isinstance(slot, str) for slot in slots)
                    and len(slots) == len(set(slots))
                    and all(slot in SLOTS for slot in slots))

        if state_slots is not None and not valid_slots(state_slots):
            raise ValueError(
                "context_projection.state_slots must be a non-empty unique list "
                "of harness card slots")
        if (not isinstance(state_slots_by_profile, dict)
                or any(profile not in VALID_REQUEST_PROFILES
                       or not valid_slots(slots, allow_empty=True)
                       for profile, slots in state_slots_by_profile.items())):
            raise ValueError(
                "context_projection.state_slots_by_profile must map request "
                "profiles to unique harness card slot lists")
        valid_criteria = set(TYPES) | {"question"}
        if (not isinstance(state_slots_by_criterion, dict)
                or any(criterion not in valid_criteria
                       or not valid_slots(slots, allow_empty=True)
                       for criterion, slots in state_slots_by_criterion.items())):
            raise ValueError(
                "context_projection.state_slots_by_criterion must map typed "
                "criteria to unique harness card slot lists")
        slot_sets = ([] if state_slots is None else [state_slots]) + list(
            state_slots_by_profile.values()) + list(
                state_slots_by_criterion.values())
        for slots in slot_sets:
            if "symbolic" in slots and "experience" not in slots:
                raise ValueError(
                    "symbolic state projection requires experience in the same profile")
            if ("experience" in slots
                    and not bool(skills.get("enabled", True))):
                raise ValueError("experience state projection requires skills.enabled")
            if ("procedure" in slots
                    and not bool(plan.get("enabled", False))):
                raise ValueError("procedure state projection requires plan_memory.enabled")
            if ("lifecycle" in slots
                    and not bool(transition_state.get("enabled", False))):
                raise ValueError("lifecycle state projection requires transition_state.enabled")
        if staged_profiles:
            # E for a staged profile is deliberately absent from the action
            # prompt. It is rendered only after the interaction model's response token. A
            # criterion-scoped E slot could bypass that separation, so reject
            # it as well instead of relying on runtime conventions.
            for profile in staged_profiles:
                profile_slots = state_slots_by_profile.get(
                    profile, state_slots if state_slots is not None else [])
                if "experience" in profile_slots:
                    raise ValueError(
                        "action_then_content requires Experience to be absent "
                        f"from the action prompt for {profile}")
            if any("experience" in slots
                   for slots in state_slots_by_criterion.values()):
                raise ValueError(
                    "action_then_content is incompatible with a criterion-scoped "
                    "Experience action-prompt slot")
        for profile in prompt_projection_by_profile:
            profile_slots = state_slots_by_profile.get(
                profile, state_slots if state_slots is not None else [])
            if "experience" not in profile_slots:
                raise ValueError(
                    "skills.prompt_projection_by_profile requires an explicit "
                    f"experience slot for {profile}")
        if projection.get("control_role", "user") not in {"user", "system", "assistant"}:
            raise ValueError(
                "context_projection.control_role must be user, system, or assistant")
        if projection.get("header_memory", "all") not in {"all", "on_demand", "none"}:
            raise ValueError(
                "context_projection.header_memory must be all, on_demand, or none")
        memory_profiles = projection.get("header_memory_by_profile", {})
        valid_request_profiles = VALID_REQUEST_PROFILES
        transition_profiles = transition_state.get("profiles", [])
        if (not isinstance(transition_profiles, list)
                or any(value not in valid_request_profiles
                       for value in transition_profiles)):
            raise ValueError("transition_state.profiles contains an unknown request profile")
        if (not isinstance(memory_profiles, dict)
                or any(name not in valid_request_profiles
                       or value not in {"all", "on_demand", "none"}
                       for name, value in memory_profiles.items())):
            raise ValueError(
                "context_projection.header_memory_by_profile must map request profiles "
                "to all, on_demand, or none")
        memory_facets = projection.get("header_memory_by_question_facet", {})
        valid_question_facets = {
            "observed_state", "immediate_intent", "assistance", "general"
        }
        if (not isinstance(memory_facets, dict)
                or any(name not in valid_question_facets
                       or value not in {"all", "on_demand", "none", "nonredundant"}
                       for name, value in memory_facets.items())):
            raise ValueError(
                "context_projection.header_memory_by_question_facet must map "
                "generic question facets to all, on_demand, none, or nonredundant")
        transparent_facets = projection.get(
            "transparent_if_redundant_question_facets", [])
        if (not isinstance(transparent_facets, list)
                or any(name not in valid_question_facets
                       for name in transparent_facets)):
            raise ValueError(
                "context_projection.transparent_if_redundant_question_facets "
                "must list generic question facets")
        minimal_states = projection.get(
            "minimal_state_if_redundant_by_question_facet", {})
        if (not isinstance(minimal_states, dict)
                or any(name not in valid_question_facets
                       or not isinstance(cue, str) or not cue.strip()
                       for name, cue in minimal_states.items())):
            raise ValueError(
                "context_projection.minimal_state_if_redundant_by_question_facet "
                "must map generic question facets to nonempty answer-free cues")
        if projection.get("event_lifetime", "append") not in {"append", "current_system"}:
            raise ValueError(
                "context_projection.event_lifetime must be append or current_system")
        if projection.get("decision_contract", "none") not in {"none", "stable_router"}:
            raise ValueError(
                "context_projection.decision_contract must be none or stable_router")
        valid_response_history = {
            "full", "action_marker", "receipt", "active_receipt",
            "latest_full", "loop_latest_full", "state_only"}
        if projection.get("response_history", "full") not in valid_response_history:
            raise ValueError(
                "context_projection.response_history is invalid")
        history_profiles = projection.get("response_history_by_profile", {})
        if (not isinstance(history_profiles, dict)
                or any(profile not in valid_request_profiles
                       or value not in valid_response_history
                       for profile, value in history_profiles.items())):
            raise ValueError(
                "context_projection.response_history_by_profile is invalid")
        # Production/self-evolved policies may expose no silence receipt or
        # one prefix-stable receipt per visible epoch. Replaying every silent
        # action creates a growing action prior. The runner's legacy
        # ``--history-silence keep`` remains available only as an explicit
        # ablation outside the policy evolution surface.
        valid_silence_history = {"drop", "first_per_epoch"}
        if projection.get("silence_history", "drop") not in valid_silence_history:
            raise ValueError("context_projection.silence_history is invalid")
        silence_history_profiles = projection.get(
            "silence_history_by_profile", {})
        if (not isinstance(silence_history_profiles, dict)
                or any(profile not in valid_request_profiles
                       or value not in valid_silence_history
                       for profile, value in silence_history_profiles.items())):
            raise ValueError(
                "context_projection.silence_history_by_profile is invalid")
        silence_history_criteria = projection.get(
            "silence_history_by_criterion", {})
        if (not isinstance(silence_history_criteria, dict)
                or any(criterion not in valid_criteria
                       or value not in valid_silence_history
                       for criterion, value in silence_history_criteria.items())):
            raise ValueError(
                "context_projection.silence_history_by_criterion is invalid")
        silence_focus_gate = projection.get(
            "silence_history_requires_specific_focus", [])
        if (not isinstance(silence_focus_gate, list)
                or len(set(silence_focus_gate)) != len(silence_focus_gate)
                or any(criterion not in valid_criteria
                       or criterion not in silence_history_criteria
                       for criterion in silence_focus_gate)):
            raise ValueError(
                "context_projection.silence_history_requires_specific_focus "
                "must name configured criteria")
        belief_query_modes = projection.get(
            "belief_query_mode_by_criterion", {})
        if (not isinstance(belief_query_modes, dict)
                or any(criterion not in valid_criteria
                       or mode not in {"full_request", "question_stem"}
                       for criterion, mode in belief_query_modes.items())):
            raise ValueError(
                "context_projection.belief_query_mode_by_criterion is invalid")
        initial_presence_gate = projection.get(
            "criterion_projection_requires_initial_presence", [])
        configured_criterion_projections = (
            set(state_slots_by_criterion)
            | set(silence_history_criteria)
            | set(belief_query_modes)
        )
        if (not isinstance(initial_presence_gate, list)
                or len(set(initial_presence_gate)) != len(initial_presence_gate)
                or any(criterion not in valid_criteria
                       or criterion not in configured_criterion_projections
                       for criterion in initial_presence_gate)):
            raise ValueError(
                "context_projection."
                "criterion_projection_requires_initial_presence must name "
                "configured criteria")
        prompt_router = projection.get("prompt_router", {})
        if not isinstance(prompt_router, dict):
            raise ValueError("context_projection.prompt_router must be an object")
        profiles = prompt_router.get("profiles", {})
        if not isinstance(profiles, dict) or any(
                not isinstance(name, str) or not isinstance(profile, str)
                for name, profile in profiles.items()):
            raise ValueError("context_projection.prompt_router.profiles must map strings")
        criteria_profiles = prompt_router.get("criteria_profiles", {})
        if (not isinstance(criteria_profiles, dict)
                or any(criterion not in valid_criteria
                       or profile not in valid_request_profiles
                       for criterion, profile in criteria_profiles.items())):
            raise ValueError(
                "context_projection.prompt_router.criteria_profiles must map "
                "known criteria to request profiles")
        for flag in ("compose_active_criteria", "temporary_question_profile"):
            if flag in prompt_router and not isinstance(
                    prompt_router[flag], bool):
                raise ValueError(
                    f"context_projection.prompt_router.{flag} must be boolean")
        if (prompt_router.get("compose_active_criteria", False)
                and not criteria_profiles):
            raise ValueError(
                "context_projection.prompt_router.compose_active_criteria "
                "requires criteria_profiles")
        if prompt_router.get("refresh", "initial_request") not in {
                "initial_request", "causal_request_events"}:
            raise ValueError(
                "context_projection.prompt_router.refresh must be initial_request "
                "or causal_request_events")
        if (prompt_router.get("temporary_question_profile", False)
                and prompt_router.get("refresh", "initial_request")
                != "causal_request_events"):
            raise ValueError(
                "context_projection.prompt_router.temporary_question_profile "
                "requires refresh=causal_request_events")
        cooldown = projection.get("response_cooldown", {})
        if not isinstance(cooldown, dict):
            raise ValueError("context_projection.response_cooldown must be an object")
        if not 0 <= float(cooldown.get("seconds", 0)) <= 5:
            raise ValueError("context_projection.response_cooldown.seconds must be in [0,5]")
        if "enabled" in cooldown and not isinstance(cooldown["enabled"], bool):
            raise ValueError("context_projection.response_cooldown.enabled must be boolean")
        if "profile" in cooldown and not isinstance(cooldown["profile"], str):
            raise ValueError("context_projection.response_cooldown.profile must be a string")
        response_epoch = projection.get("response_epoch", {})
        if not isinstance(response_epoch, dict):
            raise ValueError("context_projection.response_epoch must be an object")
        if "enabled" in response_epoch and not isinstance(response_epoch["enabled"], bool):
            raise ValueError("context_projection.response_epoch.enabled must be boolean")
        if response_epoch.get("keep_last_response", True) is not True:
            raise ValueError(
                "context_projection.response_epoch.keep_last_response must be true")
        if not 0 <= int(response_epoch.get("lookback_turns", 0)) <= 32:
            raise ValueError(
                "context_projection.response_epoch.lookback_turns must be in [0,32]")
        lookback_by_profile = response_epoch.get("lookback_by_profile", {})
        if (not isinstance(lookback_by_profile, dict)
                or any(profile not in valid_request_profiles
                       or not 0 <= int(lookback) <= 32
                       for profile, lookback in lookback_by_profile.items())):
            raise ValueError(
                "context_projection.response_epoch.lookback_by_profile is invalid")
        if not 0 <= int(response_epoch.get("evidence_lookback_turns", 0)) <= 32:
            raise ValueError(
                "context_projection.response_epoch.evidence_lookback_turns "
                "must be in [0,32]")
        evidence_lookback_by_profile = response_epoch.get(
            "evidence_lookback_by_profile", {})
        if (not isinstance(evidence_lookback_by_profile, dict)
                or any(profile not in valid_request_profiles
                       or not 0 <= int(lookback) <= 32
                       for profile, lookback
                       in evidence_lookback_by_profile.items())):
            raise ValueError(
                "context_projection.response_epoch."
                "evidence_lookback_by_profile is invalid")
        if response_epoch.get(
                "evidence_replay_mode", "image_only") != "image_only":
            raise ValueError(
                "context_projection.response_epoch.evidence_replay_mode "
                "must be image_only")
        valid_epoch_triggers = {
            "every_response", "burst", "repeated_response", "loop_then_every",
            "burst_then_every"}
        if response_epoch.get("trigger", "every_response") not in valid_epoch_triggers:
            raise ValueError(
                "context_projection.response_epoch.trigger is invalid")
        if not 1 <= int(response_epoch.get("burst_gap_turns", 1)) <= 4:
            raise ValueError(
                "context_projection.response_epoch.burst_gap_turns must be in [1,4]")
        epoch_profiles = response_epoch.get("profiles", [])
        if (not isinstance(epoch_profiles, list)
                or any(value not in valid_request_profiles for value in epoch_profiles)):
            raise ValueError(
                "context_projection.response_epoch.profiles contains an unknown request profile")
        trigger_by_profile = response_epoch.get("trigger_by_profile", {})
        if (not isinstance(trigger_by_profile, dict)
                or any(profile not in valid_request_profiles
                       or trigger not in valid_epoch_triggers
                       for profile, trigger in trigger_by_profile.items())):
            raise ValueError(
                "context_projection.response_epoch.trigger_by_profile is invalid")
        burst_gap_by_profile = response_epoch.get("burst_gap_by_profile", {})
        if (not isinstance(burst_gap_by_profile, dict)
                or any(profile not in valid_request_profiles
                       or not 1 <= int(gap) <= 4
                       for profile, gap in burst_gap_by_profile.items())):
            raise ValueError(
                "context_projection.response_epoch.burst_gap_by_profile is invalid")
        if not 0.0 <= float(response_epoch.get("repeat_threshold", 0.9)) <= 1.0:
            raise ValueError(
                "context_projection.response_epoch.repeat_threshold must be in [0,1]")
        if not 1 <= int(response_epoch.get("repeat_history", 8)) <= 16:
            raise ValueError(
                "context_projection.response_epoch.repeat_history must be in [1,16]")
        if not 1 <= int(response_epoch.get("repeat_gap_turns", 16)) <= 60:
            raise ValueError(
                "context_projection.response_epoch.repeat_gap_turns must be in [1,60]")
        if projection.get("response_history") == "latest_full":
            raise ValueError(
                "latest_full response history must be profile-scoped with an "
                "every-response epoch")
        for profile, history_mode in history_profiles.items():
            if history_mode not in {"latest_full", "loop_latest_full"}:
                continue
            effective_trigger = trigger_by_profile.get(
                profile, response_epoch.get("trigger", "every_response"))
            required_trigger = ("every_response" if history_mode == "latest_full"
                                else "loop_then_every")
            if (not response_epoch.get("enabled", False)
                    or profile not in epoch_profiles
                    or effective_trigger != required_trigger):
                raise ValueError(
                    f"{history_mode} response history requires a {required_trigger} "
                    f"epoch for profile {profile}")
        window_by_skill = event_parser.get("window_by_skill", {})
        if not isinstance(window_by_skill, dict) or any(
                not isinstance(name, str) or not 1 <= int(value) <= 32
                for name, value in window_by_skill.items()):
            raise ValueError("event_parser.window_by_skill must map names to [1,32]")
        preferences = (self.evidence_policy or {}).get("preference_memory", {})
        if not isinstance(preferences, dict):
            raise ValueError("evidence_policy.preference_memory must be an object")
        if any(not isinstance(values, list) or
               any(not isinstance(value, str) for value in values)
               for values in preferences.values()):
            raise ValueError("preference memory values must be string lists")

    def apply_runtime(self, args) -> None:
        evidence = self.evidence_policy or {}
        intervention = self.intervention_policy or {}
        response = self.response_policy or {}
        if "memory_parts" in evidence:
            args.memory_parts = evidence["memory_parts"]
        if "said_keep" in evidence:
            args.said_keep = int(evidence["said_keep"])
        if "memory_style" in evidence:
            args.memory_style = str(evidence["memory_style"])
        projection = evidence.get("context_projection") or {}
        if ("state_slots" in projection
                or "state_slots_by_profile" in projection
                or "state_slots_by_criterion" in projection):
            # A production policy must be self-contained: omitting CARDS in a
            # launcher cannot silently turn B/P/E back into an empty shell.
            all_slots = list(projection.get("state_slots") or [])
            for slots in (projection.get("state_slots_by_profile") or {}).values():
                all_slots.extend(slots)
            for slots in (projection.get("state_slots_by_criterion") or {}).values():
                all_slots.extend(slots)
            args.card = ",".join(dict.fromkeys(all_slots)) or "shell"
        skills = evidence.get("skills") or {}
        if "directory" in skills:
            args.skill_dir = (HARNESS_ROOT / skills["directory"]).resolve()
        if "novelty_rescue" in intervention:
            args.novelty_rescue = bool(intervention["novelty_rescue"])
        if "novelty_threshold" in intervention:
            args.novelty_threshold = float(intervention["novelty_threshold"])
        if "novelty_min_gap" in intervention:
            args.novelty_min_gap = int(intervention["novelty_min_gap"])
        if "novelty_history" in intervention:
            args.novelty_history = int(intervention["novelty_history"])
        if "verifier" in response:
            args.response_verifier = response["verifier"]
        if "verify_frames" in response:
            args.verify_frames = int(response["verify_frames"])
        if "action_policy" in response:
            args.verifier_action_policy = response["action_policy"]

    def evidence_focus(self, active_request: str) -> str:
        evidence = self.evidence_policy or {}
        # Legacy policies used `rules` as summarizer instructions.  New
        # policies separate lifecycle/model-authority documentation from the
        # only rules B is allowed to see, preventing task-control prose from
        # leaking into video-fact summaries.
        rules = evidence.get("memory_focus_rules")
        if rules is None:
            rules = evidence.get("rules", {})
        body = "\n".join(f"- {value}" for value in rules.values())
        focus = active_request or "No active request; preserve generally reusable evidence."
        result = f"{focus}\nMemory focus policy:\n{body}" if body else focus
        if evidence.get("plan_memory", {}).get("enabled", False):
            result += (
                "\nPLAN MEMORY CONTRACT: The request may be procedural. When it is, "
                "append only visibly supported PLAN fields using the exact field "
                "names required by the summarization prompt."
            )
        return result

    def response_suffix(self) -> str:
        rules = (self.response_policy or {}).get("rules", {})
        parts = []
        if rules:
            body = "\n".join(f"- {value}" for value in rules.values())
            parts.append("[EVOLVED RESPONSE POLICY]\n" + body)
        preferences = (self.evidence_policy or {}).get("preference_memory", {})
        if preferences:
            lines = ["[USER PREFERENCE MEMORY; preferences, not visual facts]"]
            for key, values in preferences.items():
                if values:
                    lines.append(f"{key}: " + "; ".join(values[:12]))
            parts.append("\n".join(lines))
        return ("\n\n" + "\n\n".join(parts)) if parts else ""

    def response_focus(self, active_request: str) -> str:
        suffix = self.response_suffix().strip()
        return f"{active_request}\n{suffix}" if suffix else active_request

    def dump(self) -> dict:
        return {"schema": "streamvibe-harness-policy", "name": self.name, "status": self.status,
                "evidence_policy": self.evidence_policy or {},
                "intervention_policy": self.intervention_policy or {},
                "response_policy": self.response_policy or {},
                "provenance": self.provenance or {}}
def closed_choice_request(active_criteria: set[str] | frozenset[str]) -> bool:
    """Return whether typed request state requires a bounded choice answer."""
    return "answer_mcq" in active_criteria

