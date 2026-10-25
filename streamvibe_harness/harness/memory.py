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

"""Three-tier memory management for StreamVibe Harness.

Short-term memory retains raw frames. Mid-term memory holds up to M chunk
summaries; long-term memory holds up to L blocks consolidated from M summaries
each, with FIFO eviction. Coverage is Ts * (1 + M + L*M) seconds.

Summarization starts `lookahead` turns before a chunk boundary and joins at the
boundary. Frames arriving after submission carry into the next chunk to retain
continuous coverage. `stall_s` records time spent waiting at boundaries.
"""
from __future__ import annotations

import time
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Iterable

from .belief import BeliefIndex


@dataclass(frozen=True)
class MemoryConfig:
    """Chunk duration, tier capacities, and summarization lookahead."""

    Ts: int = 200
    M: int = 5
    L: int = 42
    lookahead: int | None = None  # default Ts // 4
    said_keep: int = 8            # most recent utterances kept in the memory text

    def __post_init__(self):
        if self.Ts < 2:
            raise ValueError(f"Ts must be >= 2, got {self.Ts}")
        if self.M < 1 or self.L < 0:
            raise ValueError(f"require M >= 1 and L >= 0, got M={self.M} L={self.L}")
        if self.effective_lookahead >= self.Ts:
            raise ValueError(
                f"lookahead {self.effective_lookahead} must be < Ts {self.Ts}; a "
                "summary submitted at or before the chunk start would cover no frames"
            )

    @property
    def effective_lookahead(self) -> int:
        return max(1, self.Ts // 4) if self.lookahead is None else int(self.lookahead)

    @property
    def coverage_s(self) -> int:
        """Maximum temporal coverage: Ts * (1 + M + L*M)."""
        return self.Ts * (1 + self.M + self.L * self.M)


@dataclass
class MidSummary:
    chunk: int
    t0: float
    t1: float
    text: str


@dataclass
class LongBlock:
    t0: float
    t1: float
    text: str


@dataclass
class SaidEvent:
    """A delivered response retained in the interaction history.
    
    Record every response, including narration, so it remains available after
    its raw conversation turn leaves short-term memory.
    """

    second: float
    query: str
    answer: str


class ThreeTierMemory:
    """Per-lane memory state machine. Pure: it never touches a model itself.

    The engine drives it with `before_step`, handing in a callable that can
    produce the frames for a turn range; the agent (if any) is what turns those
    frames into text.
    """

    def __init__(self, cfg: MemoryConfig, agent=None, lane_id: str = "",
                 parts_mode: str = "all", plan_config: dict | None = None,
                 strict_ordered_events: bool = False):
        if parts_mode not in ("all", "summaries", "said"):
            raise ValueError(f"bad parts_mode {parts_mode!r}")
        self.parts_mode = parts_mode
        self.cfg = cfg
        self.agent = agent
        self.lane_id = lane_id
        # `mid` is the current, not-yet-consolidated group.  Once it reaches M
        # summaries the whole group is moved into `_long_inputs` and submitted
        # to the long-term agent.  In particular this is NOT a rolling window:
        # each mid summary must belong to exactly one long block, otherwise the
        # nominal Ts*(1+M+L*M) coverage collapses to roughly Ts*(M+L).
        self.mid: deque[MidSummary] = deque(maxlen=cfg.M)
        self.long: deque[LongBlock] = deque(maxlen=cfg.L)
        self.qa: deque[SaidEvent] = deque(maxlen=cfg.said_keep)
        self.chunk = 0
        self._submitted: dict[int, object] = {}
        self._long_pending: object | None = None
        self._long_inputs: list[MidSummary] = []
        self._snapshot: str = ""
        self._snapshot_chunk: int = -1
        # Diagnostics, reported per lane.
        self.stall_s: float = 0.0
        self.n_mid: int = 0
        self.n_long: int = 0
        self.n_evicted: int = 0
        self.n_agent_failed: int = 0
        self.n_agent_empty: int = 0
        self.n_answer_dedup: int = 0
        self.n_plan_calls: int = 0
        self.n_plan_failed: int = 0
        self.focus_query: str = ""
        self.strict_ordered_events = bool(strict_ordered_events)
        from .planning import ProceduralPlanMemory
        self.plan = ProceduralPlanMemory.from_config(plan_config)
        self.belief = BeliefIndex(max_facts=max(64, cfg.M * max(1, cfg.L + 1) * 8))

    # -- geometry -------------------------------------------------------------

    def chunk_of(self, turn: int) -> int:
        return int(turn) // self.cfg.Ts

    def chunk_short_start(self, chunk: int) -> int:
        """First turn held as raw vision tokens while `chunk` is current.

        Chunk 0 starts at 0.  Every later chunk also carries the tail of its
        predecessor that the mid-term summary did not cover (see module
        docstring), so that no second falls between two tiers.
        """
        if chunk <= 0:
            return 0
        return max(0, chunk * self.cfg.Ts - self.cfg.effective_lookahead + 1)

    def short_start(self, turn: int) -> int:
        return self.chunk_short_start(self.chunk_of(turn))

    def submit_turn(self, chunk: int) -> int:
        """Turn at which chunk `chunk`'s mid-term summary is submitted."""
        return chunk * self.cfg.Ts + self.cfg.Ts - self.cfg.effective_lookahead

    # -- driving --------------------------------------------------------------

    def before_step(
        self,
        turn: int,
        frames_for: Callable[[int, int], list],
        second_of: Callable[[int], float],
    ) -> None:
        """Advance the memory to `turn`. Call once per turn, before prompt build."""
        cfg = self.cfg
        ci = self.chunk_of(turn)

        if self.agent is not None:
            sub = self.submit_turn(ci)
            if turn == sub and ci not in self._submitted:
                # The range summarised is what short-term actually HELD for this
                # chunk, which starts at chunk_short_start, not at ci*Ts. Those
                # differ by the tail carried over from the previous chunk, and
                # summarising from ci*Ts instead drops `lookahead - 1` seconds at
                # every single boundary -- a hole that widens with stream length
                # and is invisible in any per-step metric.
                lo, hi = self.chunk_short_start(ci), turn
                self._submitted[ci] = self.agent.submit_mid(
                    frames_for(lo, hi), second_of(lo), second_of(hi),
                    lane=self.lane_id, focus=self.focus_query
                )

        if ci != self.chunk:
            # Boundary: the chunk that just ended must be in the Memory Text
            # before the new chunk's one-time prefill happens.
            # A long-term job submitted M chunks ago has had a full chunk of
            # mainline inference in which to finish.  Join it before adding the
            # newest mid summary; this hides long consolidation instead of
            # submitting and immediately waiting at the same boundary.
            self._join_long()
            self._join_mid(ci - 1, second_of)
            self.chunk = ci
            self._snapshot = self._render()
            self._snapshot_chunk = ci
        elif self._snapshot_chunk != ci:
            self._snapshot = self._render()
            self._snapshot_chunk = ci

    def _join_mid(self, chunk: int, second_of) -> None:
        fut = self._submitted.pop(chunk, None)
        if fut is None:
            return
        t0 = time.perf_counter()
        text, failed = _result(fut)
        self.stall_s += time.perf_counter() - t0
        self.n_agent_failed += int(failed)
        if not text:
            self.n_agent_empty += int(not failed)
            return
        lo = self.chunk_short_start(chunk)
        hi = self.submit_turn(chunk)
        summary = MidSummary(chunk, second_of(lo), second_of(hi), text)
        self.plan.update_from_evidence(text, summary.t0, summary.t1)
        if self.cfg.L == 0 and len(self.mid) == self.mid.maxlen and self.mid:
            self.belief.drop_through(self.mid[0].t1)
        self.mid.append(summary)
        self.belief.update(
            text, summary.t0, summary.t1, source="mid", chunk=chunk,
            strict_ordered_events=self.strict_ordered_events)
        self.n_mid += 1
        if len(self.mid) == self.cfg.M and self.cfg.L > 0:
            # Each long-term block is consolidated from M
            # *consecutive* mid summaries. Move this complete, non-overlapping
            # group out of the live mid buffer. `_render` continues to expose
            # it while the asynchronous long job is pending, so no information
            # disappears during consolidation.
            if self._long_pending is not None:
                raise RuntimeError("previous long-term consolidation was not joined")
            self._long_inputs = list(self.mid)
            self.mid.clear()
            if self.agent is not None:
                self._long_pending = self.agent.submit_long(
                    [s.text for s in self._long_inputs],
                    self._long_inputs[0].t0,
                    self._long_inputs[-1].t1,
                    lane=self.lane_id,
                    focus=self.focus_query,
                )

    def _join_long(self) -> None:
        fut, self._long_pending = self._long_pending, None
        if fut is None:
            return
        t0 = time.perf_counter()
        text, failed = _result(fut)
        self.stall_s += time.perf_counter() - t0
        self.n_agent_failed += int(failed)
        inputs, self._long_inputs = self._long_inputs, []
        if not inputs:
            return
        if not text:
            # A failed/empty consolidation must not silently erase M chunks of
            # history. Fall back to the original summaries as one textual long
            # block; it is less compressed, but preserves coverage and keeps
            # later groups non-overlapping.
            self.n_agent_empty += int(not failed)
            text = "\n".join(s.text for s in inputs)
        if len(self.long) == self.long.maxlen and self.long.maxlen:
            self.belief.drop_through(self.long[0].t1)
            self.n_evicted += 1  # FIFO eviction
        self.long.append(
            LongBlock(inputs[0].t0, inputs[-1].t1, text)
        )
        self.n_long += 1

    # -- rendering ------------------------------------------------------------

    def memory_text(self) -> str:
        """The Memory Text for the CURRENT chunk.

        Snapshotted at the chunk boundary and never recomputed inside a chunk.
        Prefix-cache reuse requires that the
        Memory Text sits at the front of every request in the chunk, so if it
        changed mid-chunk every cached KV block behind it would be invalidated.
        """
        return self._snapshot

    def evidence_items(self):
        """Structured items for deterministic active-help retrieval skills."""
        from .skills import EvidenceItem
        items = [EvidenceItem(f.t0, f.t1, f.text, "fact")
                 for f in self.belief.retrieve(self.focus_query, k=24)]
        rendered_plan = self.plan.render()
        if rendered_plan:
            items.append(EvidenceItem(0.0, float(self.chunk * self.cfg.Ts),
                                      rendered_plan, "plan"))
        return items

    def belief_text(self, focus: str = "", k: int | None = None) -> str:
        """Structured B facts, separate from memory hierarchy/control prose."""
        return self.belief.render(focus, k)

    def _render(self) -> str:
        """Render the configured memory components.
        
        `summaries` includes mid-term and long-term visual summaries; `said` includes
        previously delivered responses; `all` includes both.
        """
        parts: list[str] = []
        want_summaries = self.parts_mode in ("all", "summaries")
        want_said = self.parts_mode in ("all", "said")
        plan_text = self.plan.render()
        if plan_text and want_summaries:
            parts.append(plan_text)
        if self.long and want_summaries:
            parts.append("## Older observed evidence")
            for b in self.long:
                parts.append(f"[{b.t0:.0f}s-{b.t1:.0f}s] {b.text}")
        visible_mid = [*self._long_inputs, *self.mid]
        if visible_mid and want_summaries:
            parts.append("## Recent observed evidence")
            for s in visible_mid:
                parts.append(f"[{s.t0:.0f}s-{s.t1:.0f}s] {s.text}")
        if self.qa and want_said:
            parts.append(
                "## Recent assistant outputs (not visual evidence)\n"
                "Use these only to avoid an exact duplicate when nothing changed. "
                "They do not complete the standing request and must not suppress "
                "a response to new evidence or a new state change."
            )
            for e in self.qa:
                if e.query:
                    parts.append(f"[{e.second:.0f}s] (for: {e.query}) {e.answer}")
                else:
                    parts.append(f"[{e.second:.0f}s] {e.answer}")
        return "\n".join(parts)

    def record_answer(self, second: float, query: str, answer: str) -> None:
        if answer:
            # The gate can repeat one hallucinated completion for many adjacent
            # seconds. Replaying eight byte-identical mistakes in the next
            # chunk both amplifies the mistake and creates a strong "stop
            # talking" prior. Keep one recency-updated copy instead.
            key = _answer_key(answer)
            duplicates = [e for e in self.qa if _answer_key(e.answer) == key]
            if duplicates:
                self.n_answer_dedup += len(duplicates)
                kept = [e for e in self.qa if _answer_key(e.answer) != key]
                self.qa.clear()
                self.qa.extend(kept)
            self.qa.append(SaidEvent(second, query or "", answer))

    def set_focus(self, query: str) -> None:
        """Set the standing request used by future memory-agent submissions."""
        self.focus_query = " ".join((query or "").split())[:1200]

    def set_plan_goal(self, query: str) -> bool:
        """Activate cached procedural state only for an explicit help request."""
        return self.plan.set_goal(query)

    def submit_plan_reference(self, query: str):
        """Launch a request-only plan/cue compile for a new active goal."""
        changed = self.plan.set_goal(query)
        if not self.plan.active or not self.plan.cue_planner_enabled:
            return None
        if not changed and self.plan.reference_stages:
            return None
        submit = getattr(self.agent, "submit_plan", None) if self.agent else None
        if submit is None:
            return None
        self.n_plan_calls += 1
        return submit(query, lane=self.lane_id)

    def join_plan_reference(self, future) -> bool:
        if future is None:
            return False
        text, failed = _result(future)
        self.n_plan_failed += int(failed)
        return self.plan.update_reference(text)

    def dump(self, mid_chars: int = 600, long_chars: int = 900) -> dict:
        """Return rendered memory text for inspecting the contents of each tier."""
        return {
            "long": [{"t0": b.t0, "t1": b.t1, "text": b.text[:long_chars]}
                     for b in self.long],
            "mid": [{"t0": s.t0, "t1": s.t1, "text": s.text[:mid_chars]}
                    for s in [*self._long_inputs, *self.mid]],
            "said": [{"second": e.second, "query": e.query[:200],
                      "answer": e.answer[:300]} for e in self.qa],
            "belief": self.belief.dump(),
            "timeline": self.belief.timeline_dump(),
        }

    def stats(self) -> dict:
        return {
            "mid_summaries": self.n_mid,
            "long_blocks": self.n_long,
            "long_evicted": self.n_evicted,
            "memory_stall_s": round(self.stall_s, 3),
            "agent_failed": self.n_agent_failed,
            "agent_empty": self.n_agent_empty,
            "answer_dedup": self.n_answer_dedup,
            "belief_facts": len(self.belief),
            "ordered_events": self.belief.event_count,
            "ordered_event_version": self.belief.event_version,
            "plan_calls": self.n_plan_calls,
            "plan_failed": self.n_plan_failed,
            "coverage_s": self.cfg.coverage_s,
            "plan": self.plan.dump(),
        }


def _result(fut) -> tuple[str, bool]:
    """Unwrap a future-ish. Returns (text, failed).

    A memory agent that times out or errors costs recall; it must not cost the
    trajectory, or an agent-side hiccup would be indistinguishable from the
    model choosing silence for the rest of the video. But the failure is
    COUNTED and reported on every row: a run whose agent quietly died is not a
    result about memory, and `agent_failed > 0` is what says so.
    """
    try:
        return ((fut.result() if hasattr(fut, "result") else fut) or ""), False
    except Exception as exc:  # noqa: BLE001
        print(f"[harness] memory agent failed, continuing without it: {exc}", flush=True)
        return "", True


def _answer_key(text: str) -> str:
    """Conservative identity key for repeated assistant outputs."""
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()
