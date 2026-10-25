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

"""Memory agents for StreamVibe Harness.

The mid-term agent summarizes frame chunks; the long-term agent consolidates
those summaries. A separate inference server keeps memory-agent requests from
evicting the online model's cached prefixes and permits different context limits.

Agent output is sanitized before it enters the online prompt so literal action
markers cannot be mistaken for the online model's control tokens.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import threading
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from .prompts import (
    LONG_TERM_AGENT_PROMPT,
    MEMORY_STYLES,
    MID_TERM_AGENT_PROMPT,
    PLAN_CUE_AGENT_PROMPT,
    REQUEST_BLIND_OBSERVATION_PROMPT,
    RESPONSE_VERIFIER_PROMPT,
    current_event_prompt,
)

CONTROL_MARKERS = ("</response>", "</silence>", "<|silent|>")


def sanitize(text: str) -> str:
    """Strip control markup an agent may have written into free text."""
    value = text or ""
    for marker in CONTROL_MARKERS:
        value = value.replace(marker, marker.strip("<>|/").upper())
    return value.strip()


def _done(value):
    fut: Future = Future()
    fut.set_result(value)
    return fut


class _TranscriptFuture:
    """Record one asynchronous agent result exactly once when it is joined."""

    def __init__(self, future, callback):
        self._future = future
        self._callback = callback
        self._lock = threading.Lock()
        self._resolved = False
        self._value = None

    def result(self, *args, **kwargs):
        with self._lock:
            if not self._resolved:
                value = self._future.result(*args, **kwargs)
                self._callback(value)
                self._value = value
                self._resolved = True
        return self._value


class MemoryTranscriptAgent:
    """Record or replay pre-decision cognitive-agent products for causal A/B runs.

    The transcript freezes only outputs produced *before* the interaction
    model: video summaries, long consolidation, request-side plans, and current
    event normalisation.  It never records or rewrites the interaction model outputs.  Entries are
    addressed by agent kind + trajectory id + per-kind causal ordinal, and a
    compact input signature must match during replay.  A missing or mismatched
    record raises synchronously so a candidate cannot silently degrade to an
    empty-memory run through ``memory._result``'s normal fail-open behaviour.
    """

    SCHEMA = "streamvibe-memory-transcript"

    def __init__(self, delegate, *, replay_input: Path | None = None,
                 record_output: Path | None = None):
        if replay_input is not None and record_output is not None:
            raise ValueError("memory transcript replay and recording are mutually exclusive")
        self.delegate = delegate
        self.replay_input = Path(replay_input) if replay_input is not None else None
        self.record_output = (Path(record_output)
                              if record_output is not None else None)
        self.mode = ("replay" if self.replay_input is not None
                     else "record" if self.record_output is not None else "off")
        self._lock = threading.Lock()
        self._ordinals: dict[tuple[str, str], int] = defaultdict(int)
        self._calls_by_lane: dict[str, int] = defaultdict(int)
        self._entries: dict[str, dict] = {}
        self._handle = None
        if self.replay_input is not None:
            if not self.replay_input.is_file():
                raise FileNotFoundError(
                    f"memory transcript does not exist: {self.replay_input}")
            for line_number, raw in enumerate(
                    self.replay_input.read_text(encoding="utf-8").splitlines(), 1):
                if not raw.strip():
                    continue
                entry = json.loads(raw)
                if entry.get("schema") != self.SCHEMA:
                    raise ValueError(
                        f"{self.replay_input}:{line_number}: bad memory transcript schema")
                key = str(entry.get("key") or "")
                if not key:
                    raise ValueError(
                        f"{self.replay_input}:{line_number}: empty key")
                previous = self._entries.get(key)
                if previous is not None:
                    valid_resolution = (
                        not previous.get("resolved", False)
                        and entry.get("resolved", False)
                        and previous.get("signature") == entry.get("signature"))
                    if not valid_resolution:
                        raise ValueError(
                            f"{self.replay_input}:{line_number}: invalid duplicate key "
                            f"{key!r}")
                self._entries[key] = entry
        elif self.record_output is not None:
            self.record_output.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.record_output.open("w", encoding="utf-8")

    @staticmethod
    def _text_hash(*values) -> str:
        payload = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _identity(self, kind: str, lane: str) -> tuple[str, int]:
        lane = str(lane or "")
        with self._lock:
            ordinal = self._ordinals[(lane, kind)]
            self._ordinals[(lane, kind)] += 1
            self._calls_by_lane[lane] += 1
        return f"{kind}:{lane}:{ordinal}", ordinal

    def _submit(self, kind: str, lane: str, signature: dict, submit):
        key, ordinal = self._identity(kind, lane)
        if self.mode == "replay":
            entry = self._entries.get(key)
            if entry is None:
                raise RuntimeError(f"missing frozen memory product {key}")
            if entry.get("signature") != signature:
                raise RuntimeError(
                    f"frozen memory input mismatch for {key}: "
                    f"expected={entry.get('signature')} actual={signature}")
            # An unresolved entry was submitted near the end of the control
            # stream but never joined into any Header.  The paired candidate
            # has the same stream geometry, so its value is intentionally
            # irrelevant and must not trigger a fresh nondeterministic call.
            return _done(str(entry.get("output") or ""))
        future = submit()
        if self.mode != "record":
            return future

        submitted = {
            "schema": self.SCHEMA,
            "key": key,
            "kind": kind,
            "lane": str(lane or ""),
            "ordinal": ordinal,
            "signature": signature,
            "resolved": False,
            "output": "",
        }
        with self._lock:
            self._handle.write(json.dumps(submitted, ensure_ascii=False) + "\n")
            self._handle.flush()

        def record(value) -> None:
            entry = {
                "schema": self.SCHEMA,
                "key": key,
                "kind": kind,
                "lane": str(lane or ""),
                "ordinal": ordinal,
                "signature": signature,
                "resolved": True,
                "output": str(value or ""),
            }
            encoded = json.dumps(entry, ensure_ascii=False) + "\n"
            with self._lock:
                self._handle.write(encoded)
                self._handle.flush()

        return _TranscriptFuture(future, record)

    def submit_mid(self, frames, t0, t1, lane="", focus=""):
        signature = {"frames": len(frames), "t0": float(t0), "t1": float(t1),
                     "focus": str(focus or "")}
        return self._submit(
            "mid", lane, signature,
            lambda: self.delegate.submit_mid(frames, t0, t1, lane=lane, focus=focus))

    def submit_long(self, summaries, t0, t1, lane="", focus=""):
        signature = {"summaries": len(summaries), "t0": float(t0),
                     "t1": float(t1), "focus": str(focus or ""),
                     "summary_sha256": self._text_hash(*summaries)}
        return self._submit(
            "long", lane, signature,
            lambda: self.delegate.submit_long(
                summaries, t0, t1, lane=lane, focus=focus))

    def submit_plan(self, request="", lane=""):
        signature = {"request": str(request or "")}
        return self._submit(
            "plan", lane, signature,
            lambda: self.delegate.submit_plan(request, lane=lane))

    def submit_event(self, frames, t0, t1, request="", reference="",
                     previous="", lane="", prompt_profile="base"):
        signature = {
            "frames": len(frames), "t0": float(t0), "t1": float(t1),
            "request": str(request or ""), "reference": str(reference or ""),
            "previous": str(previous or ""),
            "prompt_profile": str(prompt_profile or "base"),
        }
        return self._submit(
            "event", lane, signature,
            lambda: self.delegate.submit_event(
                frames, t0, t1, request=request, reference=reference,
                previous=previous, lane=lane, prompt_profile=prompt_profile))

    def submit_verify(self, frames, focus="", memory="", candidate="",
                      recent_answers="", lane=""):
        signature = {
            "frames": len(frames), "focus": str(focus or ""),
            "memory_sha256": self._text_hash(memory),
            "candidate_sha256": self._text_hash(candidate),
            "recent_answers_sha256": self._text_hash(recent_answers),
        }
        return self._submit(
            "verify", lane, signature,
            lambda: self.delegate.submit_verify(
                frames, focus=focus, memory=memory, candidate=candidate,
                recent_answers=recent_answers, lane=lane))

    def health(self) -> str:
        if self.mode == "replay":
            return f"memory-transcript:{len(self._entries)}"
        health = getattr(self.delegate, "health", None)
        return health() if health is not None else self.mode

    def transcript_state(self, lane: str = "") -> dict:
        return {
            "mode": self.mode,
            "entries_loaded": len(self._entries),
            "calls": int(self._calls_by_lane.get(str(lane or ""), 0)),
        }

    def close(self):
        try:
            if self._handle is not None:
                self._handle.close()
        finally:
            self.delegate.close()


class NullAgent:
    """Disable mid-term and long-term memory for ablation runs."""

    def submit_mid(self, frames, t0, t1, lane="", focus=""):
        return _done("")

    def submit_long(self, summaries, t0, t1, lane="", focus=""):
        return _done("")

    def submit_verify(self, frames, focus="", memory="", candidate="",
                      recent_answers="", lane=""):
        return _done(candidate)

    def submit_plan(self, request="", lane=""):
        return _done("")

    def submit_event(self, frames, t0, t1, request="", reference="",
                     previous="", lane="", prompt_profile="base"):
        return _done("")

    def close(self):
        pass


class FakeAgent:
    """Deterministic stand-in so the tiers can be tested without a GPU."""

    def __init__(self):
        self.mid_calls: list[tuple] = []
        self.long_calls: list[tuple] = []
        self.verify_calls: list[tuple] = []
        self.plan_calls: list[tuple] = []
        self.event_calls: list[tuple] = []

    def submit_mid(self, frames, t0, t1, lane="", focus=""):
        self.mid_calls.append((len(frames), t0, t1, lane, focus))
        return _done(f"MID({t0:.0f}-{t1:.0f};n={len(frames)})")

    def submit_long(self, summaries, t0, t1, lane="", focus=""):
        self.long_calls.append((len(summaries), t0, t1, lane, focus))
        return _done(f"LONG({t0:.0f}-{t1:.0f};m={len(summaries)})")

    def submit_verify(self, frames, focus="", memory="", candidate="",
                      recent_answers="", lane=""):
        self.verify_calls.append(
            (len(frames), focus, memory, candidate, recent_answers, lane))
        return _done(candidate)

    def submit_plan(self, request="", lane=""):
        self.plan_calls.append((request, lane))
        return _done(
            "REFERENCE STAGES: inspect state | carry out requested task\n"
            "CONTINUE IF: the same action remains visibly in progress\n"
            "RESPONSE-RELEVANT IF: a requested transition or visible error first appears\n"
            "DEVIATION IF: a visible state conflicts with the request\n"
            "RECOVERY REFERENCE: identify the mismatch and give one concrete correction")

    def submit_event(self, frames, t0, t1, request="", reference="",
                     previous="", lane="", prompt_profile="base"):
        self.event_calls.append(
            (len(frames), t0, t1, request, reference, previous, lane,
             prompt_profile))
        if prompt_profile == "causal_alignment":
            return _done(
                "[REQUEST-BLIND OBSERVATION]\n"
                "CURRENT ENTITIES: visible task object\n"
                "CURRENT ACTIONS: none\n"
                "CURRENT TEXT: none\n"
                "CURRENT RELATIONS: none\n"
                "TEMPORAL CHANGE: none\n"
                "[REQUEST ALIGNMENT]\n"
                "CURRENT STATE: visible task state\n"
                "VISUAL NOVELTY: CONTINUING\n"
                "TASK EVIDENCE: CONTEXT_ONLY\n"
                "MATCH KIND: NONE\n"
                "GROUNDING: NONE\n"
                "ANCHOR: none\n"
                "EVIDENCE: none")
        return _done(
            "CURRENT STATE: visible task state\n"
            "VISUAL NOVELTY: CONTINUING\n"
            "TASK EVIDENCE: CONTEXT_ONLY\n"
            "MATCH KIND: NONE\n"
            "EVIDENCE: none")

    def close(self):
        pass


class VLLMServerAgent:
    """Qwen3-VL-4B-Instruct behind a vLLM OpenAI-compatible server."""

    def __init__(self, base_url: str, model: str, *, mid_max_tokens: int = 768,
                 long_max_tokens: int = 512, max_frames: int = 64,
                 jpeg_quality: int = 80, workers: int = 8, timeout: float = 300.0,
                 memory_style: str = "evidence"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        if memory_style not in MEMORY_STYLES:
            raise ValueError(f"unknown memory style {memory_style!r}")
        self.memory_style = memory_style
        self.mid_prompt = MEMORY_STYLES[memory_style]
        self.mid_max_tokens = mid_max_tokens
        self.long_max_tokens = long_max_tokens


        self.max_frames = max_frames
        self.jpeg_quality = jpeg_quality
        self.timeout = timeout
        self._pool = ThreadPoolExecutor(max_workers=workers,
                                        thread_name_prefix="memagent")

    # -- plumbing -------------------------------------------------------------

    def _encode(self, image) -> str:
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=self.jpeg_quality)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    def _subsample(self, frames):
        n = len(frames)
        if n <= self.max_frames:
            return frames
        step = n / float(self.max_frames)
        return [frames[min(n - 1, int(i * step))] for i in range(self.max_frames)]

    def _chat(self, content, max_tokens: int, *, sanitise: bool = True) -> str:
        import requests

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
        r = requests.post(f"{self.base_url}/chat/completions", json=payload,
                          timeout=self.timeout)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
        return sanitize(text) if sanitise else (text or "").strip()

    # -- the two agents -------------------------------------------------------

    def _mid(self, frames, t0, t1, focus="") -> str:
        if not frames:
            return ""
        content = [{"type": "text",
                    "text": self.mid_prompt.format(
                        t0=t0, t1=t1, focus=focus or "No explicit request; retain major events.")}]
        for image in self._subsample(frames):
            content.append({"type": "image_url",
                            "image_url": {"url": self._encode(image)}})
        return self._chat(content, self.mid_max_tokens)

    def _long(self, summaries, t0, t1, focus="") -> str:
        if not summaries:
            return ""
        body = "\n\n".join(f"[summary {i + 1}] {s}" for i, s in enumerate(summaries))
        text = LONG_TERM_AGENT_PROMPT.format(n=len(summaries), t0=t0, t1=t1,
                                             summaries=body,
                                             focus=focus or "No explicit request; retain major events.")
        return self._chat([{"type": "text", "text": text}], self.long_max_tokens)

    def _verify(self, frames, focus="", memory="", candidate="",
                recent_answers="") -> str:
        content = [{
            "type": "text",
            "text": RESPONSE_VERIFIER_PROMPT.format(
                focus=focus or "No explicit request.",
                memory=memory or "No older evidence available.",
                candidate=candidate,
                recent_answers=recent_answers or "None.",
            ),
        }]
        for image in self._subsample(frames):
            content.append({"type": "image_url",
                            "image_url": {"url": self._encode(image)}})
        # Keep the action markers: unlike summaries, this result is parsed as
        # an actual streaming action.
        return self._chat(content, 96, sanitise=False)

    def _plan(self, request="") -> str:
        if not request:
            return ""
        prompt = PLAN_CUE_AGENT_PROMPT.format(request=request)
        return self._chat([{"type": "text", "text": prompt}], 256)

    def _event(self, frames, t0, t1, request="", reference="", previous="",
               prompt_profile="base") -> str:
        if not frames:
            return ""
        if prompt_profile == "causal_alignment":
            prior_blind = "none"
            marker = "[REQUEST-BLIND OBSERVATION]"
            alignment_marker = "[REQUEST ALIGNMENT]"
            if marker in (previous or ""):
                prior_blind = previous.split(marker, 1)[1]
                prior_blind = prior_blind.split(alignment_marker, 1)[0].strip()
            observe_prompt = REQUEST_BLIND_OBSERVATION_PROMPT.format(
                n=len(frames), t0=t0, t1=t1,
                previous_observation=prior_blind or "none")
            observe_content = [{"type": "text", "text": observe_prompt}]
            for image in self._subsample(frames):
                observe_content.append({"type": "image_url",
                                        "image_url": {"url": self._encode(image)}})
            observation = self._chat(observe_content, 160)
            align_prompt = current_event_prompt(prompt_profile).format(
                n=len(frames), t0=t0, t1=t1,
                request=request or "No active request.",
                reference=reference or "No typed reference available.",
                previous=previous or "none",
                observation=observation or "No supported observation.")
            alignment = self._chat(
                [{"type": "text", "text": align_prompt}], 160)
            return ("[REQUEST-BLIND OBSERVATION]\n" + observation
                    + "\n[REQUEST ALIGNMENT]\n" + alignment)
        prompt = current_event_prompt(prompt_profile).format(
            n=len(frames), t0=t0, t1=t1,
            request=request or "No active request.",
            reference=reference or "No typed reference available.",
            previous=previous or "none",
        )
        content = [{"type": "text", "text": prompt}]
        for image in self._subsample(frames):
            content.append({"type": "image_url",
                            "image_url": {"url": self._encode(image)}})
        return self._chat(content, 128)

    def chat_text(self, prompt: str, max_tokens: int = 768) -> str:
        """Synchronous text-only call (E-stage extract/consolidate; never on the real-time path)."""
        return self._chat([{"type": "text", "text": prompt}], max_tokens, sanitise=False)

    def submit_mid(self, frames, t0, t1, lane="", focus=""):
        return self._pool.submit(self._mid, frames, t0, t1, focus)

    def submit_long(self, summaries, t0, t1, lane="", focus=""):
        return self._pool.submit(self._long, summaries, t0, t1, focus)

    def submit_verify(self, frames, focus="", memory="", candidate="",
                      recent_answers="", lane=""):
        return self._pool.submit(
            self._verify, frames, focus, memory, candidate, recent_answers)

    def submit_plan(self, request="", lane=""):
        return self._pool.submit(self._plan, request)

    def submit_event(self, frames, t0, t1, request="", reference="",
                     previous="", lane="", prompt_profile="base"):
        return self._pool.submit(
            self._event, frames, t0, t1, request, reference, previous,
            prompt_profile)

    def health(self) -> str:
        """Fail loudly at startup if the agent is unreachable.

        `_result` deliberately swallows a per-call agent failure so one bad
        summary cannot kill a trajectory -- but that same tolerance means a
        server that never came up at all would silently turn the `harness` arm
        into the `short_only` arm, and the ablation would report a real-looking
        "no effect". So the reachability of the agent is checked ONCE, up front,
        where it can still be a hard error.
        """
        import requests

        r = requests.get(f"{self.base_url}/models", timeout=30)
        r.raise_for_status()
        names = [m.get("id") for m in r.json().get("data", [])]
        if self.model not in names:
            raise RuntimeError(
                f"memory agent at {self.base_url} serves {names}, not {self.model!r}")
        return self.model

    def close(self):
        self._pool.shutdown(wait=False, cancel_futures=True)


def build_agent(kind: str, **kw):
    if kind in ("none", "null", ""):
        return NullAgent()
    if kind == "fake":
        return FakeAgent()
    if kind == "vllm":
        base = kw.pop("base_url", None) or os.environ.get(
            "HARNESS_MEMORY_URL", "http://127.0.0.1:8765/v1")
        model = kw.pop("model", None) or os.environ.get(
            "HARNESS_MEMORY_MODEL", "memory-agent")
        return VLLMServerAgent(base, model, **kw)
    raise ValueError(f"unknown memory agent kind: {kind!r}")
