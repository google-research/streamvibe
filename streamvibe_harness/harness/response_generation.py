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

"""Same-model response continuation for the Streaming Video Harness.

The interaction model owns both decisions.  In ``action_then_content`` mode it
first emits exactly one learned control token from the ordinary decision
prompt.  Only when that token is ``</response>`` is the same model invoked
again, with the chosen token already prefixed, to continue the response body.

This is deliberately not a verifier, filter, router, or rewrite: the Harness
cannot change the first model token, and no continuation can turn a response
back into silence.  The small pure helpers live here so those invariants can be
unit-tested without loading vLLM.
"""
from __future__ import annotations

from copy import deepcopy


GENERATION_MODES = frozenset({"joint", "action_then_content"})
CONTENT_PROJECTIONS = frozenset({"answer_only", "answer_schema"})
ACTION_TOKENS = frozenset({"</silence>", "</response>"})

CONTENT_SUPPORT_TAG = "[RESPONSE CONTENT SUPPORT; action already chosen by StreamVibe-VL]"
CONTENT_SUPPORT_CONTRACT = (
    "Use this only to realize the body after the already-chosen </response> "
    "token. It cannot change whether to speak. Ground the wording in the "
    "causal request, frames, and state already present; do not emit another "
    "action token."
)
ANSWER_SCHEMA_CONTRACT = (
    "The action is already fixed to </response>. The format-only Skill line "
    "below is the mandatory schema for the response body, not a suggestion "
    "about when to speak. Fulfill the active user request directly. Do not "
    "narrate the frame unless the schema asks for description, reporting, or "
    "explanation. For guidance, address the user with exactly one concrete "
    "next action grounded in visible progress. Do not repeat an earlier body "
    "or emit another action token."
)


def committed_history_action(text: str, *, model_action: str,
                             response_committed: bool,
                             silence_token: str = "</silence>") -> str:
    """Serialize the response state that was actually shown to the user.

    The model action and displayed bytes are immutable.  A displayed non-empty
    response is preserved verbatim for later causal turns, including the rare
    case where the model emitted prose without its required wrapper.  An empty
    response proposal rejected by commit hygiene remains silence.
    """
    if model_action == "response":
        return text if response_committed else silence_token
    value = (text or "").strip()
    return text if (value.startswith("</")
                    or value.startswith("<|silent|>")) else silence_token


def strict_action_token(text: str) -> str | None:
    """Return a valid one-token action, otherwise ``None``.

    The staged path is intentionally stricter than the scorer.  A malformed
    first token falls back to ordinary joint generation from the same prompt;
    it is never guessed, repaired, or mapped by the Harness.
    """
    value = (text or "").strip()
    return value if value in ACTION_TOKENS else None


def leading_action_token(text: str) -> str | None:
    """Read a frozen joint generation's leading action without using its body."""
    value = (text or "").lstrip()
    for token in ACTION_TOKENS:
        if value.startswith(token):
            return token
    return None


def response_support_messages(messages: list[dict], support: str,
                              projection: str = "answer_only") -> list[dict]:
    """Copy a decision conversation and add content-only support to its system turn."""
    value = (support or "").strip()
    if not value:
        raise ValueError("response content support must be non-empty")
    copied = deepcopy(messages)
    if not copied or copied[0].get("role") != "system":
        raise ValueError("response content support requires a leading system turn")
    if projection not in CONTENT_PROJECTIONS:
        raise ValueError(f"unknown response content projection {projection!r}")
    contract = (ANSWER_SCHEMA_CONTRACT
                if projection == "answer_schema"
                else CONTENT_SUPPORT_CONTRACT)
    copied[0]["content"] = (
        str(copied[0].get("content") or "")
        + "\n\n"
        + CONTENT_SUPPORT_TAG
        + "\n"
        + contract
        + "\n"
        + value
    )
    return copied


def compose_response(action_token: str, continuation: str) -> str:
    """Serialize a chosen response token plus the same model's continuation.

    Only whitespace between the two model generations may be inserted.  The
    continuation is otherwise byte-preserved; in particular this helper never
    suppresses, replaces, audits, or rewrites model content.
    """
    if strict_action_token(action_token) != "</response>":
        raise ValueError("a response continuation requires </response>")
    body = continuation or ""
    separator = "" if not body or body[0].isspace() else " "
    return action_token.strip() + separator + body
