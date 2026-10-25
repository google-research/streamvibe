#!/usr/bin/env bash
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

set -euo pipefail
PACKAGE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$PACKAGE_ROOT/StreamVibe-RL/config.env"
: "${RL_SOURCE_MANIFEST:?Set the normalized input manifest}"
: "${PREP_OUTPUT:?Set a new directory for prepared RL data}"
: "${CT_OPENAI_API_KEY:?Supply the rubric judge credential in the runtime environment}"
[[ ! -e "$PREP_OUTPUT" ]] || { echo "Choose a new PREP_OUTPUT directory" >&2; exit 2; }
mkdir -p "$PREP_OUTPUT"
export RL_INTERMEDIATE_MANIFEST="$PREP_OUTPUT/intermediate.jsonl"
python3 "$PACKAGE_ROOT/data/convert_rl_annotations.py"
python3 "$PACKAGE_ROOT/data/build_rl_segments.py" \
  --rows "$RL_INTERMEDIATE_MANIFEST" --output "$PREP_OUTPUT/segments.jsonl" --cap 52 --segment
export CT_SHORT_SRC="$PREP_OUTPUT/segments.jsonl"
export CT_SHORT_OUT="$PREP_OUTPUT/short_windows.jsonl"
python3 "$PACKAGE_ROOT/data/build_rl_windows.py"
export CT_RUBRIC_SRC="$CT_SHORT_OUT"
export CT_RUBRIC_OUT="$PREP_OUTPUT/rubrics.jsonl"
python3 "$PACKAGE_ROOT/StreamVibe-RL/generate_rubrics.py"
SRC="$CT_SHORT_OUT" RUB="$CT_RUBRIC_OUT" OUT="$PREP_OUTPUT/StreamVibe-RL-train.jsonl" \
  python3 "$PACKAGE_ROOT/data/attach_rl_rubrics.py"
