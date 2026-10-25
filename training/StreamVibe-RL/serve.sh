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
source "${RL_CONFIG:-$PACKAGE_ROOT/StreamVibe-RL/config.env}"
: "${SFT_MODEL:?Set SFT_MODEL to the exported SFT model directory}"
# One process per rollout GPU; allocate separate devices from the trainer.
exec python3 "$PACKAGE_ROOT/StreamVibe-RL/serve.py" \
  --model-path "$SFT_MODEL" --port "${ENGINE_PORT:-18000}" "$@"
