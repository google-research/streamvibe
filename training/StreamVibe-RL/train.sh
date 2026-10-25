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
: "${RL_DATA:?Set RL_DATA to StreamVibe-RL-train.jsonl}"
: "${ENGINES_FILE:?Set ENGINES_FILE to a file with one engine URL per line}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to the RL run parent directory}"
NNODES=${NNODES:-4}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NODE_RANK=${NODE_RANK:-0}
if (( NNODES > 1 )); then
  : "${MASTER_ADDR:?Set the rendezvous host for multi-node training}"
else
  MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
fi
MASTER_PORT=${MASTER_PORT:-29501}
export CT_CONFIG_FALLBACK="$SFT_MODEL"
if [[ ${DRY_RUN:-0} != 1 ]]; then
  : "${CT_OPENAI_API_KEY:?Supply the judge credential in the runtime environment}"
  for path in "$SFT_MODEL/config.json" "$RL_DATA" "$ENGINES_FILE"; do
    [[ -s "$path" ]] || { echo "Missing required input: $path" >&2; exit 2; }
  done
fi
cmd=(torchrun --nnodes "$NNODES" --nproc-per-node "$GPUS_PER_NODE"
  --node-rank "$NODE_RANK" --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT"
  "$PACKAGE_ROOT/StreamVibe-RL/train.py" --exp "${EXP:-StreamVibe-RL}"
  --model-path "$SFT_MODEL" --data "$RL_DATA" --out-root "$OUTPUT_ROOT"
  --engines-file "$ENGINES_FILE" --publish-port "${PUBLISH_PORT:-18777}"
  --prompts-per-step 64 --group-size 8 --turns-per-traj 11 --lr 1e-6
  --steps "${STEPS:-500}" --save-freq "${SAVE_EVERY:-50}" --judge-concurrency 32
  --w-local 0.0 --max-pixels 32768 --clip-eps 0.2 --kl-beta 0.05 --sigma-min 0.1
  "$@")
if [[ ${DRY_RUN:-0} == 1 ]]; then printf '%q ' "${cmd[@]}"; printf '\n'; exit 0; fi
exec "${cmd[@]}"
