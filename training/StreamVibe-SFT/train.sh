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

# Run once on each allocated node with matching rendezvous settings.
set -euo pipefail
PACKAGE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
: "${BASE_MODEL:?Set BASE_MODEL to the local Qwen3.5-9B directory}"
: "${TRAIN_MANIFEST:?Set TRAIN_MANIFEST to StreamVibe-SFT-train.jsonl}"
: "${DEV_MANIFEST:?Set DEV_MANIFEST to StreamVibe-SFT-dev.jsonl}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a new SFT output directory}"
NNODES=${NNODES:-19}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
GRAD_ACCUM=${GRAD_ACCUM:-4}
NODE_RANK=${NODE_RANK:-0}
if (( NNODES > 1 )); then
  : "${MASTER_ADDR:?Set the rendezvous host for multi-node training}"
else
  MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
fi
MASTER_PORT=${MASTER_PORT:-29500}
export MIX_DIST_TIMEOUT_S=${MIX_DIST_TIMEOUT_S:-1800}
WORLD=$((NNODES * GPUS_PER_NODE))
DS_CONFIG="$OUTPUT_DIR/deepspeed.json"
if [[ ${DRY_RUN:-0} != 1 ]]; then
  for path in "$BASE_MODEL/config.json" "$TRAIN_MANIFEST" "$DEV_MANIFEST"; do
    [[ -f "$path" ]] || { echo "Missing required input: $path" >&2; exit 2; }
  done
  python3 "$PACKAGE_ROOT/StreamVibe-SFT/deepspeed_config.py" \
    --output "$DS_CONFIG" --world-size "$WORLD" --grad-accum "$GRAD_ACCUM"
fi
cmd=(torchrun --nnodes "$NNODES" --nproc-per-node "$GPUS_PER_NODE"
  --node-rank "$NODE_RANK" --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT"
  "$PACKAGE_ROOT/StreamVibe-SFT/train.py"
  --model "$BASE_MODEL" --train-manifest "$TRAIN_MANIFEST" --eval-manifest "$DEV_MANIFEST"
  --output-dir "$OUTPUT_DIR" --deepspeed-config "$DS_CONFIG"
  --epochs 1 --learning-rate 1e-5 --beta 0.9996 --lambda-text 2.5
  --turn-prompt-mode shared --loss-mode shared --turn-loss-ratio 0.5
  --max-updates "${MAX_UPDATES:-1385}" --warmup-ratio 0.03 --scheduler-total-updates 1385
  --gradient-accumulation-steps "$GRAD_ACCUM" --expected-world-size "$WORLD"
  --num-nodes "$NNODES" --gpus-per-node "$GPUS_PER_NODE" --num-workers 0
  --save-every-updates "${SAVE_EVERY:-50}" --log-every-updates 1 --keep-step-checkpoints 0)
[[ ${RESUME:-0} == 1 ]] && cmd+=(--resume)
cmd+=("$@")
if [[ ${DRY_RUN:-0} == 1 ]]; then printf '%q ' "${cmd[@]}"; printf '\n'; exit 0; fi
exec "${cmd[@]}"
