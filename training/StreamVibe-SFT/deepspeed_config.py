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

"""Generate the StreamVibe-SFT ZeRO-2 configuration for the requested world size."""
import argparse
import json
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--grad-accum", type=int, default=4)
    args = parser.parse_args()
    if args.world_size < 1 or args.grad_accum < 1:
        parser.error("world size and accumulation must be positive")
    config = {
        "bf16": {"enabled": True}, "gradient_accumulation_steps": args.grad_accum,
        "gradient_clipping": 1.0, "steps_per_print": 10,
        "train_batch_size": args.world_size * args.grad_accum,
        "train_micro_batch_size_per_gpu": 1, "wall_clock_breakdown": False,
        "zero_optimization": {
            "stage": 2, "overlap_comm": False, "contiguous_gradients": True,
            "reduce_bucket_size": 200000000, "allgather_bucket_size": 200000000,
            "reduce_scatter": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_name(args.output.name + f".{os.getpid()}.partial")
    temp.write_text(json.dumps(config, indent=2) + "\n")
    temp.replace(args.output)

if __name__ == "__main__":
    main()
