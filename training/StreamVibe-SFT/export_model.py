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

"""Export distributed SFT training state to Hugging Face model and processor assets."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from transformers import AutoModelForImageTextToText, AutoProcessor

# Copied from the base model rather than the checkpoint: training does not touch
# them, and the processor snapshot saved next to the checkpoint omits some.
BASE_ASSETS = (
    "config.json",
    "generation_config.json",
    "chat_template.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="directory holding the ZeRO tags, e.g. .../checkpoints",
    )
    parser.add_argument("--tag", default="epoch_00")
    parser.add_argument(
        "--model-state-bf16",
        action="store_true",
        help=(
            "load the complete BF16 module mapping from mp_rank_00_model_states.pt "
            "instead of reconstructing FP32 weights from optimizer shards; intended "
            "for evaluation-only recovery of a ZeRO-2 checkpoint"
        ),
    )
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument(
        "--processor-dir",
        type=Path,
        required=True,
        help="tokenizer_with_control_tokens written by the trainer",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"choose a new export directory: {args.output}")
    tag_dir = args.checkpoint_dir / args.tag
    if not tag_dir.is_dir():
        raise FileNotFoundError(f"no such checkpoint tag: {tag_dir}")

    staging = args.output.with_name(args.output.name + ".partial")
    if staging.exists():
        raise FileExistsError(f"export staging already exists: {staging}")
    staging.mkdir(parents=True)

    if args.model_state_bf16:
        model_state_path = tag_dir / "mp_rank_00_model_states.pt"
        if not model_state_path.is_file():
            raise FileNotFoundError(f"missing model state: {model_state_path}")
        print(f"[1/5] loading complete BF16 module from {model_state_path}", flush=True)
        payload = torch.load(
            model_state_path, map_location="cpu", mmap=True, weights_only=False
        )
        state_dict = payload.get("module")
        if not isinstance(state_dict, dict) or not state_dict:
            raise RuntimeError(f"checkpoint has no complete module mapping: {model_state_path}")
        non_bf16 = {
            str(value.dtype)
            for value in state_dict.values()
            if torch.is_tensor(value) and value.dtype != torch.bfloat16
        }
        if non_bf16:
            raise RuntimeError(f"model-state recovery expected BF16 tensors, got {non_bf16}")
    else:
        print(f"[1/5] consolidating ZeRO shards from {tag_dir}", flush=True)
        state_dict = get_fp32_state_dict_from_zero_checkpoint(
            str(args.checkpoint_dir), tag=args.tag
        )
    print(f"      {len(state_dict)} tensors", flush=True)

    print(f"[2/5] instantiating {args.base_model}", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.base_model, dtype=torch.bfloat16, local_files_only=True
    )

    print("[3/5] loading trained weights", flush=True)
    state_dict = {key: value.to(torch.bfloat16) for key, value in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    # ``freeze_vit_train_mergers`` froze model.visual.* apart from the two merger
    # stacks, and DeepSpeed can be configured to leave frozen parameters out of the
    # checkpoint.  Those keys are legitimately absent: the base weights already
    # sitting in the model are exactly what training kept.  Anything else missing
    # means the graph and the shards disagree, and a partial load that still scores
    # plausibly is much worse than stopping here.
    def is_frozen_vision(key: str) -> bool:
        if not key.startswith("model.visual."):
            return False
        tail = key[len("model.visual.") :]
        return not (
            tail.startswith("merger.") or tail.startswith("deepstack_merger_list.")
        )

    unexplained = [key for key in missing if not is_frozen_vision(key)]
    if unexplained or unexpected:
        raise RuntimeError(
            "state dict does not match the model graph; refusing to write a "
            f"partially-loaded checkpoint.\nmissing={sorted(unexplained)[:10]}\n"
            f"unexpected={sorted(unexpected)[:10]}"
        )
    print(
        f"      loaded; {len(missing)} frozen vision tensors kept from base", flush=True
    )

    processor = AutoProcessor.from_pretrained(args.processor_dir, local_files_only=True)
    embedding_rows = model.get_input_embeddings().weight.shape[0]
    highest_token = max(processor.tokenizer.get_vocab().values())
    if highest_token >= embedding_rows:
        raise RuntimeError(
            f"tokenizer reaches id {highest_token} but the embedding table has "
            f"{embedding_rows} rows; the base config can no longer be reused"
        )

    print(f"[4/5] writing bf16 safetensors to {staging}", flush=True)
    model.save_pretrained(staging, safe_serialization=True)
    processor.save_pretrained(staging)
    for name in BASE_ASSETS:
        source = args.base_model / name
        if source.is_file() and not (staging / name).is_file():
            shutil.copy2(source, staging / name)

    (staging / "export_provenance.json").write_text(
        json.dumps(
            {
                "source_checkpoint": str(tag_dir),
                "base_model": str(args.base_model),
                "processor_dir": str(args.processor_dir),
                "dtype": "bfloat16",
                "embedding_rows": int(embedding_rows),
                "highest_token_id": int(highest_token),
                "weight_source": (
                    "complete_bf16_model_state"
                    if args.model_state_bf16
                    else "zero_optimizer_shards_fp32"
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"[5/5] publishing {args.output}", flush=True)
    staging.rename(args.output)
    print("EXPORT_OK", args.output, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
