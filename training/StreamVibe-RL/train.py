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

"""Localized trajectory GRPO for StreamVibe.

The manuscript configuration combines 0.7 R_align + 0.3 R_glob and broadcasts
group-relative trajectory advantages to retained causal turns. A frozen SFT
reference supplies KL regularization. Distributed FSDP trainers publish policy
updates to persistent rollout workers."""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import math
import os
import random
import socketserver
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist

from frames import materialize_video_frame_window, frame_path, process_image
from prompting import build_messages_native, pick_gt_critical_steps, split_images
import reward as reward_mod
import rollout as rollout_mod


DEFAULTS = dict(
    prompts_per_step=64,
    group_size=8,
    turns_per_traj=11,
    lr=1e-6,
    clip_eps=0.2,
    kl_beta=0.05,
    sigma_min=0.1,
    w_local=0.5,
    m_response=1.0,
    m_window_silence=0.4,
    m_far_silence=0.0,
    window_mask_s=3.0,
    steps=50,
    save_freq=5,


    micro_batch_rows=int(os.environ.get("CT_MICRO_BATCH_ROWS", "4")),
    dapo_max_waves=2,
    judge_concurrency=48,
    malformed_abort_rate=0.20,
    malformed_abort_patience=3,
    min_pixels=3136,
    max_pixels=65536,
)


def trajectory_advantages(rewards: list[float], sigma_min: float) -> list[float]:
    mean = sum(rewards) / len(rewards)
    var = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    std = max(math.sqrt(var), sigma_min)
    return [(r - mean) / std for r in rewards]


# Manuscript: R = 0.7 R_align + 0.3 R_glob; no format reward.
# R_turn/rubric_c are the retained internal field names for these two terms.
W_TURN = float(os.environ.get("CT_W_TURN", "0.7"))
W_TRAJ = float(os.environ.get("CT_W_TRAJ", "0.3"))


def trajectory_objectives(scores: list[dict]) -> list[float]:
    """Return the scalar objective that GRPO normalizes for each lane.

    The optional trajectory-soft-F1 mode combines whole-trajectory soft-F1 with the trajectory rubric.
    ``CT_RUBRIC_W`` is a real coefficient here (zero disables the rubric call
    in reward.py and also removes it from the objective), not merely a gate.
    """
    if reward_mod.REWARD_MODE == "StreamVibe-RL":
        return [
            W_TURN * float(score.get("R_turn") or 0.0)
            + W_TRAJ * float(score.get("rubric_c") or 0.0)
            for score in scores
        ]
    if reward_mod.REWARD_MODE == "trajectory_soft_f1":
        return [
            float(score["R"])
            + reward_mod.RUBRIC_W * float(score.get("rubric_c") or 0.0)
            for score in scores
        ]
    return [float(score["R"]) for score in scores]


def same_second_baseline(turn_scores: list[dict[float, float]]) -> dict[float, float]:
    """Mean r_t per second across the group's lanes (lanes missing a second are
    excluded from that second's mean)."""
    sums: dict[float, float] = {}
    counts: dict[float, int] = {}
    for scores in turn_scores:
        for t, r in scores.items():
            sums[t] = sums.get(t, 0.0) + r
            counts[t] = counts.get(t, 0) + 1
    return {t: sums[t] / counts[t] for t in sums}


def turn_mask(kind: str, t: float, gt: dict, cfg: dict) -> float:
    """m_t: response turns 1.0, silence near a GT (targets + protected) 0.4,
    far silence 0.0.  Malformed turns count as response turns (their negative
    trajectory advantage must reach the malformed tokens)."""
    if kind != "silence":
        return cfg["m_response"]
    anchors = list(gt.get("help_points") or []) + list(gt.get("protected_help_points") or [])
    near = any(abs(t - reward_mod.hp_time(hp)) <= cfg["window_mask_s"] for hp in anchors)
    return cfg["m_window_silence"] if near else cfg["m_far_silence"]


def compose_group_rows(sample, lanes, scores, cfg, rng) -> list[dict]:
    """All training rows for one group, each carrying its scalar advantage."""
    rewards = trajectory_objectives(scores)
    a_traj = trajectory_advantages(rewards, cfg["sigma_min"])
    baseline = same_second_baseline([score["turn_scores"] for score in scores])
    # use_turn_rewards: pure turn-local GRPO.  The trajectory channel double-counted every
    # event and punished good turns for bad ones in the same lane; the mask
    # built to contain it zeroed 76% of all seconds.  A_t = r_t - same-second
    # group mean, and the only trajectory-level signal left is malformed -> -1.
    use_turn_rewards = reward_mod.REWARD_MODE == "turn_local"
    use_soft_f1 = reward_mod.REWARD_MODE == "trajectory_soft_f1"


    miss_abs = float(os.environ.get("CT_MISS_ABS", "0") or 0)
    use_streamvibe_reward = reward_mod.REWARD_MODE == "StreamVibe-RL"
    if use_soft_f1:
        # Trajectory GRPO: R = softF1 + lambda_rubric * rubric.
        # 每车道一个常数优势, 沉默行也算 —— 赢的车道的沉默压小了 F1 分母, 应分享这份推力。
        pass
    gt = sample["gt"]


    rubric_cs = [s.get("rubric_c") for s in scores]
    spoke = [c for c in rubric_cs if c is not None]
    rubric_mean = sum(spoke) / len(spoke) if len(spoke) >= 2 else None
    # 全组一致漏掉的 gold: 没有任何 lane 在其正窗口 (与 R_turn 同一套非对称
    # 窗口) 内开过口。只罚这类窗口 —— 只要有一条 lane 命中, 相对优势已在
    # 工作, 再叠加就是双重计价。
    unanimous_missed_golds: list[float] = []
    if miss_abs > 0:
        for _hp in (gt.get("help_points") or []):
            _gt_t = reward_mod.hp_time(_hp)
            _hit = any(
                _kind != "silence" and reward_mod.in_positive_window(_t, _gt_t)
                for _lane in lanes
                for (_t, _kind, _act) in _lane.turns
            )
            if not _hit:
                unanimous_missed_golds.append(_gt_t)
    rows = []
    for lane, score, a_i in zip(lanes, scores, a_traj):
        if not lane.steps:
            continue
        retained = pick_gt_critical_steps(
            lane.steps, lane.turns, gt.get("help_points") or [],
            cfg["turns_per_traj"], rng,
        )
        turn_by_t = {t: (kind, action) for t, kind, action in lane.turns}
        malformed = bool(score["metrics"].get("malformed"))
        for step in retained:
            t = float(step["at"])
            kind, action = turn_by_t[t]
            local = score["turn_scores"].get(t, 0.0) - baseline.get(t, 0.0)
            if use_streamvibe_reward:
                # Broadcast the two-term trajectory advantage to retained turns.
                advantage = a_i
                if (miss_abs > 0 and kind == "silence"
                        and any(reward_mod.in_positive_window(t, _g)
                                for _g in unanimous_missed_golds)):
                    advantage = a_i - miss_abs
            elif use_soft_f1:
                advantage = -1.0 if malformed else a_i
            elif use_turn_rewards:
                advantage = -1.0 if malformed else local
                # rubric rides only on response turns: silence contributed
                # nothing to the text ensemble and must not share its fate
                if (not malformed and rubric_mean is not None
                        and score.get("rubric_c") is not None and kind != "silence"):
                    advantage += reward_mod.RUBRIC_W * (score["rubric_c"] - rubric_mean)
            else:
                advantage = a_i * turn_mask(kind, t, gt, cfg) + cfg["w_local"] * local
            actions_before = {
                f: a for f, a in lane.actions.items() if f < step["at"]
            }
            rows.append(
                {
                    "problem": sample["problem"],
                    "at": step["at"],
                    "frames": step["frames"],
                    "actions_before": actions_before,
                    "action": action,
                    "kind": kind,
                    "advantage": float(advantage),
                    "video_path": sample["video_path"],
                    "required_frames": sample["required_frames"],
                }
            )
    return rows


# --------------------------------------------------------------------------- #
# Row tensorization (per rank)
# --------------------------------------------------------------------------- #
class RowBuilder:
    def __init__(self, processor, tokenizer, cfg):
        self.processor = processor
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self._dir_cache: dict[tuple[str, int], str] = {}

    def _frames_dir(self, video_path: str, required: list[int]) -> str:
        # Key by (video, furthest frame): two samples sharing a video but with
        # different windows must not reuse a shallower extraction (caching by
        # path alone skipped the extension and crashed 24 ranks on missing
        # frames).  fps=1 decoding materializes every frame <= furthest, so
        # equal-furthest reuse is safe regardless of the exact required set.
        key = (video_path, max(required))
        if key not in self._dir_cache:
            self._dir_cache[key] = materialize_video_frame_window(video_path, required)
        return self._dir_cache[key]

    def build(self, row: dict) -> dict:
        conv = build_messages_native(
            row["problem"], row["at"], row["frames"], row["actions_before"], interval_s=1.0
        )
        conv = [
            m if m["role"] != "user" else {"role": "user", "content": split_images(m["content"])}
            for m in conv
        ]
        tmpl_kwargs = {}
        if os.environ.get("CT_ENABLE_THINKING") == "0":
            # Qwen3.5 defaults to thinking mode, which destroys the streaming
            # decision format; the SFT and every qwen3.5 eval disable it.
            tmpl_kwargs["enable_thinking"] = False
        chat = self.processor.apply_chat_template(
            conv, tokenize=False, add_generation_prompt=True, **tmpl_kwargs)
        frames_dir = self._frames_dir(row["video_path"], row["required_frames"])
        images = [
            process_image(frame_path(frames_dir, f), self.cfg["min_pixels"], self.cfg["max_pixels"])
            for f in row["frames"]
        ]
        model_inputs = self.processor(images=images, text=[chat], return_tensors="pt")
        prompt_ids = model_inputs["input_ids"][0]
        target_ids = self.tokenizer(row["action"], add_special_tokens=False)["input_ids"] + [self.im_end]
        target_ids = torch.tensor(target_ids, dtype=prompt_ids.dtype)
        input_ids = torch.cat([prompt_ids, target_ids])
        loss_mask = torch.zeros(len(input_ids), dtype=torch.bool)
        loss_mask[len(prompt_ids):] = True
        out = {
            "input_ids": input_ids,
            "loss_mask": loss_mask,
            "advantage": row["advantage"],
            "kind": row["kind"],
        }
        for key in ("pixel_values", "image_grid_thw"):
            if key in model_inputs:
                out[key] = model_inputs[key]
        if "mm_token_type_ids" in model_inputs:
            # transformers 5.x Qwen3.5: per-token image/text marker; the
            # appended target tokens are all text (0)
            mm = model_inputs["mm_token_type_ids"][0]
            out["mm_token_type_ids"] = torch.cat(
                [mm, torch.zeros(len(target_ids), dtype=mm.dtype)]
            )
        return out


def _pad_stack(rows: list[dict], pad_id: int, device):
    max_len = max(len(r["input_ids"]) for r in rows)
    input_ids = torch.full((len(rows), max_len), pad_id, dtype=rows[0]["input_ids"].dtype)
    attention = torch.zeros((len(rows), max_len), dtype=torch.bool)
    loss_mask = torch.zeros((len(rows), max_len), dtype=torch.bool)
    for i, r in enumerate(rows):
        n = len(r["input_ids"])
        input_ids[i, :n] = r["input_ids"]
        attention[i, :n] = True
        loss_mask[i] = torch.nn.functional.pad(r["loss_mask"], (0, max_len - n))
    mm_ttids = None
    if "mm_token_type_ids" in rows[0]:
        mm_ttids = torch.zeros((len(rows), max_len), dtype=rows[0]["mm_token_type_ids"].dtype)
        for i, r in enumerate(rows):
            n = len(r["mm_token_type_ids"])
            mm_ttids[i, :n] = r["mm_token_type_ids"]
    pixel_values = torch.cat([r["pixel_values"] for r in rows]) if "pixel_values" in rows[0] else None
    grid = torch.cat([r["image_grid_thw"] for r in rows]) if "image_grid_thw" in rows[0] else None
    adv = torch.tensor([r["advantage"] for r in rows], dtype=torch.float32)
    batch = {
        "input_ids": input_ids.to(device),
        "attention_mask": attention.to(device),
        "loss_mask": loss_mask.to(device),
        "advantages": adv.to(device),
    }
    if pixel_values is not None:
        batch["pixel_values"] = pixel_values.to(device)
    if grid is not None:
        batch["image_grid_thw"] = grid.to(device)
    if mm_ttids is not None:
        batch["mm_token_type_ids"] = mm_ttids.to(device)
    return batch


def _token_logprobs(model, batch) -> torch.Tensor:
    kwargs = {k: batch[k] for k in ("pixel_values", "image_grid_thw", "mm_token_type_ids") if k in batch}
    logits = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        # use_cache must be OFF: Qwen3.5's linear-attention conv state lands in
        # the cache during the first forward, so checkpoint recompute sees a
        # +4-frame conv input and dies with a metadata mismatch.  (HF's own
        # checkpointing flag used to force this off; the torch-level wrapper
        # does not.)
        use_cache=False,
        **kwargs,
    ).logits[:, :-1]
    labels = batch["input_ids"][:, 1:]
    # Keep logits in model dtype and gather per row: a full-vocab FP32 copy +
    # log_softmax output is ~4x the bf16 logits and can exhaust GPU memory.
    # All three passes
    # (old / ref / update) share this exact path, so at lr=0 ratio == 1.
    rows = []
    for row_logits, row_labels in zip(logits, labels):
        row_lp = torch.log_softmax(row_logits, dim=-1)
        rows.append(torch.gather(row_lp, -1, row_labels.unsqueeze(-1)).squeeze(-1))
    return torch.stack(rows).float()  # [B, L-1]


def _shifted_mask(batch) -> torch.Tensor:
    return batch["loss_mask"][:, 1:]


# --------------------------------------------------------------------------- #
# HF checkpoint save (rank0, gathered full state dict)
# --------------------------------------------------------------------------- #
def save_hf_dir(state_dict: dict, out_dir: str, base_dir: str) -> list[str]:
    from safetensors.torch import save_file

    os.makedirs(out_dir, exist_ok=True)
    shard_size = 4 << 30
    shards: list[dict] = [{}]
    sizes = [0]
    for name, tensor in state_dict.items():
        nbytes = tensor.numel() * tensor.element_size()
        if sizes[-1] + nbytes > shard_size and shards[-1]:
            shards.append({})
            sizes.append(0)
        shards[-1][name] = tensor
        sizes[-1] += nbytes
    names, weight_map = [], {}
    for i, shard in enumerate(shards):
        name = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        save_file(shard, os.path.join(out_dir, name))
        names.append(name)
        for key in shard:
            weight_map[key] = name
    index = {"metadata": {"total_size": sum(sizes)}, "weight_map": weight_map}
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as fh:
        json.dump(index, fh)
    if not os.path.isdir(base_dir):
        alt = os.environ.get("CT_CONFIG_FALLBACK", "")
        print(f"[warn] config template dir gone: {base_dir}"
              + (f" -- falling back to {alt}" if alt else " -- skipping templates"),
              flush=True)
        base_dir = alt if alt and os.path.isdir(alt) else None
    for name in (os.listdir(base_dir) if base_dir else []):
        if name.endswith((".json", ".txt", ".jinja")) and not name.startswith("model"):
            src = os.path.join(base_dir, name)
            dst = os.path.join(out_dir, name)
            if os.path.isfile(src) and not os.path.exists(dst):
                with open(src, "rb") as s, open(dst, "wb") as d:
                    d.write(s.read())
    return names + ["model.safetensors.index.json"]


class _PublishServer:
    """Serves the publish root over HTTP for the engine nodes."""

    def __init__(self, root: str, port: int):
        os.makedirs(root, exist_ok=True)
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=root)
        handler.log_message = lambda *a, **k: None

        class Server(socketserver.ThreadingTCPServer):
            allow_reuse_address = True
            daemon_threads = True

        self.server = Server(("0.0.0.0", port), handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


# A healthy weight update is a local-disk load: ~3s per engine plus the 17GB
# download.  1800s meant one hung engine stalled the whole run for 30 minutes
# per attempt (x4 retries = 2 hours) before the pool-drop logic could fire.
def _post_json(url: str, payload: dict, timeout: float = 420.0) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--engines-file", required=True)
    parser.add_argument("--publish-port", type=int, default=18777)
    parser.add_argument("--judge-mode", default="gemini", choices=["gemini", "lexical", "none"])
    parser.add_argument("--seed", type=int, default=20260821)
    # Resume: point --model-path at a saved checkpoint, keep --ref-model-path
    # at the ORIGINAL SFT base (the KL anchor must not move), and skip the
    # samples already consumed.
    parser.add_argument("--ref-model-path", default=None)
    parser.add_argument("--start-step", type=int, default=1)
    parser.add_argument("--data-skip", type=int, default=0)
    for key, value in DEFAULTS.items():
        arg = "--" + key.replace("_", "-")
        parser.add_argument(arg, type=type(value), default=value)
    args = parser.parse_args()
    cfg = {key: getattr(args, key) for key in DEFAULTS}

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(hours=6))
    # FSDP's HYBRID_SHARD builds its intra/inter-node groups with a bare
    # dist.new_group() (torch/distributed/fsdp/_init_utils.py:242), so those
    # groups keep the 10-minute NCCL default rather than the 6h set above.  On
    # the longer jitter trajectories one lagging rank then blows an ALLREDUCE
    # timeout and takes all 64 ranks down.  Make derived groups inherit it.
    _orig_new_group = dist.new_group
    def _new_group_long_timeout(*a, **kw):
        kw.setdefault("timeout", timedelta(minutes=30))
        return _orig_new_group(*a, **kw)
    dist.new_group = _new_group_long_timeout
    import torch.distributed.distributed_c10d as _c10d
    _c10d.new_group = _new_group_long_timeout
    device = torch.device("cuda", local_rank)

    out_dir = os.path.join(args.out_root, args.exp)
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        # The guard protects against silently overwriting a finished run, but it
        # also made --start-step unusable: every resume died on "output dir not
        # empty" before reaching the training loop, so the crash-recovery path
        # was dead code all along.  A resume is exactly the case where the dir
        # is expected to be populated.
        if os.listdir(out_dir) and args.start_step <= 1:
            raise RuntimeError(f"output dir not empty: {out_dir}")

    from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
    from torch.distributed.fsdp import (
        FullStateDictConfig,
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
        StateDictType,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    processor = AutoProcessor.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    def load_wrapped(train: bool):
        source = args.model_path if train else (args.ref_model_path or args.model_path)
        # AutoModel resolves the family from config (Qwen3-VL and Qwen3.5 both);
        # flash-attn may be absent in this env -- fall back to SDPA.
        model = AutoModelForImageTextToText.from_pretrained(
            source, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        # transformers 5.x + FSDP: the HF gradient_checkpointing_enable flag
        # is a no-op in this combo (FIRST-fwd probe still held ~50GB and the
        # run OOMed at step 3) -- layers are instead checkpoint-wrapped at the
        # torch level AFTER FSDP construction, below.
        layer_classes = {
            type(layer)
            for layer in model.modules()
            if type(layer).__name__.endswith(("DecoderLayer", "VisionBlock", "Qwen3VLVisionBlock"))
        }
        wrapped = FSDP(
            model,
            # HYBRID_SHARD: shard within each 8-GPU node, replicate across
            # nodes.  All-gathers stay on NVLink; only grad reduction crosses
            # the network.  8.77B bf16 / 8-way shard fits easily.
            sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            auto_wrap_policy=functools.partial(
                transformer_auto_wrap_policy, transformer_layer_cls=layer_classes
            ),
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.bfloat16,
            ),
            device_id=device,
            use_orig_params=True,
        )


        if train and os.environ.get("CT_GRAD_CKPT", "0") == "1":
            # llama-recipes recipe: wrap decoder/vision layers inside the FSDP
            # tree so backward recomputes activations instead of holding them
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                CheckpointImpl,
                apply_activation_checkpointing,
                checkpoint_wrapper,
            )
            names = {c.__name__ for c in layer_classes}
            apply_activation_checkpointing(
                wrapped,
                checkpoint_wrapper_fn=functools.partial(
                    checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
                ),
                check_fn=lambda m: type(m).__name__ in names,
            )
        if not train:
            wrapped.eval()
            for p in wrapped.parameters():
                p.requires_grad_(False)
        return wrapped

    policy = load_wrapped(train=True)
    ref = load_wrapped(train=False)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg["lr"], weight_decay=0.0)

    builder = RowBuilder(processor, tokenizer, cfg)
    pad_id = tokenizer.pad_token_id or 0

    engines: list[str] = []
    dataset: list[dict] = []
    publish_root = os.path.join(os.environ.get("STREAMVIBE_PUBLISH_ROOT", "outputs/published_weights"), args.exp)
    if rank == 0:
        with open(args.engines_file) as fh:
            engines = [line.strip() for line in fh if line.strip()]
        with open(args.data) as fh:
            dataset = [json.loads(line) for line in fh]
        random.Random(args.seed).shuffle(dataset)
        _PublishServer(publish_root, args.publish_port)
        print(f"[train] engines={len(engines)} dataset={len(dataset)} out={out_dir}", flush=True)

    rollout_cfg = rollout_mod.RolloutConfig(
        engine_urls=engines,
        temperature=1.0,
        top_p=1.0,
        min_pixels=cfg["min_pixels"],
        max_pixels=cfg["max_pixels"],
        seed_base=args.seed,
    )
    # resuming past one epoch gives data_skip > len(dataset); keep it in range
    data_cursor = args.data_skip % max(1, len(dataset))
    rng = np.random.default_rng(args.seed)
    metrics_path = os.path.join(out_dir, "metrics.jsonl")
    hostname = os.environ.get("PUBLISH_HOST", os.uname().nodename)
    malformed_streak = 0

    def _draw(n: int) -> list[dict]:
        nonlocal data_cursor
        out = []
        while len(out) < n:
            if data_cursor >= len(dataset):
                random.Random(args.seed + data_cursor).shuffle(dataset)
                data_cursor = 0        # fresh epoch; data_skip may exceed len(dataset)
            out.append(dataset[data_cursor])
            data_cursor += 1
        return out

    def next_samples(count: int) -> list[dict]:
        rows = _draw(count)
        prepared = []
        skipped = 0
        def _prep(r):
            # A single unusable row must not kill a multi-hour run: some clips
            # are shorter than their annotated window, so 1-FPS extraction of
            # the last required frame raises.  Drop the row and carry on; the
            # count is reported so silent shrinkage stays visible.
            try:
                return rollout_mod.prepare_sample(r, rollout_cfg)
            except Exception as exc:
                return exc
        with ThreadPoolExecutor(max_workers=16) as pool:
            for sample, row in zip(pool.map(_prep, rows), rows):
                if isinstance(sample, Exception):
                    skipped += 1
                    if skipped <= 3:
                        print(f"[warn] skipping row {row.get('id')}: {sample}", flush=True)
                    continue
                videos = row["videos"]
                sample["video_path"] = videos if isinstance(videos, str) else list(videos)[0]
                start, end = sample["start"], sample["end"]
                prefix = sorted(sample["prefix_actions"])
                sample["required_frames"] = prefix + list(range(start, end + 1))
                prepared.append(sample)
        if skipped:
            print(f"[warn] {skipped}/{len(rows)} rows unusable this step", flush=True)
        
        
        attempts = 0
        while len(prepared) < count and attempts < 8:
            attempts += 1
            extra = _draw(max(4, (count - len(prepared)) * 2))
            with ThreadPoolExecutor(max_workers=16) as pool:
                for sample, row in zip(pool.map(_prep, extra), extra):
                    if isinstance(sample, Exception):
                        continue
                    videos = row["videos"]
                    sample["video_path"] = videos if isinstance(videos, str) else list(videos)[0]
                    start, end = sample["start"], sample["end"]
                    prefix = sorted(sample["prefix_actions"])
                    sample["required_frames"] = prefix + list(range(start, end + 1))
                    prepared.append(sample)
                    if len(prepared) >= count:
                        break
            print(f"[warn] topped up to {len(prepared)}/{count} (attempt {attempts})", flush=True)
        if not prepared:
            raise RuntimeError("every row in this batch was unusable after retries")
        return prepared[:count]

    def rollout_and_score(samples: list[dict], gen_call: int):
        lanes = rollout_mod.rollout_batch(samples, cfg["group_size"], rollout_cfg, gen_call)
        by_sample: dict[int, list] = {}
        for lane in lanes:
            by_sample.setdefault(lane.sample_index, []).append(lane)
        results = []

        def score_one(item):
            index, sample = item
            group = sorted(by_sample[index], key=lambda ln: ln.g)
            steps = [reward_mod.steps_from_turns(lane.turns, getattr(lane, "margins", None)) for lane in group]
            scores = reward_mod.score_group(
                sample["problem"], sample["gt"], steps, judge_mode=args.judge_mode
            )
            return sample, group, scores

        with ThreadPoolExecutor(max_workers=max(1, cfg["judge_concurrency"])) as pool:
            for result in pool.map(score_one, list(enumerate(samples))):
                results.append(result)
        return results

    for step in range(args.start_step, cfg["steps"] + 1):
        step_t0 = time.time()
        rows_payload: list[dict] = []
        step_stats: dict = {}
        if rank == 0:
            groups = rollout_and_score(next_samples(cfg["prompts_per_step"]), gen_call=step * 10)

            def _degenerate(scores) -> bool:
                # use_turn_rewards trains on per-second advantages, so "no signal" means no
                # second where the lanes' r_t differ -- not zero R variance.
                if reward_mod.REWARD_MODE == "turn_local":
                    return not reward_mod.group_turn_signal([s["turn_scores"] for s in scores])
                return bool(np.std(trajectory_objectives(scores)) < 1e-9)

            for wave in range(cfg["dapo_max_waves"]):
                degenerate = [
                    i for i, (_, _, scores) in enumerate(groups) if _degenerate(scores)
                ]
                if not degenerate:
                    break
                fresh = rollout_and_score(next_samples(len(degenerate)), gen_call=step * 10 + wave + 1)
                kept = [g for i, g in enumerate(groups) if i not in set(degenerate)]
                groups = kept + fresh
                groups = groups[: cfg["prompts_per_step"]]

            all_rewards, all_objectives, rubric_values = [], [], []
            malformed, responses, turns_total = 0, 0, 0
            zero_var = 0
            
            
            rubric_group_stds: list[float] = []
            rubric_no_grad_groups = 0
            comp: dict[str, list[float]] = {}
            for sample, lanes_g, scores in groups:
                if _degenerate(scores):
                    zero_var += 1
                _grp_rub = [float(s["rubric_c"]) for s in scores if s.get("rubric_c") is not None]
                if len(_grp_rub) >= 2:
                    rubric_group_stds.append(float(np.std(_grp_rub)))
                else:
                    rubric_no_grad_groups += 1
                rows_payload.extend(compose_group_rows(sample, lanes_g, scores, cfg, rng))
                all_objectives.extend(trajectory_objectives(scores))
                for lane, score in zip(lanes_g, scores):
                    all_rewards.append(score["R"])
                    if score.get("rubric_c") is not None:
                        rubric_values.append(float(score["rubric_c"]))
                    malformed += sum(k == "malformed" for _, k, _ in lane.turns)
                    responses += sum(k == "response" for _, k, _ in lane.turns)
                    turns_total += len(lane.turns)
                    # Reward components, already computed per trajectory by
                    # reward.score_group: logging them separates "speaks more"
                    # from "speaks better" without waiting for an eval.
                    for key, value in (score.get("metrics") or {}).items():
                        comp.setdefault(key, []).append(float(value))


            _p_all: list[float] = []
            _una_tot = _una = _una_speak = 0
            _by_src: dict[str, dict] = {}
            for _sample, _lanes_g, _scores in groups:
                _src = str(_sample.get("source") or "unknown")
                _b = _by_src.setdefault(_src, {"p": [], "resp": 0, "turns": 0, "false": 0.0, "traj": 0})
                for _lane in _lanes_g:
                    for _pv in (getattr(_lane, "margins", None) or {}).values():
                        _p_all.append(float(_pv)); _b["p"].append(float(_pv))
                _acts: dict[float, list[str]] = {}
                for _lane in _lanes_g:
                    for _t, _kind, _ in _lane.turns:
                        _acts.setdefault(float(_t), []).append(_kind)
                for _ks in _acts.values():
                    if len(_ks) < 2:
                        continue
                    _una_tot += 1
                    if len(set(_ks)) == 1:
                        _una += 1
                        if _ks[0] == "response":
                            _una_speak += 1
                for _lane, _score in zip(_lanes_g, _scores):
                    _b["resp"] += sum(k == "response" for _, k, _ in _lane.turns)
                    _b["turns"] += len(_lane.turns)
                    _b["false"] += float((_score.get("metrics") or {}).get("false_response", 0.0))
                    _b["traj"] += 1
            _mp = lambda v: float(np.mean(v)) if len(v) else 0.0
            by_source = {
                k: {
                    "mean_p": round(_mp(v["p"]), 4),
                    "response_rate": round(v["resp"] / max(1, v["turns"]), 4),
                    "false_per_traj": round(v["false"] / max(1, v["traj"]), 4),
                    "traj": v["traj"],
                }
                for k, v in sorted(_by_src.items())
            }
            step_stats = {
                "step": step,
                
                "mean_p": _mp(_p_all),
                "p_gt095": _mp([p > 0.95 for p in _p_all]),
                "p_lt005": _mp([p < 0.05 for p in _p_all]),
                "unanimous_frac": _una / max(1, _una_tot),
                "unanimous_speak_frac": _una_speak / max(1, _una_tot),
                # --- 死区: 优势恒为 0 的训练行占比 (>0.5 说明奖励在打空) ---
                # 逐 turn 的 malformed_rate 会严重低估影响: reward.py:435 是
                # any(not step.valid) -> 一条轨迹任意一个坏 turn 就整条 R=-1。
                # 轨迹 11 turn -> 46 turn 后, 同样的逐 turn 率会放大 4 倍。
                "malformed_traj_rate": (
                    float(np.mean(comp.get("malformed", [0.0]))) if comp.get("malformed") else 0.0
                ),
                "dead_row_frac": (
                    _mp([abs(r["advantage"]) < 1e-9 for r in rows_payload]) if rows_payload else 0.0
                ),
                "by_source": by_source,
                "reward_mean": float(np.mean(all_rewards)),
                "reward_std": float(np.std(all_rewards)),
                "rubric_weight": float(reward_mod.RUBRIC_W),
                "rubric_mean": float(np.mean(rubric_values)) if rubric_values else None,
                "rubric_coverage": len(rubric_values) / max(1, len(all_rewards)),
                "rubric_std": float(np.std(rubric_values)) if rubric_values else None,
                "rubric_group_std_mean": (
                    float(np.mean(rubric_group_stds)) if rubric_group_stds else None
                ),
                # 组内 rubric 提供不了梯度的组数: 开口车道 <2, 或方差为 0。
                # 验收: rubric_zero_var_groups / groups 应显著小于 1.0。
                "rubric_zero_var_groups": (
                    rubric_no_grad_groups
                    + sum(1 for s in rubric_group_stds if s < 1e-6)
                ),
                "objective_mean": float(np.mean(all_objectives)),
                "objective_std": float(np.std(all_objectives)),
                "response_rate": responses / max(1, turns_total),
                "malformed_rate": malformed / max(1, turns_total),
                "zero_variance_groups": zero_var,
                "groups": len(groups),
                "rows": len(rows_payload),
                "adv_mean": float(np.mean([r["advantage"] for r in rows_payload])) if rows_payload else 0.0,
                "adv_abs_mean": float(np.mean([abs(r["advantage"]) for r in rows_payload])) if rows_payload else 0.0,
                "rollout_s": time.time() - step_t0,
            }
            # --- comprehensive quality breakdown (per trajectory means) --- #
            traj = max(1, len(all_rewards))
            step_stats.update(
                {
                    "hit_recall": float(np.mean(comp.get("hit_training_recall", [0.0]))),
                    # use_streamvibe_reward: 逐步格式合法率 (论文 R_fmt)。malformed_traj_rate 是被
                    # "一坏全坏" 放大过的整轨率, fmt_score 才是真实的格式质量。
                    "fmt_score": float(np.mean(comp.get("fmt_score", [0.0]))),
                    "content_score": float(np.mean(comp.get("content", [0.0]))),
                    "timing_score": float(np.mean(comp.get("timing", [0.0]))),
                    "silence_score": float(np.mean(comp.get("appropriate_silence", [0.0]))),
                    "false_per_traj": float(sum(comp.get("false_response", [])) / traj),
                    "mistimed_per_traj": float(sum(comp.get("mistiming", [])) / traj),
                    "duplicate_per_traj": float(sum(comp.get("duplicate", [])) / traj),
                    "miss_per_traj": float(sum(comp.get("miss", [])) / traj),
                    # composite: hits minus wrong-place talk, per trajectory.
                    # Rises only when the model speaks BETTER, not merely more.
                    "quality_score": float(
                        np.mean(comp.get("hit_training_recall", [0.0]))
                        - (sum(comp.get("false_response", [])) + sum(comp.get("mistiming", []))
                           + sum(comp.get("duplicate", []))) / traj
                    ),
                }
            )

        payload = [rows_payload, step_stats]
        dist.broadcast_object_list(payload, src=0)
        rows_payload, step_stats = payload
        # Every rank must run the SAME number of micro-batches: with
        # micro_batch_rows=1 each row is one forward/backward, and each of those
        # carries FSDP collectives.  rows_payload[rank::world] gives the leading
        # ranks one extra row whenever the total is not a multiple of `world`,
        # so those ranks enter a collective the trailing ranks have already left
        # -- a silent deadlock (observed: ranks 0-47 rows=44, ranks 48-63
        # rows=43).  Pad with zero-advantage copies so the count is uniform;
        # they contribute no gradient but keep the collective schedule aligned.
        if rows_payload:
            rem = len(rows_payload) % world
            if rem:
                pad = []
                for i in range(world - rem):
                    filler = dict(rows_payload[i % len(rows_payload)])
                    filler["advantage"] = 0.0
                    pad.append(filler)
                rows_payload = rows_payload + pad
                if rank == 0:
                    print(f"[align] padded {len(pad)} zero-advantage rows "
                          f"to reach {len(rows_payload)} ({world} ranks)", flush=True)
        my_rows = rows_payload[rank::world]

        # ---------------- tensorize ---------------- #
        t_build = time.time()
        built = [builder.build(row) for row in my_rows]
        micro_batches = [
            built[i:i + cfg["micro_batch_rows"]]
            for i in range(0, len(built), cfg["micro_batch_rows"])
        ]

        max_len = max((len(r["input_ids"]) for r in built), default=0)
        n_img_tokens = max(
            (int(r["image_grid_thw"].prod(-1).sum()) for r in built if "image_grid_thw" in r),
            default=0,
        )
        print(
            f"[mem] rank={rank} rows={len(built)} max_len={max_len} "
            f"max_img_patches={n_img_tokens} "
            f"alloc={torch.cuda.memory_allocated() / 1e9:.1f}GB",
            flush=True,
        )

        # active token count, global
        local_tokens = sum(int(r["loss_mask"].sum()) - 1 for r in built)  # -1: shift
        token_tensor = torch.tensor([local_tokens], dtype=torch.float64, device=device)
        dist.all_reduce(token_tensor)
        global_tokens = max(1.0, float(token_tensor.item()))

        # ---------------- ref logprobs ---------------- #
        # old logprobs are NOT recomputed: with exactly one optimizer update
        # per batch (on-policy), old == pol, so ratio == exp(pol - pol.detach())
        # == 1 and the gradient reduces exactly to A * dlog(pi).  This removes
        # one full forward pass over every row per step.
        ref_lp = []
        with torch.no_grad():
            for mb in micro_batches:
                batch = _pad_stack(mb, pad_id, device)
                ref_lp.append(_token_logprobs(ref, batch).cpu())
        print(
            f"[mem] rank={rank} after old/ref alloc={torch.cuda.memory_allocated() / 1e9:.1f}GB "
            f"peak={torch.cuda.max_memory_allocated() / 1e9:.1f}GB",
            flush=True,
        )
        torch.cuda.reset_peak_memory_stats()

        # ---------------- update ---------------- #
        optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), device=device)
        kl_sum = torch.zeros((), device=device)
        clip_frac_n = 0.0
        for mb, refl in zip(micro_batches, ref_lp):
            batch = _pad_stack(mb, pad_id, device)
            pol = _token_logprobs(policy, batch)
            if rank < 8 and not globals().get("_fwd_mem_printed"):
                globals()["_fwd_mem_printed"] = True
                print(f"[mem] rank={rank} after FIRST fwd alloc={torch.cuda.memory_allocated()/1e9:.1f}GB "
                      f"peak={torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)
            mask = _shifted_mask(batch)
            old = pol.detach()
            refl = refl.to(device)
            adv = batch["advantages"].unsqueeze(-1).expand_as(pol)
            ratio = torch.exp(pol - old)
            unclipped = ratio * adv
            clipped = torch.clamp(ratio, 1.0 - cfg["clip_eps"], 1.0 + cfg["clip_eps"]) * adv
            pg = -torch.min(unclipped, clipped)
            log_delta = refl - pol
            kl = torch.exp(log_delta) - log_delta - 1.0  # k3, >= 0
            token_loss = (pg + cfg["kl_beta"] * kl) * mask
            loss = token_loss.sum() / global_tokens
            loss.backward()
            with torch.no_grad():
                loss_sum += loss.detach()
                kl_sum += (kl * mask).sum() / global_tokens
                clip_frac_n += float(((clipped < unclipped) & mask).sum())
        print(
            f"[mem] rank={rank} after update alloc={torch.cuda.memory_allocated() / 1e9:.1f}GB "
            f"peak={torch.cuda.max_memory_allocated() / 1e9:.1f}GB",
            flush=True,
        )
        dist.all_reduce(loss_sum)
        dist.all_reduce(kl_sum)
        if not torch.isfinite(loss_sum):
            raise FloatingPointError(f"non-finite loss at step {step}: {loss_sum.item()}")
        grad_norm = policy.clip_grad_norm_(1.0)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite grad norm at step {step}")
        optimizer.step()

        # ---------------- publish + checkpoint ---------------- #
        cfg_save = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # noqa: F811
        from torch.distributed.fsdp import StateDictType  # noqa: F811
        with FSDP.state_dict_type(policy, StateDictType.FULL_STATE_DICT, cfg_save):
            state = policy.state_dict()
        if rank == 0:
            state = {k: v.to(torch.bfloat16) for k, v in state.items()}
            pub_dir = os.path.join(publish_root, f"step_{step}")
            files = save_hf_dir(state, pub_dir, args.model_path)
            weight_files = [f for f in files if f.endswith(".safetensors")]
            url_base = f"http://{hostname}:{args.publish_port}/step_{step}"

            def update(engine_url):
                last = None
                for attempt in range(2):
                    try:
                        return _post_json(
                            f"{engine_url}/update_weights",
                            {"url_base": url_base, "files": weight_files, "step": step, "exp": args.exp},
                        )
                    except Exception as exc:  # noqa: BLE001 — transient resets seen in prod
                        last = exc
                        time.sleep(5.0 * (attempt + 1))
                # A single flaky engine must not kill a 50-step run: report the
                # failure, drop the engine from the pool for later steps, and
                # keep training on the survivors.
                return {"ok": False, "engine": engine_url, "error": str(last)}

            failed_engines = []
            with ThreadPoolExecutor(max_workers=len(engines)) as pool:
                for result in pool.map(update, list(engines)):
                    if not result.get("ok"):
                        failed_engines.append(result.get("engine"))
                        print(f"[warn] weight update failed, dropping engine: {result}", flush=True)
            if failed_engines:
                engines = [u for u in engines if u not in set(failed_engines)]
                rollout_cfg.engine_urls = engines
                print(f"[warn] engine pool now {len(engines)} engines", flush=True)
                if not engines:
                    raise RuntimeError("all rollout engines failed weight update")
            if step % cfg["save_freq"] == 0:


                ckpt_root = os.environ.get("CT_CKPT_DIR") or out_dir
                os.makedirs(ckpt_root, exist_ok=True)
                ckpt_dir = os.path.join(ckpt_root, f"step_{step}")
                save_hf_dir(state, ckpt_dir, args.model_path)
                ckpt_gcs = os.environ.get("CT_CKPT_GCS", "").rstrip("/")
                if ckpt_gcs:
                    import subprocess as _sp

                    _sp.Popen(
                        [
                            "bash", "-c",
                            f"gsutil -q -m rsync -r {ckpt_dir} {ckpt_gcs}/step_{step} "
                            f"&& printf done | gsutil -q cp - {ckpt_gcs}/step_{step}/.complete "
                            f"&& touch {ckpt_dir}/.gcs_done "
                            f"|| echo 'GCS UPLOAD FAILED step_{step}' >> {ckpt_root}/gcs_upload_errors.log",
                        ],
                        start_new_session=True,
                        stdout=_sp.DEVNULL,
                        stderr=_sp.DEVNULL,
                    )
                    print(f"[ckpt] step_{step} saved to {ckpt_dir}, async upload -> {ckpt_gcs}/step_{step}", flush=True)


                import shutil as _shutil

                keep = os.path.realpath(args.model_path)
                for name in os.listdir(ckpt_root):
                    cand = os.path.join(ckpt_root, name)
                    if (
                        name.startswith("step_")
                        and name != f"step_{step}"
                        and os.path.isdir(cand)
                        and os.path.realpath(cand) != keep
                        and (not ckpt_gcs or os.path.exists(os.path.join(cand, ".gcs_done")))
                    ):
                        _shutil.rmtree(cand, ignore_errors=True)
            prev = os.path.join(publish_root, f"step_{step - 1}")
            if os.path.isdir(prev):
                import shutil

                shutil.rmtree(prev, ignore_errors=True)

            step_stats.update(
                {
                    "loss": float(loss_sum.item()),
                    "kl": float(kl_sum.item()),
                    "grad_norm": float(grad_norm),
                    "train_s": time.time() - t_build,
                    "step_s": time.time() - step_t0,
                }
            )
            with open(metrics_path, "a") as fh:
                fh.write(json.dumps(step_stats) + "\n")
            print(f"[train] {json.dumps(step_stats)}", flush=True)

            if step_stats["malformed_rate"] > cfg["malformed_abort_rate"]:
                malformed_streak += 1
            else:
                malformed_streak = 0
            if malformed_streak >= cfg["malformed_abort_patience"]:
                raise RuntimeError(
                    f"malformed rate above {cfg['malformed_abort_rate']} for "
                    f"{malformed_streak} consecutive steps"
                )
        del state
        dist.barrier()

    if rank == 0:
        print("[train] done", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
