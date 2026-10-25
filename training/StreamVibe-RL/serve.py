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

"""Hugging Face inference server for StreamVibe-RL.

HTTP endpoints used by rollout generation and training weight publication:
  POST /v1/chat/completions: chat with image inputs, token IDs, and optional logprobs.
  POST /update_weights: download and load published training weights.
  GET /health: server health and current weight step.

Requests run sequentially per process; use one process per GPU for parallelism.
"""
from __future__ import annotations

import argparse
import base64
import fcntl
import io
import json
import os
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

WEIGHTS_ROOT = os.environ.get("CT_LITE_WEIGHTS_ROOT", "outputs/cache/policy_weights")


def _decode_messages(messages: list[dict]):
    """OpenAI-style messages with data-URL images -> (hf_messages, PIL images)."""
    from PIL import Image

    out, images = [], []
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            out.append({"role": m["role"], "content": content})
            continue
        parts = []
        for item in content:
            if item.get("type") == "text":
                parts.append({"type": "text", "text": item["text"]})
            elif item.get("type") == "image_url":
                url = item["image_url"]["url"]
                b64 = url.split(",", 1)[1]
                images.append(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"))
                parts.append({"type": "image"})
        out.append({"role": m["role"], "content": parts})
    return out, images


class _Req:
    """一个在队列里等待被合批的请求。"""
    __slots__ = ("payload", "done", "result", "error")

    def __init__(self, payload: dict):
        self.payload = payload
        self.done = threading.Event()
        self.result: "dict | None" = None
        self.error: "BaseException | None" = None


class Engine:
    def __init__(self, model_path: str):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map="cuda:0", local_files_only=True
        )
        self.model.eval()
        self.lock = threading.Lock()   # one generate at a time per GPU
        self.weight_step = -1

        # ---- micro-batching ----
        # 原实现每次 generate() 只喂 1 条, GPU 当单请求推理机用: rollout 段
        # 占了每步 67% 的时间而训练卡全程 0% 利用率。这里补上 vLLM 版
        # serve_engine.py 早就有的 --max-batch/--batch-wait-ms 机制: 收集
        # BATCH_WAIT_MS 窗口内到达的请求, 合并成一次 generate()。
        # 批处理必须左 padding, 否则短序列会从 pad 位开始生成。
        tok = self.processor.tokenizer
        tok.padding_side = "left"
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        self._q: "list[_Req]" = []
        self._q_lock = threading.Lock()
        self._q_evt = threading.Condition(self._q_lock)
        self._batch_max = int(os.environ.get("CT_ENGINE_MAX_BATCH", "8"))
        self._batch_wait = float(os.environ.get("CT_ENGINE_BATCH_WAIT_MS", "40")) / 1000.0
        threading.Thread(target=self._batch_loop, daemon=True).start()

    # ------------------------------------------------------------------ #
    def submit_chat(self, payload: dict) -> dict:
        """入队等待合批。单条请求的语义与旧实现完全一致, 只是可能和别的
        请求共享一次 generate()。"""
        req = _Req(payload)
        with self._q_evt:
            self._q.append(req)
            self._q_evt.notify()
        req.done.wait()
        if req.error is not None:
            raise req.error
        return req.result

    # ------------------------------------------------------------------ #
    def _batch_loop(self) -> None:
        while True:
            with self._q_evt:
                while not self._q:
                    self._q_evt.wait()
                # 攒一个窗口: 让同一轮里紧随其后的请求有机会汇合
                self._q_evt.wait(self._batch_wait)
                batch = self._q[: self._batch_max]
                del self._q[: len(batch)]
            if not batch:
                continue
            try:
                results = self._run_batch([r.payload for r in batch])
                for r, res in zip(batch, results):
                    r.result = res
            except BaseException as exc:  # noqa: BLE001
                for r in batch:
                    r.error = exc
            finally:
                for r in batch:
                    r.done.set()

    # ------------------------------------------------------------------ #
    def _run_batch(self, payloads: "list[dict]") -> "list[dict]":
        torch = self.torch
        n = len(payloads)
        texts, images_all = [], []
        for p in payloads:
            msgs, imgs = _decode_messages(p["messages"])
            texts.append(self.processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False))
            images_all.append(imgs)

        kwargs = {"text": texts, "return_tensors": "pt", "padding": True}
        if any(images_all):
            # 变长多图: processor 接受 list[list[Image]], 逐条对齐 grid
            kwargs["images"] = images_all if n > 1 else images_all[0]
        inputs = self.processor(**kwargs).to("cuda:0")

        # 采样参数取批内第一条 —— rollout 对整批用同一组 temperature/top_p,
        # 不同的只有 seed。若出现异质请求, 退化为逐条处理(见下方 fallback)。
        temp = float(payloads[0].get("temperature", 1.0))
        top_p = float(payloads[0].get("top_p", 1.0))
        max_new = int(payloads[0].get("max_tokens", 48))
        want_lp = any(bool(p.get("logprobs")) for p in payloads)

        # seed: generate() 不接受 per-sequence generator (transformers 4.57),
        # 所以整批设一个种子。组内多样性仍然成立 —— 批内每条序列从同一个 RNG
        # 流里取到不同的 multinomial 抽样。失去的是"同 seed 必得同输出"的
        # 逐请求可复现性; 该性质对 GRPO 不是必需的(训练本就有引擎分配、线程
        # 调度、judge 等多处非确定性), 而组内多样性是必需的, 它被保留了。
        seeds = [p.get("seed") for p in payloads if p.get("seed") is not None]
        with self.lock, torch.no_grad():
            if seeds:
                torch.manual_seed(int(min(seeds)))
            gen = self.model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=temp > 0,
                temperature=max(temp, 1e-5),
                top_p=top_p,
                output_scores=want_lp,
                # transformers 的 GenerationConfig 默认 top_k=50, do_sample 时会挂
                # TopKLogitsWarper, 把 top-50 之外全部屏蔽成 -inf。scores 是处理后
                # 的结果, 诊断 p(response) 必须用未经处理的 logits。
                output_logits=want_lp,
                return_dict_in_generate=True,
                pad_token_id=self.processor.tokenizer.pad_token_id
                or self.processor.tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]   # 左 padding 后各条等长
        _raw = getattr(gen, "logits", None) or getattr(gen, "scores", None)
        eos_id = self.processor.tokenizer.eos_token_id
        pad_id = self.processor.tokenizer.pad_token_id
        out: list[dict] = []
        for i, p in enumerate(payloads):
            seq = gen.sequences[i][prompt_len:]
            # 批内其它序列更长时, 本条会被 pad 补齐, 截掉尾部 pad
            toks = [int(t) for t in seq.tolist()]
            while toks and pad_id is not None and toks[-1] == pad_id and toks[-1] != eos_id:
                toks.pop()
            text_out = self.processor.tokenizer.decode(toks, skip_special_tokens=False)
            result = {
                "choices": [{"message": {"role": "assistant", "content": text_out}}],
                "token_ids": toks,
            }
            if p.get("logprobs") and _raw:
                # _raw[0] 是第一个生成步的 logits, 形状 [batch, vocab]; 取第 i 行
                lp = torch.log_softmax(_raw[0][i].float(), dim=-1)
                top = torch.topk(lp, 5)
                ftl = {int(j): float(v) for j, v in zip(top.indices.tolist(), top.values.tolist())}
                # 调用方点名要的 token 一律精确给出。只给 top-5 时, 决策 token 一旦
                # 掉出前五就两个都缺失, 归一化恰好得 0.5 —— 一个看起来像数据的退化值。
                for tid in (p.get("lp_token_ids") or []):
                    tid = int(tid)
                    if 0 <= tid < lp.shape[-1]:
                        ftl[tid] = float(lp[tid])
                result["first_token_logprobs"] = ftl
            out.append(result)
        return out

    # ------------------------------------------------------------------ #
    def update_weights(self, spec: dict) -> dict:
        step = int(spec["step"])
        url_base = spec["url_base"].rstrip("/")
        files = list(spec["files"])
        root = os.path.join(WEIGHTS_ROOT, str(spec.get("exp", "default")))
        local_dir = os.path.join(root, f"step_{step}")
        os.makedirs(root, exist_ok=True)
        marker = os.path.join(local_dir, ".complete")
        with open(os.path.join(root, f"step_{step}.lock"), "a") as lockf:
            fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
            if not os.path.isfile(marker):
                os.makedirs(local_dir, exist_ok=True)
                for name in files:
                    dest = os.path.join(local_dir, name)
                    tmp = dest + ".part"
                    with urllib.request.urlopen(f"{url_base}/{name}", timeout=600) as r, open(tmp, "wb") as w:
                        while True:
                            chunk = r.read(1 << 22)
                            if not chunk:
                                break
                            w.write(chunk)
                    os.replace(tmp, dest)
                with open(marker, "w") as m:
                    m.write("ok")
        from safetensors.torch import load_file

        with self.lock:
            state = {}
            for name in files:
                if name.endswith(".safetensors"):
                    state.update(load_file(os.path.join(local_dir, name)))
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            del state
        self.weight_step = step
        # retention: keep only the newest step dir (17GB each fills the disk fast)
        for exp_name in os.listdir(WEIGHTS_ROOT):
            exp_dir = os.path.join(WEIGHTS_ROOT, exp_name)
            if not os.path.isdir(exp_dir):
                continue
            for d in os.listdir(exp_dir):
                if d.startswith("step_") and d != f"step_{step}":
                    import shutil
                    shutil.rmtree(os.path.join(exp_dir, d), ignore_errors=True)
        return {"ok": True, "step": step, "missing": len(missing), "unexpected": len(unexpected)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--port", type=int, required=True)
    # accepted for launch-script compatibility; unused by the HF backend
    ap.add_argument("--max-model-len", type=int, default=12288)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--limit-images", type=int, default=64)
    args = ap.parse_args()
    engine = Engine(args.model_path)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"ok": True, "weight_step": engine.weight_step})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            try:
                payload = json.loads(self.rfile.read(length).decode())
            except json.JSONDecodeError:
                self._send(400, {"error": "bad json"})
                return
            try:
                if self.path == "/v1/chat/completions":
                    self._send(200, engine.submit_chat(payload))
                elif self.path == "/update_weights":
                    self._send(200, engine.update_weights(payload))
                else:
                    self._send(404, {"error": "not found"})
            except Exception as exc:  # noqa: BLE001
                self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    class _Server(ThreadingHTTPServer):
        # 默认 request_queue_size=5 (socket backlog): 并发 >20 时新连接直接被
        # 内核拒绝 -> ConnectionResetError。rollout 一轮会朝单个引擎打进
        # (lanes/engines) 个并发请求, PPS 开大后必然撞这堵墙。
        request_queue_size = 512
        daemon_threads = True

    server = _Server(("0.0.0.0", args.port), Handler)
    print(f"[serve_hf_engine] ready on :{args.port} model={args.model_path}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
