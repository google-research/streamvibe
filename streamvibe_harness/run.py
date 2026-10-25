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

"""Run StreamVibe Harness with a local interaction model and memory server."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-samples',type=int,default=1)
    p.add_argument('--max-turns',type=int,default=16)
    p.add_argument('--memory-url',default=os.getenv('HARNESS_MEMORY_URL','http://127.0.0.1:8765/v1'))
    p.add_argument('--memory-model',default=os.getenv('HARNESS_MEMORY_MODEL','memory-agent'))
    p.add_argument('--dry-run',action='store_true',help='Print launch settings without loading models or calling services')
    a=p.parse_args()
    checkpoint=a.checkpoint.resolve()
    source=a.input.resolve()
    output=a.output.resolve()
    if not source.is_file():p.error('Input JSONL does not exist')
    if not a.dry_run and not checkpoint.is_dir():p.error('Checkpoint directory does not exist')
    if a.max_samples<0 or a.max_turns<0:p.error('Limits must be non-negative; zero means unrestricted')
    command=[sys.executable,'streamvibe_harness/run_streamvibe_harness.py',
             '--model','StreamVibe-VL','--model-path',os.path.relpath(checkpoint, ROOT),
             '--task','StreamVibe-Bench','--input',os.path.relpath(source, ROOT),'--output',os.path.relpath(output, ROOT),
             '--trace-output',os.path.relpath(output.with_suffix('.trace.jsonl'), ROOT),
             '--arm','harness_32','--visual-window','32',
             '--harness-policy','streamvibe_harness/policy.json',
             '--system-prompt-profile','training_native','--history-silence','keep',
             '--predecision-only','--memory-agent','vllm',
             '--memory-url',a.memory_url,'--memory-model',a.memory_model,
             '--lanes','1','--max-samples',str(a.max_samples),'--max-turns',str(a.max_turns)]
    env=os.environ.copy()
    env['STREAMVIBE_INFERENCE_BACKEND']='hf'
    env['STREAMVIBE_HARNESS_GENOME']='streamvibe_harness/genome.json'
    if a.dry_run:
        print(json.dumps({'command':command,'backend':'hf',
                          'genome':'streamvibe_harness/genome.json','policy':'streamvibe_harness/policy.json',
                          'network_or_model_calls':False},indent=2))
        return 0
    output.parent.mkdir(parents=True,exist_ok=True)
    return subprocess.call(command,cwd=ROOT,env=env)


if __name__=='__main__':
    sys.exit(main())
