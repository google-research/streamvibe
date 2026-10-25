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

"""Write every usable StreamVibe-SFT video annotation as one complete-video sample."""
from __future__ import annotations
import collections, hashlib, json, os
from pathlib import Path

ROOT=Path(os.environ['STREAMVIBE_SOURCE_ROOT'])
POOL_ROOT=Path(os.environ['STREAMVIBE_POOL_ROOT'])
OUT=POOL_ROOT/'StreamVibe-SFT/manifests/StreamVibe-SFT-video.jsonl'
REPORT=POOL_ROOT/'StreamVibe-SFT/manifests/StreamVibe-SFT-video-report.json'


DESIRED_TARGETS={'chat':300000,'event_grounding':243682,'narration':267286}
TARGETS={'chat':300000,'event_grounding':220826,'narration':247990}
EVENT={'event_action_understanding','temporal_grounding','state_transition_procedure_next',
       'counting_time_repeated_trigger','anomaly_event_alert'}
def rows(path):
 with Path(path).open() as f:
  for line in f:
   if line.strip(): yield json.loads(line)
def cls(r):
 t=str(r.get('primary_task') or '')
 return 'narration' if t=='narration_commentary' else ('event_grounding' if t in EVENT else 'chat')
def identity(r,uri):
 return f"official:{r.get('source_canonical') or r.get('source_raw') or ''}:{r.get('video_name') or Path(uri).name}"
def sig(r,uri):
 p=json.dumps([r.get('source_canonical') or r.get('source_raw') or '',r.get('video_name') or Path(uri).name,
               r.get('question') or [],r.get('responses') or []],ensure_ascii=False,sort_keys=True,separators=(',',':'))
 return hashlib.sha256(p.encode()).hexdigest()
def sources():
 for r in rows(ROOT/'manifests/video_train.jsonl'):
  yield cls(r),r,str(r.get('resolved_uri') or ''),'legacy'
 base=POOL_ROOT/'StreamVibe-SFT/candidates'
 for c in ('chat','event_grounding','narration'):
  for r in rows(base/f'{c}.jsonl'):
   # ``resolved_uri`` in these legacy candidates points at a temporary
   # content-addressed staging copy.  Those replicas were intentionally
   # removed, while ``source_local_uri`` is the authoritative surviving
   # source (which may itself be on POOL_ROOT).  Do not resurrect stale replica
   # paths in the StreamVibe-SFT manifest.
   resolved=str(r.get('resolved_uri') or '')
   source=str(r.get('source_local_uri') or '')
   stale_prefix=str(base.parent/'media')+'/'
   uri=(source or resolved) if resolved.startswith(stale_prefix) else (resolved or source)
   yield c,r,uri,'new'
 expanded=POOL_ROOT/'StreamVibe-SFT/additional_candidates'
 for c in ('chat','event_grounding','narration'):
  for r in rows(expanded/f'{c}.jsonl'):
   # Existing home media may remain in place; every newly downloaded source
   # already has a POOL_ROOT source_local_uri.
   yield c,r,str(r.get('source_local_uri') or r.get('resolved_uri') or ''),'expanded'
def main():
 OUT.parent.mkdir(parents=True,exist_ok=True)
 rl_rows=list(rows(POOL_ROOT/'StreamVibe-RL/source_manifests/StreamVibe-RL-source.jsonl'))
 # Video reuse across SFT/RL/dev is allowed; only the actual annotated sample
 # must be disjoint.  This preserves valid independent annotations attached to
 # the same complete video.
 rl_annotation_ids={str(r.get('annotation_id') or '') for r in rl_rows if r.get('annotation_id')}
 dev_paths=[ROOT/'manifests/dev.jsonl',
            POOL_ROOT/'StreamVibe-RL/source_manifests/criterion_timing_dev.jsonl',
            POOL_ROOT/'StreamVibe-RL/source_manifests/streaming_dev.jsonl']
 dev_annotation_ids={str(r.get('annotation_id') or '') for p in dev_paths for r in rows(p) if r.get('annotation_id')}
 seen_s,seen_a=set(),set(); counts=collections.Counter(); media=set(); ids=set()
 tmp=OUT.with_suffix('.jsonl.partial')
 with tmp.open('w') as out:
  for category,r,uri,origin in sources():
   if counts[category]>=TARGETS[category]: continue
   ann=str(r.get('annotation_id') or ''); signature=sig(r,uri)
   if ann in rl_annotation_ids:
    counts['excluded_rl']+=1; continue
   if ann in dev_annotation_ids: counts['excluded_dev']+=1; continue
   if signature in seen_s or (ann and ann in seen_a): counts['excluded_duplicate']+=1; continue
   seen_s.add(signature); seen_a.add(ann)
   final=dict(r); duration=float(final['duration_sec'])
   final['resolved_uri']=uri
   final['timing_class']=category; final['source_task_type']=category; final['row_kind']='streaming'
   final['window_start_sec']=0.0; final['window_end_sec']=round(duration,3)
   # Use the trainer's existing positive enum; the explicit 0..duration bounds
   # are what make this a complete-video sample (not the enum name).
   final['window_type']='response_centered_positive'; final['window_id']=hashlib.sha256(bytes.fromhex("76382d77686f6c6500") + str(ann).encode()).hexdigest()
   final.pop('source_local_uri',None)
   out.write(json.dumps(final,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n')
   counts[category]+=1; counts[f'{category}_{origin}']+=1; media.add(final['resolved_uri']); ids.add(final['window_id'])
  out.flush(); os.fsync(out.fileno())
 os.replace(tmp,OUT)
 total=sum(counts[c] for c in ('chat','event_grounding','narration'))
 shortfall={c:max(0,TARGETS[c]-counts[c]) for c in TARGETS}
 gap_to_original={c:max(0,DESIRED_TARGETS[c]-counts[c]) for c in DESIRED_TARGETS}
 report={'output':str(OUT),'rows':total,'targets':TARGETS,'desired_targets':DESIRED_TARGETS,
         'shortfall':shortfall,'gap_to_original':gap_to_original,'accepted_available_maximum':True,
         'class_counts':{c:counts[c] for c in ('chat','event_grounding','narration')},
         'unique_rows':len(ids),'unique_media':len(media),'window_policy':'one complete video per annotation; no sliding/cropped windows',
         'audit':dict(counts)}
 REPORT.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n'); print(json.dumps(report,sort_keys=True),flush=True)
 # The user accepted the largest valid available set after external media
 # recovery was exhausted.  A nonzero gap is therefore reported, not hidden
 # or padded with duplicate/windows.
if __name__=='__main__': main()
