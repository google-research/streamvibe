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

"""Frame materialization and image preprocessing for streaming rollouts.

``materialize_video_frame_window`` extracts 1-FPS JPEGs from the source MP4 into
node-local storage under a per-video flock (eight workers on one node must not
extract the same video concurrently).  ``process_image`` is the exact resize the
training processor sees, so rollout and training share one image geometry.
"""

from __future__ import annotations

import fcntl
import hashlib
import math
import os
import shutil
import subprocess
import tempfile

from PIL import Image

DEFAULT_CACHE_ROOT = "outputs/cache/video_frames"


def process_image(image, min_pixels: int | None, max_pixels: int | None):
    if isinstance(image, str):
        image = Image.open(image)
    image.load()  # avoid "Too many open files" errors
    if max_pixels is not None and (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))
    if min_pixels is not None and (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def frame_path(frames_dir: str, idx: int) -> str:
    return os.path.join(frames_dir, f"f_{idx}.jpg")


def materialize_video_frame_window(
    video_path: str,
    required_frames: list[int],
    cache_root: str = DEFAULT_CACHE_ROOT,
) -> str:
    """Materialize zero-based 1-FPS frames from an MP4 into node-local storage."""
    required = sorted(set(int(frame) for frame in required_frames if int(frame) >= 0))
    if not required:
        raise ValueError("required_frames must contain at least one non-negative frame")
    if not video_path or not os.path.isfile(video_path):
        raise FileNotFoundError(f"source video is unavailable: {video_path}")

    key = hashlib.sha256(os.path.realpath(video_path).encode()).hexdigest()[:24]
    local_dir = os.path.join(cache_root, key)
    complete_path = f"{local_dir}.complete-through"
    os.makedirs(cache_root, exist_ok=True)

    def complete_through() -> int:
        try:
            with open(complete_path, encoding="utf-8") as marker:
                return int(marker.read().strip())
        except (OSError, ValueError):
            return -1

    furthest = max(required)
    if complete_through() >= furthest and all(
        os.path.isfile(frame_path(local_dir, frame)) for frame in required
    ):
        return local_dir

    with open(f"{local_dir}.lock", "a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        os.makedirs(local_dir, exist_ok=True)
        if complete_through() < furthest or not all(
            os.path.isfile(frame_path(local_dir, frame)) for frame in required
        ):
            ffmpeg = os.environ.get("CT_FFMPEG", "ffmpeg")
            # Decode into a staging directory and publish each completed JPEG
            # atomically: a reader must never open a partially-written pathname.
            staging_dir = tempfile.mkdtemp(prefix=f".{key}.", dir=cache_root)
            try:
                result = subprocess.run(
                    [
                        ffmpeg, "-nostdin", "-loglevel", "error", "-y",
                        "-i", video_path,
                        "-t", str(furthest + 1),
                        "-vf", "fps=1",
                        "-q:v", "2",
                        "-start_number", "0",
                        os.path.join(staging_dir, "f_%d.jpg"),
                    ],
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    raise RuntimeError(f"ffmpeg frame materialization failed: {result.stderr[-500:]}")
                missing_staged = [
                    frame for frame in required
                    if not os.path.isfile(os.path.join(staging_dir, f"f_{frame}.jpg"))
                ]
                # The fps filter can omit the final integer-second sample for
                # videos whose duration ends shortly after that second; recover
                # any such samples with an exact seek.
                for frame in missing_staged:
                    output_path = os.path.join(staging_dir, f"f_{frame}.jpg")
                    exact_result = subprocess.run(
                        [
                            ffmpeg, "-nostdin", "-loglevel", "error", "-y",
                            "-ss", str(frame),
                            "-i", video_path,
                            "-frames:v", "1",
                            "-q:v", "2",
                            output_path,
                        ],
                        capture_output=True,
                        text=True,
                    )
                    if exact_result.returncode != 0:
                        raise RuntimeError(
                            "ffmpeg exact frame materialization failed "
                            f"at {frame}s: {exact_result.stderr[-500:]}"
                        )
                missing_staged = [
                    frame for frame in required
                    if not os.path.isfile(os.path.join(staging_dir, f"f_{frame}.jpg"))
                ]
                if missing_staged:
                    raise RuntimeError(
                        "source video did not materialize required 1-FPS frames "
                        f"{missing_staged[:8]}: {video_path}"
                    )
                for name in os.listdir(staging_dir):
                    if name.startswith("f_") and name.endswith(".jpg"):
                        os.replace(os.path.join(staging_dir, name), os.path.join(local_dir, name))
                marker_tmp = f"{complete_path}.{os.getpid()}.tmp"
                with open(marker_tmp, "w", encoding="utf-8") as marker:
                    marker.write(str(furthest))
                os.replace(marker_tmp, complete_path)
            finally:
                shutil.rmtree(staging_dir, ignore_errors=True)
        missing = [
            frame for frame in required
            if not os.path.isfile(frame_path(local_dir, frame))
        ]
        if missing:
            raise RuntimeError(
                f"source video did not materialize required 1-FPS frames {missing[:8]}: {video_path}"
            )
    return local_dir
