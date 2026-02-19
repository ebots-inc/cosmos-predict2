#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Transcode only Decord-unreadable MP4s in-place to H.264.
#
# Why: training/IDM pipelines decode videos with Decord; some MP4s (codec/container)
# can fail to load. This script probes each MP4 with Decord and fixes only failures.

import argparse
import os
import shutil
import subprocess
import tempfile

from decord import VideoReader, cpu


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fix decord-unreadable MP4 files by transcoding in-place to H.264.")
    p.add_argument("--dataset_path", type=str, required=True, help="Dataset root (videos root: <dataset_path>/videos).")
    return p.parse_args()


def _iter_mp4s(root: str):
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".mp4"):
                yield os.path.join(dirpath, fn)


def _video_needs_transcode(video_path: str) -> bool:
    try:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        if len(vr) <= 0:
            return True
        _ = vr[0].asnumpy()
        del vr
        return False
    except Exception:
        return True


def _transcode_inplace(src: str, ffmpeg_exe: str) -> bool:
    dst_dir = os.path.dirname(src) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_h264_", suffix=".mp4", dir=dst_dir)
    os.close(fd)
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i",
        src,
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        tmp_path,
    ]
    try:
        subprocess.run(cmd, check=True)
        os.replace(tmp_path, src)
        return True
    except Exception as e:
        print(f"ffmpeg failed for {src}: {e}")
        return False
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def main() -> int:
    args = parse_args()
    dataset_path = os.path.expanduser(args.dataset_path)
    videos_root = os.path.join(dataset_path, "videos")

    if not os.path.isdir(videos_root):
        print(f"videos_root not found: {videos_root}")
        return 2

    mp4_paths = sorted(_iter_mp4s(videos_root))
    if not mp4_paths:
        print(f"No .mp4 files found under: {videos_root}")
        return 0

    bad = [p for p in mp4_paths if _video_needs_transcode(p)]
    if not bad:
        print(f"All videos decord-readable. scanned={len(mp4_paths)} replaced=0")
        return 0

    ffmpeg_exe = shutil.which("ffmpeg")
    if not ffmpeg_exe:
        print("ffmpeg not found; cannot transcode failing videos (install with: sudo apt install -y ffmpeg)")
        print(f"scanned={len(mp4_paths)} needs_transcode={len(bad)}")
        return 2

    replaced = 0
    for src in bad:
        ok = _transcode_inplace(src, ffmpeg_exe)
        if ok:
            replaced += 1
            print(f"Replaced with training-compatible H.264: {src}")

    print(f"scanned={len(mp4_paths)} needs_transcode={len(bad)} replaced={replaced}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

