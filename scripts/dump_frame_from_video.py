#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Dump a single frame from an MP4 to a PNG.
#
# Example:
#   python -m scripts.dump_frame_from_video \
#     --video_path "/path/to/dataset/videos_tiled/observation.images.tiled/chunk-000/episode_000123.mp4"

import argparse
import os
import tempfile

import numpy as np
from decord import VideoReader, cpu
from PIL import Image


def _expand_path(p: str) -> str:
    return os.path.expandvars(os.path.expanduser(p))


def _atomic_save_png(frame: np.ndarray, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_frame_", suffix=".png", dir=os.path.dirname(out_path) or ".")
    os.close(fd)
    try:
        Image.fromarray(frame).save(tmp_path)
        os.replace(tmp_path, out_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Save a single frame from an MP4 as a PNG image.")
    p.add_argument("--video_path", type=str, required=True, help="Path to an MP4 (supports ~ and $VARS).")
    p.add_argument("--frame_index", type=int, default=4, help="0-based frame index to export (default: 4).")
    p.add_argument(
        "--out_path",
        type=str,
        default="",
        help='Output PNG path. Default: "<video_dir>/<video_stem>_frame%06d.png" % frame_index',
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    video_path = _expand_path(args.video_path)
    frame_index = int(args.frame_index)
    if frame_index < 0:
        raise ValueError(f"frame_index must be >= 0, got {frame_index}")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    n = len(vr)
    if n <= frame_index:
        raise ValueError(f"Video has only {n} frames, cannot read index={frame_index}: {video_path}")

    frame = vr[frame_index].asnumpy()
    del vr

    if frame.ndim != 3 or frame.shape[2] not in (3, 4):
        raise ValueError(f"Expected HxWx3 (or HxWx4) uint8 frame, got shape={frame.shape}")
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8, copy=False)

    if args.out_path:
        out_path = _expand_path(args.out_path)
    else:
        stem = os.path.splitext(os.path.basename(video_path))[0]
        out_path = os.path.join(os.path.dirname(video_path), f"{stem}_frame{frame_index:06d}.png")

    _atomic_save_png(frame, out_path)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

