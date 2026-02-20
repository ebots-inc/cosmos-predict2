#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Tile multi-view LeRobot MP4s into a single 2x2 MP4 without resizing.
#
# Layout:
#   [cam_high        | cam_left_wrist ]
#   [cam_right_wrist | black          ]

import argparse
import os
import tempfile
from typing import Iterator

import imageio
import numpy as np
from decord import VideoReader, cpu


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tile LeRobot multi-view MP4s into a single 2x2 MP4 (no resizing)."
    )
    p.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="LeRobot dataset root containing videos/.",
    )
    p.add_argument(
        "--video_key_high",
        type=str,
        default="observation.images.cam_high",
        help="Folder name under videos/ for the high cam.",
    )
    p.add_argument(
        "--video_key_left",
        type=str,
        default="observation.images.cam_left_wrist",
        help="Folder name under videos/ for the left wrist cam.",
    )
    p.add_argument(
        "--video_key_right",
        type=str,
        default="observation.images.cam_right_wrist",
        help="Folder name under videos/ for the right wrist cam.",
    )
    p.add_argument(
        "--out_video_key",
        type=str,
        default="observation.images.tiled",
        help="Folder name under videos/chunk-*/ to write outputs.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs.",
    )
    return p.parse_args()


def _iter_episode_mp4s(videos_root: str, video_key: str) -> Iterator[str]:
    # Expected LeRobot layout:
    #   videos/chunk-XXX/<video_key>/episode_XXXXXX.mp4
    for dirpath, _, filenames in os.walk(videos_root):
        base = os.path.basename(dirpath)
        if base != video_key:
            continue
        chunk_dir = os.path.basename(os.path.dirname(dirpath))
        if not chunk_dir.startswith("chunk-"):
            continue
        for fn in filenames:
            if fn.lower().endswith(".mp4") and fn.startswith("episode_"):
                yield os.path.join(dirpath, fn)


def _replace_video_key(rel_path: str, old: str, new: str) -> str:
    parts = rel_path.split(os.sep)
    for i, p in enumerate(parts):
        if p == old:
            parts[i] = new
            return os.sep.join(parts)
    return rel_path


def _remove_video_key(rel_path: str, key: str) -> str:
    parts = rel_path.split(os.sep)
    for i, p in enumerate(parts):
        if p == key:
            del parts[i]
            break
    return os.sep.join(parts)


def _atomic_open_writer(dst_path: str, fps: float):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=".tmp_tiled_", suffix=".mp4", dir=os.path.dirname(dst_path)
    )
    os.close(fd)
    writer = imageio.get_writer(tmp_path, fps=fps)
    return writer, tmp_path


def _tile_one(
    high_path: str,
    left_path: str,
    right_path: str,
    out_path: str,
    overwrite: bool,
) -> None:
    if (not overwrite) and os.path.exists(out_path):
        return

    vr_h = VideoReader(high_path, ctx=cpu(0), num_threads=1)
    vr_l = VideoReader(left_path, ctx=cpu(0), num_threads=1)
    vr_r = VideoReader(right_path, ctx=cpu(0), num_threads=1)

    n = min(len(vr_h), len(vr_l), len(vr_r))
    if n <= 0:
        raise ValueError(f"Empty video among: {high_path}, {left_path}, {right_path}")

    fps = float(vr_h.get_avg_fps() or 0.0)
    if fps <= 0:
        fps = 10.0

    f0h = vr_h[0].asnumpy()
    f0l = vr_l[0].asnumpy()
    f0r = vr_r[0].asnumpy()
    if f0h.ndim != 3 or f0h.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 frames, got {f0h.shape} from {high_path}")

    h, w = f0h.shape[:2]
    if f0l.shape[:2] != (h, w) or f0r.shape[:2] != (h, w):
        raise ValueError(
            "Views have different resolutions; refusing to resize. "
            f"high={f0h.shape}, left={f0l.shape}, right={f0r.shape}"
        )

    blank = np.zeros((h, w, 3), dtype=np.uint8)

    writer, tmp_path = _atomic_open_writer(out_path, fps=fps)
    try:
        for i in range(n):
            fh = vr_h[i].asnumpy()
            fl = vr_l[i].asnumpy()
            fr = vr_r[i].asnumpy()

            top = np.concatenate([fh, fl], axis=1)  # (h, 2w, 3)
            bot = np.concatenate([fr, blank], axis=1)  # (h, 2w, 3)
            tiled = np.concatenate([top, bot], axis=0)  # (2h, 2w, 3)
            writer.append_data(tiled)

        writer.close()
        os.replace(tmp_path, out_path)
    except Exception:
        try:
            writer.close()
        except Exception:
            pass
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        raise


def main() -> int:
    args = _parse_args()
    dataset_path = os.path.expanduser(args.dataset_path)
    videos_root = os.path.join(dataset_path, "videos")
    out_root = videos_root

    if not os.path.isdir(videos_root):
        print(f"videos/ not found: {videos_root}")
        return 2

    high_paths = sorted(_iter_episode_mp4s(videos_root, args.video_key_high))
    if not high_paths:
        print(f"No episodes found under: {os.path.join(videos_root, 'chunk-*', args.video_key_high)}")
        return 0

    for hp in high_paths:
        rel_full = os.path.relpath(hp, videos_root)
        lp = os.path.join(videos_root, _replace_video_key(rel_full, args.video_key_high, args.video_key_left))
        rp = os.path.join(videos_root, _replace_video_key(rel_full, args.video_key_high, args.video_key_right))

        if not os.path.exists(lp) or not os.path.exists(rp):
            print("Missing view(s), skipping:")
            print(f"  high:  {hp}")
            print(f"  left:  {lp}")
            print(f"  right: {rp}")
            continue

        rel_out = _replace_video_key(rel_full, args.video_key_high, args.out_video_key)
        out_path = os.path.join(out_root, rel_out)
        _tile_one(hp, lp, rp, out_path, overwrite=args.overwrite)
        print(f"Wrote tiled: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

