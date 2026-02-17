#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Convert images to 3-channel RGB in-place.
#
# Why: `Video2WorldPipeline` loads conditioning images via PIL + `to_tensor()`.
# RGBA/LA/P images become 4-channel tensors and crash the tokenizer conv3d
# (expects 3 input channels).

import argparse
import os
import tempfile

from PIL import Image


def _iter_images(root: str):
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith((".png", ".webp", ".jpg", ".jpeg")):
                yield os.path.join(dirpath, fn)


def _atomic_save_rgb(img: Image.Image, dst_path: str) -> None:
    dst_dir = os.path.dirname(dst_path) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_rgb_", suffix=os.path.splitext(dst_path)[1], dir=dst_dir)
    os.close(fd)
    try:
        # Save as RGB with the same extension. For PNG/WEBP this removes alpha.
        img.save(tmp_path)
        os.replace(tmp_path, dst_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def main() -> int:
    p = argparse.ArgumentParser(description="Convert RGBA/LA/P images to RGB in-place.")
    p.add_argument("--root", type=str, required=True, help="Root directory to scan recursively.")
    p.add_argument("--dryrun", action="store_true", help="Only print what would be converted.")
    args = p.parse_args()

    root = os.path.expanduser(args.root)
    converted = 0
    scanned = 0
    for path in _iter_images(root):
        scanned += 1
        try:
            with Image.open(path) as im:
                if im.mode in {"RGBA", "LA", "P"}:
                    if args.dryrun:
                        print(f"would_convert {im.mode} {path}")
                        continue
                    rgb = im.convert("RGB")
                    _atomic_save_rgb(rgb, path)
                    converted += 1
        except Exception as e:
            print(f"skip_error {path}: {type(e).__name__}: {e}")

    print(f"scanned={scanned} converted={converted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

