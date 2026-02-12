# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
import pickle
import shutil

import numpy as np
from tqdm import tqdm

from imaginaire.auxiliary.text_encoder import CosmosT5TextEncoder, CosmosT5TextEncoderConfig
from imaginaire.constants import T5_MODEL_DIR

"""example command
python -m scripts.get_t5_embeddings_from_groot_dataset --dataset_path datasets/benchmark_train/gr1
"""


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute T5 embeddings for text prompts")
    parser.add_argument(
        "--dataset_path", type=str, default="datasets/benchmark_train/gr1", help="Root path to the dataset"
    )
    parser.add_argument(
        "--dataset_format",
        type=str,
        choices=["groot_csv", "lerobot"],
        default="groot_csv",
        help="Dataset format to read prompts/videos from.",
    )
    parser.add_argument(
        "--prompt_prefix", type=str, default="The robot arm is performing a task. ", help="Prefix of the prompt"
    )
    parser.add_argument("--max_length", type=int, help="Maximum length of the text embedding")
    parser.add_argument("--cache_dir", type=str, default=T5_MODEL_DIR, help="Directory to cache the T5 model")
    parser.add_argument(
        "--meta_csv",
        type=str,
        default="",
        help="Metadata csv file. Default: <dataset_path>/metadata.csv",
    )
    parser.add_argument(
        "--episodes_jsonl",
        type=str,
        default="",
        help="(LeRobot) Path to meta/episodes.jsonl. Default: <dataset_path>/meta/episodes.jsonl",
    )
    parser.add_argument(
        "--video_src_dir",
        type=str,
        default="",
        help="(LeRobot) Source dir for videos. Default: <dataset_path>/videos/chunk-000/observation.images.cam_high",
    )
    # Default behavior: enabled for LeRobot, disabled for groot_csv (handled in main()).
    parser.add_argument(
        "--copy_videos",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="(LeRobot) Copy videos into a flat <dataset_path>/videos/*.mp4 layout.",
    )
    return parser.parse_args()


def main(args) -> None:
    t5_xxl_dir = os.path.join(args.dataset_path, "t5_xxl")
    os.makedirs(t5_xxl_dir, exist_ok=True)
    meta_txt_dir = os.path.join(args.dataset_path, "metas")
    os.makedirs(meta_txt_dir, exist_ok=True)
    videos_flat_dir = os.path.join(args.dataset_path, "videos")
    os.makedirs(videos_flat_dir, exist_ok=True)

    # Auto-detect LeRobot if user didn't explicitly set it and CSV metadata isn't available.
    detected_format = args.dataset_format
    if detected_format == "groot_csv":
        meta_csv_default = args.meta_csv or os.path.join(args.dataset_path, "metadata.csv")
        episodes_jsonl_default = os.path.join(args.dataset_path, "meta", "episodes.jsonl")
        if not os.path.exists(meta_csv_default) and os.path.exists(episodes_jsonl_default):
            detected_format = "lerobot"

    if detected_format == "lerobot":
        episodes_jsonl = args.episodes_jsonl or os.path.join(args.dataset_path, "meta", "episodes.jsonl")
        video_src_dir = args.video_src_dir or os.path.join(
            args.dataset_path, "videos", "chunk-000", "observation.images.cam_high"
        )
        copy_videos = True if args.copy_videos is None else bool(args.copy_videos)

        # (video_filename_for_outputs, prompt_raw, dst_video_path_flat_or_none, src_video_path_or_none)
        meta_items: list[tuple[str, str, str | None, str | None]] = []
        with open(episodes_jsonl, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ep = json.loads(line)
                episode_index = int(ep["episode_index"])
                tasks = ep.get("tasks") or []
                if not tasks:
                    print(f"Skipping episode {episode_index:06d}: missing/empty tasks")
                    continue
                prompt_raw = str(tasks[0])
                src_video_filename = f"episode_{episode_index:06d}.mp4"
                # Output should be a simple numeric filename like "0.mp4"
                video_filename = f"{episode_index}.mp4"
                src_video_path = os.path.join(video_src_dir, src_video_filename)
                if not os.path.exists(src_video_path):
                    print(f"Skipping episode {episode_index:06d}: missing source video {src_video_path}")
                    continue
                dst_video_path = os.path.join(videos_flat_dir, video_filename) if copy_videos else None
                meta_items.append((video_filename, prompt_raw, dst_video_path, src_video_path))
    else:
        meta_csv = args.meta_csv or os.path.join(args.dataset_path, "metadata.csv")
        meta_lines = open(meta_csv).readlines()[1:]
        # (video_filename, prompt_raw, dst_video_path, src_video_path)
        meta_items = []
        for meta_line in meta_lines:
            video_filename, prompt_raw = meta_line.split(",", 1)
            prompt_raw = prompt_raw.strip("\n")
            meta_items.append((video_filename, prompt_raw, None, None))

    # Initialize T5
    encoder_config = CosmosT5TextEncoderConfig(ckpt_path=args.cache_dir)
    encoder = CosmosT5TextEncoder(config=encoder_config)

    for video_filename, prompt, dst_video_path, src_video_path in tqdm(meta_items):
        # LeRobot: optionally copy to flat videos/ layout expected by training Dataset()
        if dst_video_path is not None and not os.path.exists(dst_video_path):
            os.makedirs(os.path.dirname(dst_video_path), exist_ok=True)
            assert src_video_path is not None
            shutil.copy2(src_video_path, dst_video_path)

        if prompt.startswith('"') and prompt.endswith('"'):
            # Remove the quotes for robocasa dataset
            prompt = prompt[1:-1]
        prompt = args.prompt_prefix + prompt
        meta_txt_filename = os.path.join(meta_txt_dir, os.path.basename(video_filename).replace(".mp4", ".txt"))
        with open(meta_txt_filename, "w") as fp:
            fp.write(prompt)

        t5_xxl_filename = os.path.join(t5_xxl_dir, os.path.basename(video_filename).replace(".mp4", ".pickle"))
        if os.path.exists(t5_xxl_filename):
            print(f"Skipping {t5_xxl_filename} because it already exists")
            # Skip if the file already exists
            continue

        print(f"encoding prompt: {prompt}")

        # Compute T5 embeddings
        encoded_text, mask_bool = encoder.encode_prompts(prompt, max_length=args.max_length, return_mask=True)
        attn_mask = mask_bool.long()
        lengths = attn_mask.sum(dim=1).cpu()

        encoded_text = encoded_text.cpu().numpy().astype(np.float16)

        # trim zeros to save space
        encoded_text = [encoded_text[batch_id][: lengths[batch_id]] for batch_id in range(encoded_text.shape[0])]

        # Save T5 embeddings as pickle file
        with open(t5_xxl_filename, "wb") as fp:
            pickle.dump(encoded_text, fp)  # list of np.ndarray in (len, 1024)


if __name__ == "__main__":
    args = parse_args()
    main(args)
