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
import os
import pickle
import json

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
    return parser.parse_args()


def main(args) -> None:
    t5_xxl_dir = os.path.join(args.dataset_path, "t5_xxl")
    os.makedirs(t5_xxl_dir, exist_ok=True)
    meta_txt_dir = os.path.join(args.dataset_path, "metas")
    os.makedirs(meta_txt_dir, exist_ok=True)

    meta_csv_default = os.path.join(args.dataset_path, "metadata.csv")
    episodes_jsonl = os.path.join(args.dataset_path, "meta", "episodes.jsonl")
    lerobot_detected = (not os.path.exists(meta_csv_default)) and os.path.exists(episodes_jsonl)

    # Collect (video_filename_for_outputs, prompt)
    meta_items: list[tuple[str, str]] = []

    if lerobot_detected:
        info_json = os.path.join(args.dataset_path, "meta", "info.json")
        chunk_size = None
        if os.path.exists(info_json):
            try:
                with open(info_json, "r") as f:
                    info = json.load(f)
                chunk_size = info.get("chunks_size", None)
                if chunk_size is not None:
                    chunk_size = int(chunk_size)
                    if chunk_size <= 0:
                        chunk_size = None
            except Exception:
                chunk_size = None

        tiled_root = os.path.join(
            args.dataset_path, "videos_tiled", "observation.images.tiled"
        )

        with open(episodes_jsonl, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ep = json.loads(line)
                episode_index = int(ep["episode_index"])
                tasks = ep.get("tasks") or []
                if not tasks:
                    continue
                prompt = str(tasks[0])

                chunk_index = (episode_index // chunk_size) if chunk_size is not None else 0
                tiled_mp4 = os.path.join(
                    tiled_root,
                    f"chunk-{chunk_index:03d}",
                    f"episode_{episode_index:06d}.mp4",
                )
                if not os.path.exists(tiled_mp4):
                    continue

                # Only used for naming outputs (metas/*.txt and t5_xxl/*.pickle).
                video_filename = f"{episode_index}.mp4"
                meta_items.append((video_filename, prompt))
    else:
        meta_csv = args.meta_csv or meta_csv_default
        meta_lines = open(meta_csv).readlines()[1:]
        for meta_line in meta_lines:
            video_filename, prompt = meta_line.split(",", 1)
            prompt = prompt.strip("\n")
            meta_items.append((video_filename, prompt))

    # Initialize T5
    encoder_config = CosmosT5TextEncoderConfig(ckpt_path=args.cache_dir)
    encoder = CosmosT5TextEncoder(config=encoder_config)

    for video_filename, prompt in tqdm(meta_items):
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
