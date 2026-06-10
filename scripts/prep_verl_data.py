"""Convert SwiLatent annotated JSONL to verl-compatible Parquet format.

Usage:
    python scripts/prep_verl_data.py \
        --input_file data_hf/data/math_openr1_all.jsonl \
        --output_dir data_verl \
        --val_ratio 0.05
"""

import argparse
import json
import os
import random

import datasets


SYSTEM_PROMPT = "You are a helpful assistant."
INSTRUCTION_SUFFIX = ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="data_verl")
    parser.add_argument("--val_ratio", type=float, default=0.05,
                        help="Fraction of data for validation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.input_file) as f:
        raw_data = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(raw_data)} samples from {args.input_file}")

    processed = []
    for idx, item in enumerate(raw_data):
        question = item.get("instruction", "")
        inp = item.get("input", "")
        if inp:
            question = f"{question}\n{inp}"

        ground_truth = item.get("ground_truth_answer", "")
        if not ground_truth:
            continue

        row = {
            "data_source": "swilatent_math",
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": str(ground_truth),
            },
            "extra_info": {
                "index": idx,
                "dataset_index": item.get("dataset_index", idx),
            },
        }
        processed.append(row)

    print(f"Processed {len(processed)} valid samples")

    random.shuffle(processed)
    n_val = max(1, int(len(processed) * args.val_ratio))
    val_data = processed[:n_val]
    train_data = processed[n_val:]

    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    os.makedirs(args.output_dir, exist_ok=True)

    train_ds = datasets.Dataset.from_list(train_data)
    val_ds = datasets.Dataset.from_list(val_data)

    train_path = os.path.join(args.output_dir, "train.parquet")
    val_path = os.path.join(args.output_dir, "test.parquet")

    train_ds.to_parquet(train_path)
    val_ds.to_parquet(val_path)

    print(f"Saved: {train_path} ({len(train_data)} rows)")
    print(f"Saved: {val_path} ({len(val_data)} rows)")


if __name__ == "__main__":
    main()
