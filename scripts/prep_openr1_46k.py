"""Download and prepare Openr1-Math-46k-8192 for SwiLatent GRPO training.

The dataset is already in verl-compatible format. This script converts it
to the JSONL format expected by train_phase3_grpo.py's MathQuestionDataset.

Usage:
    python scripts/prep_openr1_46k.py --output_dir data_openr1_46k

    # If HuggingFace is slow, download manually first:
    #   git lfs install
    #   git clone https://huggingface.co/datasets/Elliott/Openr1-Math-46k-8192
    # Then:
    python scripts/prep_openr1_46k.py \
        --local_path Openr1-Math-46k-8192 \
        --output_dir data_openr1_46k
"""

import argparse
import json
import os
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_path", type=str, default=None,
                        help="Local path to cloned dataset (skip download)")
    parser.add_argument("--output_dir", type=str, default="data_openr1_46k")
    parser.add_argument("--val_ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from datasets import load_dataset

    print("Loading dataset...")
    if args.local_path:
        ds = load_dataset(args.local_path, split="train")
    else:
        ds = load_dataset("Elliott/Openr1-Math-46k-8192", split="train")

    print(f"Loaded {len(ds)} samples")
    print(f"Columns: {ds.column_names}")
    print(f"data_source values: {set(ds[:100]['data_source'])}")

    converted = []
    skipped = 0
    for row in ds:
        prompt_messages = row.get("prompt", [])
        reward_model = row.get("reward_model", {})
        ground_truth = reward_model.get("ground_truth", "")
        data_source = row.get("data_source", "unknown")

        if not ground_truth or not prompt_messages:
            skipped += 1
            continue

        # Extract question from prompt messages
        question = ""
        for msg in prompt_messages:
            if msg.get("role") == "user":
                question = msg.get("content", "")
                break

        if not question:
            skipped += 1
            continue

        converted.append({
            "instruction": question,
            "input": "",
            "ground_truth_answer": str(ground_truth),
            "data_source": data_source,
        })

    print(f"Converted: {len(converted)}, Skipped: {skipped}")

    random.seed(args.seed)
    random.shuffle(converted)

    n_val = max(1, int(len(converted) * args.val_ratio))
    val_data = converted[:n_val]
    train_data = converted[n_val:]

    os.makedirs(args.output_dir, exist_ok=True)

    train_path = os.path.join(args.output_dir, "train.jsonl")
    val_path = os.path.join(args.output_dir, "val.jsonl")

    with open(train_path, "w") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(val_path, "w") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Train: {len(train_data)} → {train_path}")
    print(f"Val:   {len(val_data)} → {val_path}")


if __name__ == "__main__":
    main()
