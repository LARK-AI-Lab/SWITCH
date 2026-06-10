"""
Phase 1 – SFT Data Preprocessing
==================================
Converts math_openr1_all.jsonl (with extra metadata fields) into a clean
alpaca-format JSONL suitable for LLaMA-Factory SFT training.

What it does
------------
1. Reads the raw JSONL (fields: instruction, input, output, is_correct,
   predicted_answer, ground_truth_answer, dataset_index).
2. Keeps only samples where  is_correct=True  (all samples currently, kept
   as a safety gate for future re-annotation runs).
3. Strips the metadata fields; outputs only: instruction, input, output.
4. Writes to <output_dir>/math_openr1_swi_sft.jsonl  (LLaMA-Factory data dir).

Usage
-----
    python scripts/preprocess_sft_data.py \
        --input  /data/jiayu/SwiLatent/logs/math_openr1_all.jsonl \
        --output /data/jiayu/LLaMA-Factory/data/math_openr1_swi_sft.jsonl

Output statistics are printed to stdout.
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="/data/jiayu/SwiLatent/logs/math_openr1_all.jsonl",
        help="Path to raw annotated JSONL file.",
    )
    parser.add_argument(
        "--output",
        default="/data/jiayu/LLaMA-Factory/data/math_openr1_swi_sft.jsonl",
        help="Destination path for the clean LLaMA-Factory JSONL.",
    )
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = kept = skipped_incorrect = skipped_no_swi = 0

    with input_path.open() as fin, output_path.open("w") as fout:
        for raw_line in fin:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            total += 1
            try:
                d = json.loads(raw_line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Line {total}: JSON parse error – {e}", file=sys.stderr)
                continue

            # ── Quality gates ────────────────────────────────────────────
            if not d.get("is_correct", True):
                skipped_incorrect += 1
                continue

            output_text = d.get("output", "")
            if "<swi>" not in output_text:
                # Keep samples without <swi> as well – they serve as
                # negative examples (model learns NOT every response needs <swi>)
                skipped_no_swi += 1  # just count for info, still keep them

            # ── Extract alpaca fields only ───────────────────────────────
            record = {
                "instruction": d["instruction"],
                "input":       d.get("input", ""),
                "output":      output_text,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Input  : {input_path}  ({total} lines)")
    print(f"Output : {output_path}")
    print(f"  Kept             : {kept}")
    print(f"  Skipped (wrong)  : {skipped_incorrect}")
    print(f"  No <swi> (kept!) : {skipped_no_swi}")


if __name__ == "__main__":
    main()
