#!/usr/bin/env python3
"""Quick analysis of Phase 2 training log."""
import json
import sys
from collections import defaultdict

log_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/phase2_1_v2/trainer_log.jsonl"

with open(log_path) as f:
    lines = [json.loads(l) for l in f if l.strip()]

steps = [l for l in lines if l.get("event") == "train_step"]
stage_data = defaultdict(list)
for s in steps:
    stage_data[s["stage"]].append(s["loss"])

print("Stage | Logged | AvgLoss | First3              | Last3")
print("-" * 75)
for st in sorted(stage_data):
    ls = stage_data[st]
    f3 = ", ".join("{:.2f}".format(x) for x in ls[:3])
    l3 = ", ".join("{:.2f}".format(x) for x in ls[-3:])
    avg = sum(ls) / len(ls)
    print("  {:>2}  |  {:>3}   |  {:.3f}  | {:20s} | {}".format(st, len(ls), avg, f3, l3))

print()
print("Total logged entries: {}".format(len(steps)))
print("Latest global_step: {}".format(steps[-1]["global_step"]))
avg_per_stage = len(steps) // max(len(stage_data), 1)
print("Avg logged steps per stage: {} (x10 = ~{} opt steps)".format(avg_per_stage, avg_per_stage * 10))
print("Expected: 9030 samples / batch16 = 564 opt steps per epoch")
