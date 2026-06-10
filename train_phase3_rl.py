"""
Phase 3 – verl GRPO RL Training
================================
Uses verl's built-in ``main_ppo`` entry point with GRPO advantage
estimation. The SwiLatent-specific parts are:

1. **Custom reward** (``src/rl/reward.py:compute_score``):
   correctness + efficiency (bonus for <swi> usage when correct) + format.

2. **Merged model** (``scripts/merge_lora.py``):
   Phase 2 LoRA adapter merged into a standalone Qwen3-8B checkpoint
   that vLLM can load directly for rollout generation.

3. **Data** (``scripts/prep_verl_data.py``):
   Training JSONL converted to verl's Parquet format with
   ``data_source``, ``prompt``, ``reward_model.ground_truth`` columns.

Strategy A: Fixed Latent Steps
------------------------------
During RL, vLLM generates text with standard autoregressive decoding
(the model writes ``<swi>...</swi>`` as text, not latent inference).
RL optimizes correctness and tag usage via the reward function.

At evaluation time, ``CoconutSwiModel.generate(min_latent_steps=16)``
performs actual latent inference with hidden-state injection.

Usage
-----
See ``scripts/train_phase3_rl.sh`` for the full launch command.
The entry point is verl's built-in trainer::

    python3 -m verl.trainer.main_ppo \\
        algorithm.adv_estimator=grpo \\
        reward.custom_reward_function.path=src/rl/reward.py \\
        reward.custom_reward_function.name=compute_score \\
        ...

Preparation Steps
-----------------
1. Merge LoRA into standalone model::

    python scripts/merge_lora.py \\
        --base_model <base_qwen3_8b_path> \\
        --adapter_path outputs/phase2_2_v1_bf16/stage_008 \\
        --output_dir outputs/phase2_2_merged

2. Prepare data in verl format::

    python scripts/prep_verl_data.py \\
        --input_file data_hf/data/math_openr1_all.jsonl \\
        --output_dir data_verl

3. Launch RL training::

    bash scripts/train_phase3_rl.sh

Future Strategies
-----------------
- **Strategy B**: Gradually decrease ``min_latent_steps`` (16 → 0) during
  RL by implementing a custom rollout worker that uses
  ``CoconutSwiModel.generate()`` instead of vLLM.
- **Strategy C**: Remove minimum entirely, add latent-depth reward to
  encourage multi-step latent reasoning organically.
"""
