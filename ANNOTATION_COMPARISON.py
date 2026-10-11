"""
Compare annotation strategies: Teacher Forcing vs Generation-Based

This script demonstrates why Generation-Based annotation is superior.
"""

print("="*80)
print("ANNOTATION STRATEGY COMPARISON")
print("="*80)

print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                    TEACHER FORCING (Old Method)                           ║
╚═══════════════════════════════════════════════════════════════════════════╝

Process:
1. Load ground truth solution: "Step 1... Step 2... #### 42"
2. Feed to model token-by-token (teacher forcing)
3. Compute entropy at each step based on MODEL'S prediction
4. Mark high-entropy tokens with <swi>

Problems:
❌ Model sees GROUND TRUTH tokens, not its own generations
❌ Distribution mismatch: training ≠ inference behavior
❌ Entropy reflects model's surprise at GROUND TRUTH, not reasoning difficulty
❌ 70% of samples wrap final answer with <swi> (incorrect)
❌ No guarantee model will generate same tokens during training

Example Issue:
  Ground truth: "48/2 = 24"
  Model prediction: "48 divided by 2 equals 24"
  → Model is "surprised" by ground truth formatting
  → High entropy, but NOT because reasoning is hard!

╔═══════════════════════════════════════════════════════════════════════════╗
║                  GENERATION-BASED (New Method)                            ║
╚═══════════════════════════════════════════════════════════════════════════╝

Process:
1. Let model GENERATE its own solution
2. Track entropy during generation
3. Annotate <swi> where entropy was HIGH during generation
4. Use generated solution as training data

Advantages:
✅ Model generates naturally (inference-like conditions)
✅ Entropy reflects ACTUAL reasoning uncertainty
✅ <swi> placement matches model's natural reasoning flow
✅ Protects final answer region explicitly
✅ Can filter for only CORRECT solutions

Example:
  Model generates: "First, divide 48 by 2..."
  → Low entropy (confident in basic operation)
  
  Model generates: "To find combinations, we need <swi>..."
  → High entropy (complex combinatorial reasoning)
  → System marks this region for latent thinking

╔═══════════════════════════════════════════════════════════════════════════╗
║                         KEY DIFFERENCES                                   ║
╚═══════════════════════════════════════════════════════════════════════════╝

┌─────────────────────┬──────────────────────┬────────────────────────────┐
│ Aspect              │ Teacher Forcing      │ Generation-Based           │
├─────────────────────┼──────────────────────┼────────────────────────────┤
│ Input to model      │ Ground truth tokens  │ Model's own generations    │
│ Entropy meaning     │ Surprise at GT       │ Actual uncertainty         │
│ Data quality        │ 70% wraps answers    │ Natural <swi> placement    │
│ Train/test match    │ Mismatch             │ Aligned                    │
│ Can filter quality  │ No                   │ Yes (only correct)         │
│ Follows SWIR paper  │ Partially            │ Closely                    │
└─────────────────────┴──────────────────────┴────────────────────────────┘

╔═══════════════════════════════════════════════════════════════════════════╗
║                     RECOMMENDED WORKFLOW                                  ║
╚═══════════════════════════════════════════════════════════════════════════╝

Step 1: Generate and Annotate Training Data
```bash
python annotate_gsm8k_generation_based.py \\
    --model_name Qwen/Qwen2.5-Coder-3B-Instruct \\
    --max_samples 1000 \\
    --only_correct \\
    --entropy_percentile 60 \\
    --output_path data/gsm8k_swir_stage1.jsonl
```

Step 2: Convert to Stage 2 Format (Coconut)
```bash
python convert_to_latent_format.py \\
    --input data/gsm8k_swir_stage1.jsonl \\
    --output data/gsm8k_swir_stage2.json \\
    --num_latents 2
```

Step 3: Train Stage 1 (Learn when to output <swi>)
```bash
cd /data/jiayu/LLaMA-Factory
llamafactory-cli train scripts/stage1_config.yaml
```

Step 4: Train Stage 2 (Coconut latent reasoning)
```bash
python scripts/run_stage2_coconut_v2.py \\
    --base_model outputs/stage1 \\
    --data_path data/gsm8k_swir_stage2.json
```

╔═══════════════════════════════════════════════════════════════════════════╗
║                    PARAMETER RECOMMENDATIONS                              ║
╚═══════════════════════════════════════════════════════════════════════════╝

Annotation Parameters:
  --window_size 10          # Smooth entropy over 10 tokens
  --entropy_percentile 60   # Mark top 40% entropy regions
  --min_swi_length 3        # Minimum 3 tokens per <swi> region
  --only_correct           # Only use correct solutions

Tuning Guidance:
  • Higher percentile (70-80) → fewer, more selective <swi> regions
  • Lower percentile (40-50) → more aggressive latent usage
  • Larger window → smoother, longer <swi> regions
  • Start conservative (60-70), then adjust based on results

""")

print("="*80)
print("\nReady to test? Run:")
print("  python annotate_gsm8k_generation_based.py --max_samples 20")
print("="*80)
