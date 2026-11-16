# Task 1 — LoRA fine-tuning and evaluation

**Model:** `Qwen/Qwen2.5-0.5B-Instruct` (0.5B parameters, within the 1.5B limit).  
**Method:** causal-LM supervised fine-tuning with PEFT LoRA.  
**Task:** classify customer support messages into five intents and emit a strict JSON object.

## Run (Linux GPU / Colab recommended)

```bash
pip install -r ../requirements.txt
python generate_dataset.py
python finetune.py --epochs 3 --batch-size 4 --grad-accum 4
python evaluate.py
```

The deterministic generator creates 300 examples and a stratified split with 240 train / 60 held-out examples. Evaluation reports JSON validity, intent accuracy, and exact JSON match for both the base model and the fine-tuned adapter. The test partition is not used during training.

## LoRA configuration and rationale

Defaults: rank `r=8`, alpha `16`, dropout `0.05`; target attention projections `q_proj,v_proj`; learning rate `2e-4`; three epochs. A small rank keeps the trainable adapter compact and is a reasonable starting point for a 0.5B model. The selected attention projections adapt token-attention behaviour without updating the full base weights. Report the trainable parameter count printed by the script.

## Failure analysis

After evaluation, inspect `results/failures.jsonl`. Pick three real held-out failures and explain (1) the input ambiguity, (2) the model's output, and (3) a concrete data/prompt/training fix. Do not write hypothetical failures as observed failures.

## 7B memory estimate

`memory_estimate.py` calculates parameter-weight storage for 7B parameters at FP32, FP16/BF16, and 4-bit and adds a clearly labelled optimizer/activation allowance. These are planning estimates, not measured GPU memory. Full fine-tuning needs gradients and optimizer states; LoRA freezes base weights and trains only adapters; QLoRA additionally stores the frozen base in 4-bit and dequantizes for compute.
