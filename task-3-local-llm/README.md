# Task 3 — Local LLM quantization benchmark

**Model:** `Qwen/Qwen2.5-0.5B-Instruct`.  
**Settings:** FP16 (baseline) and bitsandbytes NF4 4-bit.  
**Workload:** identical 10 fixed prompts, greedy decoding, maximum 128 new tokens.

## Run

Use Linux with an NVIDIA CUDA GPU and a compatible bitsandbytes installation (Colab GPU is suitable). 4-bit bitsandbytes generally requires CUDA; this benchmark deliberately stops with a clear error rather than pretending a CPU 4-bit run is equivalent.

```bash
pip install -r ../requirements.txt
python benchmark.py --settings fp16 4bit
python serving_estimate.py --users 20 --context 2048 --new-tokens 512 --gpu-gb 24
```

The script writes per-prompt output and timing to `results/benchmark_<setting>.csv`, plus aggregate throughput and peak CUDA memory in `results/summary.json`. It uses the same prompt set and generation parameters for both model settings. Tokens/sec is computed from generated output tokens divided by measured generation wall time; model loading time is excluded and reported separately.

## What to compare

Discuss average generated tokens/sec, median latency, peak allocated / reserved GPU memory, and qualitative output differences. Review the paired outputs in the CSV rather than relying on a vague claim that quantization is “similar.” Small quantized models can change wording, factuality, and instruction-following.

## KV cache and quantization

Autoregressive decoding stores attention keys and values from prior tokens so each next token does not recompute the entire prefix. KV cache grows with batch/concurrent sequences, context length, layer count, key/value heads, head dimension, and cache precision. Weight quantization reduces model-weight memory; it does not automatically quantize the KV cache. A 4-bit model can therefore still run out of memory with long contexts and large concurrency.

The serving estimator calculates an idealized KV-cache requirement using the architecture parameters and reports a planning buffer. Memory fit alone does not prove 20-user SLA: throughput, batching, prefill bursts, scheduler overhead, context distribution, and latency targets must also be load-tested.
