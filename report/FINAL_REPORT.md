# Tndrr LLM R&D Assignment — Final Report

**Candidate:** Karthik Reddy  
**Date:** 17 November 2026  
**Repository:** https://github.com/895YB/tndrr-llm-assignment  
**Hardware:** Google Colab NVIDIA Tesla T4, 15,360 MiB (15 GiB-class) VRAM; NVIDIA driver 580.82.07; CUDA 13.0. CPU model, system RAM, OS image details and Python version were not captured in the supplied run log and are therefore not asserted here.  
**Software:** Run log recorded PyTorch 2.14.1+cu130, Transformers 4.57.6, PEFT 0.21.2, bitsandbytes 0.50.2, sentence-transformers 5.7.0, and PyMuPDF 2.1. Python version was not captured.

> Measurements below are from the Colab outputs shared for this experiment. Result artifacts generated locally in Colab must also be added to the repository before submission. No unmeasured serving-load or quality claims are made.

## Executive summary

This assignment evaluated LoRA fine-tuning, a from-scratch retrieval pipeline, and local inference under FP16 and NF4 4-bit weights using Qwen2.5-0.5B-Instruct. Fine-tuning used 240 synthetic training records and 60 template-held-out examples. LoRA trained 540,672 parameters (0.1093% of 494,573,440 total parameters). Intent accuracy rose from 23.33% to 30.00%, while valid-JSON rate fell from 100% to 60%, showing that the small training set improved classification only modestly and harmed output-format reliability.

The RFC 9110 retrieval experiment evaluated 15 authored questions. 256-word chunks achieved Recall@5 of 73.33% (11/15), compared with 60.00% (9/15) for 512-word chunks. Two status-code definition questions remained retrieval failures at both chunk sizes.

On the same ten prompts, FP16 reached 32.68 aggregate tokens/second with 1.008 GB peak allocated CUDA memory. NF4 4-bit reached 23.77 tokens/second with 0.553 GB peak allocated memory. The planning estimator estimated 4.93 GB for 20 concurrent sequences on a 24 GB GPU, but this is a memory estimate, not a measured concurrency test or an SLA guarantee.

## Task 1 — Fine-tuning

- **Model:** `Qwen/Qwen2.5-0.5B-Instruct`, approximately 0.5B parameters (494,573,440 parameters reported by the fine-tuning run).
- **Dataset:** deterministic synthetic customer-support intent data generated with seed 17. Five intents were used: billing, technical, account, shipping, cancellation. The generator held out one complete base-message template per intent, with twelve surface variants per template, yielding 240 training and 60 held-out records. This reduces near-duplicate leakage across the split.
- **LoRA configuration:** rank 8, alpha 16, dropout 0.05, target modules `q_proj` and `v_proj`; learning rate 2e-4; 3 epochs; per-device batch size 4; gradient accumulation 4; AdamW; weight decay 0.01; FP16 training and gradient checkpointing on CUDA.
- **Trainable parameters:** 540,672 / 494,573,440 = 0.1093%.
- **Training result:** reported train loss 0.48214; training runtime 39.16 seconds. The provided log did not include a peak CUDA-memory measurement for this training run, so none is reported.
- **Hardware:** Tesla T4, 15,360 MiB reported VRAM; driver 580.82.07; CUDA 13.0.

| Metric | Base model | LoRA model |
|---|---:|---:|
| Held-out intent accuracy | 23.33% (14/60) | 30.00% (18/60) |
| Valid JSON rate | 100.00% (60/60) | 60.00% (36/60) |
| Exact JSON rate | 23.33% (14/60) | 30.00% (18/60) |

The adapter improved intent accuracy by 6.67 percentage points and exact JSON accuracy by 6.67 points, but valid JSON fell by 40 percentage points. Exact JSON here means the parsed output exactly equals the expected one-key object `{"intent": "<gold-intent>"}`.

### Three observed failures

The following are the three held-out examples inspected from the fine-tuned evaluation output:

1. **Receipt request (billing).** Input: “Hi support, I need a copy of my receipt”. Expected intent: `billing`. The output did not produce the required JSON and instead described the issue as shipping-related. This is both an intent and format failure. A plausible cause is insufficient lexical/semantic generalization from the small synthetic billing templates. Improve coverage with more receipt/invoice/payment phrasings, add hard negative examples contrasting billing and shipping, and validate output against the allowed intent schema.
2. **Sign-in error (technical).** Input: “Hi support, I get an error when I sign in I would appreciate your help.” Expected intent: `technical`. The output listed intent numbers rather than returning the required JSON object. The classifier appears to have drifted into an explanation/list format. Reinforce the exact output contract with more varied training examples, a stronger system/instruction prefix, and constrained decoding or post-generation schema validation.
3. **Subscription closure (cancellation).** Input: “How do I close my subscription This is urgent.” Expected intent: `cancellation`. The model returned `{"intent":"subscription"}`, which is valid JSON but uses a label outside the five permitted classes. The failure indicates that output-format validity alone does not ensure label validity. Constrain generation to the permitted labels and add cancellation examples using “close subscription,” “end plan,” and “stop renewal.”

These examples are representative inspected failures; the full generated `results/failures.jsonl` contains the evaluation failure records and should be committed with the other result artifacts.

### 7B memory estimate

Output from `task-1-finetuning/memory_estimate.py`:

```text
FP32         weights=28.00 GB
FP16/BF16    weights=14.00 GB
INT8         weights=7.00 GB
4-bit        weights=3.50 GB
Full FT illustrative: 112.00 GB before activations/workspace
LoRA illustrative (fp16 base, 1% trainable): 15.12 GB before activations/workspace
QLoRA illustrative (4-bit base, 1% trainable): 4.62 GB before activations/workspace
These are estimates, not peak allocated memory. Quantization metadata, padding, activations, optimizer choice, and fragmentation change real usage.
```

The weight-only figures assume exactly 7 billion parameters and 4, 2, 1, or 0.5 bytes per parameter. The illustrative full fine-tuning estimate adds FP16 gradients, FP32 master weights, and two FP32 Adam moments to FP16 weights (16 bytes per parameter total). The LoRA and QLoRA examples freeze the base and assume trainable adapters equal to 1% of base parameter count, with the same illustrative trainable-parameter accounting. These are conservative planning illustrations, not measured peak memory. Actual LoRA adapter size depends on target modules and rank; activations, temporary workspaces, quantization metadata, padding and allocator fragmentation are not fully captured. The actual adapter in this experiment had 540,672 trainable parameters, not 1% of the base.

## Task 2 — RAG

- **Source:** RFC 9110, downloaded from the RFC Editor URL `https://www.rfc-editor.org/rfc/rfc9110.pdf`. The run output reported 194 PDF pages; all 194 pages had extractable text.
- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`.
- **Retrieval:** PDF text extraction with PyMuPDF; page-local word chunks; sentence-transformer embeddings; NumPy cosine similarity; top-5 retrieval. No LangChain or LlamaIndex was used.
- **Grounded answer-generation model:** `Qwen/Qwen2.5-0.5B-Instruct`. The RAG script generates answers using retrieved excerpts and page labels.
- **Questions:** 15 authored questions with gold source pages identified by matching question anchors against extracted PDF text.
- **Chunk overlap:** 64 words.

| Chunk size | Number of chunks | Recall@5 |
|---:|---:|---:|
| 256 words | 465 | 73.33% (11/15) |
| 512 words | 244 | 60.00% (9/15) |

The 256-word setting performed 13.33 percentage points better on this question set. This is a result for these authored questions and this source, not a universal claim that smaller chunks always perform better.

### Two observed retrieval failures

1. **q07 — “What is the meaning of the 3xx status-code class?”** Gold source page: 132. With 256-word chunks, the top-5 pages were 125, 145, 125, 145, 126. With 512-word chunks, they were 125, 126, 145, 146, 127. The gold page did not appear in either top five. Retrieval repeatedly selected general status-code material around page 125 rather than the specific 3xx definition. The tested experiment compared chunk sizes only; no BM25, hybrid retrieval or reranker fix was implemented/measured. Proposed next fix: section-aware chunking that retains the specific heading with its definition, then hybrid BM25+dense candidate retrieval and reranking.
2. **q08 — “What does the 4xx status-code class indicate?”** Gold source page: 137. With 256-word chunks, the top-5 pages were 125, 145, 125, 126, 141. With 512-word chunks, they were 125, 126, 145, 129, 127. The gold page was absent from both top fives. As with q07, broad status-code context outranked the exact definition. No retrieval correction was tested, so there is no before/after fix score to report. Proposed next fix: heading-aware chunks, lexical term matching for tokens such as “4xx” and “Client Error,” and a reranker over a larger candidate set.

### Design discussion

- **BM25:** a sparse lexical ranking method based on term frequency, inverse document frequency and document-length normalization. It can strongly reward exact query terms such as “3xx,” “4xx,” “Redirection,” and “Client Error,” which dense semantic similarity may underweight.
- **Cross-encoder reranker:** a model that jointly scores each query-passage pair. It is typically applied to a candidate set returned by BM25 and/or dense retrieval, reranking those candidates by query-specific relevance. It costs more computation than a single embedding search but can improve ordering among a manageable candidate set.
- **Why hybrid retrieval/reranking may help:** RFCs contain precise identifiers, section labels and technical terms. Dense embeddings can retrieve broadly related status-code explanations, while sparse retrieval can emphasize exact class tokens and terminology. A reranker may then distinguish the precise definition from general overview passages. These are proposed improvements; they were not benchmarked in this run.

## Task 3 — Quantization and local serving

- **Model:** `Qwen/Qwen2.5-0.5B-Instruct`.
- **GPU:** NVIDIA Tesla T4, 15,360 MiB VRAM, driver 580.82.07, CUDA 13.0.
- **Workload:** identical 10 fixed prompts from `prompts.json`; greedy decoding (`do_sample=False`); maximum 128 new tokens. FP16 and bitsandbytes NF4 4-bit settings were benchmarked separately.
 
| Setting | Aggregate tokens/sec | Median prompt latency | Peak allocated VRAM | Peak reserved VRAM |
|---|---:|---:|---:|---:|
| FP16 | 32.68 | 3.789 s | 1.008 GB | 1.026 GB |
| 4-bit NF4 | 23.77 | 5.403 s | 0.553 GB | 1.021 GB |

- **Generated tokens:** 1,280 for FP16 and 1,274 for NF4 4-bit.
- **Model load time:** 5.882 s for FP16 and 2.458 s for NF4 4-bit (reported separately from generation throughput).
- **Qualitative output comparison:** the benchmark saved paired prompt/output records in `benchmark_fp16.csv` and `benchmark_4bit.csv`. Those CSV contents were not provided in the report evidence, so no claim is made that outputs were identical or that one setting had better answer quality. Inspect the paired outputs by prompt ID and add concrete examples after committing/reviewing those CSVs.
- **KV cache:** during autoregressive generation, attention key and value tensors from prior tokens are retained so the model can process the next token without recomputing the full preceding context. KV cache memory grows with concurrent sequences, sequence length, number of transformer layers, key/value heads, head dimension and cache precision. Weight quantization does not automatically quantize the KV cache.
- **Quantization trade-off:** NF4 represents model weights using four bits, with additional scales/metadata and FP16 compute in this configuration. In this run it reduced peak allocated CUDA memory from 1.008 GB to 0.553 GB (about 45.1% lower), while aggregate generation throughput decreased from 32.68 to 23.77 tokens/sec (about 27.3% lower). The benchmark did not calculate a formal output-quality score.
- **20 concurrent users on a 24 GB GPU:** estimator output for 20 users, 2,048 context tokens, 512 generated tokens per user, 2,560-token sequence length:
- KV cache estimate: 0.63 GB.
- 4-bit model weights estimate: 0.30 GB for a 0.5B parameter model.
- Runtime and memory fragmentation planning buffer: 4.00 GB.
- Estimated planning total: 4.93 GB out of 24 GB GPU memory.
- Estimated memory headroom: 19.07 GB.

This is a theoretical memory estimate, not a confirmed serving-capacity result. A representative 20-client load test was not performed. Therefore, this estimate does not guarantee reliable serving for 20 concurrent users.

```text
Assumptions: 4-bit model weights with 20% quantization metadata allowance; FP16 KV cache;
users=20, context=2048, generated_tokens=512, sequence=2560
KV cache estimate: 0.63 GB
4-bit weights estimate: 0.30 GB for 0.50B parameters
Runtime / fragmentation planning buffer: 4.00 GB
Estimated planning total: 4.93 GB of 24.00 GB GPU memory
Memory-only headroom: 19.07 GB
This is not a capacity guarantee: benchmark prefill/decode throughput, batching, peak memory and p95 latency with a representative 20-client load test.
```

This calculation indicates a theoretical memory planning total below 24 GB under the stated assumptions. It does not demonstrate that the system can sustain 20 real users at a target throughput or latency. A representative concurrent load test must measure prompt prefill, token decode throughput, batching/scheduler overhead, peak memory and p95 latency. The run was an estimator only; no 20-client load test was performed.

## Conclusion

The assignment demonstrated a full LoRA training/evaluation loop, a transparent RAG implementation without orchestration frameworks, and a paired local inference benchmark. The measured LoRA result improved intent accuracy modestly but exposed a serious regression in valid JSON output, emphasizing that structured-output reliability needs its own metric and controls. On the RFC question set, 256-word chunks outperformed 512-word chunks, but exact status-code definition queries remained misses at both sizes. NF4 lowered allocated GPU memory but also lowered observed generation speed relative to FP16 on the T4.

The next experiments should expand and balance the intent dataset, enforce the allowed JSON schema during decoding, add section-aware/hybrid retrieval with a reranker and measure Recall@5 again, inspect paired FP16/4-bit answers for quality differences, and conduct a real 20-client load test with defined latency/throughput targets.
