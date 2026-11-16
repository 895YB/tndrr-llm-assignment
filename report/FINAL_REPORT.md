# Tndrr LLM R&D Assignment — Final Report

**Candidate:** [Name]  
**Date:** [Date]  
**Repository:** https://github.com/895YB/tndrr-llm-assignment  
**Hardware:** [GPU model / VRAM, CPU, system RAM, OS, driver, CUDA]  
**Software:** [Python, PyTorch, Transformers, PEFT, bitsandbytes versions]

> Complete all result fields from the generated artifacts. Do not submit placeholders or theoretical estimates as measured results.

## Executive summary
[Summarize approach, major measured findings, limitations.]

## Task 1 — Fine-tuning
- Model and parameter count:
- Dataset generation and deterministic split: 240 train / 60 held-out (verify files).
- LoRA configuration: rank 8, alpha 16, dropout 0.05, target modules q_proj/v_proj; learning rate and epochs actually used:
- Trainable parameter count from run:
- Hardware, peak VRAM, training time:

| Metric | Base model | LoRA model |
|---|---:|---:|
| Held-out intent accuracy | | |
| Valid JSON rate | | |
| Exact JSON rate | | |

### Three observed failures
1. [Paste a real row from results/failures.jsonl; explain root cause and fix.]
2. [Real failure.]
3. [Real failure.]

### 7B memory estimate
[Paste task-1-finetuning/memory_estimate.py output. Explain weights versus gradients, optimizer state, adapters and activations. State assumptions.]

## Task 2 — RAG
- Source: RFC 9110 PDF from RFC Editor; actual downloaded page count:
- Embedding model:
- Retrieval implementation: direct sentence-transformer embeddings + NumPy cosine similarity (no LangChain / LlamaIndex):
- LLM used for grounded generation:
- Chunk overlap: 64 words.

| Chunk size | Number of chunks | Recall@5 |
|---:|---:|---:|
| 256 words | | |
| 512 words | | |

### Two observed retrieval failures
1. [Question, gold page, retrieved pages, cause, tested fix and measured before/after.]
2. [Same fields.]

### Design discussion
- BM25:
- Cross-encoder reranker:
- Why hybrid retrieval / reranking could help this source:

## Task 3 — Quantization and local serving
- Model:
- GPU:
- Same 10 prompts and decoding settings were used for both:
 
| Setting | Aggregate tokens/sec | Median prompt latency | Peak allocated VRAM | Peak reserved VRAM |
|---|---:|---:|---:|---:|
| FP16 | | | | |
| 4-bit NF4 | | | | |

- Qualitative output comparison: [Give concrete prompt IDs / examples.]
- KV-cache explanation:
- Quantization explanation and quality / memory trade-off:
- 20 concurrent users on 24 GB: [Paste serving_estimate.py result and explain limitations. Distinguish memory feasibility from throughput/SLA.]

## Conclusion
[What worked, measured trade-offs, and next experiments.]
