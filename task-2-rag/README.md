# Task 2 — RAG from scratch (no LangChain / LlamaIndex)

This pipeline downloads **RFC 9110, HTTP Semantics** from the RFC Editor as a PDF, verifies that it contains at least 30 pages, extracts page text, embeds chunks with Sentence Transformers, and retrieves with NumPy cosine similarity. It then passes the retrieved passages to a small instruction model for grounded answer generation.

## Run

```bash
pip install -r ../requirements.txt
python rag.py --chunk-sizes 256 512 --top-k 5
```

First run downloads the PDF and embedding/model weights. Use a GPU runtime for generation when possible. Add `--skip-generation` to run document processing and retrieval evaluation only.

## Dataset / evaluation

The 15 authored questions in `questions.json` include a reference phrase (anchor) from the RFC. The script maps each anchor to its actual PDF page and then to the corresponding chunk. It reports Recall@5 separately for 256-word and 512-word chunks, saves each question's top-5 source pages and passages, and writes `results/metrics.json`. The anchor-to-page mapping is printed and persisted so the source evidence is auditable.

If an anchor cannot be found in your local extracted PDF, the run stops instead of silently evaluating a smaller / changed question set. The RFC is a 194-page standards document; the runtime check verifies the copy actually downloaded.

## Retrieval failure analysis

Inspect `results/retrieval_256.jsonl` and `results/retrieval_512.jsonl`. Document two real misses: question, gold page, retrieved pages, likely cause (e.g. acronym mismatch, terminology mismatch, relevant detail split at chunk boundary), and measured fix. Useful fixes to test include overlap, query expansion, BM25+dense hybrid retrieval, and reranking. Do not claim a fix improved Recall@5 unless rerun metrics show it.

## BM25 and reranker

BM25 is a sparse lexical ranking function based on term frequency, inverse document frequency, and document-length normalization. It is effective for exact technical terms, identifiers, and rare words, but can miss semantic paraphrases. A cross-encoder reranker jointly scores each query-passage pair and reorders a shortlist; it is more computationally expensive than bi-encoder retrieval but can improve relevance at the cost of an extra inference step.
