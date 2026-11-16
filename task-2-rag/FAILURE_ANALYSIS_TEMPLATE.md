# RAG retrieval failure analysis

Complete this after running `rag.py`.

## Failure 1
- Question ID / query:
- Gold RFC page(s):
- Top-5 retrieved pages:
- Why retrieval missed (lexical mismatch, chunk boundary, document structure, embedding issue, other):
- Fix applied:
- Recall@5 before / after (from actual runs):

## Failure 2
- Question ID / query:
- Gold RFC page(s):
- Top-5 retrieved pages:
- Why retrieval missed:
- Fix applied:
- Recall@5 before / after:

Candidate fixes to test: change overlap, query expansion, combine BM25 with dense scores, or rerank top-20 dense candidates with a cross-encoder. Retain the same 15 questions and report each measured result.
