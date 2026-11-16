# Run the assignment on Google Colab (GPU)

The repository is designed to run on a Linux CUDA notebook. Your laptop does not need a discrete GPU if you use Colab.

## 1. Start a GPU runtime
Open Google Colab, create a notebook, choose **Runtime → Change runtime type → T4 GPU** (or another available NVIDIA GPU). Confirm availability:

```python
!nvidia-smi
```

## 2. Clone and install

```bash
!git clone https://github.com/895YB/tndrr-llm-assignment.git
%cd tndrr-llm-assignment
!pip install -r requirements.txt
```

Restart the runtime only if Colab asks you to after installation.

## 3. Task 1 — Fine-tune and evaluate

```bash
%cd /content/tndrr-llm-assignment/task-1-finetuning
!python generate_dataset.py
!python finetune.py --epochs 3 --batch-size 4 --grad-accum 4
!python evaluate.py
!python memory_estimate.py
```

Expected dataset sizes are 240 training rows and 60 held-out rows. Check `results/metrics.json` and inspect at least three actual records in `results/failures.jsonl`. If there are fewer than three failures, explicitly state that and discuss the observed model limitations instead of inventing failed predictions.

## 4. Task 2 — Build and evaluate RAG

```bash
%cd /content/tndrr-llm-assignment/task-2-rag
!python rag.py --chunk-sizes 256 512 --top-k 5
```

First run downloads the RFC 9110 PDF and model weights. The script verifies the document has at least 30 pages. Check `results/metrics.json`, `results/gold_sources.json`, and both retrieval JSONL files. Report two real retrieval misses if any; if all queries are retrieved, analyze the weakest-ranked examples and label them correctly as near-misses rather than failures.

## 5. Task 3 — Compare FP16 and 4-bit

```bash
%cd /content/tndrr-llm-assignment/task-3-local-llm
!python benchmark.py --settings fp16 4bit
!python serving_estimate.py --users 20 --context 2048 --new-tokens 512 --gpu-gb 24
```

Check `results/summary.json` and the two per-prompt CSV files. Both modes use the same 10 prompts and greedy decoding. Record the Colab GPU model and VRAM shown by `nvidia-smi`.

## 6. Finish the report and commit evidence

Complete `report/FINAL_REPORT.md` with the actual outputs. Also commit the generated `metrics.json`, `failures.jsonl`, RAG retrieval JSONL, and benchmark CSV/summary into their respective `results/` folders. Generated model weights and downloaded source PDF are intentionally excluded from Git.

If pushing from Colab, configure Git identity and use a GitHub token with repository write access. Never paste the token into a committed notebook or file. Alternatively, download the generated result files and upload them through GitHub's web interface.

**Important:** Runtime, GPU availability, model download time, peak VRAM, accuracy, Recall@5, and throughput vary. Preserve the exact observed values; do not substitute expected or estimated metrics.
