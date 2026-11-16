"""Validate syntax, manifests, fixed prompt/question counts and held-out split without model downloads."""
import ast, json, subprocess, sys, tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent
required=[
 "task-1-finetuning/finetune.py","task-1-finetuning/evaluate.py","task-1-finetuning/generate_dataset.py",
 "task-2-rag/rag.py","task-2-rag/questions.json",
 "task-3-local-llm/benchmark.py","task-3-local-llm/prompts.json",
 "report/FINAL_REPORT.md","COLAB_RUNBOOK.md"
]
missing=[p for p in required if not (ROOT/p).is_file()]
if missing: raise SystemExit(f"Missing required files: {missing}")
py_files=list(ROOT.rglob("*.py"))
for path in py_files: ast.parse(path.read_text(encoding="utf-8"),filename=str(path))
questions=json.loads((ROOT/"task-2-rag/questions.json").read_text(encoding="utf-8"))
prompts=json.loads((ROOT/"task-3-local-llm/prompts.json").read_text(encoding="utf-8"))
if len(questions)<15: raise SystemExit(f"Expected at least 15 RAG questions; found {len(questions)}")
if len(prompts)!=10: raise SystemExit(f"Expected exactly 10 benchmark prompts; found {len(prompts)}")
with tempfile.TemporaryDirectory() as tmp:
 script=ROOT/"task-1-finetuning/generate_dataset.py"
 subprocess.run([sys.executable,str(script),"--out",tmp],check=True)
 train=[json.loads(x) for x in (Path(tmp)/"train.jsonl").read_text().splitlines()]
 test=[json.loads(x) for x in (Path(tmp)/"test.jsonl").read_text().splitlines()]
 if len(test)<50: raise SystemExit(f"Held-out examples below 50: {len(test)}")
 if {x["template_id"] for x in train}&{x["template_id"] for x in test}: raise SystemExit("Template leakage found between train and test.")
 print(f"Dataset validation: {len(train)} train / {len(test)} held out; no shared templates.")
print(f"Python syntax: {len(py_files)} files passed.")
print(f"RAG questions: {len(questions)}; local benchmark prompts: {len(prompts)}.")
print("Static project validation passed. This does not execute GPU-dependent experiments.")
