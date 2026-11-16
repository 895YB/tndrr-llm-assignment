"""RAG pipeline implemented directly with PyMuPDF, SentenceTransformers, NumPy and Transformers."""
import argparse, json, re, urllib.request
from pathlib import Path
import fitz
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

PDF_URL="https://www.rfc-editor.org/rfc/rfc9110.pdf"
EMBED_MODEL="sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
def norm(s): return re.sub(r"\s+"," ",s).strip().lower()
def download_pdf(path):
 path.parent.mkdir(parents=True,exist_ok=True)
 if not path.exists():
  print(f"Downloading {PDF_URL}")
  urllib.request.urlretrieve(PDF_URL,path)
 doc=fitz.open(path)
 if len(doc)<30: raise RuntimeError(f"Source PDF has only {len(doc)} pages; expected >=30.")
 pages=[{"page":i+1,"text":doc[i].get_text("text")} for i in range(len(doc))]
 if sum(bool(x["text"].strip()) for x in pages)<30: raise RuntimeError("Fewer than 30 pages contain extractable text.")
 print(f"Loaded RFC 9110: {len(doc)} pages; {sum(bool(x['text'].strip()) for x in pages)} pages with text.")
 return pages
def make_chunks(pages,size,overlap=64):
 chunks=[]
 for p in pages:
  words=p["text"].split()
  for start in range(0,len(words),max(1,size-overlap)):
   part=" ".join(words[start:start+size])
   if part.strip(): chunks.append({"id":f"p{p['page']}_w{start}","page":p["page"],"text":part})
 return chunks
def cosine_matrix(a,b):
 a=a/(np.linalg.norm(a,axis=1,keepdims=True)+1e-12)
 b=b/(np.linalg.norm(b,axis=1,keepdims=True)+1e-12)
 return a@b.T
def load_questions(path): return json.loads(path.read_text(encoding="utf-8"))
def resolve_gold(questions,pages):
 out=[]
 for q in questions:
  matches=[p["page"] for p in pages if norm(q["anchor"]) in norm(p["text"])]
  if not matches: raise RuntimeError(f"Question {q['id']} anchor not found in extracted PDF: {q['anchor']!r}. Check source version or anchor.")
  out.append({**q,"gold_pages":matches})
 return out
def retrieve(questions,chunks,embedder,k):
 docs=[c["text"] for c in chunks]
 dv=embedder.encode(docs,convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=True)
 qv=embedder.encode([q["question"] for q in questions],convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=False)
 sims=qv@dv.T; rows=[]; hits=0
 for i,q in enumerate(questions):
  idx=np.argsort(-sims[i])[:k]
  chosen=[chunks[int(j)] for j in idx]
  hit=any(c["page"] in q["gold_pages"] for c in chosen); hits+=int(hit)
  rows.append({"id":q["id"],"question":q["question"],"anchor":q["anchor"],"gold_pages":q["gold_pages"],"recall_hit":hit,
   "top5":[{"chunk_id":c["id"],"page":c["page"],"score":float(sims[i,j]),"text":c["text"]} for c,j in zip(chosen,idx)]})
 return hits/len(questions),rows
def generate_answers(rows,model_name):
 tok=AutoTokenizer.from_pretrained(model_name)
 dtype=torch.float16 if torch.cuda.is_available() else torch.float32
 model=AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=dtype,device_map="auto" if torch.cuda.is_available() else None).eval()
 device=next(model.parameters()).device
 for r in rows:
  ctx="\n\n".join(f"[RFC 9110 page {x['page']}] {x['text']}" for x in r["top5"][:3])
  prompt=("Answer using only the supplied RFC excerpts. If the excerpts do not contain the answer, say that the provided passages are insufficient. Cite page numbers.\n"
          f"Question: {r['question']}\nExcerpts:\n{ctx}\nAnswer:")
  ins=tok(prompt,return_tensors="pt",truncation=True,max_length=3500).to(device)
  with torch.no_grad(): ids=model.generate(**ins,max_new_tokens=140,do_sample=False,pad_token_id=tok.eos_token_id)
  r["answer"]=tok.decode(ids[0][ins["input_ids"].shape[1]:],skip_special_tokens=True)
 return rows
def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--pdf",default="data/rfc9110.pdf"); ap.add_argument("--chunk-sizes",type=int,nargs="+",default=[256,512]); ap.add_argument("--top-k",type=int,default=5); ap.add_argument("--skip-generation",action="store_true"); ap.add_argument("--embedding-model",default=EMBED_MODEL); ap.add_argument("--llm",default=LLM_MODEL); a=ap.parse_args()
 pdf=Path(a.pdf); pages=download_pdf(pdf); questions=resolve_gold(load_questions(Path("questions.json")),pages)
 out=Path("results"); out.mkdir(exist_ok=True); (out/"gold_sources.json").write_text(json.dumps(questions,indent=2))
 embedder=SentenceTransformer(a.embedding_model,device="cuda" if torch.cuda.is_available() else "cpu")
 metrics={}
 for size in a.chunk_sizes:
  chunks=make_chunks(pages,size)
  recall,rows=retrieve(questions,chunks,embedder,a.top_k)
  metrics[str(size)]={"chunk_words":size,"overlap_words":64,"questions":len(questions),"recall_at_5":recall,"chunks":len(chunks)}
  print(f"Chunk size {size} words: Recall@{a.top_k} = {recall:.3f} ({round(recall*len(questions))}/{len(questions)})")
  if size==a.chunk_sizes[-1] and not a.skip_generation: rows=generate_answers(rows,a.llm)
  with (out/f"retrieval_{size}.jsonl").open("w",encoding="utf-8") as f:
   for r in rows: f.write(json.dumps(r,ensure_ascii=False)+"\n")
 (out/"metrics.json").write_text(json.dumps(metrics,indent=2))
 print(json.dumps(metrics,indent=2)); print("Inspect saved JSONL for retrieval misses and cite page numbers from gold_sources.json.")
if __name__=="__main__": main()
