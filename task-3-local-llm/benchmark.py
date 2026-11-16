"""Same 10 prompts, two precision settings; benchmark only on actual available CUDA hardware."""
import argparse,csv,json,time
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
MODEL="Qwen/Qwen2.5-0.5B-Instruct"
def load_model(setting,model_name):
 if setting=="fp16":
  return AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.float16,device_map="auto")
 if setting=="4bit":
  if not torch.cuda.is_available(): raise RuntimeError("4-bit bitsandbytes benchmark requires a compatible CUDA GPU. Run in a Linux GPU runtime.")
  q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
  return AutoModelForCausalLM.from_pretrained(model_name,quantization_config=q,device_map="auto",torch_dtype=torch.float16)
 raise ValueError(setting)
def main():
 p=argparse.ArgumentParser(); p.add_argument("--model",default=MODEL); p.add_argument("--settings",nargs="+",default=["fp16","4bit"],choices=["fp16","4bit"]); p.add_argument("--prompts",default="prompts.json"); p.add_argument("--max-new-tokens",type=int,default=128); p.add_argument("--out",default="results"); a=p.parse_args()
 prompts=json.loads(Path(a.prompts).read_text(encoding="utf-8"))
 if len(prompts)!=10: raise SystemExit(f"Expected exactly 10 fixed prompts, got {len(prompts)}.")
 if not torch.cuda.is_available(): raise SystemExit("Run on a CUDA GPU to produce comparable FP16/4-bit GPU measurements.")
 out=Path(a.out); out.mkdir(parents=True,exist_ok=True); tok=AutoTokenizer.from_pretrained(a.model); summary={}
 for setting in a.settings:
  torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
  load_start=time.perf_counter(); model=load_model(setting,a.model).eval(); load_seconds=time.perf_counter()-load_start
  rows=[]; total_tokens=0; total_seconds=0.0
  for i,prompt in enumerate(prompts,1):
   msg=[{"role":"user","content":prompt}]
   text=tok.apply_chat_template(msg,tokenize=False,add_generation_prompt=True)
   inputs=tok(text,return_tensors="pt").to("cuda")
   torch.cuda.synchronize(); start=time.perf_counter()
   with torch.no_grad(): ids=model.generate(**inputs,max_new_tokens=a.max_new_tokens,do_sample=False,pad_token_id=tok.eos_token_id)
   torch.cuda.synchronize(); elapsed=time.perf_counter()-start
   generated=ids[0][inputs["input_ids"].shape[1]:]
   n=int(generated.numel()); total_tokens+=n; total_seconds+=elapsed
   rows.append({"prompt_id":i,"prompt":prompt,"output":tok.decode(generated,skip_special_tokens=True),"generated_tokens":n,"generation_seconds":elapsed,"tokens_per_second":n/max(elapsed,1e-9)})
  allocated=torch.cuda.max_memory_allocated()/1e9; reserved=torch.cuda.max_memory_reserved()/1e9
  with (out/f"benchmark_{setting}.csv").open("w",newline="",encoding="utf-8") as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
  summary[setting]={"model":a.model,"prompts":len(rows),"total_generated_tokens":total_tokens,"total_generation_seconds":total_seconds,"aggregate_tokens_per_second":total_tokens/max(total_seconds,1e-9),"mean_prompt_tokens_per_second":sum(x["tokens_per_second"] for x in rows)/len(rows),"median_prompt_latency_seconds":sorted(x["generation_seconds"] for x in rows)[len(rows)//2],"model_load_seconds":load_seconds,"peak_cuda_allocated_GB":allocated,"peak_cuda_reserved_GB":reserved}
  del model; torch.cuda.empty_cache()
  print(setting,json.dumps(summary[setting],indent=2))
 (out/"summary.json").write_text(json.dumps(summary,indent=2))
 print(f"Saved real run results under {out.resolve()}")
if __name__=="__main__": main()
