"""Compare base and LoRA model on the identical held-out set."""
import json,re,csv
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import PeftModel
MODEL="Qwen/Qwen2.5-0.5B-Instruct"
def parse(s):
 m=re.search(r"\{.*?\}",s,re.S)
 if not m: return None
 try:
  x=json.loads(m.group())
  return x if isinstance(x,dict) else None
 except Exception: return None
def run(model,tok,rows):
 out=[]
 device=next(model.parameters()).device
 for r in rows:
  prompt=f"Classify this customer support message into one intent: billing, technical, account, shipping, cancellation. Return only JSON.\nMessage: {r['text']}\nJSON: "
  ins=tok(prompt,return_tensors="pt").to(device)
  with torch.no_grad(): ids=model.generate(**ins,max_new_tokens=40,do_sample=False,pad_token_id=tok.eos_token_id)
  gen=tok.decode(ids[0][ins["input_ids"].shape[1]:],skip_special_tokens=True)
  obj=parse(gen); pred=obj.get("intent") if obj else None
  out.append({"text":r["text"],"gold":r["intent"],"prediction":pred,"raw":gen,"valid_json":obj is not None,"exact":obj=={"intent":r["intent"]}})
 return out
def score(rows):
 n=len(rows); return {"n":n,"intent_accuracy":sum(x["prediction"]==x["gold"] for x in rows)/n,"valid_json_rate":sum(x["valid_json"] for x in rows)/n,"exact_json_rate":sum(x["exact"] for x in rows)/n}
def main():
 test=[json.loads(x) for x in Path("data/test.jsonl").read_text().splitlines()]
 if len(test)<50: raise SystemExit("Need at least 50 held-out examples.")
 tok=AutoTokenizer.from_pretrained(MODEL); dtype=torch.float16 if torch.cuda.is_available() else torch.float32
 base=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=dtype,device_map="auto" if torch.cuda.is_available() else None).eval()
 base_rows=run(base,tok,test); del base
 if torch.cuda.is_available(): torch.cuda.empty_cache()
 adapter=Path("outputs/lora-adapter")
 if not adapter.exists(): raise SystemExit("Fine-tuned adapter not found. Run finetune.py first.")
 tuned_base=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=dtype,device_map="auto" if torch.cuda.is_available() else None)
 tuned=PeftModel.from_pretrained(tuned_base,str(adapter)).eval()
 tuned_rows=run(tuned,tok,test)
 Path("results").mkdir(exist_ok=True)
 for name,rows in [("base",base_rows),("lora",tuned_rows)]:
  with open(f"results/{name}_predictions.jsonl","w") as f:
   for x in rows: f.write(json.dumps(x,ensure_ascii=False)+"\n")
 failures=[x for x in tuned_rows if not x["exact"]]
 with open("results/failures.jsonl","w") as f:
  for x in failures: f.write(json.dumps(x,ensure_ascii=False)+"\n")
 report={"base":score(base_rows),"lora":score(tuned_rows),"heldout_examples":len(test),"lora_failures":len(failures)}
 Path("results/metrics.json").write_text(json.dumps(report,indent=2))
 print(json.dumps(report,indent=2)); print("Review results/failures.jsonl and document three actual failures in the report.")
if __name__=="__main__": main()
