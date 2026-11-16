"""LoRA fine-tuning for Qwen2.5-0.5B-Instruct."""
import argparse, json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType

MODEL="Qwen/Qwen2.5-0.5B-Instruct"
class IntentDataset(Dataset):
 def __init__(self, rows, tok, max_len=256):
  self.items=[]
  for r in rows:
   prompt=f"Classify this customer support message into one intent: billing, technical, account, shipping, cancellation. Return only JSON.\nMessage: {r['text']}\nJSON: "
   full=prompt+r["target"]+tok.eos_token
   enc=tok(full,truncation=True,max_length=max_len)
   prompt_len=len(tok(prompt, truncation=True,max_length=max_len)["input_ids"])
   labels=enc["input_ids"].copy()
   labels[:min(prompt_len,len(labels))]=[-100]*min(prompt_len,len(labels))
   self.items.append((enc["input_ids"],enc["attention_mask"],labels))
 def __len__(self): return len(self.items)
 def __getitem__(self,i):
  x,m,y=self.items[i]; return {"input_ids":torch.tensor(x),"attention_mask":torch.tensor(m),"labels":torch.tensor(y)}
class Pad:
 def __init__(self,pad): self.pad=pad
 def __call__(self,items):
  n=max(len(x["input_ids"]) for x in items); out={}
  for k in items[0]:
   fill=-100 if k=="labels" else self.pad if k=="input_ids" else 0
   out[k]=torch.stack([torch.nn.functional.pad(x[k],(0,n-len(x[k])),value=fill) for x in items])
  return out
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--model",default=MODEL); p.add_argument("--epochs",type=int,default=3)
 p.add_argument("--batch-size",type=int,default=4); p.add_argument("--grad-accum",type=int,default=4)
 p.add_argument("--lr",type=float,default=2e-4); p.add_argument("--r",type=int,default=8); p.add_argument("--alpha",type=int,default=16)
 p.add_argument("--out",default="outputs/lora-adapter"); a=p.parse_args()
 train_file=Path("data/train.jsonl")
 if not train_file.exists(): raise SystemExit("Run python generate_dataset.py first.")
 rows=[json.loads(x) for x in train_file.read_text().splitlines()]
 tok=AutoTokenizer.from_pretrained(a.model); tok.pad_token=tok.pad_token or tok.eos_token
 dtype=torch.float16 if torch.cuda.is_available() else torch.float32
 model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=dtype,device_map="auto" if torch.cuda.is_available() else None)
 model.config.use_cache=False
 if hasattr(model,"enable_input_require_grads"): model.enable_input_require_grads()
 cfg=LoraConfig(task_type=TaskType.CAUSAL_LM,r=a.r,lora_alpha=a.alpha,lora_dropout=0.05,target_modules=["q_proj","v_proj"],bias="none")
 model=get_peft_model(model,cfg); model.print_trainable_parameters()
 ds=IntentDataset(rows,tok)
 args=TrainingArguments(output_dir="outputs/trainer",num_train_epochs=a.epochs,per_device_train_batch_size=a.batch_size,gradient_accumulation_steps=a.grad_accum,learning_rate=a.lr,logging_steps=10,save_strategy="no",report_to="none",fp16=torch.cuda.is_available(),gradient_checkpointing=True,optim="adamw_torch",weight_decay=0.01)
 Trainer(model=model,args=args,train_dataset=ds,data_collator=Pad(tok.pad_token_id)).train()
 Path(a.out).mkdir(parents=True,exist_ok=True); model.save_pretrained(a.out); tok.save_pretrained(a.out)
 print(f"Adapter saved to {a.out}. Trainable adapter parameters are printed above.")
if __name__=="__main__": main()
