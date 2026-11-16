"""Create deterministic synthetic intent data with a clean held-out split."""
import argparse, json, random
from pathlib import Path

INTENTS = {
 "billing": ["I was charged twice for my subscription", "Why is my invoice higher this month", "Please explain this unexpected payment", "My card was billed after I cancelled", "I need a copy of my receipt"],
 "technical": ["The app crashes when I open settings", "I cannot upload a document", "The dashboard keeps loading forever", "I get an error when I sign in", "Notifications are not appearing"],
 "account": ["I need to change the email on my account", "I forgot my password and cannot log in", "Please help me update my profile", "My account has been locked", "How can I enable two factor authentication"],
 "shipping": ["Where is my order and tracking number", "My package has not arrived yet", "The parcel was delivered to the wrong address", "Can I change the delivery address", "The item arrived damaged in transit"],
 "cancellation": ["Please cancel my current plan", "I want to stop the automatic renewal", "How do I close my subscription", "Cancel the order I placed yesterday", "I no longer need this service"]
}
PREFIX = ["", "Hello, ", "Hi support, ", "Could you help: ", "I have an issue: ", "Please assist, "]
SUFFIX = ["", " Please advise.", " This is urgent.", " Thanks.", " Can you look into it?", " I would appreciate your help."]

def main():
 p=argparse.ArgumentParser(); p.add_argument("--out",default="data"); p.add_argument("--seed",type=int,default=17); a=p.parse_args()
 rng=random.Random(a.seed); rows=[]
 # 5 x 5 base messages x 12 deterministic phrasing variants = 300 records.
 for intent, messages in INTENTS.items():
  for i,msg in enumerate(messages):
   for j in range(12):
    prefix=PREFIX[(i+j)%len(PREFIX)]; suffix=SUFFIX[(i*2+j)%len(SUFFIX)]
    text=(prefix+msg+suffix).strip()
    rows.append({"text":text,"intent":intent,"target":json.dumps({"intent":intent},separators=(",",":"))})
 rng.shuffle(rows)
 # Stratification: take 12 per class for test; remaining 48/class train.
 train=[]; test=[]
 for intent in INTENTS:
  group=[x for x in rows if x["intent"]==intent]; rng.shuffle(group)
  test.extend(group[:12]); train.extend(group[12:])
 rng.shuffle(train); rng.shuffle(test)
 out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
 for name,data in [("train.jsonl",train),("test.jsonl",test)]:
  with (out/name).open("w",encoding="utf-8") as f:
   for row in data: f.write(json.dumps(row,ensure_ascii=False)+"\n")
 print(f"Wrote {len(train)} train and {len(test)} held-out records to {out.resolve()}")
if __name__=="__main__": main()
