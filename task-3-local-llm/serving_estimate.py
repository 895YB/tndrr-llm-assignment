"""Idealized inference memory estimator (not a load test)."""
import argparse
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--users",type=int,default=20); p.add_argument("--context",type=int,default=2048); p.add_argument("--new-tokens",type=int,default=512); p.add_argument("--gpu-gb",type=float,default=24)
 p.add_argument("--params-b",type=float,default=0.5); p.add_argument("--kv-layers",type=int,default=24); p.add_argument("--kv-heads",type=int,default=2); p.add_argument("--head-dim",type=int,default=64); p.add_argument("--kv-bytes",type=float,default=2)
 a=p.parse_args()
 seq=a.context+a.new_tokens
 kv=2*a.kv_layers*seq*a.kv_heads*a.head_dim*a.kv_bytes*a.users/1e9
 # 4-bit theoretical weights plus ~20% allowance for scales/metadata; practical allocations vary.
 weights=a.params_b*1e9*0.5*1.2/1e9
 runtime_buffer=4.0
 total=kv+weights+runtime_buffer
 print("Assumptions: 4-bit model weights with 20% quantization metadata allowance; FP16 KV cache;")
 print(f"users={a.users}, context={a.context}, generated_tokens={a.new_tokens}, sequence={seq}")
 print(f"KV cache estimate: {kv:.2f} GB")
 print(f"4-bit weights estimate: {weights:.2f} GB for {a.params_b:.2f}B parameters")
 print(f"Runtime / fragmentation planning buffer: {runtime_buffer:.2f} GB")
 print(f"Estimated planning total: {total:.2f} GB of {a.gpu_gb:.2f} GB GPU memory")
 print("Memory-only headroom:",f"{a.gpu_gb-total:.2f} GB" if total<=a.gpu_gb else f"over budget by {total-a.gpu_gb:.2f} GB")
 print("This is not a capacity guarantee: benchmark prefill/decode throughput, batching, peak memory and p95 latency with a representative 20-client load test.")
if __name__=="__main__": main()
