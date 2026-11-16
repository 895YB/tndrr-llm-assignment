"""Approximate 7B weight, gradient, optimizer, and adapter memory; GB are decimal."""
PARAMS=7_000_000_000
def gb(bytes_): return bytes_/1e9
# Weight storage only, independent of GPU/runtime.
for label,bpp in [("FP32",4),("FP16/BF16",2),("INT8",1),("4-bit",0.5)]:
 print(f"{label:12s} weights={gb(PARAMS*bpp):5.2f} GB")
# Illustrative common Adam full fine-tuning estimate: fp16 weights + fp16 grads +
# fp32 master weights + two fp32 Adam moments. Activations/workspace not included.
full=PARAMS*(2+2+4+4+4)
print(f"Full FT illustrative: {gb(full):.2f} GB before activations/workspace")
# LoRA: frozen fp16 base; trainable adapter assumed 1% of base (must refine with actual rank/config).
trainable=PARAMS*0.01
lora=PARAMS*2+trainable*(2+2+4+4+4)
print(f"LoRA illustrative (fp16 base, 1% trainable): {gb(lora):.2f} GB before activations/workspace")
# QLoRA: frozen 4-bit base plus 1% trainable adapter with the same illustrative accounting.
qlora=PARAMS*0.5+trainable*(2+2+4+4+4)
print(f"QLoRA illustrative (4-bit base, 1% trainable): {gb(qlora):.2f} GB before activations/workspace")
print("These are estimates, not peak allocated memory. Quantization metadata, padding, activations, optimizer choice, and fragmentation change real usage.")
