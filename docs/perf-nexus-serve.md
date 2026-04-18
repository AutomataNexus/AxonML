# nexus-serve decode performance — 2026-04-18

This is a living scoreboard of per-model decode tok/s on the reference dev box
(RTX 5070 Ti Laptop, Blackwell, 12 GB GDDR7; WSL2 Ubuntu; single-query decode,
temperature 0.0, 64 tokens generated from prompt `"The capital of France is"`).

## Current scoreboard

| Model | Quant | tok/s | Notes |
|---|---|---|---|
| **BitNet-b1.58-2B-4T** | I2_S (1.58-bit) | **50** | GPU kernel landed 2026-04-18 (7× over CPU) |
| Qwen3-0.6B | Q4_K_M | ~124 | Landed 2026-04-17 (arch expansion session) |
| OLMoE 1B-7B | Q4_K_M | 103 | Fused gate+up+SwiGLU + scaled-add (2026-04-17/18) |
| Mamba-130M | Q4_K_M | 101 | SSM, GPU-resident selective scan |
| Qwen3-1.7B | Q4_K_M | ~86 | 2026-04-17 |
| Qwen3-4B-Thinking | Q4_K_M | ~52 | 2026-04-17 |
| Phi-3-mini 4k-instruct | Q4_K_M | 40 | Fused Q5_K QKV 2026-04-18 |
| DeepSeek-R1-Distill-Qwen-7B | Q4_K_M | 35.5 | 2026-04-17/18 |
| Llama-3.2-3B-instruct | Q4_K_M | ~66 | 2026-04-17 |
| Falcon-7B (legacy, parallel-attn) | Q4_K_M | 22 | Q8_0 LM-head GPU + fused Q5_1 QKV 2026-04-18 |

## What moved 2026-04-18

Five commits this push:

| Commit | Change | Delta |
|---|---|---|
| `21178bc` | Fused Q5_K QKV matmul (`q5k_gemv_fused_qkv_f32`) | Phi-3 29.5 → 40.2 |
| `e6e4c31` | Q8_0 GEMV kernel (`q8_0_gemv_f32`) | Falcon LM head: CPU → GPU, 10.3 → 21.5 |
| `29bea9c` | Fused Q5_1 QKV (`q5_1_gemv_fused_qkv_f32`) | Falcon 21.5 → 22.0 |
| `e2de677` | **BitNet I2_S GPU kernel** (`i2s_gemv_f32`) | **BitNet 7 → 50 (7.1×)** |
| (reverted) | Q6_K v2 two-warp — regressed Phi-3 and Falcon, left at v1 | — |

## Kernel inventory (decode GEMV paths)

| Quant | Kernel | Block (weights/bytes) | Pattern | Consumers |
|---|---|---|---|---|
| Q4_K | `q4k_gemv_f32` v2 | 256/144 | 2 warps/row, uint32 qs + float4 act | Most models |
| Q5_K | `q5k_gemv_f32` v2 | 256/176 | 2 warps/row | Phi-3 attn_qkv |
| Q5_0 | `q5_0_gemv_f32` v2 | 32/22 | 2 warps/row | Falcon attn_output, ffn_up |
| Q5_1 | `q5_1_gemv_f32` v2 | 32/24 | 2 warps/row | Falcon attn_qkv (split) |
| Q6_K | `q6k_gemv_f32` v1 | 256/210 | 1 warp/row — **do not upgrade to v2** | LM head on most models, ffn_down on Falcon/Phi-3 |
| Q8_0 | `q8_0_gemv_f32` v2 | 32/34 | 2 warps/row | Falcon LM head |
| I2_S | `i2s_gemv_f32` v2 | 128/32 | 2 warps/row, ternary decode | BitNet-2B bodies |

## Fused QKV kernels

For architectures that split the GGUF `attn_qkv` tensor into Q/K/V Weights at
load time, a fused QKV kernel re-merges them into one launch:

| Kernel | Architectures that use it |
|---|---|
| `fused_qkv_q4k_matmul_gpu` | DeepSeek-R1-7B, Qwen2/Qwen3 family (Q4_K attn_qkv) |
| `fused_qkv_bias_q4k_matmul_gpu` | Qwen2 with Q/K/V biases |
| `fused_qkv_q5k_matmul_gpu` | Phi-3-mini (Q5_K attn_qkv) |
| `fused_qkv_q5_1_matmul_gpu` | Falcon-7B (Q5_1 attn_qkv, MQA n_kv_heads=1) |

Pattern: one grid of `ceil((q_out + k_out + v_out) / ROWS_PER_CTA)`; each warp's
`global_row` maps into Q, K, or V by threshold. Each function requires all three
Weights to be the same quant type — they fall back to three separate `matmul`
calls otherwise.

## Non-levers (do not attempt)

- **`__ldg` on Q6_K weight reads**: regresses ~3 tok/s. Texture cache not a win
  for the already-coalesced Q6_K access pattern.
- **Q6_K v2 (two warps per row)**: regresses ~1.5 tok/s on Phi-3, ~0.6 on
  Falcon. Q6_K v1 already fills SM occupancy at realistic shapes (LM head 32k+
  rows, ffn_down 4544+). The __syncthreads + shared-mem combine is pure
  overhead.
- **Bulk `pool_alloc_uninit` → `pool_alloc` sweep**: crashed WSL on 2026-04-17.
  Change pool allocation call sites one at a time with a bench in between. See
  memory feedback `feedback_no_bulk_memory_sweeps.md`.

## Bandwidth ceiling

RTX 5070 Ti Laptop GDDR7 peak: ~450 GB/s. Current utilization (by model):

| Model | Weights (Q4_K or I2_S) | tok/s × weights | Effective BW | % of peak |
|---|---|---|---|---|
| BitNet-2B | 500 MB | 50 × 0.5 = 25 | 25 GB/s | 6% |
| Phi-3-mini | ~2.2 GB | 40 × 2.2 = 88 | 88 GB/s | 20% |
| Falcon-7B | ~4.1 GB | 22 × 4.1 = 90 | 90 GB/s | 20% |
| DeepSeek-7B | ~4.4 GB | 35.5 × 4.4 = 156 | 156 GB/s | 35% |
| Qwen3-0.6B | ~0.4 GB | 124 × 0.4 = 50 | 50 GB/s | 11% |

Theoretical 65% gain possible across the board if we hit 90% of peak. Realistic
paths to get there:

1. **Fused norm+matmul kernels** — eliminate one global-memory roundtrip per
   matmul per layer. Expected 5-10%.
2. **Multi-block unroll on thin kernels** (I2_S, Q5_0, Q5_1) — 1.5-2× on
   memory-underutilized blocks.
3. **CUDA streams for parallel architectures** (Falcon attn ‖ FFN) — 1.3-1.5×
   Falcon. Blocked by `CudaStream: !Send+!Sync` backend constraint.
4. **TC-f16 path for prefill** (m ≥ 32 only) — 2-3× prefill speedup. Not a
   decode win (f16 weights are 3.6× bigger than Q4_K).

## How to reproduce

```bash
cd /opt/AxonML
cargo build --release --features cuda

# Per-model launch template (change --model and --port):
nohup target/release/nexus-serve \
  --model /opt/AxonML/models/bitnet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  --port 11440 --quantized > /tmp/nex.log 2>&1 &

# Wait ready, then 3-run warm bench:
until curl -s http://127.0.0.1:11440/health | grep -q ok; do sleep 1; done
for i in 1 2 3; do
  T0=$(date +%s.%N)
  curl -s -X POST http://127.0.0.1:11440/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"prompt":"The capital of France is","max_tokens":64,"temperature":0.0}' \
    > /tmp/r$i.json
  T1=$(date +%s.%N)
  python3 -c "import json; d=json.load(open('/tmp/r$i.json')); \
    print(f\"run$i: {d['usage']['completion_tokens']/($T1-$T0):.2f} tok/s\")"
done
```

Discard run 1 (cold). Runs 2 and 3 should be within ±1 tok/s of each other.

## Related docs

- Kernel-level architecture: `/opt/AxonML/crates/axonml-core/src/backends/cuda_kernels/*.cu` (each `.cu` has a header block explaining the block layout and kernel design)
- BitNet I2_S format spec: `/opt/AxonML/crates/axonml-quant/src/bitnet.rs` header
- Weight dispatch: `/opt/AxonML/nexus-serve/src/model/weight.rs::Weight::matmul`
- Fused-QKV helpers: `/opt/AxonML/nexus-serve/src/model/weight.rs::fused_qkv_*_matmul_gpu`
