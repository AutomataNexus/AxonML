# Draft-Model Distillation for Speculative Decoding

**Goal**: train a small draft model that, paired with DeepSeek-R1-Distill-Qwen-7B as the target via speculative decoding in nexus-serve, reaches **≥80% acceptance rate** at γ=3 — the threshold at which spec decoding actually beats baseline A (forward_one production path) on our hardware.

**Why**: `spec_bench` on 2026-04-18 showed the 1.5B Qwen2.5 draft gives only 50.8% acceptance at γ=3, so spec tops out at ~9.5 tok/s versus baseline's ~37 tok/s. Per-round cost = draft(~85ms for γ=3) + verify(~30ms with GPU-native verify). Emits avg 2.02 tokens at 50.8%, so ≈17 tok/s ceiling even with a 2× verify speedup. To beat baseline we need either a faster draft OR higher acceptance — training a smaller, tightly-aligned draft hits both.

## Acceptance rate → throughput math

Per spec round:
- Draft cost = γ × T_draft
- Verify cost = T_verify (one batched forward over γ target tokens)
- Emitted tokens = 1 + α × (γ − 1) where α = acceptance probability

Throughput = (1 + α × (γ − 1)) / (γ × T_draft + T_verify)

For DeepSeek-7B target, T_verify ~30ms post-f3ecf87 (GPU-native prefill). For draft at Qwen3-0.6B speed (110 tok/s) = 9ms/token.

| draft | γ | α | emitted | round cost | tok/s |
|---|---|---|---|---|---|
| 1.5B current | 3 | 0.508 | 2.02 | 85 + 30 = 115ms | 17.5 |
| 0.6B current | 3 | 0.508 | 2.02 | 27 + 30 = 57ms | 35.4 |
| 0.6B distilled | 3 | 0.80 | 2.60 | 27 + 30 = 57ms | 45.6 |
| 0.6B distilled | 4 | 0.75 | 3.25 | 36 + 30 = 66ms | 49.2 |
| 0.6B distilled | 5 | 0.70 | 3.80 | 45 + 30 = 75ms | 50.7 |

A 0.6B-sized draft with 75-80% acceptance projects to **45-50 tok/s** — comfortably above the 37 tok/s baseline.

## Architecture choice

**Student**: Qwen3-0.6B architecture (qwen3 arch dispatch, includes QK-norm).

Why Qwen3-0.6B and not Qwen2.5-0.5B:
- Matches target's tokenizer (both 152064-token Qwen BPE)
- Already runs at ~110 tok/s in nexus-serve (fastest small model we have)
- Qwen3 QK-norm layers are already wired in nexus-serve for decode

**Target** (frozen teacher): `DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf`. Must dequantize on-the-fly during training — either (a) decompress to f16 in memory once at start (~9GB, fits in 12GB VRAM if student is on CPU), or (b) teacher-inference with nexus-serve as a side service and feed logits to the Rust trainer via IPC.

Option (a) is simpler and faster — go with that.

## Dataset

**Tokens needed**: ~500M-1B for a meaningful distillation run. Rule of thumb: 10-20× the student parameter count.

**Source**: Web text. Specifically:
- FineWeb (open, high-quality web corpus) — 15TB, use a 500M-1B slice
- The Stack v2 (code) — optional supplement for technical reasoning
- Prompts from the nexus-agent task distribution (domain alignment)

**Pipeline**:
1. Download a 2-4GB slice of FineWeb
2. Tokenize with Qwen BPE → flat uint32 token stream on disk
3. Pre-shuffle into 2048-token sequences

Dataset at `/opt/datasets/fineweb-qwen/tokens.bin`.

## Loss function

Two-term loss at each token position:

```
L = α × CE(student, ground_truth_token) + (1 − α) × KL(teacher || student)
```

- α = 0.1 (90% distillation, 10% classical next-token)
- KL divergence over the full 152064-vocab distribution

For KL: teacher produces logits → softmax with temperature T (typically 2-4) → distribution P. Student logits → softmax T → distribution Q. Loss = Σ P · (log P − log Q).

Temperature T is a key hyperparameter — T=1 is hard labels, higher T smooths.

## Training hyperparameters

- **Optimizer**: AdamW, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1
- **LR schedule**: linear warmup (500 steps) → cosine decay to 10% of peak
- **Batch**: 4 × 2048-token sequences (effective 8192 tokens/step)
- **Steps**: ~50k for 500M tokens @ 10k tok/step
- **Grad clip**: 1.0
- **Mixed precision**: bf16 if framework supports, else fp32 (AxonML currently fp32-only)
- **Checkpoint every**: 1000 steps

Estimated wall time on RTX 5070 Ti Laptop: ~24-48h for 50k steps (depends on teacher forward-pass throughput).

## Evaluation

After every 5000 steps, snapshot student weights and run:

```bash
/opt/AxonML/nexus-serve/target/release/spec_bench \
  --target /opt/AxonML/models/deepseek-r1-distill-qwen-7b/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf \
  --draft  /opt/AxonML/checkpoints/draft_distill/step_N.gguf \
  --prompt "Write a paragraph about the history of computing." \
  -n 128 -g 3 -q
```

Report acceptance rate and tok/s per γ ∈ {2, 3, 4, 5}. Early stopping when α plateaus above 0.75 for γ=3.

## Export to GGUF for spec_bench

The training framework writes AxonML-native checkpoints (`.axonml`). For `spec_bench` to load the draft, it needs GGUF. Conversion:

1. Reload trained checkpoint's state dict
2. Repack tensor weights into GGUF layout (Q4_K or f16 — start with f16 for correctness, quantize to Q4_K after)
3. Write GGUF with Qwen3 architecture metadata (`general.architecture = "qwen3"`)
4. Validate by loading in nexus-serve and running a quick completion

A conversion utility `tools/axonml_to_gguf.rs` will live in `llm-training/tools/`.

## Scope & estimate

| Component | Effort | Notes |
|---|---|---|
| Qwen3 architecture in axonml-llm (training-friendly) | 1-2 days | Adapt from `llama.rs` + add QK-norm |
| Dataset prep pipeline | 1 day | Download + tokenize + shuffle |
| Teacher logit generation (via frozen target) | 1 day | Either inline f16 dequant or IPC to nexus-serve |
| KL+CE loss head + trainer loop | 1 day | Mostly boilerplate on llm-training's scaffolding |
| GGUF export utility | 0.5 day | Pack + metadata |
| Training run | 1-2 days | Wall-clock during actual training |
| Evaluation + iteration | 0.5-1 day | Tune T, α, γ on spec_bench results |
| **Total** | **~7-10 days** | Plus wall-clock training time |

## Starting point — where to resume

1. `llm-training/src/bin/train_draft_distill.rs` (stub) — CLI + config struct
2. Clone `llm-training/src/bin/train_llama.rs` as the starting template
3. Add Qwen3 architecture module to `axonml-llm` (likely `qwen3.rs`)
4. Implement KL-divergence loss in `axonml-nn/src/loss.rs` (may already exist — check)
5. Wire target-logit generator: load DeepSeek-7B as `InferenceEngine` with dequant to f16, run `forward_one` per token, return logits tensor

## Related

- `spec_bench.rs` at `/opt/AxonML/nexus-serve/src/bin/` — the acceptance-rate measurement tool
- `/home/devops/.claude/projects/-opt-AxonML/memory/project_arch_expansion_2026_04_17.md` — session context
- `/home/devops/.claude/projects/-opt-AxonML/memory/project_v0_6_2_release_2026_04_17.md` — original spec_bench negative-result memo
