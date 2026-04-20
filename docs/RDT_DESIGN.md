# AxonML RDT — Recurrent-Depth Transformer design

**Task:** #58
**Status:** Blocked on Oracle-7B v2 + Qwen3-0.6B draft landing (need both for evaluation baselines on the same Claude Code corpus).
**Reference:** Geiping et al. 2025, *Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach* (Huginn-3.5B).

---

## 1. Why

Standard transformers do a fixed K depth. At inference you can't trade more compute for more accuracy without either (a) growing parameters or (b) emitting more chain-of-thought text. RDT gives us lever (c): **re-use the same recurrent core block K times at test-time**, increasing effective depth without touching parameters.

For the Oracle use case specifically:

- Easy agentic queries (single `Read` tool_use) → K=4 iterations suffices
- Hard multi-step tasks (chained greps + edits + tests) → K=16 costs more compute but lifts accuracy
- K is user-tunable per-request — exposed as `num_steps` on `/v1/messages`

A small RDT (400M–1B params) with test-time K scaling may approach Oracle-7B quality on hard queries while beating it on easy ones, at a fraction of the weight footprint.

---

## 2. Architecture

### Three-stack layout

```
tokens ──► Embedding ──► Prelude (N_p layers) ──► e
                                                  │
                              ┌───────────────────┘
                              ▼
                       h_0 := e  (seed hidden state)
                              │
                              ▼
   ┌──► Core(h_t + e)   N_c layers, shared across t   ──► Block(h_t, e)
   │                                                       │
   │             h_{t+1} = α·h_t + β·e + Block(h_t, e)     │
   │                           ▲                           │
   └───────────────────────────┴───────────────────────────┘   × K iterations

                      h_K ──► Coda (N_d layers) ──► lm_head ──► logits
```

### Update rule

```
h_{t+1} = α · h_t + β · e + Block(h_t + e)
```

- `Block` is an `N_c`-layer stack of standard transformer decoder layers (re-using `Qwen3DecoderLayer`: RMSNorm + GQA + SwiGLU MLP)
- Input to `Block` is `h_t + e` — simple element-wise sum, no extra projection (matches Huginn's "injected" variant)
- `α`, `β` are **learnable scalar parameters**, initialized to α=0.5, β=0.5 (paper's default)
- The same core weights are re-applied each iteration t ∈ {0, …, K-1}

### Layer counts (small / Oracle-candidate)

| Config | N_p (prelude) | N_c (core) | N_d (coda) | hidden | heads | kv_heads | params |
|---|---|---|---|---|---|---|---|
| rdt-tiny | 2 | 4 | 2 | 1024 | 16 | 8 | ~200M |
| rdt-small | 2 | 6 | 2 | 1536 | 24 | 8 | ~500M |
| rdt-mid | 4 | 8 | 4 | 2048 | 32 | 8 | ~1.2B |

Param budget is mostly in the shared core, not the prelude/coda. Test-time compute scales with K on the core only — prelude/coda run once.

---

## 3. Training

### Sampled K per step

```
K ~ U{k_min, k_max}      # sample uniformly each minibatch
k_min = 4, k_max = 16     # default; tune per model size
```

Sampling ensures the model generalizes across iteration counts. Fixed-K training degrades when K varies at inference.

### Full-unroll backprop

Backprop through all K core iterations. Memory cost scales linearly with K — at k_max=16, 4096 seq, hidden=1536, activation memory ≈ 30 GB per GPU at bs=1 (A100 40GB tight). Mitigations:

- **Gradient checkpointing** on the core (re-compute each iteration's forward during backward) — brings memory to O(1) in K at ~30% throughput cost. Use `axonml-nn::GradientCheckpoint` wrapper.
- **Smaller bs** with higher grad accum.
- Consider **truncated backprop through depth** (only backprop through last k_truncate iterations) as a future optimization if we want k_max=32.

### Loss

Standard cross-entropy on the final coda output. No auxiliary losses in v1.

### Corpus

Same `corpus_train.jsonl` / `corpus_val.jsonl` (1,067 / 57 sessions, combined WSL + extwsl Claude Code traces). DeepSeek-R1 template already baked in. Tokenizer: Qwen2 BPE (shared with the Oracle-7B target for fair A/B).

---

## 4. GGUF format

New architecture ID: `rdt`. Metadata keys:

| Key | Type | Meaning |
|---|---|---|
| `rdt.context_length` | u32 | max seq len |
| `rdt.embedding_length` | u32 | hidden size |
| `rdt.feed_forward_length` | u32 | MLP intermediate |
| `rdt.prelude.block_count` | u32 | N_p |
| `rdt.core.block_count` | u32 | N_c |
| `rdt.coda.block_count` | u32 | N_d |
| `rdt.attention.head_count` | u32 | num Q heads |
| `rdt.attention.head_count_kv` | u32 | num KV heads (GQA) |
| `rdt.recurrent.k_default` | u32 | default K at inference (e.g. 8) |
| `rdt.recurrent.k_min` | u32 | training k_min |
| `rdt.recurrent.k_max` | u32 | training k_max |
| `rdt.recurrent.alpha` | f32 | learned α (scalar) |
| `rdt.recurrent.beta` | f32 | learned β (scalar) |
| `rdt.attention.layer_norm_rms_epsilon` | f32 | RMSNorm ε |
| `rdt.rope.freq_base` | f32 | RoPE θ |

Tensor naming convention (prefix tells the reader where each weight sits):

```
prelude.blk.0.attn_q.weight          # prelude layer 0
prelude.blk.1.ffn_down.weight
core.blk.0.attn_q.weight             # core layer 0 (re-applied K times)
core.blk.{N_c-1}.ffn_down.weight
coda.blk.0.attn_q.weight             # coda layer 0
output_norm.weight                   # final RMSNorm before lm_head
output.weight                        # lm_head (tied with token_embd if flagged)
token_embd.weight
```

---

## 5. Inference in nexus-serve

### Dispatch

`InferenceConfig::architecture = "rdt"` activates a new forward path.

### Forward

```rust
fn forward_rdt(&self, token: u32, pos: usize, num_steps: usize) -> Tensor<f32> {
    let e = self.prelude.forward(self.embed(token));
    let mut h = e.clone();
    for _ in 0..num_steps {
        let block_out = self.core.forward(&h.add(&e));
        h = h.mul_scalar(self.alpha)
              .add(&e.mul_scalar(self.beta))
              .add(&block_out);
    }
    self.coda.forward(&h)
}
```

### KV cache

One KV cache per **core layer**, but since the core runs K times, the cache across iterations can either:

- **Fresh each iteration (v1)** — zero-init KV at start of each new token step, re-build during the K iterations. Simplest, matches the paper. KV is thrown away between tokens but we keep a separate "session KV" for the Prelude and Coda layers.
- **Persistent across iterations (optimization)** — cache the K-th iteration's KV and re-use. Adds complexity; defer.

### Request-level K

New field on `MessagesRequest`: `num_steps: Option<usize>`. Defaults to `rdt.recurrent.k_default` from GGUF metadata. Clamped to `[1, 64]`.

Propagate through `generate_stream` into the per-token forward path.

---

## 6. Files to create / modify

| File | Change |
|---|---|
| `crates/axonml-llm/src/rdt.rs` | NEW — `RDTConfig`, `RDTPrelude`, `RDTCore`, `RDTCoda`, `RDT`, `RDTForCausalLM`, learnable `alpha` / `beta` scalars |
| `crates/axonml-llm/src/lib.rs` | `pub mod rdt;` + re-exports |
| `crates/axonml-llm/src/gguf_export.rs` | `export_rdt_to_gguf` |
| `crates/axonml-llm/src/gguf_loader.rs` | `load_rdt_from_gguf` |
| `llm-training/src/bin/train_rdt.rs` | NEW — training loop with K-sampling + checkpoint + monitor (respect feedback_training_control memory) |
| `llm-training/Cargo.toml` | `[[bin]] train_rdt` |
| `nexus-serve/src/model/gguf.rs` | Recognize `rdt` architecture |
| `nexus-serve/src/model/inference.rs` | `forward_rdt` + dispatch in forward_one |
| `nexus-serve/src/api/messages.rs` | `num_steps` field on `MessagesRequest`, plumb into `generate_stream` |
| `nexus-serve/src/model/inference.rs::stop_tokens()` | Treat as Qwen2 vocab (same family, reuse token IDs) |

---

## 7. Evaluation plan

Against the held-out 57-session val set, measure:

| Metric | How | Why |
|---|---|---|
| Val perplexity at K ∈ {1, 2, 4, 8, 16, 24} | CE over val corpus with fixed K | test-time compute scaling curve |
| Tool-use accuracy at K=8 | parse `<tool_use>{json}</tool_use>` on held-out queries, compare to gold | agentic quality |
| Throughput tok/s at K=8 | `spec_bench` with new rdt backend | wall-clock viability |
| Compare to Oracle-7B at K=1, 8, 16 | side-by-side on same val prompts | does test-time K match static-7B? |

Pass bar: rdt-small (~500M) at K=16 matches Oracle-7B on tool-use accuracy within 5 points, while decoding faster than 25 tok/s.

---

## 8. Upgrade path → OpenMythos RDT (v2/v3)

The landed v1 is the simplified Huginn formulation. OpenMythos (source:
user-supplied spec, 2026-04-20) extends the architecture with six
stability + capacity upgrades. Planned as an incremental migration —
each bullet is landable as its own task on top of the v1 module.

### v2 — Stability + capacity (high-priority)

1. **A and B as learnable matrices** (not scalars).
   `h_{t+1} = A·h_t + B·e + Block(h_t + e)`
   A, B ∈ ℝ^{hidden × hidden}. Dramatically increases expressive capacity
   of the recurrent update. Store as two `nn::Linear` layers without bias,
   init to identity-scaled-by-0.5 to preserve v1 behavior at init.

2. **LTI spectral-radius constraint on A** (`ρ(A) < 1`).
   Paper reference: Parcae architecture (Prairie et al. 2026). Prevents
   residual explosion across deep loops — the dominant training-instability
   failure mode in looped transformers. Enforce by construction via one of:
   - Periodic re-projection: after each optimizer step, compute
     largest-singular-value of A and rescale if ≥ 1.
   - Spectral normalization layer (Miyato et al. 2018 power-iteration
     variant).
   Start with periodic re-projection — simpler, works.

3. **DeepSeekMoE FFN inside the core block** (replaces SwiGLU).
   Fine-grained routed experts (e.g. 64 experts, top-k=6) + shared
   always-active experts (e.g. 2 shared). Router picks different expert
   subsets at each depth — each loop iteration is computationally
   distinct despite shared base weights. Adds domain breadth; looping
   gives reasoning depth.
   Requires: `DeepSeekMoERouter`, expert weight storage that indexes by
   `(layer_id, depth_t)` rather than just `layer_id`.

### v3 — Inference efficiency + adaptive compute

4. **Multi-Latent Attention (MLA)** from DeepSeek-V2 (replaces GQA).
   Caches a compressed low-rank KV latent rather than full K/V tensors →
   10–20× KV memory reduction at production context lengths. Requires
   nexus-serve KV cache rework — latent space is projected down before
   storage, projected back up inside the attention kernel.

5. **Adaptive Computation Time (ACT) halting**.
   Per-position learned scalar halting head. Each loop iteration, the
   head predicts "should I stop here?". Positions that converged early
   exit; hard positions keep looping. Prevents the "overthinking" failure
   mode where excessive K drifts past the solution into noise.
   Requires: per-position halting probabilities, cumulative-halting-sum
   threshold (typically 1-ε), and a ponder cost loss term during training
   so the model doesn't over-halt or under-halt systematically.

6. **Depth-Wise LoRA adapters**.
   Small rank-r (e.g. r=4 or r=8) LoRA-style adapters at each depth step,
   giving each loop iteration slightly distinct behavior without the full
   parameter bloat of fully-distinct per-depth weights. Bridges the gap
   between v1's pure weight-tying (all K iterations identical) and the
   depth-indexed DeepSeekMoE routing from v2's bullet 3.
   Storage: K × r × hidden × 2 additional params (A, B low-rank per depth).

### Remaining open questions (unchanged from v1 scope)

- **Shared vs per-iteration LayerNorm**: Huginn uses one shared norm
  across iterations. Paper Appendix B suggests per-iteration norms yield
  slightly better ppl but double the norm-param count. v1: shared.
- **Gradient checkpointing granularity**: per-iteration (outer loop) vs
  per-core-layer (inner). v1: per-iteration — simpler, works in unsloth
  wrapper. v3 candidate: inner for memory-constrained training at large K.

---

## 8a. Production Oracle path — distillation from Oracle-7B

The baseline `train_rdt` binary (landed, task #58) does from-scratch
next-token training. Good for pipeline validation but insufficient for a
production Oracle — 18 M tokens of Claude Code traces against an
rdt-mid (~1.2 B params) from scratch underfits the model capacity.

Production flow uses **knowledge distillation from the already-trained
Oracle-7B teacher** onto the rdt-mid student. Pieces:

1. **New binary `train_rdt_distill`** (task #61) — a fusion of
   `train_rdt`'s K-sampling core-iteration loop with
   `train_draft_distill`'s CE+KL dual-loss head:

   ```text
   L = α · CE(student_logits_at_K, next_token_labels)
     + (1 − α) · KL(student_logits_at_K, teacher_logits, T)
   ```

   Teacher (Oracle-7B) loaded frozen from its GGUF via
   `load_qwen3_from_gguf`. Student (rdt-mid) sampled at K each batch.
   Teacher produces logits once per batch (no K dimension since target
   has fixed depth); student's K-th iteration logits are matched
   against it. Same α=0.1, T=3 defaults as the non-RDT draft distill
   trainer.

2. **Colab A100 runbook** — memory budget:

   | Component | bf16 | GB |
   |---|---|---|
   | Oracle-7B teacher | fp16 weights | 14 |
   | rdt-mid student weights | bf16 | 2.4 |
   | AdamW optimizer state (β1, β2, m, v) | fp32 shadow | 9.6 |
   | Activations at K=16, seq=1024, bs=4 | bf16 | 5 |
   | **Total** | | **~31 GB** — fits A100 40 GB with 9 GB headroom |

   Training time: ~4–6 hours for 5 epochs on 18 M tokens. Using 3 K-values
   per step (mixed micro-batch) instead of single K reduces K-distribution
   mismatch without tripling wallclock.

3. **Weight export**: run `export_rdt_to_gguf` on the trained student to
   produce `oracle-rdt-mid-q4km.gguf` (quantized via llama.cpp's
   `llama-quantize Q4_K_M` after an fp16 export — same as the existing
   Oracle pipeline).

4. **nexus-serve wiring**: task #60 must land before the production
   rdt-mid can actually serve. `num_steps` request param routes to the
   runtime K value; default = `rdt.recurrent.k_default` from GGUF
   metadata.

5. **Deploy to NexusOracle**: replace `oracle-r1-distill-q4km.gguf` in
   the `claude_local_base_url` pointer target with the rdt-mid GGUF.
   The Oracle app's reasoning toggle (task #53, landed) still works;
   the new model swaps in at the nexus-serve layer.

### Why distillation beats from-scratch for this corpus

- 18 M tokens is ~2 orders of magnitude too small for from-scratch
  pretraining of a 1.2 B-param model (typical Qwen2.5-1.5B pretraining
  budget is ~2 T tokens).
- Oracle-7B already captures the DeepSeek-R1-Distill reasoning style and
  the tool-use convention from our LoRA fine-tune. Distilling those
  behaviors transfers capability density per token much more efficiently
  than ground-truth CE alone.
- Test-time K at inference then gives us the *additional* compute lever
  the fixed-depth Oracle-7B doesn't have — that's the Oracle→RDT upgrade
  value.

## 9. Ordering / gating

1. ⏳ Oracle-7B v2 Colab run finishes → baseline GGUF at `/opt/AxonML/models/oracle-distill/oracle-r1-distill-q4km.gguf`
2. ⏳ Qwen3-0.6B draft Colab run finishes → draft GGUF at same dir
3. ✅ Task #57 (no-cuda build clean) landed — unblocks stable CI
4. **Then** kick off Task #58 in this order: `rdt.rs` module → unit-test the K-iteration forward numerically (single-batch, hand-computed α·h + β·e + block) → `train_rdt` binary → smoke train a `rdt-tiny` on the Claude Code corpus → GGUF export → nexus-serve inference → full eval matrix.

Estimated walltime for the series: 4–5 working days. First trainable model ~day 2.

---

## Author

Andrew Jewell Sr. — AutomataNexus LLC
ORCID: 0009-0005-2158-7060
Created: 2026-04-20
