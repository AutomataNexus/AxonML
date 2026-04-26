# Trident-Coder personal-corpus training — Colab A100 kit

End-to-end runbook for the from-scratch 1B Trident-Coder training run on the
user's personal corpus (`personal-trident.bin`, 25.58 M tokens, trident-coder-bpe
pre-tokenized).

This is the cloud arm of "Path 1 fully operational" — the laptop arm
(`--config laptop`, ~110 M params, 12 GB VRAM) was verified end-to-end through
`train_trident_code → export_trident_gguf → nexus-serve` in commits
`96957c3` + `4636616`. The 1B run target is the same chain on Colab A100 80GB.

## Drive staging (one-time)

Create `G:\My Drive\trident-personal\` and copy in:

| File | Source on the dev box | Size |
|---|---|---|
| `personal-trident.bin` | `/opt/datasets/personal-trident.bin` | 97.6 MB |
| `tokenizer.json` | `/opt/AxonML/tokenizers/trident-coder-bpe/tokenizer.json` | 2.2 MB |
| `go.sh` | `llm-training/notebooks/trident_personal_colab/go.sh` (this kit) | 6 KB |

(Optional — the cell-by-cell notebook variant of `go.sh` lives at
`trident_personal_colab.ipynb` once added.)

## Colab cell — one-liner

```python
!bash /content/drive/MyDrive/trident-personal/go.sh
```

`go.sh` is idempotent. Re-running after a VM recycle skips:

- rustup install (if `~/.cargo/bin/cargo` exists)
- repo clone (if `/content/AxonML/.git` exists)
- sm_80 PTX regen (if `/content/AxonML/.sm80_ptx_done` exists)
- cargo build (if the three target binaries already exist)
- dataset copy (if `/content/datasets/personal-trident.bin` exists)

Training resumes from the latest checkpoint on Drive (`--resume latest`) so
the only thing a VM recycle costs is the time since the last
`--checkpoint-every-steps` flush.

## Override knobs (env vars)

```bash
COMMIT=89c1230               # pin a specific axonml commit (default: HEAD)
TRIDENT_CFG=1b               # smoke | laptop | 1b | 3b (default: 1b)
TRIDENT_STEPS=100000         # total optimizer steps
TRIDENT_SEQ=1024             # context length (A100-safe default)
TRIDENT_BS=1                 # micro-batch (A100-safe default)
TRIDENT_LR=3e-4              # peak LR
TRIDENT_WARMUP=1000          # linear warmup steps
TRIDENT_CKPT_EVERY=1000      # rotating step-level checkpoint cadence
TRIDENT_KEEP_K=5             # keep-last-K rotating window
```

`go.sh` reads these from the environment and passes them to `train_trident_code`.

## What the run produces

Streams to `G:\My Drive\trident-personal\`:

- `train.log` — appended `tee` of every line the trainer prints
- `ckpts/checkpoint_step_<N>.axonml` × keep-last-K rotating
- `ckpts/checkpoint_best.axonml` — lowest validation loss
- `ckpts/checkpoint_final.axonml` — written on clean exit (SIGTERM, lifecycle stop)

Plus the always-on browser monitor at the URL printed by the trainer
(e.g. `http://127.0.0.1:32953`) and a control socket
`/tmp/axonml-train-<pid>.sock`. From a separate Colab cell:

```python
!{REPO}/llm-training/target/release/train_ctl status
!{REPO}/llm-training/target/release/train_ctl pause
!{REPO}/llm-training/target/release/train_ctl checkpoint   # ad-hoc save
!{REPO}/llm-training/target/release/train_ctl resume
!{REPO}/llm-training/target/release/train_ctl stop         # clean shutdown
```

## After training

```bash
# On the dev box, after pulling checkpoint_final.axonml back from Drive:
export_trident_gguf \
    --config 1b \
    --checkpoint /opt/AxonML/llm-training/checkpoints/trident-1b/checkpoint_final.axonml \
    --out      /mnt/d/AxonML\ Models/trident-1b/trident-personal-1b.gguf \
    --tokenizer /opt/AxonML/tokenizers/trident-coder-bpe/tokenizer.json \
    --vocab-size 32000 \
    --name trident-personal-1b
```

Then load via `nexus-serve --model trident-personal-1b.gguf --port 11436 --quantized`.
nexus-serve loads it as `architecture: bitnet-b1.58` via the existing I2_S
dispatch path; expected decode throughput is in the BitNet-2B family
(~50 t/s on the 5070 Ti Laptop, scaling per the kernel bandwidth analysis
in `project_perf_push_2026_04_18.md`).

## Hyperparameter rationale (default 1B run)

| Knob | Default | Reason |
|---|---:|---|
| `--steps` | 100 000 | 4 epochs over 25.58 M tokens at bs=4 seq=4096 ≈ 4 × 25.58 M / (4 × 4096) ≈ 6 250 steps × 16 grad-accum-equivalent ≈ 100 k |
| `--seq-len` | 1024 | A100 80 GB-safe. seq scales activations quadratically via `[bs,heads,seq,seq]` scores — 1024 → 1.5 GB scores total; 2048 → 6 GB. Code training works fine at 1024. |
| `--batch-size` | 1 | A100 80 GB headroom at the new default: shadow ≈ 4 GB + Adam moments ≈ 8 GB + scores 1.5 GB + saved-input now CPU-staged + activations ≈ 4 GB ⇒ ~50 GB free for scratch/cuBLAS. |
| `--lr` | 3e-4 | Canonical for 1B-class from-scratch with cosine + warmup (matches the published Unsloth recipe used by `ORACLE_LORA_FINETUNE.md`) |
| `--warmup-steps` | 1 000 | 1 % of total — standard for 100 k step runs |
| `--checkpoint-every-steps` | 1 000 | Colab VM recycles can hit at any point; loses at most 1 000 × ~3-5 s/step = <90 min of progress |
| `--keep-last-k` | 5 | 5 × 1 GB ≈ 5 GB on Drive (Drive quota: typically 15-100 GB free) |

## See also

- `/opt/AxonML/llm-training/notebooks/rdt_distill_oracle_colab.ipynb` — the
  prior-art Colab notebook for the RDT distill kick (LESSONS L112-L116
  came out of that session).
- `project_trident_personal_2026_04_25.md` (auto-memory) — current
  workstream state and resume-next-session pointers.
- `project_q1_0_bonsai_2026_04_24.md` — the Bonsai Q1_0 inference work
  that motivated treating Trident as our single personal model.
