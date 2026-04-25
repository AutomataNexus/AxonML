#!/usr/bin/env bash
# Colab A100 80GB entry script — Trident-Coder 1B from-scratch training
# on the user's personal corpus.
#
# Drive layout (G:\My Drive\trident-personal\ ↔ /content/drive/MyDrive/trident-personal/):
#   personal-trident.bin       97.6 MB · 25.58 M tokens, trident-coder-bpe pre-tokenized
#   tokenizer.json             2.2 MB · 32 k vocab byte-level BPE
#   go.sh                      this file
#   train.log                  appended foreground tee output
#   ckpts/                     `--checkpoint-every-steps 1000` rotating + final
#
# Idempotent: each phase guards on a sentinel so re-runs after a Colab VM
# recycle skip already-completed phases (rustup, repo clone, PTX regen,
# cargo build, dataset copy). Re-runs always pick up where they left off.
#
# One-liner Colab cell:
#   !bash /content/drive/MyDrive/trident-personal/go.sh
#
# Lessons baked in (LESSONS.md L112-L116, RDT distill 2026-04-23 session):
#   L112  axonml-llm Cargo.toml uses relative `../crates/*` paths now
#   L113  llm-training is a standalone workspace; target/ is at
#         llm-training/target/, NOT repo root
#   L114  committed .ptx targets sm_89 (RTX 5070 Ti Laptop). A100 is sm_80
#         and rejects sm_89 PTX → CUDA_ERROR_INVALID_PTX. We regenerate
#         every kernel for sm_80 before `cargo build`.
#   L115  Google Drive FUSE is unreliable for multi-MB sequential reads
#         during long-running processes. Copy the dataset to local NVMe
#         (/content/datasets/) before training.
#   L116  RDT-mid OOM signature was activation-stack sized — Trident 1B
#         is bs=4 seq=4096, similar shape. Keep step-level checkpoints
#         (`--checkpoint-every-steps 1000`) so we never lose >1000 steps
#         of progress to a Colab VM recycle.

set -euo pipefail

DRIVE=/content/drive/MyDrive/trident-personal
LOCAL_REPO=/content/AxonML
LOCAL_DATA=/content/datasets
COMMIT="${COMMIT:-HEAD}"   # override with COMMIT=<sha> bash go.sh

mkdir -p "$DRIVE/ckpts"
mkdir -p "$LOCAL_DATA"

# ---------------------------------------------------------------------------
# 1. Rust toolchain
# ---------------------------------------------------------------------------
if [ ! -f "$HOME/.cargo/bin/cargo" ]; then
  echo "[go.sh] Installing rustup..."
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain stable --profile minimal
fi
. "$HOME/.cargo/env"

# ---------------------------------------------------------------------------
# 2. Clone repo at $COMMIT
# ---------------------------------------------------------------------------
if [ ! -d "$LOCAL_REPO/.git" ]; then
  echo "[go.sh] Cloning AutomataNexus/AxonML to $LOCAL_REPO ..."
  git clone --depth 200 https://github.com/AutomataNexus/AxonML.git "$LOCAL_REPO"
fi
cd "$LOCAL_REPO"
git fetch origin --depth 200
git checkout "$COMMIT"
echo "[go.sh] Repo at $(git rev-parse --short HEAD) on $(git rev-parse --abbrev-ref HEAD)"

# ---------------------------------------------------------------------------
# 3. Regenerate sm_80 PTX (LESSONS L114)
# ---------------------------------------------------------------------------
KERNEL_DIR="$LOCAL_REPO/crates/axonml-core/src/backends/cuda_kernels"
SENTINEL_PTX="$LOCAL_REPO/.sm80_ptx_done"
if [ ! -f "$SENTINEL_PTX" ]; then
  echo "[go.sh] Regenerating PTX kernels for sm_80 (A100)..."
  cd "$KERNEL_DIR"
  for cu in *.cu; do
    ptx="${cu%.cu}.ptx"
    echo "         $cu → $ptx"
    nvcc -ptx -arch=sm_80 --use_fast_math "$cu" -o "$ptx"
  done
  cd "$LOCAL_REPO"
  touch "$SENTINEL_PTX"
fi

# ---------------------------------------------------------------------------
# 4. Build the trainer (release, --features cuda)
# ---------------------------------------------------------------------------
TRAIN_BIN="$LOCAL_REPO/llm-training/target/release/train_trident_code"
EXPORT_BIN="$LOCAL_REPO/llm-training/target/release/export_trident_gguf"
CTL_BIN="$LOCAL_REPO/llm-training/target/release/train_ctl"
if [ ! -x "$TRAIN_BIN" ] || [ ! -x "$EXPORT_BIN" ] || [ ! -x "$CTL_BIN" ]; then
  echo "[go.sh] Building train_trident_code + export_trident_gguf + train_ctl ..."
  cd "$LOCAL_REPO/llm-training"
  cargo build --release --features cuda \
    --bin train_trident_code \
    --bin export_trident_gguf \
    --bin train_ctl
fi

# ---------------------------------------------------------------------------
# 5. Copy dataset to local NVMe (LESSONS L115)
# ---------------------------------------------------------------------------
DATASET="$LOCAL_DATA/personal-trident.bin"
if [ ! -f "$DATASET" ]; then
  echo "[go.sh] Copying personal-trident.bin from Drive to /content/ (LESSONS L115)..."
  cp "$DRIVE/personal-trident.bin" "$DATASET"
fi
ls -la "$DATASET"

# Tokenizer — keep on Drive; loaded once at training start, no hot-path I/O.
TOKENIZER="$DRIVE/tokenizer.json"
if [ ! -f "$TOKENIZER" ]; then
  echo "[go.sh] WARN: tokenizer.json missing from Drive."
  echo "        Copy it once from /opt/AxonML/tokenizers/trident-coder-bpe/tokenizer.json"
  echo "        before the first run; otherwise --tokenizer falls back to the in-repo path."
  TOKENIZER="$LOCAL_REPO/tokenizers/trident-coder-bpe/tokenizer.json"
fi

# ---------------------------------------------------------------------------
# 6. Train.  Foreground stream + tee — Colab VM recycles don't survive
#    nohup, but they DO survive across step-level checkpoints, so we just
#    keep things visible.
# ---------------------------------------------------------------------------
CFG="${TRIDENT_CFG:-1b}"        # smoke | laptop | 1b | 3b
STEPS="${TRIDENT_STEPS:-100000}"
# A100 80 GB-safe defaults. AxonML autograd retains every intermediate,
# so bs=4 seq=4096 OOMs even at 1.19 B params. Larger VRAM (H100, A100
# 80 GB pair, etc.) can bump these — but bench memory headroom first.
SEQ="${TRIDENT_SEQ:-2048}"
BS="${TRIDENT_BS:-1}"
LR="${TRIDENT_LR:-3e-4}"
WARMUP="${TRIDENT_WARMUP:-1000}"
CKPT_EVERY="${TRIDENT_CKPT_EVERY:-1000}"
KEEP_K="${TRIDENT_KEEP_K:-5}"

echo "[go.sh] Launching trainer:"
echo "         config=$CFG steps=$STEPS seq=$SEQ bs=$BS lr=$LR warmup=$WARMUP"
echo "         ckpt_every=$CKPT_EVERY keep_last=$KEEP_K"
echo "         output → $DRIVE/ckpts"
echo "         log    → $DRIVE/train.log (tee, appended)"

cd "$LOCAL_REPO/llm-training"
"$TRAIN_BIN" \
  --config "$CFG" \
  --tokenizer "$TOKENIZER" \
  --dataset "$DATASET" \
  --out "$DRIVE/ckpts" \
  --seq-len "$SEQ" \
  --batch-size "$BS" \
  --steps "$STEPS" \
  --warmup-steps "$WARMUP" \
  --lr "$LR" \
  --checkpoint-every-steps "$CKPT_EVERY" \
  --keep-last-k "$KEEP_K" \
  --resume latest \
  2>&1 | tee -a "$DRIVE/train.log"
