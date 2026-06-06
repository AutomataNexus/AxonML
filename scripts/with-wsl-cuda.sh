#!/bin/bash
# with-wsl-cuda.sh — prepend the real WSL libcuda so --features cuda builds/runs don't
# pick up the /opt/cuda-stubs shadow in ldconfig (cuArray3DCreate_v2 etc undefined).
#
# Usage:
#   ./scripts/with-wsl-cuda.sh cargo run --release --features cuda --bin foo
#   ./scripts/with-wsl-cuda.sh cargo test --features cuda --test bar
#   LD_LIBRARY_PATH=/usr/lib/wsl/lib cargo ...   (manual equivalent)
#
# Why: /opt/cuda-stubs (for Hailo DFC / TF) contains a stub libcuda.so.1 that wins
# in the cache on some WSL setups. Real driver lives in /usr/lib/wsl/lib (installed
# by the Windows NVIDIA driver). Without this, you get runtime link errors only on
# the first CUDA call that hits a missing symbol (e.g. cuArray3DCreate_v2 from
# certain tensor/alloc paths).
#
# Safe on non-WSL (the dir just won't exist or have no effect).
# Idempotent (won't duplicate the path).

set -euo pipefail

export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

exec "$@"
