# axonml-subbit

Sub-bit **vector-quantized (VQ)** GPU primitives for AxonML: the fused sub-bit
matmul kernels, grouped MoE experts, device-side router top-k, resident index
assignment, offload tiles, and the combine/cat/gather device ops. Everything
here runs a projection straight off packed codebook indices, so a model can be
served or trained from a sub-bit artifact without reconstructing dense weights.

## How it relates to the core crates

`axonml-subbit` depends on `axonml-tensor` and `axonml-core` through their
**public APIs only** — `get_cuda_backend()`, `CudaBackend::{context, stream}`,
`Tensor::{as_cuda_slice_read, as_cuda_slice_write, from_storage}`,
`pool_alloc_uninit`, `Storage::from_cuda_slice`. It loads its **own** CUDA
modules (`kernels/*.ptx`) via cudarc, so the core crates carry none of the
kernels or wrappers.

The core crates are extended through extension traits rather than by editing
their source:

| Item | Kind | Purpose |
|------|------|---------|
| `SubbitTensorExt` (for `Tensor<f32>`) | trait | `router_topk`, `vq_moe_combine[_dev]`, `vq_fused_fwd[_resident]`, `vq_fused_bwd`, `vq_gather_denorm_[cuda\|resident]`, `vq_normalize_cuda`, `cat_gpu`, `dtod_write_at` |
| `SubbitBackendExt` (for `CudaBackend`) | trait | the raw kernel launch wrappers (`vq_fused_matmul_*`, `vq_router_topk_f32`, `vq_moe_combine_f32`, `vq_assign_f32`, …) |
| `SubbitEmbeddingExt` (for `Tensor<f32>`) | trait | `embedding_scatter_add_resident[_accum]` |
| `VqGroupedExperts` | struct | concatenated MoE experts, grouped GEMV (`forward`, `forward_dev`) |
| `RouterSel` | struct | device-resident top-k router selection (indices + weights) |
| `ResidentAssign` | struct | GPU-resident per-tile VQ index+scale, uploaded once |

Bring the ops into scope with `use axonml_subbit::{…, SubbitTensorExt};`.
The crate is empty without the `cuda` feature.

## Switches

| Variable | Effect |
|----------|--------|
| `AXONML_TF32` | run the reconstruct-tile GEMMs on tf32 tensor cores |
| `AXONML_FP4` / `AXONML_FP4_BWD` | NVFP4 forward / backward on Blackwell (sm_120a) |
| `AXONML_VQ_WARP_ROWS` | row-count ceiling for the warp-per-column GEMV |
| `AXONML_NH_MOE_DEDUP_SYNC` | synchronise after each MoE dedup launch to attribute faults |

## Kernels

`kernels/*.cu` are the CUDA sources; `kernels/*.ptx` are the checked-in compiled
modules (clang → `--cuda-gpu-arch=sm_89`, JITs to newer archs). Regenerate with:

```
clang -O3 -x cuda --cuda-device-only --cuda-gpu-arch=sm_89 \
      --cuda-path=/usr/local/cuda-12.9 -nocudalib -S \
      -o kernels/vq_fused_matmul.ptx kernels/vq_fused_matmul.cu
```

Kernels use only inline-PTX / hardware intrinsics (no libdevice) so they JIT
cleanly under `-nocudalib`.
