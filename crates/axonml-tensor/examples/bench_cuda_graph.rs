//! CUDA-graph capture POC / blocker diagnostic.
//!
//! # Status
//!
//! **Stream is now capture-capable** (backend moved from `default_stream()`
//! to `new_stream()` — the default NULL stream rejects capture with
//! `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`).
//!
//! **Capture still panics** on any real op with
//! `CUDA_ERROR_STREAM_CAPTURE_ISOLATION: dependency created on uncaptured
//! work in another stream`. Root cause: our memory pool path calls into
//! the CUDA async allocator (`cuMemAllocAsync`) during kernel setup, and
//! the driver's internal memory-pool service runs on its own stream,
//! creating a cross-stream dependency the capture refuses.
//!
//! # What this bench does
//!
//! 1. Runs a 20-deep chain of elementwise adds eagerly and reports
//!    per-call latency — sanity check.
//! 2. Tries to capture the same chain into a CUDA graph. This panics
//!    today with the ISOLATION error; the panic is intentional and left
//!    in so the next person running this sees exactly where we stop.
//!
//! # What's needed to unblock
//!
//! Graph-safe memory needs one of:
//!
//! 1. **Pre-bound workspace tensors.** Every op on the captured path
//!    writes into a tensor allocated *before* capture starts. No
//!    `pool_alloc_*` calls during capture. This is how PyTorch's
//!    `CUDAGraph.capture_begin()` works — you must pre-allocate all
//!    intermediate buffers (they expose this via the `cuda_graph_pool`
//!    + explicit output tensors in custom ops).
//!
//! 2. **Stream-ordered allocator with explicit memory pool.** Create a
//!    `cudaMemPool_t` with `cudaMemPoolAttrReleaseThreshold = UINT64_MAX`
//!    and set it as the allocator for our stream. Allocations during
//!    capture then happen *on the captured stream* and the graph records
//!    them as `cudaGraphAddMemAllocNode` + `cudaGraphAddMemFreeNode`.
//!    On replay the same virtual addresses are reused deterministically.
//!    cudarc doesn't yet expose `cudaMallocFromPoolAsync` as a safe API,
//!    so this path would need a small unsafe wrapper around
//!    `cuda::driver::sys::cuMemAllocFromPoolAsync` in axonml-core.
//!
//! Option (1) is the cleaner fit for this codebase because it keeps the
//! autograd graph deterministic: the forward path would pre-compute its
//! output shapes, allocate the workspace tensors once, then all steps
//! share those pointers. Backward similarly.
//!
//! # Why it's still worth fixing
//!
//! profile_train_step bottom line: total 49 s / step, backward 16.5 s,
//! of which MatMulBackward alone is 13 s × 197 = ~66 ms per call on
//! kernels whose pure compute is <2 ms. That 30× multiplier is the WSL2
//! + Blackwell per-kernel stream-submit latency. CUDA graph replay
//! collapses it into a single submit per training step (instead of
//! ~1 200 per backward). If pre-allocated workspaces make capture succeed,
//! the projected gain is measured in tens of seconds per step.
//!
//! Run:
//!   cargo run --release --features cuda -p axonml-tensor --example bench_cuda_graph

use std::time::Instant;

use axonml_core::Device;
use axonml_core::backends::cuda::{cuda_sync, get_cuda_backend};
use axonml_tensor::Tensor;

fn main() {
    let n_elems = 1_048_576;
    let chain_depth = 20;
    let device = Device::Cuda(0);

    let a0 = mkrand(&[n_elems], 0.11, device);
    let b = mkrand(&[n_elems], 0.07, device);
    let n_iter = 100;

    // ---------- Eager baseline ----------
    for _ in 0..3 {
        let mut x = a0.clone();
        for _ in 0..chain_depth {
            x = x.add(&b).unwrap();
        }
        cuda_sync();
        std::hint::black_box(x);
    }
    cuda_sync();
    let t = Instant::now();
    for _ in 0..n_iter {
        let mut x = a0.clone();
        for _ in 0..chain_depth {
            x = x.add(&b).unwrap();
        }
        std::hint::black_box(x);
    }
    cuda_sync();
    let eager_us = t.elapsed().as_micros() as f64 / n_iter as f64;
    println!(
        "eager chain (depth={chain_depth})          {:>9.1} µs/iter  ({:.1} µs/add)",
        eager_us,
        eager_us / chain_depth as f64
    );

    // ---------- Graph capture attempt ----------
    let cuda = get_cuda_backend().expect("CUDA backend required");
    let stream = cuda.stream();

    // Pre-warm the pool (doesn't help, but rules out cold allocation paths).
    for _ in 0..5 {
        let mut x = a0.clone();
        for _ in 0..chain_depth {
            x = x.add(&b).unwrap();
        }
        std::hint::black_box(x);
    }
    cuda_sync();

    use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};
    println!("\nattempting stream capture (expect STREAM_CAPTURE_ISOLATION panic);");
    println!("see file header for what's needed to unblock this…");
    stream
        .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
        .expect("begin_capture failed — default stream? wrong mode?");
    let mut captured_x = a0.clone();
    for _ in 0..chain_depth {
        // This panics the first time around with:
        //   CUDA_ERROR_STREAM_CAPTURE_ISOLATION — dependency created on
        //   uncaptured work in another stream
        // because the pool_alloc path's cuMemAllocAsync talks to the
        // driver's memory-pool service stream.
        captured_x = captured_x.add(&b).unwrap();
    }
    let graph = stream
        .end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
        .expect("end_capture failed")
        .expect("graph empty");

    cuda_sync();
    let t = Instant::now();
    for _ in 0..n_iter {
        graph.launch().expect("graph launch failed");
    }
    cuda_sync();
    let graph_us = t.elapsed().as_micros() as f64 / n_iter as f64;
    println!(
        "graph replay (depth={chain_depth})         {:>9.1} µs/iter  ({:.1} µs/add)",
        graph_us,
        graph_us / chain_depth as f64
    );
    println!(
        "\ngraph speedup: {:.2}× on a {}-deep add chain",
        eager_us / graph_us.max(1.0),
        chain_depth
    );
    println!(
        "(captured_x pins output memory: shape={:?})",
        captured_x.shape()
    );
}

fn mkrand(shape: &[usize], seed: f32, dev: Device) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|i| seed + ((i % 19) as f32) * 0.003).collect();
    Tensor::from_vec(data, shape)
        .unwrap()
        .to_device(dev)
        .unwrap()
}
