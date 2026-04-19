//! CUDA-graph capture POC / blocker diagnostic.
//!
//! # Status: WORKING
//!
//! CUDA graph capture + replay is live. Two fixes were needed:
//!
//! 1. **Named stream instead of default NULL.** `cuStreamBeginCapture_v2`
//!    rejects the default stream. `CudaBackend::new` now uses
//!    `ctx.new_stream()`.
//!
//! 2. **Event-tracking disabled.** cudarc's `PushKernelArg` for
//!    `&CudaSlice` adds two events per slice arg, guarded by
//!    `ctx.is_managing_stream_synchronization()`. Those events were
//!    recorded pre-capture and caused the `STREAM_CAPTURE_ISOLATION`
//!    error at launch time. `CudaBackend::new` now calls
//!    `ctx.disable_event_tracking()` at startup — safe for AxonML's
//!    single-stream workload because there's no cross-stream hand-off
//!    to synchronize.
//!
//! With both fixes, a pre-bound-buffer kernel chain captures + replays
//! cleanly and replay is measurably faster than eager submit.
//!
//! # What this bench does
//!
//! 1. Runs a 20-deep chain of elementwise adds eagerly and reports
//!    per-call latency — sanity check.
//! 2. Tries to capture the same chain into a CUDA graph. This panics
//!    today with the ISOLATION error; the panic is intentional and left
//!    in so the next person running this sees exactly where we stop.
//!
//! # Remaining hurdle for full training-step capture
//!
//! This POC pre-allocates workspace tensors outside capture. For a real
//! training step wrapped in capture we additionally need `pool_alloc_*` to
//! be capture-safe: a pool miss today calls `cuMemAllocAsync`, which the
//! driver serializes on its internal memory-pool service stream (a
//! separate form of cross-stream dependency from the cudarc events one).
//! Wrapping `cuMemAllocFromPoolAsync` from an explicit `CUmemoryPool`
//! configured with `CU_MEMPOOL_ATTR_RELEASE_THRESHOLD = UINT64_MAX` lets
//! allocations during capture record as graph `MemAllocNode`s and
//! preserves their virtual addresses across replays.
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
    for &chain_depth in &[10usize, 20, 50, 100, 200] {
        run_depth(chain_depth);
        println!();
    }
}

fn run_depth(chain_depth: usize) {
    let n_elems = 1_048_576;
    let device = Device::Cuda(0);

    let a0 = mkrand(&[n_elems], 0.11, device);
    let b = mkrand(&[n_elems], 0.07, device);
    let n_iter = 100;

    // ---------- Eager baseline (via Tensor API with pool allocs) ----------
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
        "eager chain (depth={chain_depth}, via Tensor::add)  {:>9.1} µs/iter  ({:.1} µs/add)",
        eager_us,
        eager_us / chain_depth as f64
    );

    // ---------- Graph capture with pre-bound buffers ----------
    // Skip the pool entirely — pre-allocate two ping-pong output tensors
    // OUTSIDE capture, then during capture just launch add_f32 kernels
    // into them. No cuMemAllocAsync happens on the captured stream.
    let cuda = get_cuda_backend().expect("CUDA backend required");
    let stream = cuda.stream();

    let buf0 = mkrand(&[n_elems], 0.0, device);
    let buf1 = mkrand(&[n_elems], 0.0, device);

    // Seed buf0 from a0.
    cuda_sync();
    let t_warm = Instant::now();
    for _ in 0..5 {
        eager_add_ping_pong(cuda, &a0, &b, &buf0, &buf1, chain_depth);
    }
    cuda_sync();
    let eager_ppg_us = t_warm.elapsed().as_micros() as f64 / 5.0;
    println!(
        "eager ping-pong (depth={chain_depth}, raw kernel)    {:>9.1} µs/iter",
        eager_ppg_us
    );

    use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};
    println!("\ncapturing stream (pre-bound buffers, event-tracking off)…");
    stream
        .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
        .expect("begin_capture failed");
    eager_add_ping_pong(cuda, &a0, &b, &buf0, &buf1, chain_depth);
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
        "graph replay (depth={chain_depth}, pre-bound)       {:>9.1} µs/iter  ({:.1} µs/add)",
        graph_us,
        graph_us / chain_depth as f64
    );
    println!(
        "\ngraph vs raw-eager:  {:.2}× speedup",
        eager_ppg_us / graph_us.max(1.0)
    );
    println!(
        "graph vs Tensor API: {:.2}× speedup",
        eager_us / graph_us.max(1.0)
    );
}

/// Fires `depth` `add_f32` kernels alternating between `buf0` and `buf1` as
/// the running accumulator. First iteration reads from `a0`; subsequent
/// iterations read from the previous output. No pool allocations, no Tensor
/// wrapping — just raw kernel launches into pre-allocated buffers so the
/// captured work stream is pure compute.
fn eager_add_ping_pong(
    cuda: &axonml_core::backends::cuda::CudaBackend,
    a0: &Tensor<f32>,
    b: &Tensor<f32>,
    buf0: &Tensor<f32>,
    buf1: &Tensor<f32>,
    depth: usize,
) {
    let n = a0.numel();

    // Step 0: buf0 = a0 + b — scope the guards so they release before the loop.
    {
        let a_slice = a0.as_cuda_slice_read();
        let b_slice = b.as_cuda_slice_read();
        let mut buf0_guard = buf0.as_cuda_slice_write();
        cuda.add_f32(buf0_guard.slice_mut(), a_slice.slice(), b_slice.slice(), n)
            .expect("add_f32 step 0");
    }

    // Ping-pong: alternate reading from buf0/buf1 and writing the other.
    let mut src_is_buf0 = true;
    for _ in 1..depth {
        let b_slice = b.as_cuda_slice_read();
        if src_is_buf0 {
            let src_guard = buf0.as_cuda_slice_read();
            let mut dst_guard = buf1.as_cuda_slice_write();
            cuda.add_f32(dst_guard.slice_mut(), src_guard.slice(), b_slice.slice(), n)
                .expect("add_f32 step");
        } else {
            let src_guard = buf1.as_cuda_slice_read();
            let mut dst_guard = buf0.as_cuda_slice_write();
            cuda.add_f32(dst_guard.slice_mut(), src_guard.slice(), b_slice.slice(), n)
                .expect("add_f32 step");
        }
        src_is_buf0 = !src_is_buf0;
    }
}

fn mkrand(shape: &[usize], seed: f32, dev: Device) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|i| seed + ((i % 19) as f32) * 0.003).collect();
    Tensor::from_vec(data, shape)
        .unwrap()
        .to_device(dev)
        .unwrap()
}
