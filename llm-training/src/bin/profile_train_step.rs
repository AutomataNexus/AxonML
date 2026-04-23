//! Profile a single Qwen3 forward + backward + optimizer step.
//!
//! Goal: bucket the 45s/step distillation wall-clock into forward vs
//! backward vs optimizer, and (with `AXONML_PROFILE_BACKWARD=1` set)
//! print per-backward-op timing so we know which GradFn to fix next.
//!
//! This is NOT a full distill — no teacher, no dataset, just random
//! token IDs into a fresh tiny-Qwen3 and a shifted-CE loss. The point
//! is to make one step fast to iterate on, not to train a model.
//!
//! Run (with CUDA):
//!   cargo run --release --features cuda --bin profile_train_step
//!
//! Run with per-backward-op breakdown:
//!   AXONML_PROFILE_BACKWARD=1 \
//!     cargo run --release --features cuda --bin profile_train_step

use std::time::Instant;

use axonml_core::Device;
use axonml_llm::qwen3::{Qwen3Config, Qwen3ForCausalLM};
use axonml_optim::{AdamW, Optimizer};
use axonml_tensor::Tensor;
use llm_training::shifted_cross_entropy;

fn main() {
    // Scaled down from the real distill config (bs=4 seq=512) to fit the
    // 12 GB laptop GPU once activations live fully on-device. The per-step
    // shape of the graph (# ops, # kernel launches) is preserved — bs and
    // seq only change allocation size, not op count.
    let bs = 2;
    let seq = 256;

    // Qwen3-0.6B shape — the real distill target.
    let mut cfg = Qwen3Config::qwen3_0_6b();
    // Keep vocab manageable so random labels don't dominate CE cost.
    cfg.vocab_size = 1024;
    cfg.max_position_embeddings = seq;

    let device = pick_device();
    println!("profile_train_step — Qwen3-0.6B-shaped, bs={bs} seq={seq}, device={device:?}");
    println!();

    let t_build = Instant::now();
    let model = Qwen3ForCausalLM::new(&cfg);
    for p in model.parameters() {
        p.to_device(device);
    }
    println!(
        "  model init + to_device   {:.2} ms",
        t_build.elapsed().as_secs_f64() * 1000.0
    );

    let params = model.parameters();
    let mut optimizer = AdamW::new(params, 1e-4);

    // Random token IDs. u32 Tensors stay on CPU (GPU path is f32-only).
    let ids: Vec<u32> = (0..bs * seq)
        .map(|i| (i as u32) % cfg.vocab_size as u32)
        .collect();
    let input_ids = Tensor::<u32>::from_vec(ids.clone(), &[bs, seq]).unwrap();
    let labels = Tensor::<u32>::from_vec(ids, &[bs, seq]).unwrap();

    // Warm-up step — first step does one-time CUDA initialization, PTX
    // parsing, pool priming; throws off the next-step measurement.
    println!("\n--- warm-up step (not counted) ---");
    run_step(&model, &mut optimizer, &input_ids, &labels, device);

    // Hot step — all the numbers we care about.
    println!("\n--- hot step ---");
    print_pool_stats("before hot step");
    let breakdown = run_step(&model, &mut optimizer, &input_ids, &labels, device);
    print_pool_stats("after hot step");
    println!();
    println!(
        "  forward                  {:>8.1} ms",
        breakdown.forward_ms
    );
    println!("  loss (shifted CE)        {:>8.1} ms", breakdown.loss_ms);
    println!(
        "  backward                 {:>8.1} ms",
        breakdown.backward_ms
    );
    println!("  optimizer.step           {:>8.1} ms", breakdown.optim_ms);
    println!("  ───────────────────────────────────");
    println!(
        "  TOTAL                    {:>8.1} ms",
        breakdown.forward_ms + breakdown.loss_ms + breakdown.backward_ms + breakdown.optim_ms
    );
    println!();
    // =========================================================================
    // CUDA-graph capture + replay of the full training step
    // =========================================================================
    #[cfg(feature = "cuda")]
    try_graph_capture_step(&model, &mut optimizer, &input_ids, &labels, device);

    println!(
        "Set AXONML_PROFILE_BACKWARD=1 for per-backward-op breakdown. Current: {}",
        if std::env::var("AXONML_PROFILE_BACKWARD").is_ok() {
            "ON"
        } else {
            "OFF"
        }
    );
}

fn pick_device() -> Device {
    #[cfg(feature = "cuda")]
    {
        Device::Cuda(0)
    }
    #[cfg(not(feature = "cuda"))]
    {
        Device::Cpu
    }
}

struct Breakdown {
    forward_ms: f64,
    loss_ms: f64,
    backward_ms: f64,
    optim_ms: f64,
}

fn run_step(
    model: &Qwen3ForCausalLM,
    optimizer: &mut AdamW,
    input_ids: &Tensor<u32>,
    labels: &Tensor<u32>,
    device: Device,
) -> Breakdown {
    let _ = device;
    optimizer.zero_grad();
    // Drain any leftover GPU work before we start measuring.
    sync();

    // Forward
    let t_fwd = Instant::now();
    let logits = model.forward_ids(input_ids);
    sync();
    let forward_ms = t_fwd.elapsed().as_secs_f64() * 1000.0;

    // Shifted CE loss: drop last logit position, use [1..] labels.
    let t_loss = Instant::now();
    let loss = shifted_cross_entropy(&logits, labels);
    sync();
    let loss_ms = t_loss.elapsed().as_secs_f64() * 1000.0;

    // Backward
    let t_bwd = Instant::now();
    loss.backward();
    sync();
    let backward_ms = t_bwd.elapsed().as_secs_f64() * 1000.0;

    // Optimizer step
    let t_opt = Instant::now();
    optimizer.step();
    sync();
    let optim_ms = t_opt.elapsed().as_secs_f64() * 1000.0;

    Breakdown {
        forward_ms,
        loss_ms,
        backward_ms,
        optim_ms,
    }
}

/// Like `run_step` but with NO stream syncs — safe to call inside a CUDA
/// stream capture region. Every `cuda_sync` invalidates capture.
fn run_step_no_sync(
    model: &Qwen3ForCausalLM,
    optimizer: &mut AdamW,
    input_ids: &Tensor<u32>,
    labels: &Tensor<u32>,
) {
    optimizer.zero_grad();
    let logits = model.forward_ids(input_ids);
    let loss = shifted_cross_entropy(&logits, labels);
    loss.backward();
    optimizer.step();
}

/// Attempts CUDA-graph capture of the full training step (forward +
/// loss + backward + optimizer.step), replays it N times, and reports
/// wall-clock delta vs eager. Swallows `STREAM_CAPTURE_ISOLATION` panics
/// with a diagnostic so the profiler doesn't abort if a remaining
/// non-captured stream dependency is hit.
#[cfg(feature = "cuda")]
fn try_graph_capture_step(
    model: &Qwen3ForCausalLM,
    optimizer: &mut AdamW,
    input_ids: &Tensor<u32>,
    labels: &Tensor<u32>,
    device: Device,
) {
    use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};
    println!("\n--- CUDA graph capture attempt ---");
    let cuda = match axonml_core::backends::cuda::get_cuda_backend() {
        Some(b) => b,
        None => {
            println!("  (no CUDA backend; skipping)");
            return;
        }
    };
    let stream = cuda.stream();

    // Extra warmup: pool needs to be saturated for every size the step
    // touches so no cuMemAllocAsync fires during capture.
    for _ in 0..2 {
        run_step(model, optimizer, input_ids, labels, device);
    }
    sync();

    // Fine-grained capture status checks to pinpoint the first op that
    // invalidates the stream (subsequent ops all report
    // CAPTURE_INVALIDATED which is useless for debugging).
    use cudarc::driver::sys::CUstreamCaptureStatus;
    let check = |tag: &str| {
        let st = stream
            .capture_status()
            .unwrap_or(CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_INVALIDATED);
        if st != CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE {
            println!("  [capture dead after: {tag}] status={st:?}");
            true
        } else {
            false
        }
    };

    // Keep the Rust pool active during capture — it's LIFO-deterministic
    // per bucket, so allocations made at capture-time hand out the same
    // pointers the replay-time allocations will also receive. The driver's
    // cuMemAllocAsync + MemAllocNode path would let CUDA manage memory
    // but kernel nodes in the captured graph don't re-bind to the new
    // allocations on replay; our Rust pool dodges that by keeping the
    // pointers stable.
    // Pre-warm: run a few full steps so every Qwen3 bucket is populated.
    // Pool is LIFO per bucket → subsequent captures hand out the same
    // pointers the replay-time allocations will.
    for _ in 0..3 {
        run_step(model, optimizer, input_ids, labels, device);
    }
    sync();

    // Capture pen retains every CudaSlice produced inside the capture
    // scope so cudarc's `&slice.cu_device_ptr` kernel args stay at stable
    // host addresses through every graph.launch() replay. Without this,
    // intermediate tensors drop at end-of-statement and their host memory
    // is reclaimed → graph.launch hits CUDA_ERROR_ILLEGAL_ADDRESS reading
    // freed host bytes.
    let (result, pen) = axonml_core::backends::cuda_pool::with_capture_pen(|| {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            stream
                .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
                .expect("begin_capture");
            if check("begin_capture") {
                return None;
            }
            optimizer.zero_grad();
            if check("zero_grad") {
                return None;
            }
            let logits = model.forward_ids(input_ids);
            if check("forward_ids") {
                return None;
            }
            let loss = llm_training::shifted_cross_entropy(&logits, labels);
            if check("shifted_ce") {
                return None;
            }
            loss.backward();
            if check("backward") {
                return None;
            }
            optimizer.step();
            if check("optim.step") {
                return None;
            }
            // flags = 0 via transmute — cudarc 0.19's enum for our CUDA
            // version only names AUTO_FREE_ON_LAUNCH. No auto-free means
            // the graph owns its internal allocations for its whole
            // lifetime and replay reuses the same virtual addresses.
            let flags_zero: CUgraphInstantiate_flags =
                unsafe { std::mem::transmute::<u32, CUgraphInstantiate_flags>(0) };
            Some(
                stream
                    .end_capture(flags_zero)
                    .expect("end_capture")
                    .expect("graph empty"),
            )
        }))
    });
    println!(
        "  capture pen retained: f32 × {}, u32 × {} (total {})",
        pen.f32_count(),
        pen.u32_count(),
        pen.total()
    );
    // Ensure any in-flight capture is ended even if we early-returned so
    // the next eager op doesn't see stale capture state.
    let _ = stream
        .end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH);
    let result = result.map(|o| o.expect("graph capture short-circuited"));

    match result {
        Ok(graph) => {
            sync();
            let t = Instant::now();
            let n_iter = 5;
            for _ in 0..n_iter {
                graph.launch().expect("graph launch");
            }
            sync();
            let replay_ms = t.elapsed().as_secs_f64() * 1000.0 / n_iter as f64;
            println!("  graph replay:  {replay_ms:>8.1} ms / step (× {n_iter} replays)");
            println!(
                "  (compare to eager TOTAL above; gain = launch-overhead delta × kernel count)"
            );
            // Graph dropped here. Safe to return pen slices to the pool.
            drop(graph);
            pen.release();
        }
        Err(e) => {
            let msg = panic_msg(&e);
            println!("  capture FAILED: {msg}");
            // Still release pen slices so the pool recovers the buffers.
            pen.release();
        }
    }
}

#[cfg(feature = "cuda")]
fn panic_msg(e: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = e.downcast_ref::<&'static str>() {
        (*s).to_string()
    } else if let Some(s) = e.downcast_ref::<String>() {
        s.clone()
    } else {
        "<unknown panic payload>".into()
    }
}

#[inline]
fn sync() {
    #[cfg(feature = "cuda")]
    {
        let _ = axonml_core::backends::cuda::cuda_sync();
    }
}

fn print_pool_stats(label: &str) {
    #[cfg(feature = "cuda")]
    {
        let pool = axonml_core::backends::cuda_pool::get_memory_pool();
        let (hits, misses, returns, bytes) = pool.stats();
        println!(
            "  [pool {label}] hits={hits} misses={misses} returns={returns} pooled={}MB",
            bytes / (1024 * 1024)
        );
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = label;
    }
}
