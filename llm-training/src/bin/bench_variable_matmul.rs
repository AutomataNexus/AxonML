//! Does Variable::matmul autograd overhead account for the 90s backward?
//!
//! profile_train_step sees ~90s backward on Qwen3-0.6B. Raw Tensor
//! matmul submits in 9 µs/call (bench_linear_bwd stream-backlog).
//! GPU compute is ~1.45 ms/call. So the delta between submit+compute
//! (~600ms for 400 calls) and observed (90s for ~600 calls) is 150×.
//!
//! This bench chains Tensor::matmul + Variable::matmul + Variable::matmul+backward
//! so we catch any autograd-specific cost (graph registration, GradFn
//! alloc, Arc churn, Tensor clone).

use std::time::Instant;

use axonml_autograd::Variable;
use axonml_core::Device;
use axonml_tensor::Tensor;

#[cfg(feature = "cuda")]
fn sync() {
    axonml_core::backends::cuda::cuda_sync();
}
#[cfg(not(feature = "cuda"))]
fn sync() {}

fn main() {
    let m = 2048;
    let k = 1024;
    let n = 3072;
    let device = pick_device();

    let a_t = mkrand(&[m, k], 0.1, device);
    let b_t = mkrand(&[k, n], 0.2, device);

    let a = Variable::new(a_t.clone(), true);
    let b = Variable::new(b_t.clone(), true);

    for _ in 0..3 {
        let _ = a.matmul(&b);
    }
    sync();

    // 1. Raw Tensor::matmul (no autograd)
    let n_iter = 200;
    sync();
    let t = Instant::now();
    for _ in 0..n_iter {
        let c = a_t.matmul(&b_t).unwrap();
        std::hint::black_box(c);
    }
    sync();
    println!(
        "Tensor::matmul (no autograd)   {:>7.1} µs/call",
        t.elapsed().as_micros() as f64 / n_iter as f64
    );

    // 2. Variable::matmul (autograd on, no backward, no graph clear)
    sync();
    let t = Instant::now();
    for _ in 0..n_iter {
        let c = a.matmul(&b);
        std::hint::black_box(c);
    }
    sync();
    println!(
        "Variable::matmul (no bwd)      {:>7.1} µs/call",
        t.elapsed().as_micros() as f64 / n_iter as f64
    );

    // 3. Variable::matmul + backward (graph clears after each)
    let n_iter = 50;
    sync();
    let t = Instant::now();
    for _ in 0..n_iter {
        let c = a.matmul(&b);
        let loss = c.sum();
        loss.backward();
    }
    sync();
    println!(
        "Variable matmul+sum+backward   {:>7.1} µs/call   [contains 1 bwd matmul]",
        t.elapsed().as_micros() as f64 / n_iter as f64
    );

    // 4. Deep graph: chain matmuls + backward once. Mimics Qwen3 depth.
    let sq_t = mkrand(&[n, n], 0.05, device);
    let sq = Variable::new(sq_t.clone(), true);

    for depth in &[5usize, 10, 20, 40] {
        sync();
        let t = Instant::now();
        let n_reps = 5;
        for _ in 0..n_reps {
            let a_nsq = mkrand(&[m, n], 0.03, device);
            let mut x = Variable::new(a_nsq, true);
            for _ in 0..*depth {
                x = x.matmul(&sq);
            }
            let loss = x.sum();
            loss.backward();
        }
        sync();
        let total_us = t.elapsed().as_micros() as f64;
        let per_matmul_us = total_us / n_reps as f64 / (2 * *depth) as f64;
        println!(
            "depth={:2} × {} reps: {:>9.1} µs/run  →  {:>7.1} µs per matmul (fwd+bwd)",
            depth, n_reps, total_us / n_reps as f64, per_matmul_us
        );
    }
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

fn mkrand(shape: &[usize], seed: f32, dev: Device) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|i| seed + ((i % 17) as f32) * 0.003).collect();
    Tensor::from_vec(data, shape)
        .unwrap()
        .to_device(dev)
        .unwrap()
}
