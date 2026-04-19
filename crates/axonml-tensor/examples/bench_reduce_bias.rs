//! Microbench — bias-gradient reduction: matmul(ones,grad) vs sum_dim(0).
//!
//! reduce_grad_for_broadcast currently does:
//!   ones(m) [CPU alloc+H2D] → reshape [1,m] → matmul grad[m,n] → [1,n] → reshape [n]
//! Alternative:
//!   grad.sum_dim(0, false)  (fully GPU, no H2D, no matmul)
//!
//! Shape: [bs*seq, hidden] = [4*512, 896] — Qwen3-0.6B attention output bias grad.

use std::time::Instant;

use axonml_core::Device;
use axonml_tensor::Tensor;

fn main() {
    let m = 4 * 512;
    let n = 896;
    let shape = [m, n];
    let numel = m * n;
    let device = Device::Cuda(0);

    let g_data: Vec<f32> = (0..numel).map(|i| ((i % 13) as f32) * 0.01).collect();
    let grad = Tensor::from_vec(g_data, &shape)
        .unwrap()
        .to_device(device.clone())
        .unwrap();

    // Approach A: ones + matmul (current reduce_grad_for_broadcast path)
    let approach_a = |g: &Tensor<f32>| -> Tensor<f32> {
        let ones_data = vec![1.0f32; m];
        let ones = Tensor::from_vec(ones_data, &[1, m])
            .unwrap()
            .to_device(g.device())
            .unwrap();
        ones.matmul(g).unwrap().reshape(&[n as isize]).unwrap()
    };

    // Approach B: sum_dim(0, false) (pure GPU reduction)
    let approach_b = |g: &Tensor<f32>| -> Tensor<f32> { g.sum_dim(0, false) };

    for _ in 0..5 {
        let _ = approach_a(&grad);
        let _ = approach_b(&grad);
    }

    for n_iter in [50, 200, 500] {
        let t0 = Instant::now();
        for _ in 0..n_iter {
            let _ = approach_a(&grad);
        }
        let _ = approach_a(&grad).to_vec();
        let dt = t0.elapsed();
        let per_us = dt.as_micros() as f64 / (n_iter + 1) as f64;
        println!(
            "A (ones+matmul) × {}  = {:.3} ms, {:.1} µs/call",
            n_iter,
            dt.as_secs_f64() * 1000.0,
            per_us
        );
    }

    println!();

    for n_iter in [50, 200, 500] {
        let t0 = Instant::now();
        for _ in 0..n_iter {
            let _ = approach_b(&grad);
        }
        let _ = approach_b(&grad).to_vec();
        let dt = t0.elapsed();
        let per_us = dt.as_micros() as f64 / (n_iter + 1) as f64;
        println!(
            "B (sum_dim 0) × {}  = {:.3} ms, {:.1} µs/call",
            n_iter,
            dt.as_secs_f64() * 1000.0,
            per_us
        );
    }

    // Correctness
    let a = approach_a(&grad).to_vec();
    let b = approach_b(&grad).to_vec();
    let max_abs = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max);
    println!("\nmax_abs_diff = {max_abs:.6e}");
}
