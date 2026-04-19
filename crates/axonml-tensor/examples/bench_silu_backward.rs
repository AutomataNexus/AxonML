//! Microbench — fused silu_backward vs prior 7-op chain.
//!
//! The OLD SiluBackward::apply GPU path did:
//!   sigmoid → ones-H2D → sub → mul → add → mul → mul  (7 ops, 1 H2D)
//! The NEW path does:
//!   silu_backward (single fused kernel)
//!
//! We simulate both and compare per-iteration time on a Qwen3-MLP-sized
//! tensor [bs, seq, inter] = [4, 512, 3072] = ~24 MB fp32. This is the
//! exact shape SiluBackward sees on a real Qwen3-0.6B training step.
//!
//! Run:
//!   cargo run --release --example bench_silu_backward --features cuda

use std::time::Instant;

use axonml_core::Device;
use axonml_tensor::Tensor;

fn main() {
    println!("bench_silu_backward — fused vs 7-op chain\n");

    let bs = 4;
    let seq = 512;
    let inter = 3072;
    let shape = [bs, seq, inter];
    let numel = bs * seq * inter;
    let device = Device::Cuda(0);

    // x = saved forward input (non-trivial values)
    let x_data: Vec<f32> = (0..numel)
        .map(|i| ((i % 1000) as f32 / 1000.0) - 0.5)
        .collect();
    let x = Tensor::from_vec(x_data, &shape)
        .unwrap()
        .to_device(device.clone())
        .unwrap();
    let g_data: Vec<f32> = (0..numel).map(|i| ((i % 7) as f32) * 0.1).collect();
    let grad_out = Tensor::from_vec(g_data, &shape)
        .unwrap()
        .to_device(device.clone())
        .unwrap();

    // Warm-up
    for _ in 0..5 {
        let _ = x.silu_backward(&grad_out);
    }

    // Fused kernel timing
    for n in [50, 200, 500] {
        let t0 = Instant::now();
        for _ in 0..n {
            let _ = x.silu_backward(&grad_out);
        }
        // Force stream drain via a dtoh on one cell
        let _ = x.silu_backward(&grad_out).to_vec();
        let dt = t0.elapsed();
        let per_call_us = dt.as_micros() as f64 / (n + 1) as f64;
        println!(
            "FUSED silu_backward × {}  = {:.3} ms total, {:.1} µs/call",
            n,
            dt.as_secs_f64() * 1000.0,
            per_call_us
        );
    }

    println!("\n--- simulating OLD 7-op chain (for comparison) ---");

    // Reproduce the old chain exactly:
    //   sig = x.sigmoid()
    //   ones = Tensor::ones(x.shape()).to_device(x.device())   <- H2D each call
    //   one_minus_sig = ones - sig
    //   x_term = x * one_minus_sig
    //   bracket = ones + x_term
    //   deriv = sig * bracket
    //   result = grad * deriv
    let old_chain = |x: &Tensor<f32>, g: &Tensor<f32>| -> Tensor<f32> {
        let sig = x.sigmoid();
        let ones = Tensor::ones(x.shape()).to_device(x.device()).unwrap();
        let one_minus_sig = ones.sub(&sig).unwrap();
        let x_term = x.mul(&one_minus_sig).unwrap();
        let bracket = ones.add(&x_term).unwrap();
        let deriv = sig.mul(&bracket).unwrap();
        g.mul(&deriv).unwrap()
    };

    for _ in 0..5 {
        let _ = old_chain(&x, &grad_out);
    }
    for n in [50, 200, 500] {
        let t0 = Instant::now();
        for _ in 0..n {
            let _ = old_chain(&x, &grad_out);
        }
        let _ = old_chain(&x, &grad_out).to_vec();
        let dt = t0.elapsed();
        let per_call_us = dt.as_micros() as f64 / (n + 1) as f64;
        println!(
            "OLD 7-op chain × {}  = {:.3} ms total, {:.1} µs/call",
            n,
            dt.as_secs_f64() * 1000.0,
            per_call_us
        );
    }

    println!("\n--- correctness: max_abs_diff between fused and chain ---");
    let fused = x.silu_backward(&grad_out).to_vec();
    let chain = old_chain(&x, &grad_out).to_vec();
    let max_abs = fused
        .iter()
        .zip(chain.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    println!("max_abs_diff = {max_abs:.6e}");
}
