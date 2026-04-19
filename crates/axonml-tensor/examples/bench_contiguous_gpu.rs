//! Microbench — how much does `Tensor<f32>::contiguous()` cost per call
//! for a non-contiguous tensor, and what's the cost of the whole
//! transpose + contiguous + elementwise chain?
//!
//! Run:
//!   cargo run --release --example bench_contiguous --features cuda
//!
//! Baselines matter: if contiguous takes <500µs per call, the htod
//! round-trip inside it isn't the primary training bottleneck; if it's
//! multi-millisecond, it IS the leverage point.

use std::time::Instant;

use axonml_core::Device;
use axonml_tensor::Tensor;

fn main() {
    println!("bench_contiguous — Tensor op overhead audit\n");

    // Pick a shape close to a Qwen3-0.6B attention intermediate:
    // [bs, n_heads, seq, head_dim] = [4, 16, 512, 64] = 2,097,152 f32 = 8 MB
    let bs = 4;
    let n_heads = 16;
    let seq = 512;
    let head_dim = 64;

    let device = Device::Cuda(0);
    let numel = bs * n_heads * seq * head_dim;
    let data: Vec<f32> = (0..numel).map(|i| i as f32 * 0.001).collect();

    // Contiguous GPU tensor, shape [bs, seq, n_heads, head_dim]
    let t = Tensor::from_vec(data, &[bs, seq, n_heads, head_dim])
        .expect("from_vec")
        .to_device(device.clone())
        .expect("to_device");

    // Transpose(1, 2) — [bs, n_heads, seq, head_dim], NON-contiguous
    let t_nc = t.transpose(1, 2).expect("transpose");
    println!("t_nc.is_contiguous() = {}", t_nc.is_contiguous());
    println!("t_nc.shape() = {:?}", t_nc.shape());
    println!("t_nc.strides() = {:?}\n", t_nc.strides());

    // Warm-up: 5 calls to amortize first-call overhead (kernel cache, pool alloc)
    for _ in 0..5 {
        let _ = t_nc.contiguous();
    }

    // Benchmark: N iters of contiguous
    for n in [100, 500, 1000] {
        let t0 = Instant::now();
        for _ in 0..n {
            let _ = t_nc.contiguous();
        }
        // cudaDeviceSynchronize via one final to_vec to force stream drain
        let dt = t0.elapsed();
        let per_call_us = dt.as_micros() as f64 / n as f64;
        println!(
            "contiguous × {}  = {:.3} ms total, {:.1} µs/call",
            n,
            dt.as_secs_f64() * 1000.0,
            per_call_us
        );
    }

    println!("\n--- baseline: plain add on contiguous tensors ---");
    let a = Tensor::from_vec(vec![1.0f32; numel], &[numel])
        .unwrap()
        .to_device(device.clone())
        .unwrap();
    let b = a.clone();
    for _ in 0..5 {
        let _ = a.add(&b);
    }
    for n in [100, 500, 1000] {
        let t0 = Instant::now();
        for _ in 0..n {
            let _ = a.add(&b);
        }
        let dt = t0.elapsed();
        let per_call_us = dt.as_micros() as f64 / n as f64;
        println!(
            "tensor.add  (contig) × {}  = {:.3} ms total, {:.1} µs/call",
            n,
            dt.as_secs_f64() * 1000.0,
            per_call_us
        );
    }

    println!("\n--- combo: transpose → contiguous → add (realistic) ---");
    let other_full = Tensor::from_vec(vec![0.5f32; numel], &[bs, n_heads, seq, head_dim])
        .unwrap()
        .to_device(device.clone())
        .unwrap();
    for _ in 0..5 {
        let t_c = t_nc.contiguous();
        let _ = t_c.add(&other_full);
    }
    for n in [100, 500, 1000] {
        let t0 = Instant::now();
        for _ in 0..n {
            let t_c = t_nc.contiguous();
            let _ = t_c.add(&other_full);
        }
        let dt = t0.elapsed();
        let per_call_us = dt.as_micros() as f64 / n as f64;
        println!(
            "transpose.contig + add × {}  = {:.3} ms total, {:.1} µs/call",
            n,
            dt.as_secs_f64() * 1000.0,
            per_call_us
        );
    }
}
