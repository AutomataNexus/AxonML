//! Correctness + perf: GPU RMSNorm forward + backward vs CPU reference.
//!
//! Before: RMSNorm::forward and RMSNormBackward::apply in axonml-llm::llama
//! were CPU-only (full D2H of x + weight + grad_output, O(m*n) CPU loop,
//! H2D of result). On profile_train_step that showed ~61 ms/call × 85 =
//! ~5.2 s/step for backward + a proportional cost in forward.
//!
//! After: `Tensor::rms_norm_batched` (existing kernel) for forward and a
//! new `rms_norm_bwd_batched_f32` kernel for backward. One CTA per row,
//! dual-reduction (sum_sq + dot) in shmem, then per-thread write.

use std::time::Instant;

use axonml_core::Device;
use axonml_core::backends::cuda::cuda_sync;
use axonml_tensor::Tensor;

fn main() {
    // Qwen3-0.6B: bs*seq=2048 tokens, hidden=1024
    let m = 4 * 512;
    let n = 1024;
    let eps = 1e-6;
    let device = Device::Cuda(0);

    let x_gpu = mkrand(&[m, n], 0.1, device);
    let w_gpu = mkrand(&[n], 0.9, device);
    let g_gpu = mkrand(&[m, n], 0.01, device);

    let x_cpu = x_gpu.to_device(Device::Cpu).unwrap();
    let w_cpu = w_gpu.to_device(Device::Cpu).unwrap();
    let g_cpu = g_gpu.to_device(Device::Cpu).unwrap();

    // ---------- Correctness: forward ----------
    let fwd_gpu = x_gpu.rms_norm_batched(&w_gpu, m, n, eps);
    let fwd_cpu = x_cpu.rms_norm_batched(&w_cpu, m, n, eps);
    let max_fwd = max_abs_diff(&fwd_gpu.to_vec(), &fwd_cpu.to_vec());
    println!("forward  max_abs_diff (GPU vs CPU) = {max_fwd:.4e}");
    assert!(max_fwd < 5e-3, "forward correctness fail");

    // ---------- Correctness: backward ----------
    let bwd_gpu = x_gpu.rms_norm_bwd_batched(&w_gpu, &g_gpu, m, n, eps);
    let bwd_cpu = x_cpu.rms_norm_bwd_batched(&w_cpu, &g_cpu, m, n, eps);
    let max_bwd = max_abs_diff(&bwd_gpu.to_vec(), &bwd_cpu.to_vec());
    println!("backward max_abs_diff (GPU vs CPU) = {max_bwd:.4e}");
    assert!(max_bwd < 5e-3, "backward correctness fail");
    println!("PASS correctness\n");

    // ---------- Perf: forward ----------
    for _ in 0..5 {
        let _ = x_gpu.rms_norm_batched(&w_gpu, m, n, eps);
    }
    cuda_sync();
    let n_iter = 100;
    let t = Instant::now();
    for _ in 0..n_iter {
        let out = x_gpu.rms_norm_batched(&w_gpu, m, n, eps);
        std::hint::black_box(out);
    }
    cuda_sync();
    println!(
        "forward  GPU [m={m}, n={n}]   {:>7.1} µs/call",
        t.elapsed().as_micros() as f64 / n_iter as f64
    );

    // ---------- Perf: backward ----------
    for _ in 0..5 {
        let _ = x_gpu.rms_norm_bwd_batched(&w_gpu, &g_gpu, m, n, eps);
    }
    cuda_sync();
    let t = Instant::now();
    for _ in 0..n_iter {
        let out = x_gpu.rms_norm_bwd_batched(&w_gpu, &g_gpu, m, n, eps);
        std::hint::black_box(out);
    }
    cuda_sync();
    println!(
        "backward GPU [m={m}, n={n}]   {:>7.1} µs/call",
        t.elapsed().as_micros() as f64 / n_iter as f64
    );

    // CPU baseline — what the old RMSNormBackward::apply was doing.
    let t = Instant::now();
    let _ = x_cpu.rms_norm_bwd_batched(&w_cpu, &g_cpu, m, n, eps);
    let cpu_us = t.elapsed().as_micros();
    println!("backward CPU reference (1 call)    {cpu_us} µs");
}

fn mkrand(shape: &[usize], seed: f32, dev: Device) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|i| seed + ((i % 19) as f32) * 0.004).collect();
    Tensor::from_vec(data, shape)
        .unwrap()
        .to_device(dev)
        .unwrap()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}
