//! Correctness + perf: fused (residual_add + RMSNorm) vs the 2-op reference.

use std::time::Instant;

use axonml_core::Device;
use axonml_core::backends::cuda::cuda_sync;
use axonml_tensor::Tensor;

fn main() {
    let m = 4 * 512;
    let n = 1024;
    let eps = 1e-6;
    let device = Device::Cuda(0);

    let a = mkrand(&[m, n], 0.11, device);
    let b = mkrand(&[m, n], 0.07, device);
    let w = mkrand(&[n], 0.9, device);

    // ---------- Correctness ----------
    let (fused_out, fused_sum) = a.add_rmsnorm_batched(&b, &w, m, n, eps);

    // Reference: a + b → RMSNorm.
    let ref_sum = a.add(&b).unwrap();
    let ref_out = ref_sum.rms_norm_batched(&w, m, n, eps);

    let max_sum = max_abs_diff(&fused_sum.to_vec(), &ref_sum.to_vec());
    let max_out = max_abs_diff(&fused_out.to_vec(), &ref_out.to_vec());
    println!("sum max_abs_diff = {max_sum:.4e}");
    println!("out max_abs_diff = {max_out:.4e}");
    assert!(max_sum < 1e-5, "sum correctness fail");
    assert!(max_out < 1e-3, "out correctness fail");
    println!("PASS correctness\n");

    // ---------- Perf: fused ----------
    for _ in 0..5 {
        let _ = a.add_rmsnorm_batched(&b, &w, m, n, eps);
    }
    cuda_sync();
    let n_iter = 100;
    let t = Instant::now();
    for _ in 0..n_iter {
        let _ = a.add_rmsnorm_batched(&b, &w, m, n, eps);
    }
    cuda_sync();
    println!(
        "fused add+rmsnorm   [{m}, {n}] {:>7.1} µs/call",
        t.elapsed().as_micros() as f64 / n_iter as f64
    );

    // ---------- Perf: 2-op reference ----------
    for _ in 0..5 {
        let s = a.add(&b).unwrap();
        let _ = s.rms_norm_batched(&w, m, n, eps);
    }
    cuda_sync();
    let t = Instant::now();
    for _ in 0..n_iter {
        let s = a.add(&b).unwrap();
        let _ = s.rms_norm_batched(&w, m, n, eps);
    }
    cuda_sync();
    println!(
        "ref   add + rmsnorm             {:>7.1} µs/call",
        t.elapsed().as_micros() as f64 / n_iter as f64
    );
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
