//! Correctness + perf: fused SwiGLU backward vs the 2-op reference.
//!
//! Reference: forward = silu(gate) * up. Backward = SiluBackward + MulBackward.
//! Fused: one kernel producing (grad_gate, grad_up).

use std::time::Instant;

use axonml_core::Device;
use axonml_core::backends::cuda::cuda_sync;
use axonml_tensor::Tensor;

fn main() {
    // Qwen3-0.6B MLP intermediate: bs*seq × inter = 4*512 × 3072 = 6M.
    let m = 4 * 512;
    let inter = 3072;
    let shape = [m, inter];
    let device = Device::Cuda(0);

    let gate = mkrand(&shape, 0.1, device);
    let up = mkrand(&shape, 0.2, device);
    let grad = mkrand(&shape, 0.01, device);

    // ---------- Correctness ----------
    let (gg_fused, gu_fused) = gate.swiglu_bwd(&up, &grad);

    // Reference CPU computation.
    let g_cpu = gate.to_device(Device::Cpu).unwrap().to_vec();
    let u_cpu = up.to_device(Device::Cpu).unwrap().to_vec();
    let go_cpu = grad.to_device(Device::Cpu).unwrap().to_vec();
    let n = g_cpu.len();
    let mut ref_gg = vec![0.0f32; n];
    let mut ref_gu = vec![0.0f32; n];
    for i in 0..n {
        let g = g_cpu[i];
        let u = u_cpu[i];
        let go = go_cpu[i];
        let sig = 1.0f32 / (1.0 + (-g).exp());
        let silu = g * sig;
        let deriv = sig * (1.0 + g * (1.0 - sig));
        ref_gg[i] = go * u * deriv;
        ref_gu[i] = go * silu;
    }

    let max_gg = max_abs_diff(&gg_fused.to_vec(), &ref_gg);
    let max_gu = max_abs_diff(&gu_fused.to_vec(), &ref_gu);
    println!("grad_gate max_abs_diff = {max_gg:.4e}");
    println!("grad_up   max_abs_diff = {max_gu:.4e}");
    assert!(max_gg < 1e-3, "grad_gate correctness fail");
    assert!(max_gu < 1e-3, "grad_up correctness fail");
    println!("PASS correctness\n");

    // ---------- Perf: fused ----------
    for _ in 0..5 {
        let _ = gate.swiglu_bwd(&up, &grad);
    }
    cuda_sync();
    let n_iter = 100;
    let t = Instant::now();
    for _ in 0..n_iter {
        let _ = gate.swiglu_bwd(&up, &grad);
    }
    cuda_sync();
    println!(
        "fused swiglu_bwd   [{m}, {inter}] {:>7.1} µs/call",
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
