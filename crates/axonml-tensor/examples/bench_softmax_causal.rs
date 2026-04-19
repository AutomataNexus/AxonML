//! Correctness + perf: fused causal-scaled softmax vs the 3-op reference.
//!
//! Reference (what Qwen3 attention used to do):
//!   y = (scores * scale + causal_mask).softmax(-1)
//! Fused single kernel:
//!   y = softmax_causal_scaled(scores, tq, tk, offset, scale)
//!
//! Also tests the backward pass: given saved `p` and upstream grad, the
//! fused kernel produces `grad_scores = scale * p * (grad_out - dot(p, grad_out))`
//! in a single launch vs. the separate SoftmaxBackward + MulScalarBackward.

use std::time::Instant;

use axonml_core::Device;
use axonml_core::backends::cuda::cuda_sync;
use axonml_tensor::Tensor;

fn main() {
    // Qwen3-0.6B attention shape: [bs, heads, seq, seq]
    let bs = 4;
    let heads = 16;
    let seq = 512;
    let scale = 1.0f32 / (128.0f32).sqrt();
    let offset = 0usize;
    let device = Device::Cuda(0);

    let scores = mkrand(&[bs, heads, seq, seq], 0.01, device);
    let grad_out = mkrand(&[bs, heads, seq, seq], 0.003, device);

    // ---------- Correctness: forward ----------
    let fused = scores.softmax_causal_scaled(seq, seq, offset, scale);

    // Reference path: scale → + mask → softmax
    let scaled = scores.mul_scalar(scale);
    let mask = build_causal_mask(seq, seq, offset, device);
    let masked = scaled.add(&mask).unwrap();
    let ref_out = masked.softmax(-1);

    let max_fwd = max_abs_diff(&fused.to_vec(), &ref_out.to_vec());
    println!("forward  max_abs_diff (fused vs ref) = {max_fwd:.4e}");
    assert!(max_fwd < 5e-4, "forward correctness fail");

    // ---------- Correctness: backward ----------
    let fused_bwd = fused.softmax_causal_scaled_bwd(&grad_out, seq, scale);

    // Reference backward: standard softmax bwd on `ref_out` gives d(scaled),
    // then d(scores) = d(scaled) * scale.
    let ref_bwd = ref_softmax_bwd(&ref_out, &grad_out, seq).mul_scalar(scale);
    let max_bwd = max_abs_diff(&fused_bwd.to_vec(), &ref_bwd.to_vec());
    println!("backward max_abs_diff (fused vs ref) = {max_bwd:.4e}");
    assert!(max_bwd < 5e-4, "backward correctness fail");
    println!("PASS correctness\n");

    // ---------- Perf: forward ----------
    for _ in 0..5 {
        let _ = scores.softmax_causal_scaled(seq, seq, offset, scale);
    }
    cuda_sync();
    let n_iter = 100;
    let t = Instant::now();
    for _ in 0..n_iter {
        let _ = scores.softmax_causal_scaled(seq, seq, offset, scale);
    }
    cuda_sync();
    println!(
        "fused forward  [{bs},{heads},{seq},{seq}] {:>7.1} µs/call",
        t.elapsed().as_micros() as f64 / n_iter as f64
    );

    cuda_sync();
    let t = Instant::now();
    for _ in 0..n_iter {
        let scaled = scores.mul_scalar(scale);
        let masked = scaled.add(&mask).unwrap();
        let _ = masked.softmax(-1);
    }
    cuda_sync();
    println!(
        "ref 3-op chain (scale+add+softmax)        {:>7.1} µs/call",
        t.elapsed().as_micros() as f64 / n_iter as f64
    );

    // ---------- Perf: backward ----------
    for _ in 0..5 {
        let _ = fused.softmax_causal_scaled_bwd(&grad_out, seq, scale);
    }
    cuda_sync();
    let t = Instant::now();
    for _ in 0..n_iter {
        let _ = fused.softmax_causal_scaled_bwd(&grad_out, seq, scale);
    }
    cuda_sync();
    println!(
        "\nfused backward  {:>7.1} µs/call",
        t.elapsed().as_micros() as f64 / n_iter as f64
    );
}

fn build_causal_mask(tq: usize, tk: usize, offset: usize, dev: Device) -> Tensor<f32> {
    let mut data = vec![0.0f32; tq * tk];
    for i in 0..tq {
        let pos = offset + i;
        for j in 0..tk {
            if j > pos {
                data[i * tk + j] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_vec(data, &[1, 1, tq, tk])
        .unwrap()
        .to_device(dev)
        .unwrap()
}

// Standard softmax backward wrt softmax input, given saved softmax output p:
//   grad_in[r, j] = p[r, j] * (grad_out[r, j] - Σ_k p[r, k] * grad_out[r, k])
fn ref_softmax_bwd(p: &Tensor<f32>, grad_out: &Tensor<f32>, tk: usize) -> Tensor<f32> {
    let p_cpu = p.to_device(Device::Cpu).unwrap().to_vec();
    let g_cpu = grad_out.to_device(Device::Cpu).unwrap().to_vec();
    let total = p_cpu.len();
    let num_rows = total / tk;
    let mut out = vec![0.0f32; total];
    for r in 0..num_rows {
        let base = r * tk;
        let mut dot = 0.0f32;
        for j in 0..tk {
            dot += p_cpu[base + j] * g_cpu[base + j];
        }
        for j in 0..tk {
            out[base + j] = p_cpu[base + j] * (g_cpu[base + j] - dot);
        }
    }
    Tensor::from_vec(out, p.shape())
        .unwrap()
        .to_device(p.device())
        .unwrap()
}

fn mkrand(shape: &[usize], seed: f32, dev: Device) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|i| seed + ((i % 23) as f32) * 0.005).collect();
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
