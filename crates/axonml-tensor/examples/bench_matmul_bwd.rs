//! Verify correctness + perf of the new 3D batched matmul (GPU vs CPU).
//!
//! Previously the 3D batched matmul on GPU did full D2H, per-batch H2D/D2H,
//! and CPU-side reassembly — ~313 ms/call on Qwen3-0.6B MatMulBackward.
//! Now replaced with `gemm_strided_batched_f32` in one on-device call.
//!
//! This bench verifies the GPU result matches the CPU reference matmul
//! exactly (within fp32 tolerance) AND measures the new per-call cost.

use std::time::Instant;

use axonml_core::Device;
use axonml_tensor::Tensor;

fn main() {
    let bs = 4;
    let seq = 512;
    let hidden = 1024;
    let inter = 3072;

    let device = Device::Cuda(0);

    // ---------- 3D × 3D batched matmul (the actual MatMulBackward path) ----------
    //   [bs, seq, hidden] @ [bs, hidden, inter] → [bs, seq, inter]
    let lhs_3d = mkrand(&[bs, seq, hidden], 0.11, device);
    let rhs_3d = mkrand(&[bs, hidden, inter], 0.07, device);
    let lhs_cpu = lhs_3d.to_device(Device::Cpu).unwrap();
    let rhs_cpu = rhs_3d.to_device(Device::Cpu).unwrap();

    println!(
        "=== correctness: [bs={bs}, seq={seq}, hidden={hidden}] @ [bs, hidden, inter={inter}] ==="
    );
    let gpu_out = lhs_3d.matmul(&rhs_3d).unwrap();
    let cpu_out = lhs_cpu.matmul(&rhs_cpu).unwrap();
    let max_abs = max_abs_diff(&gpu_out.to_vec(), &cpu_out.to_vec());
    println!("max_abs_diff (GPU vs CPU)   = {max_abs:.4e}");
    let ok = max_abs < 1e-2;
    println!(
        "{}",
        if ok {
            "PASS"
        } else {
            "FAIL — GPU batched matmul disagrees with CPU"
        }
    );

    // ---------- 3D × 3D batched matmul perf ----------
    let n_iter = 30;
    // warm
    for _ in 0..3 {
        let _ = lhs_3d.matmul(&rhs_3d).unwrap();
    }
    let t0 = Instant::now();
    for _ in 0..n_iter {
        let out = lhs_3d.matmul(&rhs_3d).unwrap();
        std::hint::black_box(out);
    }
    let _ = lhs_3d.matmul(&rhs_3d).unwrap().to_vec();
    let dt = t0.elapsed();
    println!(
        "3D batched matmul           {:>7.2} µs/call   [{:.1}ms total, {} iters]",
        dt.as_micros() as f64 / (n_iter + 1) as f64,
        dt.as_secs_f64() * 1000.0,
        n_iter,
    );

    // ---------- 4D×4D attention matmul (Q@K^T shape) ----------
    let heads = 16;
    let head_dim = 128;
    let s = 512;
    let q_4d = mkrand(&[bs, heads, s, head_dim], 0.13, device);
    let k_4d = mkrand(&[bs, heads, s, head_dim], 0.09, device);
    // Q @ K^T: [bs,heads,seq,hd] @ [bs,heads,hd,seq] (transpose last2)
    for _ in 0..3 {
        let kt = k_4d.transpose(2, 3).unwrap();
        let _ = q_4d.matmul(&kt).unwrap();
    }
    let t1 = Instant::now();
    for _ in 0..n_iter {
        let kt = k_4d.transpose(2, 3).unwrap();
        let out = q_4d.matmul(&kt).unwrap();
        std::hint::black_box(out);
    }
    let _ = q_4d
        .matmul(&k_4d.transpose(2, 3).unwrap())
        .unwrap()
        .to_vec();
    let dt = t1.elapsed();
    println!(
        "Q @ K^T 4D [{bs},{heads},{s},{head_dim}] × [{bs},{heads},{head_dim},{s}]   {:>7.2} µs/call",
        dt.as_micros() as f64 / (n_iter + 1) as f64
    );

    // ---------- 4D×4D attention backward shapes ----------
    // grad_lhs = go[4,16,512,512] @ rhs_t[4,16,512,128] → [4,16,512,128]
    let go_attn = mkrand(&[bs, heads, s, s], 0.01, device);
    let rhs_attn = mkrand(&[bs, heads, s, head_dim], 0.12, device);
    for _ in 0..3 {
        let _ = go_attn.matmul(&rhs_attn).unwrap();
    }
    let t2 = Instant::now();
    for _ in 0..n_iter {
        let out = go_attn.matmul(&rhs_attn).unwrap();
        std::hint::black_box(out);
    }
    let _ = go_attn.matmul(&rhs_attn).unwrap().to_vec();
    println!(
        "attn_bwd go@rhs 4D [{bs},{heads},{s},{s}] @ [{bs},{heads},{s},{head_dim}]  {:>7.2} µs/call",
        t2.elapsed().as_micros() as f64 / (n_iter + 1) as f64
    );

    // ---------- 4D × non-contiguous 4D — the REAL attention backward shape ----------
    // saved_lhs comes from softmax forward (contig). lhs_t = saved_lhs.transpose(2,3) is a VIEW.
    // Then we do lt.matmul(&go) — non-contig 4D × contig 4D.
    let saved_lhs = mkrand(&[bs, heads, s, s], 0.1, device);
    let go2 = mkrand(&[bs, heads, s, head_dim], 0.01, device);

    // Force everything to stream first
    let _ = saved_lhs.to_vec();
    let _ = go2.to_vec();

    for _ in 0..3 {
        let lt = saved_lhs.transpose(2, 3).unwrap();
        let _ = lt.matmul(&go2).unwrap();
    }

    // Single-call timing with explicit sync
    use axonml_core::backends::cuda::cuda_sync;
    cuda_sync();
    let t_single = Instant::now();
    let lt = saved_lhs.transpose(2, 3).unwrap();
    let out = lt.matmul(&go2).unwrap();
    cuda_sync();
    let single_us = t_single.elapsed().as_micros();
    std::hint::black_box(out);
    println!("SINGLE attn-bwd lhs_t @ go (with sync)   {} µs", single_us);

    cuda_sync();
    let t_many = Instant::now();
    for _ in 0..n_iter {
        let lt = saved_lhs.transpose(2, 3).unwrap();
        let out = lt.matmul(&go2).unwrap();
        std::hint::black_box(out);
    }
    cuda_sync();
    let many_us = t_many.elapsed().as_micros();
    println!(
        "AVERAGED attn-bwd lhs_t @ go (with sync) {} µs/call over {} iter",
        many_us / n_iter as u128,
        n_iter
    );
}

fn mkrand(shape: &[usize], seed: f32, dev: Device) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|i| seed + ((i % 17) as f32) * 0.003).collect();
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
