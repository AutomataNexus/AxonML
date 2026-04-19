//! Correctness: head-major RoPE forward + backward vs CPU reference.

use axonml_core::Device;
use axonml_tensor::Tensor;

fn main() {
    let bs = 4;
    let n_heads = 16;
    let seq = 128;
    let head_dim = 128;
    let theta = 1_000_000.0f32;
    let pos_start = 0usize;
    let device = Device::Cuda(0);

    let x_gpu = mkrand(&[bs, n_heads, seq, head_dim], 0.1, device);
    let x_cpu = x_gpu.to_device(Device::Cpu).unwrap();

    // Forward GPU vs CPU.
    let fwd_gpu = x_gpu.apply_rope_split_halves_bhsd(bs, n_heads, seq, head_dim, theta, pos_start);
    let fwd_cpu = x_cpu.apply_rope_split_halves_bhsd(bs, n_heads, seq, head_dim, theta, pos_start);
    let max_fwd = max_abs_diff(&fwd_gpu.to_vec(), &fwd_cpu.to_vec());
    println!("RoPE forward  max_abs_diff = {max_fwd:.4e}");
    assert!(max_fwd < 1e-4, "RoPE fwd correctness fail");

    // Backward GPU vs CPU.
    let go_gpu = mkrand(&[bs, n_heads, seq, head_dim], 0.01, device);
    let go_cpu = go_gpu.to_device(Device::Cpu).unwrap();
    let bwd_gpu = go_gpu.rope_split_halves_bhsd_bwd(bs, n_heads, seq, head_dim, theta, pos_start);
    let bwd_cpu = go_cpu.rope_split_halves_bhsd_bwd(bs, n_heads, seq, head_dim, theta, pos_start);
    let max_bwd = max_abs_diff(&bwd_gpu.to_vec(), &bwd_cpu.to_vec());
    println!("RoPE backward max_abs_diff = {max_bwd:.4e}");
    assert!(max_bwd < 1e-4, "RoPE bwd correctness fail");

    // repeat_kv correctness (bs=2, 8 kv-heads, n_rep=2).
    let kv_heads = 8;
    let n_rep = 2;
    let kv_gpu = mkrand(&[bs, kv_heads, seq, head_dim], 0.3, device);
    let kv_cpu = kv_gpu.to_device(Device::Cpu).unwrap();
    let rep_gpu = kv_gpu.repeat_kv(bs, kv_heads, n_rep, seq, head_dim);
    let rep_cpu = kv_cpu.repeat_kv(bs, kv_heads, n_rep, seq, head_dim);
    let max_rep = max_abs_diff(&rep_gpu.to_vec(), &rep_cpu.to_vec());
    println!("repeat_kv     max_abs_diff = {max_rep:.4e}");
    assert!(max_rep < 1e-6, "repeat_kv correctness fail");

    println!("\nPASS all correctness");
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
