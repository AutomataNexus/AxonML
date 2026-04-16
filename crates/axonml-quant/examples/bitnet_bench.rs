//! Standalone micro-bench for `axonml_quant::bitnet::matmul_i2s`.
//!
//! Replicates the shape profile of a BitNet b1.58-2B decode step and
//! measures wall time. Run in isolation from nexus-serve so rayon has
//! nothing to compete with. Lets us tell apart "the kernel is slow" from
//! "the server context is blocking rayon."
//!
//! Usage:
//!   cargo run --release --example bitnet_bench -p axonml-quant

use std::time::Instant;

use axonml_quant::bitnet::{I2S_BLOCK_SIZE, I2S_BYTES_PER_BLOCK, I2sBlock, matmul_i2s};

const SCALE: f32 = 0.05;

fn make_weights(n: usize, k: usize) -> Vec<u8> {
    assert!(k % I2S_BLOCK_SIZE == 0);
    let blocks = k / I2S_BLOCK_SIZE;
    let mut out = Vec::with_capacity(n * blocks * I2S_BYTES_PER_BLOCK);
    for j in 0..n {
        for b in 0..blocks {
            let mut vals = [0i8; I2S_BLOCK_SIZE];
            for t in 0..I2S_BLOCK_SIZE {
                // Deterministic ternary pattern so decode is stable.
                let seed = (j as u32).wrapping_mul(131) ^ (b as u32).wrapping_mul(7) ^ (t as u32);
                vals[t] = match seed % 3 {
                    0 => 0,
                    1 => 1,
                    _ => -1,
                };
            }
            let blk = I2sBlock::pack(&vals);
            out.extend_from_slice(&blk.to_bytes());
        }
    }
    out
}

fn make_activations(m: usize, k: usize) -> Vec<f32> {
    (0..m * k).map(|i| (i as f32 * 0.001).sin()).collect()
}

fn bench(m: usize, k: usize, n: usize, iters: usize) {
    let weights = make_weights(n, k);
    let acts = make_activations(m, k);
    let mut out = vec![0.0f32; m * n];

    // Warmup.
    matmul_i2s(&acts, m, k, &weights, n, SCALE, &mut out);

    let t = Instant::now();
    for _ in 0..iters {
        matmul_i2s(&acts, m, k, &weights, n, SCALE, &mut out);
    }
    let wall = t.elapsed().as_secs_f64();
    let per_call_us = wall * 1e6 / iters as f64;
    let gflops = (m as f64 * k as f64 * n as f64 * iters as f64) / wall / 1e9;
    println!(
        "m={m:>2} k={k:>5} n={n:>6} iters={iters:>4}  wall={wall:6.3}s  per_call={per_call_us:7.1}μs  ~{gflops:.1} Gops/s (ternary)"
    );
}

fn main() {
    let threads = rayon::current_num_threads();
    println!("rayon::current_num_threads() = {threads}");
    println!();

    // Decode-step shapes (m=1, BitNet-2B hidden=2560, ffn=6912, kv=640).
    println!("=== decode (m=1) — dominant during token generation ===");
    bench(1, 2560, 2560, 200); // q_proj / o_proj
    bench(1, 2560, 640, 200); // k_proj / v_proj (GQA)
    bench(1, 2560, 6912, 200); // gate / up
    bench(1, 6912, 2560, 200); // down

    println!();
    println!("=== prefill (m=24) — one-shot at start of generation ===");
    bench(24, 2560, 2560, 50);
    bench(24, 2560, 640, 50);
    bench(24, 2560, 6912, 50);
    bench(24, 6912, 2560, 50);

    println!();
    println!("One full decode step = 4×q/o + 2×(k+v) + 2×(gate+up) + 1×down per layer × 30 layers");
    println!("  ≈ 7 matmuls/layer × 30 layers = 210 matmuls/token");
}
