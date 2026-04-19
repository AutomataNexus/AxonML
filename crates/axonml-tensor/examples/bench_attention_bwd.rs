//! Microbench — fused attention backward.
//!
//! Prior bug: fused_attention_bwd_cuda allocated 3 gradient buffers via
//!   `cuda.htod_copy(&vec![0.0f32; N]).ok()?`
//! That's 3 CPU zero-vec allocs + 3 full H2D copies per call. For Qwen3-0.6B
//! shapes: total_q = total_kv = ~8 MB each × 3 = ~24 MB H2D per layer.
//! With 30 layers per backward, this is ~720 MB/step of pure PCIe traffic
//! plus CPU memset churn — all wasted because GPU memset_zeros is ~100× faster.
//!
//! After fix: pool_alloc zeros on-GPU via cuMemsetD8Async, no CPU work.
//!
//! Shape here: Qwen3-0.6B bs=4, seq=512, heads=16, head_dim=64
//!   total_q = total_kv = 4*16*512*64 = 2,097,152 floats = 8 MB/buffer.

use std::time::Instant;

use axonml_core::Device;
use axonml_tensor::Tensor;

fn main() {
    let bs = 4;
    let heads = 16;
    let seq = 512;
    let head_dim = 64;
    let shape = [bs, heads, seq, head_dim];
    let numel = bs * heads * seq * head_dim;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let device = Device::Cuda(0);

    let mk = |seed: f32| {
        let data: Vec<f32> = (0..numel)
            .map(|i| seed + ((i % 17) as f32) * 0.01)
            .collect();
        Tensor::from_vec(data, &shape)
            .unwrap()
            .to_device(device.clone())
            .unwrap()
    };

    let q = mk(0.1);
    let k = mk(0.2);
    let v = mk(0.3);
    let o = mk(0.4);
    let go = mk(0.01);

    // Warm-up
    for _ in 0..5 {
        let _ = q
            .fused_attention_bwd_cuda(&k, &v, &o, &go, scale, true)
            .unwrap();
    }

    for n in [20, 100, 300] {
        let t0 = Instant::now();
        for _ in 0..n {
            let (gq, _gk, _gv) = q
                .fused_attention_bwd_cuda(&k, &v, &o, &go, scale, true)
                .unwrap();
            std::hint::black_box(gq);
        }
        // Stream drain
        let (drain, _, _) = q
            .fused_attention_bwd_cuda(&k, &v, &o, &go, scale, true)
            .unwrap();
        let _ = drain.to_vec();
        let dt = t0.elapsed();
        let per_call_us = dt.as_micros() as f64 / (n + 1) as f64;
        println!(
            "attention_bwd × {}  = {:.3} ms total, {:.1} µs/call",
            n,
            dt.as_secs_f64() * 1000.0,
            per_call_us
        );
    }

    let per_layer_us = 0.0; // filled below
    println!("\nNote: ~30 layers per training step — multiply µs/call by 30.");
    let _ = per_layer_us;
}
