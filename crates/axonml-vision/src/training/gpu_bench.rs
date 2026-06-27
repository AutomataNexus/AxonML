//! GPU Benchmarks — CUDA Throughput Measurement Suite
//!
//! Test-only GPU benchmarks gated by `#[cfg(test)]`. Each test prints
//! warmup-excluded latency and FPS for a representative model / resolution
//! configuration so CUDA-path performance can be tracked over time. Coverage:
//! ResNet18 forward at 224x224 and an isolated Conv2d sweep over five
//! representative backbone shapes (stem, s1, s2, s3, s4) to compare CUDA
//! im2col+GEMM against CPU.
//!
//! # File
//! `crates/axonml-vision/src/training/gpu_bench.rs`
//!
//! # Author
//! Andrew Jewell Sr. — AutomataNexus LLC
//! ORCID: 0009-0005-2158-7060
//!
//! # Updated
//! April 16, 2026 11:15 PM EST
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

#[cfg(test)]
mod tests {
    use axonml_autograd::Variable;
    use axonml_nn::Module;
    use axonml_tensor::Tensor;
    use std::time::Instant;

    // =========================================================================
    // ResNet18 GPU Benchmark
    // =========================================================================

    #[test]
    fn gpu_bench_resnet18_224() {
        use crate::models::resnet::ResNet;

        let model = ResNet::resnet18(1000);
        let input = Variable::new(
            Tensor::from_vec(vec![0.5f32; 3 * 224 * 224], &[1, 3, 224, 224]).unwrap(),
            false,
        );

        println!("\n--- ResNet18 224x224 ---");

        let _ = model.forward(&input);

        let start = Instant::now();
        let iters = 3;
        for _ in 0..iters {
            let _ = model.forward(&input);
        }
        let elapsed = start.elapsed();
        let ms = elapsed.as_secs_f64() * 1000.0 / iters as f64;
        println!("  forward: {ms:.1}ms ({:.1} FPS)", 1000.0 / ms);
    }

    // =========================================================================
    // Conv2d Isolation Benchmark (CUDA im2col+GEMM vs CPU)
    // =========================================================================

    #[test]
    fn gpu_bench_conv2d_isolated() {
        use axonml_nn::{Conv2d, Module};

        println!("\n--- Conv2d CUDA Benchmark (isolated layers) ---");

        // Realistic layer sizes from a detector backbone
        let configs: &[(usize, usize, usize, usize, usize)] = &[
            // (in_ch, out_ch, spatial, kernel, label_id)
            (3, 16, 320, 3, 0),   // stem: 3→16, 320x320, 3x3 stride 2
            (16, 32, 160, 3, 1),  // stage1: 16→32, 160x160
            (32, 64, 80, 3, 2),   // stage2: 32→64, 80x80
            (64, 128, 40, 3, 3),  // stage3: 64→128, 40x40
            (128, 256, 20, 3, 4), // stage4: 128→256, 20x20
        ];
        let labels = [
            "stem 3→16 320²",
            "s1 16→32 160²",
            "s2 32→64 80²",
            "s3 64→128 40²",
            "s4 128→256 20²",
        ];

        for &(ic, oc, spatial, ks, idx) in configs {
            let conv = Conv2d::with_options(ic, oc, (ks, ks), (1, 1), (ks / 2, ks / 2), true);
            let input = Variable::new(
                Tensor::from_vec(
                    vec![0.5f32; ic * spatial * spatial],
                    &[1, ic, spatial, spatial],
                )
                .unwrap(),
                false,
            );

            // Warmup
            let _ = conv.forward(&input);

            let start = Instant::now();
            let iters = 3;
            for _ in 0..iters {
                let _ = conv.forward(&input);
            }
            let elapsed = start.elapsed();
            let ms = elapsed.as_secs_f64() * 1000.0 / iters as f64;
            println!("  {}: {ms:.1}ms", labels[idx]);
        }
    }
}
