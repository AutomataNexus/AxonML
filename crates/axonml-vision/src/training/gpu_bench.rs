//! GPU Benchmarks — Measure CUDA-accelerated inference throughput
//!
//! # File
//! `crates/axonml-vision/src/training/gpu_bench.rs`
//!
//! # Author
//! Andrew Jewell Sr - AutomataNexus
//!
//! # Updated
//! March 8, 2026
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

#[cfg(test)]
mod tests {
    use axonml_autograd::Variable;
    use axonml_nn::Module;
    use axonml_optim::Optimizer;
    use axonml_tensor::Tensor;
    use std::time::Instant;

    // =========================================================================
    // Helios GPU Benchmarks
    // =========================================================================

    #[test]
    fn gpu_bench_helios_nano_64() {
        use crate::models::helios::Helios;

        let model = Helios::nano(80);
        let input = Variable::new(
            Tensor::from_vec(vec![0.5f32; 3 * 64 * 64], &[1, 3, 64, 64]).unwrap(),
            false,
        );

        println!("\n--- Helios-Nano 64x64 ---");

        // Warmup
        let _ = model.forward_train(&input);

        let start = Instant::now();
        let iters = 5;
        for _ in 0..iters {
            let _ = model.forward_train(&input);
        }
        let elapsed = start.elapsed();
        let ms = elapsed.as_secs_f64() * 1000.0 / iters as f64;
        println!("  forward_train: {ms:.1}ms ({:.1} FPS)", 1000.0 / ms);

        // Detect
        let start = Instant::now();
        for _ in 0..iters {
            let _ = model.detect(&input, 0.25, 0.45);
        }
        let elapsed = start.elapsed();
        let ms = elapsed.as_secs_f64() * 1000.0 / iters as f64;
        println!("  detect (NMS):  {ms:.1}ms ({:.1} FPS)", 1000.0 / ms);
    }

    #[test]
    fn gpu_bench_helios_nano_320() {
        use crate::models::helios::Helios;

        let model = Helios::nano(80);
        let input = Variable::new(
            Tensor::from_vec(vec![0.5f32; 3 * 320 * 320], &[1, 3, 320, 320]).unwrap(),
            false,
        );

        println!("\n--- Helios-Nano 320x320 ---");

        // Warmup
        let _ = model.forward_train(&input);

        let start = Instant::now();
        let iters = 3;
        for _ in 0..iters {
            let _ = model.forward_train(&input);
        }
        let elapsed = start.elapsed();
        let ms = elapsed.as_secs_f64() * 1000.0 / iters as f64;
        println!("  forward_train: {ms:.1}ms ({:.1} FPS)", 1000.0 / ms);
    }

    #[test]
    fn gpu_bench_helios_nano_640() {
        use crate::models::helios::Helios;

        let model = Helios::nano(80);
        let input = Variable::new(
            Tensor::from_vec(vec![0.5f32; 3 * 640 * 640], &[1, 3, 640, 640]).unwrap(),
            false,
        );

        println!("\n--- Helios-Nano 640x640 (standard detection size) ---");

        // Warmup
        let _ = model.forward_train(&input);

        let start = Instant::now();
        let iters = 2;
        for _ in 0..iters {
            let _ = model.forward_train(&input);
        }
        let elapsed = start.elapsed();
        let ms = elapsed.as_secs_f64() * 1000.0 / iters as f64;
        println!("  forward_train: {ms:.1}ms ({:.1} FPS)", 1000.0 / ms);
    }

    #[test]
    fn gpu_bench_helios_small_320() {
        use crate::models::helios::Helios;

        let model = Helios::small(80);
        let input = Variable::new(
            Tensor::from_vec(vec![0.5f32; 3 * 320 * 320], &[1, 3, 320, 320]).unwrap(),
            false,
        );

        println!("\n--- Helios-Small 320x320 ---");

        let _ = model.forward_train(&input);

        let start = Instant::now();
        let iters = 2;
        for _ in 0..iters {
            let _ = model.forward_train(&input);
        }
        let elapsed = start.elapsed();
        let ms = elapsed.as_secs_f64() * 1000.0 / iters as f64;
        println!("  forward_train: {ms:.1}ms ({:.1} FPS)", 1000.0 / ms);
    }

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
    // Training step GPU benchmark
    // =========================================================================

    #[test]
    fn gpu_bench_helios_train_step() {
        use crate::models::helios::{Helios, HeliosLoss};

        let model = Helios::nano(2);
        let mut optimizer = axonml_optim::Adam::new(model.parameters(), 1e-3);
        let loss_fn = HeliosLoss::new(2, 16);

        let gt_boxes = vec![vec![[16.0, 16.0, 80.0, 80.0]]];
        let gt_classes = vec![vec![0usize]];

        println!("\n--- Helios-Nano Training Step (128x128) ---");

        // Warmup
        let input = Variable::new(
            Tensor::from_vec(vec![0.5f32; 3 * 128 * 128], &[1, 3, 128, 128]).unwrap(),
            false,
        );
        optimizer.zero_grad();
        let out = model.forward_train(&input);
        let (loss, _, _, _) = loss_fn.compute(&out, &gt_boxes, &gt_classes, 2);
        loss.backward();
        optimizer.step();

        let start = Instant::now();
        let iters = 3;
        for _ in 0..iters {
            let input = Variable::new(
                Tensor::from_vec(vec![0.5f32; 3 * 128 * 128], &[1, 3, 128, 128]).unwrap(),
                false,
            );
            optimizer.zero_grad();
            let out = model.forward_train(&input);
            let (loss, _, _, _) = loss_fn.compute(&out, &gt_boxes, &gt_classes, 2);
            loss.backward();
            optimizer.step();
        }
        let elapsed = start.elapsed();
        let ms = elapsed.as_secs_f64() * 1000.0 / iters as f64;
        println!("  train_step: {ms:.1}ms ({:.1} steps/s)", 1000.0 / ms);
    }

    // =========================================================================
    // Conv2d Isolation Benchmark (CUDA im2col+GEMM vs CPU)
    // =========================================================================

    #[test]
    fn gpu_bench_conv2d_isolated() {
        use axonml_nn::{Conv2d, Module};

        println!("\n--- Conv2d CUDA Benchmark (isolated layers) ---");

        // Realistic layer sizes from Helios backbone
        let configs: &[(usize, usize, usize, usize, usize)] = &[
            // (in_ch, out_ch, spatial, kernel, label_id)
            (3, 16, 320, 3, 0),    // stem: 3→16, 320x320, 3x3 stride 2
            (16, 32, 160, 3, 1),   // stage1: 16→32, 160x160
            (32, 64, 80, 3, 2),    // stage2: 32→64, 80x80
            (64, 128, 40, 3, 3),   // stage3: 64→128, 40x40
            (128, 256, 20, 3, 4),  // stage4: 128→256, 20x20
        ];
        let labels = ["stem 3→16 320²", "s1 16→32 160²", "s2 32→64 80²", "s3 64→128 40²", "s4 128→256 20²"];

        for &(ic, oc, spatial, ks, idx) in configs {
            let conv = Conv2d::with_options(ic, oc, (ks, ks), (1, 1), (ks / 2, ks / 2), true);
            let input = Variable::new(
                Tensor::from_vec(vec![0.5f32; ic * spatial * spatial], &[1, ic, spatial, spatial]).unwrap(),
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
