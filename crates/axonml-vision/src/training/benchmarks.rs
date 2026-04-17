//! Vision Inference Benchmarks — Per-Model Latency / Throughput Sweeps
//!
//! Test-only inference benchmarks for the axonml-vision model zoo. The module
//! is gated by `#[cfg(test)]` and exposes one `#[test]` per model that
//! constructs the model, warms it up, and times forward passes across a few
//! batch sizes. Helpers `dummy_input` build constant input tensors,
//! `bench_forward` returns `(latency_ms, images_per_sec)` for any `Module`,
//! and `print_bench` emits a one-line summary. Coverage includes LeNet,
//! SimpleCNN, MLP, ResNet18, VGG16, Vision Transformer, NanoDet, BlazeFace,
//! Nexus and Phantom (each with its custom forward), Mnemosyne identity,
//! a LeNet training-step throughput test (forward + backward + Adam step),
//! a parameter-count + memory profile across the zoo plus Helios variants,
//! and a Helios-Nano end-to-end detection latency benchmark.
//!
//! # File
//! `crates/axonml-vision/src/training/benchmarks.rs`
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
    use std::time::Instant;

    use axonml_autograd::Variable;
    use axonml_nn::Module;
    use axonml_tensor::Tensor;

    // =========================================================================
    // Helpers
    // =========================================================================

    /// Create a dummy input batch of given shape.
    fn dummy_input(shape: &[usize]) -> Variable {
        let size: usize = shape.iter().product();
        Variable::new(Tensor::from_vec(vec![0.5f32; size], shape).unwrap(), false)
    }

    /// Benchmark a model's forward pass.
    /// Returns (latency_ms, images_per_sec).
    fn bench_forward<M: Module>(
        model: &M,
        input: &Variable,
        warmup: usize,
        iters: usize,
    ) -> (f64, f64) {
        let batch_size = input.shape()[0];

        // Warmup
        for _ in 0..warmup {
            let _ = model.forward(input);
        }

        // Timed iterations
        let start = Instant::now();
        for _ in 0..iters {
            let _ = model.forward(input);
        }
        let elapsed = start.elapsed();

        let total_ms = elapsed.as_secs_f64() * 1000.0;
        let latency_ms = total_ms / iters as f64;
        let images_per_sec = (batch_size * iters) as f64 / elapsed.as_secs_f64();

        (latency_ms, images_per_sec)
    }

    fn print_bench(model_name: &str, batch: usize, latency_ms: f64, ips: f64) {
        println!(
            "  {model_name:20} | batch={batch:3} | latency={latency_ms:8.2}ms | {ips:10.1} img/s"
        );
    }

    // =========================================================================
    // LeNet
    // =========================================================================

    #[test]
    fn benchmark_lenet() {
        use crate::models::lenet::LeNet;

        let model = LeNet::new();
        println!("\n--- LeNet (MNIST 28x28) ---");

        for &batch in &[1, 32, 128] {
            let input = dummy_input(&[batch, 1, 28, 28]);
            let (lat, ips) = bench_forward(&model, &input, 2, 10);
            print_bench("LeNet", batch, lat, ips);
        }
    }

    // =========================================================================
    // SimpleCNN
    // =========================================================================

    #[test]
    fn benchmark_simplecnn() {
        use crate::models::lenet::SimpleCNN;

        let model = SimpleCNN::for_mnist();
        println!("\n--- SimpleCNN (MNIST 28x28) ---");

        for &batch in &[1, 32] {
            let input = dummy_input(&[batch, 1, 28, 28]);
            let (lat, ips) = bench_forward(&model, &input, 2, 10);
            print_bench("SimpleCNN", batch, lat, ips);
        }
    }

    // =========================================================================
    // MLP
    // =========================================================================

    #[test]
    fn benchmark_mlp() {
        use crate::models::lenet::MLP;

        let model = MLP::for_mnist();
        println!("\n--- MLP (MNIST flat 784) ---");

        for &batch in &[1, 32, 128] {
            let input = dummy_input(&[batch, 784]);
            let (lat, ips) = bench_forward(&model, &input, 2, 10);
            print_bench("MLP", batch, lat, ips);
        }
    }

    // =========================================================================
    // ResNet18
    // =========================================================================

    #[test]
    fn benchmark_resnet18() {
        use crate::models::resnet::ResNet;

        let model = ResNet::resnet18(10);
        println!("\n--- ResNet18 (CIFAR 32x32) ---");

        for &batch in &[1, 8] {
            let input = dummy_input(&[batch, 3, 32, 32]);
            let (lat, ips) = bench_forward(&model, &input, 1, 5);
            print_bench("ResNet18", batch, lat, ips);
        }
    }

    // =========================================================================
    // VGG16
    // =========================================================================

    #[test]
    fn benchmark_vgg16() {
        use crate::models::vgg::VGG;

        let model = VGG::vgg16(10);
        println!("\n--- VGG16 (224x224) ---");

        // VGG classifier expects 7x7 feature maps, requiring 224x224 input
        // Single iteration only — VGG16 is ~40s/img on CPU (133M params)
        let input = dummy_input(&[1, 3, 224, 224]);
        let (lat, ips) = bench_forward(&model, &input, 0, 1);
        print_bench("VGG16", 1, lat, ips);
    }

    // =========================================================================
    // ViT
    // =========================================================================

    #[test]
    fn benchmark_vit() {
        use crate::models::transformer::VisionTransformer;

        // Small ViT for CIFAR
        let model = VisionTransformer::new(32, 8, 3, 10, 64, 2, 4, 128, 0.0);
        println!("\n--- ViT-Small (CIFAR 32x32, patch=8, d=64) ---");

        for &batch in &[1, 8] {
            let input = dummy_input(&[batch, 3, 32, 32]);
            let (lat, ips) = bench_forward(&model, &input, 2, 5);
            print_bench("ViT-Small", batch, lat, ips);
        }
    }

    // =========================================================================
    // NanoDet
    // =========================================================================

    #[test]
    fn benchmark_nanodet() {
        use crate::models::nanodet::NanoDet;

        let model = NanoDet::new(1);
        println!("\n--- NanoDet (64x64, 1 class) ---");

        let input = dummy_input(&[1, 3, 64, 64]);
        let (lat, ips) = bench_forward(&model, &input, 2, 5);
        print_bench("NanoDet-64", 1, lat, ips);

        let input = dummy_input(&[1, 3, 128, 128]);
        let (lat, ips) = bench_forward(&model, &input, 1, 3);
        print_bench("NanoDet-128", 1, lat, ips);
    }

    // =========================================================================
    // BlazeFace
    // =========================================================================

    #[test]
    fn benchmark_blazeface() {
        use crate::models::blazeface::BlazeFace;

        let model = BlazeFace::new();
        println!("\n--- BlazeFace (128x128) ---");

        let input = dummy_input(&[1, 3, 128, 128]);
        let (lat, ips) = bench_forward(&model, &input, 2, 5);
        print_bench("BlazeFace", 1, lat, ips);
    }

    // =========================================================================
    // Nexus (custom forward — not Module trait)
    // =========================================================================

    #[test]
    fn benchmark_nexus() {
        use crate::models::nexus::Nexus;

        let mut model = Nexus::new();
        model.eval();
        println!("\n--- Nexus (64x64) ---");

        let input = dummy_input(&[1, 3, 64, 64]);

        // Warmup
        let _ = model.detect(&input);

        let iters = 3;
        let start = Instant::now();
        for _ in 0..iters {
            let _ = model.detect(&input);
        }
        let elapsed = start.elapsed();
        let lat = elapsed.as_secs_f64() * 1000.0 / iters as f64;
        let ips = iters as f64 / elapsed.as_secs_f64();
        print_bench("Nexus-64", 1, lat, ips);
    }

    // =========================================================================
    // Phantom (custom forward — not Module trait)
    // =========================================================================

    #[test]
    fn benchmark_phantom() {
        use crate::models::phantom::Phantom;

        let mut model = Phantom::new();
        model.eval();
        println!("\n--- Phantom (64x64) ---");

        let input = dummy_input(&[1, 3, 64, 64]);

        // Warmup
        for _ in 0..2 {
            let _ = model.detect_frame(&input);
        }

        let iters = 5;
        let start = Instant::now();
        for _ in 0..iters {
            let _ = model.detect_frame(&input);
        }
        let elapsed = start.elapsed();
        let lat = elapsed.as_secs_f64() * 1000.0 / iters as f64;
        let ips = iters as f64 / elapsed.as_secs_f64();
        print_bench("Phantom-64", 1, lat, ips);
    }

    // =========================================================================
    // Mnemosyne
    // =========================================================================

    #[test]
    fn benchmark_mnemosyne() {
        use crate::models::biometric::MnemosyneIdentity;

        let model = MnemosyneIdentity::new();
        println!("\n--- Mnemosyne (face 32x32) ---");

        for &batch in &[1, 8] {
            let input = dummy_input(&[batch, 3, 32, 32]);
            let (lat, ips) = bench_forward(&model, &input, 2, 10);
            print_bench("Mnemosyne", batch, lat, ips);
        }
    }

    // =========================================================================
    // Training Throughput (LeNet)
    // =========================================================================

    #[test]
    fn benchmark_training_step() {
        use crate::models::lenet::LeNet;
        use axonml_nn::CrossEntropyLoss;
        use axonml_optim::{Adam, Optimizer};

        let model = LeNet::new();
        let mut optimizer = Adam::new(model.parameters(), 0.001);
        let loss_fn = CrossEntropyLoss::new();
        let batch_size = 16;

        let input = dummy_input(&[batch_size, 1, 28, 28]);
        let targets = Variable::new(
            Tensor::from_vec(
                (0..batch_size).map(|i| (i % 10) as f32).collect(),
                &[batch_size],
            )
            .unwrap(),
            false,
        );

        println!("\n--- Training Step (LeNet, batch=16) ---");

        // Warmup
        for _ in 0..2 {
            optimizer.zero_grad();
            let logits = model.forward(&input);
            let loss = loss_fn.compute(&logits, &targets);
            loss.backward();
            optimizer.step();
        }

        // Timed
        let iters = 5;
        let start = Instant::now();
        for _ in 0..iters {
            optimizer.zero_grad();
            let logits = model.forward(&input);
            let loss = loss_fn.compute(&logits, &targets);
            loss.backward();
            optimizer.step();
        }
        let elapsed = start.elapsed();
        let step_ms = elapsed.as_secs_f64() * 1000.0 / iters as f64;
        let ips = (batch_size * iters) as f64 / elapsed.as_secs_f64();

        println!("  step_time={step_ms:.1}ms | {ips:.1} img/s (forward+backward+step)");
    }

    // =========================================================================
    // Memory Profiling
    // =========================================================================

    #[test]
    fn benchmark_param_counts() {
        use crate::models::biometric::MnemosyneIdentity;
        use crate::models::blazeface::BlazeFace;
        use crate::models::lenet::{LeNet, MLP, SimpleCNN};
        use crate::models::nanodet::NanoDet;
        use crate::models::resnet::ResNet;
        use crate::models::transformer::VisionTransformer;

        println!("\n--- Parameter Counts ---");

        let models: Vec<(&str, Vec<axonml_nn::Parameter>)> = vec![
            ("LeNet", LeNet::new().parameters()),
            ("SimpleCNN-MNIST", SimpleCNN::for_mnist().parameters()),
            ("MLP-MNIST", MLP::for_mnist().parameters()),
            ("ResNet18", ResNet::resnet18(10).parameters()),
            (
                "ViT-Small",
                VisionTransformer::new(32, 8, 3, 10, 64, 2, 4, 128, 0.0).parameters(),
            ),
            ("NanoDet-1", NanoDet::new(1).parameters()),
            ("BlazeFace", BlazeFace::new().parameters()),
            ("Mnemosyne", MnemosyneIdentity::new().parameters()),
        ];

        for (name, params) in &models {
            let total: usize = params
                .iter()
                .map(|p| p.variable().data().to_vec().len())
                .sum();
            let size_mb = total as f64 * 4.0 / 1_048_576.0;
            println!("  {name:20} | {total:>10} params | {size_mb:6.2} MB (f32)");
        }

        // Helios sizes
        use crate::models::helios::Helios;
        println!("\n--- Helios Variants ---");
        for (name, model) in [
            ("Helios-Nano", Helios::nano(80)),
            ("Helios-Small", Helios::small(80)),
        ] {
            let params = model.parameters();
            let total: usize = params
                .iter()
                .map(|p| p.variable().data().to_vec().len())
                .sum();
            let size_mb = total as f64 * 4.0 / 1_048_576.0;
            println!("  {name:20} | {total:>10} params | {size_mb:6.2} MB (f32)");
        }
    }

    // =========================================================================
    // Helios Inference Benchmark
    // =========================================================================

    #[test]
    fn benchmark_helios_nano_inference() {
        use crate::models::helios::Helios;

        let model = Helios::nano(80);
        let input = Variable::new(
            Tensor::from_vec(vec![0.5; 3 * 64 * 64], &[1, 3, 64, 64]).unwrap(),
            false,
        );

        println!("\n--- Helios-Nano Inference (64x64) ---");
        let warmup = 1;
        let iters = 3;

        for _ in 0..warmup {
            let _ = model.detect(&input, 0.5, 0.45);
        }

        let start = std::time::Instant::now();
        for _ in 0..iters {
            let _ = model.detect(&input, 0.5, 0.45);
        }
        let elapsed = start.elapsed();
        let latency_ms = elapsed.as_secs_f64() * 1000.0 / iters as f64;
        let fps = iters as f64 / elapsed.as_secs_f64();
        println!("  latency={latency_ms:.1}ms | {fps:.1} FPS");
    }
}
