//! Simple Training Example — XOR Problem with a Two-Layer MLP
//!
//! Minimal end-to-end training demonstration using the AxonML framework.
//! Prints the framework version and enabled features, constructs a two-layer
//! MLP (`Linear(2,4)` -> sigmoid -> `Linear(4,1)` -> sigmoid) to learn the
//! XOR function, trains for 1000 epochs with `Adam` optimizer (lr=0.1) and
//! manual MSE loss (`(output - target)^2`), logs loss every 200 epochs, and
//! evaluates all four XOR inputs with rounded prediction vs. expected output.
//!
//! # File
//! `crates/axonml/examples/simple_training.rs`
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

use axonml::prelude::*;
use axonml_core::Device;

// =============================================================================
// Main Entry Point
// =============================================================================

fn main() {
    println!("=== Axonml ML Framework - Simple Training Example ===\n");

    // Print version and features
    println!("Version: {}", axonml::version());
    println!("Features: {}\n", axonml::features());

    // Device selection (GPU when cuda feature enabled — see L02 and deficiency #1)
    #[cfg(feature = "cuda")]
    let device = {
        println!("CUDA enabled — targeting Device::Cuda(0) for params + data (required for real MatMulBackward on GPU)");
        Device::Cuda(0)
    };
    #[cfg(not(feature = "cuda"))]
    let device = Device::Cpu;

    // -------------------------------------------------------------------------
    // Dataset (XOR Problem)
    // -------------------------------------------------------------------------

    // 1. Create a simple dataset (XOR problem)
    println!("1. Creating XOR dataset...");
    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let targets = vec![0.0, 1.0, 1.0, 0.0]; // XOR outputs

    println!("   Inputs: {inputs:?}");
    println!("   Targets: {targets:?}\n");

    // -------------------------------------------------------------------------
    // Model and Optimizer
    // -------------------------------------------------------------------------

    // 2. Create a simple MLP model
    println!("2. Creating MLP model (2 -> 4 -> 1)...");
    let linear1 = Linear::new(2, 4);
    let linear2 = Linear::new(4, 1);

    println!("   Layer 1: Linear(2, 4)");
    println!("   Layer 2: Linear(4, 1)\n");

    // Move parameters (and later inputs) to the target device.
    // This is mandatory for --features cuda to exercise the real GPU MatMulBackward path (deficiency #1 repro).
    if device.is_gpu() {
        println!("   Moving Linear parameters to GPU...");
        let mut params_for_move = linear1.parameters();
        params_for_move.extend(linear2.parameters());
        for p in &params_for_move {
            p.to_device(device);
        }
    }

    // 3. Create optimizer
    println!("3. Creating Adam optimizer (lr=0.1)...");
    let params = [linear1.parameters(), linear2.parameters()].concat();
    let mut optimizer = Adam::new(params, 0.1);
    println!("   Optimizer created!\n");

    // -------------------------------------------------------------------------
    // Training Loop
    // -------------------------------------------------------------------------

    // 4. Training loop
    // For perf blowup repro (def-1) use short runs: EPOCHS=80 cargo run --release --features cuda --example simple_training
    let epochs: usize = std::env::var("EPOCHS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1000);
    println!("4. Training for {} epochs (override with EPOCHS=...)", epochs);

    for epoch in 0..epochs {
        let mut total_loss = 0.0;

        for (input, &target) in inputs.iter().zip(targets.iter()) {
            // Create input tensor (on device when GPU to drive GPU matmuls + MatMulBackward)
            let x_t = Tensor::from_vec(input.clone(), &[1, 2]).unwrap();
            let x_t = if device.is_gpu() { x_t.to_device(device).unwrap() } else { x_t };
            let x = Variable::new(x_t, true);

            // Forward pass
            let h = linear1.forward(&x);
            let h = h.sigmoid();
            let output = linear2.forward(&h);
            let output = output.sigmoid();

            // Create target tensor
            let y_t = Tensor::from_vec(vec![target], &[1, 1]).unwrap();
            let y_t = if device.is_gpu() { y_t.to_device(device).unwrap() } else { y_t };
            let y = Variable::new(y_t, false);

            // Compute MSE loss manually: (output - target)^2
            let diff = output.sub_var(&y);
            let loss = diff.mul_var(&diff);

            total_loss += loss.data().to_vec()[0];

            // Backward pass
            loss.backward();

            // Update weights
            optimizer.step();
            optimizer.zero_grad();
        }

        if epoch % 200 == 0 || epoch == epochs - 1 {
            println!("   Epoch {}: Loss = {:.6}", epoch, total_loss / 4.0);
        }
    }

    // -------------------------------------------------------------------------
    // Evaluation
    // -------------------------------------------------------------------------

    // 5. Test the trained model
    println!("\n5. Testing trained model...");
    for (input, &expected) in inputs.iter().zip(targets.iter()) {
        let x_t = Tensor::from_vec(input.clone(), &[1, 2]).unwrap();
        let x_t = if device.is_gpu() { x_t.to_device(device).unwrap() } else { x_t };
        let x = Variable::new(x_t, false);

        let h = linear1.forward(&x);
        let h = h.sigmoid();
        let output = linear2.forward(&h);
        let output = output.sigmoid();

        let pred = output.data().to_vec()[0];
        let rounded = if pred > 0.5 { 1.0 } else { 0.0 };

        println!(
            "   Input: {input:?} -> Predicted: {pred:.4} (rounded: {rounded}) | Expected: {expected}"
        );
    }

    println!("\n=== Training Complete! ===");
}
