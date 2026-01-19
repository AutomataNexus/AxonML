//! Simple Training Example
//!
//! Demonstrates a complete training loop with Axonml.

use axonml::prelude::*;

fn main() {
    println!("=== Axonml ML Framework - Simple Training Example ===\n");

    // Print version and features
    println!("Version: {}", axonml::version());
    println!("Features: {}\n", axonml::features());

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

    // 2. Create a simple MLP model
    println!("2. Creating MLP model (2 -> 4 -> 1)...");
    let linear1 = Linear::new(2, 4);
    let linear2 = Linear::new(4, 1);

    println!("   Layer 1: Linear(2, 4)");
    println!("   Layer 2: Linear(4, 1)\n");

    // 3. Create optimizer
    println!("3. Creating Adam optimizer (lr=0.1)...");
    let params = [linear1.parameters(), linear2.parameters()].concat();
    let mut optimizer = Adam::new(params, 0.1);
    println!("   Optimizer created!\n");

    // 4. Training loop
    println!("4. Training for 1000 epochs...");
    let epochs = 1000;

    for epoch in 0..epochs {
        let mut total_loss = 0.0;

        for (input, &target) in inputs.iter().zip(targets.iter()) {
            // Create input tensor
            let x = Variable::new(Tensor::from_vec(input.clone(), &[1, 2]).unwrap(), true);

            // Forward pass
            let h = linear1.forward(&x);
            let h = h.sigmoid();
            let output = linear2.forward(&h);
            let output = output.sigmoid();

            // Create target tensor
            let y = Variable::new(Tensor::from_vec(vec![target], &[1, 1]).unwrap(), false);

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

    // 5. Test the trained model
    println!("\n5. Testing trained model...");
    for (input, &expected) in inputs.iter().zip(targets.iter()) {
        let x = Variable::new(Tensor::from_vec(input.clone(), &[1, 2]).unwrap(), false);

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
