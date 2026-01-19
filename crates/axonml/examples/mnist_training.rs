//! MNIST Training Example
//!
//! Demonstrates training on the synthetic MNIST dataset.

use axonml::prelude::*;

fn main() {
    println!("=== Axonml ML Framework - MNIST Training Example ===\n");

    // 1. Create synthetic MNIST dataset
    println!("1. Creating SyntheticMNIST dataset (500 samples)...");
    let train_dataset = SyntheticMNIST::new(500);
    let test_dataset = SyntheticMNIST::new(100);

    println!("   Training samples: {}", train_dataset.len());
    println!("   Test samples: {}\n", test_dataset.len());

    // 2. Create DataLoader
    println!("2. Creating DataLoader (batch_size=32)...");
    let train_loader = DataLoader::new(train_dataset, 32);
    println!("   Number of batches: {}\n", train_loader.len());

    // 3. Create a simple CNN model
    println!("3. Creating SimpleCNN model...");
    let model = SimpleCNN::new(1, 10);
    let params = model.parameters();
    println!("   Parameters: {}\n", params.len());

    // 4. Create optimizer
    println!("4. Creating SGD optimizer (lr=0.01)...");
    let mut optimizer = SGD::new(params, 0.01);

    // 5. Training loop
    println!("5. Training for 5 epochs...\n");
    let epochs = 5;

    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        let mut batch_count = 0;

        for batch in train_loader.iter() {
            // Reshape batch data for CNN: [batch, 1, 28, 28]
            let batch_size = batch.data.shape()[0];
            let input_data = batch.data.to_vec();
            let input = Variable::new(
                Tensor::from_vec(input_data, &[batch_size, 1, 28, 28]).unwrap(),
                true,
            );

            // Forward pass
            let output = model.forward(&input);

            // Simple loss: mean of output (for testing)
            let loss = output.mean();

            total_loss += loss.data().to_vec()[0];
            batch_count += 1;

            // Backward pass
            loss.backward();

            // Update weights
            optimizer.step();
            optimizer.zero_grad();
        }

        println!(
            "   Epoch {}: Avg Loss = {:.6}",
            epoch + 1,
            total_loss / batch_count as f32
        );
    }

    // 6. Test the model
    println!("\n6. Testing model...");
    let test_loader = DataLoader::new(test_dataset, 32);

    let mut correct = 0;
    let mut total = 0;

    for batch in test_loader.iter() {
        let batch_size = batch.data.shape()[0];
        let input_data = batch.data.to_vec();
        let input = Variable::new(
            Tensor::from_vec(input_data, &[batch_size, 1, 28, 28]).unwrap(),
            false,
        );

        let output = model.forward(&input);
        let output_data = output.data().to_vec();

        // Get predictions (argmax)
        let label_data = batch.targets.to_vec();

        for i in 0..batch_size {
            let start = i * 10;
            let end = start + 10;
            let sample_output = &output_data[start..end];

            // Find argmax
            let pred = sample_output
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            // Get true label (argmax of one-hot)
            let label_start = i * 10;
            let label_end = label_start + 10;
            let sample_label = &label_data[label_start..label_end];
            let true_label = sample_label
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            if pred == true_label {
                correct += 1;
            }
            total += 1;
        }
    }

    println!(
        "   Accuracy: {}/{} ({:.2}%)",
        correct,
        total,
        100.0 * correct as f32 / total as f32
    );

    println!("\n=== Training Complete! ===");
}
