//! End-to-end integration test for the entire Axonml framework.
//! This test simulates what a real user would do.

use axonml::prelude::*;

/// Test 1: Basic tensor operations work
#[test]
fn test_tensor_operations() {
    // Create tensors
    let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let b = Tensor::<f32>::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();

    // Basic operations
    let c = a.add(&b).unwrap();
    assert_eq!(c.to_vec(), vec![6.0, 8.0, 10.0, 12.0]);

    // Matrix multiplication
    let d = a.matmul(&b).unwrap();
    assert_eq!(d.shape(), &[2, 2]);

    println!("✓ Tensor operations work");
}

/// Test 2: Autograd works (forward + backward)
#[test]
fn test_autograd_training_step() {
    // Create variables with gradients
    let x = Variable::new(
        Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap(),
        true
    );
    let target = Variable::new(
        Tensor::from_vec(vec![2.0, 4.0, 6.0, 8.0], &[2, 2]).unwrap(),
        false
    );

    // Forward pass
    let pred = x.mul_scalar(2.0);

    // Loss
    let loss = pred.mse_loss(&target);

    // Backward pass
    loss.backward();

    // Check gradient exists
    assert!(x.grad().is_some());
    println!("✓ Autograd forward/backward works");
}

/// Test 3: Neural network module works
#[test]
fn test_neural_network() {
    // Build a simple MLP
    let model = Sequential::new()
        .add(Linear::new(4, 8))
        .add(ReLU)
        .add(Linear::new(8, 2));

    // Forward pass
    let input = Variable::new(
        Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]).unwrap(),
        false
    );
    let output = model.forward(&input);

    assert_eq!(output.data().shape(), &[1, 2]);
    println!("✓ Neural network forward pass works");
}

/// Test 4: Optimizer updates weights
#[test]
fn test_optimizer() {
    let model = Linear::new(4, 2);
    let mut optimizer = Adam::new(model.parameters(), 0.01);

    // Get initial weights
    let initial_weight = model.parameters()[0].data().to_vec();

    // Do a training step
    let input = Variable::new(
        Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]).unwrap(),
        false
    );
    let target = Variable::new(
        Tensor::from_vec(vec![1.0, 0.0], &[1, 2]).unwrap(),
        false
    );

    let output = model.forward(&input);
    let loss = output.mse_loss(&target);

    // Check that loss is computed
    let loss_val = loss.data().to_vec()[0];
    assert!(loss_val > 0.0, "Loss should be positive");

    loss.backward();

    // Check that gradients exist on leaf params
    let has_grad = model.parameters().iter().any(|p| p.grad().is_some());

    optimizer.step();
    optimizer.zero_grad();

    // Weights should have changed
    let updated_weight = model.parameters()[0].data().to_vec();
    assert_ne!(initial_weight, updated_weight, "Weights should be updated by optimizer");

    println!("✓ Optimizer updates weights");
    println!("  Initial weight[0]: {:.4}", initial_weight[0]);
    println!("  Updated weight[0]: {:.4}", updated_weight[0]);
}

/// Test 5: DataLoader batches data correctly
#[test]
fn test_dataloader() {
    use axonml::data::{Dataset, DataLoader, InMemoryDataset};

    // Create a simple dataset
    let data: Vec<(Tensor<f32>, Tensor<f32>)> = (0..100)
        .map(|i| {
            let x = Tensor::from_vec(vec![i as f32; 4], &[4]).unwrap();
            let y = Tensor::from_vec(vec![(i % 10) as f32], &[1]).unwrap();
            (x, y)
        })
        .collect();

    let dataset = InMemoryDataset::new(data);
    let loader = DataLoader::new(dataset, 10);

    assert_eq!(loader.len(), 10); // 100 samples / 10 batch_size

    let mut batch_count = 0;
    for _batch in loader.iter() {
        batch_count += 1;
    }
    assert_eq!(batch_count, 10);

    println!("✓ DataLoader batching works");
}

/// Test 6: Full training loop (XOR problem)
#[test]
fn test_full_training_loop() {
    // XOR dataset
    let inputs = [
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let targets = [0.0, 1.0, 1.0, 0.0];

    // Model
    let model = Sequential::new()
        .add(Linear::new(2, 8))
        .add(ReLU)
        .add(Linear::new(8, 1))
        .add(Sigmoid);

    let mut optimizer = Adam::new(model.parameters(), 0.1);

    // Training loop
    let mut final_loss = f32::MAX;
    for _epoch in 0..100 {
        let mut epoch_loss = 0.0;

        for (input, &target) in inputs.iter().zip(targets.iter()) {
            let x = Variable::new(
                Tensor::from_vec(input.clone(), &[1, 2]).unwrap(),
                false
            );
            let y = Variable::new(
                Tensor::from_vec(vec![target], &[1, 1]).unwrap(),
                false
            );

            let pred = model.forward(&x);
            let loss = pred.mse_loss(&y);

            epoch_loss += loss.data().to_vec()[0];

            loss.backward();
            optimizer.step();
            optimizer.zero_grad();
        }

        final_loss = epoch_loss / 4.0;
    }

    // Training should converge - XOR is a simple problem
    assert!(final_loss < 0.01, "Training didn't converge, loss: {}", final_loss);
    assert!(!final_loss.is_nan(), "Loss is NaN");

    println!("✓ Full training loop works (final loss: {:.4})", final_loss);
}

/// Test 7: Vision transforms work
#[test]
fn test_vision_transforms() {
    use axonml::vision::{Resize, ImageNormalize, CenterCrop};
    use axonml::data::Transform;

    // Create a fake image tensor [C, H, W]
    let image = Tensor::<f32>::ones(&[3, 32, 32]);

    // Apply transforms
    let resized = Resize::new(64, 64).apply(&image);
    assert_eq!(resized.shape(), &[3, 64, 64]);

    let cropped = CenterCrop::new(16, 16).apply(&image);
    assert_eq!(cropped.shape(), &[3, 16, 16]);

    let normalized = ImageNormalize::imagenet().apply(&image);
    assert_eq!(normalized.shape(), &[3, 32, 32]);

    println!("✓ Vision transforms work");
}

/// Test 8: Text tokenization works
#[test]
fn test_text_tokenization() {
    use axonml::text::{WhitespaceTokenizer, Tokenizer, Vocab};

    let tokenizer = WhitespaceTokenizer::new();
    let tokens = tokenizer.tokenize("hello world from axonml");
    assert_eq!(tokens, vec!["hello", "world", "from", "axonml"]);

    // Build vocab
    let mut vocab = Vocab::new();
    for token in &tokens {
        vocab.add_token(token);
    }

    // Encode
    let token_refs: Vec<&str> = tokens.iter().map(|s| s.as_str()).collect();
    let ids = vocab.encode(&token_refs);
    assert_eq!(ids.len(), 4);

    // Decode
    let decoded = vocab.decode(&ids);
    assert_eq!(decoded.len(), tokens.len());

    println!("✓ Text tokenization works");
}

/// Test 9: Complete CNN on synthetic MNIST
#[test]
fn test_cnn_mnist() {
    use axonml::vision::{SyntheticMNIST, SimpleCNN};
    use axonml::data::DataLoader;

    // Load synthetic dataset
    let dataset = SyntheticMNIST::new(32);
    assert_eq!(dataset.len(), 32);

    // Create model (1 channel for MNIST, 10 classes)
    let model = SimpleCNN::new(1, 10);

    // Create dataloader
    let loader = DataLoader::new(dataset, 8);

    // One forward pass through a batch
    let mut success = false;
    for batch in loader.iter() {
        let input = Variable::new(batch.data, false);
        let output = model.forward(&input);

        // Output should be [batch_size, 10] for 10 classes
        assert_eq!(output.data().shape()[1], 10);
        success = true;
        break;
    }

    assert!(success, "No batches processed");
    println!("✓ CNN on MNIST works");
}

/// Test 10: Learning rate scheduler works
#[test]
fn test_lr_scheduler() {
    let model = Linear::new(4, 2);
    let mut optimizer = SGD::new(model.parameters(), 0.1);
    let mut scheduler = StepLR::new(&optimizer, 10, 0.1);

    let initial_lr = optimizer.get_lr();

    // Simulate epochs
    for _ in 0..15 {
        scheduler.step(&mut optimizer);
    }

    let final_lr = optimizer.get_lr();
    assert!(final_lr < initial_lr, "LR should have decreased");

    println!("✓ Learning rate scheduler works");
}
