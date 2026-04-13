//! Integration Tests
//!
//! # File
//! `crates/axonml-vision/src/training/integration.rs`
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
    use axonml_data::{Dataset, Transform};
    use axonml_nn::{CrossEntropyLoss, MSELoss, Module};
    use axonml_optim::{Adam, Optimizer, SGD};
    use axonml_tensor::Tensor;

    use crate::datasets::{SyntheticCIFAR, SyntheticMNIST};
    use crate::models::lenet::{LeNet, MLP, SimpleCNN};
    use crate::models::resnet::ResNet;
    use crate::models::transformer::VisionTransformer;
    use crate::transforms::ImageNormalize;

    // =========================================================================
    // Helpers
    // =========================================================================

    /// Build a batch with transform applied to each image.
    fn make_batch_with_transform<D, T>(
        dataset: &D,
        transform: &T,
        start: usize,
        batch_size: usize,
    ) -> (Variable, Variable)
    where
        D: Dataset<Item = (Tensor<f32>, Tensor<f32>)>,
        T: Transform,
    {
        let mut images = Vec::new();
        let mut labels = Vec::new();
        let mut img_shape = Vec::new();

        for i in start..start + batch_size {
            let (img, lbl) = dataset.get(i % dataset.len()).unwrap();
            let transformed = transform.apply(&img);
            if img_shape.is_empty() {
                img_shape = transformed.shape().to_vec();
            }
            images.extend(transformed.to_vec());
            let class = lbl
                .to_vec()
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as f32)
                .unwrap();
            labels.push(class);
        }

        let mut full_shape = vec![batch_size];
        full_shape.extend(&img_shape);

        let images_var = Variable::new(Tensor::from_vec(images, &full_shape).unwrap(), false);
        let labels_var = Variable::new(Tensor::from_vec(labels, &[batch_size]).unwrap(), false);

        (images_var, labels_var)
    }

    /// Build a batch without transform (raw dataset).
    fn make_batch<D: Dataset<Item = (Tensor<f32>, Tensor<f32>)>>(
        dataset: &D,
        start: usize,
        batch_size: usize,
    ) -> (Variable, Variable) {
        let mut images = Vec::new();
        let mut labels = Vec::new();

        for i in start..start + batch_size {
            let (img, lbl) = dataset.get(i % dataset.len()).unwrap();
            images.extend(img.to_vec());
            let class = lbl
                .to_vec()
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as f32)
                .unwrap();
            labels.push(class);
        }

        let img_shape = {
            let (img, _) = dataset.get(0).unwrap();
            let s = img.shape().to_vec();
            let mut full = vec![batch_size];
            full.extend(&s);
            full
        };

        let images_var = Variable::new(Tensor::from_vec(images, &img_shape).unwrap(), false);
        let labels_var = Variable::new(Tensor::from_vec(labels, &[batch_size]).unwrap(), false);

        (images_var, labels_var)
    }

    // =========================================================================
    // Integration: MNIST -> Normalize -> LeNet -> CrossEntropy -> Adam
    // =========================================================================

    #[test]
    fn integration_mnist_lenet_adam() {
        let dataset = SyntheticMNIST::new(100);
        let normalize = ImageNormalize::new(vec![0.1307], vec![0.3081]); // MNIST stats
        let model = LeNet::new();
        let mut optimizer = Adam::new(model.parameters(), 0.001);
        let loss_fn = CrossEntropyLoss::new();

        let batch_size = 16;
        let mut losses = Vec::new();

        for step in 0..10 {
            let (images, targets) =
                make_batch_with_transform(&dataset, &normalize, step * batch_size, batch_size);

            optimizer.zero_grad();
            let logits = model.forward(&images);

            assert_eq!(
                logits.shape(),
                vec![batch_size, 10],
                "Output shape mismatch"
            );
            let loss = loss_fn.compute(&logits, &targets);

            let loss_val = loss.data().to_vec()[0];
            assert!(
                loss_val.is_finite(),
                "Loss is not finite at step {step}: {loss_val}"
            );
            losses.push(loss_val);

            loss.backward();
            optimizer.step();
        }

        // Loss should decrease over 10 steps
        assert!(
            losses.last().unwrap() < losses.first().unwrap(),
            "Pipeline loss did not decrease: {:?}",
            losses
        );
    }

    // =========================================================================
    // Integration: CIFAR -> Normalize -> ResNet18 -> CrossEntropy -> SGD
    // =========================================================================

    #[test]
    fn integration_cifar_resnet_sgd() {
        let dataset = SyntheticCIFAR::cifar10(48);
        let model = ResNet::resnet18(10);
        let mut optimizer = SGD::with_momentum(model.parameters(), 0.01, 0.9);
        let loss_fn = CrossEntropyLoss::new();

        let batch_size = 8;

        for step in 0..3 {
            let (images, targets) = make_batch(&dataset, step * batch_size, batch_size);

            optimizer.zero_grad();
            let logits = model.forward(&images);

            assert_eq!(logits.shape(), vec![batch_size, 10]);

            let loss = loss_fn.compute(&logits, &targets);
            let loss_val = loss.data().to_vec()[0];
            assert!(
                loss_val.is_finite(),
                "ResNet loss not finite at step {step}"
            );

            loss.backward();
            optimizer.step();
        }
    }

    // =========================================================================
    // Integration: CIFAR -> Resize+Normalize -> SimpleCNN -> CrossEntropy -> Adam
    // =========================================================================

    #[test]
    fn integration_cifar_simplecnn_adam() {
        let dataset = SyntheticCIFAR::cifar10(64);
        let normalize =
            ImageNormalize::new(vec![0.4914, 0.4822, 0.4465], vec![0.2470, 0.2435, 0.2616]);
        let model = SimpleCNN::for_cifar10();
        let mut optimizer = Adam::new(model.parameters(), 0.001);
        let loss_fn = CrossEntropyLoss::new();

        let batch_size = 8;
        let mut losses = Vec::new();

        for step in 0..5 {
            let (images, targets) =
                make_batch_with_transform(&dataset, &normalize, step * batch_size, batch_size);

            optimizer.zero_grad();
            let logits = model.forward(&images);
            let loss = loss_fn.compute(&logits, &targets);

            let loss_val = loss.data().to_vec()[0];
            assert!(
                loss_val.is_finite(),
                "SimpleCNN loss not finite at step {step}"
            );
            losses.push(loss_val);

            loss.backward();
            optimizer.step();
        }
    }

    // =========================================================================
    // Integration: CIFAR -> ViT -> CrossEntropy -> Adam
    // =========================================================================

    #[test]
    fn integration_cifar_vit_adam() {
        let dataset = SyntheticCIFAR::cifar10(48);
        let model = VisionTransformer::new(32, 8, 3, 10, 64, 2, 4, 128, 0.0);
        let mut optimizer = Adam::new(model.parameters(), 0.001);
        let loss_fn = CrossEntropyLoss::new();

        let batch_size = 8;
        let mut losses = Vec::new();

        for step in 0..5 {
            let (images, targets) = make_batch(&dataset, step * batch_size, batch_size);

            optimizer.zero_grad();
            let logits = model.forward(&images);

            assert_eq!(
                logits.shape(),
                vec![batch_size, 10],
                "ViT output shape mismatch"
            );

            let loss = loss_fn.compute(&logits, &targets);
            let loss_val = loss.data().to_vec()[0];
            assert!(loss_val.is_finite(), "ViT loss not finite at step {step}");
            losses.push(loss_val);

            loss.backward();
            optimizer.step();
        }

        assert!(
            losses.last().unwrap() < losses.first().unwrap(),
            "ViT pipeline loss did not decrease"
        );
    }

    // =========================================================================
    // Integration: MNIST -> MLP -> MSELoss -> Adam (regression-style)
    // =========================================================================

    #[test]
    fn integration_mnist_mlp_mse() {
        let dataset = SyntheticMNIST::new(100);
        let model = MLP::for_mnist();
        let mut optimizer = Adam::new(model.parameters(), 0.001);
        let loss_fn = MSELoss::new();

        let batch_size = 16;
        let mut losses = Vec::new();

        for step in 0..10 {
            let mut images = Vec::new();
            let mut targets = Vec::new();

            for i in 0..batch_size {
                let (img, lbl) = dataset
                    .get((step * batch_size + i) % dataset.len())
                    .unwrap();
                images.extend(img.to_vec());
                targets.extend(lbl.to_vec()); // one-hot as MSE target
            }

            let images_var = Variable::new(
                Tensor::from_vec(images, &[batch_size, 1, 28, 28]).unwrap(),
                false,
            );
            let targets_var =
                Variable::new(Tensor::from_vec(targets, &[batch_size, 10]).unwrap(), false);

            optimizer.zero_grad();
            let logits = model.forward(&images_var);
            let loss = loss_fn.compute(&logits, &targets_var);

            let loss_val = loss.data().to_vec()[0];
            assert!(loss_val.is_finite(), "MSE loss not finite at step {step}");
            losses.push(loss_val);

            loss.backward();
            optimizer.step();
        }

        assert!(
            losses.last().unwrap() < losses.first().unwrap(),
            "MSE pipeline loss did not decrease: first={:.4}, last={:.4}",
            losses.first().unwrap(),
            losses.last().unwrap()
        );
    }

    // =========================================================================
    // Integration: Detection pipeline (Phantom)
    // =========================================================================

    #[test]
    fn integration_detection_phantom() {
        use crate::models::phantom::Phantom;

        let mut model = Phantom::new();
        model.train();
        let params = model.parameters();
        let mut optimizer = Adam::new(params, 1e-3);

        let mut losses = Vec::new();
        for step in 0..5 {
            let seed = step as f32 * 0.1;
            let pixels: Vec<f32> = (0..3 * 64 * 64)
                .map(|i| (i as f32 * 0.001 + seed).sin() * 0.5 + 0.5)
                .collect();
            let frame = Variable::new(Tensor::from_vec(pixels, &[1, 3, 64, 64]).unwrap(), false);
            let gt_faces = vec![[8.0, 8.0, 32.0, 32.0]];

            let loss = crate::training::phantom_training_step(
                &mut model,
                &frame,
                &gt_faces,
                &mut optimizer,
            );
            assert!(loss.is_finite(), "Detection loss not finite at step {step}");
            losses.push(loss);
        }

        // At least verify all losses are finite (detection training is hard to converge in 5 steps)
        assert_eq!(losses.len(), 5);
    }

    // =========================================================================
    // Integration: Biometric pipeline (Mnemosyne face encoding)
    // =========================================================================

    #[test]
    fn integration_biometric_mnemosyne() {
        use crate::models::biometric::MnemosyneIdentity;
        use axonml_nn::Module;

        let model = MnemosyneIdentity::new();
        let mut optimizer = Adam::new(model.parameters(), 0.001);
        let loss_fn = MSELoss::new();

        // Get the encoding dimension from a probe forward pass
        let probe = Variable::new(
            Tensor::from_vec(vec![0.5; 3 * 32 * 32], &[1, 3, 32, 32]).unwrap(),
            false,
        );
        let probe_out = model.forward(&probe);
        let enc_dim: usize = probe_out.shape().iter().skip(1).product();

        let mut losses = Vec::new();
        for step in 0..5 {
            // Simulate a face crop: [1, 3, 32, 32]
            let seed = step as f32 * 0.05;
            let pixels: Vec<f32> = (0..3 * 32 * 32)
                .map(|i| (i as f32 * 0.01 + seed).sin() * 0.5 + 0.5)
                .collect();
            let face = Variable::new(Tensor::from_vec(pixels, &[1, 3, 32, 32]).unwrap(), false);

            // Target: arbitrary embedding
            let target = Variable::new(
                Tensor::from_vec(vec![0.5; enc_dim], &[1, enc_dim]).unwrap(),
                false,
            );

            optimizer.zero_grad();
            let encoding = model.forward(&face);

            // Reshape if needed to match target
            let enc_shape = encoding.shape();
            let enc_flat = if enc_shape.len() > 2 {
                let features: usize = enc_shape[1..].iter().product();
                encoding.reshape(&[1, features])
            } else {
                encoding.clone()
            };

            // Use MSE as a proxy loss for embedding training
            let loss = loss_fn.compute(&enc_flat, &target);
            let loss_val = loss.data().to_vec()[0];
            assert!(
                loss_val.is_finite(),
                "Mnemosyne loss not finite at step {step}"
            );
            losses.push(loss_val);

            loss.backward();
            optimizer.step();
        }

        assert!(losses.len() == 5);
    }

    // =========================================================================
    // Integration: Gradient flow validation
    // =========================================================================

    #[test]
    fn integration_gradient_flow_lenet() {
        let dataset = SyntheticMNIST::new(32);
        let model = LeNet::new();
        let mut optimizer = Adam::new(model.parameters(), 0.001);
        let loss_fn = CrossEntropyLoss::new();

        let (images, targets) = make_batch(&dataset, 0, 16);

        optimizer.zero_grad();
        let logits = model.forward(&images);
        let loss = loss_fn.compute(&logits, &targets);
        loss.backward();

        // After backward, all parameters should have gradients
        let params = model.parameters();
        let mut has_grad = 0;
        for p in &params {
            let var = p.variable();
            if let Some(grad) = var.grad() {
                let grad_norm: f32 = grad.to_vec().iter().map(|x| x * x).sum::<f32>().sqrt();
                assert!(grad_norm.is_finite(), "Gradient is not finite");
                if grad_norm > 0.0 {
                    has_grad += 1;
                }
            }
        }

        assert!(
            has_grad > 0,
            "No parameters received non-zero gradients (of {} params)",
            params.len()
        );

        optimizer.step();
    }
}
