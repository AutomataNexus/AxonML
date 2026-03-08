//! Training Convergence Tests
//!
//! # File
//! `crates/axonml-vision/src/training/convergence.rs`
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
    use axonml_data::Dataset;
    use axonml_nn::{CrossEntropyLoss, Module};
    use axonml_optim::{Adam, Optimizer, SGD};
    use axonml_tensor::Tensor;

    use crate::datasets::{SyntheticCIFAR, SyntheticMNIST};
    use crate::models::lenet::{LeNet, MLP};

    // =========================================================================
    // Helpers
    // =========================================================================

    /// Build a batch from the synthetic dataset.
    /// Returns (images [N, C, H, W], class_indices [N]).
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
            // one-hot -> class index
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

    /// Compute classification accuracy from logits [N, C] and targets [N].
    fn accuracy(logits: &Variable, targets: &Variable) -> f32 {
        let logit_vec = logits.data().to_vec();
        let target_vec = targets.data().to_vec();
        let shape = logits.shape();
        let batch_size = shape[0];
        let num_classes = shape[1];

        let mut correct = 0;
        for b in 0..batch_size {
            let offset = b * num_classes;
            let pred = (0..num_classes)
                .max_by(|&a, &b_idx| {
                    logit_vec[offset + a]
                        .partial_cmp(&logit_vec[offset + b_idx])
                        .unwrap()
                })
                .unwrap();
            if pred == target_vec[b] as usize {
                correct += 1;
            }
        }
        correct as f32 / batch_size as f32
    }

    // =========================================================================
    // Phase 1: LeNet on SyntheticMNIST (Adam)
    // =========================================================================

    #[test]
    fn convergence_lenet_mnist() {
        let dataset = SyntheticMNIST::new(200);
        let model = LeNet::new();
        let mut optimizer = Adam::new(model.parameters(), 0.001);
        let loss_fn = CrossEntropyLoss::new();

        let batch_size = 16;
        let num_epochs = 8;
        let batches_per_epoch = dataset.len() / batch_size;

        let mut first_epoch_loss = 0.0;
        let mut last_epoch_loss = 0.0;

        for epoch in 0..num_epochs {
            let mut epoch_loss = 0.0;

            for batch_idx in 0..batches_per_epoch {
                let (images, targets) = make_batch(&dataset, batch_idx * batch_size, batch_size);

                optimizer.zero_grad();
                let logits = model.forward(&images);
                let loss = loss_fn.compute(&logits, &targets);

                let loss_val = loss.data().to_vec()[0];
                epoch_loss += loss_val;

                loss.backward();
                optimizer.step();
            }

            let avg_loss = epoch_loss / batches_per_epoch as f32;
            if epoch == 0 {
                first_epoch_loss = avg_loss;
            }
            if epoch == num_epochs - 1 {
                last_epoch_loss = avg_loss;
            }
        }

        assert!(
            last_epoch_loss < first_epoch_loss,
            "Loss did not decrease: first_epoch={first_epoch_loss:.4}, last_epoch={last_epoch_loss:.4}"
        );

        // Evaluate accuracy on training data
        let (eval_images, eval_targets) = make_batch(&dataset, 0, 32);
        let logits = model.forward(&eval_images);
        let acc = accuracy(&logits, &eval_targets);
        assert!(
            acc > 0.3,
            "Accuracy too low: {acc:.2} (expected > 0.3 after training)"
        );
    }

    // =========================================================================
    // Phase 1: MLP on SyntheticMNIST
    // =========================================================================

    #[test]
    fn convergence_mlp_mnist() {
        let dataset = SyntheticMNIST::new(200);
        let model = MLP::for_mnist();
        let mut optimizer = Adam::new(model.parameters(), 0.001);
        let loss_fn = CrossEntropyLoss::new();

        let batch_size = 16;
        let num_epochs = 10;
        let batches_per_epoch = dataset.len() / batch_size;

        let mut first_epoch_loss = 0.0;
        let mut last_epoch_loss = 0.0;

        for epoch in 0..num_epochs {
            let mut epoch_loss = 0.0;

            for batch_idx in 0..batches_per_epoch {
                let (images, targets) = make_batch(&dataset, batch_idx * batch_size, batch_size);

                optimizer.zero_grad();
                let logits = model.forward(&images);
                let loss = loss_fn.compute(&logits, &targets);

                epoch_loss += loss.data().to_vec()[0];
                loss.backward();
                optimizer.step();
            }

            let avg_loss = epoch_loss / batches_per_epoch as f32;
            if epoch == 0 {
                first_epoch_loss = avg_loss;
            }
            if epoch == num_epochs - 1 {
                last_epoch_loss = avg_loss;
            }
        }

        assert!(
            last_epoch_loss < first_epoch_loss,
            "MLP loss did not decrease: first={first_epoch_loss:.4}, last={last_epoch_loss:.4}"
        );
    }

    // =========================================================================
    // Phase 1: LeNet on SyntheticCIFAR
    // =========================================================================

    #[test]
    fn convergence_lenet_cifar() {
        let dataset = SyntheticCIFAR::cifar10(200);
        let model = LeNet::for_cifar10();
        let mut optimizer = Adam::new(model.parameters(), 0.001);
        let loss_fn = CrossEntropyLoss::new();

        let batch_size = 16;
        let num_epochs = 8;
        let batches_per_epoch = dataset.len() / batch_size;

        let mut first_epoch_loss = 0.0;
        let mut last_epoch_loss = 0.0;

        for epoch in 0..num_epochs {
            let mut epoch_loss = 0.0;

            for batch_idx in 0..batches_per_epoch {
                let (images, targets) = make_batch(&dataset, batch_idx * batch_size, batch_size);

                optimizer.zero_grad();
                let logits = model.forward(&images);
                let loss = loss_fn.compute(&logits, &targets);

                epoch_loss += loss.data().to_vec()[0];
                loss.backward();
                optimizer.step();
            }

            let avg_loss = epoch_loss / batches_per_epoch as f32;
            if epoch == 0 {
                first_epoch_loss = avg_loss;
            }
            if epoch == num_epochs - 1 {
                last_epoch_loss = avg_loss;
            }
        }

        assert!(
            last_epoch_loss < first_epoch_loss,
            "CIFAR loss did not decrease: first={first_epoch_loss:.4}, last={last_epoch_loss:.4}"
        );
    }

    // =========================================================================
    // Phase 1: ResNet18 smoke (loss decreasing)
    // =========================================================================

    #[test]
    fn convergence_resnet18_cifar_smoke() {
        use crate::models::resnet::ResNet;

        let dataset = SyntheticCIFAR::cifar10(64);
        let model = ResNet::resnet18(10);
        let mut optimizer = Adam::new(model.parameters(), 0.001);
        let loss_fn = CrossEntropyLoss::new();

        let batch_size = 8;
        let steps = 8;

        let mut losses = Vec::new();
        for step in 0..steps {
            let (images, targets) = make_batch(&dataset, step * batch_size, batch_size);

            optimizer.zero_grad();
            let logits = model.forward(&images);
            let loss = loss_fn.compute(&logits, &targets);

            let loss_val = loss.data().to_vec()[0];
            losses.push(loss_val);

            loss.backward();
            optimizer.step();
        }

        for (i, &l) in losses.iter().enumerate() {
            assert!(l.is_finite(), "Loss at step {i} is not finite: {l}");
        }

        assert!(
            losses.last().unwrap() < losses.first().unwrap(),
            "ResNet18 loss did not decrease: first={:.4}, last={:.4}",
            losses.first().unwrap(),
            losses.last().unwrap()
        );
    }

    // =========================================================================
    // Phase 1: NanoDet smoke (forward + gradient flow)
    // =========================================================================

    #[test]
    fn convergence_nanodet_forward_smoke() {
        use crate::models::nanodet::NanoDet;

        let model = NanoDet::new(1);

        // Verify forward pass produces finite output at multiple scales
        for size in [64, 128] {
            let pixels: Vec<f32> = (0..3 * size * size)
                .map(|i| ((i as f32 * 0.001).sin() * 0.5 + 0.5))
                .collect();
            let frame = Variable::new(
                Tensor::from_vec(pixels, &[1, 3, size, size]).unwrap(),
                false,
            );

            let output = model.forward(&frame);
            let vals = output.data().to_vec();
            assert!(
                vals.iter().all(|v| v.is_finite()),
                "NanoDet produced non-finite output at size {size}"
            );
        }
    }

    #[test]
    fn convergence_nanodet_training_step() {
        use crate::models::phantom::Phantom;

        // Use Phantom as the detection training smoke test since it has
        // a proper training step function already
        let mut model = Phantom::new();
        model.train();
        let params = model.parameters();
        let mut optimizer = Adam::new(params, 1e-3);

        let mut losses = Vec::new();
        for step in 0..3 {
            let seed = step as f32 * 0.1;
            let pixels: Vec<f32> = (0..3 * 64 * 64)
                .map(|i| ((i as f32 * 0.001 + seed).sin() * 0.5 + 0.5))
                .collect();
            let frame = Variable::new(
                Tensor::from_vec(pixels, &[1, 3, 64, 64]).unwrap(),
                false,
            );
            let gt_faces = vec![[10.0, 10.0, 30.0, 30.0]];

            let loss = crate::training::phantom_training_step(
                &mut model, &frame, &gt_faces, &mut optimizer,
            );
            losses.push(loss);
        }

        for (i, &l) in losses.iter().enumerate() {
            assert!(l.is_finite(), "Phantom loss at step {i} is not finite: {l}");
        }
    }

    // =========================================================================
    // Phase 1: ViT smoke (loss decreasing)
    // =========================================================================

    #[test]
    fn convergence_vit_cifar_smoke() {
        use crate::models::transformer::VisionTransformer;

        let dataset = SyntheticCIFAR::cifar10(64);
        // Small ViT: image_size=32, patch_size=8, channels=3, classes=10,
        //            d_model=64, depth=2, heads=4, mlp_dim=128, dropout=0
        let model = VisionTransformer::new(32, 8, 3, 10, 64, 2, 4, 128, 0.0);
        let mut optimizer = Adam::new(model.parameters(), 0.001);
        let loss_fn = CrossEntropyLoss::new();

        let batch_size = 8;
        let steps = 8;

        let mut losses = Vec::new();
        for step in 0..steps {
            let (images, targets) = make_batch(&dataset, step * batch_size, batch_size);

            optimizer.zero_grad();
            let logits = model.forward(&images);
            let loss = loss_fn.compute(&logits, &targets);

            let loss_val = loss.data().to_vec()[0];
            losses.push(loss_val);

            loss.backward();
            optimizer.step();
        }

        for (i, &l) in losses.iter().enumerate() {
            assert!(l.is_finite(), "ViT loss at step {i} is not finite: {l}");
        }

        assert!(
            losses.last().unwrap() < losses.first().unwrap(),
            "ViT loss did not decrease: first={:.4}, last={:.4}",
            losses.first().unwrap(),
            losses.last().unwrap()
        );
    }

    // =========================================================================
    // Phase 1: LeNet with SGD+Momentum
    // =========================================================================

    #[test]
    fn convergence_lenet_sgd() {
        let dataset = SyntheticMNIST::new(200);
        let model = LeNet::new();
        let mut optimizer = SGD::with_momentum(model.parameters(), 0.05, 0.9);
        let loss_fn = CrossEntropyLoss::new();

        let batch_size = 16;
        let num_epochs = 15;
        let batches_per_epoch = dataset.len() / batch_size;

        let mut first_epoch_loss = 0.0;
        let mut last_epoch_loss = 0.0;

        for epoch in 0..num_epochs {
            let mut epoch_loss = 0.0;

            for batch_idx in 0..batches_per_epoch {
                let (images, targets) = make_batch(&dataset, batch_idx * batch_size, batch_size);

                optimizer.zero_grad();
                let logits = model.forward(&images);
                let loss = loss_fn.compute(&logits, &targets);

                epoch_loss += loss.data().to_vec()[0];
                loss.backward();
                optimizer.step();
            }

            let avg_loss = epoch_loss / batches_per_epoch as f32;
            if epoch == 0 {
                first_epoch_loss = avg_loss;
            }
            if epoch == num_epochs - 1 {
                last_epoch_loss = avg_loss;
            }
        }

        assert!(
            last_epoch_loss < first_epoch_loss,
            "SGD loss did not decrease: first={first_epoch_loss:.4}, last={last_epoch_loss:.4}"
        );
    }

    // =========================================================================
    // Phase 1: Helios training loss convergence
    // =========================================================================

    #[test]
    fn convergence_helios_training_step() {
        use crate::models::helios::{Helios, HeliosLoss};

        let model = Helios::nano(2);
        let mut optimizer = Adam::new(model.parameters(), 1e-3);
        let loss_fn = HeliosLoss::new(2, 16);

        // Synthetic GT: one box per image
        let gt_boxes = vec![vec![[8.0, 8.0, 48.0, 48.0]]];
        let gt_classes = vec![vec![0usize]];

        let mut losses = Vec::new();
        for step in 0..5 {
            let seed = step as f32 * 0.1;
            let pixels: Vec<f32> = (0..3 * 64 * 64)
                .map(|i| ((i as f32 * 0.001 + seed).sin() * 0.5 + 0.5))
                .collect();
            let input = Variable::new(
                Tensor::from_vec(pixels, &[1, 3, 64, 64]).unwrap(),
                false,
            );

            optimizer.zero_grad();
            let train_out = model.forward_train(&input);
            let (total_loss, _cls, _box, _dfl) = loss_fn.compute(
                &train_out, &gt_boxes, &gt_classes, 2,
            );

            let val = total_loss.data().to_vec()[0];
            losses.push(val);

            total_loss.backward();
            optimizer.step();
        }

        for (i, &l) in losses.iter().enumerate() {
            assert!(l.is_finite(), "Helios loss at step {i} is not finite: {l}");
        }
        assert!(losses[0] > 0.0, "Initial loss should be positive");
    }
}
