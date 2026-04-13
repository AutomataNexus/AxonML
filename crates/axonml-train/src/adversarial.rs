//! Adversarial Training Utilities
//!
//! # File
//! `crates/axonml-train/src/adversarial.rs`
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

use axonml_autograd::Variable;
use axonml_nn::Module;
use axonml_optim::Optimizer;
use axonml_tensor::Tensor;

// =============================================================================
// AdversarialTrainer
// =============================================================================

/// Adversarial training wrapper for dual-network (generator/discriminator) setups.
///
/// Manages alternating training of two networks with configurable step ratios.
///
/// # Example
/// ```ignore
/// use axonml_train::adversarial::AdversarialTrainer;
///
/// let mut trainer = AdversarialTrainer::new(gen_optimizer, disc_optimizer);
/// trainer.set_disc_steps_per_gen(5); // Train D 5x per G step
///
/// for batch in data {
///     let (g_loss, d_loss) = trainer.step(&gen, &disc, &real, &noise, |pred, target| {
///         bce_loss(pred, target)
///     });
/// }
/// ```
pub struct AdversarialTrainer {
    gen_optimizer: Box<dyn Optimizer>,
    disc_optimizer: Box<dyn Optimizer>,
    disc_steps_per_gen: usize,
    current_step: usize,
}

impl AdversarialTrainer {
    /// Creates a new adversarial trainer.
    pub fn new(gen_optimizer: Box<dyn Optimizer>, disc_optimizer: Box<dyn Optimizer>) -> Self {
        Self {
            gen_optimizer,
            disc_optimizer,
            disc_steps_per_gen: 1,
            current_step: 0,
        }
    }

    /// Sets how many discriminator steps per generator step.
    pub fn set_disc_steps_per_gen(&mut self, steps: usize) {
        self.disc_steps_per_gen = steps;
    }

    /// Performs one adversarial training step.
    ///
    /// # Arguments
    /// * `generator` - Generator model
    /// * `discriminator` - Discriminator model
    /// * `real_data` - Real data batch
    /// * `noise` - Random noise input for generator
    /// * `loss_fn` - Loss function taking (predictions, targets) → scalar loss
    ///
    /// # Returns
    /// Tuple of (generator_loss, discriminator_loss)
    pub fn step<G, D, F>(
        &mut self,
        generator: &G,
        discriminator: &D,
        real_data: &Variable,
        noise: &Variable,
        loss_fn: F,
    ) -> (f32, f32)
    where
        G: Module,
        D: Module,
        F: Fn(&Variable, &Variable) -> Variable,
    {
        // Train discriminator
        self.disc_optimizer.zero_grad();

        // Discriminator on real data
        let real_pred = discriminator.forward(real_data);
        let real_labels = Variable::new(
            Tensor::from_vec(vec![1.0; real_pred.numel()], &real_pred.shape()).unwrap(),
            false,
        );
        let d_real_loss = loss_fn(&real_pred, &real_labels);

        // Discriminator on fake data
        let fake_data = generator.forward(noise);
        let fake_pred = discriminator.forward(&fake_data);
        let fake_labels = Variable::new(
            Tensor::from_vec(vec![0.0; fake_pred.numel()], &fake_pred.shape()).unwrap(),
            false,
        );
        let d_fake_loss = loss_fn(&fake_pred, &fake_labels);

        // Combined discriminator loss
        let d_loss = d_real_loss.add_var(&d_fake_loss);
        let d_loss_val = d_loss.data().to_vec()[0] * 0.5;
        d_loss.backward();
        self.disc_optimizer.step();

        // Train generator (only every disc_steps_per_gen steps)
        let mut g_loss_val = 0.0;
        self.current_step += 1;

        if self.current_step % self.disc_steps_per_gen == 0 {
            self.gen_optimizer.zero_grad();

            let fake_data = generator.forward(noise);
            let fake_pred = discriminator.forward(&fake_data);
            let gen_labels = Variable::new(
                Tensor::from_vec(vec![1.0; fake_pred.numel()], &fake_pred.shape()).unwrap(),
                false,
            );
            let g_loss = loss_fn(&fake_pred, &gen_labels);
            g_loss_val = g_loss.data().to_vec()[0];
            g_loss.backward();
            self.gen_optimizer.step();
        }

        (g_loss_val, d_loss_val)
    }

    /// Returns references to the optimizers.
    pub fn optimizers(&self) -> (&dyn Optimizer, &dyn Optimizer) {
        (self.gen_optimizer.as_ref(), self.disc_optimizer.as_ref())
    }
}

// =============================================================================
// FGSM Attack
// =============================================================================

/// Fast Gradient Sign Method (FGSM) adversarial example generation.
///
/// Creates adversarial perturbations by taking a single step in the direction
/// of the gradient sign, bounded by epsilon.
///
/// # Arguments
/// * `input` - Original input (must have requires_grad=true)
/// * `grad` - Gradient of the loss with respect to input
/// * `epsilon` - Maximum perturbation magnitude (L-infinity norm)
///
/// # Returns
/// Perturbed input: `input + epsilon * sign(grad)`
///
/// # Example
/// ```ignore
/// use axonml_train::adversarial::fgsm_attack;
///
/// // 1. Forward pass with gradient tracking
/// let input = Variable::new(tensor, true);
/// let output = model.forward(&input);
/// let loss = loss_fn(&output, &target);
/// loss.backward();
///
/// // 2. Get input gradient and generate adversarial example
/// let grad = input.grad().unwrap();
/// let adv_input = fgsm_attack(&input, &grad, 0.03);
/// ```
pub fn fgsm_attack(input: &Variable, grad: &Tensor<f32>, epsilon: f32) -> Variable {
    let input_data = input.data().to_vec();
    let grad_data = grad.to_vec();

    assert_eq!(
        input_data.len(),
        grad_data.len(),
        "Input and gradient must have same size"
    );

    let perturbed: Vec<f32> = input_data
        .iter()
        .zip(grad_data.iter())
        .map(|(x, g)| {
            let sign = if *g > 0.0 {
                1.0
            } else if *g < 0.0 {
                -1.0
            } else {
                0.0
            };
            x + epsilon * sign
        })
        .collect();

    Variable::new(Tensor::from_vec(perturbed, &input.shape()).unwrap(), false)
}

// =============================================================================
// PGD Attack
// =============================================================================

/// Projected Gradient Descent (PGD) adversarial example generation.
///
/// Iterative version of FGSM that takes multiple smaller steps, projecting
/// back to the epsilon-ball after each step. More powerful than FGSM.
///
/// # Arguments
/// * `model` - The model to attack
/// * `input` - Original clean input
/// * `target` - True labels
/// * `epsilon` - Maximum perturbation (L-infinity bound)
/// * `alpha` - Step size per iteration
/// * `num_steps` - Number of PGD iterations
/// * `loss_fn` - Loss function: (prediction, target) → scalar loss Variable
///
/// # Returns
/// Adversarial input within epsilon-ball of original
///
/// # Example
/// ```ignore
/// use axonml_train::adversarial::pgd_attack;
///
/// let adv_input = pgd_attack(
///     &model, &input, &target,
///     0.03,   // epsilon
///     0.01,   // step size
///     7,      // iterations
///     |pred, target| cross_entropy(pred, target),
/// );
/// ```
pub fn pgd_attack<F>(
    model: &dyn Module,
    input: &Variable,
    target: &Variable,
    epsilon: f32,
    alpha: f32,
    num_steps: usize,
    loss_fn: F,
) -> Variable
where
    F: Fn(&Variable, &Variable) -> Variable,
{
    let original_data = input.data().to_vec();
    let shape = input.shape();
    let n = original_data.len();

    // Start from original input (can also start from random point in epsilon ball)
    let mut perturbed = original_data.clone();

    for _ in 0..num_steps {
        // Create variable with gradient tracking
        let adv_input = Variable::new(Tensor::from_vec(perturbed.clone(), &shape).unwrap(), true);

        // Forward + loss
        let output = model.forward(&adv_input);
        let loss = loss_fn(&output, target);
        loss.backward();

        // Get gradient and take step
        if let Some(grad) = adv_input.grad() {
            let grad_data = grad.to_vec();
            for i in 0..n {
                let sign = if grad_data[i] > 0.0 {
                    1.0
                } else if grad_data[i] < 0.0 {
                    -1.0
                } else {
                    0.0
                };
                perturbed[i] += alpha * sign;

                // Project back to epsilon-ball around original
                let delta = perturbed[i] - original_data[i];
                perturbed[i] = original_data[i] + delta.clamp(-epsilon, epsilon);
            }
        }
    }

    Variable::new(Tensor::from_vec(perturbed, &shape).unwrap(), false)
}

// =============================================================================
// Adversarial Training Helper
// =============================================================================

/// Performs one step of adversarial training (AT) on a model.
///
/// Generates adversarial examples using PGD and trains on them alongside
/// clean examples. This improves model robustness.
///
/// # Arguments
/// * `model` - Model being trained
/// * `optimizer` - Optimizer for the model
/// * `clean_input` - Clean training batch
/// * `target` - Labels
/// * `epsilon` - Perturbation budget
/// * `pgd_steps` - Number of PGD steps for adversarial generation
/// * `loss_fn` - Loss function
///
/// # Returns
/// Tuple of (clean_loss, adversarial_loss)
pub fn adversarial_training_step<F>(
    model: &dyn Module,
    optimizer: &mut dyn Optimizer,
    clean_input: &Variable,
    target: &Variable,
    epsilon: f32,
    pgd_steps: usize,
    loss_fn: F,
) -> (f32, f32)
where
    F: Fn(&Variable, &Variable) -> Variable,
{
    let alpha = epsilon / pgd_steps.max(1) as f32 * 2.0;

    // Generate adversarial examples
    let adv_input = pgd_attack(
        model,
        clean_input,
        target,
        epsilon,
        alpha,
        pgd_steps,
        &loss_fn,
    );

    // Train on combined clean + adversarial
    optimizer.zero_grad();

    let clean_output = model.forward(clean_input);
    let clean_loss = loss_fn(&clean_output, target);
    let clean_loss_val = clean_loss.data().to_vec()[0];

    let adv_output = model.forward(&adv_input);
    let adv_loss = loss_fn(&adv_output, target);
    let adv_loss_val = adv_loss.data().to_vec()[0];

    // Combined loss
    let total_loss = clean_loss.add_var(&adv_loss);
    total_loss.backward();
    optimizer.step();

    (clean_loss_val, adv_loss_val)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use axonml_nn::{Linear, Sequential};
    use std::ops::Neg;

    #[test]
    fn test_fgsm_attack_perturbation_bound() {
        let input_data = vec![1.0, 2.0, 3.0, 4.0];
        let input = Variable::new(Tensor::from_vec(input_data.clone(), &[1, 4]).unwrap(), true);

        // Gradient pointing in positive direction
        let grad = Tensor::from_vec(vec![0.5, -0.3, 0.0, 1.0], &[1, 4]).unwrap();

        let epsilon = 0.1;
        let adv = fgsm_attack(&input, &grad, epsilon);
        let adv_data = adv.data().to_vec();

        // Check perturbation is within epsilon bound
        for (orig, perturbed) in input_data.iter().zip(adv_data.iter()) {
            let delta = (perturbed - orig).abs();
            assert!(
                delta <= epsilon + 1e-6,
                "Perturbation {} exceeds epsilon {}",
                delta,
                epsilon
            );
        }
    }

    #[test]
    fn test_fgsm_attack_sign_direction() {
        let input = Variable::new(
            Tensor::from_vec(vec![0.0, 0.0, 0.0], &[1, 3]).unwrap(),
            true,
        );
        let grad = Tensor::from_vec(vec![1.0, -1.0, 0.0], &[1, 3]).unwrap();

        let epsilon = 0.5;
        let adv = fgsm_attack(&input, &grad, epsilon);
        let adv_data = adv.data().to_vec();

        assert!((adv_data[0] - 0.5).abs() < 1e-6); // positive grad → +epsilon
        assert!((adv_data[1] - (-0.5)).abs() < 1e-6); // negative grad → -epsilon
        assert!((adv_data[2] - 0.0).abs() < 1e-6); // zero grad → no change
    }

    #[test]
    fn test_fgsm_attack_shape_preserved() {
        let input = Variable::new(Tensor::from_vec(vec![1.0; 12], &[3, 4]).unwrap(), true);
        let grad = Tensor::from_vec(vec![0.1; 12], &[3, 4]).unwrap();

        let adv = fgsm_attack(&input, &grad, 0.01);
        assert_eq!(adv.shape(), vec![3, 4]);
    }

    #[test]
    fn test_pgd_attack_shape() {
        let model = Sequential::new().add(Linear::new(4, 2));

        let input = Variable::new(Tensor::from_vec(vec![1.0; 4], &[1, 4]).unwrap(), true);
        let target = Variable::new(Tensor::from_vec(vec![1.0, 0.0], &[1, 2]).unwrap(), false);

        let adv = pgd_attack(&model, &input, &target, 0.1, 0.03, 3, |pred, tgt| {
            // Simple MSE loss
            let diff = pred.add_var(&tgt.neg());
            let sq = diff.mul_var(&diff);
            sq.mean()
        });

        assert_eq!(adv.shape(), vec![1, 4]);
    }

    #[test]
    fn test_pgd_attack_within_epsilon() {
        let model = Sequential::new().add(Linear::new(4, 2));

        let input_data = vec![1.0, 2.0, 3.0, 4.0];
        let input = Variable::new(Tensor::from_vec(input_data.clone(), &[1, 4]).unwrap(), true);
        let target = Variable::new(Tensor::from_vec(vec![1.0, 0.0], &[1, 2]).unwrap(), false);

        let epsilon = 0.1;
        let adv = pgd_attack(&model, &input, &target, epsilon, 0.03, 5, |pred, tgt| {
            let diff = pred.add_var(&tgt.neg());
            let sq = diff.mul_var(&diff);
            sq.mean()
        });

        let adv_data = adv.data().to_vec();
        for (orig, perturbed) in input_data.iter().zip(adv_data.iter()) {
            let delta = (perturbed - orig).abs();
            assert!(
                delta <= epsilon + 1e-5,
                "PGD perturbation {} exceeds epsilon {}",
                delta,
                epsilon
            );
        }
    }

    #[test]
    fn test_adversarial_trainer_creation() {
        use axonml_optim::Adam;

        let gen_model = Sequential::new().add(Linear::new(10, 4));
        let disc_model = Sequential::new().add(Linear::new(4, 1));

        let gen_opt = Box::new(Adam::new(gen_model.parameters(), 0.001));
        let disc_opt = Box::new(Adam::new(disc_model.parameters(), 0.001));

        let mut trainer = AdversarialTrainer::new(gen_opt, disc_opt);
        trainer.set_disc_steps_per_gen(3);

        let (gen_opt_ref, disc_opt_ref) = trainer.optimizers();
        assert_eq!(gen_opt_ref.parameters().len(), 2); // weight + bias
        assert_eq!(disc_opt_ref.parameters().len(), 2);
    }
}
