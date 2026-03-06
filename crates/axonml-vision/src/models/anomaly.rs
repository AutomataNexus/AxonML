//! Anomaly Detection Models
//!
//! # Models
//!
//! - **PatchCore**: Feature-based anomaly detection using a pretrained backbone
//!   and k-NN memory bank. No training on anomalies needed — fit on normal data only.
//!
//! - **StudentTeacher**: Lightweight anomaly detector for edge deployment.
//!   A student network is trained to match a frozen teacher on normal data.
//!   Anomalies cause disagreement between student and teacher features.
//!
//! # Reference
//!
//! - PatchCore: "Towards Total Recall in Industrial Anomaly Detection"
//!   (Roth et al., 2022) <https://arxiv.org/abs/2106.08265>
//! - Student-Teacher: "Uninformed Students" (Bergmann et al., 2020)

use axonml_autograd::Variable;
use axonml_nn::{BatchNorm2d, Conv2d, Linear, Module, Parameter, ReLU};
use axonml_tensor::Tensor;

use crate::ops::AnomalyResult;

// =============================================================================
// PatchCore
// =============================================================================

/// PatchCore anomaly detector.
///
/// Uses a frozen pretrained backbone (e.g., ResNet) to extract patch features
/// from normal images, stored in a memory bank. At inference, anomaly score
/// is the distance from the test patch features to the nearest normal feature.
pub struct PatchCore {
    /// Feature extractor (frozen backbone)
    feature_extractor: PatchCoreBackbone,
    /// Memory bank of normal patch features [M, D]
    memory_bank: Vec<Vec<f32>>,
    /// Feature dimension
    feature_dim: usize,
    /// Anomaly threshold (learned during fit)
    threshold: f32,
    /// Whether the model has been fit
    is_fitted: bool,
}

/// Simplified backbone for feature extraction.
struct PatchCoreBackbone {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    conv2: Conv2d,
    bn2: BatchNorm2d,
    conv3: Conv2d,
    bn3: BatchNorm2d,
    relu: ReLU,
}

impl PatchCoreBackbone {
    fn new(in_channels: usize, feature_dim: usize) -> Self {
        Self {
            conv1: Conv2d::with_options(in_channels, 64, (3, 3), (2, 2), (1, 1), true),
            bn1: BatchNorm2d::new(64),
            conv2: Conv2d::with_options(64, 128, (3, 3), (2, 2), (1, 1), true),
            bn2: BatchNorm2d::new(128),
            conv3: Conv2d::with_options(128, feature_dim, (3, 3), (2, 2), (1, 1), true),
            bn3: BatchNorm2d::new(feature_dim),
            relu: ReLU,
        }
    }

    fn forward(&self, x: &Variable) -> Variable {
        let x = self.relu.forward(&self.bn1.forward(&self.conv1.forward(x)));
        let x = self.relu.forward(&self.bn2.forward(&self.conv2.forward(&x)));
        self.relu.forward(&self.bn3.forward(&self.conv3.forward(&x)))
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        p.extend(self.conv1.parameters());
        p.extend(self.bn1.parameters());
        p.extend(self.conv2.parameters());
        p.extend(self.bn2.parameters());
        p.extend(self.conv3.parameters());
        p.extend(self.bn3.parameters());
        p
    }
}

impl PatchCore {
    /// Create a new PatchCore model.
    ///
    /// # Arguments
    /// - `in_channels`: Input image channels (3 for RGB)
    /// - `feature_dim`: Feature dimension for patch embeddings
    pub fn new(in_channels: usize, feature_dim: usize) -> Self {
        Self {
            feature_extractor: PatchCoreBackbone::new(in_channels, feature_dim),
            memory_bank: Vec::new(),
            feature_dim,
            threshold: 0.0,
            is_fitted: false,
        }
    }

    /// Create PatchCore for RGB images with default settings.
    pub fn default_rgb() -> Self {
        Self::new(3, 256)
    }

    /// Fit the memory bank on normal (non-anomalous) images.
    ///
    /// # Arguments
    /// - `normal_images`: Batch of normal images `[N, C, H, W]`
    /// - `coreset_ratio`: Fraction of patches to keep (0.0-1.0, lower = faster)
    pub fn fit(&mut self, normal_images: &Variable, coreset_ratio: f32) {
        let features = self.extract_patch_features(normal_images);

        // Coreset subsampling: keep a representative subset
        let n = features.len();
        let keep = ((n as f32 * coreset_ratio) as usize).max(1);

        // Simple uniform subsampling (greedy coreset would be better but expensive)
        let step = (n as f32 / keep as f32).ceil() as usize;
        self.memory_bank = features.into_iter().step_by(step.max(1)).collect();

        // Compute threshold as max distance in normal data
        self.threshold = self.compute_threshold();
        self.is_fitted = true;
    }

    /// Predict anomaly for a single image or batch.
    ///
    /// # Arguments
    /// - `images`: `[N, C, H, W]` input images
    ///
    /// # Returns
    /// Vector of `AnomalyResult`, one per image.
    pub fn predict(&self, images: &Variable) -> Vec<AnomalyResult> {
        assert!(self.is_fitted, "PatchCore must be fit before prediction");

        let features = self.extract_patch_features(images);
        let shape = images.shape();
        let n = shape[0];

        // Group features by image
        let patches_per_image = features.len() / n;
        let mut results = Vec::new();

        for img_idx in 0..n {
            let start = img_idx * patches_per_image;
            let end = start + patches_per_image;
            let img_features = &features[start..end];

            // Compute max distance to nearest memory bank entry
            let mut max_dist = 0.0f32;
            let feat_h = (patches_per_image as f32).sqrt() as usize;
            let feat_w = feat_h;
            let mut heatmap_data = vec![0.0f32; feat_h * feat_w];

            for (pidx, patch) in img_features.iter().enumerate() {
                let dist = self.nearest_distance(patch);
                if dist > max_dist {
                    max_dist = dist;
                }
                if pidx < heatmap_data.len() {
                    heatmap_data[pidx] = dist;
                }
            }

            let heatmap = if feat_h > 0 && feat_w > 0 {
                Some(Tensor::from_vec(heatmap_data, &[feat_h, feat_w]).unwrap())
            } else {
                None
            };

            results.push(AnomalyResult {
                score: max_dist,
                is_anomalous: max_dist > self.threshold,
                heatmap,
            });
        }

        results
    }

    /// Extract patch-level features from images.
    fn extract_patch_features(&self, images: &Variable) -> Vec<Vec<f32>> {
        let feat_map = self.feature_extractor.forward(images);
        let data = feat_map.data().to_vec();
        let shape = feat_map.shape();
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);

        let mut patches = Vec::new();
        for b in 0..n {
            for y in 0..h {
                for x in 0..w {
                    let mut patch = vec![0.0f32; c];
                    for ch in 0..c {
                        patch[ch] = data[b * c * h * w + ch * h * w + y * w + x];
                    }
                    patches.push(patch);
                }
            }
        }

        patches
    }

    /// Find distance to nearest neighbor in memory bank.
    fn nearest_distance(&self, query: &[f32]) -> f32 {
        self.memory_bank
            .iter()
            .map(|mem| {
                query
                    .iter()
                    .zip(mem.iter())
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .sum::<f32>()
                    .sqrt()
            })
            .fold(f32::MAX, f32::min)
    }

    /// Compute anomaly threshold from normal data distances.
    fn compute_threshold(&self) -> f32 {
        if self.memory_bank.len() < 2 {
            return 1.0;
        }

        // Threshold = mean + 3*std of nearest neighbor distances among normal patches
        let mut distances = Vec::new();
        for (i, patch) in self.memory_bank.iter().enumerate() {
            let mut min_dist = f32::MAX;
            for (j, other) in self.memory_bank.iter().enumerate() {
                if i == j {
                    continue;
                }
                let dist: f32 = patch
                    .iter()
                    .zip(other.iter())
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .sum::<f32>()
                    .sqrt();
                min_dist = min_dist.min(dist);
            }
            distances.push(min_dist);
        }

        let mean = distances.iter().sum::<f32>() / distances.len() as f32;
        let var = distances.iter().map(|&d| (d - mean) * (d - mean)).sum::<f32>()
            / distances.len() as f32;
        mean + 3.0 * var.sqrt()
    }
}

impl Module for PatchCore {
    fn forward(&self, x: &Variable) -> Variable {
        self.feature_extractor.forward(x)
    }

    fn parameters(&self) -> Vec<Parameter> {
        self.feature_extractor.parameters()
    }

    fn train(&mut self) {}
    fn eval(&mut self) {}
}

// =============================================================================
// Student-Teacher Anomaly Detection
// =============================================================================

/// Student-Teacher anomaly detector for edge deployment.
///
/// A lightweight student network is trained to match a frozen teacher
/// on normal data. Anomalies produce disagreement between the two.
pub struct StudentTeacher {
    /// Teacher network (frozen after initialization)
    teacher: TeacherNet,
    /// Student network (trained to match teacher on normal data)
    student: StudentNet,
    /// Anomaly threshold
    threshold: f32,
}

struct TeacherNet {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    conv2: Conv2d,
    bn2: BatchNorm2d,
    relu: ReLU,
}

struct StudentNet {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    conv2: Conv2d,
    bn2: BatchNorm2d,
    relu: ReLU,
}

impl TeacherNet {
    fn new(in_channels: usize, feature_dim: usize) -> Self {
        Self {
            conv1: Conv2d::with_options(in_channels, 64, (3, 3), (2, 2), (1, 1), true),
            bn1: BatchNorm2d::new(64),
            conv2: Conv2d::with_options(64, feature_dim, (3, 3), (2, 2), (1, 1), true),
            bn2: BatchNorm2d::new(feature_dim),
            relu: ReLU,
        }
    }

    fn forward(&self, x: &Variable) -> Variable {
        let x = self.relu.forward(&self.bn1.forward(&self.conv1.forward(x)));
        self.relu.forward(&self.bn2.forward(&self.conv2.forward(&x)))
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        p.extend(self.conv1.parameters());
        p.extend(self.bn1.parameters());
        p.extend(self.conv2.parameters());
        p.extend(self.bn2.parameters());
        p
    }
}

impl StudentNet {
    fn new(in_channels: usize, feature_dim: usize) -> Self {
        Self {
            conv1: Conv2d::with_options(in_channels, 32, (3, 3), (2, 2), (1, 1), true),
            bn1: BatchNorm2d::new(32),
            conv2: Conv2d::with_options(32, feature_dim, (3, 3), (2, 2), (1, 1), true),
            bn2: BatchNorm2d::new(feature_dim),
            relu: ReLU,
        }
    }

    fn forward(&self, x: &Variable) -> Variable {
        let x = self.relu.forward(&self.bn1.forward(&self.conv1.forward(x)));
        self.relu.forward(&self.bn2.forward(&self.conv2.forward(&x)))
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        p.extend(self.conv1.parameters());
        p.extend(self.bn1.parameters());
        p.extend(self.conv2.parameters());
        p.extend(self.bn2.parameters());
        p
    }
}

impl StudentTeacher {
    /// Create a Student-Teacher anomaly detector.
    pub fn new(in_channels: usize, feature_dim: usize) -> Self {
        Self {
            teacher: TeacherNet::new(in_channels, feature_dim),
            student: StudentNet::new(in_channels, feature_dim),
            threshold: 1.0,
        }
    }

    /// Create for RGB images with default settings.
    pub fn default_rgb() -> Self {
        Self::new(3, 128)
    }

    /// Compute training loss (MSE between teacher and student features).
    ///
    /// Train the student to match the teacher on normal data only.
    pub fn training_loss(&self, normal_images: &Variable) -> Variable {
        let teacher_feat = self.teacher.forward(normal_images);
        let student_feat = self.student.forward(normal_images);

        // MSE loss between features
        let diff = teacher_feat.sub_var(&student_feat);
        let sq = diff.mul_var(&diff);
        sq.mean()
    }

    /// Predict anomaly based on student-teacher disagreement.
    pub fn predict(&self, images: &Variable) -> Vec<AnomalyResult> {
        let teacher_feat = self.teacher.forward(images);
        let student_feat = self.student.forward(images);
        let shape = teacher_feat.shape();
        let n = shape[0];

        let t_data = teacher_feat.data().to_vec();
        let s_data = student_feat.data().to_vec();
        let elems_per_image = t_data.len() / n;

        let mut results = Vec::new();
        for b in 0..n {
            let start = b * elems_per_image;
            let end = start + elems_per_image;

            let mse: f32 = t_data[start..end]
                .iter()
                .zip(s_data[start..end].iter())
                .map(|(&t, &s)| (t - s) * (t - s))
                .sum::<f32>()
                / elems_per_image as f32;

            let score = mse.sqrt();

            results.push(AnomalyResult {
                score,
                is_anomalous: score > self.threshold,
                heatmap: None,
            });
        }

        results
    }

    /// Set anomaly threshold.
    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold;
    }

    /// Get student parameters for training.
    pub fn student_parameters(&self) -> Vec<Parameter> {
        self.student.parameters()
    }
}

impl Module for StudentTeacher {
    fn forward(&self, x: &Variable) -> Variable {
        self.student.forward(x)
    }

    fn parameters(&self) -> Vec<Parameter> {
        // Only student is trainable
        self.student.parameters()
    }

    fn train(&mut self) {
        self.student.bn1.train();
        self.student.bn2.train();
    }

    fn eval(&mut self) {
        self.student.bn1.eval();
        self.student.bn2.eval();
        self.teacher.bn1.eval();
        self.teacher.bn2.eval();
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_patchcore_creation() {
        let model = PatchCore::default_rgb();
        assert_eq!(model.feature_dim, 256);
        assert!(!model.is_fitted);
    }

    #[test]
    fn test_patchcore_fit_and_predict() {
        let mut model = PatchCore::new(1, 32);

        // Fit on "normal" images
        let normal = Variable::new(
            Tensor::from_vec(vec![0.5; 4 * 1 * 32 * 32], &[4, 1, 32, 32]).unwrap(),
            false,
        );
        model.fit(&normal, 0.5);
        assert!(model.is_fitted);
        assert!(!model.memory_bank.is_empty());

        // Predict on test image
        let test = Variable::new(
            Tensor::from_vec(vec![0.5; 1 * 1 * 32 * 32], &[1, 1, 32, 32]).unwrap(),
            false,
        );
        let results = model.predict(&test);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_student_teacher_creation() {
        let model = StudentTeacher::default_rgb();
        let params = model.parameters();
        assert!(!params.is_empty());
    }

    #[test]
    fn test_student_teacher_loss() {
        let model = StudentTeacher::new(1, 32);
        let normal = Variable::new(
            Tensor::from_vec(vec![0.5; 2 * 1 * 16 * 16], &[2, 1, 16, 16]).unwrap(),
            true,
        );
        let loss = model.training_loss(&normal);
        assert!(loss.numel() == 1);
    }

    #[test]
    fn test_student_teacher_predict() {
        let model = StudentTeacher::new(1, 32);
        let test = Variable::new(
            Tensor::from_vec(vec![0.5; 2 * 1 * 16 * 16], &[2, 1, 16, 16]).unwrap(),
            false,
        );
        let results = model.predict(&test);
        assert_eq!(results.len(), 2);
    }
}
