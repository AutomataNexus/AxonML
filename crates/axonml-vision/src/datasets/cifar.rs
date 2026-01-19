//! CIFAR Datasets - Object Recognition
//!
//! Provides loaders for CIFAR-10 and CIFAR-100 datasets.
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

use axonml_data::Dataset;
use axonml_tensor::Tensor;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

// =============================================================================
// CIFAR-10 Dataset
// =============================================================================

/// The CIFAR-10 dataset.
///
/// Contains 60,000 32x32 color images in 10 classes:
/// airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.
pub struct CIFAR10 {
    images: Vec<Vec<f32>>,
    labels: Vec<u8>,
    train: bool,
}

impl CIFAR10 {
    /// Creates a new CIFAR-10 dataset from files in the specified directory.
    ///
    /// Expected files (extracted from cifar-10-batches-bin):
    /// - `data_batch_1.bin` through `data_batch_5.bin` (for training)
    /// - `test_batch.bin` (for testing)
    pub fn new<P: AsRef<Path>>(root: P, train: bool) -> Result<Self, String> {
        let root = root.as_ref();

        let mut images = Vec::new();
        let mut labels = Vec::new();

        if train {
            // Load all 5 training batches
            for i in 1..=5 {
                let filename = format!("data_batch_{i}.bin");
                let (batch_images, batch_labels) = Self::load_batch(root, &filename)?;
                images.extend(batch_images);
                labels.extend(batch_labels);
            }
        } else {
            // Load test batch
            let (batch_images, batch_labels) = Self::load_batch(root, "test_batch.bin")?;
            images = batch_images;
            labels = batch_labels;
        }

        Ok(Self {
            images,
            labels,
            train,
        })
    }

    /// Loads a single CIFAR batch file.
    fn load_batch<P: AsRef<Path>>(
        root: P,
        filename: &str,
    ) -> Result<(Vec<Vec<f32>>, Vec<u8>), String> {
        let path = root.as_ref().join(filename);
        let file = File::open(&path).map_err(|e| format!("Could not open {path:?}: {e}"))?;
        let mut reader = BufReader::new(file);

        let mut images = Vec::with_capacity(10000);
        let mut labels = Vec::with_capacity(10000);

        // Each record: 1 byte label + 3072 bytes image (32*32*3)
        let record_size = 1 + 32 * 32 * 3;
        let mut buffer = vec![0u8; record_size];

        loop {
            match reader.read_exact(&mut buffer) {
                Ok(()) => {
                    labels.push(buffer[0]);

                    // Image is stored as R, G, B channels (each 1024 bytes)
                    let image: Vec<f32> = buffer[1..].iter().map(|&b| f32::from(b) / 255.0).collect();
                    images.push(image);
                }
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(format!("Error reading batch: {e}")),
            }
        }

        Ok((images, labels))
    }

    /// Returns the class names.
    #[must_use] pub fn class_names() -> Vec<&'static str> {
        vec![
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
    }

    /// Returns whether this is the training set.
    #[must_use] pub fn is_train(&self) -> bool {
        self.train
    }

    /// Returns the number of classes.
    #[must_use] pub fn num_classes(&self) -> usize {
        10
    }

    /// Returns the image dimensions (3, 32, 32).
    #[must_use] pub fn image_size(&self) -> (usize, usize, usize) {
        (3, 32, 32)
    }
}

impl Dataset for CIFAR10 {
    type Item = (Tensor<f32>, Tensor<f32>);

    fn len(&self) -> usize {
        self.images.len()
    }

    fn get(&self, index: usize) -> Option<Self::Item> {
        if index >= self.images.len() {
            return None;
        }

        // Image as 3x32x32 tensor
        let image = Tensor::from_vec(self.images[index].clone(), &[3, 32, 32]).unwrap();

        // Label as one-hot encoded tensor
        let mut label_vec = vec![0.0f32; 10];
        label_vec[self.labels[index] as usize] = 1.0;
        let label = Tensor::from_vec(label_vec, &[10]).unwrap();

        Some((image, label))
    }
}

// =============================================================================
// CIFAR-100 Dataset
// =============================================================================

/// The CIFAR-100 dataset.
///
/// Contains 60,000 32x32 color images in 100 fine classes grouped into 20 coarse classes.
pub struct CIFAR100 {
    images: Vec<Vec<f32>>,
    fine_labels: Vec<u8>,
    coarse_labels: Vec<u8>,
    train: bool,
}

impl CIFAR100 {
    /// Creates a new CIFAR-100 dataset from files in the specified directory.
    ///
    /// Expected files:
    /// - train.bin (for training)
    /// - test.bin (for testing)
    pub fn new<P: AsRef<Path>>(root: P, train: bool) -> Result<Self, String> {
        let root = root.as_ref();
        let filename = if train { "train.bin" } else { "test.bin" };

        let path = root.join(filename);
        let file = File::open(&path).map_err(|e| format!("Could not open {path:?}: {e}"))?;
        let mut reader = BufReader::new(file);

        let mut images = Vec::new();
        let mut fine_labels = Vec::new();
        let mut coarse_labels = Vec::new();

        // Each record: 1 byte coarse label + 1 byte fine label + 3072 bytes image
        let record_size = 2 + 32 * 32 * 3;
        let mut buffer = vec![0u8; record_size];

        loop {
            match reader.read_exact(&mut buffer) {
                Ok(()) => {
                    coarse_labels.push(buffer[0]);
                    fine_labels.push(buffer[1]);

                    let image: Vec<f32> = buffer[2..].iter().map(|&b| f32::from(b) / 255.0).collect();
                    images.push(image);
                }
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(format!("Error reading file: {e}")),
            }
        }

        Ok(Self {
            images,
            fine_labels,
            coarse_labels,
            train,
        })
    }

    /// Returns the number of fine classes.
    #[must_use] pub fn num_fine_classes(&self) -> usize {
        100
    }

    /// Returns the number of coarse classes.
    #[must_use] pub fn num_coarse_classes(&self) -> usize {
        20
    }

    /// Returns whether this is the training set.
    #[must_use] pub fn is_train(&self) -> bool {
        self.train
    }

    /// Gets an item with both fine and coarse labels.
    #[must_use] pub fn get_with_coarse(&self, index: usize) -> Option<(Tensor<f32>, Tensor<f32>, Tensor<f32>)> {
        if index >= self.images.len() {
            return None;
        }

        let image = Tensor::from_vec(self.images[index].clone(), &[3, 32, 32]).unwrap();

        let mut fine_vec = vec![0.0f32; 100];
        fine_vec[self.fine_labels[index] as usize] = 1.0;
        let fine_label = Tensor::from_vec(fine_vec, &[100]).unwrap();

        let mut coarse_vec = vec![0.0f32; 20];
        coarse_vec[self.coarse_labels[index] as usize] = 1.0;
        let coarse_label = Tensor::from_vec(coarse_vec, &[20]).unwrap();

        Some((image, fine_label, coarse_label))
    }
}

impl Dataset for CIFAR100 {
    type Item = (Tensor<f32>, Tensor<f32>);

    fn len(&self) -> usize {
        self.images.len()
    }

    fn get(&self, index: usize) -> Option<Self::Item> {
        if index >= self.images.len() {
            return None;
        }

        let image = Tensor::from_vec(self.images[index].clone(), &[3, 32, 32]).unwrap();

        // Use fine labels by default
        let mut label_vec = vec![0.0f32; 100];
        label_vec[self.fine_labels[index] as usize] = 1.0;
        let label = Tensor::from_vec(label_vec, &[100]).unwrap();

        Some((image, label))
    }
}

// =============================================================================
// Synthetic CIFAR (for testing without files)
// =============================================================================

/// A synthetic CIFAR-like dataset for testing.
pub struct SyntheticCIFAR {
    size: usize,
    num_classes: usize,
}

impl SyntheticCIFAR {
    /// Creates a synthetic CIFAR-10 dataset with the specified size.
    #[must_use] pub fn cifar10(size: usize) -> Self {
        Self {
            size,
            num_classes: 10,
        }
    }

    /// Creates a synthetic CIFAR-100 dataset with the specified size.
    #[must_use] pub fn cifar100(size: usize) -> Self {
        Self {
            size,
            num_classes: 100,
        }
    }

    /// Creates a small test dataset (100 samples, CIFAR-10 style).
    #[must_use] pub fn small() -> Self {
        Self::cifar10(100)
    }

    /// Creates a standard training-size dataset (50000 samples).
    #[must_use] pub fn train() -> Self {
        Self::cifar10(50000)
    }

    /// Creates a standard test-size dataset (10000 samples).
    #[must_use] pub fn test() -> Self {
        Self::cifar10(10000)
    }

    /// Returns the class names for CIFAR-10.
    #[must_use] pub fn class_names(&self) -> Option<Vec<&'static str>> {
        if self.num_classes == 10 {
            Some(CIFAR10::class_names())
        } else {
            None
        }
    }
}

impl Dataset for SyntheticCIFAR {
    type Item = (Tensor<f32>, Tensor<f32>);

    fn len(&self) -> usize {
        self.size
    }

    fn get(&self, index: usize) -> Option<Self::Item> {
        if index >= self.size {
            return None;
        }

        let label = (index % self.num_classes) as u8;
        let seed = index as u32;

        // Generate a 3x32x32 image with color patterns based on class
        let mut image = vec![0.0f32; 3 * 32 * 32];

        // Base colors per class (for first 10 classes)
        let class_colors: [(f32, f32, f32); 10] = [
            (0.8, 0.2, 0.2), // airplane - red
            (0.2, 0.2, 0.8), // automobile - blue
            (0.2, 0.8, 0.2), // bird - green
            (0.8, 0.5, 0.2), // cat - orange
            (0.5, 0.3, 0.1), // deer - brown
            (0.7, 0.7, 0.2), // dog - yellow
            (0.2, 0.6, 0.2), // frog - green-ish
            (0.6, 0.4, 0.2), // horse - tan
            (0.3, 0.3, 0.3), // ship - gray
            (0.5, 0.5, 0.8), // truck - light blue
        ];

        let (r_base, g_base, b_base) = class_colors[(label as usize) % 10];

        for c in 0..3 {
            let channel_base = match c {
                0 => r_base,
                1 => g_base,
                _ => b_base,
            };

            for y in 0..32 {
                for x in 0..32 {
                    let i = y * 32 + x;
                    let noise_seed = seed
                        .wrapping_mul(1103515245)
                        .wrapping_add(12345 + (c * 1024 + i) as u32);
                    let noise = ((noise_seed % 256) as f32 / 255.0 - 0.5) * 0.3;

                    // Add some structure
                    let center_x = (x as f32 - 16.0) / 16.0;
                    let center_y = (y as f32 - 16.0) / 16.0;
                    let dist = (center_x * center_x + center_y * center_y).sqrt();
                    let pattern = (1.0 - dist).max(0.0);

                    let value = channel_base * (0.5 + 0.5 * pattern) + noise;
                    image[c * 32 * 32 + i] = value.clamp(0.0, 1.0);
                }
            }
        }

        let image_tensor = Tensor::from_vec(image, &[3, 32, 32]).unwrap();

        let mut label_vec = vec![0.0f32; self.num_classes];
        label_vec[label as usize] = 1.0;
        let label_tensor = Tensor::from_vec(label_vec, &[self.num_classes]).unwrap();

        Some((image_tensor, label_tensor))
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_cifar10() {
        let dataset = SyntheticCIFAR::small();

        assert_eq!(dataset.len(), 100);

        let (image, label) = dataset.get(0).unwrap();
        assert_eq!(image.shape(), &[3, 32, 32]);
        assert_eq!(label.shape(), &[10]);

        // Check label is one-hot
        let label_vec = label.to_vec();
        let sum: f32 = label_vec.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_synthetic_cifar100() {
        let dataset = SyntheticCIFAR::cifar100(100);

        let (image, label) = dataset.get(0).unwrap();
        assert_eq!(image.shape(), &[3, 32, 32]);
        assert_eq!(label.shape(), &[100]);
    }

    #[test]
    fn test_synthetic_cifar_image_range() {
        let dataset = SyntheticCIFAR::small();

        let (image, _) = dataset.get(42).unwrap();
        let image_vec = image.to_vec();

        // All values should be in [0, 1]
        for val in image_vec {
            assert!((0.0..=1.0).contains(&val));
        }
    }

    #[test]
    fn test_synthetic_cifar_deterministic() {
        let dataset = SyntheticCIFAR::small();

        let (img1, lbl1) = dataset.get(5).unwrap();
        let (img2, lbl2) = dataset.get(5).unwrap();

        assert_eq!(img1.to_vec(), img2.to_vec());
        assert_eq!(lbl1.to_vec(), lbl2.to_vec());
    }

    #[test]
    fn test_synthetic_cifar_labels() {
        let dataset = SyntheticCIFAR::cifar10(20);

        for i in 0..10 {
            let (_, label) = dataset.get(i).unwrap();
            let label_vec = label.to_vec();
            assert!((label_vec[i % 10] - 1.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_cifar10_class_names() {
        let names = CIFAR10::class_names();
        assert_eq!(names.len(), 10);
        assert_eq!(names[0], "airplane");
        assert_eq!(names[9], "truck");
    }

    #[test]
    fn test_synthetic_cifar_different_classes() {
        let dataset = SyntheticCIFAR::small();

        // Different classes should have different color patterns
        let (img0, _) = dataset.get(0).unwrap(); // class 0
        let (img1, _) = dataset.get(1).unwrap(); // class 1

        // They should be different
        assert_ne!(img0.to_vec(), img1.to_vec());
    }
}
