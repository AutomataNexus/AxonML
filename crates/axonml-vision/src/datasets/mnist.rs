//! MNIST Dataset - Handwritten Digit Recognition
//!
//! Provides loaders for the MNIST dataset of handwritten digits.
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

use axonml_data::Dataset;
use axonml_tensor::Tensor;
use byteorder::{BigEndian, ReadBytesExt};
use flate2::read::GzDecoder;
use std::fs::File;
use std::io::Read;
use std::path::Path;

// =============================================================================
// MNIST Dataset
// =============================================================================

/// The MNIST dataset of handwritten digits.
///
/// Contains 60,000 training images and 10,000 test images of 28x28 grayscale digits.
pub struct MNIST {
    images: Vec<Vec<f32>>,
    labels: Vec<u8>,
    train: bool,
}

impl MNIST {
    /// Creates a new MNIST dataset from files in the specified directory.
    ///
    /// Expected files:
    /// - train-images-idx3-ubyte.gz (or uncompressed)
    /// - train-labels-idx1-ubyte.gz (or uncompressed)
    /// - t10k-images-idx3-ubyte.gz (or uncompressed)
    /// - t10k-labels-idx1-ubyte.gz (or uncompressed)
    pub fn new<P: AsRef<Path>>(root: P, train: bool) -> Result<Self, String> {
        let root = root.as_ref();

        let (images_file, labels_file) = if train {
            ("train-images-idx3-ubyte", "train-labels-idx1-ubyte")
        } else {
            ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")
        };

        let images = Self::load_images(root, images_file)?;
        let labels = Self::load_labels(root, labels_file)?;

        if images.len() != labels.len() {
            return Err(format!(
                "Image count ({}) does not match label count ({})",
                images.len(),
                labels.len()
            ));
        }

        Ok(Self {
            images,
            labels,
            train,
        })
    }

    /// Loads images from the IDX file format.
    fn load_images<P: AsRef<Path>>(root: P, base_name: &str) -> Result<Vec<Vec<f32>>, String> {
        let root = root.as_ref();

        // Try gzipped first, then uncompressed
        let file = Self::open_file(root, base_name)?;
        let mut reader: Box<dyn Read> = if base_name.ends_with(".gz") {
            Box::new(GzDecoder::new(file))
        } else {
            // Check if we opened the .gz version
            let path = root.join(format!("{base_name}.gz"));
            if path.exists() {
                let file = File::open(&path).map_err(|e| e.to_string())?;
                Box::new(GzDecoder::new(file))
            } else {
                Box::new(file)
            }
        };

        // Read magic number
        let magic = reader.read_u32::<BigEndian>().map_err(|e| e.to_string())?;
        if magic != 2051 {
            return Err(format!("Invalid magic number for images: {magic}"));
        }

        let num_images = reader.read_u32::<BigEndian>().map_err(|e| e.to_string())? as usize;
        let rows = reader.read_u32::<BigEndian>().map_err(|e| e.to_string())? as usize;
        let cols = reader.read_u32::<BigEndian>().map_err(|e| e.to_string())? as usize;

        let image_size = rows * cols;
        let mut images = Vec::with_capacity(num_images);

        for _ in 0..num_images {
            let mut buffer = vec![0u8; image_size];
            reader.read_exact(&mut buffer).map_err(|e| e.to_string())?;

            // Convert to f32 normalized to [0, 1]
            let image: Vec<f32> = buffer.iter().map(|&b| f32::from(b) / 255.0).collect();
            images.push(image);
        }

        Ok(images)
    }

    /// Loads labels from the IDX file format.
    fn load_labels<P: AsRef<Path>>(root: P, base_name: &str) -> Result<Vec<u8>, String> {
        let root = root.as_ref();

        let file = Self::open_file(root, base_name)?;
        let mut reader: Box<dyn Read> = if base_name.ends_with(".gz") {
            Box::new(GzDecoder::new(file))
        } else {
            let path = root.join(format!("{base_name}.gz"));
            if path.exists() {
                let file = File::open(&path).map_err(|e| e.to_string())?;
                Box::new(GzDecoder::new(file))
            } else {
                Box::new(file)
            }
        };

        // Read magic number
        let magic = reader.read_u32::<BigEndian>().map_err(|e| e.to_string())?;
        if magic != 2049 {
            return Err(format!("Invalid magic number for labels: {magic}"));
        }

        let num_labels = reader.read_u32::<BigEndian>().map_err(|e| e.to_string())? as usize;

        let mut labels = vec![0u8; num_labels];
        reader.read_exact(&mut labels).map_err(|e| e.to_string())?;

        Ok(labels)
    }

    /// Opens a file, trying gzipped version first.
    fn open_file<P: AsRef<Path>>(root: P, base_name: &str) -> Result<File, String> {
        let root = root.as_ref();

        // Try gzipped first
        let gz_path = root.join(format!("{base_name}.gz"));
        if gz_path.exists() {
            return File::open(&gz_path).map_err(|e| e.to_string());
        }

        // Try uncompressed
        let path = root.join(base_name);
        if path.exists() {
            return File::open(&path).map_err(|e| e.to_string());
        }

        Err(format!(
            "Could not find {base_name} or {base_name}.gz in {root:?}"
        ))
    }

    /// Returns whether this is the training set.
    #[must_use]
    pub fn is_train(&self) -> bool {
        self.train
    }

    /// Returns the number of classes (10 for digits 0-9).
    #[must_use]
    pub fn num_classes(&self) -> usize {
        10
    }

    /// Returns the image dimensions (28, 28).
    #[must_use]
    pub fn image_size(&self) -> (usize, usize) {
        (28, 28)
    }
}

impl Dataset for MNIST {
    type Item = (Tensor<f32>, Tensor<f32>);

    fn len(&self) -> usize {
        self.images.len()
    }

    fn get(&self, index: usize) -> Option<Self::Item> {
        if index >= self.images.len() {
            return None;
        }

        // Image as 1x28x28 tensor
        let image = Tensor::from_vec(self.images[index].clone(), &[1, 28, 28]).unwrap();

        // Label as one-hot encoded tensor
        let mut label_vec = vec![0.0f32; 10];
        label_vec[self.labels[index] as usize] = 1.0;
        let label = Tensor::from_vec(label_vec, &[10]).unwrap();

        Some((image, label))
    }
}

// =============================================================================
// FashionMNIST
// =============================================================================

/// The Fashion-MNIST dataset.
///
/// Same format as MNIST but with clothing items instead of digits.
pub struct FashionMNIST {
    inner: MNIST,
}

impl FashionMNIST {
    /// Creates a new `FashionMNIST` dataset.
    ///
    /// Expected files have the same names as MNIST but should be downloaded
    /// from the Fashion-MNIST repository.
    pub fn new<P: AsRef<Path>>(root: P, train: bool) -> Result<Self, String> {
        Ok(Self {
            inner: MNIST::new(root, train)?,
        })
    }

    /// Returns the class names.
    #[must_use]
    pub fn class_names() -> Vec<&'static str> {
        vec![
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot",
        ]
    }

    /// Returns whether this is the training set.
    #[must_use]
    pub fn is_train(&self) -> bool {
        self.inner.is_train()
    }

    /// Returns the number of classes.
    #[must_use]
    pub fn num_classes(&self) -> usize {
        10
    }
}

impl Dataset for FashionMNIST {
    type Item = (Tensor<f32>, Tensor<f32>);

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn get(&self, index: usize) -> Option<Self::Item> {
        self.inner.get(index)
    }
}

// =============================================================================
// Synthetic MNIST (for testing without files)
// =============================================================================

/// A synthetic MNIST-like dataset for testing.
pub struct SyntheticMNIST {
    size: usize,
}

impl SyntheticMNIST {
    /// Creates a synthetic MNIST dataset with the specified size.
    #[must_use]
    pub fn new(size: usize) -> Self {
        Self { size }
    }

    /// Creates a small test dataset (100 samples).
    #[must_use]
    pub fn small() -> Self {
        Self::new(100)
    }

    /// Creates a standard training-size dataset (60000 samples).
    #[must_use]
    pub fn train() -> Self {
        Self::new(60000)
    }

    /// Creates a standard test-size dataset (10000 samples).
    #[must_use]
    pub fn test() -> Self {
        Self::new(10000)
    }
}

impl Dataset for SyntheticMNIST {
    type Item = (Tensor<f32>, Tensor<f32>);

    fn len(&self) -> usize {
        self.size
    }

    fn get(&self, index: usize) -> Option<Self::Item> {
        if index >= self.size {
            return None;
        }

        // Generate a deterministic "random" image based on index
        let label = (index % 10) as u8;
        let seed = index as u32;

        // Create a simple pattern based on the label
        let mut image = vec![0.0f32; 28 * 28];
        for i in 0..28 * 28 {
            // Simple pseudo-random based on seed and position
            let val = ((seed.wrapping_mul(1103515245).wrapping_add(12345 + i as u32)) % 256) as f32
                / 255.0;
            // Add some structure based on label
            let y = i / 28;
            let x = i % 28;
            let center_dist = ((y as i32 - 14).pow(2) + (x as i32 - 14).pow(2)) as f32;
            let label_pattern = (-(center_dist / (50.0 + f32::from(label) * 10.0))).exp();
            image[i] = (val * 0.3 + label_pattern * 0.7).clamp(0.0, 1.0);
        }

        let image_tensor = Tensor::from_vec(image, &[1, 28, 28]).unwrap();

        let mut label_vec = vec![0.0f32; 10];
        label_vec[label as usize] = 1.0;
        let label_tensor = Tensor::from_vec(label_vec, &[10]).unwrap();

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
    fn test_synthetic_mnist() {
        let dataset = SyntheticMNIST::small();

        assert_eq!(dataset.len(), 100);

        let (image, label) = dataset.get(0).unwrap();
        assert_eq!(image.shape(), &[1, 28, 28]);
        assert_eq!(label.shape(), &[10]);

        // Check label is one-hot
        let label_vec = label.to_vec();
        let sum: f32 = label_vec.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_synthetic_mnist_labels() {
        let dataset = SyntheticMNIST::new(20);

        for i in 0..10 {
            let (_, label) = dataset.get(i).unwrap();
            let label_vec = label.to_vec();

            // Label should be one-hot with 1.0 at position i % 10
            assert!((label_vec[i % 10] - 1.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_synthetic_mnist_image_range() {
        let dataset = SyntheticMNIST::small();

        let (image, _) = dataset.get(42).unwrap();
        let image_vec = image.to_vec();

        // All values should be in [0, 1]
        for val in image_vec {
            assert!((0.0..=1.0).contains(&val));
        }
    }

    #[test]
    fn test_synthetic_mnist_deterministic() {
        let dataset = SyntheticMNIST::small();

        let (img1, lbl1) = dataset.get(5).unwrap();
        let (img2, lbl2) = dataset.get(5).unwrap();

        assert_eq!(img1.to_vec(), img2.to_vec());
        assert_eq!(lbl1.to_vec(), lbl2.to_vec());
    }

    #[test]
    fn test_synthetic_mnist_out_of_bounds() {
        let dataset = SyntheticMNIST::new(10);

        assert!(dataset.get(9).is_some());
        assert!(dataset.get(10).is_none());
    }
}
