//! Vision Datasets
//!
//! # File
//! `crates/axonml-vision/src/datasets/mod.rs`
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

pub mod cifar;
pub mod coco;
pub mod mnist;
pub mod wider_face;

pub use cifar::{SyntheticCIFAR, CIFAR10, CIFAR100};
pub use coco::{CocoAnnotation, CocoDataset};
pub use mnist::{FashionMNIST, SyntheticMNIST, MNIST};
pub use wider_face::WiderFaceDataset;
