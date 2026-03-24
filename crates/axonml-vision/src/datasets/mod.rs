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

pub use cifar::{CIFAR10, CIFAR100, SyntheticCIFAR};
pub use coco::{CocoAnnotation, CocoDataset};
pub use mnist::{FashionMNIST, MNIST, SyntheticMNIST};
pub use wider_face::WiderFaceDataset;
