//! Vision Datasets
//!
//! Provides dataset loaders for common computer vision datasets.
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

pub mod cifar;
pub mod coco;
pub mod mnist;
pub mod wider_face;

pub use cifar::{SyntheticCIFAR, CIFAR10, CIFAR100};
pub use coco::{CocoAnnotation, CocoDataset};
pub use mnist::{FashionMNIST, SyntheticMNIST, MNIST};
pub use wider_face::WiderFaceDataset;
