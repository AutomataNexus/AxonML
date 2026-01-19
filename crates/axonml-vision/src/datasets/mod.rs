//! Vision Datasets
//!
//! Provides dataset loaders for common computer vision datasets.
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

pub mod cifar;
pub mod mnist;

pub use cifar::{SyntheticCIFAR, CIFAR10, CIFAR100};
pub use mnist::{FashionMNIST, SyntheticMNIST, MNIST};
