//! Vision Datasets — Dataset Module Re-exports
//!
//! Module hub for vision datasets. Re-exports `CIFAR10`, `CIFAR100`, and
//! `SyntheticCIFAR` from the cifar module; `MNIST`, `FashionMNIST`, and
//! `SyntheticMNIST` from the mnist module; `CocoDataset` and `CocoAnnotation`
//! from the coco module; and `WiderFaceDataset` from the wider_face module.
//!
//! # File
//! `crates/axonml-vision/src/datasets/mod.rs`
//!
//! # Author
//! Andrew Jewell Sr. — AutomataNexus LLC
//! ORCID: 0009-0005-2158-7060
//!
//! # Updated
//! April 16, 2026 11:15 PM EST
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

// =============================================================================
// Sub-Modules
// =============================================================================

pub mod cifar;
pub mod coco;
pub mod mnist;
pub mod wider_face;

// =============================================================================
// Re-Exports
// =============================================================================

pub use cifar::{CIFAR10, CIFAR100, SyntheticCIFAR};
pub use coco::{CocoAnnotation, CocoDataset};
pub use mnist::{FashionMNIST, MNIST, SyntheticMNIST};
pub use wider_face::WiderFaceDataset;
