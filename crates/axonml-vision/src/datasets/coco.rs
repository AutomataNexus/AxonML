//! COCO Dataset — Object Detection Benchmark
//!
//! # File
//! `crates/axonml-vision/src/datasets/coco.rs`
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

use axonml_tensor::Tensor;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

// =============================================================================
// COCO JSON Schema
// =============================================================================

#[derive(Deserialize)]
struct CocoJson {
    images: Vec<CocoImage>,
    annotations: Vec<CocoAnno>,
    categories: Vec<CocoCategory>,
}

#[derive(Deserialize)]
struct CocoImage {
    id: u64,
    file_name: String,
    width: u32,
    height: u32,
}

#[derive(Deserialize)]
struct CocoAnno {
    image_id: u64,
    category_id: u64,
    bbox: [f32; 4], // [x, y, width, height] in pixels
    #[allow(dead_code)]
    area: f32,
    iscrowd: u32,
}

#[derive(Deserialize)]
struct CocoCategory {
    id: u64,
    #[allow(dead_code)]
    name: String,
}

// =============================================================================
// CocoAnnotation
// =============================================================================

/// A single COCO object annotation.
#[derive(Debug, Clone)]
pub struct CocoAnnotation {
    /// Bounding box [x1, y1, x2, y2] normalized to [0, 1].
    pub bbox: [f32; 4],
    /// Category ID (0-indexed, remapped from COCO's non-contiguous IDs).
    pub category_id: usize,
}

// =============================================================================
// CocoDataset
// =============================================================================

/// COCO format object detection dataset.
pub struct CocoDataset {
    /// Image directory path.
    _image_dir: PathBuf,
    /// Per-image entries: (image_path, original_size, annotations).
    entries: Vec<CocoEntry>,
    /// Target image size (height, width) for resizing.
    target_size: (usize, usize),
    /// Number of categories.
    num_classes: usize,
}

struct CocoEntry {
    image_path: PathBuf,
    _orig_w: u32,
    _orig_h: u32,
    annotations: Vec<CocoAnnotation>,
}

impl CocoDataset {
    /// Create a COCO dataset loader.
    ///
    /// - `image_dir`: Path to image directory (e.g., `train2017/`).
    /// - `annotation_json`: Path to annotation JSON (e.g., `instances_train2017.json`).
    /// - `target_size`: (height, width) for image resizing.
    pub fn new<P: AsRef<Path>>(
        image_dir: P,
        annotation_json: P,
        target_size: (usize, usize),
    ) -> Result<Self, String> {
        let image_dir = image_dir.as_ref().to_path_buf();

        let json_str = std::fs::read_to_string(annotation_json.as_ref())
            .map_err(|e| format!("Failed to read COCO annotations: {e}"))?;

        let coco: CocoJson = serde_json::from_str(&json_str)
            .map_err(|e| format!("Failed to parse COCO JSON: {e}"))?;

        // Build category ID remapping (COCO IDs are non-contiguous)
        let mut cat_remap: HashMap<u64, usize> = HashMap::new();
        let mut sorted_cats: Vec<u64> = coco.categories.iter().map(|c| c.id).collect();
        sorted_cats.sort_unstable();
        for (new_id, &old_id) in sorted_cats.iter().enumerate() {
            cat_remap.insert(old_id, new_id);
        }
        let num_classes = sorted_cats.len();

        // Build image info map
        let _image_map: HashMap<u64, &CocoImage> =
            coco.images.iter().map(|img| (img.id, img)).collect();

        // Group annotations by image
        let mut anno_map: HashMap<u64, Vec<&CocoAnno>> = HashMap::new();
        for anno in &coco.annotations {
            if anno.iscrowd == 0 {
                anno_map.entry(anno.image_id).or_default().push(anno);
            }
        }

        // Build entries
        let mut entries = Vec::new();
        for img in &coco.images {
            let annos = anno_map.get(&img.id);
            let mut annotations = Vec::new();

            if let Some(img_annos) = annos {
                for anno in img_annos {
                    let x = anno.bbox[0];
                    let y = anno.bbox[1];
                    let w = anno.bbox[2];
                    let h = anno.bbox[3];

                    if w > 0.0 && h > 0.0 {
                        if let Some(&cat_id) = cat_remap.get(&anno.category_id) {
                            annotations.push(CocoAnnotation {
                                bbox: [
                                    x / img.width as f32,
                                    y / img.height as f32,
                                    (x + w) / img.width as f32,
                                    (y + h) / img.height as f32,
                                ],
                                category_id: cat_id,
                            });
                        }
                    }
                }
            }

            if !annotations.is_empty() {
                entries.push(CocoEntry {
                    image_path: image_dir.join(&img.file_name),
                    _orig_w: img.width,
                    _orig_h: img.height,
                    annotations,
                });
            }
        }

        Ok(Self {
            _image_dir: image_dir,
            entries,
            target_size,
            num_classes,
        })
    }

    /// Number of images with annotations.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Number of object categories.
    pub fn num_classes(&self) -> usize {
        self.num_classes
    }

    /// Get a sample: (image_tensor [3, H, W], annotations).
    ///
    /// Bounding boxes are normalized to [0, 1].
    pub fn get(&self, index: usize) -> Option<(Tensor<f32>, Vec<CocoAnnotation>)> {
        let entry = self.entries.get(index)?;
        let (th, tw) = self.target_size;

        let img = crate::image_io::load_image_resized(&entry.image_path, th, tw).ok()?;

        Some((img, entry.annotations.clone()))
    }

    /// Get annotations only (without loading image).
    pub fn get_annotations(&self, index: usize) -> Option<&[CocoAnnotation]> {
        self.entries.get(index).map(|e| e.annotations.as_slice())
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coco_json_parse() {
        let json = r#"{
            "images": [
                {"id": 1, "file_name": "img1.jpg", "width": 640, "height": 480},
                {"id": 2, "file_name": "img2.jpg", "width": 320, "height": 240}
            ],
            "annotations": [
                {"image_id": 1, "category_id": 1, "bbox": [10.0, 20.0, 100.0, 200.0], "area": 20000.0, "iscrowd": 0},
                {"image_id": 1, "category_id": 3, "bbox": [50.0, 60.0, 30.0, 40.0], "area": 1200.0, "iscrowd": 0},
                {"image_id": 2, "category_id": 1, "bbox": [5.0, 5.0, 50.0, 50.0], "area": 2500.0, "iscrowd": 0},
                {"image_id": 2, "category_id": 2, "bbox": [0.0, 0.0, 10.0, 10.0], "area": 100.0, "iscrowd": 1}
            ],
            "categories": [
                {"id": 1, "name": "person"},
                {"id": 2, "name": "car"},
                {"id": 3, "name": "dog"}
            ]
        }"#;

        let coco: CocoJson = serde_json::from_str(json).unwrap();
        assert_eq!(coco.images.len(), 2);
        assert_eq!(coco.annotations.len(), 4);
        assert_eq!(coco.categories.len(), 3);

        // Verify non-crowd annotations are 3 (one is iscrowd=1)
        let non_crowd: Vec<_> = coco.annotations.iter().filter(|a| a.iscrowd == 0).collect();
        assert_eq!(non_crowd.len(), 3);
    }

    #[test]
    fn test_coco_category_remapping() {
        // COCO uses non-contiguous category IDs (1-90 with gaps)
        let mut cat_remap: HashMap<u64, usize> = HashMap::new();
        let cats = vec![1, 3, 5, 10]; // Non-contiguous
        for (new_id, &old_id) in cats.iter().enumerate() {
            cat_remap.insert(old_id, new_id);
        }

        assert_eq!(cat_remap[&1], 0);
        assert_eq!(cat_remap[&3], 1);
        assert_eq!(cat_remap[&5], 2);
        assert_eq!(cat_remap[&10], 3);
    }

    #[test]
    fn test_bbox_normalization() {
        // Image 640x480, bbox [10, 20, 100, 200] (x, y, w, h)
        let x = 10.0f32;
        let y = 20.0f32;
        let w = 100.0f32;
        let h = 200.0f32;
        let img_w = 640.0f32;
        let img_h = 480.0f32;

        let normalized = [x / img_w, y / img_h, (x + w) / img_w, (y + h) / img_h];

        assert!((normalized[0] - 10.0 / 640.0).abs() < 1e-5);
        assert!((normalized[1] - 20.0 / 480.0).abs() < 1e-5);
        assert!((normalized[2] - 110.0 / 640.0).abs() < 1e-5);
        assert!((normalized[3] - 220.0 / 480.0).abs() < 1e-5);

        // All values should be in [0, 1]
        assert!(normalized.iter().all(|&v| v >= 0.0 && v <= 1.0));
    }
}
