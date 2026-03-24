//! WIDER FACE Dataset — Face Detection Benchmark
//!
//! # File
//! `crates/axonml-vision/src/datasets/wider_face.rs`
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
use std::path::{Path, PathBuf};

// =============================================================================
// WiderFaceDataset
// =============================================================================

/// A single WIDER FACE annotation entry.
#[derive(Debug, Clone)]
pub struct WiderFaceEntry {
    /// Relative image path within the dataset.
    pub image_path: PathBuf,
    /// Face bounding boxes as [x1, y1, x2, y2] in pixel coordinates.
    pub bboxes: Vec<[f32; 4]>,
}

/// WIDER FACE dataset for face detection training.
///
/// Parses the WIDER FACE annotation format and loads images on demand.
pub struct WiderFaceDataset {
    /// Root directory containing WIDER_train/images/ etc.
    root: PathBuf,
    /// Parsed entries.
    entries: Vec<WiderFaceEntry>,
    /// Target image size (height, width) for resizing.
    target_size: (usize, usize),
    /// Whether to apply random horizontal flip.
    pub flip_augment: bool,
}

impl WiderFaceDataset {
    /// Create a WIDER FACE dataset loader.
    ///
    /// - `root`: Path to the dataset root (containing WIDER_train/ and wider_face_split/).
    /// - `split`: "train" or "val".
    /// - `target_size`: (height, width) to resize images to.
    pub fn new<P: AsRef<Path>>(
        root: P,
        split: &str,
        target_size: (usize, usize),
    ) -> Result<Self, String> {
        let root = root.as_ref().to_path_buf();

        let anno_filename = match split {
            "train" => "wider_face_train_bbx_gt.txt",
            "val" => "wider_face_val_bbx_gt.txt",
            _ => return Err(format!("Unknown split: {split}. Use 'train' or 'val'.")),
        };

        let anno_path = root.join("wider_face_split").join(anno_filename);
        let image_dir = match split {
            "train" => root.join("WIDER_train").join("images"),
            "val" => root.join("WIDER_val").join("images"),
            _ => unreachable!(),
        };

        let entries = Self::parse_annotations(&anno_path, &image_dir)?;

        Ok(Self {
            root,
            entries,
            target_size,
            flip_augment: false,
        })
    }

    /// Parse WIDER FACE annotation file.
    ///
    /// Format:
    /// ```text
    /// relative/path/to/image.jpg
    /// num_faces
    /// x1 y1 w h blur expression illumination invalid occlusion pose
    /// ...
    /// ```
    fn parse_annotations(
        anno_path: &Path,
        image_dir: &Path,
    ) -> Result<Vec<WiderFaceEntry>, String> {
        let content = std::fs::read_to_string(anno_path)
            .map_err(|e| format!("Failed to read annotations: {e}"))?;

        let lines: Vec<&str> = content.lines().collect();
        let mut entries = Vec::new();
        let mut i = 0;

        while i < lines.len() {
            let image_path = lines[i].trim();
            i += 1;
            if i >= lines.len() {
                break;
            }

            let num_faces: usize = lines[i]
                .trim()
                .parse()
                .map_err(|e| format!("Failed to parse num_faces at line {i}: {e}"))?;
            i += 1;

            let mut bboxes = Vec::new();
            for _ in 0..num_faces {
                if i >= lines.len() {
                    break;
                }
                let parts: Vec<f32> = lines[i]
                    .split_whitespace()
                    .filter_map(|s| s.parse().ok())
                    .collect();
                i += 1;

                if parts.len() >= 4 {
                    let x1 = parts[0];
                    let y1 = parts[1];
                    let w = parts[2];
                    let h = parts[3];

                    // Skip invalid boxes (w=0 or h=0)
                    if w > 0.0 && h > 0.0 {
                        bboxes.push([x1, y1, x1 + w, y1 + h]);
                    }
                }
            }

            // Handle case where num_faces is 0 (one line with "0 0 0 0 0 0 0 0 0 0")
            if num_faces == 0 {
                i += 1; // Skip the dummy line
            }

            if !bboxes.is_empty() {
                entries.push(WiderFaceEntry {
                    image_path: image_dir.join(image_path),
                    bboxes,
                });
            }
        }

        Ok(entries)
    }

    /// Get entry count.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get a single sample: (image_tensor [3, H, W], normalized_bboxes).
    ///
    /// Bounding boxes are normalized to [0, 1] relative to target_size.
    pub fn get(&self, index: usize) -> Option<(Tensor<f32>, Vec<[f32; 4]>)> {
        let entry = self.entries.get(index)?;
        let (th, tw) = self.target_size;

        // Load and resize image
        let img = crate::image_io::load_image_resized(&entry.image_path, th, tw).ok()?;

        // Load original image dimensions for bbox scaling
        let orig_img = image::open(&entry.image_path).ok()?;
        let (orig_w, orig_h) = (orig_img.width() as f32, orig_img.height() as f32);

        // Normalize bboxes to [0, 1] and scale to target size
        let bboxes: Vec<[f32; 4]> = entry
            .bboxes
            .iter()
            .map(|b| {
                [
                    (b[0] / orig_w).clamp(0.0, 1.0),
                    (b[1] / orig_h).clamp(0.0, 1.0),
                    (b[2] / orig_w).clamp(0.0, 1.0),
                    (b[3] / orig_h).clamp(0.0, 1.0),
                ]
            })
            .collect();

        Some((img, bboxes))
    }

    /// Get annotation only (without loading image).
    pub fn get_annotation(&self, index: usize) -> Option<&WiderFaceEntry> {
        self.entries.get(index)
    }

    /// Root directory.
    pub fn root(&self) -> &Path {
        &self.root
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_wider_face_format() {
        // Create temporary annotation file
        let dir = std::env::temp_dir().join("wider_face_test");
        let split_dir = dir.join("wider_face_split");
        let image_dir = dir.join("WIDER_train").join("images");
        std::fs::create_dir_all(&split_dir).ok();
        std::fs::create_dir_all(&image_dir).ok();

        let anno_content = "0--Parade/0_Parade_001.jpg\n3\n\
            78 221 7 8 2 0 0 0 0 0\n\
            78 238 14 17 2 0 0 0 0 0\n\
            113 212 11 15 2 0 0 0 0 0\n\
            0--Parade/0_Parade_002.jpg\n1\n\
            100 100 50 60 0 0 0 0 0 0\n";

        std::fs::write(split_dir.join("wider_face_train_bbx_gt.txt"), anno_content).unwrap();

        let entries = WiderFaceDataset::parse_annotations(
            &split_dir.join("wider_face_train_bbx_gt.txt"),
            &image_dir,
        )
        .unwrap();

        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].bboxes.len(), 3);
        assert_eq!(entries[1].bboxes.len(), 1);

        // Verify bbox conversion (x1, y1, x1+w, y1+h)
        let b = &entries[1].bboxes[0];
        assert!((b[0] - 100.0).abs() < 1e-5);
        assert!((b[1] - 100.0).abs() < 1e-5);
        assert!((b[2] - 150.0).abs() < 1e-5);
        assert!((b[3] - 160.0).abs() < 1e-5);

        // Cleanup
        std::fs::remove_dir_all(&dir).ok();
    }
}
