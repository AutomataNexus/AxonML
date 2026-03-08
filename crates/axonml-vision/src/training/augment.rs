//! Detection Augmentations — Mosaic, MixUp, HSV Jitter, Random Affine
//!
//! YOLO-style data augmentation pipeline for object detection training.
//! All augmentations operate on (image, boxes, classes) triples and
//! correctly transform bounding boxes alongside images.
//!
//! @version 0.1.0

use axonml_tensor::Tensor;
use rand::Rng;

// =============================================================================
// Detection Sample
// =============================================================================

/// A detection sample: image + annotations.
#[derive(Debug, Clone)]
pub struct DetSample {
    /// Image tensor [C, H, W] in [0, 1].
    pub image: Tensor<f32>,
    /// Bounding boxes [M, 4] as (x1, y1, x2, y2) in pixel coords.
    pub boxes: Vec<[f32; 4]>,
    /// Class labels [M].
    pub classes: Vec<usize>,
}

impl DetSample {
    /// Create from components.
    pub fn new(image: Tensor<f32>, boxes: Vec<[f32; 4]>, classes: Vec<usize>) -> Self {
        Self { image, boxes, classes }
    }

    /// Image height.
    pub fn height(&self) -> usize {
        self.image.shape()[1]
    }

    /// Image width.
    pub fn width(&self) -> usize {
        self.image.shape()[2]
    }

    /// Number of annotations.
    pub fn num_objects(&self) -> usize {
        self.boxes.len()
    }
}

// =============================================================================
// Mosaic Augmentation
// =============================================================================

/// 4-image mosaic augmentation (YOLOv4+).
///
/// Combines 4 images into a single mosaic by placing them in quadrants
/// around a randomly chosen center point. Dramatically improves detection
/// of small objects and reduces need for large batch sizes.
pub struct Mosaic {
    /// Target output size (height, width).
    pub target_size: (usize, usize),
}

impl Mosaic {
    /// Create mosaic with target output size.
    pub fn new(target_h: usize, target_w: usize) -> Self {
        Self { target_size: (target_h, target_w) }
    }

    /// Apply mosaic to 4 samples. Returns a single merged sample.
    ///
    /// Panics if `samples.len() != 4`.
    pub fn apply(&self, samples: &[DetSample]) -> DetSample {
        assert_eq!(samples.len(), 4, "Mosaic requires exactly 4 samples");

        let (th, tw) = self.target_size;
        let channels = samples[0].image.shape()[0];
        let mut rng = rand::thread_rng();

        // Random center point (within 25%-75% of image)
        let cx = rng.gen_range(tw / 4..3 * tw / 4);
        let cy = rng.gen_range(th / 4..3 * th / 4);

        let mut mosaic_data = vec![0.5f32; channels * th * tw]; // grey fill
        let mut all_boxes = Vec::new();
        let mut all_classes = Vec::new();

        // Quadrant regions: (dst_x_start, dst_y_start, dst_w, dst_h)
        let regions = [
            (0, 0, cx, cy),           // top-left
            (cx, 0, tw - cx, cy),     // top-right
            (0, cy, cx, th - cy),     // bottom-left
            (cx, cy, tw - cx, th - cy), // bottom-right
        ];

        for (i, sample) in samples.iter().enumerate() {
            let (dx, dy, dw, dh) = regions[i];
            if dw == 0 || dh == 0 {
                continue;
            }

            let src_h = sample.height();
            let src_w = sample.width();
            let src_data = sample.image.to_vec();

            // Scale factors from source to destination region
            let sx = src_w as f32 / dw as f32;
            let sy = src_h as f32 / dh as f32;

            // Copy pixels with nearest-neighbor resize
            for c in 0..channels {
                for y in 0..dh {
                    let src_y = ((y as f32 * sy) as usize).min(src_h - 1);
                    for x in 0..dw {
                        let src_x = ((x as f32 * sx) as usize).min(src_w - 1);
                        let src_idx = c * src_h * src_w + src_y * src_w + src_x;
                        let dst_idx = c * th * tw + (dy + y) * tw + (dx + x);
                        mosaic_data[dst_idx] = src_data[src_idx];
                    }
                }
            }

            // Transform bounding boxes
            for (bi, bbox) in sample.boxes.iter().enumerate() {
                let [bx1, by1, bx2, by2] = *bbox;

                // Scale to destination region
                let nx1 = dx as f32 + bx1 / sx;
                let ny1 = dy as f32 + by1 / sy;
                let nx2 = dx as f32 + bx2 / sx;
                let ny2 = dy as f32 + by2 / sy;

                // Clip to mosaic bounds
                let nx1 = nx1.max(0.0).min(tw as f32);
                let ny1 = ny1.max(0.0).min(th as f32);
                let nx2 = nx2.max(0.0).min(tw as f32);
                let ny2 = ny2.max(0.0).min(th as f32);

                // Keep only if box has meaningful area
                if (nx2 - nx1) > 2.0 && (ny2 - ny1) > 2.0 {
                    all_boxes.push([nx1, ny1, nx2, ny2]);
                    all_classes.push(sample.classes[bi]);
                }
            }
        }

        let image = Tensor::from_vec(mosaic_data, &[channels, th, tw]).unwrap();
        DetSample::new(image, all_boxes, all_classes)
    }
}

// =============================================================================
// MixUp Augmentation
// =============================================================================

/// MixUp augmentation for detection.
///
/// Blends two images and concatenates their annotations.
/// image_out = α * image1 + (1-α) * image2
///
/// From Zhang et al., "mixup: Beyond Empirical Risk Minimization" (ICLR 2018).
pub struct MixUp {
    /// Beta distribution parameter (higher = more uniform mixing).
    pub beta: f32,
}

impl MixUp {
    /// Create with default beta=1.5.
    pub fn new() -> Self {
        Self { beta: 1.5 }
    }

    /// Create with custom beta.
    pub fn with_beta(beta: f32) -> Self {
        Self { beta }
    }

    /// Apply MixUp to two samples. Returns blended result.
    pub fn apply(&self, a: &DetSample, b: &DetSample) -> DetSample {
        let mut rng = rand::thread_rng();

        // Sample alpha from Beta(beta, beta) — approximate with uniform for simplicity
        let alpha: f32 = rng.gen_range(0.4..0.6);

        let shape_a = a.image.shape().to_vec();
        let (c, h, w) = (shape_a[0], shape_a[1], shape_a[2]);

        // Resize b to match a if needed
        let b_data = if b.height() != h || b.width() != w {
            resize_chw(&b.image, h, w)
        } else {
            b.image.to_vec()
        };

        let a_data = a.image.to_vec();

        // Blend pixels
        let blended: Vec<f32> = a_data.iter().zip(b_data.iter())
            .map(|(&va, &vb)| alpha * va + (1.0 - alpha) * vb)
            .collect();

        let image = Tensor::from_vec(blended, &[c, h, w]).unwrap();

        // Concatenate annotations (scale b's boxes if images were different sizes)
        let mut boxes = a.boxes.clone();
        let mut classes = a.classes.clone();

        let sx = w as f32 / b.width() as f32;
        let sy = h as f32 / b.height() as f32;

        for (bi, bbox) in b.boxes.iter().enumerate() {
            boxes.push([
                bbox[0] * sx,
                bbox[1] * sy,
                bbox[2] * sx,
                bbox[3] * sy,
            ]);
            classes.push(b.classes[bi]);
        }

        DetSample::new(image, boxes, classes)
    }
}

impl Default for MixUp {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Random Horizontal Flip
// =============================================================================

/// Random horizontal flip with bounding box adjustment.
pub struct DetRandomHFlip {
    /// Flip probability.
    pub prob: f32,
}

impl DetRandomHFlip {
    /// Create with probability 0.5.
    pub fn new() -> Self {
        Self { prob: 0.5 }
    }

    /// Apply to a detection sample.
    pub fn apply(&self, sample: &DetSample) -> DetSample {
        let mut rng = rand::thread_rng();
        if rng.gen::<f32>() > self.prob {
            return sample.clone();
        }

        let shape = sample.image.shape();
        let (c, h, w) = (shape[0], shape[1], shape[2]);
        let data = sample.image.to_vec();

        // Flip image horizontally
        let mut flipped = vec![0.0f32; c * h * w];
        for ch in 0..c {
            for y in 0..h {
                for x in 0..w {
                    flipped[ch * h * w + y * w + x] = data[ch * h * w + y * w + (w - 1 - x)];
                }
            }
        }

        // Flip bounding boxes
        let w_f = w as f32;
        let boxes: Vec<[f32; 4]> = sample.boxes.iter().map(|[x1, y1, x2, y2]| {
            [w_f - x2, *y1, w_f - x1, *y2]
        }).collect();

        let image = Tensor::from_vec(flipped, &[c, h, w]).unwrap();
        DetSample::new(image, boxes, sample.classes.clone())
    }
}

impl Default for DetRandomHFlip {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// HSV Jitter
// =============================================================================

/// HSV color space jitter for detection images.
///
/// Randomly adjusts hue, saturation, and value channels.
/// Standard YOLO augmentation with h_gain=0.015, s_gain=0.7, v_gain=0.4.
pub struct HSVJitter {
    /// Hue shift range (fraction of 360°).
    pub h_gain: f32,
    /// Saturation multiplier range.
    pub s_gain: f32,
    /// Value multiplier range.
    pub v_gain: f32,
}

impl HSVJitter {
    /// Create with YOLO defaults.
    pub fn new() -> Self {
        Self {
            h_gain: 0.015,
            s_gain: 0.7,
            v_gain: 0.4,
        }
    }

    /// Apply HSV jitter. Image must be [3, H, W] RGB in [0, 1].
    pub fn apply(&self, sample: &DetSample) -> DetSample {
        let shape = sample.image.shape();
        let (c, h, w) = (shape[0], shape[1], shape[2]);
        if c != 3 {
            return sample.clone();
        }

        let mut rng = rand::thread_rng();
        let h_shift = rng.gen_range(-self.h_gain..self.h_gain);
        let s_scale = rng.gen_range(1.0 - self.s_gain..1.0 + self.s_gain);
        let v_scale = rng.gen_range(1.0 - self.v_gain..1.0 + self.v_gain);

        let data = sample.image.to_vec();
        let mut result = vec![0.0f32; c * h * w];

        for y in 0..h {
            for x in 0..w {
                let r = data[0 * h * w + y * w + x];
                let g = data[1 * h * w + y * w + x];
                let b = data[2 * h * w + y * w + x];

                // RGB -> HSV
                let (hue, sat, val) = rgb_to_hsv(r, g, b);

                // Apply jitter
                let hue = ((hue + h_shift) % 1.0 + 1.0) % 1.0;
                let sat = (sat * s_scale).clamp(0.0, 1.0);
                let val = (val * v_scale).clamp(0.0, 1.0);

                // HSV -> RGB
                let (nr, ng, nb) = hsv_to_rgb(hue, sat, val);

                result[0 * h * w + y * w + x] = nr;
                result[1 * h * w + y * w + x] = ng;
                result[2 * h * w + y * w + x] = nb;
            }
        }

        let image = Tensor::from_vec(result, &[c, h, w]).unwrap();
        DetSample::new(image, sample.boxes.clone(), sample.classes.clone())
    }
}

impl Default for HSVJitter {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Random Affine
// =============================================================================

/// Random affine transformation with bounding box adjustment.
///
/// Applies random rotation, scale, translation, and shear.
/// Bounding boxes are transformed and re-clipped.
pub struct DetRandomAffine {
    /// Rotation range in degrees.
    pub degrees: f32,
    /// Scale range: (1-scale, 1+scale).
    pub scale: f32,
    /// Translation range (fraction of image size).
    pub translate: f32,
    /// Shear range in degrees.
    pub shear: f32,
}

impl DetRandomAffine {
    /// Create with YOLO defaults.
    pub fn new() -> Self {
        Self {
            degrees: 0.0,
            scale: 0.5,
            translate: 0.1,
            shear: 0.0,
        }
    }

    /// Create with custom parameters.
    pub fn with_params(degrees: f32, scale: f32, translate: f32, shear: f32) -> Self {
        Self { degrees, scale, translate, shear }
    }

    /// Apply random affine. Returns transformed sample at the same size.
    pub fn apply(&self, sample: &DetSample) -> DetSample {
        let shape = sample.image.shape();
        let (c, h, w) = (shape[0], shape[1], shape[2]);
        let data = sample.image.to_vec();
        let mut rng = rand::thread_rng();

        // Random parameters (guard empty ranges)
        let angle = if self.degrees > 0.0 {
            rng.gen_range(-self.degrees..self.degrees) * std::f32::consts::PI / 180.0
        } else { 0.0 };
        let scale = if self.scale > 0.0 {
            rng.gen_range(1.0 - self.scale..1.0 + self.scale)
        } else { 1.0 };
        let tx = if self.translate > 0.0 {
            rng.gen_range(-self.translate..self.translate) * w as f32
        } else { 0.0 };
        let ty = if self.translate > 0.0 {
            rng.gen_range(-self.translate..self.translate) * h as f32
        } else { 0.0 };
        let shear_x = if self.shear > 0.0 {
            rng.gen_range(-self.shear..self.shear) * std::f32::consts::PI / 180.0
        } else { 0.0 };
        let shear_y = if self.shear > 0.0 {
            rng.gen_range(-self.shear..self.shear) * std::f32::consts::PI / 180.0
        } else { 0.0 };

        // Affine matrix (2x3): maps dst -> src
        let cos_a = angle.cos() * scale;
        let sin_a = angle.sin() * scale;

        // Center of image
        let cx = w as f32 * 0.5;
        let cy = h as f32 * 0.5;

        // Combined rotation+scale+shear around center, then translate
        let m00 = cos_a + shear_x.tan() * sin_a;
        let m01 = -sin_a + shear_x.tan() * cos_a;
        let m10 = sin_a + shear_y.tan() * cos_a;
        let m11 = cos_a - shear_y.tan() * sin_a;

        // Offset so rotation is around center
        let m02 = cx - m00 * cx - m01 * cy + tx;
        let m12 = cy - m10 * cx - m11 * cy + ty;

        // Inverse affine for backward mapping (dst -> src)
        let det = m00 * m11 - m01 * m10;
        if det.abs() < 1e-8 {
            return sample.clone();
        }
        let inv_det = 1.0 / det;
        let i00 = m11 * inv_det;
        let i01 = -m01 * inv_det;
        let i10 = -m10 * inv_det;
        let i11 = m00 * inv_det;
        let i02 = -(i00 * m02 + i01 * m12);
        let i12 = -(i10 * m02 + i11 * m12);

        // Warp image using inverse mapping
        let mut warped = vec![0.5f32; c * h * w]; // grey fill
        for y_dst in 0..h {
            for x_dst in 0..w {
                let x_src = i00 * x_dst as f32 + i01 * y_dst as f32 + i02;
                let y_src = i10 * x_dst as f32 + i11 * y_dst as f32 + i12;

                let xi = x_src as i32;
                let yi = y_src as i32;
                if xi >= 0 && xi < w as i32 && yi >= 0 && yi < h as i32 {
                    for ch in 0..c {
                        warped[ch * h * w + y_dst * w + x_dst] =
                            data[ch * h * w + yi as usize * w + xi as usize];
                    }
                }
            }
        }

        // Transform bounding boxes: apply forward affine to all 4 corners
        let mut new_boxes = Vec::new();
        let mut new_classes = Vec::new();

        for (bi, bbox) in sample.boxes.iter().enumerate() {
            let [x1, y1, x2, y2] = *bbox;
            let corners = [
                (x1, y1), (x2, y1),
                (x1, y2), (x2, y2),
            ];

            let mut min_x = f32::MAX;
            let mut min_y = f32::MAX;
            let mut max_x = f32::MIN;
            let mut max_y = f32::MIN;

            for (px, py) in &corners {
                let nx = m00 * px + m01 * py + m02;
                let ny = m10 * px + m11 * py + m12;
                min_x = min_x.min(nx);
                min_y = min_y.min(ny);
                max_x = max_x.max(nx);
                max_y = max_y.max(ny);
            }

            // Clip to image bounds
            min_x = min_x.max(0.0).min(w as f32);
            min_y = min_y.max(0.0).min(h as f32);
            max_x = max_x.max(0.0).min(w as f32);
            max_y = max_y.max(0.0).min(h as f32);

            // Keep if remaining area is meaningful
            let new_area = (max_x - min_x) * (max_y - min_y);
            let orig_area = (x2 - x1) * (y2 - y1);
            if new_area > 4.0 && new_area > orig_area * 0.1 {
                new_boxes.push([min_x, min_y, max_x, max_y]);
                new_classes.push(sample.classes[bi]);
            }
        }

        let image = Tensor::from_vec(warped, &[c, h, w]).unwrap();
        DetSample::new(image, new_boxes, new_classes)
    }
}

impl Default for DetRandomAffine {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// LetterBox (resize with padding)
// =============================================================================

/// Letterbox resize: scales image to fit target size while maintaining
/// aspect ratio, padding with grey (0.5).
pub struct LetterBox {
    /// Target size (height, width).
    pub target_size: (usize, usize),
}

impl LetterBox {
    /// Create letterbox for target size.
    pub fn new(target_h: usize, target_w: usize) -> Self {
        Self { target_size: (target_h, target_w) }
    }

    /// Apply letterbox resize.
    pub fn apply(&self, sample: &DetSample) -> DetSample {
        let (th, tw) = self.target_size;
        let src_h = sample.height();
        let src_w = sample.width();

        // Scale to fit
        let scale = (tw as f32 / src_w as f32).min(th as f32 / src_h as f32);
        let new_w = (src_w as f32 * scale) as usize;
        let new_h = (src_h as f32 * scale) as usize;

        // Padding offsets (center the image)
        let pad_x = (tw - new_w) / 2;
        let pad_y = (th - new_h) / 2;

        let channels = sample.image.shape()[0];
        let src_data = sample.image.to_vec();

        let mut dst_data = vec![0.5f32; channels * th * tw]; // grey fill

        // Nearest-neighbor resize into center
        for c in 0..channels {
            for y in 0..new_h {
                let sy = ((y as f32 / scale) as usize).min(src_h - 1);
                for x in 0..new_w {
                    let sx = ((x as f32 / scale) as usize).min(src_w - 1);
                    dst_data[c * th * tw + (pad_y + y) * tw + (pad_x + x)] =
                        src_data[c * src_h * src_w + sy * src_w + sx];
                }
            }
        }

        // Transform boxes
        let boxes: Vec<[f32; 4]> = sample.boxes.iter().map(|[x1, y1, x2, y2]| {
            [
                x1 * scale + pad_x as f32,
                y1 * scale + pad_y as f32,
                x2 * scale + pad_x as f32,
                y2 * scale + pad_y as f32,
            ]
        }).collect();

        let image = Tensor::from_vec(dst_data, &[channels, th, tw]).unwrap();
        DetSample::new(image, boxes, sample.classes.clone())
    }
}

// =============================================================================
// Detection Augmentation Pipeline
// =============================================================================

/// Complete detection augmentation pipeline (YOLO-style).
///
/// Applies: Mosaic(optional) → MixUp(optional) → HSV Jitter → Random Affine → H-Flip → LetterBox
pub struct DetAugPipeline {
    /// Enable mosaic augmentation.
    pub use_mosaic: bool,
    /// Mosaic probability.
    pub mosaic_prob: f32,
    /// Enable MixUp.
    pub use_mixup: bool,
    /// MixUp probability (applied after mosaic).
    pub mixup_prob: f32,
    /// HSV jitter.
    pub hsv: HSVJitter,
    /// Random affine.
    pub affine: DetRandomAffine,
    /// Horizontal flip.
    pub hflip: DetRandomHFlip,
    /// Letterbox to target size.
    pub letterbox: LetterBox,
}

impl DetAugPipeline {
    /// Create with YOLO-style defaults for a given target size.
    pub fn yolo(target_h: usize, target_w: usize) -> Self {
        Self {
            use_mosaic: true,
            mosaic_prob: 1.0,
            use_mixup: true,
            mixup_prob: 0.1,
            hsv: HSVJitter::new(),
            affine: DetRandomAffine::new(),
            hflip: DetRandomHFlip::new(),
            letterbox: LetterBox::new(target_h, target_w),
        }
    }

    /// Simple pipeline: no mosaic/mixup, just basic augmentation.
    pub fn simple(target_h: usize, target_w: usize) -> Self {
        Self {
            use_mosaic: false,
            mosaic_prob: 0.0,
            use_mixup: false,
            mixup_prob: 0.0,
            hsv: HSVJitter::new(),
            affine: DetRandomAffine::with_params(0.0, 0.2, 0.05, 0.0),
            hflip: DetRandomHFlip::new(),
            letterbox: LetterBox::new(target_h, target_w),
        }
    }

    /// Apply augmentation to a single sample (no mosaic/mixup).
    pub fn apply_single(&self, sample: &DetSample) -> DetSample {
        let s = self.hsv.apply(sample);
        let s = self.affine.apply(&s);
        let s = self.hflip.apply(&s);
        self.letterbox.apply(&s)
    }

    /// Apply full pipeline with 4 samples for mosaic.
    ///
    /// - `primary`: The main sample.
    /// - `others`: 3 additional samples for mosaic (ignored if mosaic disabled).
    /// - `mixup_partner`: Optional sample for mixup (ignored if mixup disabled).
    pub fn apply(
        &self,
        primary: &DetSample,
        others: &[DetSample],
        mixup_partner: Option<&DetSample>,
    ) -> DetSample {
        let mut rng = rand::thread_rng();

        // Step 1: Mosaic
        let mut sample = if self.use_mosaic && others.len() >= 3 && rng.gen::<f32>() < self.mosaic_prob {
            let (th, tw) = self.letterbox.target_size;
            let mosaic = Mosaic::new(th, tw);
            let four = vec![
                primary.clone(),
                others[0].clone(),
                others[1].clone(),
                others[2].clone(),
            ];
            mosaic.apply(&four)
        } else {
            primary.clone()
        };

        // Step 2: MixUp
        if self.use_mixup && rng.gen::<f32>() < self.mixup_prob {
            if let Some(partner) = mixup_partner {
                let mixup = MixUp::new();
                sample = mixup.apply(&sample, partner);
            }
        }

        // Step 3: HSV + Affine + Flip + Letterbox
        self.apply_single(&sample)
    }
}

// =============================================================================
// Helpers
// =============================================================================

/// RGB [0,1] → HSV [0,1].
fn rgb_to_hsv(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;

    let v = max;
    let s = if max > 0.0 { delta / max } else { 0.0 };

    let h = if delta < 1e-6 {
        0.0
    } else if (max - r).abs() < 1e-6 {
        ((g - b) / delta % 6.0) / 6.0
    } else if (max - g).abs() < 1e-6 {
        ((b - r) / delta + 2.0) / 6.0
    } else {
        ((r - g) / delta + 4.0) / 6.0
    };

    let h = ((h % 1.0) + 1.0) % 1.0;
    (h, s, v)
}

/// HSV [0,1] → RGB [0,1].
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let c = v * s;
    let h6 = h * 6.0;
    let x = c * (1.0 - (h6 % 2.0 - 1.0).abs());
    let m = v - c;

    let (r, g, b) = if h6 < 1.0 {
        (c, x, 0.0)
    } else if h6 < 2.0 {
        (x, c, 0.0)
    } else if h6 < 3.0 {
        (0.0, c, x)
    } else if h6 < 4.0 {
        (0.0, x, c)
    } else if h6 < 5.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };

    (r + m, g + m, b + m)
}

/// Nearest-neighbor resize for CHW tensor, returns flat vec.
fn resize_chw(tensor: &Tensor<f32>, new_h: usize, new_w: usize) -> Vec<f32> {
    let shape = tensor.shape();
    let (c, h, w) = (shape[0], shape[1], shape[2]);
    let data = tensor.to_vec();
    let mut out = vec![0.0f32; c * new_h * new_w];

    for ch in 0..c {
        for y in 0..new_h {
            let sy = (y as f32 * h as f32 / new_h as f32) as usize;
            let sy = sy.min(h - 1);
            for x in 0..new_w {
                let sx = (x as f32 * w as f32 / new_w as f32) as usize;
                let sx = sx.min(w - 1);
                out[ch * new_h * new_w + y * new_w + x] = data[ch * h * w + sy * w + sx];
            }
        }
    }

    out
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sample(c: usize, h: usize, w: usize, num_boxes: usize) -> DetSample {
        let data: Vec<f32> = (0..c * h * w)
            .map(|i| (i as f32 * 0.001).sin().abs())
            .collect();
        let image = Tensor::from_vec(data, &[c, h, w]).unwrap();

        let boxes: Vec<[f32; 4]> = (0..num_boxes)
            .map(|i| {
                let x = (i * 20) as f32;
                let y = (i * 15) as f32;
                [x, y, x + 30.0, y + 25.0]
            })
            .collect();
        let classes: Vec<usize> = (0..num_boxes).map(|i| i % 3).collect();

        DetSample::new(image, boxes, classes)
    }

    #[test]
    fn test_det_sample() {
        let s = make_sample(3, 64, 64, 2);
        assert_eq!(s.height(), 64);
        assert_eq!(s.width(), 64);
        assert_eq!(s.num_objects(), 2);
    }

    #[test]
    fn test_hflip() {
        let s = make_sample(3, 32, 32, 1);
        let flip = DetRandomHFlip { prob: 1.0 }; // always flip
        let flipped = flip.apply(&s);

        assert_eq!(flipped.image.shape(), &[3, 32, 32]);
        assert_eq!(flipped.num_objects(), 1);
        // Box x coords should be mirrored
        let orig = s.boxes[0];
        let new_box = flipped.boxes[0];
        assert!((new_box[0] - (32.0 - orig[2])).abs() < 0.01);
        assert!((new_box[2] - (32.0 - orig[0])).abs() < 0.01);
    }

    #[test]
    fn test_hsv_roundtrip() {
        let r = 0.8;
        let g = 0.3;
        let b = 0.5;
        let (h, s, v) = rgb_to_hsv(r, g, b);
        let (nr, ng, nb) = hsv_to_rgb(h, s, v);
        assert!((nr - r).abs() < 0.02, "R: {nr} vs {r}");
        assert!((ng - g).abs() < 0.02, "G: {ng} vs {g}");
        assert!((nb - b).abs() < 0.02, "B: {nb} vs {b}");
    }

    #[test]
    fn test_hsv_jitter() {
        let s = make_sample(3, 32, 32, 1);
        let jitter = HSVJitter::new();
        let result = jitter.apply(&s);
        assert_eq!(result.image.shape(), &[3, 32, 32]);
        assert_eq!(result.num_objects(), 1);
        // Boxes unchanged
        assert_eq!(result.boxes[0], s.boxes[0]);
    }

    #[test]
    fn test_letterbox() {
        let s = make_sample(3, 100, 200, 1);
        let lb = LetterBox::new(64, 64);
        let result = lb.apply(&s);
        assert_eq!(result.image.shape(), &[3, 64, 64]);
        // Image was 200x100 → scale=0.32, new=64x32, pad_y=16
        assert!(result.num_objects() == 1);
    }

    #[test]
    fn test_mosaic() {
        let samples: Vec<DetSample> = (0..4).map(|_| make_sample(3, 64, 64, 2)).collect();
        let mosaic = Mosaic::new(128, 128);
        let result = mosaic.apply(&samples);

        assert_eq!(result.image.shape(), &[3, 128, 128]);
        // Should have boxes from all 4 images (some may be clipped away)
        assert!(result.num_objects() > 0);
    }

    #[test]
    fn test_mixup() {
        let a = make_sample(3, 64, 64, 1);
        let b = make_sample(3, 64, 64, 2);
        let mixup = MixUp::new();
        let result = mixup.apply(&a, &b);

        assert_eq!(result.image.shape(), &[3, 64, 64]);
        assert_eq!(result.num_objects(), 3); // 1 + 2
    }

    #[test]
    fn test_random_affine() {
        let s = make_sample(3, 64, 64, 1);
        let affine = DetRandomAffine::with_params(10.0, 0.3, 0.1, 5.0);
        let result = affine.apply(&s);

        assert_eq!(result.image.shape(), &[3, 64, 64]);
        // May lose boxes if they go out of bounds
    }

    #[test]
    fn test_simple_pipeline() {
        let s = make_sample(3, 100, 150, 2);
        let pipeline = DetAugPipeline::simple(64, 64);
        let result = pipeline.apply_single(&s);

        assert_eq!(result.image.shape(), &[3, 64, 64]);
    }

    #[test]
    fn test_yolo_pipeline_no_mosaic() {
        let s = make_sample(3, 100, 100, 1);
        let mut pipeline = DetAugPipeline::yolo(64, 64);
        pipeline.use_mosaic = false;
        pipeline.use_mixup = false;

        let result = pipeline.apply(&s, &[], None);
        assert_eq!(result.image.shape(), &[3, 64, 64]);
    }

    #[test]
    fn test_yolo_pipeline_full() {
        let primary = make_sample(3, 80, 80, 2);
        let others: Vec<DetSample> = (0..3).map(|_| make_sample(3, 80, 80, 1)).collect();
        let mixup_partner = make_sample(3, 80, 80, 1);

        let pipeline = DetAugPipeline::yolo(64, 64);
        let result = pipeline.apply(&primary, &others, Some(&mixup_partner));

        assert_eq!(result.image.shape(), &[3, 64, 64]);
        // Should have some boxes surviving
    }
}
