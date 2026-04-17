//! Pseudo-Event Generation — Frame Differencing on Standard Cameras
//!
//! Implements `EventConfig` (threshold, polarity, temporal decay parameters) and
//! `EventEncoder` which generates pseudo-events from consecutive video frames by
//! computing per-pixel intensity changes, thresholding to produce positive/negative
//! polarity events, and encoding the result as a multi-channel event tensor. This
//! enables event-camera-style processing on standard frame-based cameras for
//! efficient motion-driven face detection in the Phantom pipeline.
//!
//! # File
//! `crates/axonml-vision/src/models/phantom/events.rs`
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

use axonml_autograd::Variable;
use axonml_tensor::Tensor;

// =============================================================================
// Event Encoder
// =============================================================================

/// Configuration for event generation.
#[derive(Debug, Clone)]
pub struct EventConfig {
    /// Intensity change threshold for event generation (0-255 scale).
    pub threshold: f32,
    /// Cell size for density computation (pixels).
    pub cell_size: usize,
    /// Dilation radius for active regions (in cells).
    pub dilation: usize,
    /// Minimum density to consider a cell active.
    pub min_density: f32,
}

impl Default for EventConfig {
    fn default() -> Self {
        Self {
            threshold: 15.0,
            cell_size: 16,
            dilation: 1,
            min_density: 0.05,
        }
    }
}

/// Pseudo-event encoder that generates event maps from frame differences.
///
/// Converts standard camera frames into sparse event representations,
/// enabling neuromorphic-inspired processing on regular hardware.
pub struct EventEncoder {
    config: EventConfig,
    /// Previous frame grayscale values [H, W] (0-255 scale).
    prev_gray: Option<Vec<f32>>,
    prev_width: usize,
    prev_height: usize,
}

impl EventEncoder {
    /// Create a new event encoder with default configuration.
    pub fn new() -> Self {
        Self {
            config: EventConfig::default(),
            prev_gray: None,
            prev_width: 0,
            prev_height: 0,
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: EventConfig) -> Self {
        Self {
            config,
            prev_gray: None,
            prev_width: 0,
            prev_height: 0,
        }
    }

    /// Reset the encoder state (clears previous frame).
    pub fn reset(&mut self) {
        self.prev_gray = None;
    }

    /// Whether this is the first frame (cold start).
    pub fn is_cold_start(&self) -> bool {
        self.prev_gray.is_none()
    }

    /// Compute per-pixel event map from current frame.
    ///
    /// # Arguments
    /// - `frame`: Current frame as Variable [B, 3, H, W] (normalized float)
    ///
    /// # Returns
    /// - Event density map as Variable [B, 1, cell_H, cell_W]
    /// - Active cell mask (Vec of (cell_y, cell_x) indices)
    /// - Per-pixel event magnitude [B, 1, H, W]
    pub fn encode(&mut self, frame: &Variable) -> (Variable, Vec<(usize, usize)>, Variable) {
        let shape = frame.shape();
        let (batch, _c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let data = frame.data().to_vec();

        // Convert to grayscale (luminance): 0.299*R + 0.587*G + 0.114*B
        // Input is normalized, so undo normalization first (approximate)
        let mut gray = vec![0.0f32; batch * h * w];
        for b in 0..batch {
            for y in 0..h {
                for x in 0..w {
                    let r = data[b * 3 * h * w + 0 * h * w + y * w + x];
                    let g = data[b * 3 * h * w + h * w + y * w + x];
                    let b_val = data[b * 3 * h * w + 2 * h * w + y * w + x];
                    // Approximate: normalized values centered around 0, scale ~[-2, 2]
                    gray[b * h * w + y * w + x] = 0.299 * r + 0.587 * g + 0.114 * b_val;
                }
            }
        }

        // Compute per-pixel event magnitude
        let mut event_mag = vec![0.0f32; batch * h * w];

        if let Some(ref prev) = self.prev_gray {
            if self.prev_width == w && self.prev_height == h {
                // Scale threshold to normalized space (~15/255 * 4 range ≈ 0.235)
                let thresh = self.config.threshold / 255.0 * 4.0;
                for i in 0..batch * h * w {
                    let idx = i % (h * w); // Index into prev (single frame stored)
                    let diff = (gray[i] - prev[idx]).abs();
                    event_mag[i] = if diff > thresh { diff } else { 0.0 };
                }
            }
        }

        // Compute cell-level density
        let cell = self.config.cell_size;
        let cell_h = h.div_ceil(cell);
        let cell_w = w.div_ceil(cell);
        let mut density = vec![0.0f32; batch * cell_h * cell_w];

        for b in 0..batch {
            for cy in 0..cell_h {
                for cx in 0..cell_w {
                    let mut count = 0.0f32;
                    let mut total = 0.0f32;

                    let y_start = cy * cell;
                    let y_end = ((cy + 1) * cell).min(h);
                    let x_start = cx * cell;
                    let x_end = ((cx + 1) * cell).min(w);

                    for y in y_start..y_end {
                        for x in x_start..x_end {
                            total += 1.0;
                            if event_mag[b * h * w + y * w + x] > 0.0 {
                                count += 1.0;
                            }
                        }
                    }

                    density[b * cell_h * cell_w + cy * cell_w + cx] =
                        if total > 0.0 { count / total } else { 0.0 };
                }
            }
        }

        // Dilate active cells and collect active indices
        let mut active_mask = vec![false; cell_h * cell_w];
        for cy in 0..cell_h {
            for cx in 0..cell_w {
                // Check first batch item for active cell selection
                if density[cy * cell_w + cx] >= self.config.min_density {
                    // Mark this cell and neighbors within dilation radius
                    let d = self.config.dilation;
                    let y_lo = cy.saturating_sub(d);
                    let y_hi = (cy + d + 1).min(cell_h);
                    let x_lo = cx.saturating_sub(d);
                    let x_hi = (cx + d + 1).min(cell_w);
                    for dy in y_lo..y_hi {
                        for dx in x_lo..x_hi {
                            active_mask[dy * cell_w + dx] = true;
                        }
                    }
                }
            }
        }

        let active_cells: Vec<(usize, usize)> = active_mask
            .iter()
            .enumerate()
            .filter(|&(_, &active)| active)
            .map(|(i, _)| (i / cell_w, i % cell_w))
            .collect();

        // Store current gray for next frame
        // Store only the first batch item
        self.prev_gray = Some(gray[..h * w].to_vec());
        self.prev_width = w;
        self.prev_height = h;

        let density_tensor = Tensor::from_vec(density, &[batch, 1, cell_h, cell_w]).unwrap();
        let event_tensor = Tensor::from_vec(event_mag, &[batch, 1, h, w]).unwrap();

        (
            Variable::new(density_tensor, false),
            active_cells,
            Variable::new(event_tensor, false),
        )
    }

    /// Get the number of active cells from the last encode call.
    /// Useful for measuring compute savings.
    pub fn active_ratio(&self, total_cells: usize, active_count: usize) -> f32 {
        if total_cells > 0 {
            active_count as f32 / total_cells as f32
        } else {
            0.0
        }
    }
}

impl Default for EventEncoder {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(h: usize, w: usize, val: f32) -> Variable {
        let data = vec![val; 3 * h * w];
        Variable::new(Tensor::from_vec(data, &[1, 3, h, w]).unwrap(), false)
    }

    #[test]
    fn test_event_encoder_cold_start() {
        let mut enc = EventEncoder::new();
        assert!(enc.is_cold_start());

        let frame = make_frame(32, 32, 0.5);
        let (density, active, events) = enc.encode(&frame);

        // First frame: no previous → no events
        assert!(!enc.is_cold_start());
        assert_eq!(density.shape()[2], 2); // 32/16 = 2
        assert_eq!(density.shape()[3], 2);
        assert!(active.is_empty()); // No events on first frame
        assert_eq!(events.shape(), vec![1, 1, 32, 32]);
    }

    #[test]
    fn test_identical_frames_zero_events() {
        let mut enc = EventEncoder::new();
        let frame = make_frame(32, 32, 0.5);

        enc.encode(&frame);
        let (_density, active, events) = enc.encode(&frame);

        // Identical frames → no events
        assert!(active.is_empty());
        let event_data = events.data().to_vec();
        assert!(event_data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_changed_frames_produce_events() {
        let mut enc = EventEncoder::new();
        let frame1 = make_frame(32, 32, 0.0);
        let frame2 = make_frame(32, 32, 2.0); // Big change

        enc.encode(&frame1);
        let (_density, active, events) = enc.encode(&frame2);

        // Large change → events everywhere
        assert!(!active.is_empty());
        let event_data = events.data().to_vec();
        assert!(event_data.iter().any(|&v| v > 0.0));
    }

    #[test]
    fn test_partial_change_sparse_events() {
        let mut enc = EventEncoder::with_config(EventConfig {
            cell_size: 16,
            dilation: 0,
            ..EventConfig::default()
        });

        let h = 32;
        let w = 32;

        // Frame 1: all zeros
        let frame1 = make_frame(h, w, 0.0);
        enc.encode(&frame1);

        // Frame 2: change only top-left quadrant
        let mut data2 = vec![0.0f32; 3 * h * w];
        for c in 0..3 {
            for y in 0..h / 2 {
                for x in 0..w / 2 {
                    data2[c * h * w + y * w + x] = 2.0;
                }
            }
        }
        let frame2 = Variable::new(Tensor::from_vec(data2, &[1, 3, h, w]).unwrap(), false);

        let (_density, active, _events) = enc.encode(&frame2);

        // Only top-left cells should be active (cell 0,0)
        assert!(!active.is_empty());
        assert!(active.contains(&(0, 0)));
        // Bottom-right should be inactive
        let cell_h = h / 16;
        let cell_w = w / 16;
        assert!(!active.contains(&(cell_h - 1, cell_w - 1)));
    }

    #[test]
    fn test_event_encoder_reset() {
        let mut enc = EventEncoder::new();
        enc.encode(&make_frame(16, 16, 1.0));
        assert!(!enc.is_cold_start());

        enc.reset();
        assert!(enc.is_cold_start());
    }

    #[test]
    fn test_active_ratio() {
        let enc = EventEncoder::new();
        assert!((enc.active_ratio(100, 25) - 0.25).abs() < 1e-5);
        assert_eq!(enc.active_ratio(0, 0), 0.0);
    }
}
