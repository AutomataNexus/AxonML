//! Differentiable Renderer
//!
//! # File
//! `crates/axonml-vision/src/models/aegis3d/renderer.rs`
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

#![allow(missing_docs)]

use super::octree::AdaptiveOctree;

// =============================================================================
// Camera Model
// =============================================================================

/// Pinhole camera model.
#[derive(Debug, Clone)]
pub struct Camera {
    /// Camera position in world space
    pub position: [f32; 3],
    /// Camera look-at direction (normalized)
    pub forward: [f32; 3],
    /// Camera up direction (normalized)
    pub up: [f32; 3],
    /// Camera right direction (normalized)
    pub right: [f32; 3],
    /// Focal length in pixels
    pub focal_length: f32,
    /// Image width
    pub width: usize,
    /// Image height
    pub height: usize,
}

impl Camera {
    /// Create a camera looking from `position` at `target`.
    pub fn look_at(
        position: [f32; 3],
        target: [f32; 3],
        width: usize,
        height: usize,
        fov_degrees: f32,
    ) -> Self {
        let forward = normalize(sub(target, position));
        let world_up = [0.0, 1.0, 0.0];
        let right = normalize(cross(forward, world_up));
        let up = normalize(cross(right, forward));
        let focal_length = (width as f32 / 2.0) / (fov_degrees.to_radians() / 2.0).tan();

        Self {
            position,
            forward,
            up,
            right,
            focal_length,
            width,
            height,
        }
    }

    /// Generate a ray direction for pixel (u, v).
    pub fn ray_direction(&self, u: f32, v: f32) -> [f32; 3] {
        let cx = self.width as f32 / 2.0;
        let cy = self.height as f32 / 2.0;

        let dx = u - cx;
        let dy = cy - v; // Flip y (image coords vs world coords)

        normalize([
            self.forward[0] * self.focal_length + self.right[0] * dx + self.up[0] * dy,
            self.forward[1] * self.focal_length + self.right[1] * dx + self.up[1] * dy,
            self.forward[2] * self.focal_length + self.right[2] * dx + self.up[2] * dy,
        ])
    }
}

// =============================================================================
// Sphere Tracing
// =============================================================================

/// Result of a single ray march.
#[derive(Debug, Clone, Copy)]
pub struct RayHit {
    /// Hit position in world space
    pub position: [f32; 3],
    /// Distance from camera
    pub depth: f32,
    /// Surface normal at hit point (estimated from SDF gradient)
    pub normal: [f32; 3],
    /// Whether the ray hit a surface
    pub hit: bool,
}

/// Configuration for sphere tracing.
#[derive(Debug, Clone)]
pub struct SphereTracingConfig {
    /// Maximum number of marching steps
    pub max_steps: usize,
    /// Distance threshold for surface hit
    pub hit_threshold: f32,
    /// Maximum ray distance before giving up
    pub max_distance: f32,
    /// Epsilon for finite-difference normal estimation
    pub normal_epsilon: f32,
}

impl Default for SphereTracingConfig {
    fn default() -> Self {
        Self {
            max_steps: 128,
            hit_threshold: 0.001,
            max_distance: 10.0,
            normal_epsilon: 0.001,
        }
    }
}

/// Differentiable renderer using sphere tracing through the octree SDF.
pub struct DifferentiableRenderer {
    pub config: SphereTracingConfig,
}

impl Default for DifferentiableRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl DifferentiableRenderer {
    pub fn new() -> Self {
        Self {
            config: SphereTracingConfig::default(),
        }
    }

    pub fn with_config(config: SphereTracingConfig) -> Self {
        Self { config }
    }

    /// March a single ray through the SDF field.
    pub fn trace_ray(
        &self,
        octree: &AdaptiveOctree,
        origin: [f32; 3],
        direction: [f32; 3],
    ) -> RayHit {
        let mut t = 0.0f32;

        for _ in 0..self.config.max_steps {
            let pos = [
                origin[0] + direction[0] * t,
                origin[1] + direction[1] * t,
                origin[2] + direction[2] * t,
            ];

            let sdf = octree.query_sdf(pos);

            if sdf.abs() < self.config.hit_threshold {
                // Hit! Estimate normal via finite differences
                let eps = self.config.normal_epsilon;
                let nx = octree.query_sdf([pos[0] + eps, pos[1], pos[2]])
                    - octree.query_sdf([pos[0] - eps, pos[1], pos[2]]);
                let ny = octree.query_sdf([pos[0], pos[1] + eps, pos[2]])
                    - octree.query_sdf([pos[0], pos[1] - eps, pos[2]]);
                let nz = octree.query_sdf([pos[0], pos[1], pos[2] + eps])
                    - octree.query_sdf([pos[0], pos[1], pos[2] - eps]);

                return RayHit {
                    position: pos,
                    depth: t,
                    normal: normalize([nx, ny, nz]),
                    hit: true,
                };
            }

            // Advance by SDF distance (sphere tracing guarantee)
            t += sdf.abs().max(self.config.hit_threshold * 0.5);

            if t > self.config.max_distance {
                break;
            }
        }

        RayHit {
            position: [0.0; 3],
            depth: self.config.max_distance,
            normal: [0.0; 3],
            hit: false,
        }
    }

    /// Render a depth map from the given camera viewpoint.
    ///
    /// # Returns
    /// `(depth_map, normal_map, hit_mask)` as flat Vec<f32> arrays.
    /// - `depth_map`: `[H, W]` depth values (max_distance for misses)
    /// - `normal_map`: `[H, W, 3]` surface normals
    /// - `hit_mask`: `[H, W]` binary mask (1.0 = hit, 0.0 = miss)
    pub fn render(&self, octree: &AdaptiveOctree, camera: &Camera) -> RenderOutput {
        let w = camera.width;
        let h = camera.height;

        let mut depth_map = vec![self.config.max_distance; h * w];
        let mut normal_map = vec![0.0f32; h * w * 3];
        let mut hit_mask = vec![0.0f32; h * w];

        for y in 0..h {
            for x in 0..w {
                let dir = camera.ray_direction(x as f32 + 0.5, y as f32 + 0.5);
                let hit = self.trace_ray(octree, camera.position, dir);

                let idx = y * w + x;
                depth_map[idx] = hit.depth;

                if hit.hit {
                    hit_mask[idx] = 1.0;
                    normal_map[idx * 3] = hit.normal[0];
                    normal_map[idx * 3 + 1] = hit.normal[1];
                    normal_map[idx * 3 + 2] = hit.normal[2];
                }
            }
        }

        RenderOutput {
            depth_map,
            normal_map,
            hit_mask,
            width: w,
            height: h,
        }
    }

    /// Render with LOD limit for edge deployment.
    pub fn render_lod(
        &self,
        octree: &AdaptiveOctree,
        camera: &Camera,
        max_lod: usize,
    ) -> RenderOutput {
        let w = camera.width;
        let h = camera.height;

        let mut depth_map = vec![self.config.max_distance; h * w];
        let mut normal_map = vec![0.0f32; h * w * 3];
        let mut hit_mask = vec![0.0f32; h * w];

        for y in 0..h {
            for x in 0..w {
                let dir = camera.ray_direction(x as f32 + 0.5, y as f32 + 0.5);
                let hit = self.trace_ray_lod(octree, camera.position, dir, max_lod);

                let idx = y * w + x;
                depth_map[idx] = hit.depth;

                if hit.hit {
                    hit_mask[idx] = 1.0;
                    normal_map[idx * 3] = hit.normal[0];
                    normal_map[idx * 3 + 1] = hit.normal[1];
                    normal_map[idx * 3 + 2] = hit.normal[2];
                }
            }
        }

        RenderOutput {
            depth_map,
            normal_map,
            hit_mask,
            width: w,
            height: h,
        }
    }

    fn trace_ray_lod(
        &self,
        octree: &AdaptiveOctree,
        origin: [f32; 3],
        direction: [f32; 3],
        max_lod: usize,
    ) -> RayHit {
        let mut t = 0.0f32;

        for _ in 0..self.config.max_steps {
            let pos = [
                origin[0] + direction[0] * t,
                origin[1] + direction[1] * t,
                origin[2] + direction[2] * t,
            ];

            let sdf = octree.query_sdf_lod(pos, max_lod);

            if sdf.abs() < self.config.hit_threshold {
                let eps = self.config.normal_epsilon;
                let nx = octree.query_sdf_lod([pos[0] + eps, pos[1], pos[2]], max_lod)
                    - octree.query_sdf_lod([pos[0] - eps, pos[1], pos[2]], max_lod);
                let ny = octree.query_sdf_lod([pos[0], pos[1] + eps, pos[2]], max_lod)
                    - octree.query_sdf_lod([pos[0], pos[1] - eps, pos[2]], max_lod);
                let nz = octree.query_sdf_lod([pos[0], pos[1], pos[2] + eps], max_lod)
                    - octree.query_sdf_lod([pos[0], pos[1], pos[2] - eps], max_lod);

                return RayHit {
                    position: pos,
                    depth: t,
                    normal: normalize([nx, ny, nz]),
                    hit: true,
                };
            }

            t += sdf.abs().max(self.config.hit_threshold * 0.5);

            if t > self.config.max_distance {
                break;
            }
        }

        RayHit {
            position: [0.0; 3],
            depth: self.config.max_distance,
            normal: [0.0; 3],
            hit: false,
        }
    }
}

/// Output from rendering.
#[derive(Debug, Clone)]
pub struct RenderOutput {
    /// Depth map `[H, W]`
    pub depth_map: Vec<f32>,
    /// Normal map `[H, W, 3]`
    pub normal_map: Vec<f32>,
    /// Hit mask `[H, W]` (1.0 = surface hit)
    pub hit_mask: Vec<f32>,
    pub width: usize,
    pub height: usize,
}

impl RenderOutput {
    /// Fraction of rays that hit a surface.
    pub fn hit_fraction(&self) -> f32 {
        let total = self.hit_mask.len() as f32;
        let hits: f32 = self.hit_mask.iter().sum();
        hits / total
    }
}

// =============================================================================
// Vector helpers
// =============================================================================

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-10 {
        return [0.0; 3];
    }
    [v[0] / len, v[1] / len, v[2] / len]
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::super::octree::AABB;
    use super::*;

    #[test]
    fn test_camera_creation() {
        let cam = Camera::look_at([0.0, 0.0, 3.0], [0.0, 0.0, 0.0], 64, 64, 60.0);
        assert_eq!(cam.width, 64);
        assert_eq!(cam.height, 64);

        // Forward should point toward -z
        assert!(cam.forward[2] < 0.0);
    }

    #[test]
    fn test_camera_ray_direction() {
        let cam = Camera::look_at([0.0, 0.0, 3.0], [0.0, 0.0, 0.0], 64, 64, 60.0);

        // Center pixel should point roughly forward
        let dir = cam.ray_direction(32.0, 32.0);
        let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
        assert!((len - 1.0).abs() < 0.01, "Direction should be normalized");
    }

    #[test]
    fn test_sphere_tracing() {
        let bounds = AABB::new([-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]);
        let octree = AdaptiveOctree::new(bounds, 2);
        let renderer = DifferentiableRenderer::new();

        // Trace a ray (may or may not hit depending on random SDF weights)
        let hit = renderer.trace_ray(&octree, [0.0, 0.0, 5.0], [0.0, 0.0, -1.0]);
        assert!(hit.depth > 0.0);
    }

    #[test]
    fn test_render_output() {
        let bounds = AABB::new([-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]);
        let octree = AdaptiveOctree::new(bounds, 2);
        let renderer = DifferentiableRenderer::with_config(SphereTracingConfig {
            max_steps: 16, // Fewer steps for fast test
            hit_threshold: 0.01,
            max_distance: 5.0,
            normal_epsilon: 0.01,
        });
        let cam = Camera::look_at([0.0, 0.0, 3.0], [0.0, 0.0, 0.0], 8, 8, 60.0);

        let output = renderer.render(&octree, &cam);
        assert_eq!(output.depth_map.len(), 64); // 8x8
        assert_eq!(output.normal_map.len(), 192); // 8x8x3
        assert_eq!(output.hit_mask.len(), 64);
    }
}
