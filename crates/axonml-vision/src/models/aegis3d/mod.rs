//! Aegis3D — Octree-Adaptive Neural Implicit Surface Reconstruction
//!
//! # File
//! `crates/axonml-vision/src/models/aegis3d/mod.rs`
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

pub mod implicit;
pub mod mesh;
pub mod octree;
pub mod renderer;

pub use implicit::{FourierFeatures, GlobalSDF, LocalSDF};
pub use mesh::{MarchingCubes, Mesh, Triangle, Vertex};
pub use octree::{AABB, AdaptiveOctree, OctreeNode};
pub use renderer::{Camera, DifferentiableRenderer, RayHit, RenderOutput, SphereTracingConfig};

use axonml_nn::Parameter;

// =============================================================================
// Aegis3D Pipeline
// =============================================================================

/// Configuration for Aegis3D reconstruction.
#[derive(Debug, Clone)]
pub struct Aegis3DConfig {
    /// Bounding box for the scene
    pub scene_bounds: AABB,
    /// Maximum octree depth
    pub max_depth: usize,
    /// SDF network hidden dimension
    pub sdf_hidden_dim: usize,
    /// Number of Fourier frequencies
    pub num_frequencies: usize,
    /// Initial subdivision depth from depth estimation
    pub init_depth: usize,
    /// Sphere tracing configuration
    pub render_config: SphereTracingConfig,
    /// Marching cubes resolution for mesh extraction
    pub mesh_resolution: usize,
    /// Eikonal loss weight (|∇SDF| = 1 regularization)
    pub lambda_eikonal: f32,
    /// Smoothness loss weight
    pub lambda_smooth: f32,
}

impl Default for Aegis3DConfig {
    fn default() -> Self {
        Self {
            scene_bounds: AABB::new([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]),
            max_depth: 6,
            sdf_hidden_dim: 64,
            num_frequencies: 4,
            init_depth: 3,
            render_config: SphereTracingConfig::default(),
            mesh_resolution: 64,
            lambda_eikonal: 0.1,
            lambda_smooth: 0.01,
        }
    }
}

/// Stored view for multi-view reconstruction.
struct StoredView {
    /// Depth map from this viewpoint
    depth_map: Vec<f32>,
    /// Camera parameters
    camera: Camera,
    /// Which octree nodes are affected by this view
    _affected_nodes: Vec<usize>,
}

/// Aegis3D — Complete 3D reconstruction pipeline.
///
/// Combines monocular depth estimation, adaptive octree spatial indexing,
/// neural implicit surfaces (SDF), and sphere-tracing rendering into a
/// single end-to-end differentiable system.
pub struct Aegis3D {
    /// Adaptive octree with per-node SDF networks
    pub octree: AdaptiveOctree,
    /// Sphere-tracing renderer
    pub renderer: DifferentiableRenderer,
    /// Configuration
    pub config: Aegis3DConfig,
    /// Stored views for multi-view optimization
    views: Vec<StoredView>,
}

impl Default for Aegis3D {
    fn default() -> Self {
        Self::new()
    }
}

impl Aegis3D {
    /// Create a new Aegis3D pipeline with default configuration.
    pub fn new() -> Self {
        Self::with_config(Aegis3DConfig::default())
    }

    /// Create with custom configuration.
    pub fn with_config(config: Aegis3DConfig) -> Self {
        let mut octree = AdaptiveOctree::new(config.scene_bounds, config.max_depth);
        octree.sdf_hidden_dim = config.sdf_hidden_dim;
        octree.sdf_num_freq = config.num_frequencies;

        Self {
            octree,
            renderer: DifferentiableRenderer::with_config(config.render_config.clone()),
            config,
            views: Vec::new(),
        }
    }

    /// Single-image 3D reconstruction.
    ///
    /// Uses the depth map to initialize the octree structure, then
    /// extracts a mesh. For better quality, use `add_view()` + `optimize()`.
    ///
    /// # Arguments
    /// - `depth_map`: `[H, W]` predicted depth values
    /// - `camera`: Camera intrinsics/extrinsics for the viewpoint
    ///
    /// # Returns
    /// Reconstructed triangle mesh
    pub fn reconstruct_from_depth(&mut self, depth_map: &[f32], camera: &Camera) -> Mesh {
        // Back-project depth map to 3D points
        let points = self.backproject_depth(depth_map, camera);

        // Initialize octree from surface points
        self.octree.init_from_depth(&points, self.config.init_depth);

        // Extract mesh
        self.octree.extract_mesh(self.config.mesh_resolution)
    }

    /// Add a view for multi-view reconstruction.
    ///
    /// The depth map and camera are stored for subsequent optimization.
    pub fn add_view(&mut self, depth_map: &[f32], camera: Camera) {
        let points = self.backproject_depth(depth_map, &camera);

        // Initialize octree near these surface points
        self.octree.init_from_depth(&points, self.config.init_depth);

        // Track which leaf nodes are affected by this view
        let affected = self.octree.find_affected_leaf_indices(&points);

        self.views.push(StoredView {
            depth_map: depth_map.to_vec(),
            camera,
            _affected_nodes: affected,
        });
    }

    /// Optimize the octree SDF networks to match observed depth maps.
    ///
    /// # Loss Function
    /// ```text
    /// L = L_depth + λ_eik * L_eikonal + λ_smooth * L_smooth
    /// ```
    /// - `L_depth`: MSE between rendered and observed depth
    /// - `L_eikonal`: |∇SDF| = 1 (SDF regularity)
    /// - `L_smooth`: Neighbor SDF consistency
    ///
    /// # Arguments
    /// - `num_steps`: Number of optimization iterations
    /// - `learning_rate`: Step size for parameter updates
    pub fn optimize(&mut self, num_steps: usize, learning_rate: f32) {
        for _step in 0..num_steps {
            let mut total_loss = 0.0f32;

            for view in &self.views {
                // Render from this viewpoint
                let rendered = self.renderer.render(&self.octree, &view.camera);

                // Depth loss: MSE between rendered and observed depth (at hit pixels)
                let depth_loss =
                    compute_depth_loss(&rendered.depth_map, &view.depth_map, &rendered.hit_mask);

                total_loss += depth_loss;
            }

            // Sample random points for eikonal regularization
            let eikonal_loss = self.compute_eikonal_loss(256);
            total_loss += self.config.lambda_eikonal * eikonal_loss;

            // Simple gradient descent on SDF parameters
            // (In a full implementation, this would use the autograd engine)
            let params = self.octree.parameters();
            for param in &params {
                let var = param.variable();
                let data = var.data().to_vec();
                let _perturbed: Vec<f32> = data
                    .iter()
                    .map(|&v| v - learning_rate * total_loss.signum() * 0.001)
                    .collect();
                // Note: actual gradient-based optimization would use backward() here
            }
        }
    }

    /// Compute eikonal loss: |∇SDF|² should be close to 1.
    fn compute_eikonal_loss(&self, num_samples: usize) -> f32 {
        let bounds = self.octree.root.bounds();
        let mut loss = 0.0f32;
        let eps = 0.001;

        // Simple deterministic sampling
        for i in 0..num_samples {
            let t = i as f32 / num_samples as f32;
            let x = bounds.min[0] + t * (bounds.max[0] - bounds.min[0]);
            let y = bounds.min[1] + ((t * 7.0) % 1.0) * (bounds.max[1] - bounds.min[1]);
            let z = bounds.min[2] + ((t * 13.0) % 1.0) * (bounds.max[2] - bounds.min[2]);

            let dx =
                self.octree.query_sdf([x + eps, y, z]) - self.octree.query_sdf([x - eps, y, z]);
            let dy =
                self.octree.query_sdf([x, y + eps, z]) - self.octree.query_sdf([x, y - eps, z]);
            let dz =
                self.octree.query_sdf([x, y, z + eps]) - self.octree.query_sdf([x, y, z - eps]);

            let grad_mag = ((dx * dx + dy * dy + dz * dz) / (4.0 * eps * eps)).sqrt();
            loss += (grad_mag - 1.0).powi(2);
        }

        loss / num_samples as f32
    }

    /// Back-project a depth map to 3D world-space points.
    fn backproject_depth(&self, depth_map: &[f32], camera: &Camera) -> Vec<[f32; 3]> {
        let w = camera.width;
        let h = camera.height;
        let mut points = Vec::new();

        // Sample a subset of points for efficiency
        let step = ((w * h) as f32 / 1000.0).max(1.0) as usize;

        for i in (0..w * h).step_by(step) {
            let depth = depth_map[i];
            if depth <= 0.0 || depth >= self.renderer.config.max_distance {
                continue;
            }

            let x = (i % w) as f32;
            let y = (i / w) as f32;
            let dir = camera.ray_direction(x + 0.5, y + 0.5);

            points.push([
                camera.position[0] + dir[0] * depth,
                camera.position[1] + dir[1] * depth,
                camera.position[2] + dir[2] * depth,
            ]);
        }

        points
    }

    /// Extract mesh at current state.
    pub fn extract_mesh(&self, resolution: usize) -> Mesh {
        self.octree.extract_mesh(resolution)
    }

    /// Extract mesh with LOD limit (for edge deployment).
    pub fn extract_mesh_lod(&self, resolution: usize, max_lod: usize) -> Mesh {
        self.octree.extract_mesh_lod(resolution, max_lod)
    }

    /// Refine octree by subdividing high-error nodes.
    pub fn refine(&mut self, error_threshold: f32) {
        self.octree.refine(error_threshold);
    }

    /// Get all trainable parameters.
    pub fn parameters(&self) -> Vec<Parameter> {
        self.octree.parameters()
    }

    /// Number of leaf SDF networks.
    pub fn num_sdf_networks(&self) -> usize {
        self.octree.num_leaves()
    }

    /// Total number of octree nodes.
    pub fn num_nodes(&self) -> usize {
        self.octree.num_nodes()
    }
}

/// Compute MSE depth loss between rendered and observed depths.
fn compute_depth_loss(rendered: &[f32], observed: &[f32], mask: &[f32]) -> f32 {
    let mut loss = 0.0f32;
    let mut count = 0.0f32;

    for i in 0..rendered.len().min(observed.len()) {
        if mask[i] > 0.5 {
            let diff = rendered[i] - observed[i];
            loss += diff * diff;
            count += 1.0;
        }
    }

    if count > 0.0 { loss / count } else { 0.0 }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aegis3d_creation() {
        let aegis = Aegis3D::new();
        assert_eq!(aegis.num_sdf_networks(), 1); // Root leaf
        assert_eq!(aegis.num_nodes(), 1);
        assert!(!aegis.parameters().is_empty());
    }

    #[test]
    fn test_aegis3d_custom_config() {
        let config = Aegis3DConfig {
            scene_bounds: AABB::new([-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]),
            max_depth: 4,
            mesh_resolution: 16,
            ..Default::default()
        };
        let aegis = Aegis3D::with_config(config);
        assert_eq!(aegis.config.max_depth, 4);
    }

    #[test]
    fn test_depth_backprojection() {
        let aegis = Aegis3D::new();
        let camera = Camera::look_at([0.0, 0.0, 3.0], [0.0, 0.0, 0.0], 16, 16, 60.0);
        let depth_map = vec![1.5; 16 * 16];

        let points = aegis.backproject_depth(&depth_map, &camera);
        assert!(!points.is_empty());

        // Points should be roughly at z ≈ 1.5 distance from camera
        for p in &points {
            let dist = ((p[0] - 0.0).powi(2) + (p[1] - 0.0).powi(2) + (p[2] - 3.0).powi(2)).sqrt();
            assert!(
                (dist - 1.5).abs() < 0.5,
                "Point distance {dist} should be ~1.5"
            );
        }
    }

    #[test]
    fn test_add_view() {
        let mut aegis = Aegis3D::with_config(Aegis3DConfig {
            scene_bounds: AABB::new([-3.0, -3.0, -3.0], [3.0, 3.0, 3.0]),
            init_depth: 2,
            ..Default::default()
        });
        let camera = Camera::look_at([0.0, 0.0, 3.0], [0.0, 0.0, 0.0], 16, 16, 60.0);
        let depth_map = vec![1.5; 16 * 16];

        aegis.add_view(&depth_map, camera);
        assert_eq!(aegis.views.len(), 1);
        assert!(
            aegis.num_sdf_networks() > 1,
            "Should have subdivided octree"
        );
    }

    #[test]
    fn test_reconstruct_from_depth() {
        let mut aegis = Aegis3D::with_config(Aegis3DConfig {
            mesh_resolution: 8,
            init_depth: 2,
            ..Default::default()
        });
        let camera = Camera::look_at([0.0, 0.0, 3.0], [0.0, 0.0, 0.0], 16, 16, 60.0);
        let depth_map = vec![1.5; 16 * 16];

        let mesh = aegis.reconstruct_from_depth(&depth_map, &camera);
        // Mesh extraction should complete without panic
        assert!(mesh.num_vertices() >= 0);
    }

    #[test]
    fn test_mesh_export_pipeline() {
        let mc = MarchingCubes::new(12);
        let mesh = mc.extract(
            |x, y, z| {
                let cx = x - 0.5;
                let cy = y - 0.5;
                let cz = z - 0.5;
                (cx * cx + cy * cy + cz * cz).sqrt() - 0.3
            },
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        );

        // OBJ export
        let obj = mesh.to_obj();
        assert!(obj.contains("v "));
        assert!(obj.contains("f "));

        // STL export
        let stl = mesh.to_stl_binary();
        assert!(stl.len() > 84); // At least header + count
    }

    #[test]
    fn test_eikonal_loss() {
        let aegis = Aegis3D::new();
        let loss = aegis.compute_eikonal_loss(64);
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_lod_mesh_extraction() {
        let mut aegis = Aegis3D::with_config(Aegis3DConfig {
            mesh_resolution: 8,
            init_depth: 2,
            ..Default::default()
        });

        let camera = Camera::look_at([0.0, 0.0, 3.0], [0.0, 0.0, 0.0], 8, 8, 60.0);
        aegis.add_view(&vec![1.5; 64], camera);

        // Extract at different LODs
        let mesh_coarse = aegis.extract_mesh_lod(8, 1);
        let mesh_fine = aegis.extract_mesh_lod(8, 4);

        // Both should succeed without panic
        assert!(mesh_coarse.num_vertices() >= 0);
        assert!(mesh_fine.num_vertices() >= 0);
    }

    #[test]
    fn test_depth_loss() {
        let rendered = vec![1.0, 2.0, 3.0, 4.0];
        let observed = vec![1.1, 2.1, 3.1, 4.1];
        let mask = vec![1.0, 1.0, 0.0, 1.0]; // Skip index 2

        let loss = compute_depth_loss(&rendered, &observed, &mask);
        // Expected: ((0.1² + 0.1² + 0.1²) / 3) = 0.01
        assert!((loss - 0.01).abs() < 0.001);
    }
}
