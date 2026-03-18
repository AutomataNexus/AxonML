//! Adaptive Octree for Spatial Acceleration
//!
//! # File
//! `crates/axonml-vision/src/models/aegis3d/octree.rs`
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

use super::implicit::LocalSDF;
use super::mesh::{MarchingCubes, Mesh};

use axonml_nn::{Module, Parameter};

// =============================================================================
// Octree Node
// =============================================================================

/// Axis-aligned bounding box for an octree node.
#[derive(Debug, Clone, Copy)]
pub struct AABB {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

impl AABB {
    pub fn new(min: [f32; 3], max: [f32; 3]) -> Self {
        Self { min, max }
    }

    /// Center of the bounding box.
    pub fn center(&self) -> [f32; 3] {
        [
            (self.min[0] + self.max[0]) * 0.5,
            (self.min[1] + self.max[1]) * 0.5,
            (self.min[2] + self.max[2]) * 0.5,
        ]
    }

    /// Half-extent (size from center to edge).
    pub fn extent(&self) -> f32 {
        (self.max[0] - self.min[0]) * 0.5
    }

    /// Check if a point is inside the AABB.
    pub fn contains(&self, point: [f32; 3]) -> bool {
        point[0] >= self.min[0]
            && point[0] <= self.max[0]
            && point[1] >= self.min[1]
            && point[1] <= self.max[1]
            && point[2] >= self.min[2]
            && point[2] <= self.max[2]
    }

    /// Compute sub-boxes for 8 children.
    pub fn subdivide(&self) -> [AABB; 8] {
        let c = self.center();
        [
            AABB::new(self.min, c),
            AABB::new([c[0], self.min[1], self.min[2]], [self.max[0], c[1], c[2]]),
            AABB::new([self.min[0], c[1], self.min[2]], [c[0], self.max[1], c[2]]),
            AABB::new([c[0], c[1], self.min[2]], [self.max[0], self.max[1], c[2]]),
            AABB::new([self.min[0], self.min[1], c[2]], [c[0], c[1], self.max[2]]),
            AABB::new([c[0], self.min[1], c[2]], [self.max[0], c[1], self.max[2]]),
            AABB::new([self.min[0], c[1], c[2]], [c[0], self.max[1], self.max[2]]),
            AABB::new(c, self.max),
        ]
    }

    /// Which octant index (0-7) a point falls into.
    pub fn octant_for(&self, point: [f32; 3]) -> usize {
        let c = self.center();
        let mut idx = 0;
        if point[0] >= c[0] {
            idx |= 1;
        }
        if point[1] >= c[1] {
            idx |= 2;
        }
        if point[2] >= c[2] {
            idx |= 4;
        }
        idx
    }
}

/// Octree node — either a leaf with an SDF network, or an internal node.
pub enum OctreeNode {
    /// Leaf node with a local SDF network.
    Leaf {
        bounds: AABB,
        sdf: LocalSDF,
        /// Depth level in the tree
        depth: usize,
        /// Accumulated error for refinement decisions
        error: f32,
    },
    /// Internal node with 8 children.
    Internal {
        bounds: AABB,
        children: Box<[OctreeNode; 8]>,
        depth: usize,
    },
    /// Empty node — no geometry expected here.
    Empty { bounds: AABB, depth: usize },
}

impl OctreeNode {
    pub fn bounds(&self) -> &AABB {
        match self {
            OctreeNode::Leaf { bounds, .. } => bounds,
            OctreeNode::Internal { bounds, .. } => bounds,
            OctreeNode::Empty { bounds, .. } => bounds,
        }
    }

    pub fn depth(&self) -> usize {
        match self {
            OctreeNode::Leaf { depth, .. } => *depth,
            OctreeNode::Internal { depth, .. } => *depth,
            OctreeNode::Empty { depth, .. } => *depth,
        }
    }
}

// =============================================================================
// Adaptive Octree
// =============================================================================

/// Adaptive octree with per-node SDF networks.
///
/// The octree adaptively subdivides near surfaces, allocating neural network
/// capacity where geometry is complex. Empty regions are pruned as `Empty`
/// nodes with zero parameters.
pub struct AdaptiveOctree {
    /// Root node
    pub root: OctreeNode,
    /// Maximum allowed depth
    pub max_depth: usize,
    /// SDF hidden dimension for local networks
    pub sdf_hidden_dim: usize,
    /// Number of Fourier frequencies for positional encoding
    pub sdf_num_freq: usize,
}

impl AdaptiveOctree {
    /// Create a new octree covering the given bounding box.
    pub fn new(bounds: AABB, max_depth: usize) -> Self {
        let center = bounds.center();
        let extent = bounds.extent();

        Self {
            root: OctreeNode::Leaf {
                bounds,
                sdf: LocalSDF::default_at(center, extent),
                depth: 0,
                error: f32::MAX,
            },
            max_depth,
            sdf_hidden_dim: 64,
            sdf_num_freq: 4,
        }
    }

    /// Initialize octree using depth map predictions.
    ///
    /// Creates initial subdivisions near predicted surface locations,
    /// dramatically reducing convergence time compared to uniform initialization.
    ///
    /// # Arguments
    /// - `depth_points`: `[N, 3]` predicted 3D surface points from monocular depth
    /// - `initial_depth`: How deep to subdivide near surface points
    pub fn init_from_depth(&mut self, depth_points: &[[f32; 3]], initial_depth: usize) {
        let target_depth = initial_depth.min(self.max_depth);

        // Subdivide nodes that contain surface points
        for point in depth_points {
            self.subdivide_at_point(*point, target_depth);
        }
    }

    /// Subdivide the tree at a specific point down to target depth.
    fn subdivide_at_point(&mut self, point: [f32; 3], target_depth: usize) {
        Self::subdivide_node(
            &mut self.root,
            point,
            target_depth,
            self.sdf_hidden_dim,
            self.sdf_num_freq,
        );
    }

    fn subdivide_node(
        node: &mut OctreeNode,
        point: [f32; 3],
        target_depth: usize,
        hidden_dim: usize,
        num_freq: usize,
    ) {
        match node {
            OctreeNode::Leaf { bounds, depth, .. } | OctreeNode::Empty { bounds, depth, .. } => {
                if !bounds.contains(point) || *depth >= target_depth {
                    return;
                }

                let sub_bounds = bounds.subdivide();
                let current_depth = *depth;
                let parent_bounds = *bounds;

                let children: [OctreeNode; 8] = std::array::from_fn(|i| {
                    let center = sub_bounds[i].center();
                    let extent = sub_bounds[i].extent();
                    if sub_bounds[i].contains(point) {
                        OctreeNode::Leaf {
                            bounds: sub_bounds[i],
                            sdf: LocalSDF::new(hidden_dim, num_freq, center, extent),
                            depth: current_depth + 1,
                            error: f32::MAX,
                        }
                    } else {
                        OctreeNode::Empty {
                            bounds: sub_bounds[i],
                            depth: current_depth + 1,
                        }
                    }
                });

                *node = OctreeNode::Internal {
                    bounds: parent_bounds,
                    children: Box::new(children),
                    depth: current_depth,
                };

                // Recurse into the child containing the point
                if let OctreeNode::Internal { children, .. } = node {
                    let octant = parent_bounds.octant_for(point);
                    Self::subdivide_node(
                        &mut children[octant],
                        point,
                        target_depth,
                        hidden_dim,
                        num_freq,
                    );
                }
            }
            OctreeNode::Internal {
                bounds, children, ..
            } => {
                if !bounds.contains(point) {
                    return;
                }
                let octant = bounds.octant_for(point);
                Self::subdivide_node(
                    &mut children[octant],
                    point,
                    target_depth,
                    hidden_dim,
                    num_freq,
                );
            }
        }
    }

    /// Query SDF value at a point.
    ///
    /// Traverses the octree to find the leaf containing the point,
    /// then evaluates that leaf's SDF network.
    pub fn query_sdf(&self, point: [f32; 3]) -> f32 {
        Self::query_node(&self.root, point)
    }

    fn query_node(node: &OctreeNode, point: [f32; 3]) -> f32 {
        match node {
            OctreeNode::Leaf { bounds, sdf, .. } => {
                if bounds.contains(point) {
                    sdf.evaluate_single(point[0], point[1], point[2])
                } else {
                    1.0 // Outside = large positive distance
                }
            }
            OctreeNode::Internal {
                bounds, children, ..
            } => {
                if !bounds.contains(point) {
                    return 1.0;
                }
                let octant = bounds.octant_for(point);
                Self::query_node(&children[octant], point)
            }
            OctreeNode::Empty { .. } => 1.0, // Empty = far from surface
        }
    }

    /// Query SDF with level-of-detail limit.
    ///
    /// Stops traversal at `max_lod` depth. Edge devices use low LOD (0-4),
    /// servers use full LOD (0-8). Same model, different resolution.
    pub fn query_sdf_lod(&self, point: [f32; 3], max_lod: usize) -> f32 {
        Self::query_node_lod(&self.root, point, max_lod)
    }

    fn query_node_lod(node: &OctreeNode, point: [f32; 3], max_lod: usize) -> f32 {
        match node {
            OctreeNode::Leaf { bounds, sdf, .. } => {
                if bounds.contains(point) {
                    sdf.evaluate_single(point[0], point[1], point[2])
                } else {
                    1.0
                }
            }
            OctreeNode::Internal {
                bounds,
                children,
                depth,
                ..
            } => {
                if !bounds.contains(point) {
                    return 1.0;
                }
                if *depth >= max_lod {
                    // At LOD limit — evaluate the first non-empty child at the query point
                    let octant = bounds.octant_for(point);
                    match &children[octant] {
                        OctreeNode::Leaf { sdf, .. } => {
                            sdf.evaluate_single(point[0], point[1], point[2])
                        }
                        _ => 1.0,
                    }
                } else {
                    let octant = bounds.octant_for(point);
                    Self::query_node_lod(&children[octant], point, max_lod)
                }
            }
            OctreeNode::Empty { .. } => 1.0,
        }
    }

    /// Collect all leaf SDF parameters for optimization.
    pub fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        Self::collect_params(&self.root, &mut params);
        params
    }

    fn collect_params(node: &OctreeNode, params: &mut Vec<Parameter>) {
        match node {
            OctreeNode::Leaf { sdf, .. } => {
                params.extend(sdf.parameters());
            }
            OctreeNode::Internal { children, .. } => {
                for child in children.as_ref() {
                    Self::collect_params(child, params);
                }
            }
            OctreeNode::Empty { .. } => {}
        }
    }

    /// Count total leaf nodes.
    pub fn num_leaves(&self) -> usize {
        Self::count_leaves(&self.root)
    }

    fn count_leaves(node: &OctreeNode) -> usize {
        match node {
            OctreeNode::Leaf { .. } => 1,
            OctreeNode::Internal { children, .. } => children.iter().map(Self::count_leaves).sum(),
            OctreeNode::Empty { .. } => 0,
        }
    }

    /// Count total nodes (leaves + internal + empty).
    pub fn num_nodes(&self) -> usize {
        Self::count_nodes(&self.root)
    }

    fn count_nodes(node: &OctreeNode) -> usize {
        match node {
            OctreeNode::Leaf { .. } | OctreeNode::Empty { .. } => 1,
            OctreeNode::Internal { children, .. } => {
                1 + children.iter().map(Self::count_nodes).sum::<usize>()
            }
        }
    }

    /// Extract a mesh from the octree SDF using marching cubes.
    pub fn extract_mesh(&self, resolution: usize) -> Mesh {
        let bounds = self.root.bounds();
        let mc = MarchingCubes::new(resolution);
        mc.extract(|x, y, z| self.query_sdf([x, y, z]), bounds.min, bounds.max)
    }

    /// Extract mesh with LOD limit.
    pub fn extract_mesh_lod(&self, resolution: usize, max_lod: usize) -> Mesh {
        let bounds = self.root.bounds();
        let mc = MarchingCubes::new(resolution);
        mc.extract(
            |x, y, z| self.query_sdf_lod([x, y, z], max_lod),
            bounds.min,
            bounds.max,
        )
    }

    /// Refine the octree by subdividing high-error leaves.
    ///
    /// After optimization, leaves with error above `threshold` are
    /// subdivided to increase local resolution.
    pub fn refine(&mut self, threshold: f32) {
        let hidden = self.sdf_hidden_dim;
        let freq = self.sdf_num_freq;
        let max_d = self.max_depth;
        Self::refine_node(&mut self.root, threshold, max_d, hidden, freq);
    }

    fn refine_node(
        node: &mut OctreeNode,
        threshold: f32,
        max_depth: usize,
        hidden_dim: usize,
        num_freq: usize,
    ) {
        match node {
            OctreeNode::Leaf {
                bounds,
                depth,
                error,
                ..
            } => {
                if *error > threshold && *depth < max_depth {
                    let sub_bounds = bounds.subdivide();
                    let current_depth = *depth;
                    let parent_bounds = *bounds;

                    let children: [OctreeNode; 8] = std::array::from_fn(|i| {
                        let center = sub_bounds[i].center();
                        let extent = sub_bounds[i].extent();
                        OctreeNode::Leaf {
                            bounds: sub_bounds[i],
                            sdf: LocalSDF::new(hidden_dim, num_freq, center, extent),
                            depth: current_depth + 1,
                            error: f32::MAX,
                        }
                    });

                    *node = OctreeNode::Internal {
                        bounds: parent_bounds,
                        children: Box::new(children),
                        depth: current_depth,
                    };
                }
            }
            OctreeNode::Internal { children, .. } => {
                for child in children.as_mut() {
                    Self::refine_node(child, threshold, max_depth, hidden_dim, num_freq);
                }
            }
            OctreeNode::Empty { .. } => {}
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aabb() {
        let aabb = AABB::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        assert_eq!(aabb.center(), [0.5, 0.5, 0.5]);
        assert!((aabb.extent() - 0.5).abs() < 1e-6);
        assert!(aabb.contains([0.5, 0.5, 0.5]));
        assert!(!aabb.contains([1.5, 0.5, 0.5]));
    }

    #[test]
    fn test_aabb_subdivide() {
        let aabb = AABB::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let children = aabb.subdivide();
        assert_eq!(children.len(), 8);

        // First child should be [0, 0.5]³
        let c0 = &children[0];
        assert!((c0.max[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_octree_creation() {
        let bounds = AABB::new([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
        let tree = AdaptiveOctree::new(bounds, 4);
        assert_eq!(tree.num_leaves(), 1);
        assert_eq!(tree.num_nodes(), 1);
    }

    #[test]
    fn test_octree_query() {
        let bounds = AABB::new([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
        let tree = AdaptiveOctree::new(bounds, 4);

        // Query should return a value (from the root leaf SDF)
        let val = tree.query_sdf([0.0, 0.0, 0.0]);
        assert!(val.is_finite());
    }

    #[test]
    fn test_octree_depth_init() {
        let bounds = AABB::new([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
        let mut tree = AdaptiveOctree::new(bounds, 4);

        // Initialize with some surface points
        tree.init_from_depth(&[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], 2);

        assert!(tree.num_leaves() > 1, "Should have subdivided");
        assert!(tree.num_nodes() > 1, "Should have more nodes");
    }

    #[test]
    fn test_octree_lod_query() {
        let bounds = AABB::new([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
        let mut tree = AdaptiveOctree::new(bounds, 4);
        tree.init_from_depth(&[[0.0, 0.0, 0.0]], 3);

        // Both LOD levels should return finite values
        let val_coarse = tree.query_sdf_lod([0.0, 0.0, 0.0], 1);
        let val_fine = tree.query_sdf_lod([0.0, 0.0, 0.0], 3);
        assert!(val_coarse.is_finite());
        assert!(val_fine.is_finite());
    }

    #[test]
    fn test_octree_parameters() {
        let bounds = AABB::new([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
        let tree = AdaptiveOctree::new(bounds, 4);
        let params = tree.parameters();
        assert!(!params.is_empty());
    }

    #[test]
    fn test_octree_mesh_extraction() {
        let bounds = AABB::new([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
        let tree = AdaptiveOctree::new(bounds, 2);

        // Extract at low resolution
        let mesh = tree.extract_mesh(8);
        // Mesh may or may not have triangles depending on SDF values
        // but should not panic
        assert!(mesh.num_vertices() >= 0);
    }
}
