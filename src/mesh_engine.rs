use crate::geometry::Vec2;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::collections::HashSet;

const PHI_INV: f32 = 0.618_034;

#[derive(Clone, Copy, Debug)]
pub struct MeshConfig {
    pub width: usize,
    pub height: usize,
    pub spacing: f32,
    pub seed: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct FractalMeshConfig {
    pub levels: usize,
    pub branches_per_region: usize,
    pub target_dimension: f32,
    pub target_nodes: usize,
    pub base_radius: f32,
    pub lateral_link_weight: f32,
    pub parent_link_weight: f32,
}

impl Default for FractalMeshConfig {
    fn default() -> Self {
        Self {
            levels: 4,
            branches_per_region: 5,
            target_dimension: 2.65,
            target_nodes: 0,
            base_radius: 0.0,
            lateral_link_weight: 0.35,
            parent_link_weight: 1.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MeshNode {
    pub id: usize,
    pub position: Vec2,
    pub depth: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct MeshEdge {
    pub a: usize,
    pub b: usize,
    pub rest_length: f32,
    pub weight: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct MeshSimplex2 {
    pub a: usize,
    pub b: usize,
    pub c: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct MeshSimplex3 {
    pub a: usize,
    pub b: usize,
    pub c: usize,
    pub d: usize,
}

#[derive(Clone, Debug)]
pub struct MeshTopology {
    pub nodes: Vec<MeshNode>,
    pub edges: Vec<MeshEdge>,
    pub simplices: Vec<MeshSimplex2>,
    pub tetrahedra: Vec<MeshSimplex3>,
}

#[derive(Clone, Debug)]
pub struct MeshStats {
    pub nodes: usize,
    pub edges: usize,
    pub triangles: usize,
    pub tetrahedra: usize,
    pub depth_layers: usize,
}

pub struct SimplicialMeshEngine;

impl SimplicialMeshEngine {
    pub fn grid(config: MeshConfig) -> MeshTopology {
        Self::grid_3d(config, 1)
    }

    pub fn fractal_3d(config: MeshConfig, fractal: FractalMeshConfig) -> MeshTopology {
        let mut rng = StdRng::seed_from_u64(config.seed);
        let width = config.width.max(1) as f32;
        let height = config.height.max(1) as f32;
        let center = Vec2::new(width * config.spacing * 0.5, height * config.spacing * 0.5);
        let radius = if fractal.base_radius > 0.0 {
            fractal.base_radius
        } else {
            width.max(height) * config.spacing * 0.42
        };
        let branches = fractal.branches_per_region.clamp(2, 12);
        let level_count = fractal.levels.clamp(1, 7);
        let target_nodes = if fractal.target_nodes > 0 {
            fractal.target_nodes.max(1)
        } else {
            usize::MAX
        };
        let dimension = fractal.target_dimension.clamp(1.2, 2.95);
        let contraction = (branches as f32).powf(-1.0 / dimension);

        let mut nodes = Vec::new();
        nodes.push(MeshNode {
            id: 0,
            position: center,
            depth: 0.0,
        });

        let mut levels = vec![vec![0_usize]];
        let mut parent_of = vec![usize::MAX];
        let mut edge_keys = HashSet::new();
        let mut edges = Vec::new();
        let mut simplices = Vec::new();
        let mut tetrahedra = Vec::new();

        for level in 1..=level_count {
            if nodes.len() >= target_nodes {
                break;
            }
            let parent_level = levels[level - 1].clone();
            let mut current_level = Vec::new();
            let scale = contraction.powi(level as i32);
            let branch_radius = radius * scale;

            for (parent_ord, &parent_id) in parent_level.iter().enumerate() {
                let parent_position = nodes[parent_id].position;
                let parent_depth = nodes[parent_id].depth;
                let mut siblings = Vec::with_capacity(branches);

                for branch in 0..branches {
                    if nodes.len() >= target_nodes {
                        break;
                    }
                    let child_id = nodes.len();
                    let t = (branch as f32 + 0.5) / branches as f32;
                    let z = 1.0 - 2.0 * t;
                    let radial = (1.0 - z * z).sqrt();
                    let theta = std::f32::consts::TAU
                        * (branch as f32 * PHI_INV + parent_ord as f32 * PHI_INV * 0.5);
                    let jitter = 1.0 + rng.gen_range(-0.04..0.04);
                    let offset = Vec2::new(theta.cos() * radial, theta.sin() * radial)
                        * branch_radius
                        * jitter;

                    nodes.push(MeshNode {
                        id: child_id,
                        position: parent_position + offset,
                        depth: parent_depth + z * branch_radius * jitter,
                    });
                    parent_of.push(parent_id);
                    current_level.push(child_id);
                    siblings.push(child_id);

                    push_edge(
                        &nodes,
                        &mut edges,
                        &mut edge_keys,
                        parent_id,
                        child_id,
                        fractal.parent_link_weight,
                    );
                }

                for pair in siblings.windows(2) {
                    let a = pair[0];
                    let b = pair[1];
                    push_edge(
                        &nodes,
                        &mut edges,
                        &mut edge_keys,
                        a,
                        b,
                        fractal.lateral_link_weight,
                    );
                    simplices.push(MeshSimplex2 {
                        a: parent_id,
                        b: a,
                        c: b,
                    });
                }

                if siblings.len() > 2 {
                    let first = siblings[0];
                    let last = siblings[siblings.len() - 1];
                    push_edge(
                        &nodes,
                        &mut edges,
                        &mut edge_keys,
                        first,
                        last,
                        fractal.lateral_link_weight,
                    );
                    simplices.push(MeshSimplex2 {
                        a: parent_id,
                        b: last,
                        c: first,
                    });
                }

                for window in siblings.windows(3) {
                    tetrahedra.push(MeshSimplex3 {
                        a: parent_id,
                        b: window[0],
                        c: window[1],
                        d: window[2],
                    });
                }
            }

            if current_level.is_empty() {
                break;
            }
            levels.push(current_level);
        }

        for level in 2..levels.len() {
            for &node_id in &levels[level] {
                let parent = parent_of[node_id];
                if parent == usize::MAX {
                    continue;
                }
                let grandparent = parent_of[parent];
                if grandparent == usize::MAX {
                    continue;
                }
                push_edge(
                    &nodes,
                    &mut edges,
                    &mut edge_keys,
                    grandparent,
                    node_id,
                    fractal.lateral_link_weight * contraction,
                );
            }
        }

        MeshTopology {
            nodes,
            edges,
            simplices,
            tetrahedra,
        }
    }

    pub fn grid_3d(config: MeshConfig, depth_layers: usize) -> MeshTopology {
        let layers = depth_layers.max(1);
        let layer_size = config.width * config.height;
        let mut rng = StdRng::seed_from_u64(config.seed);
        let mut nodes = Vec::with_capacity(layer_size * layers);

        for z in 0..layers {
            for y in 0..config.height {
                for x in 0..config.width {
                    let jitter = Vec2::new(rng.gen_range(-3.0..3.0), rng.gen_range(-3.0..3.0));
                    let id = z * layer_size + y * config.width + x;
                    nodes.push(MeshNode {
                        id,
                        position: Vec2::new(x as f32 * config.spacing, y as f32 * config.spacing)
                            + jitter,
                        depth: z as f32 * config.spacing,
                    });
                }
            }
        }

        let mut topology = MeshTopology {
            nodes,
            edges: Vec::new(),
            simplices: Vec::new(),
            tetrahedra: Vec::new(),
        };

        for z in 0..layers {
            for y in 0..config.height {
                for x in 0..config.width {
                    let id = z * layer_size + y * config.width + x;
                    if x + 1 < config.width {
                        topology.edges.push(MeshEdge {
                            a: id,
                            b: z * layer_size + y * config.width + (x + 1),
                            rest_length: config.spacing,
                            weight: 1.0,
                        });
                    }
                    if y + 1 < config.height {
                        topology.edges.push(MeshEdge {
                            a: id,
                            b: z * layer_size + (y + 1) * config.width + x,
                            rest_length: config.spacing,
                            weight: 1.0,
                        });
                    }
                    if layers > 1 && z + 1 < layers {
                        topology.edges.push(MeshEdge {
                            a: id,
                            b: (z + 1) * layer_size + y * config.width + x,
                            rest_length: config.spacing,
                            weight: 1.0,
                        });
                    }
                    if x + 1 < config.width && y + 1 < config.height {
                        let bx = z * layer_size + y * config.width + (x + 1);
                        let cy = z * layer_size + (y + 1) * config.width + x;
                        let dxy = z * layer_size + (y + 1) * config.width + (x + 1);
                        topology.edges.push(MeshEdge {
                            a: id,
                            b: dxy,
                            rest_length: config.spacing * 2.0_f32.sqrt(),
                            weight: 0.45,
                        });
                        topology.simplices.push(MeshSimplex2 {
                            a: id,
                            b: bx,
                            c: dxy,
                        });
                        topology.simplices.push(MeshSimplex2 {
                            a: id,
                            b: cy,
                            c: dxy,
                        });

                        if layers > 1 && z + 1 < layers {
                            let up = (z + 1) * layer_size + y * config.width + x;
                            topology.tetrahedra.push(MeshSimplex3 {
                                a: id,
                                b: bx,
                                c: cy,
                                d: up,
                            });
                        }
                    }
                }
            }
        }

        topology
    }
}

fn push_edge(
    nodes: &[MeshNode],
    edges: &mut Vec<MeshEdge>,
    edge_keys: &mut HashSet<(usize, usize)>,
    a: usize,
    b: usize,
    weight: f32,
) {
    let key = if a < b { (a, b) } else { (b, a) };
    if !edge_keys.insert(key) {
        return;
    }

    let pa = nodes[a].position;
    let pb = nodes[b].position;
    let dz = nodes[b].depth - nodes[a].depth;
    let rest_length = ((pb - pa).length_squared() + dz * dz).sqrt().max(1.0);
    edges.push(MeshEdge {
        a,
        b,
        rest_length,
        weight,
    });
}

impl MeshTopology {
    pub fn stats(&self, depth_layers: usize) -> MeshStats {
        MeshStats {
            nodes: self.nodes.len(),
            edges: self.edges.len(),
            triangles: self.simplices.len(),
            tetrahedra: self.tetrahedra.len(),
            depth_layers,
        }
    }
}
