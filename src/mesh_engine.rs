use crate::geometry::Vec2;
use rand::{rngs::StdRng, Rng, SeedableRng};

#[derive(Clone, Copy, Debug)]
pub struct MeshConfig {
    pub width: usize,
    pub height: usize,
    pub spacing: f32,
    pub seed: u64,
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
