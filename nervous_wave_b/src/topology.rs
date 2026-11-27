//! Topologies for Prueba B (N=128, E=1,024 sparse arms vs N=64, E=4,096 dense control).
//!
//! Six arms compared fairly with identical 64-dim hub readout (32 hubs x [Re, |psi|^2 - 0.5]):
//!   1. NervousCoherenceB      (N=128, E=1,024, bipolar ganglia + myelinated commissures + Kuramoto gate)
//!   2. NervousNoCoherenceGate (N=128, E=1,024, same nervous topology, lambda_coh = 0)
//!   3. NervousNoDelay         (N=128, E=1,024, same nervous topology, theta_e = omega_i = 0, lambda_coh = 0)
//!   4. GridLocal128           (N=128, E=1,024, 8x16 local Moore lattice, 0 long axons + Kuramoto gate)
//!   5. RandomEr128            (N=128, E=1,024, directed Erdos-Renyi in_deg=8 + Kuramoto gate)
//!   6. FlatDense4096          (N=64,  E=4,096, 4x MORE synapses + Kuramoto gate)

use crate::rng::Rng64;
use std::collections::VecDeque;

pub const N_SPARSE: usize = 128;
pub const E_SPARSE: usize = 1024;
pub const N_PAST_REC: usize = 16;
pub const N_FUT_REC: usize = 16;
pub const N_GANGLIA: usize = 8;
pub const GANGLION_SIZE: usize = 8;
pub const N_HUBS: usize = 32;
pub const N_FLAT: usize = 64;
pub const E_FLAT: usize = 4096;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TopologyKind {
    NervousCoherenceB,
    NervousNoCoherenceGate,
    NervousNoDelay,
    GridLocal128,
    RandomEr128,
    FlatDense4096,
}

impl TopologyKind {
    pub fn name(&self) -> &'static str {
        match self {
            Self::NervousCoherenceB => "nervous_coherence_b",
            Self::NervousNoCoherenceGate => "nervous_no_coherence_gate",
            Self::NervousNoDelay => "nervous_no_delay",
            Self::GridLocal128 => "grid_local_128",
            Self::RandomEr128 => "random_er_128",
            Self::FlatDense4096 => "flat_dense_4096",
        }
    }

    pub fn all() -> [TopologyKind; 6] {
        [
            Self::NervousCoherenceB,
            Self::NervousNoCoherenceGate,
            Self::NervousNoDelay,
            Self::GridLocal128,
            Self::RandomEr128,
            Self::FlatDense4096,
        ]
    }

    pub fn coherence_gate_enabled(&self) -> bool {
        !matches!(self, Self::NervousNoCoherenceGate | Self::NervousNoDelay)
    }

    pub fn phase_enabled(&self) -> bool {
        !matches!(self, Self::NervousNoDelay)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgeKind {
    Afferent,
    IntraGanglion,
    Commissure,
    HubProjection,
    LocalGrid,
    Random,
    Dense,
}

#[derive(Clone, Copy, Debug)]
pub struct Edge {
    pub src: usize,
    pub dst: usize,
    #[allow(dead_code)]
    pub kind: EdgeKind,
}

#[derive(Clone, Debug)]
pub struct GraphTopology {
    pub kind: TopologyKind,
    pub n_nodes: usize,
    pub edges: Vec<Edge>,
    pub past_receptors: Vec<usize>,
    pub future_receptors: Vec<usize>,
    pub past_ganglia: Vec<usize>,
    pub future_ganglia: Vec<usize>,
    pub hubs: Vec<usize>,
    #[allow(dead_code)]
    pub mean_in_deg: f64,
    pub syn_scale: f64,
}

impl GraphTopology {
    pub fn build(kind: TopologyKind, seed: u64) -> Self {
        match kind {
            TopologyKind::NervousCoherenceB
            | TopologyKind::NervousNoCoherenceGate
            | TopologyKind::NervousNoDelay => build_nervous_128(kind),
            TopologyKind::GridLocal128 => build_grid_local_128(),
            TopologyKind::RandomEr128 => build_random_er_128(seed),
            TopologyKind::FlatDense4096 => build_flat_dense_4096(),
        }
    }

    /// Mean shortest directed hop distance from all 32 receptors to all 32 hubs.
    pub fn mean_receptor_to_hub_hops(&self) -> f64 {
        let mut adj = vec![Vec::new(); self.n_nodes];
        for e in &self.edges {
            adj[e.src].push(e.dst);
        }
        let mut total = 0.0;
        let mut count = 0usize;
        let mut sources = Vec::with_capacity(self.past_receptors.len() + self.future_receptors.len());
        sources.extend_from_slice(&self.past_receptors);
        sources.extend_from_slice(&self.future_receptors);

        for &s in &sources {
            let mut dist = vec![usize::MAX; self.n_nodes];
            let mut q = VecDeque::new();
            dist[s] = 0;
            q.push_back(s);
            while let Some(u) = q.pop_front() {
                let du = dist[u];
                for &v in &adj[u] {
                    if dist[v] == usize::MAX {
                        dist[v] = du + 1;
                        q.push_back(v);
                    }
                }
            }
            for &h in &self.hubs {
                if dist[h] < usize::MAX {
                    total += dist[h] as f64;
                    count += 1;
                }
            }
        }
        if count == 0 {
            f64::INFINITY
        } else {
            total / count as f64
        }
    }
}

fn build_nervous_128(kind: TopologyKind) -> GraphTopology {
    let mut edges = Vec::with_capacity(E_SPARSE);

    // 1) 256 bipolar sensory afferents (16 Past -> G0..G3; 16 Future -> G4..G7)
    for r in 0..N_PAST_REC {
        for g in 0..4usize {
            let base = 32 + GANGLION_SIZE * g;
            let k1 = r % GANGLION_SIZE;
            let k2 = (r + 3 + g) % GANGLION_SIZE;
            edges.push(Edge {
                src: r,
                dst: base + k1,
                kind: EdgeKind::Afferent,
            });
            edges.push(Edge {
                src: r,
                dst: base + k2,
                kind: EdgeKind::Afferent,
            });
        }
    }
    for r in 0..N_FUT_REC {
        let src = N_PAST_REC + r;
        for g_off in 0..4usize {
            let g = 4 + g_off;
            let base = 32 + GANGLION_SIZE * g;
            let k1 = r % GANGLION_SIZE;
            let k2 = (r + 3 + g_off) % GANGLION_SIZE;
            edges.push(Edge {
                src,
                dst: base + k1,
                kind: EdgeKind::Afferent,
            });
            edges.push(Edge {
                src,
                dst: base + k2,
                kind: EdgeKind::Afferent,
            });
        }
    }

    // 2) 384 intra-ganglion unbroken resonant edges (8 ganglia x 8 neurons x 6 offsets)
    let offsets = [1usize, 2, 3, 5, 6, 7];
    for g in 0..N_GANGLIA {
        let base = 32 + GANGLION_SIZE * g;
        for k in 0..GANGLION_SIZE {
            for &off in &offsets {
                edges.push(Edge {
                    src: base + k,
                    dst: base + (k + off) % GANGLION_SIZE,
                    kind: EdgeKind::IntraGanglion,
                });
            }
        }
    }

    // 3) 128 myelinated long-range commissural tracts (16 cross-hemisphere ganglion pairs x 8)
    for gp in 0..4usize {
        for gf in 4..8usize {
            let base_p = 32 + GANGLION_SIZE * gp;
            let base_f = 32 + GANGLION_SIZE * gf;
            for m in 0..4usize {
                // Past -> Future
                edges.push(Edge {
                    src: base_p + (2 * m + gf) % GANGLION_SIZE,
                    dst: base_f + (2 * m + gp) % GANGLION_SIZE,
                    kind: EdgeKind::Commissure,
                });
                // Future -> Past
                edges.push(Edge {
                    src: base_f + (2 * m + 1 + gp) % GANGLION_SIZE,
                    dst: base_p + (2 * m + 1 + gf) % GANGLION_SIZE,
                    kind: EdgeKind::Commissure,
                });
            }
        }
    }

    // 4) 256 convergent bilateral hub projections (32 hubs x 8 ganglia)
    for h in 0..N_HUBS {
        let dst = 96 + h;
        for g in 0..N_GANGLIA {
            let base = 32 + GANGLION_SIZE * g;
            let k = (h + 3 * g) % GANGLION_SIZE;
            edges.push(Edge {
                src: base + k,
                dst,
                kind: EdgeKind::HubProjection,
            });
        }
    }

    debug_assert_eq!(edges.len(), E_SPARSE);
    let mean_in_deg = edges.len() as f64 / N_SPARSE as f64;
    GraphTopology {
        kind,
        n_nodes: N_SPARSE,
        edges,
        past_receptors: (0..16).collect(),
        future_receptors: (16..32).collect(),
        past_ganglia: (32..64).collect(),
        future_ganglia: (64..96).collect(),
        hubs: (96..128).collect(),
        mean_in_deg,
        syn_scale: 1.0 / mean_in_deg.sqrt(),
    }
}

fn grid_node_id(row: usize, col: usize) -> usize {
    match row {
        0 => col,            // Past receptors 0..16
        1 => 32 + col,       // Past-side internal 32..48
        2 => 48 + col,       // Past-side internal 48..64
        3 => 96 + col,       // Midline Hubs 96..112
        4 => 112 + col,      // Midline Hubs 112..128
        5 => 64 + col,       // Future-side internal 64..80
        6 => 80 + col,       // Future-side internal 80..96
        7 => 16 + col,       // Future receptors 16..32
        _ => unreachable!(),
    }
}

fn build_grid_local_128() -> GraphTopology {
    let rows = 8usize;
    let cols = 16usize;
    let mut edges = Vec::with_capacity(E_SPARSE);

    for r in 0..rows {
        for c in 0..cols {
            let src = grid_node_id(r, c);
            // Local Moore neighborhood (horizontal wrap mod 16, open vertical boundaries)
            for dr in -1i32..=1 {
                let nr = r as i32 + dr;
                if nr < 0 || nr >= rows as i32 {
                    continue;
                }
                for dc in -1i32..=1 {
                    if dr == 0 && dc == 0 {
                        continue;
                    }
                    let nc = (c as i32 + dc).rem_euclid(cols as i32) as usize;
                    edges.push(Edge {
                        src,
                        dst: grid_node_id(nr as usize, nc),
                        kind: EdgeKind::LocalGrid,
                    });
                }
            }
            // Boundary rows (0 and 7) have 5 Moore neighbors; add 3 strictly local (|dr|<=1, |dc|=2)
            // neighbors to match exact E=1,024 budget without creating any vertical shortcuts.
            if r == 0 || r == rows - 1 {
                let c_m2 = (c + cols - 2) % cols;
                let c_p2 = (c + 2) % cols;
                let nr_in = if r == 0 { 1 } else { rows - 2 };
                edges.push(Edge {
                    src,
                    dst: grid_node_id(r, c_m2),
                    kind: EdgeKind::LocalGrid,
                });
                edges.push(Edge {
                    src,
                    dst: grid_node_id(r, c_p2),
                    kind: EdgeKind::LocalGrid,
                });
                edges.push(Edge {
                    src,
                    dst: grid_node_id(nr_in, c_p2),
                    kind: EdgeKind::LocalGrid,
                });
            }
        }
    }

    debug_assert_eq!(edges.len(), E_SPARSE);
    let mean_in_deg = edges.len() as f64 / N_SPARSE as f64;
    GraphTopology {
        kind: TopologyKind::GridLocal128,
        n_nodes: N_SPARSE,
        edges,
        past_receptors: (0..16).collect(),
        future_receptors: (16..32).collect(),
        past_ganglia: (32..64).collect(),
        future_ganglia: (64..96).collect(),
        hubs: (96..128).collect(),
        mean_in_deg,
        syn_scale: 1.0 / mean_in_deg.sqrt(),
    }
}

fn build_random_er_128(seed: u64) -> GraphTopology {
    let mut rng = Rng64::new(seed ^ 0xE212_8B0B);
    let mut edges = Vec::with_capacity(E_SPARSE);
    let mut candidates: Vec<usize> = (0..N_SPARSE).collect();

    for dst in 0..N_SPARSE {
        rng.shuffle(&mut candidates);
        let mut added = 0usize;
        for &src in &candidates {
            if src == dst {
                continue;
            }
            edges.push(Edge {
                src,
                dst,
                kind: EdgeKind::Random,
            });
            added += 1;
            if added == 8 {
                break;
            }
        }
    }

    debug_assert_eq!(edges.len(), E_SPARSE);
    let mean_in_deg = edges.len() as f64 / N_SPARSE as f64;
    GraphTopology {
        kind: TopologyKind::RandomEr128,
        n_nodes: N_SPARSE,
        edges,
        past_receptors: (0..16).collect(),
        future_receptors: (16..32).collect(),
        past_ganglia: (32..64).collect(),
        future_ganglia: (64..96).collect(),
        hubs: (96..128).collect(),
        mean_in_deg,
        syn_scale: 1.0 / mean_in_deg.sqrt(),
    }
}

fn build_flat_dense_4096() -> GraphTopology {
    let mut edges = Vec::with_capacity(E_FLAT);
    for src in 0..N_FLAT {
        for dst in 0..N_FLAT {
            edges.push(Edge {
                src,
                dst,
                kind: EdgeKind::Dense,
            });
        }
    }
    debug_assert_eq!(edges.len(), E_FLAT);
    let mean_in_deg = edges.len() as f64 / N_FLAT as f64;
    GraphTopology {
        kind: TopologyKind::FlatDense4096,
        n_nodes: N_FLAT,
        edges,
        past_receptors: (0..16).collect(),
        future_receptors: (16..32).collect(),
        past_ganglia: (0..16).collect(),
        future_ganglia: (16..32).collect(),
        hubs: (32..64).collect(),
        mean_in_deg,
        syn_scale: 1.0 / mean_in_deg.sqrt(),
    }
}
