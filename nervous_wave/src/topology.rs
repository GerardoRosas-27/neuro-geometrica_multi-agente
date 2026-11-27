//! Directed graph topologies for Prueba A (N = 64 wave oscillators):
//!   - `NervousSystem`       (E = 256): Bipolar afferents -> 4 modular ganglia w/ lateral inhibition
//!                                      + 32 long-range myelinated commissural tracts -> 16 convergent hubs.
//!   - `GridLocalLattice`    (E = 256): 8x8 local 2D sheet, Past=row 0, Future=row 7, Hubs=rows 3..4, 0 long axons.
//!   - `RandomErdosRenyi`    (E = 256): Unstructured sparse random directed graph (exact 4 out-edges/node).
//!   - `AllToAllFlat`        (E = 4096): Dense 64x64 global flat coupling (16x more synapses).
//!   - `NervousNoDelay`      (E = 256): Exact NervousSystem topology with phase/delay disabled (real diffusion).
//!   - `NervousNoInhibition` (E = 256): Exact NervousSystem topology with inhibitory signs flipped to +1.

use crate::rng::Rng64;

pub const N_NODES: usize = 64;
pub const N_PAST_REC: usize = 8; // nodes 0..8
#[allow(dead_code)]
pub const N_FUT_REC: usize = 8; // nodes 8..16
pub const HUB_START: usize = 48; // nodes 48..64 (16 convergent readout hubs)
pub const N_HUBS: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TopologyKind {
    NervousSystem,
    GridLocalLattice,
    RandomErdosRenyi,
    AllToAllFlat,
    NervousNoDelay,
    NervousNoInhibition,
}

impl TopologyKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::NervousSystem => "nervous_system",
            Self::GridLocalLattice => "grid_local_lattice",
            Self::RandomErdosRenyi => "random_erdos_renyi",
            Self::AllToAllFlat => "all_to_all_flat",
            Self::NervousNoDelay => "nervous_no_delay",
            Self::NervousNoInhibition => "nervous_no_inhibition",
        }
    }

    pub fn is_complex_wave(self) -> bool {
        self != Self::NervousNoDelay
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DirectedEdge {
    pub src: usize,
    pub dst: usize,
    /// +1.0 for excitatory/myelinated axons, -1.0 for lateral-inhibitory interneurons (pi phase flip).
    pub sign: f64,
    /// Physical pathway class: 0=sensory, 1=local_exc, 2=local_inh, 3=myelinated_tract, 4=hub_proj, 5=generic.
    pub pathway: u8,
}

#[derive(Clone, Debug)]
pub struct WaveTopology {
    pub kind: TopologyKind,
    pub edges: Vec<DirectedEdge>,
    #[allow(dead_code)]
    pub in_deg: [usize; N_NODES],
    pub inv_sqrt_indeg: [f64; N_NODES],
    /// 2D layout coordinates (x, y) in [0, 1] x [0, 1] for physical distance & visualization.
    pub coords: [(f64, f64); N_NODES],
}

fn nervous_coords() -> [(f64, f64); N_NODES] {
    let mut c = [(0.0, 0.0); N_NODES];
    // Past receptors along left-top arc
    for i in 0..8 {
        c[i] = (0.06, 0.08 + 0.84 * (i as f64) / 7.0);
    }
    // Future receptors along right arc
    for i in 0..8 {
        c[8 + i] = (0.94, 0.08 + 0.84 * (i as f64) / 7.0);
    }
    // 4 ganglia: G0 (left-top), G1 (left-bot), G2 (right-top), G3 (right-bot)
    let centers = [(0.28, 0.28), (0.28, 0.72), (0.72, 0.28), (0.72, 0.72)];
    for (g, &(cx, cy)) in centers.iter().enumerate() {
        for k in 0..8 {
            let ang = 2.0 * std::f64::consts::PI * (k as f64) / 8.0;
            c[16 + g * 8 + k] = (cx + 0.085 * ang.cos(), cy + 0.085 * ang.sin());
        }
    }
    // 16 convergent association hubs down the central midline column (2x8)
    for h in 0..16 {
        let col = if h % 2 == 0 { 0.46 } else { 0.54 };
        let row = 0.08 + 0.84 * ((h / 2) as f64) / 7.0;
        c[48 + h] = (col, row);
    }
    c
}

fn grid_coords() -> [(f64, f64); N_NODES] {
    let mut c = [(0.0, 0.0); N_NODES];
    // Logical node -> (row, col):
    // row 0: Past 0..8; row 7: Future 8..16; rows 1,2,5,6: internal 16..48; rows 3,4: hubs 48..64
    for col in 0..8 {
        let y = 0.06 + 0.88 * (col as f64) / 7.0;
        c[col] = (0.06, y); // row 0
        c[8 + col] = (0.94, y); // row 7
        c[16 + col] = (0.06 + 0.88 * (1.0 / 7.0), y); // row 1
        c[24 + col] = (0.06 + 0.88 * (2.0 / 7.0), y); // row 2
        c[32 + col] = (0.06 + 0.88 * (5.0 / 7.0), y); // row 5
        c[40 + col] = (0.06 + 0.88 * (6.0 / 7.0), y); // row 6
        c[48 + col] = (0.06 + 0.88 * (3.0 / 7.0), y); // row 3 (hubs)
        c[56 + col] = (0.06 + 0.88 * (4.0 / 7.0), y); // row 4 (hubs)
    }
    c
}

fn build_nervous_edges(allow_inhibition: bool) -> Vec<DirectedEdge> {
    let mut edges = Vec::with_capacity(256);

    // 1. Bipolar sensory afferent bundles (64 edges):
    //    Past (0..8) -> G0 (16..24) & G1 (24..32); Future (8..16) -> G2 (32..40) & G3 (40..48)
    for p in 0..8 {
        edges.push(DirectedEdge { src: p, dst: 16 + p, sign: 1.0, pathway: 0 });
        edges.push(DirectedEdge { src: p, dst: 16 + (p + 1) % 8, sign: 1.0, pathway: 0 });
        edges.push(DirectedEdge { src: p, dst: 24 + p, sign: 1.0, pathway: 0 });
        edges.push(DirectedEdge { src: p, dst: 24 + (p + 2) % 8, sign: 1.0, pathway: 0 });
    }
    for f in 0..8 {
        let src = 8 + f;
        edges.push(DirectedEdge { src, dst: 32 + f, sign: 1.0, pathway: 0 });
        edges.push(DirectedEdge { src, dst: 32 + (f + 1) % 8, sign: 1.0, pathway: 0 });
        edges.push(DirectedEdge { src, dst: 40 + f, sign: 1.0, pathway: 0 });
        edges.push(DirectedEdge { src, dst: 40 + (f + 2) % 8, sign: 1.0, pathway: 0 });
    }

    // 2. Intra-ganglion canonical E-I microcircuits (96 edges = 4 ganglia x 24 edges):
    //    - Neurons 0..6 form an unbroken bidirectional excitatory wave ring (12 edges)
    //      and drive local inhibitory interneurons 6, 7 (6 E->I edges).
    //    - Inhibitory interneurons 6, 7 deliver lateral divisive shunting back onto 0..6 (6 I->E edges).
    for g in 0..4 {
        let base = 16 + g * 8;
        for u in 0..6 {
            // Unbroken bidirectional excitatory wave resonator around 0..6
            edges.push(DirectedEdge { src: base + u, dst: base + (u + 1) % 6, sign: 1.0, pathway: 1 });
            edges.push(DirectedEdge { src: base + u, dst: base + (u + 5) % 6, sign: 1.0, pathway: 1 });
            // E -> I drive into local inhibitory interneurons (6 and 7)
            edges.push(DirectedEdge { src: base + u, dst: base + 6 + (u % 2), sign: 1.0, pathway: 1 });
        }
        // I -> E lateral divisive shunting (interneuron 6 -> {0,2,4}, interneuron 7 -> {1,3,5})
        let inh_sign = if allow_inhibition { -1.0 } else { 1.0 };
        let inh_pw = if allow_inhibition { 2 } else { 1 };
        for &target in &[0usize, 2, 4] {
            edges.push(DirectedEdge { src: base + 6, dst: base + target, sign: inh_sign, pathway: inh_pw });
        }
        for &target in &[1usize, 3, 5] {
            edges.push(DirectedEdge { src: base + 7, dst: base + target, sign: inh_sign, pathway: inh_pw });
        }
    }

    // 3. Sparse long-range myelinated commissural tracts (32 edges):
    //    Bidirectional high-velocity bridges between Past-side (G0, G1) and Future-side (G2, G3)
    let bridge_pairs: [(usize, usize, [usize; 4]); 8] = [
        (0, 2, [0, 1, 2, 3]),
        (2, 0, [0, 1, 2, 3]),
        (1, 3, [0, 1, 2, 3]),
        (3, 1, [0, 1, 2, 3]),
        (0, 3, [2, 3, 4, 5]),
        (3, 0, [2, 3, 4, 5]),
        (1, 2, [2, 3, 4, 5]),
        (2, 1, [2, 3, 4, 5]),
    ];
    for (ga, gb, idxs) in bridge_pairs {
        let ba = 16 + ga * 8;
        let bb = 16 + gb * 8;
        for k in idxs {
            edges.push(DirectedEdge {
                src: ba + k,
                dst: bb + k,
                sign: 1.0,
                pathway: 3,
            });
        }
    }

    // 4. Convergent association hub projections (64 edges = 16 hubs x 4 converging tracts):
    //    Each hub receives 4 distinct converging tracts (2 Past-side G0/G1 + 2 Future-side G2/G3).
    for h in 0..N_HUBS {
        let dst = HUB_START + h;
        let srcs = [
            16 + (h % 6),
            24 + ((h * 5 + 1) % 6),
            32 + ((h * 3 + 2) % 6),
            40 + ((h * 7 + 3) % 6),
        ];
        for src in srcs {
            edges.push(DirectedEdge {
                src,
                dst,
                sign: 1.0,
                pathway: 4,
            });
        }
    }

    assert_eq!(edges.len(), 256);
    edges
}

fn build_grid_edges() -> Vec<DirectedEdge> {
    // Map (row 0..8, col 0..8) -> logical node id
    let node_at = |r: usize, c: usize| -> usize {
        match r {
            0 => c,              // Past receptors
            1 => 16 + c,         // Internal
            2 => 24 + c,         // Internal
            3 => 48 + c,         // Convergent Hubs (top center)
            4 => 56 + c,         // Convergent Hubs (bottom center)
            5 => 32 + c,         // Internal
            6 => 40 + c,         // Internal
            7 => 8 + c,          // Future receptors
            _ => unreachable!(),
        }
    };

    let mut edges = Vec::with_capacity(256);
    for r in 0..8 {
        for c in 0..8 {
            let src = node_at(r, c);
            let left = node_at(r, (c + 7) % 8);
            let right = node_at(r, (c + 1) % 8);
            let (v1, v2) = if r == 0 {
                // Open top boundary: strictly local forward into row 1 (NO wrap to row 7!)
                (node_at(1, c), node_at(1, (c + 1) % 8))
            } else if r == 7 {
                // Open bottom boundary: strictly local forward into row 6 (NO wrap to row 0!)
                (node_at(6, c), node_at(6, (c + 1) % 8))
            } else {
                (node_at(r - 1, c), node_at(r + 1, c))
            };
            for dst in [left, right, v1, v2] {
                edges.push(DirectedEdge {
                    src,
                    dst,
                    sign: 1.0,
                    pathway: 5,
                });
            }
        }
    }
    assert_eq!(edges.len(), 256);
    edges
}

fn build_random_er_edges(rng: &mut Rng64) -> Vec<DirectedEdge> {
    let mut edges = Vec::with_capacity(256);
    for src in 0..N_NODES {
        let mut chosen = Vec::with_capacity(4);
        while chosen.len() < 4 {
            let dst = (rng.next_u64() as usize) % N_NODES;
            if dst != src && !chosen.contains(&dst) {
                chosen.push(dst);
            }
        }
        for dst in chosen {
            edges.push(DirectedEdge {
                src,
                dst,
                sign: 1.0,
                pathway: 5,
            });
        }
    }
    assert_eq!(edges.len(), 256);
    edges
}

fn build_all_to_all_edges() -> Vec<DirectedEdge> {
    let mut edges = Vec::with_capacity(N_NODES * N_NODES);
    for src in 0..N_NODES {
        for dst in 0..N_NODES {
            edges.push(DirectedEdge {
                src,
                dst,
                sign: 1.0,
                pathway: 5,
            });
        }
    }
    assert_eq!(edges.len(), 4096);
    edges
}

impl WaveTopology {
    pub fn build(kind: TopologyKind, rng: &mut Rng64) -> Self {
        let (edges, coords) = match kind {
            TopologyKind::NervousSystem | TopologyKind::NervousNoDelay => {
                (build_nervous_edges(true), nervous_coords())
            }
            TopologyKind::NervousNoInhibition => (build_nervous_edges(false), nervous_coords()),
            TopologyKind::GridLocalLattice => (build_grid_edges(), grid_coords()),
            TopologyKind::RandomErdosRenyi => (build_random_er_edges(rng), nervous_coords()),
            TopologyKind::AllToAllFlat => (build_all_to_all_edges(), nervous_coords()),
        };

        let mut in_deg = [0usize; N_NODES];
        for e in &edges {
            in_deg[e.dst] += 1;
        }
        let mut inv_sqrt_indeg = [1.0; N_NODES];
        for i in 0..N_NODES {
            inv_sqrt_indeg[i] = 1.0 / (in_deg[i].max(1) as f64).sqrt();
        }
        Self {
            kind,
            edges,
            in_deg,
            inv_sqrt_indeg,
            coords,
        }
    }

    /// Shortest directed hop distance from a source set to the 16 convergent hubs.
    pub fn mean_hop_distance_to_hubs(&self, sources: &[usize]) -> f64 {
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); N_NODES];
        for e in &self.edges {
            adj[e.src].push(e.dst);
        }
        let mut dist = [usize::MAX; N_NODES];
        let mut queue = std::collections::VecDeque::new();
        for &s in sources {
            dist[s] = 0;
            queue.push_back(s);
        }
        while let Some(u) = queue.pop_front() {
            for &v in &adj[u] {
                if dist[v] == usize::MAX {
                    dist[v] = dist[u] + 1;
                    queue.push_back(v);
                }
            }
        }
        let mut sum = 0.0;
        for h in HUB_START..(HUB_START + N_HUBS) {
            sum += if dist[h] == usize::MAX {
                99.0
            } else {
                dist[h] as f64
            };
        }
        sum / (N_HUBS as f64)
    }

    /// Create a lesioned copy where a deterministic fraction `frac` of edges are severed.
    pub fn with_lesion(&self, frac: f64, rng: &mut Rng64) -> Vec<bool> {
        let m = self.edges.len();
        let n_cut = ((m as f64) * frac).round() as usize;
        let mut idx: Vec<usize> = (0..m).collect();
        rng.shuffle(&mut idx);
        let mut active = vec![true; m];
        for &e_idx in &idx[..n_cut.min(m)] {
            active[e_idx] = false;
        }
        active
    }
}
