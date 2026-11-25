//! Three genuinely independent domains for autonomous spectral structure learning
//! and frozen-core cross-domain transfer:
//!   - Domain A (Source, R^20): Abstract Causal & Relational Ecosystem (24 entities, 4 symmetries)
//!   - Domain B (Target 1, R^12): Continuous Coupled Nonlinear Physical Oscillators (18 oscillators)
//!   - Domain C (Target 2, R^16): Discrete Relational Graph Z_5 x {0..3} (20 symbolic nodes)
//!
//! Zero shared entities, zero shared sensory dimensions, zero connection to ORBITA or CDT-RQM.

use crate::rng::Rng64;
use std::f64::consts::PI;

#[derive(Clone, Debug)]
pub struct RelPair {
    pub a: usize,
    pub b: usize,
    pub reg: Vec<f64>,
    pub cls: usize,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct TwoHopQuery {
    pub a: usize,
    pub b: usize,
    pub c: usize,
    pub pair_ac_idx: usize,
}

#[derive(Clone, Debug)]
pub struct DomainData {
    pub name: &'static str,
    pub obs_dim: usize,
    pub reg_dim: usize,
    pub cls_dim: usize,
    pub n_entities: usize,
    /// Noisy opaque sensory observations per entity, standardized per feature.
    pub obs: Vec<Vec<f64>>,
    /// All N*(N-1) ordered pairs (a != b).
    pub pairs: Vec<RelPair>,
    /// Lookup matrix pair_idx[a * n_entities + b] = index into `pairs` (usize::MAX if a==b).
    pub pair_lookup: Vec<usize>,
    /// Ordered permutation of pair indices; prefixes of size n_few form nested train sets.
    pub perm_order: Vec<usize>,
    /// Default train count (Domain A = 220; B = 48; C = 60).
    pub default_train_n: usize,
    /// Two-hop unseen triples (Domain A).
    pub two_hop_queries: Vec<TwoHopQuery>,
    /// Ground-truth discrete node coordinates (tier, ring) for algebraic consistency in Domain C.
    pub node_tiers: Vec<usize>,
    #[allow(dead_code)]
    pub node_rings: Vec<usize>,
}

impl DomainData {
    pub fn train_indices(&self, n_train: usize) -> &[usize] {
        &self.perm_order[..n_train.min(self.perm_order.len())]
    }

    pub fn test_indices(&self, n_train: usize) -> &[usize] {
        let k = n_train.min(self.perm_order.len());
        &self.perm_order[k..]
    }

    /// Create a negative-control copy of Domain A where training targets (reg, cls)
    /// are randomly permuted across pairs — identical marginal statistics and inputs,
    /// zero genuine relational structure.
    pub fn make_shuffled_label_control(&self, rng: &mut Rng64) -> Self {
        let mut copy = self.clone();
        let n_tr = copy.default_train_n;
        let train_idx: Vec<usize> = copy.perm_order[..n_tr].to_vec();
        let mut perm = train_idx.clone();
        rng.shuffle(&mut perm);
        let orig_pairs = self.pairs.clone();
        for (&dst, &src) in train_idx.iter().zip(perm.iter()) {
            copy.pairs[dst].reg = orig_pairs[src].reg.clone();
            copy.pairs[dst].cls = orig_pairs[src].cls;
        }
        copy
    }
}

fn standardize_obs(obs: &mut [Vec<f64>]) {
    let n = obs.len();
    if n == 0 {
        return;
    }
    let d = obs[0].len();
    for j in 0..d {
        let mean = obs.iter().map(|v| v[j]).sum::<f64>() / (n as f64);
        let var = obs
            .iter()
            .map(|v| {
                let diff = v[j] - mean;
                diff * diff
            })
            .sum::<f64>()
            / (n as f64);
        let inv_std = 1.0 / (var + 1e-8).sqrt();
        for v in obs.iter_mut() {
            v[j] = (v[j] - mean) * inv_std;
        }
    }
}

fn standardize_reg_channels(pairs: &mut [RelPair], reg_dim: usize) {
    let n = pairs.len() as f64;
    for c in 0..reg_dim {
        let mean = pairs.iter().map(|p| p.reg[c]).sum::<f64>() / n;
        let var = pairs
            .iter()
            .map(|p| {
                let d = p.reg[c] - mean;
                d * d
            })
            .sum::<f64>()
            / n;
        let inv = 1.0 / (var + 1e-8).sqrt();
        for p in pairs.iter_mut() {
            p.reg[c] = (p.reg[c] - mean) * inv;
        }
    }
}

/// Build a coverage-first ordering of pairs so small prefixes cover every entity evenly
/// and remain class-balanced before filling remaining pairs in deterministic random order.
fn build_balanced_pair_order(
    n_entities: usize,
    pairs: &[RelPair],
    cls_dim: usize,
    rng: &mut Rng64,
) -> Vec<usize> {
    let total = pairs.len();
    let mut used = vec![false; total];
    let mut order = Vec::with_capacity(total);

    // Group candidate pair indices by class and shuffle each bucket.
    let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); cls_dim];
    for (idx, p) in pairs.iter().enumerate() {
        buckets[p.cls].push(idx);
    }
    for b in &mut buckets {
        rng.shuffle(b);
    }

    // Round 1: cover every entity as source/target while rotating across classes.
    let mut entity_deg = vec![0usize; n_entities];
    let mut cursors = vec![0usize; cls_dim];
    let mut step = 0usize;
    while order.len() < total {
        let mut added_any = false;
        for c_off in 0..cls_dim {
            let c = (step + c_off) % cls_dim;
            // Pick best unused pair in bucket `c` with minimum entity degree sum (lookahead up to 24).
            let b = &buckets[c];
            while cursors[c] < b.len() && used[b[cursors[c]]] {
                cursors[c] += 1;
            }
            if cursors[c] >= b.len() {
                continue;
            }
            let mut best_pos = cursors[c];
            let mut best_score = usize::MAX;
            let limit = (cursors[c] + 24).min(b.len());
            for pos in cursors[c]..limit {
                let idx = b[pos];
                if used[idx] {
                    continue;
                }
                let score = entity_deg[pairs[idx].a] + entity_deg[pairs[idx].b];
                if score < best_score {
                    best_score = score;
                    best_pos = pos;
                }
            }
            let chosen = b[best_pos];
            used[chosen] = true;
            entity_deg[pairs[chosen].a] += 1;
            entity_deg[pairs[chosen].b] += 1;
            order.push(chosen);
            added_any = true;
        }
        step += 1;
        if !added_any {
            break;
        }
    }
    debug_assert_eq!(order.len(), total);
    order
}

/// Domain A (Source): Abstract Causal & Relational Ecosystem in R^20.
/// K_A = 24 entities, 552 ordered pairs (220 train / 332 unseen), 4 relational symmetries:
///   0 = symmetric resonance (even in diff)
///   1 = directed causal dominance (odd in diff)
///   2 = chiral/cyclic cross-coupling (antisymmetric rotational)
///   3 = compositional binding (even in sum)
pub fn generate_domain_a(rng: &mut Rng64) -> DomainData {
    let n_entities = 24usize;
    let obs_dim = 20usize;
    let reg_dim = 4usize;
    let cls_dim = 4usize;

    // Latent 4D entity coordinates well-distributed in [-1.5, 1.5]^4.
    let mut latent = Vec::with_capacity(n_entities);
    for _ in 0..n_entities {
        let mut s = [0.0; 4];
        for v in &mut s {
            *v = rng.uniform_range(-1.45, 1.45);
        }
        latent.push(s);
    }

    // Unknown nonlinear sensory projection M_A (20x4) + b_A + noise.
    let mut m_a = vec![vec![0.0; 4]; obs_dim];
    let mut b_a = vec![0.0; obs_dim];
    for r in 0..obs_dim {
        for c in 0..4 {
            m_a[r][c] = rng.normal() * 0.65;
        }
        b_a[r] = rng.normal() * 0.25;
    }
    let mut obs = Vec::with_capacity(n_entities);
    for s in &latent {
        let mut x = vec![0.0; obs_dim];
        for r in 0..obs_dim {
            let mut acc = b_a[r];
            for c in 0..4 {
                acc += m_a[r][c] * s[c];
            }
            // Nonlinear sensory warping + mild measurement noise.
            x[r] = acc.tanh() + 0.15 * (0.8 * acc).sin() + rng.normal() * 0.03;
        }
        obs.push(x);
    }
    standardize_obs(&mut obs);

    // Two orthogonal unit directions for directed causal dominance spanning 2D antisymmetric subspace.
    let v_dir1 = rng.unit_vec(4);
    let mut v_dir2 = rng.unit_vec(4);
    let proj12: f64 = (0..4).map(|k| v_dir1[k] * v_dir2[k]).sum();
    let mut n2 = 0.0;
    for k in 0..4 {
        v_dir2[k] -= proj12 * v_dir1[k];
        n2 += v_dir2[k] * v_dir2[k];
    }
    let inv_n2 = 1.0 / (n2 + 1e-12).sqrt();
    for k in 0..4 {
        v_dir2[k] *= inv_n2;
    }

    let mut pairs = Vec::with_capacity(n_entities * (n_entities - 1));
    let mut pair_lookup = vec![usize::MAX; n_entities * n_entities];
    for a in 0..n_entities {
        for b in 0..n_entities {
            if a == b {
                continue;
            }
            let sa = latent[a];
            let sb = latent[b];
            let mut diff = [0.0; 4];
            let mut sum = [0.0; 4];
            let mut d2 = 0.0;
            for k in 0..4 {
                diff[k] = sb[k] - sa[k];
                sum[k] = sa[k] + sb[k];
                d2 += diff[k] * diff[k];
            }
            let dist = (d2 + 1e-12).sqrt();
            let dot_d1: f64 = (0..4).map(|k| v_dir1[k] * diff[k]).sum();
            let dot_d2: f64 = (0..4).map(|k| v_dir2[k] * diff[k]).sum();

            // 4 genuinely distinct 4D-isotropic structural symmetries:
            // y0: Isotropic symmetric resonance (even in all 4 coords of sb - sa)
            let y0 = (1.05 * dist).cos() * (-0.16 * d2).exp();
            // y1: Strictly antisymmetric 2-axis directed dominance (a -> b)
            let y1 = (0.95 * dot_d1).tanh() * (-0.08 * d2).exp() + 0.45 * (0.85 * dot_d2).sin();
            // y2: Full 4D symplectic chiral coupling across planes (0,1) and (2,3)
            let chiral = 0.80 * (sa[0] * sb[1] - sa[1] * sb[0])
                + 0.80 * (sa[2] * sb[3] - sa[3] * sb[2]);
            let y2 = chiral.sin();
            // y3: Full 4D symmetric compositional sum-binding in (sa + sb)
            let y3 = (0.75 * sum[0] + 0.65 * sum[1]).cos()
                * (0.70 * sum[2] - 0.60 * sum[3]).cos();

            let idx = pairs.len();
            pair_lookup[a * n_entities + b] = idx;
            pairs.push(RelPair {
                a,
                b,
                reg: vec![y0, y1, y2, y3],
                cls: 0,
            });
        }
    }

    standardize_reg_channels(&mut pairs, reg_dim);
    // Latent relation class = dominant standardized symmetry channel.
    for p in &mut pairs {
        let mut best_c = 0usize;
        let mut best_v = p.reg[0];
        for c in 1..4 {
            if p.reg[c] > best_v {
                best_v = p.reg[c];
                best_c = c;
            }
        }
        p.cls = best_c;
    }

    let perm_order = build_balanced_pair_order(n_entities, &pairs, cls_dim, rng);
    let default_train_n = 220usize;

    // Build 120 unseen 2-hop compositional queries (a -> b -> c) where (a,b) and (b,c)
    // are in train and (a,c) is strictly in unseen test!
    let mut is_train = vec![false; pairs.len()];
    for &idx in &perm_order[..default_train_n] {
        is_train[idx] = true;
    }
    let mut two_hop_queries = Vec::new();
    for &ac_idx in &perm_order[default_train_n..] {
        let a = pairs[ac_idx].a;
        let c = pairs[ac_idx].b;
        for b in 0..n_entities {
            if b == a || b == c {
                continue;
            }
            let ab = pair_lookup[a * n_entities + b];
            let bc = pair_lookup[b * n_entities + c];
            if is_train[ab] && is_train[bc] {
                two_hop_queries.push(TwoHopQuery {
                    a,
                    b,
                    c,
                    pair_ac_idx: ac_idx,
                });
                break;
            }
        }
    }
    rng.shuffle(&mut two_hop_queries);
    two_hop_queries.truncate(120);

    DomainData {
        name: "domain_a_causal_ecosystem",
        obs_dim,
        reg_dim,
        cls_dim,
        n_entities,
        obs,
        pairs,
        pair_lookup,
        perm_order,
        default_train_n,
        two_hop_queries,
        node_tiers: Vec::new(),
        node_rings: Vec::new(),
    }
}

/// Domain B (Target 1 — Continuous Physical Oscillators in R^12):
/// N_B = 18 coupled nonlinear Stuart-Landau / Kuramoto-Sakaguchi oscillators.
/// Observed via raw 4-step physical trajectory samples + 4 kinematic diagnostics in R^12.
/// Totally different modality, dimensionality (12 vs 20), and polar physics from Domain A!
pub fn generate_domain_b(rng: &mut Rng64) -> DomainData {
    let n_entities = 18usize;
    let obs_dim = 12usize;
    let reg_dim = 3usize;
    let cls_dim = 4usize;

    // Intrinsic physical state per oscillator: (theta, omega, amp_r, shear_eta)
    let mut theta = Vec::with_capacity(n_entities);
    let mut omega = Vec::with_capacity(n_entities);
    let mut amp = Vec::with_capacity(n_entities);
    let mut shear = Vec::with_capacity(n_entities);
    for _ in 0..n_entities {
        theta.push(rng.uniform_range(-PI, PI));
        omega.push(rng.uniform_range(0.70, 2.10));
        amp.push(rng.uniform_range(0.55, 1.45));
        shear.push(rng.uniform_range(-1.0, 1.0));
    }

    // Physical sensory trajectory observation in R^12:
    // 4 time steps t = 0.00, 0.30, 0.60, 0.90 -> (Re z(t), Im z(t)) = 8 coords
    // + 4 kinematic/spectral diagnostics: (speed, radial_energy, phase_accel, anharmonic_skew)
    let times = [0.0, 0.30, 0.60, 0.90];
    let mut obs = Vec::with_capacity(n_entities);
    for i in 0..n_entities {
        let mut x = vec![0.0; obs_dim];
        for (ti, &t) in times.iter().enumerate() {
            let ph = theta[i] + omega[i] * t + 0.16 * shear[i] * t * t;
            let r_t = amp[i] * (1.0 - 0.05 * shear[i] * t).max(0.2);
            x[2 * ti] = r_t * ph.cos() + rng.normal() * 0.025;
            x[2 * ti + 1] = r_t * ph.sin() + rng.normal() * 0.025;
        }
        // Kinematic observables directly from oscillator physics:
        let speed = amp[i] * omega[i];
        let energy = amp[i] * amp[i];
        let accel = amp[i] * (omega[i] * omega[i] + 0.32 * shear[i]);
        let skew = shear[i] * (theta[i] * 0.5).cos() + 0.3 * omega[i];
        x[8] = speed + rng.normal() * 0.025;
        x[9] = energy + rng.normal() * 0.025;
        x[10] = accel + rng.normal() * 0.025;
        x[11] = skew + rng.normal() * 0.025;
        obs.push(x);
    }
    standardize_obs(&mut obs);

    // Coupled physical interaction between oscillator i and oscillator j:
    //   f0: diffusive synchronization coherence (symmetric in i, j)
    //   f1: directed Kuramoto-Sakaguchi energy drive (i -> j, strictly antisymmetric)
    //   f2: frustrated nonlinear beat amplitude (depends on sum-phase & sum-frequency)
    let mut pairs = Vec::with_capacity(n_entities * (n_entities - 1));
    let mut pair_lookup = vec![usize::MAX; n_entities * n_entities];
    for i in 0..n_entities {
        for j in 0..n_entities {
            if i == j {
                continue;
            }
            let d_theta = theta[j] - theta[i];
            let d_omega = omega[j] - omega[i];
            let d_amp = amp[j] - amp[i];
            let d_shear = shear[j] - shear[i];

            let detune_env = (-0.55 * d_omega * d_omega).exp();
            let f0 = d_theta.cos() * detune_env * (amp[i] * amp[j]) / (1.0 + 0.30 * d_amp * d_amp);
            let f1 = d_theta.sin() * (-0.35 * d_omega * d_omega).exp() + 0.45 * d_shear.tanh();
            let f2 = (theta[i] + theta[j] + 0.55 * (shear[i] + shear[j])).cos()
                * (1.15 * (omega[i] + omega[j])).sin();

            let idx = pairs.len();
            pair_lookup[i * n_entities + j] = idx;
            pairs.push(RelPair {
                a: i,
                b: j,
                reg: vec![f0, f1, f2],
                cls: 0,
            });
        }
    }

    standardize_reg_channels(&mut pairs, reg_dim);

    // Calibrate uncoupled threshold tau_0 as the 25th percentile of max(f0, f1, f2)
    // so class 0 (uncoupled) has ~25% share and classes 1,2,3 split the strong-coupling regimes.
    let mut max_vals: Vec<f64> = pairs
        .iter()
        .map(|p| p.reg[0].max(p.reg[1]).max(p.reg[2]))
        .collect();
    max_vals.sort_by(|a, b| a.total_cmp(b));
    let tau0 = max_vals[max_vals.len() / 4];

    for p in &mut pairs {
        let m = p.reg[0].max(p.reg[1]).max(p.reg[2]);
        if m < tau0 {
            p.cls = 0; // Uncoupled / weak drift
        } else if p.reg[0] >= p.reg[1] && p.reg[0] >= p.reg[2] {
            p.cls = 1; // Diffusive in-phase synchronization
        } else if p.reg[1] >= p.reg[2] {
            p.cls = 2; // Directed Kuramoto-Sakaguchi drive i -> j
        } else {
            p.cls = 3; // Frustrated resonant beating
        }
    }

    let perm_order = build_balanced_pair_order(n_entities, &pairs, cls_dim, rng);

    DomainData {
        name: "domain_b_coupled_oscillators",
        obs_dim,
        reg_dim,
        cls_dim,
        n_entities,
        obs,
        pairs,
        pair_lookup,
        perm_order,
        default_train_n: 48,
        two_hop_queries: Vec::new(),
        node_tiers: Vec::new(),
        node_rings: Vec::new(),
    }
}

/// Domain C (Target 2 — Discrete Relational Graph & Algebraic Structure in R^16):
/// N_C = 20 symbolic nodes = Z_5 (cyclic ring) x {0, 1, 2, 3} (4-level strict hierarchy).
/// Observed only via opaque noisy embeddings in R^16 (zero discrete IDs provided).
/// Exact 4 discrete algebraic relations on 380 ordered pairs:
///   0: Same-tier equivalence (Delta_tier == 0 and Delta_ring != 1, 60 pairs, symmetric)
///   1: Strict dominance tier(u) < tier(v) (Delta_tier > 0 and Delta_ring != 1, 120 pairs, antisymmetric & transitive)
///   2: Cyclic successor (ring(v) - ring(u)) mod 5 == 1 (80 pairs, directed non-transitive)
///   3: Strict subordination tier(u) > tier(v) (Delta_tier < 0 and Delta_ring != 1, 120 pairs, inverse of 1)
pub fn generate_domain_c(rng: &mut Rng64) -> DomainData {
    let n_tiers = 4usize;
    let n_rings = 5usize;
    let n_entities = n_tiers * n_rings; // 20
    let obs_dim = 16usize;
    let reg_dim = 0usize; // Pure discrete relational classification domain!
    let cls_dim = 4usize;

    let mut node_tiers = Vec::with_capacity(n_entities);
    let mut node_rings = Vec::with_capacity(n_entities);
    let mut geom = Vec::with_capacity(n_entities);

    // Shuffle node IDs so entity index gives zero clue about (tier, ring).
    let mut coords = Vec::with_capacity(n_entities);
    for t in 0..n_tiers {
        for r in 0..n_rings {
            coords.push((t, r));
        }
    }
    rng.shuffle(&mut coords);

    for &(t, r) in &coords {
        node_tiers.push(t);
        node_rings.push(r);
        let t_norm = (t as f64 - 1.5) * 0.85;
        let angle = 2.0 * PI * (r as f64) / (n_rings as f64);
        let aux = ((t as f64) * 0.55 + angle).sin() * 0.35;
        geom.push([t_norm, angle.cos() * 1.15, angle.sin() * 1.15, aux]);
    }

    // Opaque nonlinear random mixing into R^16 + measurement noise.
    let mut m_c = vec![vec![0.0; 4]; obs_dim];
    let mut b_c = vec![0.0; obs_dim];
    for row in 0..obs_dim {
        for col in 0..4 {
            m_c[row][col] = rng.normal() * 0.70;
        }
        b_c[row] = rng.normal() * 0.20;
    }
    let mut obs = Vec::with_capacity(n_entities);
    for g in &geom {
        let mut x = vec![0.0; obs_dim];
        for row in 0..obs_dim {
            let mut acc = b_c[row];
            for col in 0..4 {
                acc += m_c[row][col] * g[col];
            }
            x[row] = acc.tanh() + rng.normal() * 0.025;
        }
        obs.push(x);
    }
    standardize_obs(&mut obs);

    let mut pairs = Vec::with_capacity(n_entities * (n_entities - 1));
    let mut pair_lookup = vec![usize::MAX; n_entities * n_entities];
    for u in 0..n_entities {
        for v in 0..n_entities {
            if u == v {
                continue;
            }
            let du_t = (node_tiers[v] as i32) - (node_tiers[u] as i32);
            let du_r = ((node_rings[v] + n_rings - node_rings[u]) % n_rings) as usize;
            let cls = if du_r == 1 {
                2 // Directed cyclic successor in Z_5
            } else if du_t == 0 {
                0 // Same-tier equivalence
            } else if du_t > 0 {
                1 // Strict hierarchical dominance u < v
            } else {
                3 // Strict subordination u > v
            };
            let idx = pairs.len();
            pair_lookup[u * n_entities + v] = idx;
            pairs.push(RelPair {
                a: u,
                b: v,
                reg: Vec::new(),
                cls,
            });
        }
    }

    let perm_order = build_balanced_pair_order(n_entities, &pairs, cls_dim, rng);

    DomainData {
        name: "domain_c_discrete_graph_algebra",
        obs_dim,
        reg_dim,
        cls_dim,
        n_entities,
        obs,
        pairs,
        pair_lookup,
        perm_order,
        default_train_n: 60,
        two_hop_queries: Vec::new(),
        node_tiers,
        node_rings,
    }
}
