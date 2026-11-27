//! Two genuinely independent benchmark domains for Prueba A:
//!   - Domain A (R^20): 4D-isotropic Causal & Relational Ecosystem (24 entities, 220 train / 332 test, 120 2-hop)
//!   - Domain C (R^16): Discrete Relational Graph Z_5 x {0..3} (20 symbolic nodes, 4 algebraic relations)

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
    #[allow(dead_code)]
    pub name: &'static str,
    pub obs_dim: usize,
    pub reg_dim: usize,
    pub cls_dim: usize,
    pub n_entities: usize,
    pub obs: Vec<Vec<f64>>,
    pub pairs: Vec<RelPair>,
    pub pair_lookup: Vec<usize>,
    pub perm_order: Vec<usize>,
    pub default_train_n: usize,
    pub two_hop_queries: Vec<TwoHopQuery>,
    pub node_tiers: Vec<usize>,
}

impl DomainData {
    pub fn train_indices(&self, n_train: usize) -> &[usize] {
        &self.perm_order[..n_train.min(self.perm_order.len())]
    }

    pub fn test_indices(&self, n_train: usize) -> &[usize] {
        let k = n_train.min(self.perm_order.len());
        &self.perm_order[k..]
    }

    pub fn with_sensory_noise(&self, sigma: f64, rng: &mut Rng64) -> Self {
        let mut copy = self.clone();
        for x in &mut copy.obs {
            for v in x.iter_mut() {
                *v += rng.normal() * sigma;
            }
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

fn build_balanced_pair_order(
    n_entities: usize,
    pairs: &[RelPair],
    cls_dim: usize,
    rng: &mut Rng64,
) -> Vec<usize> {
    let total = pairs.len();
    let mut used = vec![false; total];
    let mut order = Vec::with_capacity(total);

    let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); cls_dim];
    for (idx, p) in pairs.iter().enumerate() {
        buckets[p.cls].push(idx);
    }
    for b in &mut buckets {
        rng.shuffle(b);
    }

    let mut entity_deg = vec![0usize; n_entities];
    let mut cursors = vec![0usize; cls_dim];
    let mut step = 0usize;
    while order.len() < total {
        let mut added_any = false;
        for c_off in 0..cls_dim {
            let c = (step + c_off) % cls_dim;
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

pub fn generate_domain_a(rng: &mut Rng64) -> DomainData {
    let n_entities = 24usize;
    let obs_dim = 20usize;
    let reg_dim = 4usize;
    let cls_dim = 4usize;

    let mut latent = Vec::with_capacity(n_entities);
    for _ in 0..n_entities {
        let mut s = [0.0; 4];
        for v in &mut s {
            *v = rng.uniform_range(-1.45, 1.45);
        }
        latent.push(s);
    }

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
            x[r] = acc.tanh() + 0.15 * (0.8 * acc).sin() + rng.normal() * 0.03;
        }
        obs.push(x);
    }
    standardize_obs(&mut obs);

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

            let y0 = (1.05 * dist).cos() * (-0.16 * d2).exp();
            let y1 = (0.95 * dot_d1).tanh() * (-0.08 * d2).exp() + 0.45 * (0.85 * dot_d2).sin();
            let chiral = 0.80 * (sa[0] * sb[1] - sa[1] * sb[0])
                + 0.80 * (sa[2] * sb[3] - sa[3] * sb[2]);
            let y2 = chiral.sin();
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
    }
}

pub fn generate_domain_c(rng: &mut Rng64) -> DomainData {
    let n_tiers = 4usize;
    let n_rings = 5usize;
    let n_entities = n_tiers * n_rings;
    let obs_dim = 16usize;
    let reg_dim = 0usize;
    let cls_dim = 4usize;

    let mut node_tiers = Vec::with_capacity(n_entities);
    let mut node_rings = Vec::with_capacity(n_entities);
    let mut geom = Vec::with_capacity(n_entities);

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
                2
            } else if du_t == 0 {
                0
            } else if du_t > 0 {
                1
            } else {
                3
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
        default_train_n: 120,
        two_hop_queries: Vec::new(),
        node_tiers,
    }
}
