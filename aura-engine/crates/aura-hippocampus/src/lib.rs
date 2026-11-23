//! `aura-hippocampus`:
//! - Navigable Small-World L1 Graph (`HippocampusGraph`, HNSW-style M=16, ef=32)
//!   supporting one-shot episodic recall across 5,000+ byte horizons (Bottleneck 4)
//!   and deterministic binary persistence.
//! - `ScaledTernaryMlp` (2-layer 100% 2-bit BitNet b1.58 zero-mul classifier)
//!   and `SleepConsolidator` v2 with hippocampal replay + importance-weighted hysteresis
//!   solving Bottleneck 3's 62.6% capacity ceiling & catastrophic forgetting.

use aura_math::{LatentVec, ScaledTernaryLayer, TernaryTrainConfig, Xorshift64, LATENT_DIM};
use std::io::{Read, Write};

#[derive(Clone, Debug)]
pub struct Episode {
    pub key: LatentVec,
    pub action_id: u16,
    pub next_latent: LatentVec,
    pub surprise: i32,
    pub reward_signal: i16,
    pub timestamp: u64,
    pub payload_id: u16,
    pub salience: u16,
}

pub trait EpisodicHippocampus {
    fn recall_similar(&self, query: &LatentVec, top_k: usize) -> Vec<&Episode>;
    fn store_if_salient(&mut self, ep: Episode) -> Option<usize>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Navigable Small-World L1 Graph (HNSW-style greedy beam search in L1 Laplace metric).
#[derive(Clone, Debug)]
pub struct HippocampusGraph {
    pub episodes: Vec<Episode>,
    pub neighbors: Vec<Vec<usize>>,
    pub max_neighbors: usize, // M = 16
    pub ef_search: usize,     // ef = 32
    pub max_capacity: usize,
    pub salience_threshold: i32,
    entry_point: usize,
}

impl HippocampusGraph {
    pub fn new(max_capacity: usize, salience_threshold: i32) -> Self {
        Self {
            episodes: Vec::with_capacity(max_capacity.min(1024)),
            neighbors: Vec::with_capacity(max_capacity.min(1024)),
            max_neighbors: 16,
            ef_search: 32,
            max_capacity: max_capacity.max(8),
            salience_threshold,
            entry_point: 0,
        }
    }

    pub fn clear(&mut self) {
        self.episodes.clear();
        self.neighbors.clear();
        self.entry_point = 0;
    }

    /// Retroactively attach delayed dopamine/reward to a previously stored episodic binding
    /// (Bottleneck 4: 5,000-byte delayed credit assignment without BPTT).
    pub fn apply_delayed_reward(&mut self, ep_idx: usize, reward: i16) {
        if let Some(ep) = self.episodes.get_mut(ep_idx) {
            ep.reward_signal = ep.reward_signal.saturating_add(reward);
            if reward > 0 {
                ep.salience = ep.salience.saturating_add(reward.unsigned_abs() * 4);
            }
        }
    }

    /// Search graph for `ef` nearest L1 neighbors returning `(dist, idx)` sorted ascending.
    pub fn search_indices(&self, query: &LatentVec, ef: usize) -> Vec<(i32, usize)> {
        let n = self.episodes.len();
        if n == 0 {
            return Vec::new();
        }
        let ef = ef.max(1);
        // For tiny graphs (<= 48 nodes), exact scan is faster than graph bookkeeping
        if n <= 48 {
            let mut all: Vec<(i32, usize)> = self
                .episodes
                .iter()
                .enumerate()
                .map(|(i, ep)| (ep.key.l1_distance(query), i))
                .collect();
            all.sort_unstable_by_key(|&(d, i)| (d, i));
            all.truncate(ef);
            return all;
        }

        let mut visited = vec![false; n];
        let ep0 = self.entry_point.min(n - 1);
        let d0 = self.episodes[ep0].key.l1_distance(query);
        visited[ep0] = true;

        // Multi-seed entry: start from entry_point plus 3 evenly spaced landmarks for recall >= 95%
        let mut candidates: Vec<(i32, usize)> = vec![(d0, ep0)];
        let mut best: Vec<(i32, usize)> = vec![(d0, ep0)];
        for step in 1..4usize {
            let seed_idx = (ep0 + (n * step) / 4) % n;
            if !visited[seed_idx] {
                visited[seed_idx] = true;
                let ds = self.episodes[seed_idx].key.l1_distance(query);
                candidates.push((ds, seed_idx));
                best.push((ds, seed_idx));
            }
        }
        best.sort_unstable_by_key(|&(d, i)| (d, i));

        while !candidates.is_empty() {
            // Pop closest candidate
            let mut min_pos = 0usize;
            for i in 1..candidates.len() {
                if candidates[i].0 < candidates[min_pos].0 {
                    min_pos = i;
                }
            }
            let (curr_d, curr_idx) = candidates.swap_remove(min_pos);
            let worst_best = best.last().map(|x| x.0).unwrap_or(i32::MAX);
            if curr_d > worst_best && best.len() >= ef {
                break;
            }

            for &nb in &self.neighbors[curr_idx] {
                if nb >= n || visited[nb] {
                    continue;
                }
                visited[nb] = true;
                let nd = self.episodes[nb].key.l1_distance(query);
                let worst = best.last().map(|x| x.0).unwrap_or(i32::MAX);
                if best.len() < ef || nd < worst {
                    candidates.push((nd, nb));
                    let pos = best.partition_point(|&(d, i)| (d, i) < (nd, nb));
                    best.insert(pos, (nd, nb));
                    if best.len() > ef {
                        best.pop();
                    }
                }
            }
        }
        best
    }

    pub fn recall_with_indices(&self, query: &LatentVec, top_k: usize) -> Vec<(usize, i32, &Episode)> {
        let res = self.search_indices(query, self.ef_search.max(top_k));
        res.into_iter()
            .take(top_k)
            .map(|(d, idx)| (idx, d, &self.episodes[idx]))
            .collect()
    }

    pub fn insert(&mut self, ep: Episode) -> usize {
        if self.episodes.len() >= self.max_capacity {
            // Evict lowest-salience / oldest slot and overwrite in-place
            let mut victim = 0usize;
            for i in 1..self.episodes.len() {
                let a = (self.episodes[i].salience, self.episodes[i].timestamp);
                let b = (self.episodes[victim].salience, self.episodes[victim].timestamp);
                if a < b {
                    victim = i;
                }
            }
            self.episodes[victim] = ep;
            let nbs = self.search_indices(&self.episodes[victim].key, self.max_neighbors + 1);
            let mut new_links = Vec::with_capacity(self.max_neighbors);
            for (_, nb) in nbs {
                if nb != victim && new_links.len() < self.max_neighbors {
                    new_links.push(nb);
                }
            }
            self.neighbors[victim] = new_links;
            return victim;
        }

        let new_idx = self.episodes.len();
        let nearest = if new_idx > 0 {
            self.search_indices(&ep.key, self.ef_search.max(self.max_neighbors))
        } else {
            Vec::new()
        };

        self.episodes.push(ep);
        let mut links = Vec::with_capacity(self.max_neighbors);
        for (_, nb) in nearest.into_iter().take(self.max_neighbors) {
            links.push(nb);
        }
        self.neighbors.push(links.clone());

        for nb in links {
            let nb_list = &mut self.neighbors[nb];
            if !nb_list.contains(&new_idx) {
                nb_list.push(new_idx);
                if nb_list.len() > self.max_neighbors + 4 {
                    let nb_key = self.episodes[nb].key;
                    nb_list.sort_unstable_by_key(|&id| self.episodes[id].key.l1_distance(&nb_key));
                    nb_list.truncate(self.max_neighbors);
                }
            }
        }
        if new_idx == 0 {
            self.entry_point = 0;
        }
        new_idx
    }

    /// Salience-weighted random sampling for offline Sleep Replay.
    pub fn sample_replay(&self, count: usize, rng: &mut Xorshift64) -> Vec<usize> {
        let n = self.episodes.len();
        if n == 0 || count == 0 {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            // Tournament of 3 weighted by salience (O(1), deterministic)
            let i0 = rng.next_usize(n);
            let i1 = rng.next_usize(n);
            let i2 = rng.next_usize(n);
            let mut best = i0;
            if self.episodes[i1].salience > self.episodes[best].salience {
                best = i1;
            }
            if self.episodes[i2].salience > self.episodes[best].salience {
                best = i2;
            }
            out.push(best);
        }
        out
    }

    /// Compact deterministic binary serialization (disk persistence / mmap-ready layout).
    pub fn serialize_binary<W: Write>(&self, mut w: W) -> std::io::Result<()> {
        w.write_all(b"AURAHIP2")?;
        w.write_all(&(self.episodes.len() as u32).to_le_bytes())?;
        for ep in &self.episodes {
            let key_u8: [u8; LATENT_DIM] = std::array::from_fn(|i| ep.key.data[i] as u8);
            let nxt_u8: [u8; LATENT_DIM] = std::array::from_fn(|i| ep.next_latent.data[i] as u8);
            w.write_all(&key_u8)?;
            w.write_all(&nxt_u8)?;
            w.write_all(&ep.action_id.to_le_bytes())?;
            w.write_all(&ep.surprise.to_le_bytes())?;
            w.write_all(&ep.reward_signal.to_le_bytes())?;
            w.write_all(&ep.timestamp.to_le_bytes())?;
            w.write_all(&ep.payload_id.to_le_bytes())?;
            w.write_all(&ep.salience.to_le_bytes())?;
        }
        Ok(())
    }

    pub fn deserialize_binary<R: Read>(mut r: R, max_capacity: usize, thresh: i32) -> std::io::Result<Self> {
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic)?;
        if &magic != b"AURAHIP2" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "invalid AURAHIP2 header",
            ));
        }
        let mut len_buf = [0u8; 4];
        r.read_exact(&mut len_buf)?;
        let count = u32::from_le_bytes(len_buf) as usize;
        let mut g = Self::new(max_capacity.max(count), thresh);
        for _ in 0..count {
            let mut key_u8 = [0u8; LATENT_DIM];
            let mut nxt_u8 = [0u8; LATENT_DIM];
            r.read_exact(&mut key_u8)?;
            r.read_exact(&mut nxt_u8)?;
            let mut b2 = [0u8; 2];
            let mut b4 = [0u8; 4];
            let mut b8 = [0u8; 8];
            r.read_exact(&mut b2)?;
            let action_id = u16::from_le_bytes(b2);
            r.read_exact(&mut b4)?;
            let surprise = i32::from_le_bytes(b4);
            r.read_exact(&mut b2)?;
            let reward_signal = i16::from_le_bytes(b2);
            r.read_exact(&mut b8)?;
            let timestamp = u64::from_le_bytes(b8);
            r.read_exact(&mut b2)?;
            let payload_id = u16::from_le_bytes(b2);
            r.read_exact(&mut b2)?;
            let salience = u16::from_le_bytes(b2);

            g.insert(Episode {
                key: LatentVec {
                    data: std::array::from_fn(|i| key_u8[i] as i8),
                },
                action_id,
                next_latent: LatentVec {
                    data: std::array::from_fn(|i| nxt_u8[i] as i8),
                },
                surprise,
                reward_signal,
                timestamp,
                payload_id,
                salience,
            });
        }
        Ok(g)
    }
}

impl EpisodicHippocampus for HippocampusGraph {
    fn recall_similar(&self, query: &LatentVec, top_k: usize) -> Vec<&Episode> {
        self.search_indices(query, self.ef_search.max(top_k))
            .into_iter()
            .take(top_k)
            .map(|(_, idx)| &self.episodes[idx])
            .collect()
    }

    fn store_if_salient(&mut self, ep: Episode) -> Option<usize> {
        if ep.surprise >= self.salience_threshold || ep.reward_signal != 0 || ep.salience > 0 {
            Some(self.insert(ep))
        } else {
            None
        }
    }

    fn len(&self) -> usize {
        self.episodes.len()
    }
}

// ============================================================================
// BOTTLENECK 3 FIX: 2-Layer Scaled Ternary MLP + SleepConsolidator v2
// ============================================================================

/// 2-layer 100% 2-bit Scaled Ternary Network (`in_dim -> hidden_dim -> out_dim`).
/// Both weight matrices operate strictly on 2-bit bit-planes (`project_lut4_simd`)
/// with 1 scalar per-row scale `alpha_r` and STE + AdamW + importance hysteresis training.
#[derive(Clone, Debug)]
pub struct ScaledTernaryMlp {
    pub l1: ScaledTernaryLayer,
    pub l2: ScaledTernaryLayer,
    h_pre: Vec<f32>,
    h_act: Vec<f32>,
    dh: Vec<f32>,
}

impl ScaledTernaryMlp {
    pub fn new(in_dim: usize, hidden_dim: usize, out_dim: usize, seed: u64) -> Self {
        let mut rng = Xorshift64::new(seed ^ 0x534C_4545_5000_0003);
        Self {
            l1: ScaledTernaryLayer::new(hidden_dim, in_dim, &mut rng),
            l2: ScaledTernaryLayer::new(out_dim, hidden_dim, &mut rng),
            h_pre: vec![0.0; hidden_dim],
            h_act: vec![0.0; hidden_dim],
            dh: vec![0.0; hidden_dim],
        }
    }

    pub fn forward(&mut self, x: &[f32], out: &mut [f32]) {
        self.l1.forward(x, &mut self.h_pre);
        for i in 0..self.h_pre.len() {
            let v = self.h_pre[i];
            self.h_act[i] = if v > 0.0 { v } else { 0.1 * v };
        }
        self.l2.forward(&self.h_act, out);
    }

    /// Accumulate STE gradients for one `(x, targets_pm1)` pair using smooth margin hinge/BCE loss.
    pub fn backward_binary_targets(&mut self, x: &[f32], targets_pm1: &[f32], weight: f32) {
        let out_dim = self.l2.rows;
        let mut logits = vec![0.0f32; out_dim];
        self.forward(x, &mut logits);

        let mut dy2 = vec![0.0f32; out_dim];
        for r in 0..out_dim {
            let y = targets_pm1[r];
            let z = (-y * logits[r]).clamp(-15.0, 15.0);
            let p_err = 1.0 / (1.0 + (-z).exp());
            dy2[r] = -y * p_err * weight;
        }

        self.dh.fill(0.0);
        self.l2
            .backward_accumulate(&self.h_act, &dy2, Some(&mut self.dh));
        for i in 0..self.dh.len() {
            if self.h_pre[i] <= 0.0 {
                self.dh[i] *= 0.1;
            }
        }
        self.l1.backward_accumulate(x, &self.dh, None);
    }

    pub fn step_adamw(&mut self, batch_size: usize, cfg: &TernaryTrainConfig) {
        self.l1.step_adamw_ste(batch_size, cfg);
        self.l2.step_adamw_ste(batch_size, cfg);
    }

    pub fn consolidate_task_anchor(&mut self, samples: &[(Vec<f32>, Vec<f32>)]) {
        self.l1.importance.fill(0.0);
        self.l2.importance.fill(0.0);
        let n = samples.len().max(1);
        for (x, y) in samples {
            self.l1.zero_grad();
            self.l2.zero_grad();
            self.backward_binary_targets(x, y, 1.0);
            self.l1.accumulate_importance(1.0 / (n as f32));
            self.l2.accumulate_importance(1.0 / (n as f32));
        }
        self.l1.zero_grad();
        self.l2.zero_grad();
        self.l1.snapshot_anchor_and_normalize_importance();
        self.l2.snapshot_anchor_and_normalize_importance();
    }

    pub fn total_flips(&self) -> u64 {
        self.l1.flip_count + self.l2.flip_count
    }

    pub fn reset_flip_count(&mut self) {
        self.l1.flip_count = 0;
        self.l2.flip_count = 0;
    }

    pub fn nonzero_fraction(&self) -> f64 {
        let p1 = (self.l1.rows * self.l1.cols) as f64;
        let p2 = (self.l2.rows * self.l2.cols) as f64;
        (self.l1.ternary.nonzero_fraction() * p1 + self.l2.ternary.nonzero_fraction() * p2)
            / (p1 + p2)
    }
}

/// Configuration for `SleepConsolidator` v2.
#[derive(Clone, Copy, Debug)]
pub struct SleepConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub use_replay: bool,
    pub replay_per_batch: usize,
    pub train_cfg: TernaryTrainConfig,
}

/// Offline Sleep Consolidator: replays compressed hippocampal episodes alongside new Task B
/// samples into 2-bit BitNet b1.58 `ScaledTernaryMlp` with importance-weighted hysteresis.
pub struct SleepConsolidator;

impl SleepConsolidator {
    pub fn consolidate_with_sleep(
        model: &mut ScaledTernaryMlp,
        new_samples: &[(Vec<f32>, Vec<f32>)],
        hippocampus: &HippocampusGraph,
        cfg: &SleepConfig,
        rng: &mut Xorshift64,
    ) {
        let n_new = new_samples.len();
        if n_new == 0 {
            return;
        }
        let in_dim = model.l1.cols;
        let out_dim = model.l2.rows;
        let mut order: Vec<usize> = (0..n_new).collect();

        for epoch in 0..cfg.epochs {
            // Cosine LR schedule for clean BitNet STE quantization settling
            let mut ep_tcfg = cfg.train_cfg;
            let progress = (epoch as f32) / (cfg.epochs.max(1) as f32);
            ep_tcfg.lr = cfg.train_cfg.lr
                * (0.18 + 0.82 * (0.5 * (1.0 + (progress * std::f32::consts::PI).cos())));

            // Fisher-Yates deterministic shuffle
            for i in (1..n_new).rev() {
                let j = rng.next_usize(i + 1);
                order.swap(i, j);
            }

            let mut idx = 0usize;
            while idx < n_new {
                let end = (idx + cfg.batch_size).min(n_new);
                let mut count = 0usize;

                for &si in &order[idx..end] {
                    let (ref x, ref y) = new_samples[si];
                    model.backward_binary_targets(x, y, 1.0);
                    count += 1;
                }

                if cfg.use_replay && !hippocampus.is_empty() {
                    let replay_ids = hippocampus.sample_replay(cfg.replay_per_batch, rng);
                    for rid in replay_ids {
                        let ep = &hippocampus.episodes[rid];
                        let x_rep: Vec<f32> = ep.key.data[..in_dim]
                            .iter()
                            .map(|&v| (v as f32) / 64.0)
                            .collect();
                        let y_rep: Vec<f32> = ep.next_latent.data[..out_dim]
                            .iter()
                            .map(|&v| if v >= 0 { 1.0 } else { -1.0 })
                            .collect();
                        model.backward_binary_targets(&x_rep, &y_rep, 1.0);
                        count += 1;
                    }
                }

                model.step_adamw(count, &ep_tcfg);
                idx = end;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hippocampus_hnsw_high_recall_and_binary_roundtrip() {
        let mut rng = Xorshift64::new(0xABCD_1234);
        let mut g = HippocampusGraph::new(512, 0);
        let mut keys = Vec::new();
        for i in 0..200 {
            let mut k = LatentVec::zero();
            for d in 0..LATENT_DIM {
                k.data[d] = ((rng.next_u32() % 200) as i32 - 100) as i8;
            }
            keys.push(k);
            g.insert(Episode {
                key: k,
                action_id: (i % 8) as u16,
                next_latent: k,
                surprise: 500,
                reward_signal: 1,
                timestamp: i as u64,
                payload_id: i as u16,
                salience: 10,
            });
        }

        // Exact-match noisy query recall across 30 probes
        let mut hits = 0usize;
        for q_idx in (0..200).step_by(6).take(30) {
            let mut q = keys[q_idx];
            for d in 0..16 {
                q.data[d] = q.data[d].saturating_add(3);
            }
            let top = g.recall_similar(&q, 1);
            if top[0].payload_id as usize == q_idx {
                hits += 1;
            }
        }
        assert!(hits >= 28, "Expected >=28/30 HNSW recall, got {hits}/30");

        // Binary roundtrip test
        let mut buf = Vec::new();
        g.serialize_binary(&mut buf).unwrap();
        let g2 = HippocampusGraph::deserialize_binary(buf.as_slice(), 512, 0).unwrap();
        assert_eq!(g2.len(), g.len());
        assert_eq!(g2.episodes[42].key, g.episodes[42].key);
        assert_eq!(g2.episodes[42].payload_id, g.episodes[42].payload_id);
    }
}
