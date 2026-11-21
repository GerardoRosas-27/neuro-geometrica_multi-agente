//! `aura-hippocampus`: Memoria episódica indexada mediante grafo navegable disperso L1
//! (estilo HNSW sin dependencias externas), serialización binaria determinista y
//! motor de consolidación en Ciclo de Sueño (`SleepConsolidator`) hacia planos de bits
//! ternarios con histéresis (`TernaryMatrix`).

use aura_math::{
    LatentVec, TernaryMatrix, Xorshift64, DEFAULT_HYSTERESIS_HI, DEFAULT_HYSTERESIS_LO,
    DEFAULT_SHADOW_CAP, LATENT_DIM,
};

const GRAPH_M: usize = 16;
const GRAPH_EF: usize = 32;

/// Registro episódico para el hipocampo (fiel a la propuesta AURA §4).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Episode {
    pub key: LatentVec,
    pub action_id: u16,
    pub next_latent: LatentVec,
    pub surprise: i32,
    pub reward_signal: i16,
    pub task_tag: u8,
    pub replay_count: u32,
}

/// Contrato del subsistema hipocampal (fiel a la propuesta AURA §4).
pub trait EpisodicHippocampus {
    fn recall_similar(&self, query: &LatentVec, top_k: usize) -> Vec<&Episode>;
    fn store_if_salient(&mut self, ep: Episode);
}

/// Índice episódico navegable por distancia L1 (Energía Libre de Laplace, 0 multiplicaciones).
#[derive(Clone, Debug)]
pub struct HippocampusGraph {
    pub episodes: Vec<Episode>,
    neighbors: Vec<Vec<usize>>,
    pub max_capacity: usize,
    pub salience_threshold: i32,
    pub dedup_l1_threshold: i32,
    entry: Option<usize>,
}

impl HippocampusGraph {
    pub fn new(max_capacity: usize, salience_threshold: i32, dedup_l1_threshold: i32) -> Self {
        Self {
            episodes: Vec::with_capacity(max_capacity.min(1024)),
            neighbors: Vec::with_capacity(max_capacity.min(1024)),
            max_capacity: max_capacity.max(8),
            salience_threshold,
            dedup_l1_threshold,
            entry: None,
        }
    }

    pub fn len(&self) -> usize {
        self.episodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.episodes.is_empty()
    }

    /// Búsqueda greedy beam sobre el grafo navegable L1 (devuelve índices ordenados por distancia).
    fn search_indices(&self, query: &LatentVec, ef: usize) -> Vec<(i32, usize)> {
        let n = self.episodes.len();
        if n == 0 {
            return Vec::new();
        }
        let mut visited = vec![false; n];
        let mut candidates: Vec<(i32, usize)> = Vec::with_capacity(ef * 2);

        // Semillas iniciales: punto de entrada + anclas espaciadas deterministas para evitar islas.
        let mut seeds = [0usize; 4];
        let mut n_seeds = 0usize;
        if let Some(e) = self.entry {
            seeds[n_seeds] = e;
            n_seeds += 1;
        }
        if n > 4 {
            seeds[n_seeds] = n / 3;
            n_seeds += 1;
            seeds[n_seeds] = (2 * n) / 3;
            n_seeds += 1;
            seeds[n_seeds] = n - 1;
            n_seeds += 1;
        }

        for &s in &seeds[..n_seeds] {
            if !visited[s] {
                visited[s] = true;
                let d = self.episodes[s].key.l1_distance(query);
                candidates.push((d, s));
            }
        }

        let mut expanded = vec![false; n];
        loop {
            candidates.sort_unstable_by_key(|&(d, idx)| (d, idx));
            if candidates.len() > ef {
                candidates.truncate(ef);
            }
            // Elegir el candidato no expandido más cercano.
            let next_to_expand = candidates
                .iter()
                .find(|&&(_, idx)| !expanded[idx])
                .copied();
            let Some((_, cur)) = next_to_expand else {
                break;
            };
            expanded[cur] = true;

            for &nb in &self.neighbors[cur] {
                if nb < n && !visited[nb] {
                    visited[nb] = true;
                    let d = self.episodes[nb].key.l1_distance(query);
                    candidates.push((d, nb));
                }
            }
        }

        candidates.sort_unstable_by_key(|&(d, idx)| (d, idx));
        candidates
    }

    fn connect_node(&mut self, new_idx: usize) {
        let query = self.episodes[new_idx].key;
        let found = self.search_indices(&query, GRAPH_EF);
        let mut nb_list = Vec::with_capacity(GRAPH_M);
        for &(_, idx) in found.iter() {
            if idx == new_idx {
                continue;
            }
            nb_list.push(idx);
            if nb_list.len() >= GRAPH_M {
                break;
            }
        }
        self.neighbors[new_idx] = nb_list.clone();

        for &nb in &nb_list {
            if !self.neighbors[nb].contains(&new_idx) {
                self.neighbors[nb].push(new_idx);
                if self.neighbors[nb].len() > GRAPH_M {
                    let nb_key = self.episodes[nb].key;
                    self.neighbors[nb].sort_unstable_by_key(|&other| {
                        (self.episodes[other].key.l1_distance(&nb_key), other)
                    });
                    self.neighbors[nb].truncate(GRAPH_M);
                }
            }
        }
        if self.entry.is_none() {
            self.entry = Some(new_idx);
        }
    }

    /// Inserta incondicionalmente un episodio (deduplicando solo si es casi idéntico en clave y acción).
    pub fn insert_episode(&mut self, ep: Episode) {
        if !self.episodes.is_empty() && self.dedup_l1_threshold > 0 {
            let nearest = self.search_indices(&ep.key, 8);
            for &(dist, idx) in nearest.iter().take(4) {
                if dist <= self.dedup_l1_threshold && self.episodes[idx].action_id == ep.action_id {
                    // Actualizar en sitio el recuerdo para reflejar la transición más reciente.
                    self.episodes[idx].next_latent = ep.next_latent;
                    self.episodes[idx].surprise = ep.surprise.max(self.episodes[idx].surprise);
                    self.episodes[idx].reward_signal = ep.reward_signal;
                    return;
                }
            }
        }

        if self.episodes.len() < self.max_capacity {
            let new_idx = self.episodes.len();
            self.episodes.push(ep);
            self.neighbors.push(Vec::with_capacity(GRAPH_M));
            self.connect_node(new_idx);
        } else {
            // Reemplazar la ranura con menor saliencia preservando la topología del grafo.
            let mut victim = 0usize;
            let mut min_sal = i32::MAX;
            for (i, e) in self.episodes.iter().enumerate() {
                if e.surprise < min_sal {
                    min_sal = e.surprise;
                    victim = i;
                }
            }
            self.episodes[victim] = ep;
            self.neighbors[victim].clear();
            self.connect_node(victim);
        }
    }

    /// Recuperación episódica condicionada opcionalmente por acción.
    pub fn recall_similar_for_action(
        &self,
        query: &LatentVec,
        action_id: u16,
        top_k: usize,
    ) -> Vec<&Episode> {
        let found = self.search_indices(query, GRAPH_EF.max(top_k * 4));
        let mut out = Vec::with_capacity(top_k);
        for &(_, idx) in &found {
            if self.episodes[idx].action_id == action_id {
                out.push(&self.episodes[idx]);
                if out.len() >= top_k {
                    break;
                }
            }
        }
        out
    }

    /// Serialización binaria determinista sin dependencias externas (persistencia).
    pub fn save_to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"AURA_HIP");
        buf.extend_from_slice(&(self.max_capacity as u32).to_le_bytes());
        buf.extend_from_slice(&self.salience_threshold.to_le_bytes());
        buf.extend_from_slice(&self.dedup_l1_threshold.to_le_bytes());
        buf.extend_from_slice(&(self.episodes.len() as u32).to_le_bytes());
        for ep in &self.episodes {
            for &b in &ep.key.data {
                buf.push(b as u8);
            }
            buf.extend_from_slice(&ep.action_id.to_le_bytes());
            for &b in &ep.next_latent.data {
                buf.push(b as u8);
            }
            buf.extend_from_slice(&ep.surprise.to_le_bytes());
            buf.extend_from_slice(&ep.reward_signal.to_le_bytes());
            buf.push(ep.task_tag);
            buf.extend_from_slice(&ep.replay_count.to_le_bytes());
        }
        buf
    }

    pub fn load_from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 24 || &bytes[0..8] != b"AURA_HIP" {
            return Err("invalid AURA_HIP header".to_string());
        }
        let max_capacity = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let salience_threshold = i32::from_le_bytes(bytes[12..16].try_into().unwrap());
        let dedup_l1_threshold = i32::from_le_bytes(bytes[16..20].try_into().unwrap());
        let count = u32::from_le_bytes(bytes[20..24].try_into().unwrap()) as usize;

        let ep_size = LATENT_DIM + 2 + LATENT_DIM + 4 + 2 + 1 + 4;
        if bytes.len() != 24 + count * ep_size {
            return Err("truncated AURA_HIP payload".to_string());
        }
        let mut graph = Self::new(max_capacity, salience_threshold, dedup_l1_threshold);
        // Al cargar desactivar deduplicación temporalmente para reconstruir 1-a-1 exacto.
        let orig_dedup = graph.dedup_l1_threshold;
        graph.dedup_l1_threshold = 0;

        let mut pos = 24;
        for _ in 0..count {
            let mut key = LatentVec::zero();
            for i in 0..LATENT_DIM {
                key.data[i] = bytes[pos + i] as i8;
            }
            pos += LATENT_DIM;
            let action_id = u16::from_le_bytes(bytes[pos..pos + 2].try_into().unwrap());
            pos += 2;
            let mut next_latent = LatentVec::zero();
            for i in 0..LATENT_DIM {
                next_latent.data[i] = bytes[pos + i] as i8;
            }
            pos += LATENT_DIM;
            let surprise = i32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
            pos += 4;
            let reward_signal = i16::from_le_bytes(bytes[pos..pos + 2].try_into().unwrap());
            pos += 2;
            let task_tag = bytes[pos];
            pos += 1;
            let replay_count = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
            pos += 4;

            graph.insert_episode(Episode {
                key,
                action_id,
                next_latent,
                surprise,
                reward_signal,
                task_tag,
                replay_count,
            });
        }
        graph.dedup_l1_threshold = orig_dedup;
        Ok(graph)
    }
}

impl EpisodicHippocampus for HippocampusGraph {
    fn recall_similar(&self, query: &LatentVec, top_k: usize) -> Vec<&Episode> {
        if top_k == 0 || self.episodes.is_empty() {
            return Vec::new();
        }
        let found = self.search_indices(query, GRAPH_EF.max(top_k * 2));
        found
            .into_iter()
            .take(top_k)
            .map(|(_, idx)| &self.episodes[idx])
            .collect()
    }

    fn store_if_salient(&mut self, ep: Episode) {
        if ep.surprise >= self.salience_threshold || ep.reward_signal != 0 {
            self.insert_episode(ep);
        }
    }
}

/// Configuración del Ciclo de Sueño (permite ablaciones exactas de replay e histéresis).
#[derive(Clone, Copy, Debug)]
pub struct SleepConfig {
    pub cycles: usize,
    pub use_replay: bool,
    pub replay_ratio_pct: u8, // % de pasos del ciclo que re-juegan episodios previos del hipocampo
    pub hysteresis_hi: i16,
    pub hysteresis_lo: i16,
    pub shadow_cap: i16,
    pub step_delta: i16,
    pub target_scale_shift: u8,
    pub key_active_thresh: i8,
    pub err_deadband: i32,
}

impl Default for SleepConfig {
    fn default() -> Self {
        Self {
            cycles: 800,
            use_replay: true,
            replay_ratio_pct: 50,
            hysteresis_hi: DEFAULT_HYSTERESIS_HI,
            hysteresis_lo: DEFAULT_HYSTERESIS_LO,
            shadow_cap: DEFAULT_SHADOW_CAP,
            step_delta: 20,
            target_scale_shift: 3, // escala x8 para alinear rango de project_zero_mul
            key_active_thresh: 16,
            err_deadband: 12,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SleepReport {
    pub cycles_executed: usize,
    pub old_replays: usize,
    pub new_replays: usize,
    pub weight_flips: usize,
    pub mean_abs_err_start: f64,
    pub mean_abs_err_end: f64,
}

/// Motor de consolidación en Ciclo de Sueño (Etapa 2):
/// destila episodios nuevos y recuerdos hipocampales hacia los planos de bits ternarios
/// `TernaryMatrix` con CERO multiplicaciones y acumuladores latentes con histéresis.
pub struct SleepConsolidator;

impl SleepConsolidator {
    pub fn consolidate_into_ternary(
        slow_mat: &mut TernaryMatrix,
        hippocampus: &mut HippocampusGraph,
        new_episodes: &[Episode],
        cfg: SleepConfig,
        rng: &mut Xorshift64,
    ) -> SleepReport {
        assert_eq!(slow_mat.cols(), LATENT_DIM);
        let rows = slow_mat.rows;
        let mut pred_buf = vec![0i32; rows];
        let mut old_replays = 0usize;
        let mut new_replays = 0usize;
        let mut weight_flips = 0usize;
        let mut err_start_sum = 0.0f64;
        let mut err_start_n = 0usize;
        let mut err_end_sum = 0.0f64;
        let mut err_end_n = 0usize;

        let n_hip = hippocampus.episodes.len();
        let n_new = new_episodes.len();
        if n_new == 0 && n_hip == 0 {
            return SleepReport {
                cycles_executed: 0,
                old_replays: 0,
                new_replays: 0,
                weight_flips: 0,
                mean_abs_err_start: 0.0,
                mean_abs_err_end: 0.0,
            };
        }

        let start_window = (cfg.cycles / 10).max(1);
        let end_window_start = cfg.cycles.saturating_sub(start_window);

        for step in 0..cfg.cycles {
            let pick_old = cfg.use_replay
                && n_hip > 0
                && (n_new == 0 || (rng.next_u64() % 100) < (cfg.replay_ratio_pct as u64));

            let (key, target) = if pick_old {
                // Selección priorizada por torneo (2 candidatos -> mayor sorpresa / menor replay_count).
                let a = rng.next_usize(n_hip);
                let b = rng.next_usize(n_hip);
                let score_a = hippocampus.episodes[a].surprise
                    - (hippocampus.episodes[a].replay_count as i32) * 8;
                let score_b = hippocampus.episodes[b].surprise
                    - (hippocampus.episodes[b].replay_count as i32) * 8;
                let idx = if score_a >= score_b { a } else { b };
                hippocampus.episodes[idx].replay_count += 1;
                old_replays += 1;
                (
                    hippocampus.episodes[idx].key,
                    hippocampus.episodes[idx].next_latent,
                )
            } else {
                let idx = rng.next_usize(n_new);
                new_replays += 1;
                (new_episodes[idx].key, new_episodes[idx].next_latent)
            };

            // Proyección 0-mul en planos de bits ternarios.
            slow_mat.project_zero_mul(&key.data, &mut pred_buf);

            // Homeostasis sináptica de sueño (Tononi & Cirelli): cada 8 ciclos aplica un
            // reescalado suave por desplazamiento de bits sobre los acumuladores latentes,
            // devolviendo a 0 las conexiones ruidosas no correlacionadas mientras las señales
            // consistentes se mantienen ancladas en la banda de histéresis.
            if step > 0 && (step & 7) == 0 && cfg.hysteresis_hi > 1 {
                let cols = slow_mat.cols();
                for r in 0..rows {
                    for c in 0..cols {
                        let idx = r * cols + c;
                        let cur = slow_mat.shadow_acc[idx];
                        if cur != 0 {
                            let dec = aura_math::decay_sym_i16(cur, 5);
                            let diff = dec - cur;
                            if diff != 0 {
                                slow_mat.consolidate_with_thresholds(
                                    r,
                                    c,
                                    diff,
                                    cfg.hysteresis_hi,
                                    cfg.hysteresis_lo,
                                    cfg.shadow_cap,
                                );
                            }
                        }
                    }
                }
            }

            let mut step_err_sum = 0i64;
            let margin = 192i32;
            for (r, &pred_r) in pred_buf.iter().enumerate() {
                let tgt_sign = (target.data[r] as i32).signum();
                let signed_margin = tgt_sign * pred_r;
                let hinge_err = (margin - signed_margin).max(0);
                step_err_sum += hinge_err as i64;
                if hinge_err < cfg.err_deadband {
                    continue;
                }
                let err_sign = tgt_sign as i16;
                for c in 0..LATENT_DIM {
                    let kc = key.data[c];
                    if kc.abs() < cfg.key_active_thresh {
                        continue;
                    }
                    let delta = if kc > 0 {
                        err_sign * cfg.step_delta
                    } else {
                        -err_sign * cfg.step_delta
                    };
                    let before = slow_mat.weight_at(r, c);
                    slow_mat.consolidate_with_thresholds(
                        r,
                        c,
                        delta,
                        cfg.hysteresis_hi,
                        cfg.hysteresis_lo,
                        cfg.shadow_cap,
                    );
                    if slow_mat.weight_at(r, c) != before {
                        weight_flips += 1;
                    }
                }
            }

            let step_err_mean = (step_err_sum as f64) / (rows as f64);
            if step < start_window {
                err_start_sum += step_err_mean;
                err_start_n += 1;
            }
            if step >= end_window_start {
                err_end_sum += step_err_mean;
                err_end_n += 1;
            }
        }

        SleepReport {
            cycles_executed: cfg.cycles,
            old_replays,
            new_replays,
            weight_flips,
            mean_abs_err_start: if err_start_n > 0 {
                err_start_sum / (err_start_n as f64)
            } else {
                0.0
            },
            mean_abs_err_end: if err_end_n > 0 {
                err_end_sum / (err_end_n as f64)
            } else {
                0.0
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rand_vec(rng: &mut Xorshift64) -> LatentVec {
        let mut v = LatentVec::zero();
        for x in &mut v.data {
            *x = (rng.next_signed_byte() >> 1).clamp(-127, 127) as i8;
        }
        v
    }

    #[test]
    fn navigable_graph_recalls_exact_matches_and_roundtrips_binary() {
        let mut rng = Xorshift64::from_seed(0x1234);
        let mut hip = HippocampusGraph::new(256, 0, 0);
        for i in 0..120 {
            let k = rand_vec(&mut rng);
            let n = rand_vec(&mut rng);
            hip.insert_episode(Episode {
                key: k,
                action_id: (i % 4) as u16,
                next_latent: n,
                surprise: 100 + i as i32,
                reward_signal: 1,
                task_tag: 0,
                replay_count: 0,
            });
        }

        // Probar recuperación exacta de vectores perturbados levemente.
        let mut hits = 0usize;
        for i in 0..30 {
            let mut q = hip.episodes[i * 4].key;
            q.data[0] = q.data[0].saturating_add(3);
            q.data[1] = q.data[1].saturating_sub(3);
            let rec = hip.recall_similar(&q, 3);
            if rec.iter().any(|e| e.key == hip.episodes[i * 4].key) {
                hits += 1;
            }
        }
        assert!(hits >= 28, "navigable graph recall too low: {hits}/30");

        // Verificar persistencia binaria bit a bit.
        let bytes = hip.save_to_bytes();
        let loaded = HippocampusGraph::load_from_bytes(&bytes).expect("load_from_bytes failed");
        assert_eq!(loaded.episodes, hip.episodes);
    }
}
