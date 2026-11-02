//! Circuito Trisináptico Hipocampal (EC -> Giro Dentado -> CA3 -> CA1) y Sistemas de
//! Aprendizaje Complementarios (CLS) con Replay de Ondas Agudas-Ripples (SWR).
//!
//! Fundamentación neurocientífica:
//! 1. **McClelland, McNaughton & O'Reilly (1995)**, *Psychological Review*, 102(3), 419-457;
//!    **Kumaran, Hassabis & McClelland (2016)**, *Trends in Cognitive Sciences*, 20(7), 512-534:
//!    El cerebro evita el olvido catastrófico dividiendo el aprendizaje en un sistema rápido y
//!    disperso en el **Hipocampo** (aprendizaje episódico *one-shot*) y un sistema lento y
//!    estructurado en la **Neocorteza**, acoplados mediante reproducción (*replay*) durante el sueño.
//! 2. **Rolls (2013)**, *Frontiers in Systems Neuroscience*, 7:74:
//!    - **Giro Dentado (DG)**: Expansión dimensional + inhibición competitiva extrema (*sparse k-WTA*)
//!      mediada por fibras musgosas para **Separación de Patrones** (*Pattern Separation*).
//!    - **CA3**: Red auto-asociativa y hetero-asociativa de atractores continuos (Hopfield moderno,
//!      **Krotov & Hopfield, 2016**; **Ramsauer et al., 2020**) para **Completado de Patrones**
//!      (*Pattern Completion*) ante entradas ocluidas o ruidosas.
//!    - **CA1 (Bucle Hipocampo-VTA de Lisman & Grace, 2005, *Neuron*)**: Compara el estado
//!      cortical actual con la recuperación de CA3 para detectar novedad e impulsar dopamina.
//! 3. **Buzsáki (2015)**, *Hippocampus*, 25(10), 1073-1188:
//!    Reproducción priorizada por **Sharp-Wave Ripples (SWR)** durante el sueño NREM.

#![allow(clippy::needless_range_loop)]

use crate::rng::Xoshiro256StarStar;

/// Traza episódica almacenada en el complejo CA3-CA1 del hipocampo.
#[derive(Clone, Debug)]
pub struct EpisodicEngram {
    /// Clave sensorial/entorrinal original ($x_{\text{EC}}$).
    pub ec_input: Vec<f64>,
    /// Código ortogonal ultra-disperso del Giro Dentado ($z_{\text{DG}}$).
    pub dg_sparse_code: Vec<f64>,
    /// Representación objetivo o estado siguiente asociado en CA1 ($y_{\text{target}}$).
    pub ca1_target: Vec<f64>,
    /// Saliencia neuromoduladora (Dopamina + Sorpresa en el momento de codificación).
    pub salience: f64,
    /// Contador de reproducciones en sueño SWR.
    pub replay_count: usize,
}

/// Sistema Hipocampal Trisináptico (EC -> DG -> CA3 -> CA1).
#[derive(Clone, Debug)]
pub struct HippocampalSystem {
    pub ec_dim: usize,
    pub dg_dim: usize,
    pub target_dim: usize,
    /// Proyección aleatoria fija de la vía perforante (Corteza Entorrinal -> Giro Dentado).
    pub w_perforant_ec_to_dg: Vec<f64>,
    /// Fracción activa en el Giro Dentado (típicamente 5% a 12% de células granulares).
    pub dg_sparsity: f64,
    /// Temperatura inversa ($\beta$) del atractor asociativo de CA3 (Ramsauer et al., 2020).
    pub ca3_inverse_temp: f64,
    /// Capacidad máxima del búfer episódico CA3.
    pub max_engrams: usize,
    /// Engramas episódicos almacenados en los colaterales recurrentes de CA3.
    pub ca3_engrams: Vec<EpisodicEngram>,
}

impl HippocampalSystem {
    pub fn new(
        ec_dim: usize,
        dg_dim: usize,
        target_dim: usize,
        dg_sparsity: f64,
        ca3_inverse_temp: f64,
        max_engrams: usize,
        seed: u64,
    ) -> Self {
        let mut rng = Xoshiro256StarStar::seed_from_u64(seed);
        let scale = 1.0 / (ec_dim.max(1) as f64).sqrt();
        let w_perforant_ec_to_dg = (0..dg_dim * ec_dim).map(|_| rng.randn() * scale).collect();

        Self {
            ec_dim,
            dg_dim,
            target_dim,
            w_perforant_ec_to_dg,
            dg_sparsity: dg_sparsity.clamp(0.04, 0.25),
            ca3_inverse_temp: ca3_inverse_temp.max(1.0),
            max_engrams: max_engrams.max(16),
            ca3_engrams: Vec::new(),
        }
    }

    /// **Separación de Patrones en el Giro Dentado (DG)** (Rolls, 2013):
    /// Expande el vector de la Corteza Entorrinal a una dimensión mayor (`dg_dim`) y aplica
    /// competición inhibitoria estricta *k-Winners-Take-All* (k-WTA) seguida de normalización L2.
    pub fn dentate_gyrus_pattern_separation(&self, ec_input: &[f64]) -> Vec<f64> {
        debug_assert_eq!(ec_input.len(), self.ec_dim);
        let mut raw = vec![0.0; self.dg_dim];
        for i in 0..self.dg_dim {
            let row = i * self.ec_dim;
            let mut s = 0.0;
            for j in 0..self.ec_dim {
                s += self.w_perforant_ec_to_dg[row + j] * ec_input[j];
            }
            raw[i] = s.max(0.0); // Rectificación de células granulares
        }

        // Selección competitiva k-WTA mediada por interneuronas hilares del Giro Dentado
        let k = ((self.dg_dim as f64 * self.dg_sparsity).round() as usize).clamp(1, self.dg_dim);
        let mut indices: Vec<usize> = (0..self.dg_dim).collect();
        indices.sort_by(|&a, &b| raw[b].total_cmp(&raw[a]));

        let mut sparse = vec![0.0; self.dg_dim];
        let mut norm_sq = 0.0;
        for &idx in indices.iter().take(k) {
            let val = raw[idx];
            sparse[idx] = val;
            norm_sq += val * val;
        }

        let norm = norm_sq.sqrt();
        if norm > 1e-9 {
            for v in &mut sparse {
                *v /= norm;
            }
        }
        sparse
    }

    /// **Completado de Patrones en CA3** (Rolls, 2013; Ramsauer et al., 2020):
    /// Dada una clave sensorial posiblemente ruidosa o incompleta (`ec_query`), recupera tanto
    /// el patrón sensorial reconstruido ($\hat{x}_{\text{CA3}}$) como la asociación objetivo en CA1
    /// ($\hat{y}_{\text{CA1}}$) y la señal de novedad de CA1 (Lisman & Grace, 2005).
    pub fn ca3_pattern_completion(&self, ec_query: &[f64]) -> (Vec<f64>, Vec<f64>, f64) {
        if self.ca3_engrams.is_empty() {
            return (
                ec_query.to_vec(),
                vec![0.0; self.target_dim],
                1.0, // Máxima novedad cuando la memoria episódica está vacía
            );
        }

        let q_dg = self.dentate_gyrus_pattern_separation(ec_query);
        let q_norm = l2_norm(ec_query).max(1e-8);

        // Similitud combinada entre colaterales de Schaffer (EC directo) y fibras musgosas (DG)
        let mut scores = Vec::with_capacity(self.ca3_engrams.len());
        let mut max_sim = f64::NEG_INFINITY;

        for engram in &self.ca3_engrams {
            let e_norm = l2_norm(&engram.ec_input).max(1e-8);
            let cos_ec = dot(ec_query, &engram.ec_input) / (q_norm * e_norm);
            let cos_dg = dot(&q_dg, &engram.dg_sparse_code);
            let sim = 0.65 * cos_ec + 0.35 * cos_dg;
            if sim > max_sim {
                max_sim = sim;
            }
            scores.push(self.ca3_inverse_temp * sim);
        }

        // Softmax de energía libre del atractor de Hopfield continuo (Ramsauer et al., 2020)
        let max_score = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut weights: Vec<f64> = scores.iter().map(|&s| (s - max_score).exp()).collect();
        let sum_w: f64 = weights.iter().sum::<f64>().max(1e-12);
        for w in &mut weights {
            *w /= sum_w;
        }

        let mut completed_ec = vec![0.0; self.ec_dim];
        let mut recalled_ca1 = vec![0.0; self.target_dim];
        for (engram, &w) in self.ca3_engrams.iter().zip(weights.iter()) {
            for j in 0..self.ec_dim {
                completed_ec[j] += w * engram.ec_input[j];
            }
            for j in 0..self.target_dim {
                recalled_ca1[j] += w * engram.ca1_target[j];
            }
        }

        // Novedad detectada en CA1 (1 - similitud máxima con episodios conocidos)
        let ca1_novelty = (1.0 - max_sim).clamp(0.0, 1.0);
        (completed_ec, recalled_ca1, ca1_novelty)
    }

    /// Almacena o actualiza un episodio en CA3 mediante plasticidad rápida *one-shot*
    /// modulada por saliencia dopaminérgica/noradrenérgica (Bittner et al., 2017 BTSP).
    pub fn encode_episode(&mut self, ec_input: &[f64], ca1_target: &[f64], salience: f64) {
        let dg_sparse_code = self.dentate_gyrus_pattern_separation(ec_input);

        // Verificar si ya existe un engrama muy cercano para fusionarlo sin redundancia
        let q_norm = l2_norm(ec_input).max(1e-8);
        for engram in &mut self.ca3_engrams {
            let e_norm = l2_norm(&engram.ec_input).max(1e-8);
            let cos_ec = dot(ec_input, &engram.ec_input) / (q_norm * e_norm);
            let diff_norm = l2_dist(ec_input, &engram.ec_input);
            if cos_ec > 0.985 && diff_norm < 0.08 {
                let alpha = 0.35;
                for j in 0..self.ec_dim {
                    engram.ec_input[j] = (1.0 - alpha) * engram.ec_input[j] + alpha * ec_input[j];
                }
                for j in 0..self.target_dim {
                    engram.ca1_target[j] =
                        (1.0 - alpha) * engram.ca1_target[j] + alpha * ca1_target[j];
                }
                engram.salience = engram.salience.max(salience);
                return;
            }
        }

        if self.ca3_engrams.len() >= self.max_engrams {
            // Reemplazar el engrama de menor saliencia efectiva
            if let Some((min_idx, _)) = self
                .ca3_engrams
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.salience.total_cmp(&b.salience))
            {
                self.ca3_engrams[min_idx] = EpisodicEngram {
                    ec_input: ec_input.to_vec(),
                    dg_sparse_code,
                    ca1_target: ca1_target.to_vec(),
                    salience: salience.max(0.1),
                    replay_count: 0,
                };
            }
        } else {
            self.ca3_engrams.push(EpisodicEngram {
                ec_input: ec_input.to_vec(),
                dg_sparse_code,
                ca1_target: ca1_target.to_vec(),
                salience: salience.max(0.1),
                replay_count: 0,
            });
        }
    }

    /// Selecciona un lote de episodios priorizados por saliencia para reproducirlos durante
    /// ráfagas de **Sharp-Wave Ripples (SWR)** en el sueño NREM (Buzsáki, 2015).
    pub fn sample_sharp_wave_ripple_batch(
        &mut self,
        batch_size: usize,
        rng: &mut Xoshiro256StarStar,
    ) -> Vec<(Vec<f64>, Vec<f64>, f64)> {
        if self.ca3_engrams.is_empty() {
            return Vec::new();
        }

        let n = self.ca3_engrams.len();
        let mut weights = Vec::with_capacity(n);
        let mut total_w = 0.0;
        for e in &self.ca3_engrams {
            // Priorizar episodios con alta saliencia y equilibrar cobertura de replay
            let w = (e.salience.max(0.05)) / (1.0 + 0.05 * (e.replay_count as f64));
            weights.push(w);
            total_w += w;
        }

        let mut batch = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            let r = rng.gen_range(0.0, total_w.max(1e-9));
            let mut acc = 0.0;
            let mut chosen = n - 1;
            for (idx, &w) in weights.iter().enumerate() {
                acc += w;
                if r <= acc {
                    chosen = idx;
                    break;
                }
            }
            let engram = &mut self.ca3_engrams[chosen];
            engram.replay_count += 1;
            batch.push((
                engram.ec_input.clone(),
                engram.ca1_target.clone(),
                engram.salience,
            ));
        }
        batch
    }
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[inline]
fn l2_norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

#[inline]
fn l2_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum::<f64>()
        .sqrt()
}
