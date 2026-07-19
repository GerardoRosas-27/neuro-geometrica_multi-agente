//! Presupuesto de residuo del motor termodinámico (eje imaginario / fase).
//!
//! Formaliza la idea mínima:
//! 1) Softmax sobre candidatos → función de partición local Z
//! 2) Entropía S = -Σ p ln p  (proxy tipo Von Neumann sobre la ventana)
//! 3) Residuo R = 1 - p_chosen  (masa no renderizada)
//! 4) Reciclaje: Sueño A + atenuación extra de perdedores/distractores ∝ R
//!
//! La contabilidad se ejecuta en clone para no contaminar el replay.

use crate::native_thermo_rqm_epr::{NativeCandidateScore, NativeThermoRqmEprSubstrate};
use crate::native_thermodynamic_engine::{
    evaluate_native_suite, EngineMetrics, Lesson, LessonKind, DEFAULT_OBSERVER,
};
use crate::relational_field::ObserverId;

const EPSILON: f32 = 1.0e-8;

#[derive(Clone, Copy, Debug)]
pub struct ResidueBudgetConfig {
    pub temperature: f32,
    /// Extra de atenuación sobre distractores: base * (1 + recycle_gain * R).
    pub recycle_gain: f32,
    /// Atenuación directa de perdedores de query ∝ R * p_loser.
    pub loser_gain: f32,
    pub reinforce_gain: f32,
    pub phase_cancel_gain: f32,
    pub min_residue: f32,
    pub top_losers: usize,
    pub sleep_attempts: usize,
    pub sleep_replay_passes: usize,
}

impl Default for ResidueBudgetConfig {
    fn default() -> Self {
        Self {
            temperature: 0.85,
            recycle_gain: 0.55,
            loser_gain: 0.40,
            reinforce_gain: 0.04,
            phase_cancel_gain: 0.05,
            min_residue: 0.02,
            top_losers: 4,
            sleep_attempts: 6,
            sleep_replay_passes: 2,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ResidueAccount {
    pub partition_z: f32,
    pub entropy_s: f32,
    pub residue_r: f32,
    pub chosen_prob: f32,
    pub residual_phase: f32,
    pub free_energy_proxy: f32,
    pub loser_mass: Vec<(usize, f32)>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ResidueRecycleReport {
    pub queries: usize,
    pub recycled: usize,
    pub mean_residue: f32,
    pub mean_entropy: f32,
    pub mean_z: f32,
    pub mean_free_energy: f32,
    pub attenuated_edges: usize,
    pub reinforced_edges: usize,
}

#[derive(Clone, Debug)]
pub struct ResidueSleepReport {
    pub attempts: usize,
    pub accepted: usize,
    pub before: EngineMetrics,
    pub after: EngineMetrics,
    pub recycle: ResidueRecycleReport,
    pub decision: &'static str,
}

pub fn account_from_candidates(
    candidates: &[NativeCandidateScore],
    chosen_agents: &[usize],
    phases: &[(usize, f32)],
    config: &ResidueBudgetConfig,
) -> ResidueAccount {
    if candidates.is_empty() {
        return ResidueAccount::default();
    }

    let temperature = config.temperature.max(1.0e-3);
    let max_score = candidates
        .iter()
        .map(|c| c.score)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut weights = candidates
        .iter()
        .map(|c| ((c.score - max_score) / temperature).exp())
        .collect::<Vec<_>>();
    let z: f32 = weights.iter().sum();
    if z <= EPSILON {
        return ResidueAccount::default();
    }
    for w in &mut weights {
        *w /= z;
    }

    let entropy = -weights
        .iter()
        .filter(|&&p| p > EPSILON)
        .map(|p| p * p.ln())
        .sum::<f32>();

    let chosen_prob = candidates
        .iter()
        .zip(weights.iter())
        .filter(|(c, _)| chosen_agents.contains(&c.agent))
        .map(|(_, &p)| p)
        .sum::<f32>()
        .clamp(0.0, 1.0);
    let residue = (1.0 - chosen_prob).clamp(0.0, 1.0);

    let phase_lookup = |agent: usize| {
        phases
            .iter()
            .find(|(id, _)| *id == agent)
            .map(|(_, phase)| *phase)
            .unwrap_or(0.0)
    };

    let mut residual_sin = 0.0;
    let mut residual_cos = 0.0;
    let mut loser_mass = Vec::new();
    for (candidate, &prob) in candidates.iter().zip(weights.iter()) {
        if chosen_agents.contains(&candidate.agent) || prob <= EPSILON {
            continue;
        }
        let phase = phase_lookup(candidate.agent);
        residual_sin += prob * phase.sin();
        residual_cos += prob * phase.cos();
        loser_mass.push((candidate.agent, prob));
    }
    loser_mass.sort_by(|a, b| b.1.total_cmp(&a.1));
    loser_mass.truncate(config.top_losers.max(1));
    let residual_phase = residual_sin
        .atan2(residual_cos)
        .rem_euclid(std::f32::consts::TAU);

    ResidueAccount {
        partition_z: z,
        entropy_s: entropy,
        residue_r: residue,
        chosen_prob,
        residual_phase,
        free_energy_proxy: -temperature * z.max(EPSILON).ln(),
        loser_mass,
    }
}

pub fn phases_for_candidates(
    substrate: &NativeThermoRqmEprSubstrate,
    candidates: &[NativeCandidateScore],
) -> Vec<(usize, f32)> {
    candidates
        .iter()
        .map(|c| {
            let phase = substrate.thermal.phase.get(c.agent).copied().unwrap_or(0.0);
            (c.agent, phase)
        })
        .collect()
}

/// Contabiliza residuo sin mutar el sustrato de replay (query en clone).
pub fn probe_residue_account(
    substrate: &NativeThermoRqmEprSubstrate,
    observer: ObserverId,
    cue: &[usize],
    chosen: &[usize],
    config: &ResidueBudgetConfig,
) -> ResidueAccount {
    let mut probe = substrate.clone();
    probe_residue_account_in_place(&mut probe, observer, cue, chosen, config)
}

/// Variante para reutilizar un único clone en una secuencia de probes.
pub fn probe_residue_account_in_place(
    probe: &mut NativeThermoRqmEprSubstrate,
    observer: ObserverId,
    cue: &[usize],
    chosen: &[usize],
    config: &ResidueBudgetConfig,
) -> ResidueAccount {
    let query = probe.query(observer, 0.0, cue);
    let phases = phases_for_candidates(probe, &query.candidates);
    account_from_candidates(&query.candidates, chosen, &phases, config)
}

fn recycle_losers(
    substrate: &mut NativeThermoRqmEprSubstrate,
    observer: ObserverId,
    cue: &[usize],
    account: &ResidueAccount,
    config: &ResidueBudgetConfig,
) -> usize {
    if account.residue_r < config.min_residue || account.loser_mass.is_empty() {
        return 0;
    }
    let mut attenuated = 0usize;
    for &(loser, mass) in &account.loser_mass {
        let amount = (config.loser_gain * account.residue_r * mass).clamp(0.0, 0.75);
        if amount <= EPSILON {
            continue;
        }
        for &source in cue {
            substrate.attenuate_relation(observer, source, loser, amount);
            attenuated += 1;
            if config.phase_cancel_gain > 0.0 {
                let anti_phase = (account.residual_phase + std::f32::consts::PI)
                    .rem_euclid(std::f32::consts::TAU);
                blend_relation_phase(
                    substrate,
                    observer,
                    source,
                    loser,
                    anti_phase,
                    config.phase_cancel_gain * amount,
                );
            }
        }
    }
    attenuated
}

fn blend_relation_phase(
    substrate: &mut NativeThermoRqmEprSubstrate,
    observer: ObserverId,
    source: usize,
    target: usize,
    target_phase: f32,
    amount: f32,
) {
    let amount = amount.clamp(0.0, 1.0);
    if amount <= EPSILON {
        return;
    }
    substrate.blend_relation_phase(observer, source, target, target_phase, amount);
}

fn action_cue(lesson: &Lesson) -> Vec<usize> {
    let mut cue = lesson.local.clone();
    cue.extend_from_slice(&lesson.action);
    cue.sort_unstable();
    cue.dedup();
    cue
}

fn typed_observer(kind: LessonKind) -> ObserverId {
    match kind {
        LessonKind::Semantic => ObserverId(261_001),
        LessonKind::Episodic => ObserverId(261_002),
        LessonKind::Causal => ObserverId(261_003),
        LessonKind::Skill => ObserverId(261_004),
    }
}

fn attenuate_distractor(
    substrate: &mut NativeThermoRqmEprSubstrate,
    observer: ObserverId,
    cue: &[usize],
    distractor: &[usize],
    amount: f32,
) -> usize {
    let mut n = 0;
    for &source in cue {
        for &target in distractor {
            substrate.attenuate_relation(observer, source, target, amount);
            n += 1;
        }
    }
    n
}

fn attenuate_contrastive_remotes(
    substrate: &mut NativeThermoRqmEprSubstrate,
    observer: ObserverId,
    cue: &[usize],
    lesson: &Lesson,
    lessons: &[Lesson],
    amount: f32,
) -> usize {
    let mut n = 0;
    for other in lessons {
        if std::ptr::eq(other, lesson) || other.remote == lesson.remote {
            continue;
        }
        n += attenuate_distractor(substrate, observer, cue, &other.remote, amount);
    }
    n
}

/// Sueño A + reciclaje de residuo (contabilidad en clone).
pub fn replay_residue_sleep(
    substrate: &mut NativeThermoRqmEprSubstrate,
    lessons: &[Lesson],
    attempt: usize,
    config: &ResidueBudgetConfig,
) -> ResidueRecycleReport {
    let mut report = ResidueRecycleReport::default();
    let mut residue_sum = 0.0;
    let mut entropy_sum = 0.0;
    let mut z_sum = 0.0;
    let mut fe_sum = 0.0;

    for pass in 0..config.sleep_replay_passes.max(1) {
        let success = 0.85 + 0.05 * ((attempt + pass) % 3) as f32;
        let base_attenuation = 0.25 + 0.10 * ((attempt + pass) % 4) as f32;

        for lesson in lessons {
            let action = action_cue(lesson);
            let typed = typed_observer(lesson.kind);
            let passes = [
                (DEFAULT_OBSERVER, lesson.local.as_slice(), 1.0_f32, 0.55_f32),
                (DEFAULT_OBSERVER, action.as_slice(), 0.90, 0.50),
                (typed, lesson.local.as_slice(), 1.0, 0.45),
            ];

            for (observer, cue, success_scale, contrast) in passes {
                let account =
                    probe_residue_account(substrate, observer, cue, &lesson.remote, config);
                report.queries += 1;
                residue_sum += account.residue_r;
                entropy_sum += account.entropy_s;
                z_sum += account.partition_z;
                fe_sum += account.free_energy_proxy;

                let scale = 1.0 + config.recycle_gain * account.residue_r;
                let attenuation = (base_attenuation * scale).clamp(0.05, 0.95);

                // Baseline Sueño A (entrenar + atenuar), con atenuación escalada por R.
                substrate.train_observed_transition(
                    observer,
                    0.0,
                    cue,
                    &lesson.remote,
                    success * success_scale,
                );
                report.attenuated_edges +=
                    attenuate_distractor(substrate, observer, cue, &lesson.distractor, attenuation);
                report.attenuated_edges += attenuate_contrastive_remotes(
                    substrate,
                    observer,
                    cue,
                    lesson,
                    lessons,
                    attenuation * contrast,
                );

                let losers = recycle_losers(substrate, observer, cue, &account, config);
                if losers > 0 || account.residue_r >= config.min_residue {
                    report.recycled += 1;
                }
                report.attenuated_edges += losers;

                let reinforce = (config.reinforce_gain * account.residue_r).clamp(0.0, 0.20);
                if reinforce > EPSILON {
                    substrate.train_observed_transition(
                        observer,
                        0.0,
                        cue,
                        &lesson.remote,
                        reinforce,
                    );
                    report.reinforced_edges += cue.len() * lesson.remote.len();
                }
            }
        }
    }

    let n = report.queries.max(1) as f32;
    report.mean_residue = residue_sum / n;
    report.mean_entropy = entropy_sum / n;
    report.mean_z = z_sum / n;
    report.mean_free_energy = fe_sum / n;
    report
}

pub fn native_sleep_residue(
    substrate: NativeThermoRqmEprSubstrate,
    lessons: &[Lesson],
    config: ResidueBudgetConfig,
) -> (NativeThermoRqmEprSubstrate, ResidueSleepReport) {
    let mut best = substrate;
    let before = evaluate_native_suite(&best, lessons);
    let mut best_metrics = before;
    let mut accepted = 0;
    let mut best_recycle = ResidueRecycleReport::default();
    let mut last_recycle = ResidueRecycleReport::default();

    for attempt in 0..config.sleep_attempts.max(1) {
        let mut candidate = best.clone();
        let recycle = replay_residue_sleep(&mut candidate, lessons, attempt, &config);
        last_recycle = recycle;
        candidate.thermal.run_until_stable(4, 1.0e-5, 1.0e-5);
        let metrics = evaluate_native_suite(&candidate, lessons);
        let preserves_memory = metrics.accuracy() + 0.0001 >= before.accuracy()
            && metrics.leakage() <= best_metrics.leakage() + 0.0005;
        let improves_stats = metrics.leakage() + 0.0001 < best_metrics.leakage()
            || metrics.margin() > best_metrics.margin() + 0.0001;
        if preserves_memory && improves_stats {
            best = candidate;
            best_metrics = metrics;
            best_recycle = recycle;
            accepted += 1;
        }
    }

    if accepted == 0 {
        best_recycle = last_recycle;
    }

    let decision = if accepted > 0 && best_metrics.leakage() + 1.0e-6 < before.leakage() {
        "residue_sleep_improve"
    } else if accepted > 0 {
        "residue_sleep_accept"
    } else {
        "residue_sleep_reject"
    };

    (
        best,
        ResidueSleepReport {
            attempts: config.sleep_attempts.max(1),
            accepted,
            before,
            after: best_metrics,
            recycle: best_recycle,
            decision,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_thermodynamic_engine::{
        canonical_lessons, fresh_native_substrate, train_canonical,
    };

    fn cand(agent: usize, score: f32) -> NativeCandidateScore {
        NativeCandidateScore {
            agent,
            score,
            relational_score: score,
            thermal_multiplier: 1.0,
        }
    }

    #[test]
    fn partition_and_residue_unitarity() {
        let candidates = vec![cand(1, 2.0), cand(2, 1.0), cand(3, 0.5)];
        let phases = vec![(1, 0.0), (2, 1.0), (3, 2.0)];
        let account =
            account_from_candidates(&candidates, &[1], &phases, &ResidueBudgetConfig::default());
        assert!(account.partition_z > 0.0);
        assert!((account.chosen_prob + account.residue_r - 1.0).abs() < 1.0e-5);
        assert!(account.entropy_s > 0.0);
        assert!(!account.loser_mass.is_empty());
        assert!(account.free_energy_proxy.is_finite());
    }

    #[test]
    fn pure_state_has_near_zero_residue() {
        let candidates = vec![cand(7, 10.0), cand(8, -5.0)];
        let phases = vec![(7, 0.1), (8, 3.0)];
        let mut config = ResidueBudgetConfig::default();
        config.temperature = 0.25;
        let account = account_from_candidates(&candidates, &[7], &phases, &config);
        assert!(account.chosen_prob > 0.95);
        assert!(account.residue_r < 0.05);
        assert!(account.entropy_s < 0.3);
    }

    #[test]
    fn residue_sleep_matches_or_beats_baseline_leak_band() {
        let lessons = canonical_lessons();
        let mut substrate = fresh_native_substrate();
        train_canonical(&mut substrate, &lessons, 3);
        let before = evaluate_native_suite(&substrate, &lessons);
        let mut config = ResidueBudgetConfig::default();
        config.sleep_attempts = 6;
        let (after, report) = native_sleep_residue(substrate, &lessons, config);
        let metrics = evaluate_native_suite(&after, &lessons);
        assert!(report.recycle.queries > 0);
        assert!(metrics.accuracy() + 1.0e-6 >= before.accuracy());
        assert!(metrics.leakage() + 1.0e-6 <= before.leakage());
        // Debe acercarse al Sueño A fuerte (< 2% leak tras consolidar).
        assert!(metrics.leakage() < 0.03);
    }
}
