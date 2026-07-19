use crate::entanglement::{EntanglementConfig, EntanglementField};
use crate::native_thermo_rqm_epr::{
    NativeCandidateScore, NativeRqmQueryReport, NativeThermoRqmConfig, NativeThermoRqmEprSubstrate,
};
use crate::native_thermodynamic_cdt::NativeThermoCdtConfig;
use crate::plasticity_controller::{run_plasticity_cycle, PlasticityConfig, PlasticityReport};
use crate::relational_field::ObserverId;
use std::fs;
use std::hash::{Hash, Hasher};
use std::io;
use std::path::Path;
use std::time::{Duration, Instant};

pub const DEFAULT_TRAINED_STATE: &str = "data/native_thermo_clean.cdt_native";
pub const DEFAULT_NODES_PER_SLICE: usize = 160;
pub const DEFAULT_OBSERVER: ObserverId = ObserverId(260_001);

#[derive(Clone, Copy, Debug)]
pub struct NativeEngineConfig {
    pub eval_repeats: usize,
    pub sleep_attempts: usize,
    pub sleep_replay_passes: usize,
}

impl Default for NativeEngineConfig {
    fn default() -> Self {
        Self {
            eval_repeats: 24,
            sleep_attempts: 8,
            sleep_replay_passes: 2,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum LessonKind {
    Semantic,
    Episodic,
    Causal,
    Skill,
}

#[derive(Clone, Debug)]
pub struct Lesson {
    pub kind: LessonKind,
    pub local: Vec<usize>,
    pub action: Vec<usize>,
    pub remote: Vec<usize>,
    pub distractor: Vec<usize>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct EngineMetrics {
    pub cases: usize,
    pub correct: usize,
    pub leakage_sum: f32,
    pub margin_sum: f32,
    pub dynamics_sum: f32,
}

impl EngineMetrics {
    pub fn record(&mut self, expected: f32, distractor: f32, dynamics: f32) {
        let total = expected + distractor;
        self.cases += 1;
        self.correct += usize::from(expected > distractor);
        self.leakage_sum += if total > f32::EPSILON {
            distractor / total
        } else {
            1.0
        };
        self.margin_sum += expected - distractor;
        self.dynamics_sum += dynamics;
    }

    pub fn accuracy(self) -> f32 {
        self.correct as f32 / self.cases.max(1) as f32
    }

    pub fn leakage(self) -> f32 {
        self.leakage_sum / self.cases.max(1) as f32
    }

    pub fn margin(self) -> f32 {
        self.margin_sum / self.cases.max(1) as f32
    }

    pub fn dynamics(self) -> f32 {
        self.dynamics_sum / self.cases.max(1) as f32
    }
}

#[derive(Clone, Debug)]
pub struct NativePathPruneTarget {
    pub expected: Vec<usize>,
    pub distractor: Vec<usize>,
}

#[derive(Clone, Copy, Debug)]
pub struct NativePathPruneConfig {
    pub leakage_weight: f32,
    pub energy_weight: f32,
    pub state_weight: f32,
    pub score_weight: f32,
    pub protected_weight: f32,
    pub threshold: f32,
}

impl Default for NativePathPruneConfig {
    fn default() -> Self {
        Self {
            leakage_weight: 1.0,
            energy_weight: 0.08,
            state_weight: 0.04,
            score_weight: 0.85,
            protected_weight: 0.60,
            threshold: 0.35,
        }
    }
}

pub fn native_multi_hop_query_pruned(
    substrate: &mut NativeThermoRqmEprSubstrate,
    cue: &[usize],
    max_hops: usize,
    target: Option<&NativePathPruneTarget>,
) -> Vec<NativeCandidateScore> {
    let mut frontier = cue.to_vec();
    let mut accumulated = Vec::<NativeCandidateScore>::new();
    let mut decay = 1.0_f32;
    for _ in 0..max_hops {
        let report = substrate.query(DEFAULT_OBSERVER, 0.0, &frontier);
        merge_candidate_scores(&mut accumulated, &report.candidates, decay);
        frontier = report
            .candidates
            .iter()
            .take(10)
            .map(|candidate| candidate.agent)
            .collect();
        if frontier.is_empty() {
            break;
        }
        decay *= 0.70;
    }
    accumulated.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.agent.cmp(&b.agent))
    });
    if let Some(target) = target {
        native_free_energy_path_prune(
            substrate,
            &mut accumulated,
            target,
            NativePathPruneConfig::default(),
        );
    }
    accumulated.truncate(80);
    accumulated
}

pub fn native_free_energy_path_prune(
    substrate: &NativeThermoRqmEprSubstrate,
    candidates: &mut Vec<NativeCandidateScore>,
    target: &NativePathPruneTarget,
    config: NativePathPruneConfig,
) {
    let max_score = candidates
        .iter()
        .map(|candidate| candidate.score.max(0.0))
        .fold(f32::EPSILON, f32::max);
    candidates.retain(|candidate| {
        let energy = substrate
            .thermal
            .energy
            .get(candidate.agent)
            .copied()
            .unwrap_or(0.0)
            .abs();
        let state = substrate
            .thermal
            .thermal_state
            .get(candidate.agent)
            .copied()
            .unwrap_or(0.0)
            .abs();
        let normalized_score = candidate.score.max(0.0) / max_score;
        let is_expected = target.expected.contains(&candidate.agent);
        let is_distractor = target.distractor.contains(&candidate.agent);
        let free_energy = config.leakage_weight * f32::from(is_distractor)
            + config.energy_weight * energy
            + config.state_weight * state
            - config.score_weight * normalized_score
            - config.protected_weight * f32::from(is_expected);
        free_energy <= config.threshold || is_expected
    });
    candidates.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.agent.cmp(&b.agent))
    });
}

pub fn merge_candidate_scores(
    out: &mut Vec<NativeCandidateScore>,
    incoming: &[NativeCandidateScore],
    weight: f32,
) {
    for candidate in incoming {
        let mut candidate = candidate.clone();
        candidate.score *= weight;
        candidate.relational_score *= weight;
        if let Some(existing) = out.iter_mut().find(|item| item.agent == candidate.agent) {
            existing.score += candidate.score;
            existing.relational_score += candidate.relational_score;
        } else {
            out.push(candidate);
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct EngineBenchmark {
    pub metrics: EngineMetrics,
    pub elapsed: Duration,
    pub relations: usize,
    pub epr_links: usize,
    pub energy: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NativeSleepReport {
    pub attempts: usize,
    pub accepted: usize,
    pub before: EngineMetrics,
    pub after: EngineMetrics,
    pub before_energy: f32,
    pub after_energy: f32,
    pub before_epr_links: usize,
    pub after_epr_links: usize,
}

#[derive(Clone, Debug)]
pub struct NativeEngineRunReport {
    pub source: String,
    pub native_before_sleep: EngineBenchmark,
    pub sleep: NativeSleepReport,
    pub plasticity: PlasticityReport,
    pub native_after_sleep: EngineBenchmark,
    pub decision: NativeEngineDecision,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeEngineDecision {
    StablePass,
    NeedsTuning,
}

impl NativeEngineDecision {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::StablePass => "native_thermo_stable_pass",
            Self::NeedsTuning => "native_thermo_needs_tuning",
        }
    }
}

/// Carga un checkpoint nativo limpio, o crea y entrena uno fresco con lecciones canónicas.
pub fn run_native_thermo_engine<P: AsRef<Path>>(
    state_path: P,
    config: NativeEngineConfig,
) -> io::Result<NativeEngineRunReport> {
    let lessons = canonical_lessons();
    let path = state_path.as_ref();
    let (mut substrate, source) = match load_native_clean_checkpoint(path) {
        Ok(substrate) => (substrate, format!("loaded={}", path.display())),
        Err(_) => {
            let mut substrate = fresh_native_substrate();
            train_canonical(&mut substrate, &lessons, 4);
            (substrate, "fresh_trained=true".to_string())
        }
    };

    let native_before_sleep = benchmark_native(&substrate, &lessons, config.eval_repeats);
    let (native_slept, sleep) = native_sleep_consolidate(
        substrate,
        &lessons,
        config.sleep_attempts,
        config.sleep_replay_passes,
    );
    // Sueño B: expandir EPR por composición selectiva + consolidación A.
    let (native_slept, _prospective) =
        native_sleep_prospective(native_slept, &lessons, ProspectiveSleepConfig::default());
    let (native_slept, plasticity) =
        run_plasticity_cycle(native_slept, &lessons, PlasticityConfig::default());
    substrate = native_slept;
    let native_after_sleep = benchmark_native(&substrate, &lessons, config.eval_repeats);
    let decision = decision(native_before_sleep, native_after_sleep);

    Ok(NativeEngineRunReport {
        source,
        native_before_sleep,
        sleep,
        plasticity,
        native_after_sleep,
        decision,
    })
}

#[derive(Clone, Copy, Debug)]
pub struct ProspectiveSleepConfig {
    pub attempts: usize,
    pub futures_per_attempt: usize,
    pub cap_delta: usize,
    /// Si está activo, reserva cupo desalojando EPR de menor utilidad en vez de crecer sin límite.
    pub utility_replacement: bool,
    pub prospective_slots_per_node: usize,
    pub create_threshold_scale: f32,
    pub train_success_base: f32,
    pub contrast_scale: f32,
    pub min_future_accuracy: f32,
    pub min_useful_epr_ratio: f32,
    pub max_base_accuracy_drop: f32,
    pub max_base_leak_drift: f32,
    /// Solo acepta puentes A→B→C con prior >= este umbral.
    pub min_bridge_prior: f32,
    /// Score mínimo de arista RQM para considerar A→B / B→C.
    pub min_edge_score: f32,
    /// Si true, aplica Sueño A (consolidación/poda) tras cada intento B.
    pub consolidate_after: bool,
    pub consolidate_attempts: usize,
    pub consolidate_replay_passes: usize,
}

impl Default for ProspectiveSleepConfig {
    fn default() -> Self {
        Self {
            attempts: 4,
            futures_per_attempt: 12,
            cap_delta: 4,
            utility_replacement: true,
            prospective_slots_per_node: 1,
            create_threshold_scale: 0.85,
            train_success_base: 0.28,
            contrast_scale: 0.08,
            min_future_accuracy: 0.55,
            min_useful_epr_ratio: 0.15,
            max_base_accuracy_drop: 0.01,
            max_base_leak_drift: 0.015,
            min_bridge_prior: 0.45,
            min_edge_score: 0.20,
            consolidate_after: true,
            consolidate_attempts: 4,
            consolidate_replay_passes: 2,
        }
    }
}

#[derive(Clone, Debug)]
struct DreamLesson {
    kind: LessonKind,
    local: Vec<usize>,
    action: Vec<usize>,
    remote: Vec<usize>,
    distractor: Vec<usize>,
    prior: f32,
}

#[derive(Clone, Debug, Default)]
pub struct ProspectiveSleepReport {
    pub attempts: usize,
    pub accepted: usize,
    pub futures_generated: usize,
    pub futures_trained: usize,
    pub epr_before: usize,
    pub epr_after: usize,
    pub epr_created: usize,
    pub epr_replaced: usize,
    pub cap_before: usize,
    pub cap_after: usize,
    pub base_before: EngineMetrics,
    pub base_after: EngineMetrics,
    pub consolidate_accepted: usize,
    pub decision: &'static str,
}

/// Sueño prospectivo: puentes A→B→C selectivos, cupo EPR +Δ, luego Sueño A.
/// Objetivo: bajar fuga y crecer EPR sin degradar memoria consolidada.
pub fn native_sleep_prospective(
    substrate: NativeThermoRqmEprSubstrate,
    base_lessons: &[Lesson],
    config: ProspectiveSleepConfig,
) -> (NativeThermoRqmEprSubstrate, ProspectiveSleepReport) {
    let base_before = evaluate_native_suite(&substrate, base_lessons);
    let epr_before = substrate.entanglement.active_count();
    let cap_before = substrate.entanglement.max_links_per_node();
    let original = substrate.clone();
    let mut best = substrate;
    let mut best_base = base_before;
    let mut accepted = 0;
    let mut futures_generated = 0;
    let mut futures_trained = 0;
    let mut best_epr_after = epr_before;
    let mut best_epr_replaced = 0usize;
    let mut best_cap_after = cap_before;
    let mut consolidate_accepted_total = 0usize;

    for attempt in 0..config.attempts.max(1) {
        let mut candidate = original.clone();
        let dream_cap = if config.utility_replacement {
            cap_before
        } else {
            cap_before + config.cap_delta
        };
        candidate
            .entanglement
            .set_max_links_per_node(dream_cap.max(cap_before));
        let soft_threshold =
            candidate.entanglement.config.create_threshold * config.create_threshold_scale.max(0.1);
        candidate.entanglement.set_create_threshold(soft_threshold);

        let mut futures = generate_composition_dream_futures(
            &candidate,
            base_lessons,
            config.futures_per_attempt.max(4),
            attempt,
            config.min_bridge_prior,
            config.min_edge_score,
        );
        futures_generated += futures.len();
        if futures.is_empty() {
            continue;
        }
        futures.sort_by(|a, b| b.prior.total_cmp(&a.prior));
        let train_n = ((futures.len() + 1) / 2)
            .max(1)
            .min(futures.len().saturating_sub(1).max(1));
        let split = train_n.min(futures.len().saturating_sub(1)).max(1);
        let (train_futures, probe_futures) = futures.split_at(split);

        let epr_before_attempt = candidate.entanglement.active_count();
        let mut epr_replaced_attempt = 0usize;
        let mut epr_created_attempt = 0usize;
        for dream in train_futures {
            let (created, replaced) = train_dream_future(&mut candidate, dream, config);
            epr_created_attempt += created;
            epr_replaced_attempt += replaced;
        }
        futures_trained += train_futures.len();
        candidate.thermal.run_until_stable(4, 1.0e-5, 1.0e-5);

        let mut consolidate_accepted = 0usize;
        if config.consolidate_after {
            let (slept, sleep_a) = native_sleep_consolidate(
                candidate,
                base_lessons,
                config.consolidate_attempts.max(1),
                config.consolidate_replay_passes.max(1),
            );
            candidate = slept;
            consolidate_accepted = sleep_a.accepted;
            consolidate_accepted_total += sleep_a.accepted;
        }

        let base_metrics = evaluate_native_suite(&candidate, base_lessons);
        let future_metrics = evaluate_dream_suite(&candidate, probe_futures);
        let epr_after_attempt = candidate.entanglement.active_count();
        let epr_delta = epr_after_attempt.saturating_sub(epr_before_attempt);
        let useful_ratio = if epr_delta == 0 {
            0.0
        } else {
            future_metrics.accuracy()
        };

        let preserves_base = base_metrics.accuracy() + config.max_base_accuracy_drop
            >= base_before.accuracy()
            && base_metrics.leakage() <= base_before.leakage() + config.max_base_leak_drift;
        let futures_ok =
            !probe_futures.is_empty() && future_metrics.accuracy() >= config.min_future_accuracy;
        let epr_changed = epr_delta > 0 || epr_created_attempt > 0 || epr_replaced_attempt > 0;
        let accept = preserves_base
            && epr_changed
            && (futures_ok
                || (config.consolidate_after
                    && consolidate_accepted > 0
                    && base_metrics.accuracy() + 0.001 >= base_before.accuracy()
                    && base_metrics.leakage() <= base_before.leakage() + 0.001
                    && (useful_ratio + 1.0e-6 >= config.min_useful_epr_ratio
                        || base_metrics.leakage() + 0.001 < base_before.leakage())));

        if accept {
            best = candidate;
            best_base = base_metrics;
            best_epr_after = epr_after_attempt;
            best_epr_replaced = epr_replaced_attempt;
            best_cap_after = dream_cap;
            accepted += 1;
        }
    }

    let decision = if accepted > 0 {
        "prospective_sleep_accept"
    } else {
        "prospective_sleep_reject"
    };
    let report = ProspectiveSleepReport {
        attempts: config.attempts.max(1),
        accepted,
        futures_generated,
        futures_trained,
        epr_before,
        epr_after: best_epr_after,
        epr_created: best_epr_after.saturating_sub(epr_before),
        epr_replaced: best_epr_replaced,
        cap_before,
        cap_after: best_cap_after,
        base_before,
        base_after: best_base,
        consolidate_accepted: consolidate_accepted_total,
        decision,
    };
    (best, report)
}

fn generate_composition_dream_futures(
    substrate: &NativeThermoRqmEprSubstrate,
    base_lessons: &[Lesson],
    budget: usize,
    salt: usize,
    min_bridge_prior: f32,
    min_edge_score: f32,
) -> Vec<DreamLesson> {
    let mut out = Vec::new();
    let mut seen = std::collections::HashSet::<u64>::new();

    // Solo puentes A→B→C de alto prior (sin cruces agresivos entre lecciones).
    let mut outgoing: std::collections::HashMap<usize, Vec<(usize, f32)>> =
        std::collections::HashMap::new();
    let mut incoming_score: std::collections::HashMap<(usize, usize), f32> =
        std::collections::HashMap::new();
    for (_observer, source, target, amplitude, _phase, coherence, uncertainty, _tick) in
        substrate.relation_entries()
    {
        let score = amplitude * amplitude * coherence * (1.0 - uncertainty);
        if score < min_edge_score.max(0.05) {
            continue;
        }
        outgoing.entry(source).or_default().push((target, score));
        let key = (source, target);
        let entry = incoming_score.entry(key).or_insert(0.0);
        *entry = entry.max(score);
    }
    for edges in outgoing.values_mut() {
        edges.sort_by(|a, b| b.1.total_cmp(&a.1));
        edges.truncate(3);
    }

    // Preferir puentes anclados a lecciones del currículo (menos ruido).
    let lesson_nodes = base_lessons
        .iter()
        .flat_map(|lesson| {
            lesson
                .local
                .iter()
                .chain(lesson.remote.iter())
                .copied()
                .collect::<Vec<_>>()
        })
        .collect::<std::collections::HashSet<_>>();

    let mut bridges = Vec::new();
    for (&a, mids) in &outgoing {
        for &(b, score_ab) in mids {
            let Some(next) = outgoing.get(&b) else {
                continue;
            };
            for &(c, score_bc) in next {
                if a == c {
                    continue;
                }
                let direct = incoming_score.get(&(a, c)).copied().unwrap_or(0.0);
                if direct >= min_edge_score {
                    continue;
                }
                // Normalizar prior a [0,1] aproximando scores RQM tipicos.
                let prior =
                    (score_ab.min(score_bc) / (score_ab.min(score_bc) + 1.0)).clamp(0.0, 1.0);
                let raw = score_ab.min(score_bc);
                if prior + 1.0e-6 < min_bridge_prior && raw < min_bridge_prior {
                    // Acepta si prior normalizado o score crudo supera umbral.
                    if raw + 1.0e-6 < min_bridge_prior {
                        continue;
                    }
                }
                let anchored = lesson_nodes.contains(&a) || lesson_nodes.contains(&c);
                let rank = raw + if anchored { 1.0 } else { 0.0 };
                bridges.push((
                    rank,
                    prior.max((raw / (raw + 4.0)).clamp(0.0, 1.0)),
                    a,
                    b,
                    c,
                    anchored,
                ));
            }
        }
    }
    bridges.sort_by(|a, b| b.0.total_cmp(&a.0));
    // Diversificar un poco entre intentos sobre el top-K, sin bajar el umbral.
    if salt > 0 && bridges.len() > 4 {
        let top = bridges.len().min(12);
        let rot = (salt * 3) % top;
        bridges[..top].rotate_left(rot);
        bridges.sort_by(|a, b| b.0.total_cmp(&a.0));
    }

    for (_rank, prior, a, _b, c, _anchored) in bridges {
        let distractor = base_lessons
            .iter()
            .find(|lesson| lesson.remote.contains(&c) || lesson.local.contains(&a))
            .map(|lesson| lesson.distractor.clone())
            .filter(|d| !d.is_empty())
            .unwrap_or_else(|| {
                base_lessons
                    .iter()
                    .map(|lesson| lesson.distractor.clone())
                    .find(|d| !d.is_empty())
                    .unwrap_or_else(|| {
                        vec![c.wrapping_add(17) % substrate.thermal.node_count().max(1)]
                    })
            });
        push_dream(
            &mut out,
            &mut seen,
            DreamLesson {
                kind: LessonKind::Causal,
                local: vec![a],
                action: Vec::new(),
                remote: vec![c],
                distractor,
                prior,
            },
        );
        if out.len() >= budget {
            break;
        }
    }
    out
}

fn push_dream(
    out: &mut Vec<DreamLesson>,
    seen: &mut std::collections::HashSet<u64>,
    dream: DreamLesson,
) {
    if dream.local.is_empty() || dream.remote.is_empty() {
        return;
    }
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    dream.local.hash(&mut hasher);
    dream.remote.hash(&mut hasher);
    dream.distractor.hash(&mut hasher);
    if seen.insert(hasher.finish()) {
        out.push(dream);
    }
}

fn train_dream_future(
    substrate: &mut NativeThermoRqmEprSubstrate,
    dream: &DreamLesson,
    config: ProspectiveSleepConfig,
) -> (usize, usize) {
    // Success bajo: prior alto ayuda, pero no sobreescribe memoria consolidada.
    let success =
        (config.train_success_base + 0.20 * dream.prior.clamp(0.0, 1.0)).clamp(0.15, 0.55);
    let mut replaced = 0usize;
    let mut missing_before = Vec::new();
    for &source in &dream.local {
        for &target in &dream.remote {
            if !substrate.entanglement.has_active_link(source, target) {
                missing_before.push((source, target));
                if config.utility_replacement {
                    replaced += substrate.entanglement.reserve_pair_capacity(
                        source,
                        target,
                        config.prospective_slots_per_node,
                    );
                }
            }
        }
    }
    let epr_benefit = success + success.max(substrate.entanglement.config.create_threshold * 0.9);
    substrate.train_observed_transition_with_epr_benefit(
        DEFAULT_OBSERVER,
        0.0,
        &dream.local,
        &dream.remote,
        success,
        epr_benefit,
    );
    let created = missing_before
        .iter()
        .filter(|&&(source, target)| substrate.entanglement.has_active_link(source, target))
        .count();
    if !dream.distractor.is_empty() {
        attenuate_distractor(
            substrate,
            DEFAULT_OBSERVER,
            &dream.local,
            &dream.distractor,
            config.contrast_scale * (1.0 - dream.prior).clamp(0.05, 1.0),
        );
    }
    (created, replaced)
}

fn evaluate_dream_suite(
    substrate: &NativeThermoRqmEprSubstrate,
    dreams: &[DreamLesson],
) -> EngineMetrics {
    if dreams.is_empty() {
        return EngineMetrics::default();
    }
    let lessons = dreams
        .iter()
        .map(|dream| Lesson {
            kind: dream.kind,
            local: dream.local.clone(),
            action: dream.action.clone(),
            remote: dream.remote.clone(),
            distractor: dream.distractor.clone(),
        })
        .collect::<Vec<_>>();
    evaluate_native(substrate, &lessons, EvalMode::Normal)
}

pub fn native_sleep_consolidate(
    substrate: NativeThermoRqmEprSubstrate,
    lessons: &[Lesson],
    attempts: usize,
    replay_passes: usize,
) -> (NativeThermoRqmEprSubstrate, NativeSleepReport) {
    let mut best = substrate;
    let before = evaluate_native_suite(&best, lessons);
    let before_energy = best.thermal.report().mean_energy;
    let before_epr_links = best.entanglement.active_count();
    let mut best_metrics = before;
    let mut accepted = 0;

    for attempt in 0..attempts {
        let mut candidate = best.clone();
        replay_native_sleep(&mut candidate, lessons, attempt, replay_passes);
        candidate.thermal.run_until_stable(4, 1.0e-5, 1.0e-5);
        let metrics = evaluate_native_suite(&candidate, lessons);
        let preserves_memory = metrics.accuracy() + 0.0001 >= before.accuracy()
            && metrics.leakage() <= best_metrics.leakage() + 0.0005;
        let improves_stats = metrics.leakage() + 0.0001 < best_metrics.leakage()
            || metrics.margin() > best_metrics.margin() + 0.0001;
        if preserves_memory && improves_stats {
            best = candidate;
            best_metrics = metrics;
            accepted += 1;
        }
    }
    best.entanglement.compact_inactive();

    let report = NativeSleepReport {
        attempts,
        accepted,
        before,
        after: best_metrics,
        before_energy,
        after_energy: best.thermal.report().mean_energy,
        before_epr_links,
        after_epr_links: best.entanglement.active_count(),
    };
    (best, report)
}

pub fn benchmark_native(
    substrate: &NativeThermoRqmEprSubstrate,
    lessons: &[Lesson],
    repeats: usize,
) -> EngineBenchmark {
    let start = Instant::now();
    let mut metrics = EngineMetrics::default();
    for _ in 0..repeats {
        metrics = merge_metrics(
            metrics,
            evaluate_native(substrate, lessons, EvalMode::Normal),
        );
        metrics = merge_metrics(
            metrics,
            evaluate_native(substrate, lessons, EvalMode::ActionConditioned),
        );
        metrics = merge_metrics(metrics, evaluate_typed_native(substrate, lessons));
    }

    EngineBenchmark {
        metrics,
        elapsed: start.elapsed(),
        relations: substrate.relation_count(),
        epr_links: substrate.entanglement.active_count(),
        energy: substrate.thermal.report().mean_energy,
    }
}

pub fn canonical_lessons() -> Vec<Lesson> {
    [
        (
            LessonKind::Semantic,
            "vanchurin",
            "represent",
            "madelung",
            "noise",
        ),
        (
            LessonKind::Semantic,
            "mera",
            "compress",
            "holography",
            "flat",
        ),
        (
            LessonKind::Causal,
            "dvali",
            "stabilize",
            "criticality",
            "thermal",
        ),
        (
            LessonKind::Causal,
            "wolfram",
            "branch",
            "causal_invariance",
            "random",
        ),
        (
            LessonKind::Episodic,
            "graphity",
            "cool",
            "geometrogenesis",
            "complete",
        ),
        (
            LessonKind::Skill,
            "landauer",
            "forget",
            "dissipation",
            "free",
        ),
        (LessonKind::Episodic, "page", "retain", "retention", "loss"),
        (
            LessonKind::Skill,
            "markov",
            "separate",
            "blanket",
            "external",
        ),
    ]
    .into_iter()
    .map(|(kind, local, action, remote, distractor)| Lesson {
        kind,
        local: pattern(local, 0),
        action: pattern(action, 0),
        remote: pattern(remote, 1),
        distractor: pattern(distractor, 1),
    })
    .collect()
}

pub fn fresh_native_substrate() -> NativeThermoRqmEprSubstrate {
    NativeThermoRqmEprSubstrate::new(native_cdt_config(), native_rqm_config(), epr_config())
}

pub fn load_native_clean_checkpoint<P: AsRef<Path>>(
    path: P,
) -> Result<NativeThermoRqmEprSubstrate, String> {
    load_native_checkpoint(path)
}

/// Carga checkpoints nativos limpios o del currículo de cinco fases.
pub fn load_native_checkpoint<P: AsRef<Path>>(
    path: P,
) -> Result<NativeThermoRqmEprSubstrate, String> {
    let contents = fs::read_to_string(path).map_err(|err| err.to_string())?;
    let mut lines = contents.lines();
    let header = lines.next();
    if !matches!(
        header,
        Some("NATIVE_THERMO_RQM_EPR_CLEAN_STATE_V1")
            | Some("NATIVE_THERMO_RQM_EPR_CURRICULUM_STATE_V1")
    ) {
        return Err("version nativa invalida".to_string());
    }
    let _stats_line = lines.next().ok_or("falta stats")?;
    let thermal_config = parse_thermal_config(lines.next().ok_or("falta thermal_config")?)?;
    let rqm_config = parse_rqm_config(lines.next().ok_or("falta rqm_config")?)?;
    let mut substrate =
        NativeThermoRqmEprSubstrate::new(thermal_config, rqm_config, EntanglementConfig::default());
    let node_count = parse_count_header(lines.next().ok_or("faltan nodes")?, "nodes")?;
    for _ in 0..node_count {
        let line = lines.next().ok_or("faltan nodos")?;
        let parts = line.split_whitespace().collect::<Vec<_>>();
        if parts.len() != 8 || parts[0] != "n" {
            return Err(format!("nodo nativo invalido: {line}"));
        }
        let idx = parse_usize(parts[1], "idx")?;
        if idx < substrate.thermal.node_count() {
            substrate.thermal.thermal_state[idx] = parse_f32(parts[2], "state")?;
            substrate.thermal.amplitude[idx] = parse_f32(parts[3], "amplitude")?;
            substrate.thermal.phase[idx] = parse_f32(parts[4], "phase")?;
            substrate.thermal.temperature[idx] = parse_f32(parts[5], "temperature")?;
            substrate.thermal.energy[idx] = parse_f32(parts[6], "energy")?;
            substrate.thermal.activation[idx] = parse_f32(parts[7], "activation")?;
        }
    }
    let relation_count = parse_count_header(lines.next().ok_or("faltan relations")?, "relations")?;
    for _ in 0..relation_count {
        let line = lines.next().ok_or("faltan relaciones")?;
        let parts = line.split_whitespace().collect::<Vec<_>>();
        if parts.len() != 9 || parts[0] != "r" {
            return Err(format!("relacion nativa invalida: {line}"));
        }
        substrate.import_relation_state(
            ObserverId(parse_usize(parts[1], "observer")?),
            parse_usize(parts[2], "source")?,
            parse_usize(parts[3], "target")?,
            parse_f32(parts[4], "amplitude")?,
            parse_f32(parts[5], "phase")?,
            parse_f32(parts[6], "coherence")?,
            parse_f32(parts[7], "uncertainty")?,
            parse_u64(parts[8], "last_tick")?,
        );
    }
    let rest = lines.collect::<Vec<_>>().join("\n");
    if let Some(entanglement) = section(&rest, "entanglement_begin", "entanglement_end") {
        let mut field = EntanglementField::new(EntanglementConfig::default());
        field.apply_persistent_state(&entanglement)?;
        substrate.entanglement = field;
    }
    Ok(substrate)
}

pub fn train_canonical(
    substrate: &mut NativeThermoRqmEprSubstrate,
    lessons: &[Lesson],
    epochs: usize,
) {
    for _ in 0..epochs {
        for lesson in lessons {
            substrate.train_observed_transition(
                DEFAULT_OBSERVER,
                0.0,
                &lesson.local,
                &lesson.remote,
                1.0,
            );
            attenuate_distractor(
                substrate,
                DEFAULT_OBSERVER,
                &lesson.local,
                &lesson.distractor,
                0.35,
            );
            let action_cue = cue_for_mode(lesson, EvalMode::ActionConditioned);
            substrate.train_observed_transition(
                DEFAULT_OBSERVER,
                0.0,
                &action_cue,
                &lesson.remote,
                0.95,
            );
            let typed = typed_observer(lesson.kind);
            substrate.train_observed_transition(typed, 0.0, &lesson.local, &lesson.remote, 1.0);
        }
    }
}

fn replay_native_sleep(
    substrate: &mut NativeThermoRqmEprSubstrate,
    lessons: &[Lesson],
    attempt: usize,
    replay_passes: usize,
) {
    for pass in 0..replay_passes {
        let success = 0.85 + 0.05 * ((attempt + pass) % 3) as f32;
        let attenuation = 0.25 + 0.10 * ((attempt + pass) % 4) as f32;
        for lesson in lessons {
            substrate.train_observed_transition(
                DEFAULT_OBSERVER,
                0.0,
                &lesson.local,
                &lesson.remote,
                success,
            );
            attenuate_distractor(
                substrate,
                DEFAULT_OBSERVER,
                &lesson.local,
                &lesson.distractor,
                attenuation,
            );
            attenuate_contrastive_remotes(
                substrate,
                DEFAULT_OBSERVER,
                &lesson.local,
                lesson,
                lessons,
                attenuation * 0.55,
            );

            let action_cue = cue_for_mode(lesson, EvalMode::ActionConditioned);
            substrate.train_observed_transition(
                DEFAULT_OBSERVER,
                0.0,
                &action_cue,
                &lesson.remote,
                success * 0.90,
            );
            attenuate_distractor(
                substrate,
                DEFAULT_OBSERVER,
                &action_cue,
                &lesson.distractor,
                attenuation,
            );
            attenuate_contrastive_remotes(
                substrate,
                DEFAULT_OBSERVER,
                &action_cue,
                lesson,
                lessons,
                attenuation * 0.50,
            );

            let typed = typed_observer(lesson.kind);
            substrate.train_observed_transition(typed, 0.0, &lesson.local, &lesson.remote, success);
            attenuate_distractor(
                substrate,
                typed,
                &lesson.local,
                &lesson.distractor,
                attenuation,
            );
            attenuate_contrastive_remotes(
                substrate,
                typed,
                &lesson.local,
                lesson,
                lessons,
                attenuation * 0.45,
            );
        }
    }
}

fn attenuate_contrastive_remotes(
    substrate: &mut NativeThermoRqmEprSubstrate,
    observer: ObserverId,
    cue: &[usize],
    lesson: &Lesson,
    lessons: &[Lesson],
    amount: f32,
) {
    for other in lessons {
        if std::ptr::eq(other, lesson) || other.remote == lesson.remote {
            continue;
        }
        attenuate_distractor(substrate, observer, cue, &other.remote, amount);
    }
}

fn attenuate_distractor(
    substrate: &mut NativeThermoRqmEprSubstrate,
    observer: ObserverId,
    cue: &[usize],
    distractor: &[usize],
    amount: f32,
) {
    for &source in cue {
        for &target in distractor {
            substrate.attenuate_relation(observer, source, target, amount);
        }
    }
}

pub fn evaluate_native_suite(
    substrate: &NativeThermoRqmEprSubstrate,
    lessons: &[Lesson],
) -> EngineMetrics {
    merge_metrics(
        merge_metrics(
            evaluate_native(substrate, lessons, EvalMode::Normal),
            evaluate_native(substrate, lessons, EvalMode::ActionConditioned),
        ),
        evaluate_typed_native(substrate, lessons),
    )
}

#[derive(Clone, Copy)]
enum EvalMode {
    Normal,
    ActionConditioned,
}

fn evaluate_native(
    substrate: &NativeThermoRqmEprSubstrate,
    lessons: &[Lesson],
    mode: EvalMode,
) -> EngineMetrics {
    let mut trial = substrate.clone();
    let mut metrics = EngineMetrics::default();
    for lesson in lessons {
        let cue = cue_for_mode(lesson, mode);
        let report = trial.query(DEFAULT_OBSERVER, 0.0, &cue);
        metrics.record(
            score_native(&report.candidates, &lesson.remote),
            score_native(&report.candidates, &lesson.distractor),
            native_dynamics(&report),
        );
    }
    metrics
}

fn evaluate_typed_native(
    substrate: &NativeThermoRqmEprSubstrate,
    lessons: &[Lesson],
) -> EngineMetrics {
    let mut trial = substrate.clone();
    let mut metrics = EngineMetrics::default();
    for lesson in lessons {
        let report = trial.query(typed_observer(lesson.kind), 0.0, &lesson.local);
        metrics.record(
            score_native(&report.candidates, &lesson.remote),
            score_native(&report.candidates, &lesson.distractor),
            native_dynamics(&report),
        );
    }
    metrics
}

fn score_native(candidates: &[NativeCandidateScore], targets: &[usize]) -> f32 {
    candidates
        .iter()
        .filter(|candidate| targets.contains(&candidate.agent))
        .map(|candidate| candidate.score)
        .sum()
}

fn native_dynamics(report: &NativeRqmQueryReport) -> f32 {
    report.thermal.mean_energy.abs() + report.thermal.state_variance
}

fn cue_for_mode(lesson: &Lesson, mode: EvalMode) -> Vec<usize> {
    let mut cue = lesson.local.clone();
    if matches!(mode, EvalMode::ActionConditioned) {
        cue.extend_from_slice(&lesson.action);
        cue.sort_unstable();
        cue.dedup();
    }
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

fn merge_metrics(left: EngineMetrics, right: EngineMetrics) -> EngineMetrics {
    EngineMetrics {
        cases: left.cases + right.cases,
        correct: left.correct + right.correct,
        leakage_sum: left.leakage_sum + right.leakage_sum,
        margin_sum: left.margin_sum + right.margin_sum,
        dynamics_sum: left.dynamics_sum + right.dynamics_sum,
    }
}

fn pattern(label: &str, slice: usize) -> Vec<usize> {
    let mut out = (0..10)
        .map(|offset| {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            label.hash(&mut hasher);
            offset.hash(&mut hasher);
            slice * DEFAULT_NODES_PER_SLICE + (hasher.finish() as usize % DEFAULT_NODES_PER_SLICE)
        })
        .collect::<Vec<_>>();
    out.sort_unstable();
    out.dedup();
    out
}

fn decision(before: EngineBenchmark, after: EngineBenchmark) -> NativeEngineDecision {
    let stable = after.metrics.accuracy() >= 0.97
        && after.metrics.leakage() <= before.metrics.leakage() + 0.01
        && after.metrics.accuracy() + 0.0001 >= before.metrics.accuracy();
    if stable {
        NativeEngineDecision::StablePass
    } else {
        NativeEngineDecision::NeedsTuning
    }
}

fn native_cdt_config() -> NativeThermoCdtConfig {
    NativeThermoCdtConfig {
        slices: 4,
        nodes_per_slice: DEFAULT_NODES_PER_SLICE,
        spatial_degree: 4,
        temporal_degree: 3,
        temperature: 0.85,
        dt: 0.08,
        diffusion: 0.18,
        confinement: 0.12,
        pilot_gain: 0.55,
        phase_coupling: 0.22,
        amplitude_decay: 0.01,
        state_clamp: 4.0,
        seed: 87_301,
    }
}

fn native_rqm_config() -> NativeThermoRqmConfig {
    NativeThermoRqmConfig {
        thermal_steps_per_train: 0,
        thermal_steps_per_query: 2,
        thermal_score_gain: 0.35,
        thermal_activation_margin: f32::MAX,
        max_candidates: 128,
        max_pilot_window_nodes: 96,
        sampling_block_size: 16,
        sampling_schedule_rounds: 2,
        max_sampling_blocks: 8,
        collect_query_diagnostics: true,
        ..NativeThermoRqmConfig::default()
    }
}

fn epr_config() -> EntanglementConfig {
    EntanglementConfig {
        create_threshold: 1.0,
        max_links_per_node: 8,
        max_syncs_per_step: 512,
        contradiction_gain: 0.55,
        max_entropy: 0.9,
        max_heat: 0.9,
        ..EntanglementConfig::default()
    }
}

fn parse_thermal_config(line: &str) -> Result<NativeThermoCdtConfig, String> {
    let p = line.split_whitespace().collect::<Vec<_>>();
    if p.len() != 14 || p[0] != "thermal_config" {
        return Err(format!("thermal_config invalida: {line}"));
    }
    Ok(NativeThermoCdtConfig {
        slices: parse_usize(p[1], "slices")?,
        nodes_per_slice: parse_usize(p[2], "nodes_per_slice")?,
        spatial_degree: parse_usize(p[3], "spatial_degree")?,
        temporal_degree: parse_usize(p[4], "temporal_degree")?,
        temperature: parse_f32(p[5], "temperature")?,
        dt: parse_f32(p[6], "dt")?,
        diffusion: parse_f32(p[7], "diffusion")?,
        confinement: parse_f32(p[8], "confinement")?,
        pilot_gain: parse_f32(p[9], "pilot_gain")?,
        phase_coupling: parse_f32(p[10], "phase_coupling")?,
        amplitude_decay: parse_f32(p[11], "amplitude_decay")?,
        state_clamp: parse_f32(p[12], "state_clamp")?,
        seed: parse_u64(p[13], "seed")?,
    })
}

fn parse_rqm_config(line: &str) -> Result<NativeThermoRqmConfig, String> {
    let p = line.split_whitespace().collect::<Vec<_>>();
    if p.len() != 16 || p[0] != "rqm_config" {
        return Err(format!("rqm_config invalida: {line}"));
    }
    Ok(NativeThermoRqmConfig {
        amplitude_learning_rate: parse_f32(p[1], "amplitude_lr")?,
        coherence_learning_rate: parse_f32(p[2], "coherence_lr")?,
        uncertainty_learning_rate: parse_f32(p[3], "uncertainty_lr")?,
        phase_learning_rate: parse_f32(p[4], "phase_lr")?,
        amplitude_decay: parse_f32(p[5], "amplitude_decay")?,
        thermal_steps_per_train: parse_usize(p[6], "thermal_steps_per_train")?,
        thermal_steps_per_query: parse_usize(p[7], "thermal_steps_per_query")?,
        thermal_score_gain: parse_f32(p[8], "thermal_score_gain")?,
        thermal_activation_margin: parse_f32(p[9], "thermal_activation_margin")?,
        collect_query_diagnostics: parse_usize(p[10], "diagnostics")? != 0,
        max_candidates: parse_usize(p[11], "max_candidates")?,
        max_pilot_window_nodes: parse_usize(p[12], "max_pilot_window_nodes")?,
        sampling_block_size: parse_usize(p[13], "sampling_block_size")?,
        sampling_schedule_rounds: parse_usize(p[14], "sampling_schedule_rounds")?,
        max_sampling_blocks: parse_usize(p[15], "max_sampling_blocks")?,
    })
}

fn parse_count_header(line: &str, label: &str) -> Result<usize, String> {
    let p = line.split_whitespace().collect::<Vec<_>>();
    if p.len() != 2 || p[0] != label {
        return Err(format!("cabecera {label} invalida: {line}"));
    }
    parse_usize(p[1], label)
}

fn section(contents: &str, begin: &str, end: &str) -> Option<String> {
    let start = contents.find(begin)? + begin.len();
    let tail = &contents[start..];
    let stop = tail.find(end)?;
    Some(tail[..stop].trim_matches('\n').to_string())
}

fn parse_usize(value: &str, label: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|err| format!("{label} invalido: {err}"))
}

fn parse_u64(value: &str, label: &str) -> Result<u64, String> {
    value
        .parse::<u64>()
        .map_err(|err| format!("{label} invalido: {err}"))
}

fn parse_f32(value: &str, label: &str) -> Result<f32, String> {
    value
        .parse::<f32>()
        .map_err(|err| format!("{label} invalido: {err}"))
}
