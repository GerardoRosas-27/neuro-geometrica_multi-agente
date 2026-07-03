use std::collections::{HashMap, HashSet};
use std::fs;
use std::io;
use std::path::Path;

const TAU: f32 = std::f32::consts::TAU;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ObserverId(pub usize);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct RelationKey {
    pub observer: ObserverId,
    pub a: usize,
    pub b: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct RelationalFieldConfig {
    pub amplitude_learning_rate: f32,
    pub phase_learning_rate: f32,
    pub coherence_learning_rate: f32,
    pub uncertainty_learning_rate: f32,
    pub amplitude_decay: f32,
    pub coherence_decay: f32,
    pub uncertainty_recovery: f32,
    pub activation_threshold: f32,
}

impl Default for RelationalFieldConfig {
    fn default() -> Self {
        Self {
            amplitude_learning_rate: 0.08,
            phase_learning_rate: 0.18,
            coherence_learning_rate: 0.10,
            uncertainty_learning_rate: 0.12,
            amplitude_decay: 0.004,
            coherence_decay: 0.002,
            uncertainty_recovery: 0.01,
            activation_threshold: 0.08,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RelationalState {
    pub amplitude: f32,
    pub phase: f32,
    pub coherence: f32,
    pub uncertainty: f32,
    pub last_observed_tick: u64,
}

impl Default for RelationalState {
    fn default() -> Self {
        Self {
            amplitude: 0.05,
            phase: 0.0,
            coherence: 0.20,
            uncertainty: 0.80,
            last_observed_tick: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct CandidateScore {
    pub agent: usize,
    pub score: f32,
    pub interference: f32,
    pub probability: f32,
    pub mean_coherence: f32,
    pub mean_uncertainty: f32,
}

#[derive(Clone, Debug)]
pub struct CollapseReport {
    pub observer: ObserverId,
    pub seeds: Vec<usize>,
    pub observer_phase: f32,
    pub candidates: Vec<CandidateScore>,
    pub total_interference: f32,
    pub mean_coherence: f32,
    pub mean_uncertainty: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct SimplexPhaseReport {
    pub observer: ObserverId,
    pub vertices: [usize; 3],
    pub phase_closure: f32,
    pub coherence: f32,
    pub tension: f32,
}

#[derive(Clone, Debug)]
pub struct RelationalFieldSubstrate {
    pub config: RelationalFieldConfig,
    pub tick: u64,
    relations: HashMap<RelationKey, RelationalState>,
}

impl RelationalFieldSubstrate {
    pub fn new(config: RelationalFieldConfig) -> Self {
        Self {
            config,
            tick: 0,
            relations: HashMap::new(),
        }
    }

    pub fn relation_count(&self) -> usize {
        self.relations.len()
    }

    pub fn relation_state(
        &self,
        observer: ObserverId,
        a: usize,
        b: usize,
    ) -> Option<RelationalState> {
        let (a, b) = ordered_pair(a, b);
        self.relations.get(&RelationKey { observer, a, b }).copied()
    }

    pub fn relation_probability(&self, observer: ObserverId, a: usize, b: usize) -> Option<f32> {
        let state = self.relation_state(observer, a, b)?;
        Some(state.amplitude * state.amplitude * state.coherence * (1.0 - state.uncertainty))
    }

    pub fn phase_alignment(
        &self,
        observer: ObserverId,
        source: usize,
        target: usize,
        observer_phase: f32,
    ) -> Option<f32> {
        let phase = self.oriented_phase(observer, source, target)?;
        Some((phase - normalize_phase(observer_phase)).cos())
    }

    pub fn modulation(
        &self,
        observer: ObserverId,
        source: usize,
        target: usize,
        observer_phase: f32,
    ) -> Option<f32> {
        let probability = self.relation_probability(observer, source, target)?;
        let alignment = self
            .phase_alignment(observer, source, target, observer_phase)?
            .max(0.0);
        Some(probability * alignment)
    }

    pub fn reinforce_relation(
        &mut self,
        observer: ObserverId,
        a: usize,
        b: usize,
        target_phase: f32,
        prediction_success: f32,
    ) -> RelationalState {
        if a == b {
            return RelationalState::default();
        }

        self.tick = self.tick.wrapping_add(1);
        let (left, right) = ordered_pair(a, b);
        let phase = if left == a {
            target_phase
        } else {
            -target_phase
        };
        let key = RelationKey {
            observer,
            a: left,
            b: right,
        };
        let state = self.relations.entry(key).or_default();
        let success = prediction_success.clamp(0.0, 1.0);
        let failure = 1.0 - success;

        state.amplitude += self.config.amplitude_learning_rate * success * (1.0 - state.amplitude);
        state.amplitude -= self.config.amplitude_learning_rate * failure * state.amplitude * 0.5;
        state.amplitude = state.amplitude.clamp(0.0, 1.0);

        let phase_error = phase_delta(state.phase, phase);
        state.phase =
            normalize_phase(state.phase + self.config.phase_learning_rate * success * phase_error);
        if failure > 0.0 {
            state.phase = normalize_phase(
                state.phase
                    + self.config.phase_learning_rate
                        * failure
                        * phase_error.signum()
                        * std::f32::consts::PI
                        * 0.25,
            );
        }

        state.coherence += self.config.coherence_learning_rate
            * (success - failure * 0.5)
            * (1.0 - state.coherence);
        state.coherence = state.coherence.clamp(0.0, 1.0);

        state.uncertainty += self.config.uncertainty_learning_rate * (failure - success);
        state.uncertainty = state.uncertainty.clamp(0.0, 1.0);
        state.last_observed_tick = self.tick;
        *state
    }

    pub fn observe_pattern(
        &mut self,
        observer: ObserverId,
        seeds: &[usize],
        observer_phase: f32,
        limit: usize,
    ) -> CollapseReport {
        self.tick = self.tick.wrapping_add(1);
        let observer_phase = normalize_phase(observer_phase);
        let seeds = compact_pattern(seeds);
        let seed_set = seeds.iter().copied().collect::<HashSet<_>>();
        let mut scores = HashMap::<usize, ScoreAccumulator>::new();
        let mut total_interference = 0.0;
        let mut coherence_sum = 0.0;
        let mut uncertainty_sum = 0.0;
        let mut observations = 0_usize;

        for seed in &seeds {
            for (key, state) in self.relations.iter_mut() {
                if key.observer != observer || (key.a != *seed && key.b != *seed) {
                    continue;
                }
                let target = if key.a == *seed { key.b } else { key.a };
                if seed_set.contains(&target) {
                    continue;
                }

                state.last_observed_tick = self.tick;
                let oriented_phase = if key.a == *seed {
                    state.phase
                } else {
                    -state.phase
                };
                let interference = state.amplitude
                    * state.coherence
                    * (1.0 - state.uncertainty)
                    * (oriented_phase - observer_phase).cos();
                let probability =
                    state.amplitude * state.amplitude * state.coherence * (1.0 - state.uncertainty);
                let score = (interference.max(0.0) + probability).max(0.0);

                let entry = scores.entry(target).or_default();
                entry.score += score;
                entry.interference += interference;
                entry.probability += probability;
                entry.coherence += state.coherence;
                entry.uncertainty += state.uncertainty;
                entry.count += 1;

                total_interference += interference;
                coherence_sum += state.coherence;
                uncertainty_sum += state.uncertainty;
                observations += 1;
            }
        }

        let mut candidates = scores
            .into_iter()
            .filter_map(|(agent, score)| {
                if score.score < self.config.activation_threshold {
                    return None;
                }
                let count = score.count.max(1) as f32;
                Some(CandidateScore {
                    agent,
                    score: score.score,
                    interference: score.interference,
                    probability: score.probability,
                    mean_coherence: score.coherence / count,
                    mean_uncertainty: score.uncertainty / count,
                })
            })
            .collect::<Vec<_>>();
        candidates.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.agent.cmp(&b.agent))
        });
        candidates.truncate(limit);

        CollapseReport {
            observer,
            seeds,
            observer_phase,
            candidates,
            total_interference,
            mean_coherence: coherence_sum / observations.max(1) as f32,
            mean_uncertainty: uncertainty_sum / observations.max(1) as f32,
        }
    }

    pub fn simplex_phase_report(
        &self,
        observer: ObserverId,
        a: usize,
        b: usize,
        c: usize,
    ) -> Option<SimplexPhaseReport> {
        let ab = self.oriented_phase(observer, a, b)?;
        let bc = self.oriented_phase(observer, b, c)?;
        let ca = self.oriented_phase(observer, c, a)?;
        let phase_closure = normalize_phase(ab + bc + ca);
        let coherence = phase_closure.cos().mul_add(0.5, 0.5);
        Some(SimplexPhaseReport {
            observer,
            vertices: [a, b, c],
            phase_closure,
            coherence,
            tension: 1.0 - coherence,
        })
    }

    pub fn step_decay(&mut self) {
        self.tick = self.tick.wrapping_add(1);
        let amplitude_decay = self.config.amplitude_decay.clamp(0.0, 1.0);
        let coherence_decay = self.config.coherence_decay.clamp(0.0, 1.0);
        let uncertainty_recovery = self.config.uncertainty_recovery.clamp(0.0, 1.0);

        for state in self.relations.values_mut() {
            state.amplitude *= 1.0 - amplitude_decay * state.uncertainty;
            state.coherence *= 1.0 - coherence_decay * state.uncertainty;
            state.uncertainty += uncertainty_recovery * (1.0 - state.uncertainty);
            state.amplitude = state.amplitude.clamp(0.0, 1.0);
            state.coherence = state.coherence.clamp(0.0, 1.0);
            state.uncertainty = state.uncertainty.clamp(0.0, 1.0);
        }
    }

    pub fn save_persistent_state<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        if let Some(parent) = path.as_ref().parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, self.serialize_persistent_state())
    }

    pub fn load_persistent_state<P: AsRef<Path>>(&mut self, path: P) -> io::Result<()> {
        let contents = fs::read_to_string(path)?;
        self.apply_persistent_state(&contents)
            .map_err(|message| io::Error::new(io::ErrorKind::InvalidData, message))
    }

    pub fn serialize_persistent_state(&self) -> String {
        let mut out = String::new();
        out.push_str("SNGA_RQF_RELATIONAL_FIELD_V1\n");
        out.push_str(&format!("tick {}\n", self.tick));
        out.push_str(&format!(
            "config {:.7} {:.7} {:.7} {:.7} {:.7} {:.7} {:.7} {:.7}\n",
            self.config.amplitude_learning_rate,
            self.config.phase_learning_rate,
            self.config.coherence_learning_rate,
            self.config.uncertainty_learning_rate,
            self.config.amplitude_decay,
            self.config.coherence_decay,
            self.config.uncertainty_recovery,
            self.config.activation_threshold
        ));
        out.push_str(&format!("relations {}\n", self.relations.len()));
        let mut relations = self.relations.iter().collect::<Vec<_>>();
        relations.sort_by(|(left_key, _), (right_key, _)| {
            (left_key.observer.0, left_key.a, left_key.b).cmp(&(
                right_key.observer.0,
                right_key.a,
                right_key.b,
            ))
        });
        for (key, state) in relations {
            out.push_str(&format!(
                "r {} {} {} {:.7} {:.7} {:.7} {:.7} {}\n",
                key.observer.0,
                key.a,
                key.b,
                state.amplitude,
                state.phase,
                state.coherence,
                state.uncertainty,
                state.last_observed_tick
            ));
        }
        out.push_str("end\n");
        out
    }

    pub fn apply_persistent_state(&mut self, contents: &str) -> Result<(), String> {
        let mut lines = contents.lines();
        if lines.next() != Some("SNGA_RQF_RELATIONAL_FIELD_V1") {
            return Err("version RQF invalida".to_string());
        }
        let tick_line = lines.next().ok_or("falta tick RQF")?;
        let parts = tick_line.split_whitespace().collect::<Vec<_>>();
        if parts.len() != 2 || parts[0] != "tick" {
            return Err(format!("tick RQF invalido: {tick_line}"));
        }
        self.tick = parse_u64(parts[1], "tick")?;

        let config_line = lines.next().ok_or("falta config RQF")?;
        let parts = config_line.split_whitespace().collect::<Vec<_>>();
        if parts.len() != 9 || parts[0] != "config" {
            return Err(format!("config RQF invalida: {config_line}"));
        }
        self.config = RelationalFieldConfig {
            amplitude_learning_rate: parse_f32(parts[1], "amplitude_lr")?,
            phase_learning_rate: parse_f32(parts[2], "phase_lr")?,
            coherence_learning_rate: parse_f32(parts[3], "coherence_lr")?,
            uncertainty_learning_rate: parse_f32(parts[4], "uncertainty_lr")?,
            amplitude_decay: parse_f32(parts[5], "amplitude_decay")?,
            coherence_decay: parse_f32(parts[6], "coherence_decay")?,
            uncertainty_recovery: parse_f32(parts[7], "uncertainty_recovery")?,
            activation_threshold: parse_f32(parts[8], "activation_threshold")?,
        };

        let relations_header = lines.next().ok_or("faltan relaciones RQF")?;
        let relation_count = parse_count_header(relations_header, "relations")?;
        self.relations.clear();
        for _ in 0..relation_count {
            let line = lines.next().ok_or("faltan lineas RQF")?;
            let parts = line.split_whitespace().collect::<Vec<_>>();
            if parts.len() != 9 || parts[0] != "r" {
                return Err(format!("relacion RQF invalida: {line}"));
            }
            self.relations.insert(
                RelationKey {
                    observer: ObserverId(parse_usize(parts[1], "observer")?),
                    a: parse_usize(parts[2], "a")?,
                    b: parse_usize(parts[3], "b")?,
                },
                RelationalState {
                    amplitude: parse_f32(parts[4], "amplitude")?,
                    phase: parse_f32(parts[5], "phase")?,
                    coherence: parse_f32(parts[6], "coherence")?,
                    uncertainty: parse_f32(parts[7], "uncertainty")?,
                    last_observed_tick: parse_u64(parts[8], "last_tick")?,
                },
            );
        }
        Ok(())
    }

    fn oriented_phase(&self, observer: ObserverId, source: usize, target: usize) -> Option<f32> {
        let (a, b) = ordered_pair(source, target);
        let phase = self.relations.get(&RelationKey { observer, a, b })?.phase;
        Some(if a == source { phase } else { -phase })
    }
}

#[derive(Default)]
struct ScoreAccumulator {
    score: f32,
    interference: f32,
    probability: f32,
    coherence: f32,
    uncertainty: f32,
    count: usize,
}

fn ordered_pair(a: usize, b: usize) -> (usize, usize) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

fn compact_pattern(pattern: &[usize]) -> Vec<usize> {
    let mut out = pattern.to_vec();
    out.sort_unstable();
    out.dedup();
    out
}

fn normalize_phase(phase: f32) -> f32 {
    let mut phase = phase % TAU;
    if phase > std::f32::consts::PI {
        phase -= TAU;
    } else if phase < -std::f32::consts::PI {
        phase += TAU;
    }
    phase
}

fn phase_delta(from: f32, to: f32) -> f32 {
    normalize_phase(to - from)
}

fn parse_count_header(line: &str, label: &str) -> Result<usize, String> {
    let parts = line.split_whitespace().collect::<Vec<_>>();
    if parts.len() != 2 || parts[0] != label {
        return Err(format!("cabecera {label} invalida: {line}"));
    }
    parse_usize(parts[1], label)
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
