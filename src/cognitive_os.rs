//! Núcleo de control cognitivo sobre CDT-RQM-EPR.
//!
//! El Transformer queda fuera de este módulo: traduce lenguaje a `CognitiveTask`
//! y verbaliza `CognitiveEpisode`. El núcleo conserva memoria tipada, mantiene
//! un espacio de trabajo, explora rutas con beam search, decide y consolida
//! feedback verificado durante sueño.

use crate::native_thermo_rqm_epr::NativeThermoRqmEprSubstrate;
use crate::relational_field::ObserverId;
use std::collections::{BTreeSet, HashMap};

const COGNITIVE_OBSERVER_BASE: usize = 880_000;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum CognitiveRelation {
    MemberOf,
    BasedIn,
    CapitalOf,
    LocatedIn,
    Causes,
    Enables,
}

impl CognitiveRelation {
    pub fn observer(self) -> ObserverId {
        let offset = match self {
            Self::MemberOf => 1,
            Self::BasedIn => 2,
            Self::CapitalOf => 3,
            Self::LocatedIn => 4,
            Self::Causes => 5,
            Self::Enables => 6,
        };
        ObserverId(COGNITIVE_OBSERVER_BASE + offset)
    }
}

#[derive(Clone, Debug)]
pub struct SemanticFact {
    pub subject: String,
    pub relation: CognitiveRelation,
    pub object: String,
    pub confidence: f32,
}

#[derive(Clone, Debug)]
pub struct CognitiveTask {
    pub id: String,
    pub start: String,
    pub program: Vec<CognitiveRelation>,
}

#[derive(Clone, Debug)]
pub struct CognitiveAlternative {
    pub entity: String,
    pub score: f32,
}

#[derive(Clone, Debug)]
pub struct CognitiveStep {
    pub relation: CognitiveRelation,
    pub from: String,
    pub chosen: String,
    pub score: f32,
    pub confidence: f32,
    pub alternatives: Vec<CognitiveAlternative>,
}

#[derive(Clone, Debug)]
pub struct WorkingMemory {
    pub task_id: String,
    pub steps: Vec<CognitiveStep>,
}

#[derive(Clone, Debug)]
pub struct CognitiveEpisode {
    pub task: CognitiveTask,
    pub answer: Option<String>,
    pub confidence: f32,
    pub working_memory: WorkingMemory,
    pub verified: Option<bool>,
}

#[derive(Clone, Copy, Debug)]
pub struct CognitiveOsConfig {
    pub pattern_width: usize,
    pub beam_width: usize,
    pub alternatives_per_step: usize,
    pub minimum_score: f32,
    pub sleep_replay_strength: f32,
    pub failed_path_attenuation: f32,
}

impl Default for CognitiveOsConfig {
    fn default() -> Self {
        Self {
            pattern_width: 6,
            beam_width: 4,
            alternatives_per_step: 4,
            minimum_score: 1.0e-8,
            sleep_replay_strength: 1.0,
            failed_path_attenuation: 0.75,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CognitiveMetrics {
    pub cases: usize,
    pub correct: usize,
    pub answered: usize,
    pub confidence_sum: f32,
    pub steps_sum: usize,
}

impl CognitiveMetrics {
    pub fn accuracy(self) -> f32 {
        self.correct as f32 / self.cases.max(1) as f32
    }

    pub fn coverage(self) -> f32 {
        self.answered as f32 / self.cases.max(1) as f32
    }

    pub fn mean_confidence(self) -> f32 {
        self.confidence_sum / self.answered.max(1) as f32
    }

    pub fn mean_steps(self) -> f32 {
        self.steps_sum as f32 / self.cases.max(1) as f32
    }
}

pub trait CognitiveVerifier {
    /// Se invoca únicamente después de decidir. La traza correcta nunca entra
    /// en `think`, por lo que inferencia y supervisión permanecen separadas.
    fn expected_trace(&self, task: &CognitiveTask) -> Option<Vec<String>>;
}

#[derive(Clone, Debug)]
pub struct CognitiveOperatingSystem {
    pub substrate: NativeThermoRqmEprSubstrate,
    pub config: CognitiveOsConfig,
    entities: BTreeSet<String>,
    entity_patterns: HashMap<String, Vec<usize>>,
    next_entity_slot: usize,
    facts: Vec<SemanticFact>,
    episodic_memory: Vec<CognitiveEpisode>,
}

#[derive(Clone)]
struct BeamState {
    entity: String,
    score: f32,
    steps: Vec<CognitiveStep>,
}

impl CognitiveOperatingSystem {
    pub fn new(substrate: NativeThermoRqmEprSubstrate, config: CognitiveOsConfig) -> Self {
        Self {
            substrate,
            config,
            entities: BTreeSet::new(),
            entity_patterns: HashMap::new(),
            next_entity_slot: 0,
            facts: Vec::new(),
            episodic_memory: Vec::new(),
        }
    }

    pub fn facts(&self) -> &[SemanticFact] {
        &self.facts
    }

    pub fn episodic_memory(&self) -> &[CognitiveEpisode] {
        &self.episodic_memory
    }

    pub fn register_entity(&mut self, entity: impl Into<String>) {
        let entity = entity.into();
        if self.entity_patterns.contains_key(&entity) {
            return;
        }
        let width = self.config.pattern_width.max(1);
        let node_count = self.substrate.thermal.node_count().max(1);
        let capacity = (node_count / width).max(1);
        let slot = self.next_entity_slot;
        let nodes = if slot < capacity {
            // Un nodo por partición: asambleas dispersas y sin colisiones.
            (0..width)
                .map(|projection| projection * capacity + slot)
                .filter(|&node| node < node_count)
                .collect()
        } else {
            // Fallback reproducible cuando el registro excede capacidad.
            (0..width)
                .map(|projection| {
                    stable_hash64(
                        0xC09A_171E_05,
                        format!("overflow-entity:{entity}:{projection}").as_bytes(),
                    ) as usize
                        % node_count
                })
                .collect()
        };
        self.next_entity_slot = self.next_entity_slot.saturating_add(1);
        self.entities.insert(entity.clone());
        self.entity_patterns.insert(entity, nodes);
    }

    /// Memoria semántica supervisada. La confianza es evidencia externa, no
    /// entropía autorreferencial del modelo lingüístico.
    pub fn remember_fact(
        &mut self,
        subject: impl Into<String>,
        relation: CognitiveRelation,
        object: impl Into<String>,
        confidence: f32,
    ) {
        let subject = subject.into();
        let object = object.into();
        self.register_entity(subject.clone());
        self.register_entity(object.clone());
        let source = self.entity_nodes(&subject);
        let target = self.entity_nodes(&object);
        let confidence = confidence.clamp(0.0, 1.0);
        self.substrate.train_observed_transition(
            relation.observer(),
            relation_phase(relation),
            &source,
            &target,
            confidence,
        );
        if let Some(fact) = self.facts.iter_mut().find(|fact| {
            fact.subject == subject && fact.relation == relation && fact.object == object
        }) {
            fact.confidence = (fact.confidence + confidence) * 0.5;
        } else {
            self.facts.push(SemanticFact {
                subject,
                relation,
                object,
                confidence,
            });
        }
    }

    /// Explora y decide sin recibir respuesta esperada ni distractores.
    pub fn think(&mut self, task: &CognitiveTask) -> CognitiveEpisode {
        self.register_entity(task.start.clone());
        let mut beams = vec![BeamState {
            entity: task.start.clone(),
            score: 0.0,
            steps: Vec::new(),
        }];

        for &relation in &task.program {
            let mut expanded = Vec::new();
            for beam in &beams {
                let source = self.entity_nodes(&beam.entity);
                let report =
                    self.substrate
                        .query(relation.observer(), relation_phase(relation), &source);
                let node_scores = report
                    .candidates
                    .iter()
                    .map(|candidate| (candidate.agent, candidate.score))
                    .collect::<HashMap<_, _>>();
                let mut alternatives = self
                    .entities
                    .iter()
                    .filter(|entity| entity.as_str() != beam.entity)
                    .filter_map(|entity| {
                        let score = self
                            .entity_nodes(entity)
                            .iter()
                            .filter_map(|node| node_scores.get(node))
                            .sum::<f32>();
                        (score > self.config.minimum_score).then(|| CognitiveAlternative {
                            entity: entity.clone(),
                            score,
                        })
                    })
                    .collect::<Vec<_>>();
                alternatives.sort_by(|left, right| {
                    right
                        .score
                        .total_cmp(&left.score)
                        .then_with(|| left.entity.cmp(&right.entity))
                });
                alternatives.truncate(self.config.alternatives_per_step.max(1));
                let total = alternatives
                    .iter()
                    .map(|alternative| alternative.score.max(0.0))
                    .sum::<f32>()
                    .max(f32::EPSILON);
                for alternative in &alternatives {
                    let confidence = (alternative.score / total).clamp(0.0, 1.0);
                    let mut steps = beam.steps.clone();
                    steps.push(CognitiveStep {
                        relation,
                        from: beam.entity.clone(),
                        chosen: alternative.entity.clone(),
                        score: alternative.score,
                        confidence,
                        alternatives: alternatives.clone(),
                    });
                    expanded.push(BeamState {
                        entity: alternative.entity.clone(),
                        score: beam.score + (1.0 + alternative.score).ln(),
                        steps,
                    });
                }
            }
            expanded.sort_by(|left, right| {
                right
                    .score
                    .total_cmp(&left.score)
                    .then_with(|| left.entity.cmp(&right.entity))
            });
            expanded.truncate(self.config.beam_width.max(1));
            if expanded.is_empty() {
                beams.clear();
                break;
            }
            beams = expanded;
        }

        let best = beams.into_iter().next();
        let (answer, confidence, steps) = match best {
            Some(best) => {
                let confidence = best
                    .steps
                    .iter()
                    .map(|step| step.confidence)
                    .product::<f32>();
                (Some(best.entity), confidence, best.steps)
            }
            None => (None, 0.0, Vec::new()),
        };
        CognitiveEpisode {
            task: task.clone(),
            answer,
            confidence,
            working_memory: WorkingMemory {
                task_id: task.id.clone(),
                steps,
            },
            verified: None,
        }
    }

    /// Evaluación aislada: cada tarea parte del mismo snapshot y el verificador
    /// solo observa la decisión terminada.
    pub fn evaluate<V: CognitiveVerifier>(
        &self,
        tasks: &[CognitiveTask],
        verifier: &V,
    ) -> (CognitiveMetrics, Vec<CognitiveEpisode>) {
        let mut metrics = CognitiveMetrics::default();
        let mut episodes = Vec::with_capacity(tasks.len());
        for task in tasks {
            let mut trial = self.clone();
            let mut episode = trial.think(task);
            let expected = verifier.expected_trace(task);
            let expected_answer = expected.as_ref().and_then(|trace| trace.last());
            let correct = episode.answer.as_ref() == expected_answer;
            episode.verified = Some(correct);
            metrics.cases += 1;
            metrics.correct += usize::from(correct);
            metrics.answered += usize::from(episode.answer.is_some());
            metrics.confidence_sum += episode.confidence;
            metrics.steps_sum += episode.working_memory.steps.len();
            episodes.push(episode);
        }
        (metrics, episodes)
    }

    /// Fase de sueño supervisada por un verificador externo. Las decisiones
    /// fallidas se debilitan; la traza corregida se consolida por replay.
    pub fn dream_with_feedback<V: CognitiveVerifier>(
        &mut self,
        episodes: &[CognitiveEpisode],
        verifier: &V,
        replay_passes: usize,
    ) {
        for _ in 0..replay_passes.max(1) {
            for episode in episodes {
                let Some(correct_trace) = verifier.expected_trace(&episode.task) else {
                    continue;
                };
                let was_correct = episode.answer.as_ref() == correct_trace.last();
                if !was_correct {
                    self.attenuate_episode(episode);
                }
                self.replay_corrected_trace(&episode.task, &correct_trace);
            }
        }
        self.substrate.thermal.run_until_stable(8, 1.0e-5, 1.0e-5);
        self.episodic_memory.extend(episodes.iter().cloned());
    }

    /// Ablación: relajación sin verificador ni replay semántico.
    pub fn relax_without_feedback(&mut self, steps: usize) {
        self.substrate
            .thermal
            .run_until_stable(steps.max(1), 1.0e-5, 1.0e-5);
    }

    /// Separa memoria consolidada de estado de trabajo volátil. El sueño puede
    /// modificar relaciones semánticas, pero antes del siguiente ciclo wake el
    /// campo térmico vuelve al punto homeostático previo.
    pub fn restore_thermal_homeostasis_from(&mut self, baseline: &Self) {
        self.substrate.thermal = baseline.substrate.thermal.clone();
    }

    fn replay_corrected_trace(&mut self, task: &CognitiveTask, trace: &[String]) {
        if trace.len() != task.program.len() + 1 {
            return;
        }
        for (index, &relation) in task.program.iter().enumerate() {
            self.remember_fact(
                trace[index].clone(),
                relation,
                trace[index + 1].clone(),
                self.config.sleep_replay_strength,
            );
        }
    }

    fn attenuate_episode(&mut self, episode: &CognitiveEpisode) {
        for step in &episode.working_memory.steps {
            let source = self.entity_nodes(&step.from);
            let target = self.entity_nodes(&step.chosen);
            for &source_node in &source {
                for &target_node in &target {
                    self.substrate.attenuate_relation(
                        step.relation.observer(),
                        source_node,
                        target_node,
                        self.config.failed_path_attenuation,
                    );
                }
            }
        }
    }

    fn entity_nodes(&self, entity: &str) -> Vec<usize> {
        if let Some(nodes) = self.entity_patterns.get(entity) {
            return nodes.clone();
        }
        let node_count = self.substrate.thermal.node_count().max(1);
        let mut nodes = (0..self.config.pattern_width.max(1))
            .map(|projection| {
                stable_hash64(
                    0xC09A_171E_05,
                    format!("entity:{entity}:{projection}").as_bytes(),
                ) as usize
                    % node_count
            })
            .collect::<Vec<_>>();
        nodes.sort_unstable();
        nodes.dedup();
        nodes
    }
}

fn relation_phase(relation: CognitiveRelation) -> f32 {
    let index = relation.observer().0 - COGNITIVE_OBSERVER_BASE;
    (index as f32 * 0.73).rem_euclid(std::f32::consts::TAU)
}

/// FNV-1a versionado: a diferencia de `DefaultHasher`, su salida forma parte
/// explícita del formato cognitivo y es reproducible entre plataformas.
fn stable_hash64(seed: u64, bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64 ^ seed;
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    // Avalancha final para impedir que sufijos de proyección produzcan la
    // misma asamblea permutada al reducir el hash módulo node_count.
    hash ^= hash >> 30;
    hash = hash.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    hash ^= hash >> 27;
    hash = hash.wrapping_mul(0x94d0_49bb_1331_11eb);
    hash ^ (hash >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entanglement::EntanglementConfig;
    use crate::native_thermo_rqm_epr::NativeThermoRqmConfig;
    use crate::native_thermodynamic_cdt::NativeThermoCdtConfig;

    struct TestVerifier;

    impl CognitiveVerifier for TestVerifier {
        fn expected_trace(&self, task: &CognitiveTask) -> Option<Vec<String>> {
            (task.id == "ada-capital").then(|| {
                vec![
                    "ada".into(),
                    "red-team".into(),
                    "france".into(),
                    "paris".into(),
                ]
            })
        }
    }

    fn fresh_os() -> CognitiveOperatingSystem {
        CognitiveOperatingSystem::new(
            NativeThermoRqmEprSubstrate::new(
                NativeThermoCdtConfig {
                    slices: 4,
                    nodes_per_slice: 64,
                    seed: 7,
                    ..NativeThermoCdtConfig::default()
                },
                NativeThermoRqmConfig {
                    thermal_steps_per_train: 1,
                    thermal_steps_per_query: 0,
                    ..NativeThermoRqmConfig::default()
                },
                EntanglementConfig::default(),
            ),
            CognitiveOsConfig::default(),
        )
    }

    #[test]
    fn composes_unseen_program_without_oracle_in_inference() {
        let mut os = fresh_os();
        os.remember_fact("ada", CognitiveRelation::MemberOf, "red-team", 1.0);
        os.remember_fact("red-team", CognitiveRelation::BasedIn, "france", 1.0);
        os.remember_fact("france", CognitiveRelation::CapitalOf, "paris", 1.0);
        let task = CognitiveTask {
            id: "ada-capital".into(),
            start: "ada".into(),
            program: vec![
                CognitiveRelation::MemberOf,
                CognitiveRelation::BasedIn,
                CognitiveRelation::CapitalOf,
            ],
        };
        let episode = os.think(&task);
        assert_eq!(episode.answer.as_deref(), Some("paris"));
        assert_eq!(episode.working_memory.steps.len(), 3);
    }

    #[test]
    fn verified_sleep_corrects_a_stronger_false_route() {
        let mut os = fresh_os();
        os.remember_fact("ada", CognitiveRelation::MemberOf, "blue-team", 1.0);
        os.remember_fact("ada", CognitiveRelation::MemberOf, "red-team", 0.4);
        os.remember_fact("blue-team", CognitiveRelation::BasedIn, "italy", 1.0);
        os.remember_fact("italy", CognitiveRelation::CapitalOf, "rome", 1.0);
        os.remember_fact("red-team", CognitiveRelation::BasedIn, "france", 1.0);
        os.remember_fact("france", CognitiveRelation::CapitalOf, "paris", 1.0);
        let task = CognitiveTask {
            id: "ada-capital".into(),
            start: "ada".into(),
            program: vec![
                CognitiveRelation::MemberOf,
                CognitiveRelation::BasedIn,
                CognitiveRelation::CapitalOf,
            ],
        };
        let (_, episodes) = os.evaluate(&[task.clone()], &TestVerifier);
        assert_ne!(episodes[0].answer.as_deref(), Some("paris"));
        os.dream_with_feedback(&episodes, &TestVerifier, 6);
        let (after, _) = os.evaluate(&[task], &TestVerifier);
        assert_eq!(after.correct, 1);
    }
}
