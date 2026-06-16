use crate::simplicial::{ConceptProjection, SimplicialNetwork};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

#[derive(Clone, Copy, Debug)]
pub enum Modality {
    Language,
    Vision,
    Audio,
}

impl Modality {
    fn band(self) -> usize {
        match self {
            Self::Language => 0,
            Self::Vision => 1,
            Self::Audio => 2,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Language => "lenguaje",
            Self::Vision => "vision",
            Self::Audio => "audio",
        }
    }
}

#[derive(Clone, Debug)]
pub struct GroundedConcept {
    pub label: &'static str,
    pub language: &'static [&'static str],
    pub vision: &'static [&'static str],
    pub audio: &'static [&'static str],
}

#[derive(Clone, Debug)]
pub struct DemoTrace {
    pub message: String,
    pub projection: ConceptProjection,
}

#[derive(Clone, Debug)]
pub struct RecallReport {
    pub label: String,
    pub target_agents: usize,
    pub active_target_agents: usize,
    pub target_surprise: f32,
    pub mean_target_surprise: f32,
    pub total_free_energy: f32,
}

pub struct MultimodalDemo {
    concepts: Vec<GroundedConcept>,
    last_trace: DemoTrace,
}

impl MultimodalDemo {
    pub fn new(network: &SimplicialNetwork) -> Self {
        Self {
            concepts: vec![
                GroundedConcept {
                    label: "manzana",
                    language: &["manzana", "fruta", "dulce"],
                    vision: &["redonda", "roja", "verde", "brillante"],
                    audio: &["crujiente", "mordida"],
                },
                GroundedConcept {
                    label: "roca",
                    language: &["roca", "piedra", "mineral"],
                    vision: &["gris", "irregular", "opaca"],
                    audio: &["golpe seco", "raspado"],
                },
            ],
            last_trace: DemoTrace {
                message: "M=train multimodal, L=evocar manzana desde lenguaje, O=evocar roca"
                    .to_string(),
                projection: network.project_active_state(8),
            },
        }
    }

    pub fn train_all(&mut self, network: &mut SimplicialNetwork) {
        let concepts = self.concepts.clone();
        for concept in &concepts {
            self.train_concept(network, concept);
        }
        self.last_trace = DemoTrace {
            message: "Entrenamiento multimodal sintetico: lenguaje, vision y audio coactivados"
                .to_string(),
            projection: network.project_active_state(8),
        };
    }

    pub fn recall_language(&mut self, network: &mut SimplicialNetwork, label: &str) {
        let Some((concept_label, language)) = self
            .concepts
            .iter()
            .find(|concept| concept.label == label)
            .map(|concept| (concept.label, concept.language))
        else {
            self.last_trace = DemoTrace {
                message: format!("Concepto no encontrado: {label}"),
                projection: network.project_active_state(8),
            };
            return;
        };

        let pattern = self.encode_terms(network, Modality::Language, language);
        network.inject_pattern(&pattern, 1.35, 5);
        self.last_trace = DemoTrace {
            message: format!(
                "Evocacion desde lenguaje: '{}' activa su vecindad multimodal",
                concept_label
            ),
            projection: network.project_active_state(8),
        };
    }

    pub fn trace(&self) -> &DemoTrace {
        &self.last_trace
    }

    pub fn refresh_projection(&mut self, network: &SimplicialNetwork) {
        self.last_trace.projection = network.project_active_state(8);
    }

    pub fn concept_labels(&self) -> Vec<&'static str> {
        self.concepts.iter().map(|concept| concept.label).collect()
    }

    pub fn evaluate_recall(
        &mut self,
        network: &mut SimplicialNetwork,
        label: &str,
        steps: usize,
    ) -> Option<RecallReport> {
        let target = self.fused_pattern(network, label)?;
        self.recall_language(network, label);

        for _ in 0..steps {
            network.step();
        }

        let mut active_target_agents = 0;
        let mut target_surprise = 0.0;
        for &idx in &target {
            let surprise = network.agents[idx].surprise;
            target_surprise += surprise;
            if surprise > 0.08 {
                active_target_agents += 1;
            }
        }

        Some(RecallReport {
            label: label.to_string(),
            target_agents: target.len(),
            active_target_agents,
            target_surprise,
            mean_target_surprise: target_surprise / target.len().max(1) as f32,
            total_free_energy: network.total_free_energy(),
        })
    }

    fn train_concept(&self, network: &mut SimplicialNetwork, concept: &GroundedConcept) {
        let language = self.encode_terms(network, Modality::Language, concept.language);
        let vision = self.encode_terms(network, Modality::Vision, concept.vision);
        let audio = self.encode_terms(network, Modality::Audio, concept.audio);

        let mut fused = Vec::new();
        fused.extend(language);
        fused.extend(vision);
        fused.extend(audio);
        fused.sort_unstable();
        fused.dedup();

        network.inject_pattern(&fused, 1.2, 4);
        network.reinforce_coactivation(&fused, 0.18);
    }

    fn fused_pattern(&self, network: &SimplicialNetwork, label: &str) -> Option<Vec<usize>> {
        let concept = self
            .concepts
            .iter()
            .find(|concept| concept.label == label)?;
        let mut fused = Vec::new();
        fused.extend(self.encode_terms(network, Modality::Language, concept.language));
        fused.extend(self.encode_terms(network, Modality::Vision, concept.vision));
        fused.extend(self.encode_terms(network, Modality::Audio, concept.audio));
        fused.sort_unstable();
        fused.dedup();
        Some(fused)
    }

    fn encode_terms(
        &self,
        network: &SimplicialNetwork,
        modality: Modality,
        terms: &[&str],
    ) -> Vec<usize> {
        let len = network.agents.len().max(1);
        let band_size = (len / 3).max(1);
        let band_start = modality.band() * band_size;
        let band_end = if modality.band() == 2 {
            len
        } else {
            ((modality.band() + 1) * band_size).min(len)
        };
        let span = (band_end - band_start).max(1);

        terms
            .iter()
            .enumerate()
            .map(|(i, term)| {
                let mut hasher = DefaultHasher::new();
                modality.name().hash(&mut hasher);
                term.hash(&mut hasher);
                i.hash(&mut hasher);
                band_start + (hasher.finish() as usize % span)
            })
            .collect()
    }
}
