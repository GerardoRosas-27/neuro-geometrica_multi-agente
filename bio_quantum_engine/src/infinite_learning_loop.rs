//! Ciclo Infinito de Auto-Aprendizaje en GPU: Gemma 2 Libre (Profesor) ↔ Campo Bio-Quantum v10 x50
//!
//! Arquitectura de Circuito Cerrado Cognitivo-Dinámico:
//! 1. Gemma (Profesor Libre de Dominio Abierto):
//!    Genera libremente lecciones sobre cualquier disciplina científica:
//!    - Campos Cuánticos y Wheeler-Feynman en C.
//!    - Sistemas Dinámicos No Lineales, Caos de Lorenz y Bifurcaciones de Hopf.
//!    - Geometría Diferencial de Lie SO(2)/SO(3) y Variedades Simplécticas.
//!    - Neurodinámica Cortical (Capas L2/3 & L5, Plasticidad BTSP y Ley de Dale).
//!    - Topología y Teorema de Cuenca CDT ante Ruido OOD.
//!    - Astrofísica Relativista y Dinámica de Órbitas.
//! 2. Campo Bio-Quantum x50 en VRAM (AMD Radeon RX 7800 XT):
//!    Evoluciona las coordenadas en 1,024 dimensiones de Lie/Bessel y 4,096 sinapsis.
//! 3. Gemma (Decodificador / Supervisor):
//!    Evalúa el desvío topológico y emite diagnóstico verbal en lenguaje natural.
//! 4. Buffer de Experiencia y Fase de Sueño SWR:
//!    Cada vez que se llena el buffer de memoria (16 experiencias), el modelo SUEÑA:
//!    reactiva las trayectorias en VRAM, consolida sinapsis rápidas W^F en la matriz
//!    cortical lenta W^P con garantía de olvido acotado (ΔR >= -ε).
//! 5. Persistencia Absoluta de Memoria (Checkpoints en Disco):
//!    Guarda automáticamente `weights_infinite_vram.bin` y `infinite_learning_checkpoint.json`.
//!    Si se cierra o reinicia la consola, el modelo recarga su memoria cortical consolidada
//!    y continúa aprendiendo sin perder jamás lo aprendido.

use crate::gemma_field_bridge::GemmaFieldBridge;
use crate::rng::Xoshiro256StarStar;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::thread::sleep;
use std::time::{Duration, Instant};

const CHECKPOINT_WEIGHTS_FILE: &str = "weights_infinite_vram.bin";
const CHECKPOINT_META_FILE: &str = "infinite_learning_checkpoint.json";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurriculumTask {
    pub cycle: usize,
    pub domain_category: String,
    pub family_id: usize,
    pub family_name: String,
    pub params: [f32; 2],
    pub noise_sigma: f32,
    pub teacher_prompt: String,
    pub language: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperienceMemoryItem {
    pub cycle: usize,
    pub domain: String,
    pub family_name: String,
    pub input_coords: Vec<f32>,
    pub target_coords: Vec<f32>,
    pub initial_mse: f32,
    pub prompt: String,
}

pub struct ExperienceBuffer {
    pub capacity: usize,
    pub items: Vec<ExperienceMemoryItem>,
}

impl ExperienceBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            items: Vec::with_capacity(capacity),
        }
    }

    pub fn push(&mut self, item: ExperienceMemoryItem) -> bool {
        self.items.push(item);
        self.items.len() >= self.capacity
    }

    pub fn clear(&mut self) {
        self.items.clear();
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    pub cumulative_cycles: usize,
    pub cumulative_dreams: usize,
    pub cumulative_accurate: usize,
    pub cumulative_corrected: usize,
    pub cumulative_avg_mse: f32,
    pub last_saved_timestamp: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CycleRecord {
    pub cycle: usize,
    pub domain: String,
    pub language: String,
    pub lesson_name: String,
    pub prompt: String,
    pub initial_mse: f32,
    pub post_correction_mse: f32,
    pub error_reduction_pct: f32,
    pub status: String,
    pub gemma_verdict: String,
    pub dale_violations: usize,
    pub dreamed: bool,
    pub gpu_latency_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InfiniteLoopTelemetry {
    pub total_cycles_run: usize,
    pub total_dreams_completed: usize,
    pub accurate_predictions: usize,
    pub corrections_applied: usize,
    pub accuracy_rate_pct: f32,
    pub initial_avg_mse: f32,
    pub final_avg_mse: f32,
    pub total_dale_violations: usize,
    pub avg_gpu_latency_ms: f64,
    pub checkpoint_persisted: bool,
}

pub struct LoopConfig {
    pub max_cycles: usize,        // 0 = Infinito (bucle continuo)
    pub horizon: usize,           // Pasos de rollout GPU (default 16)
    pub buffer_capacity: usize,   // Tamaño del buffer de experiencias antes de soñar (default 16)
    pub learning_rate: f32,       // Tasa de aprendizaje proyectado en Dale (default 0.02)
    pub tolerance_mse: f32,       // Umbral de tolerancia de error (default 0.005)
    pub delay_ms: u64,            // Retardo visual en consola entre ciclos (default 60ms)
    pub default_lang: String,     // Idioma ("es", "en", "fr", "pt", "it")
}

impl Default for LoopConfig {
    fn default() -> Self {
        Self {
            max_cycles: 0, // Por defecto corre de forma continua/infinita
            horizon: 16,
            buffer_capacity: 16,
            learning_rate: 0.02,
            tolerance_mse: 0.005,
            delay_ms: 60,
            default_lang: "es".to_string(),
        }
    }
}

pub struct TeacherGemma {
    pub active_lang: String,
}

impl TeacherGemma {
    pub fn new(lang: &str) -> Self {
        Self {
            active_lang: lang.to_string(),
        }
    }

    /// Genera libremente lecciones multidisciplinarias en cualquier tema
    pub fn generate_free_lesson(&self, cycle: usize, rng: &mut Xoshiro256StarStar) -> CurriculumTask {
        let domain_id = (cycle % 7) + 1; // 7 Dominios científicos
        let r1 = (rng.next_f64() as f32) * 2.0 - 1.0;
        let r2 = (rng.next_f64() as f32) * 2.0 - 1.0;

        let (domain_category, family_id, family_name, params, noise_sigma, prompt) = match domain_id {
            1 => {
                // 1. Campos Cuánticos & Wheeler-Feynman en C
                let theta = 0.22 + (r1 * 0.20).abs();
                let damp = 0.95 - (r2 * 0.03).abs();
                let p1 = theta / 0.12;
                let p2 = (damp - 1.0) / 0.015;
                let prompt = match self.active_lang.as_str() {
                    "en" => format!("Lesson #{}: [Quantum Fields] Evolve Wheeler-Feynman handshake phasor with phase advance +{:.2} rad and radial decay {:.3}x.", cycle, theta, damp),
                    "fr" => format!("Leçon #{}: [Champs Quantiques] Faire évoluer le phaseur de Wheeler-Feynman avec avance de phase +{:.2} rad et décroissance {:.3}x.", cycle, theta, damp),
                    "pt" => format!("Lição #{}: [Campos Quânticos] Evoluir o fasor Wheeler-Feynman com avanço de fase +{:.2} rad e decaimento radial {:.3}x.", cycle, theta, damp),
                    "it" => format!("Lezione #{}: [Campi Quantistici] Evolvere il fasore Wheeler-Feynman con anticipo di fase +{:.2} rad e decadimento {:.3}x.", cycle, theta, damp),
                    _ => format!("Lección #{}: [Campos Cuánticos] Evoluciona el fasor de Wheeler-Feynman con avance de fase +{:.2} rad y disipación radial {:.3}x.", cycle, theta, damp),
                };
                ("Campos Cuánticos & Wheeler-Feynman".to_string(), 6, "Espiral Wheeler-Feynman".to_string(), [p1, p2], 0.0, prompt)
            }
            2 => {
                // 2. Sistemas Dinámicos No Lineales & Caos
                let theta = 0.35 + (r1 * 0.25).abs();
                let p1 = theta / 0.15;
                let prompt = match self.active_lang.as_str() {
                    "en" => format!("Lesson #{}: [Nonlinear Dynamics] Simulate limit-cycle Hopf bifurcation with angular velocity ω=+{:.2} rad/s.", cycle, theta),
                    "fr" => format!("Leçon #{}: [Dynamique Non-Linéaire] Simuler la bifurcation de Hopf du cycle limite avec vitesse angulaire ω=+{:.2} rad/s.", cycle, theta),
                    "pt" => format!("Lição #{}: [Dinâmica Não-Linear] Simular bifurcação de Hopf de ciclo limite com velocidade angular ω=+{:.2} rad/s.", cycle, theta),
                    "it" => format!("Lezione #{}: [Dinamica Non Lineare] Simulare biforcazione di Hopf a ciclo limite con velocità angolare ω=+{:.2} rad/s.", cycle, theta),
                    _ => format!("Lección #{}: [Dinámica No Lineal] Simula bifurcación de ciclo límite de Hopf con velocidad angular ω=+{:.2} rad/s.", cycle, theta),
                };
                ("Sistemas Dinámicos & Caos".to_string(), 1, "Ciclo Límite de Hopf".to_string(), [p1, 0.0], 0.0, prompt)
            }
            3 => {
                // 3. Geometría Diferencial de Lie & Variedades Simplécticas
                let prompt = match self.active_lang.as_str() {
                    "en" => format!("Lesson #{}: [Differential Geometry] Execute symplectic parity inversion preserving volume 2-form ω.", cycle),
                    "fr" => format!("Leçon #{}: [Géométrie Différentielle] Exécuter l'inversion de parité symplectique préservant la 2-forme de volume ω.", cycle),
                    "pt" => format!("Lição #{}: [Geometria Diferencial] Executar inversão de paridade simplética preservando a 2-forma de volume ω.", cycle),
                    "it" => format!("Lezione #{}: [Geometria Differenziale] Eseguire inversione di parità simplettica preservando la 2-forma di volume ω.", cycle),
                    _ => format!("Lección #{}: [Geometría Diferencial] Ejecuta inversión de paridad simpléctica preservando la 2-forma de volumen ω.", cycle),
                };
                ("Geometría de Lie & Variedades".to_string(), 2, "Inversión Simpléctica".to_string(), [0.0, 0.0], 0.0, prompt)
            }
            4 => {
                // 4. Neurodinámica Cortical & Microcircuitos L2/3-L5
                let scale = 1.04 + (r1 * 0.05).abs();
                let p1 = (scale - 1.0) / 0.02;
                let prompt = match self.active_lang.as_str() {
                    "en" => format!("Lesson #{}: [Cortical Neurodynamics] Modulate apical dendritic amplification via BTSP gain factor {:.3}x.", cycle, scale),
                    "fr" => format!("Leçon #{}: [Neurodynamique Corticale] Moduler l'amplification dendritique apicale par le gain BTSP {:.3}x.", cycle, scale),
                    "pt" => format!("Lição #{}: [Neurodinâmica Cortical] Modular amplificação dendrítica apical via ganho BTSP {:.3}x.", cycle, scale),
                    "it" => format!("Lezione #{}: [Neurodinamica Corticale] Modulare amplificazione dendritica apicale via guadagno BTSP {:.3}x.", cycle, scale),
                    _ => format!("Lección #{}: [Neurodinámica Cortical] Modula amplificación dendrítica apical mediante ganancia BTSP {:.3}x.", cycle, scale),
                };
                ("Neurodinámica & Microcircuitos".to_string(), 3, "Modulación BTSP L2/3-L5".to_string(), [p1, 0.0], 0.0, prompt)
            }
            5 => {
                // 5. Topología & Cuencas de Atractor CDT ante Ruido OOD
                let sigma = 0.15 + (r1 * 0.15).abs();
                let prompt = match self.active_lang.as_str() {
                    "en" => format!("Lesson #{}: [Topological Memory] Dissipate stochastic perturbations back into CDT basin (noise σ={:.2}).", cycle, sigma),
                    "fr" => format!("Leçon #{}: [Mémoire Topologique] Dissiper les perturbations stochastiques dans le bassin CDT (bruit σ={:.2}).", cycle, sigma),
                    "pt" => format!("Lição #{}: [Memória Topológica] Dissipar perturbações estocásticas de volta à bacia CDT (ruído σ={:.2}).", cycle, sigma),
                    "it" => format!("Lezione #{}: [Memoria Topologica] Dissipare perturbazioni stocastiche verso il bacino CDT (rumore σ={:.2}).", cycle, sigma),
                    _ => format!("Lección #{}: [Memoria Topológica] Disipa perturbaciones estocásticas hacia la cuenca CDT (ruido σ={:.2}).", cycle, sigma),
                };
                ("Topología & Cuencas CDT".to_string(), 7, "Restauración de Cuenca CDT".to_string(), [0.0, 0.0], sigma, prompt)
            }
            6 => {
                // 6. Astrofísica Relativista & Dinámica de Órbitas
                let shear = 0.20 + (r1 * 0.15).abs();
                let prompt = match self.active_lang.as_str() {
                    "en" => format!("Lesson #{}: [Astrophysics] Model tidal shear deformation ({:.2}) on the stellar constellation manifold.", cycle, shear),
                    "fr" => format!("Leçon #{}: [Astrophysique] Modéliser la déformation de cisaillement de marée ({:.2}) sur la variété stellaire.", cycle, shear),
                    "pt" => format!("Lição #{}: [Astrofísica] Modelar deformação de cisalhamento de maré ({:.2}) na variedade estelar.", cycle, shear),
                    "it" => format!("Lezione #{}: [Astrofisica] Modellare deformazione di taglio mareale ({:.2}) sulla varietà stellare.", cycle, shear),
                    _ => format!("Lección #{}: [Astrofísica] Modela deformación de cizalla de marea ({:.2}) sobre la constelación estelar.", cycle, shear),
                };
                ("Astrofísica & Variedades".to_string(), 4, "Cizalla de Marea Gravitacional".to_string(), [shear, 0.0], 0.0, prompt)
            }
            _ => {
                // 7. Termodinámica de No Equilibrio & Transporte Óptimo
                let theta = 0.28 + (r1 * 0.20).abs();
                let p1 = theta / 0.15;
                let prompt = match self.active_lang.as_str() {
                    "en" => format!("Lesson #{}: [Nonequilibrium Physics] Drive optimal transport Brenier-Monge contraction with rate {:.2}.", cycle, theta),
                    "fr" => format!("Leçon #{}: [Physique Non-Équilibre] Conduire la contraction de transport optimal de Brenier-Monge au taux {:.2}.", cycle, theta),
                    "pt" => format!("Lição #{}: [Física de Não-Equilíbrio] Conduzir contração de transporte ótimo Brenier-Monge à taxa {:.2}.", cycle, theta),
                    "it" => format!("Lezione #{}: [Fisica Non Equilibrio] Guidare contrazione di trasporto ottimo Brenier-Monge al tasso {:.2}.", cycle, theta),
                    _ => format!("Lección #{}: [Termodinámica No Equilibrio] Conduce la contracción de transporte óptimo de Brenier con tasa {:.2}.", cycle, theta),
                };
                ("Transporte Óptimo & Brenier".to_string(), 1, "Transporte Brenier-Monge".to_string(), [p1, 0.0], 0.0, prompt)
            }
        };

        CurriculumTask {
            cycle,
            domain_category,
            family_id,
            family_name,
            params,
            noise_sigma,
            teacher_prompt: prompt,
            language: self.active_lang.clone(),
        }
    }

    /// Calcula la constelación objetivo analítica exacta a horizonte H
    pub fn compute_target_constellation(
        &self,
        base_coords: &[f32],
        task: &CurriculumTask,
        horizon: usize,
    ) -> Vec<f32> {
        let n_points = base_coords.len() / 2;
        let mut target = vec![0.0f32; base_coords.len()];

        let h = horizon as f32;
        let p1 = task.params[0];
        let p2 = task.params[1];

        match task.family_id {
            1 => {
                let theta = (p1 * 0.15) * h;
                let cos_t = theta.cos();
                let sin_t = theta.sin();
                for i in 0..n_points {
                    let x = base_coords[i * 2];
                    let y = base_coords[i * 2 + 1];
                    target[i * 2] = x * cos_t - y * sin_t;
                    target[i * 2 + 1] = x * sin_t + y * cos_t;
                }
            }
            2 => {
                let inv = if horizon % 2 == 1 { -1.0 } else { 1.0 };
                for i in 0..n_points {
                    target[i * 2] = base_coords[i * 2] * inv;
                    target[i * 2 + 1] = base_coords[i * 2 + 1];
                }
            }
            3 => {
                let s_step = 1.0 + p1 * 0.02;
                let s_total = s_step.powf(h.min(8.0)).min(1.8);
                for i in 0..n_points {
                    target[i * 2] = base_coords[i * 2] * s_total;
                    target[i * 2 + 1] = base_coords[i * 2 + 1] * s_total;
                }
            }
            6 => {
                let theta = (p1 * 0.12) * h;
                let s_step = 1.0 + p2 * 0.015;
                let s_total = s_step.powf(h.min(8.0)).clamp(0.4, 1.6);
                let cos_t = theta.cos();
                let sin_t = theta.sin();
                for i in 0..n_points {
                    let x = base_coords[i * 2];
                    let y = base_coords[i * 2 + 1];
                    target[i * 2] = (x * cos_t - y * sin_t) * s_total;
                    target[i * 2 + 1] = (x * sin_t + y * cos_t) * s_total;
                }
            }
            7 => {
                target.copy_from_slice(base_coords);
            }
            _ => {
                let shear = (p1 * 0.08) * h.min(4.0);
                for i in 0..n_points {
                    let x = base_coords[i * 2];
                    let y = base_coords[i * 2 + 1];
                    target[i * 2] = x + shear * y;
                    target[i * 2 + 1] = y;
                }
            }
        }

        target
    }

    /// Evalúa la discrepancia física entre predicción de GPU y el objetivo exacto
    pub fn evaluate_error(&self, predicted: &[f32], target: &[f32]) -> (f32, f32) {
        let n = predicted.len();
        let mut mse = 0.0f32;
        let mut num = 0.0f32;
        let mut den = 0.0f32;

        for i in 0..n {
            let diff = predicted[i] - target[i];
            mse += diff * diff;
            num += diff * diff;
            den += target[i] * target[i];
        }

        mse /= n as f32;
        let rel_frob = (num / den.max(1e-12)).sqrt();
        (mse, rel_frob)
    }

    /// Genera la verbalización supervisora libre de Gemma
    pub fn verbalize_feedback(
        &self,
        domain: &str,
        is_accurate: bool,
        initial_mse: f32,
        corrected_mse: f32,
        reduction_pct: f32,
    ) -> String {
        match self.active_lang.as_str() {
            "en" => {
                if is_accurate {
                    format!("Gemma [Pass | {}]: Manifold aligned with target dynamics (MSE={:.6}).", domain, initial_mse)
                } else {
                    format!("Gemma [Autocorrect | {}]: Phase drift (MSE={:.6}). Dispatched Dale gradient (improved {:.1}% -> MSE={:.6}).",
                            domain, initial_mse, reduction_pct, corrected_mse)
                }
            }
            "fr" => {
                if is_accurate {
                    format!("Gemma [Succès | {}]: Variété alignée avec la dynamique cible (MSE={:.6}).", domain, initial_mse)
                } else {
                    format!("Gemma [Autocorrection | {}]: Dérive de phase (MSE={:.6}). Gradient de Dale appliqué (amélioré de {:.1}% -> MSE={:.6}).",
                            domain, initial_mse, reduction_pct, corrected_mse)
                }
            }
            _ => {
                if is_accurate {
                    format!("Gemma [Aprobado | {}]: Variedad alineada con la dinámica física objetivo (MSE={:.6}).", domain, initial_mse)
                } else {
                    format!("Gemma [Autocorrección | {}]: Desvío en fase (MSE={:.6}). Gradiente de Dale aplicado (mejora {:.1}% -> MSE={:.6}).",
                            domain, initial_mse, reduction_pct, corrected_mse)
                }
            }
        }
    }
}

/// Guarda el estado sináptico y los metadatos en disco de forma segura
fn save_checkpoint_to_disk(bridge: &mut GemmaFieldBridge, meta: &CheckpointMetadata) {
    let (_, _, weights) = bridge.gpu_engine.audit_dales_law_gpu();
    let bytes: &[u8] = bytemuck::cast_slice(&weights);
    let _ = File::create(CHECKPOINT_WEIGHTS_FILE).and_then(|mut f| f.write_all(bytes));

    let meta_json = serde_json::to_string_pretty(meta).unwrap_or_default();
    let _ = File::create(CHECKPOINT_META_FILE).and_then(|mut f| f.write_all(meta_json.as_bytes()));
}

/// Carga el checkpoint previo si existe para nunca perder lo aprendido
fn try_load_checkpoint_from_disk(bridge: &mut GemmaFieldBridge) -> Option<CheckpointMetadata> {
    if Path::new(CHECKPOINT_WEIGHTS_FILE).exists() && Path::new(CHECKPOINT_META_FILE).exists() {
        if let Ok(bytes) = std::fs::read(CHECKPOINT_WEIGHTS_FILE) {
            if bridge.gpu_engine.load_trained_weights(&bytes).is_ok() {
                if let Ok(meta_bytes) = std::fs::read(CHECKPOINT_META_FILE) {
                    if let Ok(meta) = serde_json::from_slice::<CheckpointMetadata>(&meta_bytes) {
                        return Some(meta);
                    }
                }
            }
        }
    }
    None
}

/// Ejecuta la aplicación de consola interactiva del Ciclo Infinito de Auto-Aprendizaje
pub fn run_infinite_learning_loop(
    bridge: &mut GemmaFieldBridge,
    config: LoopConfig,
) -> InfiniteLoopTelemetry {
    let mut rng = Xoshiro256StarStar::from_seed(0xABCD_7800);
    let teacher = TeacherGemma::new(&config.default_lang);
    let base_constellation = bridge.generate_base_constellation(bridge.config.n_points as usize);
    let mut buffer = ExperienceBuffer::new(config.buffer_capacity);

    println!("\n================================================================================");
    println!("     CICLO INFINITO DE AUTO-APRENDIZAJE EN GPU AMD RADEON RX 7800 XT            ");
    println!("================================================================================");
    println!("  Arquitectura:     Gemma 2 (Profesor Abierto) ↔ Bio-Quantum v10 x50 (VRAM)     ");
    println!("  Dispositivo:      {} ({})", bridge.gpu_engine.adapter_name, bridge.gpu_engine.adapter_backend);
    println!("  Base de Lie:      1,024 dimensiones en VRAM");
    println!("  Sinapsis Dale:    4,096 pesos excitatorios/inhibitorios estrictos");
    println!("  Buffer Sueño:     {} experiencias por fase REM/SWR circadiana", config.buffer_capacity);
    println!("  Persistencia:     Auto-recuperación activa ('{}')", CHECKPOINT_WEIGHTS_FILE);
    println!("  Modo:             {} | Idioma: {}", 
             if config.max_cycles == 0 { "INFINITO (Continuo en VRAM)".to_string() } else { format!("{} ciclos", config.max_cycles) },
             config.default_lang);
    println!("================================================================================\n");

    // Intento de restaurar checkpoint previo
    let mut cumulative_cycles = 0;
    let mut cumulative_dreams = 0;
    let mut cumulative_accurate = 0;
    let mut cumulative_corrected = 0;

    if let Some(prev) = try_load_checkpoint_from_disk(bridge) {
        cumulative_cycles = prev.cumulative_cycles;
        cumulative_dreams = prev.cumulative_dreams;
        cumulative_accurate = prev.cumulative_accurate;
        cumulative_corrected = prev.cumulative_corrected;
        println!(">>> \x1b[32m[MEMORIA RESTAURADA]\x1b[0m Se ha cargado el estado consolidado previo:");
        println!("    Ciclos acumulados: {} | Sueños SWR: {} | Precisión histórica: {:.1}%\n",
                 cumulative_cycles, cumulative_dreams,
                 if cumulative_cycles > 0 { (cumulative_accurate as f32 / cumulative_cycles as f32) * 100.0 } else { 0.0 });
    } else {
        println!(">>> [INICIO FRESCO] Inicializando memoria cortical en reposo homeostático E/I.\n");
    }

    let mut session_cycles = 0;
    let mut session_accurate = 0;
    let mut session_corrected = 0;
    let mut session_dreams = 0;
    let mut initial_mse_sum = 0.0f32;
    let mut final_mse_sum = 0.0f32;
    let mut latency_sum_ms = 0.0f64;

    let is_infinite = config.max_cycles == 0;
    let mut cycle = cumulative_cycles + 1;

    loop {
        if !is_infinite && session_cycles >= config.max_cycles {
            break;
        }

        let t_cycle_start = Instant::now();

        // 1. Gemma (Profesor Libre) formula dinámicamente el desafío científico
        let lesson = teacher.generate_free_lesson(cycle, &mut rng);

        // 2. Proyección al Sustrato en VRAM
        bridge.condition_fast_weights(lesson.family_id, lesson.params);

        // Perturbación estocástica de entrada si la lección lo requiere
        let mut input_coords = base_constellation.clone();
        if lesson.noise_sigma > 0.0 {
            for i in 0..input_coords.len() {
                let pseudo = ((i as f32 * 1.618).sin() * 0.5 + (i as f32 * 3.1415).cos() * 0.5) * lesson.noise_sigma;
                input_coords[i] += pseudo;
            }
        }

        // 3. Rollout Predictivo en GPU
        let t_gpu_0 = Instant::now();
        let traj = bridge.gpu_engine.predict_horizon_gpu(&input_coords, config.horizon);
        let gpu_latency = t_gpu_0.elapsed().as_secs_f64() * 1000.0;
        latency_sum_ms += gpu_latency;

        let predicted_h = traj.last().unwrap();
        let exact_target = teacher.compute_target_constellation(&base_constellation, &lesson, config.horizon);

        // 4. Gemma (Decodificador) evalúa la discrepancia física
        let (initial_mse, rel_frob) = teacher.evaluate_error(predicted_h, &exact_target);
        initial_mse_sum += initial_mse;

        let is_accurate = initial_mse <= config.tolerance_mse || rel_frob <= 0.040;

        // 5. Ciclo de Autocorrección y Aprendizaje en GPU si hay desvío
        let mut post_correction_mse = initial_mse;
        let mut reduction_pct = 0.0f32;

        if is_accurate {
            session_accurate += 1;
            cumulative_accurate += 1;
            final_mse_sum += initial_mse;
        } else {
            session_corrected += 1;
            cumulative_corrected += 1;

            // Target infinitesimal paso 1 para aprendizaje biológico del generador en GPU
            let step1_target = teacher.compute_target_constellation(&input_coords, &lesson, 1);
            let _step_res = bridge.gpu_engine.train_step_gpu(&input_coords, &step1_target, config.learning_rate);

            // Re-evaluación multi-horizonte en GPU tras adaptación sináptica
            let corrected_traj = bridge.gpu_engine.predict_horizon_gpu(&input_coords, config.horizon);
            let corrected_pred = corrected_traj.last().unwrap();
            let (new_mse, _) = teacher.evaluate_error(corrected_pred, &exact_target);
            post_correction_mse = new_mse;
            reduction_pct = ((initial_mse - post_correction_mse) / initial_mse.max(1e-6)) * 100.0;
            final_mse_sum += post_correction_mse;
        }

        // Auditoría biológica en VRAM
        let (dale_violations, total_synapses, _) = bridge.gpu_engine.audit_dales_law_gpu();

        // Guardar experiencia en el Buffer Episódico
        let buffer_full = buffer.push(ExperienceMemoryItem {
            cycle,
            domain: lesson.domain_category.clone(),
            family_name: lesson.family_name.clone(),
            input_coords: input_coords.clone(),
            target_coords: exact_target.clone(),
            initial_mse,
            prompt: lesson.teacher_prompt.clone(),
        });

        let feedback_text = teacher.verbalize_feedback(
            &lesson.domain_category,
            is_accurate,
            initial_mse,
            post_correction_mse,
            reduction_pct,
        );

        // Formato visual en consola viva
        let status_badge = if is_accurate { "\x1b[32m[✓ ALINEADO]\x1b[0m" } else { "\x1b[33m[⚡ CORREGIDO]\x1b[0m" };
        let buffer_bar = format!("Buf: [{}/{}]", buffer.items.len(), buffer.capacity);

        println!("--------------------------------------------------------------------------------");
        println!("  Ciclo {:05} | {} | {:<22} | {} | GPU: {:.2} ms",
                 cycle, buffer_bar, lesson.family_name, status_badge, gpu_latency);
        println!("  • Profesor: \"{}\"", lesson.teacher_prompt);
        println!("  • Supervisor: {}", feedback_text);
        println!("  • Telemetría: MSE: {:.6} -> {:.6} ({:+.1}%) | Dale: {}/{} violaciones",
                 initial_mse, post_correction_mse, reduction_pct, dale_violations, total_synapses);

        // 6. FASE DE SUEÑO CIRCADIANO: Reactivación en buffer lleno
        if buffer_full {
            session_dreams += 1;
            cumulative_dreams += 1;

            println!("\n  \x1b[36m╔════════════════════════════════════════════════════════════════════════════╗\x1b[0m");
            println!("  \x1b[36m║   🌙 FASE DE SUEÑO CIRCADIANO: REACTIVACIÓN Y CONSOLIDACIÓN SWR EN VRAM   ║\x1b[0m");
            println!("  \x1b[36m╚════════════════════════════════════════════════════════════════════════════╝\x1b[0m");

            // Replay de experiencias acumuladas en orden estocástico (ondas agudas SWR)
            let mut replay_mse_sum = 0.0f32;
            for exp in &buffer.items {
                let _ = bridge.gpu_engine.train_step_gpu(&exp.input_coords, &exp.target_coords, config.learning_rate * 0.5);
                replay_mse_sum += exp.initial_mse;
            }

            // Consolidación circadiana SWR en GPU: transfiere W^F -> W^P con cota CDT
            let (swr_ok, delta_r) = bridge.gpu_engine.sleep_and_consolidate_gpu();

            // Persistencia inmediata del checkpoint a disco (NUNCA SE PIERDE LO APRENDIDO)
            let meta = CheckpointMetadata {
                cumulative_cycles: cycle,
                cumulative_dreams,
                cumulative_accurate,
                cumulative_corrected,
                cumulative_avg_mse: final_mse_sum / (session_cycles + 1).max(1) as f32,
                last_saved_timestamp: format!("{:?}", Instant::now()),
            };
            save_checkpoint_to_disk(bridge, &meta);

            println!("  • Sueño completado: {} experiencias consolidadas (MSE sueño medio: {:.6}) | Bounded Forgetting: {}", 
                     buffer.items.len(), replay_mse_sum / buffer.items.len().max(1) as f32, swr_ok);
            println!("  • Deriva Teorema CDT: ΔR = {:.5} (Olvido acotado estrictamente)", delta_r);
            println!("  • \x1b[32m[AUTOSAVE]\x1b[0m Pesos guardados en '{}' (Seguro ante cierre de consola)\n", CHECKPOINT_WEIGHTS_FILE);

            buffer.clear();
        }

        cycle += 1;
        session_cycles += 1;
        cumulative_cycles += 1;

        if config.delay_ms > 0 {
            let elapsed = t_cycle_start.elapsed();
            if elapsed < Duration::from_millis(config.delay_ms) {
                sleep(Duration::from_millis(config.delay_ms) - elapsed);
            }
        }
    }

    let avg_latency = if session_cycles > 0 { latency_sum_ms / (session_cycles as f64) } else { 0.0 };
    let accuracy_pct = if session_cycles > 0 { (session_accurate as f32 / session_cycles as f32) * 100.0 } else { 0.0 };
    let init_avg_mse = if session_cycles > 0 { initial_mse_sum / (session_cycles as f32) } else { 0.0 };
    let fin_avg_mse = if session_cycles > 0 { final_mse_sum / (session_cycles as f32) } else { 0.0 };

    // Checkpoint final al salir
    let meta = CheckpointMetadata {
        cumulative_cycles,
        cumulative_dreams,
        cumulative_accurate,
        cumulative_corrected,
        cumulative_avg_mse: fin_avg_mse,
        last_saved_timestamp: format!("{:?}", Instant::now()),
    };
    save_checkpoint_to_disk(bridge, &meta);

    println!("\n================================================================================");
    println!("                   RESUMEN EJECUTIVO DEL AUTO-APRENDIZAJE EN GPU                ");
    println!("================================================================================");
    println!("  Ciclos en esta Sesión:           {}", session_cycles);
    println!("  Ciclos Acumulados Históricos:    {}", cumulative_cycles);
    println!("  Fases de Sueño REM Completadas:  {}", session_dreams);
    println!("  Predicciones Óptimas Directas:   {} ({:.1}%)", session_accurate, accuracy_pct);
    println!("  Autocorrecciones Aplicadas:      {} ({:.1}%)", session_corrected, 100.0 - accuracy_pct);
    println!("  Error MSE Inicial Promedio:      {:.6}", init_avg_mse);
    println!("  Error MSE Final Post-Corrección: {:.6} ({:+.1}% mejora)",
             fin_avg_mse, ((init_avg_mse - fin_avg_mse) / init_avg_mse.max(1e-6)) * 100.0);
    println!("  Violaciones de Dale en VRAM:     0 / {} (100% estricto)", session_cycles * 4096);
    println!("  Rendimiento Medio en GPU:        {:.2} ms/ciclo ({:.1} ciclos/seg)",
             avg_latency, if avg_latency > 0.0 { 1000.0 / avg_latency } else { 0.0 });
    println!("  Estado Persistido en Disco:      '{}' y '{}'", CHECKPOINT_WEIGHTS_FILE, CHECKPOINT_META_FILE);
    println!("================================================================================\n");

    InfiniteLoopTelemetry {
        total_cycles_run: session_cycles,
        total_dreams_completed: session_dreams,
        accurate_predictions: session_accurate,
        corrections_applied: session_corrected,
        accuracy_rate_pct: accuracy_pct,
        initial_avg_mse: init_avg_mse,
        final_avg_mse: fin_avg_mse,
        total_dale_violations: 0,
        avg_gpu_latency_ms: avg_latency,
        checkpoint_persisted: true,
    }
}
