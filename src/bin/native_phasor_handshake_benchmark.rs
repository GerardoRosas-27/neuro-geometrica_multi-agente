//! Experimento aislado de inferencia fasorial con una o dos fronteras.
//!
//! No modela medición cuántica ni demuestra retrocausalidad. Compara, sobre el
//! mismo grafo estratificado, dos algoritmos clásicos:
//! - `forward_valley`: propagación compleja y poda por intensidad local;
//! - `handshake`: la misma propagación, guiada por un mensaje conjugado que
//!   parte de una condición de frontera final conocida.

use cdt_rqm_epr::native_rng::{splitmix64, unit_from_u64};
use num_complex::Complex32;
use std::hint::black_box;
use std::time::{Duration, Instant};

const DEFAULT_TRIALS: usize = 96;
const DEFAULT_WIDTH: usize = 256;
const DEFAULT_DEPTH: usize = 10;
const DEFAULT_DEGREE: usize = 8;
const DEFAULT_BEAM: usize = 16;
const TIMING_REPEATS: usize = 8;
const EPSILON: f32 = 1.0e-12;

#[derive(Clone, Copy, Debug)]
struct Edge {
    to: usize,
    transport: Complex32,
}

#[derive(Clone, Debug)]
struct Problem {
    width: usize,
    edges: Vec<Vec<Vec<Edge>>>,
    planted: Vec<usize>,
}

#[derive(Clone, Copy, Debug, Default)]
struct InferenceResult {
    goal_probability: f32,
    goal_rank: usize,
    planted_retention: f32,
    goal_is_top1: bool,
    multiply_adds: usize,
    scored_nodes: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct Aggregate {
    goal_probability: f64,
    reciprocal_rank: f64,
    planted_retention: f64,
    top1_goals: usize,
    multiply_adds: usize,
    scored_nodes: usize,
    elapsed: Duration,
    cases: usize,
}

impl Aggregate {
    fn record(&mut self, result: InferenceResult, elapsed: Duration) {
        self.goal_probability += f64::from(result.goal_probability);
        self.reciprocal_rank += 1.0 / result.goal_rank.max(1) as f64;
        self.planted_retention += f64::from(result.planted_retention);
        self.top1_goals += usize::from(result.goal_is_top1);
        self.multiply_adds += result.multiply_adds;
        self.scored_nodes += result.scored_nodes;
        self.elapsed += elapsed;
        self.cases += 1;
    }

    fn print(self, noise: f32, method: &str) {
        let cases = self.cases.max(1) as f64;
        println!(
            "{noise:.2},{method},{:.4},{:.4},{:.4},{:.4},{:.1},{:.1},{:.3}",
            self.goal_probability / cases,
            self.reciprocal_rank / cases,
            self.planted_retention / cases,
            self.top1_goals as f64 / cases,
            self.multiply_adds as f64 / cases,
            self.scored_nodes as f64 / cases,
            self.elapsed.as_secs_f64() * 1_000.0 / cases,
        );
    }
}

fn main() {
    let trials = env_usize("PHASOR_HANDSHAKE_TRIALS", DEFAULT_TRIALS);
    let width = env_usize("PHASOR_HANDSHAKE_WIDTH", DEFAULT_WIDTH).max(8);
    let depth = env_usize("PHASOR_HANDSHAKE_DEPTH", DEFAULT_DEPTH).max(2);
    let degree = env_usize("PHASOR_HANDSHAKE_DEGREE", DEFAULT_DEGREE).clamp(2, width);
    let beam = env_usize("PHASOR_HANDSHAKE_BEAM", DEFAULT_BEAM).clamp(1, width);

    println!("benchmark=inferencia_fasorial_forward_vs_handshake");
    println!("alcance=algoritmo_clasico_aislado no_colapso_cuantico no_retrocausalidad");
    println!(
        "config,trials={trials},width={width},depth={depth},degree={degree},beam={beam},timing_repeats={TIMING_REPEATS}"
    );
    println!(
        "noise,method,goal_probability,mrr,planted_retention,goal_top1_rate,\
         complex_madds,scored_nodes,mean_ms"
    );

    for noise in [0.15_f32, 0.35, 0.55, 0.75] {
        let mut baseline = Aggregate::default();
        let mut handshake_cold = Aggregate::default();
        let mut handshake_warm = Aggregate::default();

        for trial in 0..trials {
            let seed = 0x4841_4E44_5348_414B_u64
                ^ (trial as u64).rotate_left(23)
                ^ u64::from(noise.to_bits());
            let problem = generate_problem(width, depth, degree, noise, seed);

            let started = Instant::now();
            let mut baseline_result = InferenceResult::default();
            for _ in 0..TIMING_REPEATS {
                baseline_result = black_box(forward_valley(&problem, beam));
            }
            baseline.record(
                baseline_result,
                started.elapsed().div_f64(TIMING_REPEATS as f64),
            );

            let started = Instant::now();
            let (backward, backward_ops) = backward_boundary(&problem);
            let mut cold_result = handshake(&problem, &backward, beam);
            cold_result.multiply_adds += backward_ops;
            let cold_elapsed = started.elapsed();
            handshake_cold.record(cold_result, cold_elapsed);

            let started = Instant::now();
            let mut warm_result = InferenceResult::default();
            for _ in 0..TIMING_REPEATS {
                warm_result = black_box(handshake(&problem, black_box(&backward), beam));
            }
            handshake_warm.record(
                warm_result,
                started.elapsed().div_f64(TIMING_REPEATS as f64),
            );
        }

        baseline.print(noise, "forward_valley");
        handshake_cold.print(noise, "handshake_cold");
        handshake_warm.print(noise, "handshake_cached_boundary");
        println!(
            "delta,noise={noise:.2},goal_probability={:+.4},goal_top1_rate={:+.4},\
             cold_madds_ratio={:.3},warm_madds_ratio={:.3}",
            mean_goal_probability(handshake_cold) - mean_goal_probability(baseline),
            mean_top1_rate(handshake_cold) - mean_top1_rate(baseline),
            mean_madds(handshake_cold) / mean_madds(baseline).max(1.0),
            mean_madds(handshake_warm) / mean_madds(baseline).max(1.0),
        );
    }
}

fn forward_valley(problem: &Problem, beam: usize) -> InferenceResult {
    infer(problem, None, beam)
}

fn handshake(problem: &Problem, backward: &[Vec<Complex32>], beam: usize) -> InferenceResult {
    infer(problem, Some(backward), beam)
}

fn infer(problem: &Problem, backward: Option<&[Vec<Complex32>]>, beam: usize) -> InferenceResult {
    let mut current = vec![Complex32::new(0.0, 0.0); problem.width];
    current[problem.planted[0]] = Complex32::new(1.0, 0.0);
    let mut active = vec![false; problem.width];
    active[problem.planted[0]] = true;
    let mut multiply_adds = 0usize;
    let mut scored_nodes = 0usize;
    let mut retained_layers = 1usize;

    for layer in 0..problem.edges.len() {
        let mut next = vec![Complex32::new(0.0, 0.0); problem.width];
        for from in 0..problem.width {
            if !active[from] {
                continue;
            }
            for edge in &problem.edges[layer][from] {
                next[edge.to] += current[from] * edge.transport;
                multiply_adds += 1;
            }
        }

        let scores = (0..problem.width)
            .map(|node| {
                let score = match backward {
                    Some(messages) => transaction_score(next[node], messages[layer + 1][node]),
                    None => next[node].norm_sqr(),
                };
                (node, score)
            })
            .collect::<Vec<_>>();
        scored_nodes += problem.width;
        active.fill(false);
        for (node, _) in top_k(scores, beam) {
            active[node] = true;
        }
        for node in 0..problem.width {
            if !active[node] {
                next[node] = Complex32::new(0.0, 0.0);
            }
        }

        let planted_is_active = active[problem.planted[layer + 1]];
        retained_layers += usize::from(planted_is_active);
        current = next;
    }

    let goal = problem.planted[problem.planted.len() - 1];
    let total_intensity = current.iter().map(|value| value.norm_sqr()).sum::<f32>();
    let goal_intensity = current[goal].norm_sqr();
    let goal_rank = 1 + current
        .iter()
        .enumerate()
        .filter(|(node, value)| *node != goal && value.norm_sqr() > goal_intensity)
        .count();
    InferenceResult {
        goal_probability: goal_intensity / total_intensity.max(EPSILON),
        goal_rank,
        planted_retention: retained_layers as f32 / problem.planted.len() as f32,
        goal_is_top1: goal_rank == 1,
        multiply_adds,
        scored_nodes,
    }
}

fn backward_boundary(problem: &Problem) -> (Vec<Vec<Complex32>>, usize) {
    let layers = problem.edges.len() + 1;
    let mut backward = vec![vec![Complex32::new(0.0, 0.0); problem.width]; layers];
    let goal = problem.planted[layers - 1];
    backward[layers - 1][goal] = Complex32::new(1.0, 0.0);
    let mut operations = 0usize;

    for layer in (0..problem.edges.len()).rev() {
        for from in 0..problem.width {
            let mut message = Complex32::new(0.0, 0.0);
            for edge in &problem.edges[layer][from] {
                message += edge.transport.conj() * backward[layer + 1][edge.to];
                operations += 1;
            }
            backward[layer][from] = message;
        }
        normalize(&mut backward[layer]);
    }
    (backward, operations)
}

fn transaction_score(forward: Complex32, backward: Complex32) -> f32 {
    // |f·b|² conserva la información de fase durante la propagación de ambos
    // mensajes, pero evita que una fase global arbitraria cambie el ranking.
    (forward * backward).norm_sqr()
}

fn generate_problem(
    width: usize,
    depth: usize,
    degree: usize,
    phase_noise: f32,
    seed: u64,
) -> Problem {
    let mut planted = vec![0usize; depth + 1];
    for (layer, node) in planted.iter_mut().enumerate().skip(1) {
        *node = (splitmix64(seed ^ (layer as u64).wrapping_mul(0x9E37_79B9)) as usize) % width;
    }

    let mut edges = Vec::with_capacity(depth);
    for layer in 0..depth {
        let mut layer_edges = vec![Vec::with_capacity(degree); width];
        for (from, outgoing) in layer_edges.iter_mut().enumerate() {
            let is_planted_source = from == planted[layer];
            if is_planted_source {
                let phase = symmetric_unit(seed, layer, from, 0) * phase_noise;
                outgoing.push(Edge {
                    to: planted[layer + 1],
                    transport: Complex32::from_polar(1.0, phase),
                });
            }
            let mut cursor = 0usize;
            while outgoing.len() < degree {
                let raw = splitmix64(
                    seed.rotate_left(17)
                        ^ (layer as u64).wrapping_mul(0xD1B5_4A32_D192_ED03)
                        ^ (from as u64).rotate_left(31)
                        ^ cursor as u64,
                );
                cursor += 1;
                let to = raw as usize % width;
                if outgoing.iter().any(|edge| edge.to == to) {
                    continue;
                }
                let amplitude = 0.55 + 0.55 * unit_from_u64(splitmix64(raw ^ 0xA11C_E5ED));
                let phase = std::f32::consts::TAU * unit_from_u64(splitmix64(raw ^ 0xFA53_C0DE));
                outgoing.push(Edge {
                    to,
                    transport: Complex32::from_polar(amplitude, phase),
                });
            }
        }
        edges.push(layer_edges);
    }
    Problem {
        width,
        edges,
        planted,
    }
}

fn top_k(mut scores: Vec<(usize, f32)>, limit: usize) -> Vec<(usize, f32)> {
    scores.sort_unstable_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    scores.truncate(limit.min(scores.len()));
    scores
}

fn normalize(values: &mut [Complex32]) {
    let norm = values
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f32>()
        .sqrt();
    if norm > EPSILON {
        for value in values {
            *value /= norm;
        }
    }
}

fn symmetric_unit(seed: u64, layer: usize, node: usize, salt: usize) -> f32 {
    2.0 * unit_from_u64(splitmix64(
        seed ^ (layer as u64).rotate_left(11) ^ (node as u64).rotate_left(37) ^ salt as u64,
    )) - 1.0
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn mean_goal_probability(metrics: Aggregate) -> f64 {
    metrics.goal_probability / metrics.cases.max(1) as f64
}

fn mean_top1_rate(metrics: Aggregate) -> f64 {
    metrics.top1_goals as f64 / metrics.cases.max(1) as f64
}

fn mean_madds(metrics: Aggregate) -> f64 {
    metrics.multiply_adds as f64 / metrics.cases.max(1) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backward_boundary_is_zero_off_goal_at_terminal_layer() {
        let problem = generate_problem(32, 4, 4, 0.2, 7);
        let (backward, _) = backward_boundary(&problem);
        let goal = problem.planted[4];
        assert_eq!(backward[4][goal], Complex32::new(1.0, 0.0));
        assert!(backward[4]
            .iter()
            .enumerate()
            .all(|(node, value)| node == goal || value.norm_sqr() == 0.0));
    }

    #[test]
    fn cached_handshake_does_not_count_backward_operations() {
        let problem = generate_problem(64, 5, 4, 0.3, 11);
        let (backward, backward_ops) = backward_boundary(&problem);
        let forward = forward_valley(&problem, 8);
        let guided = handshake(&problem, &backward, 8);
        assert!(backward_ops > 0);
        assert!(guided.multiply_adds <= forward.multiply_adds);
    }

    #[test]
    fn handshake_recovers_the_goal_on_low_noise_fixture() {
        let problem = generate_problem(64, 6, 5, 0.05, 19);
        let (backward, _) = backward_boundary(&problem);
        let guided = handshake(&problem, &backward, 8);
        assert_eq!(guided.goal_rank, 1, "{guided:?}");
        assert!(guided.goal_probability >= 0.90, "{guided:?}");
    }
}
