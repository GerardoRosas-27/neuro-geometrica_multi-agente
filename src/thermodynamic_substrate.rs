use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;

const EPSILON: f32 = 1.0e-6;

#[derive(Clone, Copy, Debug)]
pub struct ThermodynamicConfig {
    pub size: usize,
    pub temperature: f32,
    pub dt: f32,
    pub confinement: f32,
    pub initial_state_min: f32,
    pub initial_state_max: f32,
    pub state_clamp: f32,
    pub seed: u64,
}

impl Default for ThermodynamicConfig {
    fn default() -> Self {
        Self {
            size: 1_024,
            temperature: 1.0,
            dt: 0.01,
            confinement: 0.04,
            initial_state_min: -1.0,
            initial_state_max: 1.0,
            state_clamp: 5.0,
            seed: 0xC0DE_CAFE,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ThermodynamicStepReport {
    pub tick: u64,
    pub mean_state: f32,
    pub state_variance: f32,
    pub mean_energy: f32,
    pub free_energy_proxy: f32,
    pub boltzmann_partition_proxy: f32,
    pub max_abs_state: f32,
}

pub trait PilotForceField {
    fn write_forces(&self, forces: &mut [f32]);
}

#[derive(Clone, Debug)]
pub struct ThermodynamicSubstrate {
    pub config: ThermodynamicConfig,
    pub states: Vec<f32>,
    pub pilot_forces: Vec<f32>,
    pub local_temperatures: Vec<f32>,
    energies: Vec<f32>,
    rng: Xoshiro256PlusPlus,
    tick: u64,
}

impl ThermodynamicSubstrate {
    pub fn new(config: ThermodynamicConfig) -> Self {
        let size = config.size.max(1);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(config.seed);
        let mut states = Vec::with_capacity(size);
        for _ in 0..size {
            states.push(uniform_range(
                &mut rng,
                config.initial_state_min,
                config.initial_state_max,
            ));
        }

        Self {
            config: ThermodynamicConfig { size, ..config },
            states,
            pilot_forces: vec![0.0; size],
            local_temperatures: vec![config.temperature.max(0.0); size],
            energies: vec![0.0; size],
            rng,
            tick: 0,
        }
    }

    pub fn tick(&self) -> u64 {
        self.tick
    }

    pub fn len(&self) -> usize {
        self.states.len()
    }

    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    pub fn set_temperature(&mut self, temperature: f32) {
        self.config.temperature = temperature.max(0.0);
        self.local_temperatures.fill(self.config.temperature);
    }

    pub fn set_local_temperature(&mut self, index: usize, temperature: f32) -> bool {
        if let Some(slot) = self.local_temperatures.get_mut(index) {
            *slot = temperature.max(0.0);
            true
        } else {
            false
        }
    }

    pub fn set_pilot_force(&mut self, index: usize, force: f32) -> bool {
        if let Some(slot) = self.pilot_forces.get_mut(index) {
            *slot = force;
            true
        } else {
            false
        }
    }

    pub fn clear_pilot_forces(&mut self) {
        self.pilot_forces.fill(0.0);
    }

    pub fn apply_pilot_field(&mut self, field: &impl PilotForceField) {
        field.write_forces(&mut self.pilot_forces);
    }

    #[inline(always)]
    pub fn step_langevin(&mut self) -> ThermodynamicStepReport {
        let dt = self.config.dt.max(0.0);
        let confinement = self.config.confinement.max(0.0);
        let state_clamp = self.config.state_clamp.abs().max(EPSILON);
        let noise_seed = self.rng.gen::<u64>() ^ self.tick.rotate_left(17);

        self.states
            .par_iter_mut()
            .zip(self.energies.par_iter_mut())
            .zip(self.pilot_forces.par_iter())
            .zip(self.local_temperatures.par_iter())
            .enumerate()
            .for_each(|(i, (((state, energy), pilot_force), temperature))| {
                let noise_amplitude = (2.0 * temperature.max(0.0) * dt).sqrt();
                let thermal_noise = gaussian_from_counter(noise_seed, i as u64);
                let deterministic_force = *pilot_force - confinement * *state;
                let dx = deterministic_force * dt + thermal_noise * noise_amplitude;
                let next_state = (*state + dx).clamp(-state_clamp, state_clamp);

                *state = next_state;
                *energy = effective_energy(next_state, *pilot_force, confinement);
            });

        self.tick = self.tick.wrapping_add(1);
        self.report()
    }

    pub fn report(&self) -> ThermodynamicStepReport {
        let n = self.states.len().max(1) as f32;
        let mean_state = self.states.par_iter().copied().sum::<f32>() / n;
        let state_variance = self
            .states
            .par_iter()
            .map(|state| {
                let centered = *state - mean_state;
                centered * centered
            })
            .sum::<f32>()
            / n;
        let mean_energy = self.energies.par_iter().copied().sum::<f32>() / n;
        let max_abs_state = self
            .states
            .par_iter()
            .map(|state| state.abs())
            .reduce(|| 0.0, f32::max);
        let boltzmann_partition_proxy = self
            .energies
            .par_iter()
            .zip(self.local_temperatures.par_iter())
            .map(|(energy, temperature)| boltzmann_weight(*energy, *temperature))
            .sum::<f32>();
        let free_energy_proxy = if boltzmann_partition_proxy > EPSILON {
            -self.config.temperature.max(EPSILON) * boltzmann_partition_proxy.ln()
        } else {
            f32::INFINITY
        };

        ThermodynamicStepReport {
            tick: self.tick,
            mean_state,
            state_variance,
            mean_energy,
            free_energy_proxy,
            boltzmann_partition_proxy,
            max_abs_state,
        }
    }

    pub fn boltzmann_probabilities(&self) -> Vec<f32> {
        let weights = self
            .energies
            .par_iter()
            .zip(self.local_temperatures.par_iter())
            .map(|(energy, temperature)| boltzmann_weight(*energy, *temperature))
            .collect::<Vec<_>>();
        let partition = weights.par_iter().copied().sum::<f32>().max(EPSILON);
        weights
            .into_iter()
            .map(|weight| weight / partition)
            .collect()
    }

    pub fn has_stabilized(
        previous: ThermodynamicStepReport,
        current: ThermodynamicStepReport,
        energy_tolerance: f32,
        variance_tolerance: f32,
    ) -> bool {
        (current.mean_energy - previous.mean_energy).abs() <= energy_tolerance
            && (current.state_variance - previous.state_variance).abs() <= variance_tolerance
    }
}

#[inline(always)]
fn effective_energy(state: f32, pilot_force: f32, confinement: f32) -> f32 {
    0.5 * confinement * state * state - pilot_force * state
}

#[inline(always)]
fn boltzmann_weight(energy: f32, temperature: f32) -> f32 {
    (-energy / temperature.max(EPSILON)).exp()
}

#[inline(always)]
fn uniform_range(rng: &mut Xoshiro256PlusPlus, min: f32, max: f32) -> f32 {
    let (min, max) = if min <= max { (min, max) } else { (max, min) };
    let unit = rng.gen::<f32>();
    min + (max - min) * unit
}

#[inline(always)]
fn gaussian_from_counter(seed: u64, counter: u64) -> f32 {
    let base = seed ^ counter.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let sum = unit_from_u64(splitmix64(base))
        + unit_from_u64(splitmix64(base ^ 0xA24B_AED4_963E_E407))
        + unit_from_u64(splitmix64(base ^ 0x9FB2_1C65_1E98_DF25))
        + unit_from_u64(splitmix64(base ^ 0xC13F_A9A9_02A6_328F))
        + unit_from_u64(splitmix64(base ^ 0x91E1_0DA5_C79E_7B1D))
        + unit_from_u64(splitmix64(base ^ 0xD1B5_4A32_D192_ED03));
    sum - 3.0
}

#[inline(always)]
fn unit_from_u64(value: u64) -> f32 {
    ((value >> 40) as f32) * (1.0 / (1_u32 << 24) as f32)
}

#[inline(always)]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct ConstantPilot(f32);

    impl PilotForceField for ConstantPilot {
        fn write_forces(&self, forces: &mut [f32]) {
            forces.fill(self.0);
        }
    }

    #[test]
    fn langevin_step_keeps_buffers_aligned() {
        let mut substrate = ThermodynamicSubstrate::new(ThermodynamicConfig {
            size: 64,
            temperature: 0.5,
            dt: 0.01,
            ..ThermodynamicConfig::default()
        });

        substrate.apply_pilot_field(&ConstantPilot(0.2));
        let report = substrate.step_langevin();

        assert_eq!(substrate.len(), 64);
        assert_eq!(substrate.pilot_forces.len(), 64);
        assert_eq!(substrate.local_temperatures.len(), 64);
        assert_eq!(report.tick, 1);
        assert!(report.max_abs_state <= substrate.config.state_clamp);
    }

    #[test]
    fn boltzmann_probabilities_are_normalized() {
        let mut substrate = ThermodynamicSubstrate::new(ThermodynamicConfig {
            size: 32,
            temperature: 1.0,
            dt: 0.0,
            ..ThermodynamicConfig::default()
        });

        substrate.step_langevin();
        let total = substrate.boltzmann_probabilities().into_iter().sum::<f32>();

        assert!((total - 1.0).abs() < 1.0e-4);
    }
}
