//! Adaptador pyrochlore → MPO de largo alcance para `oxicuda-tn`.

use crate::quantum_spin_thermodynamic_engine::QuantumSpinBond;
use oxicuda_tn::dmrg::dmrg::{dmrg_two_site, DmrgConfig, DmrgResult};
use oxicuda_tn::error::TnResult;
use oxicuda_tn::handle::LcgRng;
use oxicuda_tn::mpo::auto_compress::{mpo_compress, MpoCompressConfig, MpoData};
use oxicuda_tn::mpo::mpo::{Mpo, MpoTensor};
use oxicuda_tn::mps::mps::Mps;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug)]
pub struct PyrochloreMpoConfig {
    pub exchange: f64,
    pub anisotropy: f64,
    pub initial_bond_dimension: usize,
    pub compress_max_bond: usize,
    pub compress_tolerance: f64,
    pub dmrg: DmrgConfig,
    pub seed: u64,
}

impl Default for PyrochloreMpoConfig {
    fn default() -> Self {
        Self {
            exchange: 1.0,
            anisotropy: 1.0,
            initial_bond_dimension: 8,
            compress_max_bond: 64,
            compress_tolerance: 1.0e-12,
            dmrg: DmrgConfig {
                max_sweeps: 16,
                chi_max: 64,
                trunc_tol: 1.0e-10,
                energy_tol: 1.0e-9,
                lanczos_iter: 48,
                lanczos_tol: 1.0e-10,
            },
            seed: 0x0A1C_DA7A_2026_0718,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PyrochloreDmrgReport {
    pub sites: usize,
    pub physical_bonds: usize,
    pub initial_mpo_bond_dimension: usize,
    pub mpo_bond_dimension: usize,
    pub energy: f64,
    pub sweeps: usize,
    pub final_norm: f64,
    pub max_mps_bond_dimension: usize,
    pub midpoint_entropy: f64,
}

#[derive(Clone, Copy)]
struct TwoSiteTerm {
    right_site: usize,
    channel: usize,
    right_operator: [f64; 4],
    coefficient: f64,
}

pub fn build_pyrochlore_mpo(
    sites: usize,
    bonds: &[QuantumSpinBond],
    exchange: f64,
    anisotropy: f64,
) -> TnResult<Mpo> {
    let identity = [1.0, 0.0, 0.0, 1.0];
    let raising = [0.0, 1.0, 0.0, 0.0];
    let lowering = [0.0, 0.0, 1.0, 0.0];
    let spin_z = [0.5, 0.0, 0.0, -0.5];
    let mut terms = Vec::with_capacity(bonds.len() * 3);
    let mut channels = BTreeMap::<(usize, u8), usize>::new();
    for bond in bonds {
        let (left_site, right_site) = if bond.a <= bond.b {
            (bond.a, bond.b)
        } else {
            (bond.b, bond.a)
        };
        if left_site == right_site {
            continue;
        }
        let coupling = exchange * bond.multiplicity as f64;
        let raising_channel = next_channel(&mut channels, left_site, 0);
        let lowering_channel = next_channel(&mut channels, left_site, 1);
        let z_channel = next_channel(&mut channels, left_site, 2);
        terms.push(TwoSiteTerm {
            right_site,
            channel: raising_channel,
            right_operator: lowering,
            coefficient: coupling / 2.0,
        });
        terms.push(TwoSiteTerm {
            right_site,
            channel: lowering_channel,
            right_operator: raising,
            coefficient: coupling / 2.0,
        });
        terms.push(TwoSiteTerm {
            right_site,
            channel: z_channel,
            right_operator: spin_z,
            coefficient: coupling * anisotropy,
        });
    }
    let virtual_dimension = channels.len() + 2;
    let terminal = virtual_dimension - 1;
    let mut tensors = Vec::with_capacity(sites);
    for site in 0..sites {
        let left_dimension = if site == 0 { 1 } else { virtual_dimension };
        let right_dimension = if site + 1 == sites {
            1
        } else {
            virtual_dimension
        };
        let mut tensor = MpoTensor::zeros(left_dimension, 2, 2, right_dimension)?;
        add_global_operator(&mut tensor, site, sites, 0, 0, identity, 1.0, terminal)?;
        add_global_operator(
            &mut tensor,
            site,
            sites,
            terminal,
            terminal,
            identity,
            1.0,
            terminal,
        )?;
        for channel in 1..=channels.len() {
            add_global_operator(
                &mut tensor,
                site,
                sites,
                channel,
                channel,
                identity,
                1.0,
                terminal,
            )?;
        }
        for (&(left_site, kind), &channel) in &channels {
            if site == left_site {
                let operator = match kind {
                    0 => raising,
                    1 => lowering,
                    _ => spin_z,
                };
                add_global_operator(
                    &mut tensor,
                    site,
                    sites,
                    terminal,
                    channel,
                    operator,
                    1.0,
                    terminal,
                )?;
            }
        }
        for term in &terms {
            if site == term.right_site {
                add_global_operator(
                    &mut tensor,
                    site,
                    sites,
                    term.channel,
                    0,
                    term.right_operator,
                    term.coefficient,
                    terminal,
                )?;
            }
        }
        tensors.push(tensor);
    }
    Mpo::from_tensors(tensors)
}

fn next_channel(channels: &mut BTreeMap<(usize, u8), usize>, left_site: usize, kind: u8) -> usize {
    if let Some(channel) = channels.get(&(left_site, kind)) {
        return *channel;
    }
    let channel = channels.len() + 1;
    channels.insert((left_site, kind), channel);
    channel
}

pub fn solve_pyrochlore_dmrg(
    sites: usize,
    bonds: &[QuantumSpinBond],
    config: PyrochloreMpoConfig,
) -> TnResult<(DmrgResult, PyrochloreDmrgReport)> {
    let mpo = build_pyrochlore_mpo(sites, bonds, config.exchange, config.anisotropy)?;
    let initial_mpo_bond_dimension = maximum_mpo_bond(&mpo);
    let mpo = if config.compress_max_bond > 0 {
        let data = mpo_to_data(&mpo);
        let compressed = mpo_compress(
            &data,
            &MpoCompressConfig {
                max_bond: config.compress_max_bond,
                tol: config.compress_tolerance,
            },
        )?;
        mpo_from_data(&compressed)?
    } else {
        mpo
    };
    let mpo_bond_dimension = maximum_mpo_bond(&mpo);
    let mut rng = LcgRng::new(config.seed);
    let initial = Mps::random_mps(sites, 2, config.initial_bond_dimension, &mut rng)?;
    let result = dmrg_two_site(&mpo, initial, config.dmrg, &mut rng)?;
    let final_norm = result.mps.norm()?;
    let max_mps_bond_dimension = oxicuda_tn::metrics::metrics::max_bond_dimension(&result.mps)?;
    let midpoint_entropy = if sites > 1 {
        oxicuda_tn::metrics::metrics::entanglement_entropy(&result.mps, sites / 2 - 1)?
    } else {
        0.0
    };
    let report = PyrochloreDmrgReport {
        sites,
        physical_bonds: bonds.iter().map(|bond| bond.multiplicity as usize).sum(),
        initial_mpo_bond_dimension,
        mpo_bond_dimension,
        energy: result.energy,
        sweeps: result.sweeps_done,
        final_norm,
        max_mps_bond_dimension,
        midpoint_entropy,
    };
    Ok((result, report))
}

fn mpo_to_data(mpo: &Mpo) -> MpoData {
    let mut cores = Vec::with_capacity(mpo.site_tensors.len());
    let mut bond_dims = Vec::with_capacity(mpo.site_tensors.len() + 1);
    let mut phys_dims = Vec::with_capacity(mpo.site_tensors.len());
    for (site, tensor) in mpo.site_tensors.iter().enumerate() {
        if site == 0 {
            bond_dims.push(tensor.w_l);
        }
        bond_dims.push(tensor.w_r);
        phys_dims.push((tensor.d_in, tensor.d_out));
        let mut core = vec![0.0; tensor.w_l * tensor.d_in * tensor.d_out * tensor.w_r];
        for left in 0..tensor.w_l {
            for input in 0..tensor.d_in {
                for output in 0..tensor.d_out {
                    for right in 0..tensor.w_r {
                        let index = ((left * tensor.d_in + input) * tensor.d_out + output)
                            * tensor.w_r
                            + right;
                        core[index] = tensor
                            .get(left, output, input, right)
                            .expect("validated MPO tensor indices");
                    }
                }
            }
        }
        cores.push(core);
    }
    MpoData {
        cores,
        bond_dims,
        phys_dims,
    }
}

fn mpo_from_data(data: &MpoData) -> TnResult<Mpo> {
    let mut tensors = Vec::with_capacity(data.cores.len());
    for site in 0..data.cores.len() {
        let left_dimension = data.bond_dims[site];
        let right_dimension = data.bond_dims[site + 1];
        let (input_dimension, output_dimension) = data.phys_dims[site];
        let mut tensor = MpoTensor::zeros(
            left_dimension,
            output_dimension,
            input_dimension,
            right_dimension,
        )?;
        for left in 0..left_dimension {
            for input in 0..input_dimension {
                for output in 0..output_dimension {
                    for right in 0..right_dimension {
                        let index = ((left * input_dimension + input) * output_dimension + output)
                            * right_dimension
                            + right;
                        tensor.set(left, output, input, right, data.cores[site][index])?;
                    }
                }
            }
        }
        tensors.push(tensor);
    }
    Mpo::from_tensors(tensors)
}

fn maximum_mpo_bond(mpo: &Mpo) -> usize {
    mpo.site_tensors
        .iter()
        .flat_map(|tensor| [tensor.w_l, tensor.w_r])
        .max()
        .unwrap_or(1)
}

fn add_global_operator(
    tensor: &mut MpoTensor,
    site: usize,
    sites: usize,
    global_left: usize,
    global_right: usize,
    operator: [f64; 4],
    scale: f64,
    terminal: usize,
) -> TnResult<()> {
    let local_left = if site == 0 {
        if global_left != terminal {
            return Ok(());
        }
        0
    } else {
        global_left
    };
    let local_right = if site + 1 == sites {
        if global_right != 0 {
            return Ok(());
        }
        0
    } else {
        global_right
    };
    for output in 0..2 {
        for input in 0..2 {
            let value = operator[output * 2 + input] * scale;
            if value.abs() <= f64::EPSILON {
                continue;
            }
            let current = tensor.get(local_left, output, input, local_right)?;
            tensor.set(local_left, output, input, local_right, current + value)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantum_spin_thermodynamic_engine::{
        periodic_pyrochlore_model, QuantumSpinConfig, QuantumSpinThermodynamicEngine,
    };
    use crate::symmetry_thermodynamic_substrate::SymmetryThermodynamicConfig;

    #[test]
    fn pyrochlore_mpo_dmrg_matches_exact_four_spin_energy() {
        let (geometry, bonds) =
            periodic_pyrochlore_model(1, 1, 1, SymmetryThermodynamicConfig::default()).unwrap();
        let exact = QuantumSpinThermodynamicEngine::new_neel_with_bonds(
            geometry,
            bonds.clone(),
            QuantumSpinConfig {
                spin_lattice_alpha: 0.0,
                ..QuantumSpinConfig::default()
            },
        )
        .unwrap()
        .lanczos_spectrum(40)
        .ground_energy;
        let (_, report) = solve_pyrochlore_dmrg(
            4,
            &bonds,
            PyrochloreMpoConfig {
                initial_bond_dimension: 4,
                dmrg: DmrgConfig {
                    max_sweeps: 8,
                    chi_max: 16,
                    trunc_tol: 1.0e-10,
                    energy_tol: 1.0e-9,
                    lanczos_iter: 32,
                    lanczos_tol: 1.0e-10,
                },
                ..PyrochloreMpoConfig::default()
            },
        )
        .unwrap();
        assert_eq!(report.physical_bonds, 12);
        assert!(
            report.mpo_bond_dimension < report.initial_mpo_bond_dimension,
            "{report:?}"
        );
        assert!(
            (report.energy - exact).abs() < 1.0e-6,
            "{report:?} exact={exact}"
        );
        assert!((report.final_norm - 1.0).abs() < 1.0e-8);
        assert!(report.midpoint_entropy > 0.0);
    }
}
