//! Adaptador de sitios pyrochlore al scaffold 3D-PEPS de OxiCUDA.
//!
//! OxiCUDA usa una grilla cúbica OBC. El adaptador empaqueta las cuatro
//! subredes pyrochlore en un motivo 2×2 por celda y reporta qué enlaces físicos
//! siguen siendo no locales; todavía no implementa optimización PEPS del XXZ.

use crate::quantum_spin_thermodynamic_engine::QuantumSpinBond;
use oxicuda_tn::error::TnResult;
use oxicuda_tn::handle::LcgRng;
use oxicuda_tn::peps::peps_3d::{
    peps3d_bond_dimension, peps3d_entanglement_entropy_z, peps3d_local_expectation, peps3d_n_sites,
    peps3d_norm_approx, peps3d_product_state, peps3d_random, Peps3d, Site3d,
};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PyrochlorePeps3dReport {
    pub pyrochlore_sites: usize,
    pub peps_sites: usize,
    pub lx: usize,
    pub ly: usize,
    pub lz: usize,
    pub bond_dimension: usize,
    pub approximate_norm: f64,
    pub z_entanglement_entropy: f64,
    pub mean_magnetization_z: f64,
    pub local_bonds: usize,
    pub nonlocal_bonds: usize,
    pub optimization_available: bool,
}

#[derive(Clone, Debug)]
pub struct PyrochlorePeps3dAdapter {
    pub peps: Peps3d,
    pub site_map: Vec<Site3d>,
    pub bonds: Vec<QuantumSpinBond>,
    pub cells: (usize, usize, usize),
}

impl PyrochlorePeps3dAdapter {
    pub fn product_state(
        nx: usize,
        ny: usize,
        nz: usize,
        spin_state: &[usize],
        bonds: &[QuantumSpinBond],
    ) -> TnResult<Self> {
        let site_map = pyrochlore_site_map(nx, ny, nz);
        let (lx, ly, lz) = (2 * nx, 2 * ny, nz);
        let mut cubic_state = vec![0usize; lx * ly * lz];
        for (pyrochlore_site, site) in site_map.iter().enumerate() {
            let cubic_index = site.x * ly * lz + site.y * lz + site.z;
            cubic_state[cubic_index] = spin_state[pyrochlore_site];
        }
        let peps = peps3d_product_state(lx, ly, lz, 2, &cubic_state)?;
        Ok(Self {
            peps,
            site_map,
            bonds: bonds.to_vec(),
            cells: (nx, ny, nz),
        })
    }

    pub fn random(
        nx: usize,
        ny: usize,
        nz: usize,
        bond_dimension: usize,
        bonds: &[QuantumSpinBond],
        seed: u64,
    ) -> TnResult<Self> {
        let site_map = pyrochlore_site_map(nx, ny, nz);
        let mut rng = LcgRng::new(seed);
        let peps = peps3d_random(2 * nx, 2 * ny, nz, bond_dimension, 2, &mut rng)?;
        Ok(Self {
            peps,
            site_map,
            bonds: bonds.to_vec(),
            cells: (nx, ny, nz),
        })
    }

    pub fn report(&self, boundary_bond: usize) -> TnResult<PyrochlorePeps3dReport> {
        let z_entropy = if self.peps.lz > 1 {
            peps3d_entanglement_entropy_z(&self.peps, self.peps.lz / 2 - 1)?
        } else {
            0.0
        };
        let spin_z = [0.5, 0.0, 0.0, -0.5];
        let mean_magnetization_z = self
            .site_map
            .iter()
            .map(|site| peps3d_local_expectation(&self.peps, &spin_z, site))
            .collect::<TnResult<Vec<_>>>()?
            .into_iter()
            .sum::<f64>()
            / self.site_map.len().max(1) as f64;
        let (local_bonds, nonlocal_bonds) = self.bond_locality();
        Ok(PyrochlorePeps3dReport {
            pyrochlore_sites: self.site_map.len(),
            peps_sites: peps3d_n_sites(&self.peps),
            lx: self.peps.lx,
            ly: self.peps.ly,
            lz: self.peps.lz,
            bond_dimension: peps3d_bond_dimension(&self.peps),
            approximate_norm: peps3d_norm_approx(&self.peps, boundary_bond.max(1))?,
            z_entanglement_entropy: z_entropy,
            mean_magnetization_z,
            local_bonds,
            nonlocal_bonds,
            optimization_available: false,
        })
    }

    fn bond_locality(&self) -> (usize, usize) {
        let mut local = 0;
        let mut nonlocal = 0;
        for bond in &self.bonds {
            let a = self.site_map[bond.a];
            let b = self.site_map[bond.b];
            let distance = a.x.abs_diff(b.x) + a.y.abs_diff(b.y) + a.z.abs_diff(b.z);
            if distance == 1 {
                local += bond.multiplicity as usize;
            } else {
                nonlocal += bond.multiplicity as usize;
            }
        }
        (local, nonlocal)
    }
}

pub fn pyrochlore_site_map(nx: usize, ny: usize, nz: usize) -> Vec<Site3d> {
    let offsets = [(0, 0), (0, 1), (1, 0), (1, 1)];
    let mut mapping = Vec::with_capacity(4 * nx * ny * nz);
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                for (dx, dy) in offsets {
                    mapping.push(Site3d::new(2 * x + dx, 2 * y + dy, z));
                }
            }
        }
    }
    mapping
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantum_spin_thermodynamic_engine::periodic_pyrochlore_model;
    use crate::symmetry_thermodynamic_substrate::SymmetryThermodynamicConfig;

    #[test]
    fn product_peps_maps_eight_pyrochlore_spins_and_has_zero_entropy() {
        let (_, bonds) =
            periodic_pyrochlore_model(1, 1, 2, SymmetryThermodynamicConfig::default()).unwrap();
        let spin_state = [0, 1, 0, 1, 0, 1, 0, 1];
        let adapter = PyrochlorePeps3dAdapter::product_state(1, 1, 2, &spin_state, &bonds).unwrap();
        let report = adapter.report(8).unwrap();
        assert_eq!(report.pyrochlore_sites, 8);
        assert_eq!(report.peps_sites, 8);
        assert_eq!((report.lx, report.ly, report.lz), (2, 2, 2));
        assert_eq!(report.bond_dimension, 1);
        assert!((report.approximate_norm - 1.0).abs() < 1.0e-12);
        assert!(report.z_entanglement_entropy.abs() < 1.0e-12);
        assert_eq!(report.local_bonds + report.nonlocal_bonds, 24);
        assert!(!report.optimization_available);
    }
}
