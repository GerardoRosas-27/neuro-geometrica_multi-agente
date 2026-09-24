//! Endurecimientos E21/E22 del ciclo siguiente (Sprint 1–2).
//!
//! Re-exporta runners harden + seeds DEV ciclo. La lógica pesada vive en
//! [`crate::field_autonomy_stage2_v2`]. Este módulo añade helpers puros
//! (gate flags, curriculum weights) y wrappers smoke.
//!
//! **Prohibido:** confirmation seeds `0xB300–0xB30F`.

use crate::field_autonomy_stage2_v2::{
    is_confirmation_seed, run_e21_harden, run_e21_harden_cycle, run_e22_harden,
    run_e22_harden_cycle, HyperparamLock, ResultRowV2,
};

pub use crate::field_autonomy_stage2_v2::{
    is_confirmation_seed as gate_blocks_confirmation, DEV_CYCLE_E21_SEEDS as E21_SEEDS,
    DEV_CYCLE_E22_SEEDS as E22_SEEDS,
};

/// Peso relativo de muestras en frontera dx=±2 vs interior (curriculum borde).
pub fn edge_curriculum_weight(dx: f64) -> u32 {
    if (dx.abs() - 2.0).abs() < 1e-9 {
        2
    } else {
        1
    }
}

/// Flags de brazo claim autonomía: RQM/NN/table/attractor OFF.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AutonomyGateFlags {
    pub field_only: bool,
    pub rqm_eval_off: bool,
    pub confirmation_forbidden: bool,
}

impl AutonomyGateFlags {
    pub const CLAIM_ARM: Self = Self {
        field_only: true,
        rqm_eval_off: true,
        confirmation_forbidden: true,
    };

    pub fn validates_seed(self, seed: u64) -> bool {
        if self.confirmation_forbidden && is_confirmation_seed(seed) {
            return false;
        }
        self.field_only && self.rqm_eval_off
    }
}

/// Smoke 1-seed E21 + E22 harden (hyperparams smoke). Para examples/CI corto.
pub fn run_harden_smoke_pair(seed_e21: u64, seed_e22: u64) -> (ResultRowV2, ResultRowV2) {
    let mut hp = HyperparamLock::smoke();
    hp.lock();
    (run_e21_harden(seed_e21, &hp), run_e22_harden(seed_e22, &hp))
}

pub fn run_e21_cycle_smoke() -> Vec<ResultRowV2> {
    run_e21_harden_cycle(true)
}

pub fn run_e22_cycle_smoke() -> Vec<ResultRowV2> {
    run_e22_harden_cycle(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edge_weight_is_double_at_boundary() {
        assert_eq!(edge_curriculum_weight(2.0), 2);
        assert_eq!(edge_curriculum_weight(-2.0), 2);
        assert_eq!(edge_curriculum_weight(0.0), 1);
        assert_eq!(edge_curriculum_weight(1.0), 1);
    }

    #[test]
    fn autonomy_gate_rejects_confirmation() {
        let g = AutonomyGateFlags::CLAIM_ARM;
        assert!(g.validates_seed(0xA310));
        assert!(!g.validates_seed(0xB300));
        assert!(!g.validates_seed(0xB30F));
    }

    #[test]
    fn seed_families_are_disjoint_from_lock_and_confirm() {
        for s in E21_SEEDS.iter().chain(E22_SEEDS.iter()) {
            assert!(*s >= 0xA310 && *s <= 0xA31F);
            assert!(!is_confirmation_seed(*s));
            assert!(!(0xA300..=0xA307).contains(s), "must not reuse lock DEV");
        }
        assert_eq!(E21_SEEDS.len(), 8);
        assert_eq!(E22_SEEDS.len(), 8);
    }

    #[test]
    fn harden_smoke_pair_field_only() {
        let (e21, e22) = run_harden_smoke_pair(0xA310, 0xA318);
        assert!(e21.field_only);
        assert!(e22.field_only);
        assert_ne!(e21.verdict, "INVALID");
        assert_ne!(e22.verdict, "INVALID");
    }
}
