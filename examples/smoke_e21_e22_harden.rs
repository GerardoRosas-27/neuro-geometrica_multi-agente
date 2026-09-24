//! Smoke E21/E22 harden — 1 seed cada uno (DEV ciclo). No confirmation.
//!
//! ```text
//! cargo run --example smoke_e21_e22_harden --release
//! ```

use cdt_rqm_epr::field_autonomy_stage2_v2_harden::{
    run_harden_smoke_pair, AutonomyGateFlags, E21_SEEDS, E22_SEEDS,
};

fn main() {
    let gate = AutonomyGateFlags::CLAIM_ARM;
    assert!(gate.validates_seed(E21_SEEDS[0]));
    assert!(gate.validates_seed(E22_SEEDS[0]));
    let (e21, e22) = run_harden_smoke_pair(E21_SEEDS[0], E22_SEEDS[0]);
    println!(
        "E21 harden seed=0x{:X} verdict={} leakage={} field_only={} notes={}",
        e21.seed, e21.verdict, e21.leakage_score, e21.field_only, e21.notes
    );
    println!(
        "E22 harden seed=0x{:X} verdict={} leakage={} field_only={} notes={}",
        e22.seed, e22.verdict, e22.leakage_score, e22.field_only, e22.notes
    );
    println!("confirmation not run (deferred).");
}
