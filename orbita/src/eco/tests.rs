use super::*;
fn w() -> Orbita { Orbita::from_json(&std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../artifacts/orbita/dev/weights_0xA400.json")).unwrap()) }
#[test] fn eco_paso0b_zero_delta_bit_exact() { let d = w(); let (bad, n) = paso0b(&d, 0xC400); assert!(n > 0); assert_eq!(bad, 0); }
#[test] fn eco_paso0c_d3num_layout() { assert!(d3_layout_ok()); }
#[test] fn eco_rollback_exact_and_zero_arm_parity() {
    let d = w(); let cfg = RunCfg { n_new: 4, n_known: 4, hpf: 1, sleep_steps: 20, do_e33: false };
    let r = run_seed(&d, 0xC401, &cfg, &[Arm::Eco, Arm::Zero, Arm::NoRollback]);
    assert_eq!(r["parity_zero_vs_orbita_lat"]["mismatch"], 0);
    assert_eq!(r["arms"]["eco_zero"]["auc_new"], r["arms"]["orbita_lat"]["auc_new"]);
}
#[test] fn eco_corr_trains_and_changes() {
    let mut c = Corr::new(1, true); let h = c.hash(); let d = w();
    let e = gen_stream(0xC402, 1, 0).remove(0); let b = make_base(&d, e.fam, e.p, &e.x0, &e.ys, K, None);
    let (t, _) = teacher_of(&d, &b, &e); let cdt = vec![Rec { ep: e, base: b, teacher: t, fit_err: 0.0, success: false, gate: false }];
    fit(&d, &mut c, 5, 3, |r| sample(&d, &cdt, &[0], &[0], None, r)); assert_ne!(c.hash(), h);
}
