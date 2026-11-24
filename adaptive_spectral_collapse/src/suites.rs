//! Implementation of the 4 preregistered experimental suites (`A1`–`A4`) comparing all 7 arms
//! (`adaptive_spectral`, `riemann_fixed`, `fourier_fixed`, `dyson_gue_fixed`,
//! `poisson_fixed`, `qmc_halton_fixed`, `mlp_conventional`) on unknown opaque concepts.

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;

use crate::mlp_model::MlpModel;
use crate::rng::Rng;
use crate::spectra::compute_riemann_zeros;
use crate::spectral_model::{SpectralArm, SpectralModel, CAVITY_LIMIT, M_1D};
use crate::world::{pair_partial_f1, UnknownWorld, DIM_X, K_BASE};

const SIGMA_EVAL: f64 = 0.025;

/// Executes all 4 suites (`A1`–`A4`) for one `(seed, phase)` and writes CSV results to `out_dir`.
pub fn run_seed(
    seed: u64,
    phase: &str,
    out_dir: &Path,
    export_traces: bool,
) -> std::io::Result<()> {
    fs::create_dir_all(out_dir)?;

    let world = UnknownWorld::generate(seed);
    // Compute first 96 Riemann zeros ONLY for `RiemannFixed` baseline (never passed to AdaptiveSpectral!).
    let riemann_zeros = compute_riemann_zeros(96);

    // 1. Train all 6 spectral arms + 1 conventional MLP arm strictly on base concepts (0..24) & train_pairs (110).
    let mut spectral_models: Vec<SpectralModel> = Vec::with_capacity(6);
    let mut spectral_train_ms: Vec<f64> = Vec::with_capacity(6);

    for (idx, &arm) in SpectralArm::ALL.iter().enumerate() {
        let mut init_rng = Rng::from_seed(seed, 100 + idx as u64);
        let mut train_rng = Rng::from_seed(seed, 200 + idx as u64);
        let zeros_opt = if arm == SpectralArm::RiemannFixed {
            Some(riemann_zeros.as_slice())
        } else {
            None
        };
        let mut model = SpectralModel::init(arm, zeros_opt, &mut init_rng);
        let t0 = Instant::now();
        model.train_base(&world, &mut train_rng);
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        spectral_models.push(model);
        spectral_train_ms.push(ms);
    }

    let mut mlp_init_rng = Rng::from_seed(seed, 199);
    let mut mlp_train_rng = Rng::from_seed(seed, 299);
    let mut mlp = MlpModel::new(K_BASE, &mut mlp_init_rng);
    let t0_mlp = Instant::now();
    mlp.train_base(&world, &mut mlp_train_rng);
    let mlp_train_ms = t0_mlp.elapsed().as_secs_f64() * 1000.0;

    // Pre-generate shared noisy sensory observations on UNSEEN_PAIRS and TRAIN_PAIRS for fair paired comparison.
    let mut eval_rng = Rng::from_seed(seed, 3000);
    let unseen_obs: Vec<[f64; DIM_X]> = world
        .unseen_pairs
        .iter()
        .map(|&(a, b)| world.observe_pair(a, b, SIGMA_EVAL, &mut eval_rng))
        .collect();
    let train_obs: Vec<[f64; DIM_X]> = world
        .train_pairs
        .iter()
        .map(|&(a, b)| world.observe_pair(a, b, SIGMA_EVAL, &mut eval_rng))
        .collect();

    // --- Suite A1: Unknown Input, Spectral Self-Organization & Unseen-Pair Abduction ---
    {
        let mut w = BufWriter::new(File::create(out_dir.join("metrics_a1.csv"))?);
        writeln!(
            w,
            "seed,phase,arm,n_params,train_ms,unseen_abduction_acc,unseen_abduction_f1,train_abduction_acc,collision_free_rate,init_spacing_r,learned_spacing_r,field_contrast_snr"
        )?;

        for (idx, model) in spectral_models.iter().enumerate() {
            let mut exact_u = 0usize;
            let mut f1_u = 0.0_f64;
            for (j, &(a, b)) in world.unseen_pairs.iter().enumerate() {
                let (pa, pb, _) = model.decode_sensory(&unseen_obs[j]);
                if pa == a && pb == b {
                    exact_u += 1;
                }
                f1_u += pair_partial_f1(pa, pb, a, b);
            }
            let mut exact_t = 0usize;
            for (j, &(a, b)) in world.train_pairs.iter().enumerate() {
                let (pa, pb, _) = model.decode_sensory(&train_obs[j]);
                if pa == a && pb == b {
                    exact_t += 1;
                }
            }
            let mut snr_rng = Rng::from_seed(seed, 3100 + idx as u64);
            let snr = model.field_contrast_snr(&mut snr_rng);
            let cfr = model.collision_free_rate();
            let nu = world.unseen_pairs.len() as f64;
            let nt = world.train_pairs.len() as f64;
            writeln!(
                w,
                "0x{:X},{},{},{},{:.3},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
                seed,
                phase,
                model.arm.as_str(),
                SpectralModel::TOTAL_PARAMS,
                spectral_train_ms[idx],
                (exact_u as f64) / nu,
                f1_u / nu,
                (exact_t as f64) / nt,
                cfr,
                model.initial_spacing_r,
                model.learned_spacing_r,
                snr
            )?;
        }

        // Conventional MLP row
        let mut exact_u = 0usize;
        let mut f1_u = 0.0_f64;
        for (j, &(a, b)) in world.unseen_pairs.iter().enumerate() {
            let ((pa, pb), _) = mlp.infer(&unseen_obs[j]);
            if pa == a && pb == b {
                exact_u += 1;
            }
            f1_u += pair_partial_f1(pa, pb, a, b);
        }
        let mut exact_t = 0usize;
        for (j, &(a, b)) in world.train_pairs.iter().enumerate() {
            let ((pa, pb), _) = mlp.infer(&train_obs[j]);
            if pa == a && pb == b {
                exact_t += 1;
            }
        }
        let nu = world.unseen_pairs.len() as f64;
        let nt = world.train_pairs.len() as f64;
        writeln!(
            w,
            "0x{:X},{},mlp_conventional,{},{:.3},{:.6},{:.6},{:.6},0.000000,0.000000,0.000000,0.000000",
            seed,
            phase,
            MlpModel::param_count(K_BASE),
            mlp_train_ms,
            (exact_u as f64) / nu,
            f1_u / nu,
            (exact_t as f64) / nt
        )?;
        w.flush()?;
    }

    // --- Suite A2: Unseen-Pair Prediction "On the Other Side" + Counterfactual do(c_a -> c_new) ---
    {
        // Choose counterfactual target c_new for each unseen pair (a, b) such that {c_new, b} is ALSO unseen.
        let mut cf_rng = Rng::from_seed(seed, 4000);
        let cf_targets: Vec<usize> = world
            .unseen_pairs
            .iter()
            .map(|&(a, b)| {
                let mut cands: Vec<usize> = (0..K_BASE)
                    .filter(|&c| c != a && c != b && world.is_unseen_base(c, b))
                    .collect();
                if cands.is_empty() {
                    cands = (0..K_BASE).filter(|&c| c != a && c != b).collect();
                }
                cands[cf_rng.usize_range(cands.len())]
            })
            .collect();

        let mut w = BufWriter::new(File::create(out_dir.join("metrics_a2.csv"))?);
        writeln!(
            w,
            "seed,phase,arm,unseen_pred_rmse,unseen_class_acc,cf_exact_acc,cf_pred_rmse"
        )?;

        let nu = world.unseen_pairs.len() as f64;
        for model in &spectral_models {
            let mut pred_sq = 0.0_f64;
            let mut cls_hits = 0usize;
            let mut cf_hits = 0usize;
            let mut cf_sq = 0.0_f64;

            for (j, &(a, b)) in world.unseen_pairs.iter().enumerate() {
                let u_obs = model.project_composite(&unseen_obs[j]);
                let (pa, pb, _) = model.decode_phase(&u_obs);
                let (ty1, ty2, tcls) = world.eval_outcome(a, b);
                let (py1, py2, pcls) = model.causal.predict_pair(pa, pb);
                pred_sq += (py1 - ty1).powi(2) + (py2 - ty2).powi(2);
                if pcls == tcls {
                    cls_hits += 1;
                }

                // Counterfactual do(c_a -> c_new): remove inferred cause pa, inject c_new in phase space, re-collapse!
                let c_new = cf_targets[j];
                let (cf_ta, cf_tb) = if c_new < b { (c_new, b) } else { (b, c_new) };
                // Match which inferred cause corresponds to `a` (if `a` was recovered, remove `a`; else remove `pa`).
                let rem_idx = if pb == a { pb } else { pa };
                let mut u_cf = [0.0; 3];
                for d in 0..3 {
                    u_cf[d] = u_obs[d] - model.state_coords[rem_idx][d] + model.state_coords[c_new][d];
                }
                let (cfa, cfb, _) = model.decode_phase(&u_cf);
                if cfa == cf_ta && cfb == cf_tb {
                    cf_hits += 1;
                }
                let (cty1, cty2, _) = world.eval_outcome(cf_ta, cf_tb);
                let (cpy1, cpy2, _) = model.causal.predict_pair(cfa, cfb);
                cf_sq += (cpy1 - cty1).powi(2) + (cpy2 - cty2).powi(2);
            }

            writeln!(
                w,
                "0x{:X},{},{},{:.6},{:.6},{:.6},{:.6}",
                seed,
                phase,
                model.arm.as_str(),
                (pred_sq / (2.0 * nu)).sqrt(),
                (cls_hits as f64) / nu,
                (cf_hits as f64) / nu,
                (cf_sq / (2.0 * nu)).sqrt()
            )?;
        }

        // Conventional MLP counterfactual & prediction
        let inv_s2 = std::f64::consts::FRAC_1_SQRT_2;
        let mut pred_sq = 0.0_f64;
        let mut cls_hits = 0usize;
        let mut cf_hits = 0usize;
        let mut cf_sq = 0.0_f64;
        for (j, &(a, b)) in world.unseen_pairs.iter().enumerate() {
            let ((pa, pb), (py1, py2, pcls)) = mlp.infer(&unseen_obs[j]);
            let (ty1, ty2, tcls) = world.eval_outcome(a, b);
            pred_sq += (py1 - ty1).powi(2) + (py2 - ty2).powi(2);
            if pcls == tcls {
                cls_hits += 1;
            }
            let c_new = cf_targets[j];
            let (cf_ta, cf_tb) = if c_new < b { (c_new, b) } else { (b, c_new) };
            let rem_idx = if pb == a { pb } else { pa };
            let mut x_cf = [0.0; DIM_X];
            for i in 0..DIM_X {
                x_cf[i] = unseen_obs[j][i] - inv_s2 * world.xs[rem_idx][i] + inv_s2 * world.xs[c_new][i];
            }
            let ((cfa, cfb), (cpy1, cpy2, _)) = mlp.infer(&x_cf);
            if cfa == cf_ta && cfb == cf_tb {
                cf_hits += 1;
            }
            let (cty1, cty2, _) = world.eval_outcome(cf_ta, cf_tb);
            cf_sq += (cpy1 - cty1).powi(2) + (cpy2 - cty2).powi(2);
        }
        writeln!(
            w,
            "0x{:X},{},mlp_conventional,{:.6},{:.6},{:.6},{:.6}",
            seed,
            phase,
            (pred_sq / (2.0 * nu)).sqrt(),
            (cls_hits as f64) / nu,
            (cf_hits as f64) / nu,
            (cf_sq / (2.0 * nu)).sqrt()
        )?;
        w.flush()?;
    }

    // --- Suite A3: Unseen Compositional Analogy (A:B::C:?) + Honest Noise Stress Test ---
    {
        let mut w = BufWriter::new(File::create(out_dir.join("metrics_a3.csv"))?);
        writeln!(w, "seed,phase,subtask,arm,sigma_x,exact_acc,partial_f1,pred_rmse")?;

        // A3a: 100 analogies with target D = {b, y} strictly in UNSEEN_PAIRS
        let n_an = 100usize;
        let mut an_rng = Rng::from_seed(seed, 5000);
        let mut analogies: Vec<([f64; DIM_X], [f64; DIM_X], [f64; DIM_X], (usize, usize))> =
            Vec::with_capacity(n_an);
        for _ in 0..n_an {
            let (b, y) = world.unseen_pairs[an_rng.usize_range(world.unseen_pairs.len())];
            // Pick distinct a, x outside {b, y}
            let mut pool: Vec<usize> = (0..K_BASE).filter(|&k| k != b && k != y).collect();
            let ia = an_rng.usize_range(pool.len());
            let a = pool.swap_remove(ia);
            let ix = an_rng.usize_range(pool.len());
            let x = pool[ix];
            let xa = world.observe_pair(a, x, SIGMA_EVAL, &mut an_rng);
            let xb = world.observe_pair(b, x, SIGMA_EVAL, &mut an_rng);
            let xc = world.observe_pair(a, y, SIGMA_EVAL, &mut an_rng);
            analogies.push((xa, xb, xc, (b, y)));
        }

        for model in &spectral_models {
            let mut hits = 0usize;
            let mut f1s = 0.0_f64;
            let mut sq = 0.0_f64;
            for (xa, xb, xc, tgt) in &analogies {
                let (tb, ty) = *tgt;
                let ua = model.project_composite(xa);
                let ub = model.project_composite(xb);
                let uc = model.project_composite(xc);
                let ud = [uc[0] + ub[0] - ua[0], uc[1] + ub[1] - ua[1], uc[2] + ub[2] - ua[2]];
                let (pa, pb, _) = model.decode_phase(&ud);
                if pa == tb && pb == ty {
                    hits += 1;
                }
                f1s += pair_partial_f1(pa, pb, tb, ty);
                let (ty1, ty2, _) = world.eval_outcome(tb, ty);
                let (py1, py2, _) = model.causal.predict_pair(pa, pb);
                sq += (py1 - ty1).powi(2) + (py2 - ty2).powi(2);
            }
            let nf = n_an as f64;
            writeln!(
                w,
                "0x{:X},{},analogy,{},{:.4},{:.6},{:.6},{:.6}",
                seed,
                phase,
                model.arm.as_str(),
                SIGMA_EVAL,
                (hits as f64) / nf,
                f1s / nf,
                (sq / (2.0 * nf)).sqrt()
            )?;
        }

        // MLP analogy in sensory space: x_D = x_C + (x_B - x_A)
        {
            let mut hits = 0usize;
            let mut f1s = 0.0_f64;
            let mut sq = 0.0_f64;
            for (xa, xb, xc, tgt) in &analogies {
                let (tb, ty) = *tgt;
                let mut xd = [0.0; DIM_X];
                for i in 0..DIM_X {
                    xd[i] = xc[i] + xb[i] - xa[i];
                }
                let ((pa, pb), (py1, py2, _)) = mlp.infer(&xd);
                if pa == tb && pb == ty {
                    hits += 1;
                }
                f1s += pair_partial_f1(pa, pb, tb, ty);
                let (ty1, ty2, _) = world.eval_outcome(tb, ty);
                sq += (py1 - ty1).powi(2) + (py2 - ty2).powi(2);
            }
            let nf = n_an as f64;
            writeln!(
                w,
                "0x{:X},{},analogy,mlp_conventional,{:.4},{:.6},{:.6},{:.6}",
                seed,
                phase,
                SIGMA_EVAL,
                (hits as f64) / nf,
                f1s / nf,
                (sq / (2.0 * nf)).sqrt()
            )?;
        }

        // A3b: Honest noise stress sweep across sigma_x in {0.0, 0.025, 0.050, 0.100}
        let sigmas = [0.0_f64, 0.025, 0.050, 0.100];
        let sweep_arms = [0usize, 1, 2]; // AdaptiveSpectral, RiemannFixed, FourierFixed
        let nu = world.unseen_pairs.len() as f64;

        for (s_idx, &sig) in sigmas.iter().enumerate() {
            let mut nrng = Rng::from_seed(seed, 5500 + s_idx as u64);
            let obs: Vec<[f64; DIM_X]> = world
                .unseen_pairs
                .iter()
                .map(|&(a, b)| world.observe_pair(a, b, sig, &mut nrng))
                .collect();

            for &m_idx in &sweep_arms {
                let model = &spectral_models[m_idx];
                let mut hits = 0usize;
                let mut f1s = 0.0_f64;
                let mut sq = 0.0_f64;
                for (j, &(a, b)) in world.unseen_pairs.iter().enumerate() {
                    let (pa, pb, _) = model.decode_sensory(&obs[j]);
                    if pa == a && pb == b {
                        hits += 1;
                    }
                    f1s += pair_partial_f1(pa, pb, a, b);
                    let (ty1, ty2, _) = world.eval_outcome(a, b);
                    let (py1, py2, _) = model.causal.predict_pair(pa, pb);
                    sq += (py1 - ty1).powi(2) + (py2 - ty2).powi(2);
                }
                writeln!(
                    w,
                    "0x{:X},{},noise_sweep,{},{:.4},{:.6},{:.6},{:.6}",
                    seed,
                    phase,
                    model.arm.as_str(),
                    sig,
                    (hits as f64) / nu,
                    f1s / nu,
                    (sq / (2.0 * nu)).sqrt()
                )?;
            }

            // MLP noise sweep
            let mut hits = 0usize;
            let mut f1s = 0.0_f64;
            let mut sq = 0.0_f64;
            for (j, &(a, b)) in world.unseen_pairs.iter().enumerate() {
                let ((pa, pb), (py1, py2, _)) = mlp.infer(&obs[j]);
                if pa == a && pb == b {
                    hits += 1;
                }
                f1s += pair_partial_f1(pa, pb, a, b);
                let (ty1, ty2, _) = world.eval_outcome(a, b);
                sq += (py1 - ty1).powi(2) + (py2 - ty2).powi(2);
            }
            writeln!(
                w,
                "0x{:X},{},noise_sweep,mlp_conventional,{:.4},{:.6},{:.6},{:.6}",
                seed,
                phase,
                sig,
                (hits as f64) / nu,
                f1s / nu,
                (sq / (2.0 * nu)).sqrt()
            )?;
        }
        w.flush()?;
    }

    // Export visual traces BEFORE continual mutation (for Figures 1 & 4)
    if export_traces {
        export_visual_traces(&spectral_models, out_dir)?;
    }

    // --- Suite A4: Continual Autonomous Learning of 4 Brand-New Concepts (K: 24 -> 28) ---
    {
        let mut eval4_rng = Rng::from_seed(seed, 6000);
        let old_obs: Vec<[f64; DIM_X]> = world
            .unseen_pairs
            .iter()
            .map(|&(a, b)| world.observe_pair(a, b, SIGMA_EVAL, &mut eval4_rng))
            .collect();
        let nov_old_obs: Vec<[f64; DIM_X]> = world
            .novel_old_unseen
            .iter()
            .map(|&(a, b)| world.observe_pair(a, b, SIGMA_EVAL, &mut eval4_rng))
            .collect();
        let nov_nov_obs: Vec<[f64; DIM_X]> = world
            .novel_novel_unseen
            .iter()
            .map(|&(a, b)| world.observe_pair(a, b, SIGMA_EVAL, &mut eval4_rng))
            .collect();

        let mut w = BufWriter::new(File::create(out_dir.join("metrics_a4.csv"))?);
        writeln!(
            w,
            "seed,phase,arm,adapt_ms,old_unseen_acc,old_unseen_rmse,novel_old_acc,novel_old_rmse,novel_novel_acc,novel_novel_rmse,novel_all_acc,novel_all_rmse"
        )?;

        let eval_spectral_subset = |model: &SpectralModel, pairs: &[(usize, usize)], obs: &[[f64; DIM_X]]| -> (usize, f64) {
            let mut hits = 0usize;
            let mut sq = 0.0_f64;
            for (j, &(a, b)) in pairs.iter().enumerate() {
                let (pa, pb, _) = model.decode_sensory(&obs[j]);
                if pa == a && pb == b {
                    hits += 1;
                }
                let (ty1, ty2, _) = world.eval_outcome(a, b);
                let (py1, py2, _) = model.causal.predict_pair(pa, pb);
                sq += (py1 - ty1).powi(2) + (py2 - ty2).powi(2);
            }
            (hits, sq)
        };

        for (idx, model) in spectral_models.iter_mut().enumerate() {
            let mut arng = Rng::from_seed(seed, 6100 + idx as u64);
            let t0 = Instant::now();
            model.adapt_continual(&world, &mut arng);
            let ms = t0.elapsed().as_secs_f64() * 1000.0;

            let (oh, osq) = eval_spectral_subset(model, &world.unseen_pairs, &old_obs);
            let (noh, nosq) = eval_spectral_subset(model, &world.novel_old_unseen, &nov_old_obs);
            let (nnh, nnsq) = eval_spectral_subset(model, &world.novel_novel_unseen, &nov_nov_obs);

            let n_old = world.unseen_pairs.len() as f64;
            let n_no = world.novel_old_unseen.len() as f64;
            let n_nn = world.novel_novel_unseen.len() as f64;
            let n_all = n_no + n_nn;

            writeln!(
                w,
                "0x{:X},{},{},{:.3},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
                seed,
                phase,
                model.arm.as_str(),
                ms,
                (oh as f64) / n_old,
                (osq / (2.0 * n_old)).sqrt(),
                (noh as f64) / n_no,
                (nosq / (2.0 * n_no)).sqrt(),
                (nnh as f64) / n_nn,
                (nnsq / (2.0 * n_nn)).sqrt(),
                ((noh + nnh) as f64) / n_all,
                ((nosq + nnsq) / (2.0 * n_all)).sqrt()
            )?;
        }

        // Conventional MLP continual adaptation
        let mut mlp_arng = Rng::from_seed(seed, 6199);
        let t0_m = Instant::now();
        mlp.adapt_continual(&world, &mut mlp_arng);
        let ms_m = t0_m.elapsed().as_secs_f64() * 1000.0;

        let eval_mlp_subset = |m: &MlpModel, pairs: &[(usize, usize)], obs: &[[f64; DIM_X]]| -> (usize, f64) {
            let mut hits = 0usize;
            let mut sq = 0.0_f64;
            for (j, &(a, b)) in pairs.iter().enumerate() {
                let ((pa, pb), (py1, py2, _)) = m.infer(&obs[j]);
                if pa == a && pb == b {
                    hits += 1;
                }
                let (ty1, ty2, _) = world.eval_outcome(a, b);
                sq += (py1 - ty1).powi(2) + (py2 - ty2).powi(2);
            }
            (hits, sq)
        };

        let (oh, osq) = eval_mlp_subset(&mlp, &world.unseen_pairs, &old_obs);
        let (noh, nosq) = eval_mlp_subset(&mlp, &world.novel_old_unseen, &nov_old_obs);
        let (nnh, nnsq) = eval_mlp_subset(&mlp, &world.novel_novel_unseen, &nov_nov_obs);
        let n_old = world.unseen_pairs.len() as f64;
        let n_no = world.novel_old_unseen.len() as f64;
        let n_nn = world.novel_novel_unseen.len() as f64;
        let n_all = n_no + n_nn;

        writeln!(
            w,
            "0x{:X},{},mlp_conventional,{:.3},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
            seed,
            phase,
            ms_m,
            (oh as f64) / n_old,
            (osq / (2.0 * n_old)).sqrt(),
            (noh as f64) / n_no,
            (nosq / (2.0 * n_no)).sqrt(),
            (nnh as f64) / n_nn,
            (nnsq / (2.0 * n_nn)).sqrt(),
            ((noh + nnh) as f64) / n_all,
            ((nosq + nnsq) / (2.0 * n_all)).sqrt()
        )?;
        w.flush()?;
    }

    Ok(())
}

fn export_visual_traces(models: &[SpectralModel], out_dir: &Path) -> std::io::Result<()> {
    // 1. Channel-0 continuous learned wave field W_0(u) for Adaptive, Riemann, Fourier, Poisson
    let mut w = BufWriter::new(File::create(out_dir.join("trace_1d_wave.csv"))?);
    writeln!(w, "arm,u_coord,w0_val")?;
    let n_pts = 240usize;
    let step = (2.0 * CAVITY_LIMIT) / ((n_pts - 1) as f64);
    for &idx in &[0usize, 1, 2, 4] {
        let m = &models[idx];
        for i in 0..n_pts {
            let u = -CAVITY_LIMIT + (i as f64) * step;
            writeln!(w, "{},{:.6},{:.6}", m.arm.as_str(), u, m.waves_1d[0].eval(u))?;
        }
    }
    w.flush()?;

    // 2. Learned state coordinates z_k of AdaptiveSpectral + its learned 1D frequencies
    let mut ws = BufWriter::new(File::create(out_dir.join("trace_states_freqs.csv"))?);
    writeln!(ws, "kind,channel,index,value")?;
    let adapt = &models[0];
    for (k, z) in adapt.state_coords.iter().enumerate() {
        for d in 0..3 {
            writeln!(ws, "state,{},{},{:.6}", d, k, z[d])?;
        }
    }
    for d in 0..3 {
        let mut fs: Vec<f64> = adapt.waves_1d[d].omega.to_vec();
        fs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for m in 0..M_1D {
            writeln!(ws, "freq,{},{},{:.6}", d, m, fs[m])?;
        }
    }
    ws.flush()?;
    Ok(())
}
