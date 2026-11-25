//! Scientific evaluation metrics for Suites T1..T4 on the 768-wave colliding architecture.

use crate::domains::DomainData;
use crate::models::{MlpModel, WaveModel, LATENT_D, N_COLLISIONS, REL_DIM};
use crate::rng::Rng64;

#[derive(Clone, Debug, Default)]
pub struct EvalSummary {
    pub cls_acc: f64,
    pub macro_f1: f64,
    pub reg_rmse: f64,
    pub acc_sym: f64,
    pub acc_asym: f64,
    pub nmi_unsup: f64,
    pub ari_unsup: f64,
    pub abd_top1: f64,
    pub psr: f64,
    pub ghost: f64,
    pub cfr: f64,
    pub two_hop_acc: f64,
    pub alg_consistency: f64,
    pub spec_r: f64,
    pub phase_odd_frac: f64,
}

#[inline]
pub fn argmax(xs: &[f64]) -> usize {
    let mut best_i = 0usize;
    let mut best_v = f64::NEG_INFINITY;
    for (i, &v) in xs.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    best_i
}

#[inline]
pub fn softmax_prob(logits: &[f64], target: usize) -> f64 {
    let max_l = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut sum = 0.0;
    let mut num = 0.0;
    for (i, &l) in logits.iter().enumerate() {
        let e = (l - max_l).exp();
        sum += e;
        if i == target {
            num = e;
        }
    }
    num / sum.max(1e-15)
}

/// Mean consecutive level-spacing ratio <r> of sorted radial norms ||omega1[m]|| (M = 256).
pub fn spectral_spacing_r(omega: &[f64]) -> f64 {
    let mut radii: Vec<f64> = (0..N_COLLISIONS)
        .map(|m| {
            let wm = &omega[m * LATENT_D..(m + 1) * LATENT_D];
            wm.iter().map(|x| x * x).sum::<f64>().sqrt()
        })
        .collect();
    radii.sort_by(|a, b| a.total_cmp(b));
    let mut spacings = Vec::with_capacity(N_COLLISIONS - 1);
    for i in 0..(N_COLLISIONS - 1) {
        spacings.push((radii[i + 1] - radii[i]).max(0.0));
    }
    let mut sum_r = 0.0;
    let mut cnt = 0usize;
    for i in 0..(spacings.len() - 1) {
        let a = spacings[i];
        let b = spacings[i + 1];
        let hi = a.max(b);
        let lo = a.min(b);
        if hi > 1e-9 {
            sum_r += lo / hi;
        }
        cnt += 1;
    }
    if cnt == 0 {
        0.0
    } else {
        sum_r / (cnt as f64)
    }
}

pub fn phase_odd_fraction(phi: &[f64]) -> f64 {
    if phi.is_empty() {
        return 0.0;
    }
    phi.iter().map(|&p| p.sin().powi(2)).sum::<f64>() / (phi.len() as f64)
}

pub fn unsupervised_kmeans_nmi_ari(
    embeddings: &[[f64; REL_DIM]],
    labels: &[usize],
    k: usize,
    rng: &mut Rng64,
) -> (f64, f64) {
    let n = embeddings.len();
    if n < k {
        return (0.0, 0.0);
    }

    let mut norm_emb: Vec<[f64; REL_DIM]> = Vec::with_capacity(n);
    for e in embeddings {
        let n2: f64 = e.iter().map(|x| x * x).sum();
        let inv = 1.0 / (n2 + 1e-12).sqrt();
        let mut v = [0.0; REL_DIM];
        for d in 0..REL_DIM {
            v[d] = e[d] * inv;
        }
        norm_emb.push(v);
    }

    let mut best_inertia = f64::INFINITY;
    let mut best_assign = vec![0usize; n];

    for _restart in 0..16 {
        let mut centers = vec![[0.0; REL_DIM]; k];
        let first = (rng.next_u64() as usize) % n;
        centers[0] = norm_emb[first];
        let mut min_d2 = vec![f64::INFINITY; n];
        for c in 1..k {
            let mut sum_d2 = 0.0;
            for i in 0..n {
                let d2 = sq_dist(&norm_emb[i], &centers[c - 1]);
                if d2 < min_d2[i] {
                    min_d2[i] = d2;
                }
                sum_d2 += min_d2[i];
            }
            if sum_d2 <= 1e-12 {
                centers[c] = norm_emb[(rng.next_u64() as usize) % n];
            } else {
                let mut target = rng.uniform() * sum_d2;
                let mut chosen = n - 1;
                for i in 0..n {
                    target -= min_d2[i];
                    if target <= 0.0 {
                        chosen = i;
                        break;
                    }
                }
                centers[c] = norm_emb[chosen];
            }
        }

        let mut assign = vec![0usize; n];
        let mut inertia = 0.0;
        for _iter in 0..40 {
            let mut changed = false;
            inertia = 0.0;
            for i in 0..n {
                let mut best_c = 0usize;
                let mut best_d = f64::INFINITY;
                for c in 0..k {
                    let d2 = sq_dist(&norm_emb[i], &centers[c]);
                    if d2 < best_d {
                        best_d = d2;
                        best_c = c;
                    }
                }
                if assign[i] != best_c {
                    assign[i] = best_c;
                    changed = true;
                }
                inertia += best_d;
            }
            let mut counts = vec![0usize; k];
            let mut sums = vec![[0.0; REL_DIM]; k];
            for i in 0..n {
                let c = assign[i];
                counts[c] += 1;
                for d in 0..REL_DIM {
                    sums[c][d] += norm_emb[i][d];
                }
            }
            for c in 0..k {
                if counts[c] > 0 {
                    let inv = 1.0 / (counts[c] as f64);
                    for d in 0..REL_DIM {
                        centers[c][d] = sums[c][d] * inv;
                    }
                } else {
                    centers[c] = norm_emb[(rng.next_u64() as usize) % n];
                }
            }
            if !changed {
                break;
            }
        }

        if inertia < best_inertia {
            best_inertia = inertia;
            best_assign = assign;
        }
    }

    compute_nmi_ari(labels, &best_assign, k)
}

#[inline]
fn sq_dist(a: &[f64; REL_DIM], b: &[f64; REL_DIM]) -> f64 {
    let mut s = 0.0;
    for d in 0..REL_DIM {
        let diff = a[d] - b[d];
        s += diff * diff;
    }
    s
}

fn compute_nmi_ari(y_true: &[usize], y_pred: &[usize], k: usize) -> (f64, f64) {
    let n = y_true.len() as f64;
    let mut cont = vec![vec![0usize; k]; k];
    let mut row_sum = vec![0usize; k];
    let mut col_sum = vec![0usize; k];
    for (&t, &p) in y_true.iter().zip(y_pred.iter()) {
        if t < k && p < k {
            cont[t][p] += 1;
            row_sum[t] += 1;
            col_sum[p] += 1;
        }
    }
    let mut mi = 0.0;
    for i in 0..k {
        for j in 0..k {
            let nij = cont[i][j] as f64;
            if nij > 0.0 {
                let p_ij = nij / n;
                let p_i = (row_sum[i] as f64) / n;
                let p_j = (col_sum[j] as f64) / n;
                mi += p_ij * (p_ij / (p_i * p_j)).ln();
            }
        }
    }
    let h_u: f64 = row_sum
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = (c as f64) / n;
            -p * p.ln()
        })
        .sum();
    let h_v: f64 = col_sum
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = (c as f64) / n;
            -p * p.ln()
        })
        .sum();
    let nmi = if h_u + h_v > 1e-12 {
        (2.0 * mi / (h_u + h_v)).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let comb2 = |x: usize| -> f64 {
        if x < 2 {
            0.0
        } else {
            (x * (x - 1)) as f64 * 0.5
        }
    };
    let mut sum_ij = 0.0;
    for i in 0..k {
        for j in 0..k {
            sum_ij += comb2(cont[i][j]);
        }
    }
    let sum_a: f64 = row_sum.iter().map(|&c| comb2(c)).sum();
    let sum_b: f64 = col_sum.iter().map(|&c| comb2(c)).sum();
    let total_c2 = comb2(y_true.len());
    let expected = if total_c2 > 0.0 {
        sum_a * sum_b / total_c2
    } else {
        0.0
    };
    let max_idx = 0.5 * (sum_a + sum_b);
    let denom = max_idx - expected;
    let ari = if denom.abs() > 1e-12 {
        (sum_ij - expected) / denom
    } else {
        0.0
    };
    (nmi, ari)
}

fn compute_macro_f1(y_true: &[usize], y_pred: &[usize], k: usize) -> f64 {
    let mut tp = vec![0usize; k];
    let mut fp = vec![0usize; k];
    let mut fn_ = vec![0usize; k];
    for (&t, &p) in y_true.iter().zip(y_pred.iter()) {
        if t == p {
            tp[t] += 1;
        } else {
            fp[p] += 1;
            fn_[t] += 1;
        }
    }
    let mut f1_sum = 0.0;
    let mut valid = 0usize;
    for c in 0..k {
        let denom = 2 * tp[c] + fp[c] + fn_[c];
        if denom > 0 {
            f1_sum += (2 * tp[c]) as f64 / (denom as f64);
            valid += 1;
        }
    }
    if valid == 0 {
        0.0
    } else {
        f1_sum / (valid as f64)
    }
}

#[allow(clippy::type_complexity)]
pub fn evaluate_domain(
    domain: &DomainData,
    test_indices: &[usize],
    predict_fn: &dyn Fn(usize, usize) -> (Vec<f64>, Vec<f64>, [f64; REL_DIM]),
    spec_r: f64,
    phase_odd_frac: f64,
    rng: &mut Rng64,
) -> EvalSummary {
    let n_ent = domain.n_entities;
    let n_pairs = domain.pairs.len();

    let mut all_reg: Vec<Vec<f64>> = Vec::with_capacity(n_pairs);
    let mut all_logits: Vec<Vec<f64>> = Vec::with_capacity(n_pairs);
    let mut all_z: Vec<[f64; REL_DIM]> = Vec::with_capacity(n_pairs);

    for p in &domain.pairs {
        let (reg, logits, z) = predict_fn(p.a, p.b);
        all_reg.push(reg);
        all_logits.push(logits);
        all_z.push(z);
    }

    let mut y_true = Vec::with_capacity(test_indices.len());
    let mut y_pred = Vec::with_capacity(test_indices.len());
    let mut test_z = Vec::with_capacity(test_indices.len());
    let mut correct = 0usize;
    let mut sym_ok = 0usize;
    let mut sym_tot = 0usize;
    let mut asym_ok = 0usize;
    let mut asym_tot = 0usize;
    let mut sq_err = 0.0;
    let mut reg_cnt = 0usize;

    let mut is_test = vec![false; n_pairs];
    for &idx in test_indices {
        is_test[idx] = true;
        let p = &domain.pairs[idx];
        let pred = argmax(&all_logits[idx]);
        y_true.push(p.cls);
        y_pred.push(pred);
        test_z.push(all_z[idx]);
        if pred == p.cls {
            correct += 1;
        }
        let is_sym_class = if domain.name.starts_with("domain_a") {
            p.cls == 0 || p.cls == 3
        } else if domain.name.starts_with("domain_b") {
            p.cls == 0 || p.cls == 1
        } else {
            p.cls == 0
        };
        if is_sym_class {
            sym_tot += 1;
            if pred == p.cls {
                sym_ok += 1;
            }
        } else {
            asym_tot += 1;
            if pred == p.cls {
                asym_ok += 1;
            }
        }
        for c in 0..domain.reg_dim {
            let e = all_reg[idx][c] - p.reg[c];
            sq_err += e * e;
            reg_cnt += 1;
        }
    }

    let cls_acc = correct as f64 / (test_indices.len().max(1) as f64);
    let macro_f1 = compute_macro_f1(&y_true, &y_pred, domain.cls_dim);
    let reg_rmse = if reg_cnt > 0 {
        (sq_err / (reg_cnt as f64)).sqrt()
    } else {
        0.0
    };
    let acc_sym = if sym_tot > 0 {
        sym_ok as f64 / (sym_tot as f64)
    } else {
        0.0
    };
    let acc_asym = if asym_tot > 0 {
        asym_ok as f64 / (asym_tot as f64)
    } else {
        0.0
    };

    let (nmi_unsup, ari_unsup) = unsupervised_kmeans_nmi_ari(&test_z, &y_true, domain.cls_dim, rng);

    let mut cfr_ok = 0usize;
    let mut cfr_tot = 0usize;
    for &idx in test_indices {
        let a = domain.pairs[idx].a;
        let b = domain.pairs[idx].b;
        let ref_v = &all_z[idx];
        let n_ref = ref_v.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
        let mut max_sim = f64::NEG_INFINITY;
        for k in 0..n_ent {
            if k == a || k == b {
                continue;
            }
            let kidx = domain.pair_lookup[a * n_ent + k];
            let cand = &all_z[kidx];
            let n_c = cand.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
            let dot: f64 = (0..REL_DIM).map(|d| ref_v[d] * cand[d]).sum();
            let sim = dot / (n_ref * n_c);
            if sim > max_sim {
                max_sim = sim;
            }
        }
        if max_sim < 0.96 {
            cfr_ok += 1;
        }
        cfr_tot += 1;
    }
    let cfr = cfr_ok as f64 / (cfr_tot.max(1) as f64);

    let mut abd_ok = 0usize;
    let mut abd_tot = 0usize;
    let mut psr_sum = 0.0;
    let mut ghost_sum = 0.0;

    if domain.reg_dim > 0 {
        let inv_2tau2 = 1.0 / (2.0 * 0.45 * 0.45);
        for &idx in test_indices {
            let a = domain.pairs[idx].a;
            let b = domain.pairs[idx].b;
            let y_obs = &domain.pairs[idx].reg;
            let mut best_k = usize::MAX;
            let mut best_score = f64::NEG_INFINITY;
            let mut true_score = 0.0;
            let mut off_sum = 0.0;
            let mut off_cnt = 0usize;
            for k in 0..n_ent {
                if k == a {
                    continue;
                }
                let kidx = domain.pair_lookup[a * n_ent + k];
                let y_hat = &all_reg[kidx];
                let mut d2 = 0.0;
                for c in 0..domain.reg_dim {
                    let e = y_hat[c] - y_obs[c];
                    d2 += e * e;
                }
                let score = (-d2 * inv_2tau2).exp();
                if score > best_score {
                    best_score = score;
                    best_k = k;
                }
                if k == b {
                    true_score = score;
                } else {
                    off_sum += score;
                    off_cnt += 1;
                }
            }
            if best_k == b {
                abd_ok += 1;
            }
            let mean_off = off_sum / (off_cnt.max(1) as f64);
            ghost_sum += mean_off;
            psr_sum += true_score / (mean_off + 1e-6);
            abd_tot += 1;
        }
    } else {
        for u in 0..n_ent {
            for r_star in 0..domain.cls_dim {
                let mut unseen_cands = Vec::new();
                let mut has_true = false;
                for v in 0..n_ent {
                    if v == u {
                        continue;
                    }
                    let idx = domain.pair_lookup[u * n_ent + v];
                    if is_test[idx] {
                        unseen_cands.push((v, idx));
                        if domain.pairs[idx].cls == r_star {
                            has_true = true;
                        }
                    }
                }
                if !has_true || unseen_cands.len() < 2 {
                    continue;
                }
                let mut best_idx = 0usize;
                let mut best_p = f64::NEG_INFINITY;
                let mut true_p_sum = 0.0;
                let mut true_cnt = 0usize;
                let mut off_p_sum = 0.0;
                let mut off_cnt = 0usize;
                for &(_v, idx) in &unseen_cands {
                    let p = softmax_prob(&all_logits[idx], r_star);
                    if p > best_p {
                        best_p = p;
                        best_idx = idx;
                    }
                    if domain.pairs[idx].cls == r_star {
                        true_p_sum += p;
                        true_cnt += 1;
                    } else {
                        off_p_sum += p;
                        off_cnt += 1;
                    }
                }
                if domain.pairs[best_idx].cls == r_star {
                    abd_ok += 1;
                }
                let mean_true = true_p_sum / (true_cnt.max(1) as f64);
                let mean_off = off_p_sum / (off_cnt.max(1) as f64);
                ghost_sum += mean_off;
                psr_sum += mean_true / (mean_off + 1e-6);
                abd_tot += 1;
            }
        }
    }

    let abd_top1 = abd_ok as f64 / (abd_tot.max(1) as f64);
    let psr = psr_sum / (abd_tot.max(1) as f64);
    let ghost = ghost_sum / (abd_tot.max(1) as f64);

    let two_hop_acc = if !domain.two_hop_queries.is_empty() {
        let mut ok = 0usize;
        for q in &domain.two_hop_queries {
            if argmax(&all_logits[q.pair_ac_idx]) == domain.pairs[q.pair_ac_idx].cls {
                ok += 1;
            }
        }
        ok as f64 / (domain.two_hop_queries.len() as f64)
    } else {
        0.0
    };

    let alg_consistency = if !domain.node_tiers.is_empty() {
        let mut law_sym_ok = 0usize;
        let mut law_sym_tot = 0usize;
        let mut law_inv_ok = 0usize;
        let mut law_inv_tot = 0usize;
        let mut law_cyc_ok = 0usize;
        let mut law_cyc_tot = 0usize;
        for &idx in test_indices {
            let u = domain.pairs[idx].a;
            let v = domain.pairs[idx].b;
            let rev = domain.pair_lookup[v * n_ent + u];
            let t_uv = domain.pairs[idx].cls;
            let t_vu = domain.pairs[rev].cls;
            let p_uv = argmax(&all_logits[idx]);
            let p_vu = argmax(&all_logits[rev]);
            if t_uv == 0 && t_vu == 0 {
                law_sym_tot += 1;
                if p_uv == 0 && p_vu == 0 {
                    law_sym_ok += 1;
                }
            }
            if t_uv == 1 && t_vu == 3 {
                law_inv_tot += 1;
                if p_uv == 1 && p_vu == 3 {
                    law_inv_ok += 1;
                }
            }
            if t_uv == 2 {
                law_cyc_tot += 1;
                if p_uv == 2 && p_vu != 2 {
                    law_cyc_ok += 1;
                }
            }
        }
        let mut law_trans_ok = 0usize;
        let mut law_trans_tot = 0usize;
        for u in 0..n_ent {
            for v in 0..n_ent {
                if v == u {
                    continue;
                }
                let uv = domain.pair_lookup[u * n_ent + v];
                if domain.pairs[uv].cls != 1 {
                    continue;
                }
                for w in 0..n_ent {
                    if w == u || w == v {
                        continue;
                    }
                    let vw = domain.pair_lookup[v * n_ent + w];
                    let uw = domain.pair_lookup[u * n_ent + w];
                    if is_test[uw] && domain.pairs[vw].cls == 1 && domain.pairs[uw].cls == 1 {
                        law_trans_tot += 1;
                        if argmax(&all_logits[uv]) == 1
                            && argmax(&all_logits[vw]) == 1
                            && argmax(&all_logits[uw]) == 1
                        {
                            law_trans_ok += 1;
                        }
                    }
                }
            }
        }
        let r_sym = law_sym_ok as f64 / (law_sym_tot.max(1) as f64);
        let r_inv = law_inv_ok as f64 / (law_inv_tot.max(1) as f64);
        let r_cyc = law_cyc_ok as f64 / (law_cyc_tot.max(1) as f64);
        let r_tr = law_trans_ok as f64 / (law_trans_tot.max(1) as f64);
        0.25 * (r_sym + r_inv + r_cyc + r_tr)
    } else {
        0.0
    };

    EvalSummary {
        cls_acc,
        macro_f1,
        reg_rmse,
        acc_sym,
        acc_asym,
        nmi_unsup,
        ari_unsup,
        abd_top1,
        psr,
        ghost,
        cfr,
        two_hop_acc,
        alg_consistency,
        spec_r,
        phase_odd_frac,
    }
}

pub fn eval_wave(
    model: &WaveModel,
    domain: &DomainData,
    test_indices: &[usize],
    rng: &mut Rng64,
) -> EvalSummary {
    let spec_r = spectral_spacing_r(&model.core.omega1);
    let phase_odd = phase_odd_fraction(&model.core.phi1);
    evaluate_domain(
        domain,
        test_indices,
        &|a, b| model.predict_pair(&domain.obs[a], &domain.obs[b]),
        spec_r,
        phase_odd,
        rng,
    )
}

pub fn eval_mlp(
    model: &MlpModel,
    domain: &DomainData,
    test_indices: &[usize],
    rng: &mut Rng64,
) -> EvalSummary {
    evaluate_domain(
        domain,
        test_indices,
        &|a, b| model.predict_pair(&domain.obs[a], &domain.obs[b]),
        0.0,
        0.0,
        rng,
    )
}
