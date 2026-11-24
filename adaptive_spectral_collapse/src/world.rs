//! Synthetic world of unknown opaque concepts (K_BASE = 24, K_NEW = 4, DIM_X = 16)
//! with zero hardcoded primes or spectral coordinates, strict disjoint train/unseen splits,
//! and an unobserved nonlinear symmetric compositional causal law.

use crate::rng::Rng;

pub const DIM_X: usize = 16;
pub const DIM_TRAIT: usize = 4;
pub const K_BASE: usize = 24;
pub const K_NEW: usize = 4;
pub const K_TOT: usize = K_BASE + K_NEW;

/// Training/evaluation split ratios matching PLAN.md §2 verbatim.
pub const N_TRAIN_BASE_PAIRS: usize = 110;
pub const N_NOVEL_TRAIN_PER_CONCEPT: usize = 4;

#[derive(Clone, Debug)]
pub struct UnknownWorld {
    /// Raw opaque unit-norm sensory vectors `x_k in R^{16}` for `k in 0..28`.
    pub xs: Vec<[f64; DIM_X]>,
    /// Unobserved latent causal trait vectors `v_k in R^4` (never revealed directly).
    pub traits: Vec<[f64; DIM_TRAIT]>,
    /// 110 base training pairs (39.9% of 276), connected with >= 6 appearances per base concept.
    pub train_pairs: Vec<(usize, usize)>,
    /// 166 strictly held-out unseen base pairs (60.1% of 276).
    pub unseen_pairs: Vec<(usize, usize)>,
    /// Suite A4: 16 training pairs (4 per novel concept `24..27` paired with base concepts).
    pub novel_train_pairs: Vec<(usize, usize)>,
    /// Suite A4: 80 strictly unseen `(novel, old)` pairs.
    pub novel_old_unseen: Vec<(usize, usize)>,
    /// Suite A4: 6 strictly unseen `(novel, novel)` pairs.
    pub novel_novel_unseen: Vec<(usize, usize)>,
}

impl UnknownWorld {
    pub fn generate(seed: u64) -> Self {
        let mut rng = Rng::from_seed(seed, 7001);

        // 1. Generate K_TOT = 28 arbitrary opaque unit vectors in R^16 (overcomplete, non-orthogonal).
        let mut xs = Vec::with_capacity(K_TOT);
        for _ in 0..K_TOT {
            let mut v = [0.0_f64; DIM_X];
            let mut norm2 = 0.0;
            for x in &mut v {
                let g = rng.normal();
                *x = g;
                norm2 += g * g;
            }
            let inv = 1.0 / norm2.sqrt().max(1e-12);
            for x in &mut v {
                *x *= inv;
            }
            xs.push(v);
        }

        // 2. Generate unobserved 4D causal traits per concept.
        let mut traits = Vec::with_capacity(K_TOT);
        for _ in 0..K_TOT {
            let mut tv = [0.0_f64; DIM_TRAIT];
            for t in &mut tv {
                *t = 0.7 * rng.normal();
            }
            traits.push(tv);
        }

        // 3. Build all 276 distinct base pairs and partition strictly into 110 train / 166 unseen,
        //    guaranteeing every base concept appears in >= 6 training pairs.
        let mut all_base = Vec::with_capacity(K_BASE * (K_BASE - 1) / 2);
        for a in 0..K_BASE {
            for b in (a + 1)..K_BASE {
                all_base.push((a, b));
            }
        }

        // Shuffle all_base deterministically.
        for i in (1..all_base.len()).rev() {
            let j = rng.usize_range(i + 1);
            all_base.swap(i, j);
        }

        let mut in_train = vec![false; all_base.len()];
        let mut deg = vec![0usize; K_BASE];
        let mut chosen = 0usize;

        // Pass 1: greedily ensure every concept has degree >= 6.
        for min_deg in 1..=6 {
            for (idx, &(a, b)) in all_base.iter().enumerate() {
                if !in_train[idx] && (deg[a] < min_deg || deg[b] < min_deg) && chosen < N_TRAIN_BASE_PAIRS {
                    in_train[idx] = true;
                    deg[a] += 1;
                    deg[b] += 1;
                    chosen += 1;
                }
            }
        }
        // Pass 2: fill remaining training slots up to exactly N_TRAIN_BASE_PAIRS = 110.
        for (idx, &(a, b)) in all_base.iter().enumerate() {
            if chosen >= N_TRAIN_BASE_PAIRS {
                break;
            }
            if !in_train[idx] {
                in_train[idx] = true;
                deg[a] += 1;
                deg[b] += 1;
                chosen += 1;
            }
        }

        let mut train_pairs = Vec::with_capacity(N_TRAIN_BASE_PAIRS);
        let mut unseen_pairs = Vec::with_capacity(all_base.len() - N_TRAIN_BASE_PAIRS);
        for (idx, &p) in all_base.iter().enumerate() {
            if in_train[idx] {
                train_pairs.push(p);
            } else {
                unseen_pairs.push(p);
            }
        }
        train_pairs.sort_unstable();
        unseen_pairs.sort_unstable();

        // 4. Suite A4 continual split for 4 novel concepts (24..27):
        //    Each novel concept interacts with exactly 4 distinct base concepts during adaptation;
        //    remaining 20 (novel, old) pairs per novel concept (= 80) + all 6 (novel, novel) pairs are unseen.
        let mut novel_train_pairs = Vec::with_capacity(K_NEW * N_NOVEL_TRAIN_PER_CONCEPT);
        let mut novel_old_unseen = Vec::with_capacity(K_NEW * (K_BASE - N_NOVEL_TRAIN_PER_CONCEPT));
        for nc in K_BASE..K_TOT {
            let mut old_pool: Vec<usize> = (0..K_BASE).collect();
            for i in (1..old_pool.len()).rev() {
                let j = rng.usize_range(i + 1);
                old_pool.swap(i, j);
            }
            for &old_c in &old_pool[..N_NOVEL_TRAIN_PER_CONCEPT] {
                novel_train_pairs.push((old_c, nc));
            }
            for &old_c in &old_pool[N_NOVEL_TRAIN_PER_CONCEPT..] {
                novel_old_unseen.push((old_c, nc));
            }
        }
        novel_train_pairs.sort_unstable();
        novel_old_unseen.sort_unstable();

        let mut novel_novel_unseen = Vec::with_capacity(K_NEW * (K_NEW - 1) / 2);
        for a in K_BASE..K_TOT {
            for b in (a + 1)..K_TOT {
                novel_novel_unseen.push((a, b));
            }
        }

        Self {
            xs,
            traits,
            train_pairs,
            unseen_pairs,
            novel_train_pairs,
            novel_old_unseen,
            novel_novel_unseen,
        }
    }

    /// Generates noisy raw sensory superposition `(x_a + x_b)/sqrt(2) + N(0, sigma_x^2 I)`.
    #[inline]
    pub fn observe_pair(&self, a: usize, b: usize, sigma_x: f64, rng: &mut Rng) -> [f64; DIM_X] {
        let inv_s2 = std::f64::consts::FRAC_1_SQRT_2;
        let mut out = [0.0_f64; DIM_X];
        for i in 0..DIM_X {
            let n = if sigma_x > 0.0 {
                sigma_x * rng.normal()
            } else {
                0.0
            };
            out[i] = (self.xs[a][i] + self.xs[b][i]) * inv_s2 + n;
        }
        out
    }

    /// Nonlinear symmetric relational law (PLAN.md §2).
    #[inline]
    pub fn eval_outcome(&self, a: usize, b: usize) -> (f64, f64, usize) {
        let va = &self.traits[a];
        let vb = &self.traits[b];
        let y1 = (va[0] + vb[0] + 0.85 * va[1] * vb[1]).tanh();
        let diff3 = va[3] - vb[3];
        let y2 = (va[2] + vb[2]).sin() + 0.45 * (diff3 * diff3 - 0.5).tanh();
        let cls = (if y1 >= 0.0 { 2 } else { 0 }) + (if y2 >= 0.0 { 1 } else { 0 });
        (y1, y2, cls)
    }

    /// Checks whether pair `(a, b)` (unordered) is in `unseen_pairs`.
    pub fn is_unseen_base(&self, a: usize, b: usize) -> bool {
        let pair = if a < b { (a, b) } else { (b, a) };
        self.unseen_pairs.binary_search(&pair).is_ok()
    }
}

/// Pair-level partial F1 between unordered predicted pair `(pa, pb)` and true distinct pair `(ta, tb)`.
#[inline]
pub fn pair_partial_f1(pa: usize, pb: usize, ta: usize, tb: usize) -> f64 {
    if pa == pb {
        if pa == ta || pa == tb { 0.5 } else { 0.0 }
    } else {
        let mut hits = 0usize;
        if pa == ta || pa == tb {
            hits += 1;
        }
        if pb == ta || pb == tb {
            hits += 1;
        }
        (hits as f64) / 2.0
    }
}
