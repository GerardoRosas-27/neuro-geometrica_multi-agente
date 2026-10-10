//! Token-Free Liquid Field — core de inferencia Rust (schema `tfl-v1`).
//!
//! El core solo acepta estado de campo `Ψ` (f32) y contexto estructurado
//! (id de operación). No hay texto, token IDs, tokenizer, logits ni red en
//! esta API: la auditoría E54 es estructural (los tipos no lo permiten).
//! La semántica replica `training/token_free/e46.py::step` operación a
//! operación (layout row-major, `x @ W` con `W[in, out]`, `x = [Ψ, onehot(op)]`).

use serde::Deserialize;
use std::collections::BTreeMap;

pub const SCHEMA: &str = "tfl-v1";

/// Estado de campo Ψ con dimensión declarada.
#[derive(Clone, Debug, PartialEq)]
pub struct FieldState(pub Vec<f32>);

/// Único contexto permitido: id de operación estructurada.
#[derive(Clone, Copy, Debug)]
pub struct FieldContext {
    pub op: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DynKind {
    Mlp,
    Liquid,
    Ssm,
    Linear,
    /// Liquid con K sub-pasos Euler (dt = 1/K), ciclo 2.
    LiquidSs,
    /// Residual Ψ + MLP([Ψ,c]) (análogo arquitectónico de Dφ v4).
    V4Res,
}

#[derive(Deserialize)]
struct RawTensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

#[derive(Deserialize)]
struct RawArtifact {
    schema: String,
    kind: String,
    #[serde(rename = "N")]
    n: usize,
    ops: usize,
    #[serde(default)]
    ctx_dim: Option<usize>,
    #[serde(default)]
    substeps: Option<usize>,
    params_sha256: String,
    params: BTreeMap<String, RawTensor>,
}

/// Matriz densa row-major `[rows, cols]`.
#[derive(Clone, Debug)]
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Mat {
    /// `out[j] = Σ_i x[i]·W[i,j] (+ bias)`.
    fn vecmat(&self, x: &[f32], bias: Option<&[f32]>, out: &mut [f32]) {
        match bias {
            Some(b) => out.copy_from_slice(&b[..self.cols]),
            None => out.iter_mut().for_each(|v| *v = 0.0),
        }
        for (i, &xi) in x.iter().enumerate().take(self.rows) {
            if xi == 0.0 {
                continue;
            }
            let row = &self.data[i * self.cols..(i + 1) * self.cols];
            for (o, w) in out.iter_mut().zip(row) {
                *o += xi * w;
            }
        }
    }
}

/// Parámetros entrenados y versionados.
#[derive(Clone, Debug)]
pub struct LiquidParams {
    pub kind: DynKind,
    pub n: usize,
    pub ops: usize,
    /// Dimensión del contexto estructurado (= ops si es one-hot).
    pub ctx_dim: usize,
    /// Sub-pasos por defecto (solo LiquidSs).
    pub substeps: usize,
    pub sha256: String,
    w1: Option<Mat>,
    b1: Vec<f32>,
    w2: Option<Mat>,
    b2: Vec<f32>,
    wg: Option<Mat>,
    bg: Vec<f32>,
    a: Vec<Mat>,
    b: Vec<Vec<f32>>,
    dec_w: Mat,
    dec_b: Vec<f32>,
    /// Transpuestas (por columna contigua) para actualizar solo índices activos.
    w2t: Vec<f32>,
    wgt: Vec<f32>,
    at: Vec<Vec<f32>>,
}

fn transpose(m: &Mat) -> Vec<f32> {
    let mut t = vec![0.0; m.data.len()];
    for i in 0..m.rows {
        for j in 0..m.cols {
            t[j * m.rows + i] = m.data[i * m.cols + j];
        }
    }
    t
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[derive(Clone, Copy, Debug)]
pub struct RolloutConfig {
    pub max_steps: usize,
}

#[derive(Clone, Debug, Default)]
pub struct RolloutTrace {
    pub steps: usize,
    pub norms: Vec<f32>,
    pub delta_norms: Vec<f32>,
    pub active_nodes: usize,
    /// Auditoría E54: siempre 0 por construcción del tipo.
    pub token_ids_seen: usize,
    pub llm_calls: usize,
}

fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044_715 * x * x * x)).tanh())
}
fn softplus(x: f32) -> f32 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl LiquidParams {
    pub fn from_json(text: &str) -> Result<Self, String> {
        let raw: RawArtifact = serde_json::from_str(text).map_err(|e| e.to_string())?;
        if raw.schema != SCHEMA {
            return Err(format!("schema {} != {SCHEMA}", raw.schema));
        }
        let kind = match raw.kind.as_str() {
            "mlp" => DynKind::Mlp,
            "liquid" => DynKind::Liquid,
            "ssm" => DynKind::Ssm,
            "linear" => DynKind::Linear,
            "liquid_ss" => DynKind::LiquidSs,
            "v4res" => DynKind::V4Res,
            k => return Err(format!("kind desconocido {k}")),
        };
        let mut p = raw.params;
        let mut take = |k: &str| p.remove(k);
        let mat = |t: RawTensor| Mat {
            rows: t.shape[0],
            cols: t.shape[1],
            data: t.data,
        };
        let vecof = |t: Option<RawTensor>| t.map(|t| t.data).unwrap_or_default();
        let dec_w = take("decoder.Dw").map(mat).ok_or("falta decoder.Dw")?;
        let dec_b = vecof(take("decoder.Db"));
        let (mut a, mut b) = (Vec::new(), Vec::new());
        if kind == DynKind::Linear {
            let ta = take("model.A").ok_or("falta A")?;
            let tb = take("model.b").ok_or("falta b")?;
            let n = raw.n;
            for o in 0..raw.ops {
                a.push(Mat {
                    rows: n,
                    cols: n,
                    data: ta.data[o * n * n..(o + 1) * n * n].to_vec(),
                });
                b.push(tb.data[o * n..(o + 1) * n].to_vec());
            }
        }
        let w1 = take("model.W1").map(mat);
        let w2 = take("model.W2").map(mat);
        let wg = take("model.Wg").map(mat);
        let w2t = w2.as_ref().map(transpose).unwrap_or_default();
        let wgt = wg.as_ref().map(transpose).unwrap_or_default();
        let at = a.iter().map(transpose).collect();
        Ok(Self {
            kind,
            n: raw.n,
            ops: raw.ops,
            ctx_dim: raw.ctx_dim.unwrap_or(raw.ops),
            substeps: raw.substeps.unwrap_or(1),
            sha256: raw.params_sha256,
            w1,
            b1: vecof(take("model.b1")),
            w2,
            b2: vecof(take("model.b2")),
            wg,
            bg: vecof(take("model.bg")),
            a,
            b,
            dec_w,
            dec_b,
            w2t,
            wgt,
            at,
        })
    }

    pub fn param_count(&self) -> usize {
        let m = |x: &Option<Mat>| x.as_ref().map(|m| m.data.len()).unwrap_or(0);
        m(&self.w1)
            + m(&self.w2)
            + m(&self.wg)
            + self.b1.len()
            + self.b2.len()
            + self.bg.len()
            + self.a.iter().map(|m| m.data.len()).sum::<usize>()
            + self.b.iter().map(Vec::len).sum::<usize>()
    }

    /// Una transición Ψ → Ψ' con contexto one-hot (semántica de `e46.py::step`).
    pub fn step(&self, psi: &FieldState, c: FieldContext, scratch: &mut Scratch) -> FieldState {
        assert!(c.op < self.ops, "op fuera de rango");
        let mut cv = vec![0.0f32; self.ctx_dim];
        cv[c.op] = 1.0;
        self.step_ctx(psi, &cv, None, None, scratch).0
    }

    /// Transición general: contexto vectorial estructurado, máscara top-k
    /// opcional (E47) y número de sub-pasos opcional (E48, solo LiquidSs).
    /// Devuelve (Ψ', sub-pasos calculados).
    pub fn step_ctx(
        &self,
        psi: &FieldState,
        c: &[f32],
        topk: Option<usize>,
        substeps: Option<usize>,
        scratch: &mut Scratch,
    ) -> (FieldState, usize) {
        let n = self.n;
        assert_eq!(psi.0.len(), n, "dimensión de Ψ");
        assert_eq!(c.len(), self.ctx_dim, "dimensión de contexto");
        // Selección top-k por magnitud (empates: índice menor). Coste incluido.
        let sel: Vec<usize> = match topk {
            Some(k) if k < n => {
                let mut idx: Vec<usize> = (0..n).collect();
                idx.sort_by(|&i, &j| psi.0[j].abs().total_cmp(&psi.0[i].abs()).then(i.cmp(&j)));
                idx.truncate(k);
                idx
            }
            _ => (0..n).collect(),
        };
        let mut cur = psi.0.clone();
        if self.kind == DynKind::Linear {
            let op = (0..self.ops)
                .max_by(|&i, &j| c[i].total_cmp(&c[j]).then(j.cmp(&i)))
                .unwrap_or(0);
            let at = &self.at[op];
            let mut out = cur.clone();
            if sel.len() == n {
                for j in 0..n {
                    out[j] = self.b[op][j] + dot(&cur, &at[j * n..(j + 1) * n]);
                }
            } else {
                for &j in &sel {
                    let row = &at[j * n..(j + 1) * n];
                    out[j] = self.b[op][j] + sel.iter().map(|&i| cur[i] * row[i]).sum::<f32>();
                }
            }
            return (FieldState(out), 1);
        }
        let k_sub = if self.kind == DynKind::LiquidSs {
            substeps.unwrap_or(self.substeps).max(1)
        } else {
            1
        };
        let dt = 1.0 / k_sub as f32;
        for _ in 0..k_sub {
            self.inner(&mut cur, c, &sel, dt, scratch);
        }
        (FieldState(cur), k_sub)
    }

    /// Una evaluación de F sobre los índices `sel` (resto intacto).
    fn inner(&self, cur: &mut [f32], c: &[f32], sel: &[usize], dt: f32, scratch: &mut Scratch) {
        let n = self.n;
        let x = &mut scratch.x;
        x.clear();
        x.resize(n, 0.0);
        for &i in sel {
            x[i] = cur[i];
        }
        x.extend_from_slice(c);
        let w1 = self.w1.as_ref().expect("W1");
        let w2 = self.w2.as_ref().expect("W2");
        scratch.h.resize(w1.cols, 0.0);
        w1.vecmat(x, Some(&self.b1), &mut scratch.h);
        let liquid = matches!(self.kind, DynKind::Liquid | DynKind::LiquidSs);
        if liquid {
            scratch.h.iter_mut().for_each(|v| *v = v.tanh());
        } else {
            scratch.h.iter_mut().for_each(|v| *v = gelu(*v));
        }
        let gated = matches!(
            self.kind,
            DynKind::Liquid | DynKind::LiquidSs | DynKind::Ssm
        );
        for &j in sel {
            let hd = w2.rows;
            let f = self.b2[j] + dot(&scratch.h, &self.w2t[j * hd..(j + 1) * hd]);
            let p = x[j];
            let g = if gated {
                let ind = n + self.ctx_dim;
                self.bg[j] + dot(x, &self.wgt[j * ind..(j + 1) * ind])
            } else {
                0.0
            };
            scratch.pending.push((
                j,
                match self.kind {
                    DynKind::Mlp => f,
                    DynKind::V4Res => p + f,
                    DynKind::Ssm => sigmoid(g) * p + f,
                    _ => {
                        let tau = 0.5 + softplus(g);
                        p + dt * (-p / tau + f)
                    }
                },
            ));
        }
        for (j, v) in scratch.pending.drain(..) {
            cur[j] = v;
        }
    }

    pub fn rollout(
        &self,
        psi0: &FieldState,
        ops: &[usize],
        cfg: RolloutConfig,
        scratch: &mut Scratch,
    ) -> (Vec<FieldState>, RolloutTrace) {
        let mut tr = RolloutTrace {
            active_nodes: self.n,
            ..Default::default()
        };
        let mut cur = psi0.clone();
        let mut states = Vec::with_capacity(ops.len());
        for &op in ops.iter().take(cfg.max_steps) {
            let next = self.step(&cur, FieldContext { op }, scratch);
            let d: f32 = next
                .0
                .iter()
                .zip(&cur.0)
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            tr.delta_norms.push(d.sqrt());
            tr.norms
                .push(next.0.iter().map(|v| v * v).sum::<f32>().sqrt());
            tr.steps += 1;
            states.push(next.clone());
            cur = next;
        }
        (states, tr)
    }

    /// Decoder final independiente (lineal): argmax por slot.
    pub fn decode(&self, psi: &FieldState, slots: usize) -> Vec<usize> {
        let mut logits = vec![0.0; self.dec_w.cols];
        self.dec_w.vecmat(&psi.0, Some(&self.dec_b), &mut logits);
        let v = logits.len() / slots;
        (0..slots)
            .map(|s| {
                let row = &logits[s * v..(s + 1) * v];
                (0..v)
                    .max_by(|&a, &b| row[a].total_cmp(&row[b]))
                    .unwrap_or(0)
            })
            .collect()
    }
}

#[derive(Default)]
pub struct Scratch {
    x: Vec<f32>,
    h: Vec<f32>,
    pending: Vec<(usize, f32)>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny(kind: &str) -> String {
        // N=2, ops=2, H=3: parámetros deterministas conocidos.
        let t = |shape: &[usize], f: f32| {
            let n: usize = shape.iter().product();
            let data: Vec<f32> = (0..n).map(|i| f * ((i as f32) * 0.37).sin()).collect();
            serde_json::json!({"shape": shape, "data": data})
        };
        let mut params = serde_json::json!({
            "decoder.Dw": t(&[2, 4], 1.0), "decoder.Db": t(&[4], 0.1),
        });
        let m = params.as_object_mut().unwrap();
        if kind == "linear" {
            m.insert("model.A".into(), t(&[2, 2, 2], 0.5));
            m.insert("model.b".into(), t(&[2, 2], 0.1));
        } else {
            m.insert("model.W1".into(), t(&[4, 3], 0.5));
            m.insert("model.b1".into(), t(&[3], 0.1));
            m.insert("model.W2".into(), t(&[3, 2], 0.5));
            m.insert("model.b2".into(), t(&[2], 0.1));
            if kind != "mlp" {
                m.insert("model.Wg".into(), t(&[4, 2], 0.5));
                m.insert("model.bg".into(), t(&[2], 0.2));
            }
        }
        serde_json::json!({"schema": SCHEMA, "kind": kind, "N": 2, "ops": 2,
            "params_sha256": "x", "params": params})
        .to_string()
    }

    #[test]
    fn token_free_step_is_finite_and_deterministic() {
        for k in ["mlp", "liquid", "ssm", "linear"] {
            let p = LiquidParams::from_json(&tiny(k)).unwrap();
            let mut s = Scratch::default();
            let psi = FieldState(vec![0.3, -0.7]);
            let (a, tr) = p.rollout(&psi, &[0, 1, 1, 0], RolloutConfig { max_steps: 8 }, &mut s);
            let (b, _) = p.rollout(&psi, &[0, 1, 1, 0], RolloutConfig { max_steps: 8 }, &mut s);
            assert_eq!(a, b);
            assert_eq!(tr.steps, 4);
            assert_eq!(tr.token_ids_seen + tr.llm_calls, 0);
            assert!(a.iter().all(|x| x.0.iter().all(|v| v.is_finite())));
            assert_eq!(p.decode(&a[3], 2).len(), 2);
        }
    }

    #[test]
    fn token_free_liquid_matches_hand_computation() {
        let p = LiquidParams::from_json(&tiny("liquid")).unwrap();
        let mut s = Scratch::default();
        let psi = FieldState(vec![0.3, -0.7]);
        let out = p.step(&psi, FieldContext { op: 1 }, &mut s);
        // Cálculo manual independiente.
        let w = |f: f32, i: usize| f * ((i as f32) * 0.37).sin();
        let x = [0.3f32, -0.7, 0.0, 1.0];
        let h: Vec<f32> = (0..3)
            .map(|j| (w(0.1, j) + (0..4).map(|i| x[i] * w(0.5, i * 3 + j)).sum::<f32>()).tanh())
            .collect();
        for k in 0..2 {
            let f = w(0.1, k) + (0..3).map(|j| h[j] * w(0.5, j * 2 + k)).sum::<f32>();
            let g = w(0.2, k) + (0..4).map(|i| x[i] * w(0.5, i * 2 + k)).sum::<f32>();
            let tau = 0.5 + softplus(g);
            let e = x[k] - x[k] / tau + f;
            assert!((out.0[k] - e).abs() < 1e-6);
        }
    }
}
