//! ESTELA · autodiferenciación inversa mínima (cinta por hilo) y trait `S` de escalar.
//! El forward de ESTELA es genérico en `S`: la evaluación usa `f64` y el entrenamiento `Var`.
//! `Var` realiza exactamente las mismas operaciones f64 en el mismo orden (su `.v()` es bit a bit
//! el valor de la ruta f64); eso es lo que prueba la compuerta del Paso 0.
use std::cell::RefCell;
use std::ops::{Add, Div, Mul, Neg, Sub};

pub trait S: Copy + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> + Neg<Output = Self> {
    fn c(x: f64) -> Self;
    fn v(&self) -> f64;
    fn tanh(self) -> Self;
    fn exp(self) -> Self;
    fn ln_1p(self) -> Self;
    fn cos(self) -> Self;
    fn sin(self) -> Self;
    fn sqrt(self) -> Self;
    fn add_c(self, b: f64) -> Self { self + Self::c(b) }
    fn mul_c(self, b: f64) -> Self { self * Self::c(b) }
}

impl S for f64 {
    fn c(x: f64) -> Self { x }
    fn v(&self) -> f64 { *self }
    fn tanh(self) -> Self { f64::tanh(self) }
    fn exp(self) -> Self { f64::exp(self) }
    fn ln_1p(self) -> Self { f64::ln_1p(self) }
    fn cos(self) -> Self { f64::cos(self) }
    fn sin(self) -> Self { f64::sin(self) }
    fn sqrt(self) -> Self { f64::sqrt(self) }
    fn add_c(self, b: f64) -> Self { self + b }
    fn mul_c(self, b: f64) -> Self { self * b }
}

/// nodo: hasta 2 padres con su derivada local. Índice u32::MAX = constante (sin gradiente).
#[derive(Clone, Copy)]
struct Node { p: [u32; 2], d: [f64; 2] }
const NONE: u32 = u32::MAX;

thread_local! { static TAPE: RefCell<Vec<Node>> = const { RefCell::new(Vec::new()) }; }

#[derive(Clone, Copy, Debug)]
pub struct Var { pub i: u32, pub x: f64 }

fn push(p0: u32, d0: f64, p1: u32, d1: f64, x: f64) -> Var {
    if p0 == NONE && p1 == NONE { return Var { i: NONE, x }; }
    TAPE.with(|t| { let mut t = t.borrow_mut(); t.push(Node { p: [p0, p1], d: [d0, d1] }); Var { i: (t.len() - 1) as u32, x } })
}
/// reinicia la cinta del hilo
pub fn reset() { TAPE.with(|t| t.borrow_mut().clear()); }
pub fn tape_len() -> usize { TAPE.with(|t| t.borrow().len()) }
/// variable independiente (hoja)
pub fn leaf(x: f64) -> Var { TAPE.with(|t| { let mut t = t.borrow_mut(); t.push(Node { p: [NONE, NONE], d: [0.0, 0.0] }); Var { i: (t.len() - 1) as u32, x } }) }
/// gradiente de `out` respecto de todos los nodos (vector indexado por nodo)
pub fn backward(out: Var) -> Vec<f64> {
    TAPE.with(|t| {
        let t = t.borrow();
        let mut g = vec![0.0; t.len()];
        if out.i == NONE { return g; }
        g[out.i as usize] = 1.0;
        for i in (0..=out.i as usize).rev() {
            let gi = g[i]; if gi == 0.0 { continue; }
            let n = t[i];
            for k in 0..2 { if n.p[k] != NONE { g[n.p[k] as usize] += gi * n.d[k]; } }
        }
        g
    })
}

impl Add for Var { type Output = Var; fn add(self, b: Var) -> Var { push(self.i, 1.0, b.i, 1.0, self.x + b.x) } }
impl Sub for Var { type Output = Var; fn sub(self, b: Var) -> Var { push(self.i, 1.0, b.i, -1.0, self.x - b.x) } }
impl Mul for Var { type Output = Var; fn mul(self, b: Var) -> Var { push(self.i, b.x, b.i, self.x, self.x * b.x) } }
impl Div for Var { type Output = Var; fn div(self, b: Var) -> Var { let q = self.x / b.x; push(self.i, 1.0 / b.x, b.i, -q / b.x, q) } }
impl Neg for Var { type Output = Var; fn neg(self) -> Var { push(self.i, -1.0, NONE, 0.0, -self.x) } }

impl S for Var {
    fn c(x: f64) -> Self { Var { i: NONE, x } }
    fn v(&self) -> f64 { self.x }
    fn tanh(self) -> Self { let y = self.x.tanh(); push(self.i, 1.0 - y * y, NONE, 0.0, y) }
    fn exp(self) -> Self { let y = self.x.exp(); push(self.i, y, NONE, 0.0, y) }
    fn ln_1p(self) -> Self { push(self.i, 1.0 / (1.0 + self.x), NONE, 0.0, self.x.ln_1p()) }
    fn cos(self) -> Self { push(self.i, -self.x.sin(), NONE, 0.0, self.x.cos()) }
    fn sin(self) -> Self { push(self.i, self.x.cos(), NONE, 0.0, self.x.sin()) }
    fn sqrt(self) -> Self { let y = self.x.sqrt(); push(self.i, 0.5 / y, NONE, 0.0, y) }
    fn add_c(self, b: f64) -> Self { push(self.i, 1.0, NONE, 0.0, self.x + b) }
    fn mul_c(self, b: f64) -> Self { push(self.i, b, NONE, 0.0, self.x * b) }
}
