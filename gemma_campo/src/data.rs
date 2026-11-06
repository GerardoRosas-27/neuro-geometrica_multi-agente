//! Formato de datos producido por python/encoder.py:
//!   <dir>/<split>_z.f32  : f32 LE, N x d (todas las conversaciones concatenadas)
//!   <dir>/<split>_z.json : {"d": d, "lengths": [len_conv0, len_conv1, ...]}
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Index { pub d: usize, pub lengths: Vec<usize> }

#[derive(Clone, Debug)]
pub struct Split { pub d: usize, pub z: Vec<f32>, pub lengths: Vec<usize>, pub offsets: Vec<usize> }

impl Split {
    pub fn new(d: usize, z: Vec<f32>, lengths: Vec<usize>) -> Self {
        let mut offsets = Vec::with_capacity(lengths.len());
        let mut o = 0;
        for &l in &lengths { offsets.push(o); o += l; }
        assert_eq!(o * d, z.len(), "longitudes no cuadran con z");
        Split { d, z, lengths, offsets }
    }
    pub fn load(dir: &Path, split: &str) -> std::io::Result<Self> {
        let idx: Index = serde_json::from_slice(&std::fs::read(dir.join(format!("{split}_z.json")))?)?;
        let z = read_f32(&dir.join(format!("{split}_z.f32")))?;
        Ok(Split::new(idx.d, z, idx.lengths))
    }
    pub fn conv(&self, c: usize) -> &[f32] {
        let s = self.offsets[c] * self.d;
        &self.z[s..s + self.lengths[c] * self.d]
    }
    pub fn n_tokens(&self) -> usize { self.lengths.iter().sum() }
}

pub fn read_f32(p: &Path) -> std::io::Result<Vec<f32>> {
    let b = std::fs::read(p)?;
    Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}
pub fn write_f32(p: &Path, v: &[f32]) -> std::io::Result<()> {
    let mut b = Vec::with_capacity(v.len() * 4);
    for x in v { b.extend_from_slice(&x.to_le_bytes()); }
    std::fs::write(p, b)
}
