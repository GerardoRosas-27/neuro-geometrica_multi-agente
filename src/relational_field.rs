//! Identidad de observador compartida por el motor termodinámico nativo.

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ObserverId(pub usize);
