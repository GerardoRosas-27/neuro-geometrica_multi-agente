// gemma_campo: entrenamiento SGD-momentum sin carreras de datos.
// Cada hilo escribe SOLO su propio elemento de salida; no hay `+=` compartido.
// Objetivo: L = (1/(n*d)) * sum_{b,o} (phi_b . W[:,o] - y_bo)^2 + sum_{f,o} lam_f * W_fo^2

struct Params {
    n: u32,      // muestras válidas en el batch (<= capacidad)
    f: u32,      // dimensión de features
    d: u32,      // dimensión de salida (d_field)
    nwg: u32,    // nº de workgroups de la reducción
    lr: f32,
    mom: f32,
    _p0: f32,
    _p1: f32,
};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> phi: array<f32>;        // n x f
@group(0) @binding(2) var<storage, read> y: array<f32>;          // n x d
@group(0) @binding(3) var<storage, read_write> w: array<f32>;    // f x d
@group(0) @binding(4) var<storage, read_write> v: array<f32>;    // f x d (momentum)
@group(0) @binding(5) var<storage, read_write> err: array<f32>;  // n x d
@group(0) @binding(6) var<storage, read> lam: array<f32>;        // f
@group(0) @binding(7) var<storage, read_write> partial: array<f32>; // nwg

// K1: un hilo por (b,o). Lee W, escribe err[b,o] (propio).
@compute @workgroup_size(128)
fn k_forward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= P.n * P.d) { return; }
    let b = idx / P.d;
    let o = idx % P.d;
    var acc: f32 = 0.0;
    for (var j: u32 = 0u; j < P.f; j = j + 1u) {
        acc = acc + phi[b * P.f + j] * w[j * P.d + o];
    }
    err[idx] = acc - y[idx];
}

// K2: un hilo por (f,o). Suma sobre el batch localmente y actualiza SOLO W[f,o], V[f,o].
@compute @workgroup_size(128)
fn k_update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= P.f * P.d) { return; }
    let j = idx / P.d;
    let o = idx % P.d;
    var g: f32 = 0.0;
    for (var b: u32 = 0u; b < P.n; b = b + 1u) {
        g = g + phi[b * P.f + j] * err[b * P.d + o];
    }
    g = g * (2.0 / f32(P.n * P.d)) + 2.0 * lam[j] * w[idx];
    let vn = P.mom * v[idx] + g;
    v[idx] = vn;
    w[idx] = w[idx] - P.lr * vn;
}

// K3: reducción en árbol por workgroup de err^2 -> partial[wg]. La CPU suma partial (f64).
var<workgroup> sh: array<f32, 256>;
@compute @workgroup_size(256)
fn k_reduce(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let total = P.n * P.d;
    let stride = P.nwg * 256u;
    var s: f32 = 0.0;
    var i: u32 = wid.x * 256u + lid.x;
    loop {
        if (i >= total) { break; }
        let e = err[i];
        s = s + e * e;
        i = i + stride;
    }
    sh[lid.x] = s;
    workgroupBarrier();
    var k: u32 = 128u;
    loop {
        if (k == 0u) { break; }
        if (lid.x < k) { sh[lid.x] = sh[lid.x] + sh[lid.x + k]; }
        workgroupBarrier();
        k = k / 2u;
    }
    if (lid.x == 0u) { partial[wid.x] = sh[0]; }
}
