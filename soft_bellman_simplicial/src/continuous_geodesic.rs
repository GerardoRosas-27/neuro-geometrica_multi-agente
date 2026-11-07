//! Optimización Continua Baricéntrica de Snell-Fermat sobre Complejos Simpliciales 3D
//!
//! Resuelve el límite de discretización de v2 ("Al refinar la discretización, el oráculo todavía
//! reduce su costo un 11.06%: no es una geodésica continua exacta"):
//! 1. Parametriza cada nodo interior x_m de una polilínea simplicial mediante coordenadas
//!    baricéntricas continuas \lambda_m \in \text{int}(\Delta^{k_m}) sobre su símplice soporte \text{supp}(x_m).
//! 2. Preserva exactamente el soporte simplicial de cada nodo (\lambda_{m,a} \ge \epsilon > 0), garantizando
//!    que cada segmento [p_m(\lambda_m), p_{m+1}(\lambda_{m+1})] permanezca 100% contenido en el mismo
//!    tetraedro activo y conserve exactamente su portador e impedancia de cara/volumen.
//! 3. Minimiza la acción geodésica intrasimplicial continua mediante descenso de gradiente riemanniano
//!    proyectado sobre el símplice con búsqueda de línea de Armijo, satisfaciendo la ley de refracción
//!    de Snell discreta en las fronteras de caras y aristas.

pub type V3 = [f64; 3];

#[derive(Clone, Debug)]
pub struct ContinuousGeodesicResult {
    pub initial_points: Vec<V3>,
    pub optimized_points: Vec<V3>,
    pub barycentric_coords: Vec<Vec<f64>>,
    pub supports: Vec<Vec<usize>>,
    pub initial_model_cost: f64,
    pub optimized_model_cost: f64,
    pub initial_true_cost: f64,
    pub optimized_true_cost: f64,
    pub iterations: usize,
}

/// Proyecta un vector `y` sobre el símplice interior `{ \lambda : \sum \lambda_i = 1, \lambda_i >= eps }`
pub fn project_to_simplex_interior(y: &[f64], eps: f64) -> Vec<f64> {
    let k = y.len();
    if k == 0 {
        return vec![];
    }
    if k == 1 {
        return vec![1.0];
    }
    let rem = (1.0 - (k as f64) * eps).max(0.0);
    if rem <= 1e-12 {
        return vec![1.0 / (k as f64); k];
    }
    let mut u: Vec<f64> = y.iter().map(|&v| v - eps).collect();
    u.sort_by(|a, b| b.total_cmp(a));
    let mut cssv = 0.0;
    let mut rho = 0usize;
    for (j, &val) in u.iter().enumerate() {
        cssv += val;
        let t = (cssv - rem) / ((j + 1) as f64);
        if val - t > 0.0 {
            rho = j;
        }
    }
    let sum_rho: f64 = u.iter().take(rho + 1).sum();
    let theta = (sum_rho - rem) / ((rho + 1) as f64);
    y.iter()
        .map(|&v| ((v - eps - theta).max(0.0)) + eps)
        .collect()
}

/// Calcula las coordenadas baricéntricas iniciales \lambda \in \Delta^k de un punto `p` respecto a su soporte
pub fn recover_initial_barycentric(p: V3, vertices: &[V3], support: &[usize]) -> Vec<f64> {
    match support.len() {
        0 => vec![],
        1 => vec![1.0],
        2 => {
            let a = vertices[support[0]];
            let b = vertices[support[1]];
            let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let ap = [p[0] - a[0], p[1] - a[1], p[2] - a[2]];
            let denom = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
            let t = if denom > 1e-14 {
                ((ap[0] * ab[0] + ap[1] * ab[1] + ap[2] * ab[2]) / denom).clamp(0.01, 0.99)
            } else {
                0.5
            };
            vec![1.0 - t, t]
        }
        3 => {
            let v0 = vertices[support[0]];
            let v1 = vertices[support[1]];
            let v2 = vertices[support[2]];
            let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
            let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
            let ep = [p[0] - v0[0], p[1] - v0[1], p[2] - v0[2]];
            let d00 = e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2];
            let d01 = e1[0] * e2[0] + e1[1] * e2[1] + e1[2] * e2[2];
            let d11 = e2[0] * e2[0] + e2[1] * e2[1] + e2[2] * e2[2];
            let d20 = ep[0] * e1[0] + ep[1] * e1[1] + ep[2] * e1[2];
            let d21 = ep[0] * e2[0] + ep[1] * e2[1] + ep[2] * e2[2];
            let denom = d00 * d11 - d01 * d01;
            if denom.abs() > 1e-14 {
                let l1 = (d11 * d20 - d01 * d21) / denom;
                let l2 = (d00 * d21 - d01 * d20) / denom;
                let l0 = 1.0 - l1 - l2;
                project_to_simplex_interior(&[l0, l1, l2], 1e-3)
            } else {
                vec![1.0 / 3.0; 3]
            }
        }
        k => vec![1.0 / (k as f64); k],
    }
}

pub fn eval_barycentric_point(vertices: &[V3], support: &[usize], lambda: &[f64]) -> V3 {
    let mut p = [0.0; 3];
    for (&vid, &l) in support.iter().zip(lambda) {
        let v = vertices[vid];
        p[0] += l * v[0];
        p[1] += l * v[1];
        p[2] += l * v[2];
    }
    p
}

/// Optimiza continuamente los puntos de cruce baricéntricos \lambda_m \in \Delta^{k_m} a lo largo de la ruta
pub fn optimize_barycentric_snell_polyline<FModel, FTruth>(
    mesh_vertices: &[V3],
    initial_points: &[V3],
    supports: &[Vec<usize>],
    segment_model_impedances: &[f64],
    segment_true_impedances: &[f64],
    model_segment_length_and_grad: FModel,
    true_segment_length: FTruth,
    max_iters: usize,
) -> ContinuousGeodesicResult
where
    FModel: Fn(V3, usize) -> (f64, V3),
    FTruth: Fn(V3, usize) -> f64,
{
    let m_nodes = initial_points.len();
    let mut lambdas: Vec<Vec<f64>> = (0..m_nodes)
        .map(|i| recover_initial_barycentric(initial_points[i], mesh_vertices, &supports[i]))
        .collect();
    let mut points: Vec<V3> = (0..m_nodes)
        .map(|i| eval_barycentric_point(mesh_vertices, &supports[i], &lambdas[i]))
        .collect();

    let compute_total_cost = |pts: &[V3], is_truth: bool| -> f64 {
        let mut total = 0.0;
        for s in 0..(pts.len().saturating_sub(1)) {
            let d = [
                pts[s + 1][0] - pts[s][0],
                pts[s + 1][1] - pts[s][1],
                pts[s + 1][2] - pts[s][2],
            ];
            if is_truth {
                total += segment_true_impedances[s] * true_segment_length(d, s);
            } else {
                total += segment_model_impedances[s] * model_segment_length_and_grad(d, s).0;
            }
        }
        total
    };

    let initial_model_cost = compute_total_cost(&points, false);
    let initial_true_cost = compute_total_cost(&points, true);

    if m_nodes <= 2 {
        return ContinuousGeodesicResult {
            initial_points: initial_points.to_vec(),
            optimized_points: points,
            barycentric_coords: lambdas,
            supports: supports.to_vec(),
            initial_model_cost,
            optimized_model_cost: initial_model_cost,
            initial_true_cost,
            optimized_true_cost: initial_true_cost,
            iterations: 0,
        };
    }

    let eps_simplex = 1e-3;
    let mut actual_iters = 0;

    for iter in 0..max_iters {
        let mut max_move = 0.0f64;

        // Barrido Gauss-Seidel hacia adelante y hacia atrás sobre los nodos interiores 1 .. m_nodes-1
        for idx in 1..(m_nodes - 1) {
            let supp = &supports[idx];
            if supp.len() <= 1 {
                continue; // Un vértice 0-símplice es fijo
            }

            let local_cost = |p_mid: V3, p_prev: V3, p_next: V3| -> (f64, V3) {
                let d_prev = [
                    p_mid[0] - p_prev[0],
                    p_mid[1] - p_prev[1],
                    p_mid[2] - p_prev[2],
                ];
                let d_next = [
                    p_next[0] - p_mid[0],
                    p_next[1] - p_mid[1],
                    p_next[2] - p_mid[2],
                ];
                let (l_prev, g_prev) = model_segment_length_and_grad(d_prev, idx - 1);
                let (l_next, g_next) = model_segment_length_and_grad(d_next, idx);
                let imp_prev = segment_model_impedances[idx - 1];
                let imp_next = segment_model_impedances[idx];
                let val = imp_prev * l_prev + imp_next * l_next;
                let grad_p = [
                    imp_prev * g_prev[0] - imp_next * g_next[0],
                    imp_prev * g_prev[1] - imp_next * g_next[1],
                    imp_prev * g_prev[2] - imp_next * g_next[2],
                ];
                (val, grad_p)
            };

            let p_prev = points[idx - 1];
            let p_next = points[idx + 1];
            let (curr_val, grad_p) = local_cost(points[idx], p_prev, p_next);

            // Gradiente respecto a \lambda_{idx, a}: \partial S / \partial \lambda_a = grad_p \cdot v_a
            let mut grad_lambda: Vec<f64> = supp
                .iter()
                .map(|&vid| {
                    let v = mesh_vertices[vid];
                    grad_p[0] * v[0] + grad_p[1] * v[1] + grad_p[2] * v[2]
                })
                .collect();

            // Proyección tangente de suma cero
            let mean_g = grad_lambda.iter().sum::<f64>() / (supp.len() as f64);
            for g in &mut grad_lambda {
                *g -= mean_g;
            }
            let g_norm = grad_lambda.iter().map(|x| x * x).sum::<f64>().sqrt();
            if g_norm < 1e-10 {
                continue;
            }

            // Búsqueda de línea de Armijo proyectada sobre \Delta^{k}
            let mut step = 0.25 / g_norm.max(0.5);
            for _ in 0..8 {
                let cand_raw: Vec<f64> = lambdas[idx]
                    .iter()
                    .zip(&grad_lambda)
                    .map(|(&l, &g)| l - step * g)
                    .collect();
                let cand_lambda = project_to_simplex_interior(&cand_raw, eps_simplex);
                let cand_p = eval_barycentric_point(mesh_vertices, supp, &cand_lambda);
                let (cand_val, _) = local_cost(cand_p, p_prev, p_next);
                if cand_val < curr_val - 1e-12 {
                    let diff = (0..3)
                        .map(|d| (cand_p[d] - points[idx][d]).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    max_move = max_move.max(diff);
                    lambdas[idx] = cand_lambda;
                    points[idx] = cand_p;
                    break;
                }
                step *= 0.45;
            }
        }

        actual_iters = iter + 1;
        if max_move < 1e-6 {
            break;
        }
    }

    let optimized_model_cost = compute_total_cost(&points, false);
    let optimized_true_cost = compute_total_cost(&points, true);

    ContinuousGeodesicResult {
        initial_points: initial_points.to_vec(),
        optimized_points: points,
        barycentric_coords: lambdas,
        supports: supports.to_vec(),
        initial_model_cost,
        optimized_model_cost,
        initial_true_cost,
        optimized_true_cost,
        iterations: actual_iters,
    }
}
