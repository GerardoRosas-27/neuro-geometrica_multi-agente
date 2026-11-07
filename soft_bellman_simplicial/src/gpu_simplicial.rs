//! Aceleración GPU WGPU para Soft-Bellman Simplicial 3D v3 (AMD RDNA3 / Radeon RX 7800 XT)
//!
//! Ejecuta en paralelo mediante Compute Shaders WGSL:
//! 1. Evaluación de costos métricos SPD 3x3 + modulación espectral Hodge-1 + base no lineal Fourier-Polinomial Phi + holonomía B = d1 A.
//! 2. Iteración de valor de frente de onda Soft-Bellman diferenciable sobre complejos simpliciales 3D aumentados (Steiner + refinados).
//! 3. Determinantes de volumen de Cayley-Menger y barrera logarítmica por tetraedro en GPU.

use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};
use std::time::Instant;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuSimplicialConfig {
    pub num_states: u32,
    pub num_edges: u32,
    pub num_tets: u32,
    pub goal_state: u32,
    pub tau: f32,
    pub eps: f32,
    pub step_layer: u32,
    pub pad0: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuEdgeInput {
    pub src: u32,
    pub dst: u32,
    pub num_faces: u32,
    pub carrier_dim: u32,
    pub faces: [u32; 8],
    pub dx: f32,
    pub dy: f32,
    pub dz: f32,
    pub hodge_down: f32,
    pub hodge_up: f32,
    pub holonomy_b: f32,
    pub pad0: f32,
    pub pad1: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuStateSpan {
    pub edge_start: u32,
    pub edge_count: u32,
    pub is_goal: u32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuTetInput {
    pub u: [f32; 4], // ux, uy, uz, hodge_scale
    pub v: [f32; 4], // vx, vy, vz, pad
    pub w: [f32; 4], // wx, wy, wz, pad
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GpuExecutionTelemetry {
    pub adapter_name: String,
    pub adapter_backend: String,
    pub driver_info: String,
    pub num_states_dispatched: usize,
    pub num_edges_dispatched: usize,
    pub num_tets_dispatched: usize,
    pub bellman_layers: usize,
    pub gpu_dispatch_ms: f64,
    pub max_cost_abs_diff_vs_cpu: f64,
    pub max_value_abs_diff_vs_cpu: f64,
    pub min_cm_volume2_gpu: f64,
    pub mean_cm_log_barrier_gpu: f64,
}

pub const WGSL_SIMPLICIAL_COMPUTE: &str = r#"
struct Config {
    num_states: u32,
    num_edges: u32,
    num_tets: u32,
    goal_state: u32,
    tau: f32,
    eps: f32,
    step_layer: u32,
    pad0: u32,
};

struct EdgeInput {
    src: u32,
    dst: u32,
    num_faces: u32,
    carrier_dim: u32,
    f0: u32,
    f1: u32,
    f2: u32,
    f3: u32,
    f4: u32,
    f5: u32,
    f6: u32,
    f7: u32,
    dx: f32,
    dy: f32,
    dz: f32,
    hodge_down: f32,
    hodge_up: f32,
    holonomy_b: f32,
    pad0: f32,
    pad1: f32,
};

struct StateSpan {
    edge_start: u32,
    edge_count: u32,
    is_goal: u32,
    pad: u32,
};

struct TetInput {
    u: vec4<f32>,
    v: vec4<f32>,
    w: vec4<f32>,
};

@group(0) @binding(0) var<uniform> cfg: Config;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read> edges: array<EdgeInput>;
@group(0) @binding(3) var<storage, read> spans: array<StateSpan>;
@group(0) @binding(4) var<storage, read> tets: array<TetInput>;
@group(0) @binding(5) var<storage, read_write> edge_costs: array<f32>;
@group(0) @binding(6) var<storage, read_write> v_curr: array<f32>;
@group(0) @binding(7) var<storage, read_write> v_next: array<f32>;
@group(0) @binding(8) var<storage, read_write> tet_cm_vol2: array<f32>;

fn sp(x: f32) -> f32 {
    if (x > 25.0) {
        return x;
    }
    if (x < -25.0) {
        return exp(x);
    }
    return log(1.0 + exp(x));
}

fn transform_vec(d: vec3<f32>) -> vec3<f32> {
    let l00 = 0.5 + sp(weights[0]);
    let l11 = 0.5 + sp(weights[1]);
    let l22 = 0.5 + sp(weights[2]);
    let l01 = 0.8 * tanh(weights[3]);
    let l02 = 0.8 * tanh(weights[4]);
    let l12 = 0.8 * tanh(weights[5]);
    return vec3<f32>(
        l00 * d.x + l01 * d.y + l02 * d.z,
        l11 * d.y + l12 * d.z,
        l22 * d.z
    );
}

fn spd_inner(a: vec3<f32>, b: vec3<f32>) -> f32 {
    let ta = transform_vec(a);
    let tb = transform_vec(b);
    return dot(ta, tb) + cfg.eps * dot(a, b);
}

@compute @workgroup_size(64)
fn compute_simplicial_costs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= cfg.num_edges) {
        return;
    }
    let e = edges[idx];
    let d = vec3<f32>(e.dx, e.dy, e.dz);
    let len_spd = sqrt(max(spd_inner(d, d), 1e-12));

    // Modulación intrínseca guiada por el Laplaciano de Hodge 1-cochain
    let hodge_mod = exp(
        0.15 * tanh(weights[15]) * e.hodge_down +
        0.15 * tanh(weights[16]) * e.hodge_up
    );
    let len_intrinsic = len_spd * hodge_mod;

    // Base no lineal de Fourier-Polinomios evaluada exactamente por cada cara incidente f \in e.faces
    let f_arr = array<u32, 8>(e.f0, e.f1, e.f2, e.f3, e.f4, e.f5, e.f6, e.f7);
    let nf = min(max(e.num_faces, 1u), 8u);
    var phi_sum: f32 = 0.0;
    for (var k: u32 = 0u; k < nf; k = k + 1u) {
        let fid = f_arr[k];
        let a = weights[20u + 2u * fid];
        let b = weights[20u + 2u * fid + 1u];
        let da = a - 0.5;
        let db = b - 0.5;
        let z = weights[6]
              + weights[7] * a
              + weights[8] * b
              + weights[9] * (da * da)
              + weights[10] * (db * db)
              + weights[11] * (da * db)
              + weights[12] * sin(7.0 * a)
              + weights[13] * cos(7.0 * a)
              + weights[14] * sin(7.0 * b);
        phi_sum = phi_sum + (0.3 + sp(z));
    }
    let phi = phi_sum / f32(nf);

    // 2-cocadena de flujo magnético / holonomía de coborde B = d1 A
    let holonomy_factor = exp(0.10 * tanh(weights[17]) * e.holonomy_b);

    edge_costs[idx] = max(len_intrinsic * phi * holonomy_factor, 1e-6);
}

@compute @workgroup_size(64)
fn soft_bellman_wavefront_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    if (x >= cfg.num_states) {
        return;
    }
    let span = spans[x];
    if (span.is_goal != 0u) {
        v_next[x] = 0.0;
        return;
    }
    if (span.edge_count == 0u) {
        v_next[x] = 1e30;
        return;
    }
    var m: f32 = 1e30;
    for (var k: u32 = 0u; k < span.edge_count; k = k + 1u) {
        let eid = span.edge_start + k;
        let dst = edges[eid].dst;
        let q = edge_costs[eid] + v_curr[dst];
        if (q < m) {
            m = q;
        }
    }
    if (m >= 1e29) {
        v_next[x] = 1e30;
        return;
    }
    var sum_exp: f32 = 0.0;
    for (var k: u32 = 0u; k < span.edge_count; k = k + 1u) {
        let eid = span.edge_start + k;
        let dst = edges[eid].dst;
        let q = edge_costs[eid] + v_curr[dst];
        let diff = clamp((q - m) / cfg.tau, 0.0, 80.0);
        sum_exp = sum_exp + exp(-diff);
    }
    v_next[x] = m - cfg.tau * log(max(sum_exp, 1e-20));
}

@compute @workgroup_size(64)
fn compute_cayley_menger_volumes(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= cfg.num_tets) {
        return;
    }
    let t = tets[tid];
    let u = t.u.xyz;
    let v = t.v.xyz;
    let w = t.w.xyz;
    let scale = max(t.u.w, 0.25);

    let g00 = spd_inner(u, u) * scale;
    let g01 = spd_inner(u, v) * scale;
    let g02 = spd_inner(u, w) * scale;
    let g11 = spd_inner(v, v) * scale;
    let g12 = spd_inner(v, w) * scale;
    let g22 = spd_inner(w, w) * scale;

    let det_gram = g00 * (g11 * g22 - g12 * g12)
                 - g01 * (g01 * g22 - g12 * g02)
                 + g02 * (g01 * g12 - g11 * g02);
    tet_cm_vol2[tid] = max(det_gram / 36.0, 1e-12);
}
"#;

pub struct GpuSimplicialEngine {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_name: String,
    pub adapter_backend: String,
    pub driver_info: String,
    pub cost_pipeline: wgpu::ComputePipeline,
    pub bellman_pipeline: wgpu::ComputePipeline,
    pub cm_pipeline: wgpu::ComputePipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuSimplicialEngine {
    pub fn new() -> Self {
        let instance = wgpu::Instance::default();
        let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));

        let mut chosen = None;
        for a in adapters {
            let info = a.get_info();
            if info.name.contains("AMD")
                || info.name.contains("Radeon")
                || info.name.contains("7800")
                || info.name.contains("NVIDIA")
            {
                chosen = Some(a);
                break;
            }
        }
        let adapter = chosen.unwrap_or_else(|| {
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            }))
            .expect("No GPU adapter available")
        });

        let info = adapter.get_info();
        let adapter_name = info.name.clone();
        let adapter_backend = format!("{:?}", info.backend);
        let driver_info = format!("{} ({})", info.driver, info.driver_info);

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Soft-Bellman Simplicial 3D v3 GPU Device"),
            ..Default::default()
        }))
        .expect("Failed to create WGPU device");

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Simplicial 3D v3 WGSL Shader"),
            source: wgpu::ShaderSource::Wgsl(WGSL_SIMPLICIAL_COMPUTE.into()),
        });

        let entry = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Simplicial 3D v3 Bind Group Layout"),
            entries: &[
                entry(0, wgpu::BufferBindingType::Uniform),
                entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
                entry(3, wgpu::BufferBindingType::Storage { read_only: true }),
                entry(4, wgpu::BufferBindingType::Storage { read_only: true }),
                entry(5, wgpu::BufferBindingType::Storage { read_only: false }),
                entry(6, wgpu::BufferBindingType::Storage { read_only: false }),
                entry(7, wgpu::BufferBindingType::Storage { read_only: false }),
                entry(8, wgpu::BufferBindingType::Storage { read_only: false }),
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Simplicial 3D v3 Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let cost_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Simplicial Cost Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("compute_simplicial_costs"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bellman_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Soft-Bellman Wavefront Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("soft_bellman_wavefront_step"),
            compilation_options: Default::default(),
            cache: None,
        });

        let cm_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Cayley-Menger Volume Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("compute_cayley_menger_volumes"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            device,
            queue,
            adapter_name,
            adapter_backend,
            driver_info,
            cost_pipeline,
            bellman_pipeline,
            cm_pipeline,
            bind_group_layout,
        }
    }

    pub fn execute_batch_forward_gpu(
        &self,
        weights_18: &[f64],
        face_ab: &[[f32; 2]],
        edges: &[GpuEdgeInput],
        spans: &[GpuStateSpan],
        tets: &[GpuTetInput],
        horizon: usize,
        tau: f64,
        eps: f64,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, f64) {
        let start = Instant::now();
        let mut w32 = vec![0.0f32; 20 + face_ab.len() * 2];
        for (i, &val) in weights_18.iter().take(20).enumerate() {
            w32[i] = val as f32;
        }
        for (fid, &[a, b]) in face_ab.iter().enumerate() {
            w32[20 + 2 * fid] = a;
            w32[20 + 2 * fid + 1] = b;
        }

        let num_states = spans.len() as u32;
        let num_edges = edges.len() as u32;
        let num_tets = tets.len().max(1) as u32;

        let cfg = GpuSimplicialConfig {
            num_states,
            num_edges,
            num_tets: tets.len() as u32,
            goal_state: 0,
            tau: tau as f32,
            eps: eps as f32,
            step_layer: 0,
            pad0: 0,
        };

        let mut v_init = vec![1e30f32; spans.len()];
        for (i, s) in spans.iter().enumerate() {
            if s.is_goal != 0 {
                v_init[i] = 0.0;
            }
        }

        let dummy_tet = [GpuTetInput {
            u: [1.0, 0.0, 0.0, 1.0],
            v: [0.0, 1.0, 0.0, 0.0],
            w: [0.0, 0.0, 1.0, 0.0],
        }];
        let tet_slice = if tets.is_empty() { &dummy_tet[..] } else { tets };

        let cfg_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Config Buffer"),
            contents: bytemuck::bytes_of(&cfg),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let w_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Weights Buffer"),
            contents: bytemuck::cast_slice(&w32),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let e_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Edges Buffer"),
            contents: bytemuck::cast_slice(edges),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let s_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Spans Buffer"),
            contents: bytemuck::cast_slice(spans),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let t_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tets Buffer"),
            contents: bytemuck::cast_slice(tet_slice),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let cost_size = (edges.len().max(1) * 4) as u64;
        let state_size = (spans.len().max(1) * 4) as u64;
        let tet_size = (tet_slice.len() * 4) as u64;

        let cost_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Edge Costs Buffer"),
            size: cost_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let v_a_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("V Buffer A"),
            contents: bytemuck::cast_slice(&v_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });
        let v_b_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("V Buffer B"),
            contents: bytemuck::cast_slice(&v_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });
        let cm_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("CM Vol2 Buffer"),
            size: tet_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let cost_readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cost Readback"),
            size: cost_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let v_readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("V Readback"),
            size: state_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let cm_readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("CM Readback"),
            size: tet_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let make_bg = |va: &wgpu::Buffer, vb: &wgpu::Buffer| {
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Simplicial Step BG"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: cfg_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: w_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: e_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: s_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: t_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: cost_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: va.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 7, resource: vb.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 8, resource: cm_buf.as_entire_binding() },
                ],
            })
        };

        let bg_ab = make_bg(&v_a_buf, &v_b_buf);
        let bg_ba = make_bg(&v_b_buf, &v_a_buf);

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Simplicial v3 Command Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Cost + CM Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.cost_pipeline);
            pass.set_bind_group(0, &bg_ab, &[]);
            pass.dispatch_workgroups((num_edges + 63) / 64, 1, 1);

            pass.set_pipeline(&self.cm_pipeline);
            pass.set_bind_group(0, &bg_ab, &[]);
            pass.dispatch_workgroups((num_tets + 63) / 64, 1, 1);
        }

        let state_wg = (num_states + 63) / 64;
        for step in 0..horizon {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Soft-Bellman Wavefront Step"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.bellman_pipeline);
            if step % 2 == 0 {
                pass.set_bind_group(0, &bg_ab, &[]);
            } else {
                pass.set_bind_group(0, &bg_ba, &[]);
            }
            pass.dispatch_workgroups(state_wg, 1, 1);
        }

        let final_v_buf = if horizon % 2 == 0 { &v_a_buf } else { &v_b_buf };
        encoder.copy_buffer_to_buffer(&cost_buf, 0, &cost_readback, 0, cost_size);
        encoder.copy_buffer_to_buffer(final_v_buf, 0, &v_readback, 0, state_size);
        encoder.copy_buffer_to_buffer(&cm_buf, 0, &cm_readback, 0, tet_size);

        let sub_idx = self.queue.submit(std::iter::once(encoder.finish()));

        let read_vec = |buf: &wgpu::Buffer, wait_idx: Option<wgpu::SubmissionIndex>| -> Vec<f32> {
            let slice = buf.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |res| {
                let _ = tx.send(res);
            });
            if let Some(idx) = wait_idx {
                let _ = self.device.poll(wgpu::PollType::Wait {
                    submission_index: Some(idx),
                    timeout: None,
                });
            } else {
                let _ = self.device.poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                });
            }
            let _ = rx.recv().unwrap();
            let mapped = slice.get_mapped_range().unwrap();
            let out: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
            drop(mapped);
            buf.unmap();
            out
        };

        let costs_out = read_vec(&cost_readback, Some(sub_idx));
        let v_out = read_vec(&v_readback, None);
        let cm_out = read_vec(&cm_readback, None);
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

        (costs_out, v_out, cm_out, elapsed_ms)
    }
}
