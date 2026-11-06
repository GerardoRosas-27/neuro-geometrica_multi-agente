//! Entrenador GPU (wgpu/WGSL). Sin carreras: ver train.wgsl.
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params { n: u32, f: u32, d: u32, nwg: u32, lr: f32, mom: f32, _p0: f32, _p1: f32 }

pub const NWG: u32 = 64;

#[allow(dead_code)]
pub struct Gpu {
    device: wgpu::Device, queue: wgpu::Queue,
    pub adapter_name: String,
    cap: usize, f: usize, d: usize,
    params: wgpu::Buffer, phi: wgpu::Buffer, y: wgpu::Buffer, w: wgpu::Buffer, v: wgpu::Buffer,
    err: wgpu::Buffer, partial: wgpu::Buffer, rb_partial: wgpu::Buffer, rb_w: wgpu::Buffer,
    p_fwd: wgpu::ComputePipeline, p_upd: wgpu::ComputePipeline, p_red: wgpu::ComputePipeline,
    bg_fwd: wgpu::BindGroup, bg_upd: wgpu::BindGroup, bg_red: wgpu::BindGroup,
}

fn sbuf(dev: &wgpu::Device, label: &str, bytes: u64) -> wgpu::Buffer {
    dev.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: bytes.max(16),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false })
}

impl Gpu {
    /// `cap`: tamaño máx. de batch. `lam`: lambda por feature. `w0`: pesos iniciales (f x d).
    pub fn new(cap: usize, f: usize, d: usize, lam: &[f32], w0: &[f32]) -> Result<Self, String> {
        let instance = wgpu::Instance::default();
        let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
        let mut chosen = None;
        for a in adapters { let n = a.get_info().name; if n.contains("AMD") || n.contains("Radeon") { chosen = Some(a); break; } }
        let adapter = match chosen { Some(a) => a, None => pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() })).map_err(|e| format!("sin adaptador GPU: {e:?}"))? };
        let info = adapter.get_info();
        let mut limits = wgpu::Limits::default();
        let al = adapter.limits();
        limits.max_storage_buffer_binding_size = al.max_storage_buffer_binding_size;
        limits.max_buffer_size = al.max_buffer_size;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("gemma_campo"), required_limits: limits, ..Default::default() })).map_err(|e| format!("{e:?}"))?;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("train.wgsl"), source: wgpu::ShaderSource::Wgsl(crate::WGSL.into()) });
        let fb = 4u64;
        let params = device.create_buffer(&wgpu::BufferDescriptor { label: Some("params"), size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let phi = sbuf(&device, "phi", (cap * f) as u64 * fb);
        let y = sbuf(&device, "y", (cap * d) as u64 * fb);
        let w = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("w"), contents: bytemuck::cast_slice(w0),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC });
        let v = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("v"), contents: bytemuck::cast_slice(&vec![0f32; f * d]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC });
        let lamb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("lam"), contents: bytemuck::cast_slice(lam),
            usage: wgpu::BufferUsages::STORAGE });
        let err = sbuf(&device, "err", (cap * d) as u64 * fb);
        let partial = sbuf(&device, "partial", NWG as u64 * fb);
        let rb = |l: &str, s: u64| device.create_buffer(&wgpu::BufferDescriptor { label: Some(l), size: s,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let rb_partial = rb("rb_partial", NWG as u64 * fb);
        let rb_w = rb("rb_w", (f * d) as u64 * fb);
        let pipe = |ep: &str| device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(ep), layout: None, module: &module,
            entry_point: Some(ep), compilation_options: Default::default(), cache: None });
        let (p_fwd, p_upd, p_red) = (pipe("k_forward"), pipe("k_update"), pipe("k_reduce"));
        let bg = |p: &wgpu::ComputePipeline, binds: &[(u32, &wgpu::Buffer)]| {
            let entries: Vec<wgpu::BindGroupEntry> = binds.iter().map(|(b, buf)| wgpu::BindGroupEntry { binding: *b, resource: buf.as_entire_binding() }).collect();
            device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &p.get_bind_group_layout(0), entries: &entries })
        };
        let bg_fwd = bg(&p_fwd, &[(0, &params), (1, &phi), (2, &y), (3, &w), (5, &err)]);
        let bg_upd = bg(&p_upd, &[(0, &params), (1, &phi), (3, &w), (4, &v), (5, &err), (6, &lamb)]);
        let bg_red = bg(&p_red, &[(0, &params), (5, &err), (7, &partial)]);
        Ok(Gpu { device, queue, adapter_name: format!("{} ({:?})", info.name, info.backend), cap, f, d, params, phi, y, w, v, err, partial,
            rb_partial, rb_w, p_fwd, p_upd, p_red, bg_fwd, bg_upd, bg_red })
    }

    fn read(&self, src: &wgpu::Buffer, dst: &wgpu::Buffer, n: usize) -> Vec<f32> {
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(src, 0, dst, 0, (n * 4) as u64);
        let idx = self.queue.submit(Some(enc.finish()));
        let slice = dst.slice(..(n * 4) as u64);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.device.poll(wgpu::PollType::Wait { submission_index: Some(idx), timeout: None });
        let out = { let data = slice.get_mapped_range().expect("map"); bytemuck::cast_slice::<u8, f32>(&data).to_vec() };
        dst.unmap();
        out
    }

    /// Un paso: err con W actual, pérdida de datos (antes de actualizar) reducida en árbol, luego W,V <- update.
    /// `update=false` -> solo forward+pérdida (para val o tests).
    pub fn step(&self, phi: &[f32], y: &[f32], n: usize, lr: f32, mom: f32, update: bool) -> f64 {
        assert!(n <= self.cap && n > 0);
        let p = Params { n: n as u32, f: self.f as u32, d: self.d as u32, nwg: NWG, lr, mom, _p0: 0.0, _p1: 0.0 };
        self.queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&p));
        self.queue.write_buffer(&self.phi, 0, bytemuck::cast_slice(&phi[..n * self.f]));
        self.queue.write_buffer(&self.y, 0, bytemuck::cast_slice(&y[..n * self.d]));
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            cp.set_pipeline(&self.p_fwd); cp.set_bind_group(0, &self.bg_fwd, &[]);
            cp.dispatch_workgroups(((n * self.d) as u32).div_ceil(128), 1, 1);
        }
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            cp.set_pipeline(&self.p_red); cp.set_bind_group(0, &self.bg_red, &[]);
            cp.dispatch_workgroups(NWG, 1, 1);
        }
        if update {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            cp.set_pipeline(&self.p_upd); cp.set_bind_group(0, &self.bg_upd, &[]);
            cp.dispatch_workgroups(((self.f * self.d) as u32).div_ceil(128), 1, 1);
        }
        self.queue.submit(Some(enc.finish()));
        let parts = self.read(&self.partial, &self.rb_partial, NWG as usize);
        parts.iter().map(|x| *x as f64).sum::<f64>() / (n * self.d) as f64
    }

    pub fn weights(&self) -> Vec<f32> { self.read(&self.w, &self.rb_w, self.f * self.d) }
}
