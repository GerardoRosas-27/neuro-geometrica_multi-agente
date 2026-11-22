//! `aura-genesis`: The 4 Biological DNA Decompression & Developmental Subroutines for AURA v3.
//!
//! 1. **Chemical Gradient Routing (`MorphogenGenome` + `MorphogenRouter`)**:
//!    Compact 144-byte morphogen genome (Shh, Wnt, BMP, EphA/ephrin-A, EphB/ephrin-B) +
//!    autonomous 3D growth-cone navigation -> >1,000x compression vs explicit CSR and
//!    retinotopic neighborhood preservation (`topo_corr > 0.90`).
//! 2. **Canonical Cortical Microcircuit (`CanonicalColumnTemplate` + `CanonicalCortex`)**:
//!    Single 6-layer vertical microcircuit template (Mountcastle: L1, L2/3, L4, L5, L6 +
//!    E/I lateral inhibition) instantiated across cortical areas with ZERO modality-specific
//!    code, enabling Mriganka Sur cross-modal rewiring (`A1 -> Visual` orientation detectors).
//! 3. **Overproduction + Microglia Pruning + Physical Myelination (`DevelopmentalPruner`)**:
//!    Jesse Conel inverted-U curve (`Newborn < ChildhoodPeak >> AdolescenceCompiled`),
//!    physical removal of unused synapses from CSR + saltatory myelination (`axon_delay -> 1`).
//! 4. **Prenatal Synthetic Curriculum (`PrenatalCurriculum`)**:
//!    Spontaneous coherent 2D retinal waves, fetal motor twitches (efference copy), and
//!    synthetic rhythmic byte waves executed strictly BEFORE real data (zero leakage).

use aura_cortex::{CorticalFabric, FabricConfig, Synapse, SynapseKind, CORTEX_TILE};
use aura_math::{LatentVec, Xorshift64, LATENT_DIM};
use aura_world::EnsembleWorldModel;

// ============================================================================
// SUBROUTINE 1: Chemical Gradient Routing (MorphogenGenome + Growth Cones)
// ============================================================================

/// 3D embryonic position of a neuron inside the developing cortical sheet `[0, 1]^3`.
#[derive(Clone, Copy, Debug)]
pub struct EmbryoNeuronPos {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    /// 0 = Gamma/Sensory (L4), 1 = Beta/Assoc (L2/3), 2 = Theta/Deep-Apical (L5/L6)
    pub band: u8,
}

/// Compact 144-byte DNA morphogen genome (36 `f32` genes — ZERO explicit edge addresses!).
/// Encodes 3D chemical fields (Shh, Wnt, BMP, EphA/ephrin-A, EphB/ephrin-B) and local
/// growth-cone navigation rules.
#[derive(Clone, Copy, Debug)]
pub struct MorphogenGenome {
    // Sonic Hedgehog (Shh): ventral-dorsal laminar attractors along z
    pub shh_z_sensory: f32,
    pub shh_z_assoc: f32,
    pub shh_z_apical: f32,
    pub shh_gain: f32,

    // Wnt (anterior-posterior x) & BMP (medial-lateral y) global polarity gradients
    pub wnt_x_slope: f32,
    pub wnt_x_offset: f32,
    pub bmp_y_slope: f32,
    pub bmp_y_offset: f32,

    // EphA/ephrin-A & EphB/ephrin-B counter-gradients (Sperry retinotopic chemoaffinity)
    pub epha_gain: f32,
    pub ephb_gain: f32,
    pub eph_repulsion_bias: f32,
    pub lateral_cohesion: f32,

    // Autonomous growth-cone navigation physics
    pub cone_step_size: f32,
    pub cone_max_steps: f32,
    pub capture_radius: f32,
    pub branch_jitter: f32,

    // Developmental synaptic budgets & axonal delay scaling
    pub newborn_fanout: f32,
    pub peak_fanout: f32,
    pub ff_frac_sensory: f32,
    pub ff_frac_assoc: f32,
    pub fb_frac_apical: f32,
    pub lat_frac_assoc: f32,
    pub delay_base_unmyelinated: f32,
    pub delay_per_dist: f32,

    // Canonical laminar & oscillatory frequency genes
    pub freq_gamma_base: f32,
    pub freq_beta_base: f32,
    pub freq_theta_base: f32,
    pub exc_init_prob: f32,
    pub inh_init_prob: f32,
    pub apical_exc_bias: f32,
    pub phase_spatial_slope_x: f32,
    pub phase_spatial_slope_y: f32,
    pub gain_ff_shift: f32,
    pub gain_fb_shift: f32,
    pub prune_target_keep: f32,
    pub myelin_top_frac: f32,
}

impl Default for MorphogenGenome {
    fn default() -> Self {
        Self {
            shh_z_sensory: 0.12,
            shh_z_assoc: 0.50,
            shh_z_apical: 0.88,
            shh_gain: 2.2,

            wnt_x_slope: 1.0,
            wnt_x_offset: 0.0,
            bmp_y_slope: 1.0,
            bmp_y_offset: 0.0,

            epha_gain: 2.6,
            ephb_gain: 2.6,
            eph_repulsion_bias: 0.02,
            lateral_cohesion: 0.14,

            cone_step_size: 0.14,
            cone_max_steps: 10.0,
            capture_radius: 0.14,
            branch_jitter: 0.04,

            newborn_fanout: 10.0,
            peak_fanout: 38.0,
            ff_frac_sensory: 1.0,
            ff_frac_assoc: 0.45,
            fb_frac_apical: 0.72,
            lat_frac_assoc: 0.15,
            delay_base_unmyelinated: 2.5,
            delay_per_dist: 3.5,

            freq_gamma_base: 19.0,
            freq_beta_base: 9.0,
            freq_theta_base: 3.0,
            exc_init_prob: 0.20,
            inh_init_prob: 0.16,
            apical_exc_bias: 0.42,
            phase_spatial_slope_x: 16.0,
            phase_spatial_slope_y: 16.0,
            gain_ff_shift: 2.0,
            gain_fb_shift: 2.0,
            prune_target_keep: 0.32,
            myelin_top_frac: 0.72,
        }
    }
}

impl MorphogenGenome {
    pub const SERIALIZED_BYTES: usize = 36 * 4; // 144 bytes exact

    /// Serialize genome into exactly 144 deterministic little-endian bytes.
    pub fn to_bytes(&self) -> [u8; Self::SERIALIZED_BYTES] {
        let genes: [f32; 36] = [
            self.shh_z_sensory,
            self.shh_z_assoc,
            self.shh_z_apical,
            self.shh_gain,
            self.wnt_x_slope,
            self.wnt_x_offset,
            self.bmp_y_slope,
            self.bmp_y_offset,
            self.epha_gain,
            self.ephb_gain,
            self.eph_repulsion_bias,
            self.lateral_cohesion,
            self.cone_step_size,
            self.cone_max_steps,
            self.capture_radius,
            self.branch_jitter,
            self.newborn_fanout,
            self.peak_fanout,
            self.ff_frac_sensory,
            self.ff_frac_assoc,
            self.fb_frac_apical,
            self.lat_frac_assoc,
            self.delay_base_unmyelinated,
            self.delay_per_dist,
            self.freq_gamma_base,
            self.freq_beta_base,
            self.freq_theta_base,
            self.exc_init_prob,
            self.inh_init_prob,
            self.apical_exc_bias,
            self.phase_spatial_slope_x,
            self.phase_spatial_slope_y,
            self.gain_ff_shift,
            self.gain_fb_shift,
            self.prune_target_keep,
            self.myelin_top_frac,
        ];
        let mut out = [0u8; Self::SERIALIZED_BYTES];
        for (i, g) in genes.iter().enumerate() {
            out[i * 4..(i + 1) * 4].copy_from_slice(&g.to_le_bytes());
        }
        out
    }

    /// Deserialize genome from 144 little-endian bytes.
    pub fn from_bytes(bytes: &[u8; Self::SERIALIZED_BYTES]) -> Self {
        let mut g = [0.0f32; 36];
        for i in 0..36 {
            g[i] = f32::from_le_bytes([
                bytes[i * 4],
                bytes[i * 4 + 1],
                bytes[i * 4 + 2],
                bytes[i * 4 + 3],
            ]);
        }
        Self {
            shh_z_sensory: g[0],
            shh_z_assoc: g[1],
            shh_z_apical: g[2],
            shh_gain: g[3],
            wnt_x_slope: g[4],
            wnt_x_offset: g[5],
            bmp_y_slope: g[6],
            bmp_y_offset: g[7],
            epha_gain: g[8],
            ephb_gain: g[9],
            eph_repulsion_bias: g[10],
            lateral_cohesion: g[11],
            cone_step_size: g[12],
            cone_max_steps: g[13],
            capture_radius: g[14],
            branch_jitter: g[15],
            newborn_fanout: g[16],
            peak_fanout: g[17],
            ff_frac_sensory: g[18],
            ff_frac_assoc: g[19],
            fb_frac_apical: g[20],
            lat_frac_assoc: g[21],
            delay_base_unmyelinated: g[22],
            delay_per_dist: g[23],
            freq_gamma_base: g[24],
            freq_beta_base: g[25],
            freq_theta_base: g[26],
            exc_init_prob: g[27],
            inh_init_prob: g[28],
            apical_exc_bias: g[29],
            phase_spatial_slope_x: g[30],
            phase_spatial_slope_y: g[31],
            gain_ff_shift: g[32],
            gain_fb_shift: g[33],
            prune_target_keep: g[34],
            myelin_top_frac: g[35],
        }
    }
}

/// Chemical-gradient autonomous growth-cone router (`MorphogenRouter`).
pub struct MorphogenRouter;

impl MorphogenRouter {
    /// Local chemical concentration field `U(x, y, z)` sensed by a growth cone carrying
    /// receptor levels `(rec_epha, rec_ephb)` and laminar affinity `target_z`.
    /// Purely local continuous function of `(x, y, z)` — ZERO neuron ID lookups!
    #[inline]
    pub fn chemical_energy(
        p: (f32, f32, f32),
        rec_epha: f32,
        rec_ephb: f32,
        target_z: f32,
        g: &MorphogenGenome,
    ) -> f32 {
        // Ligand concentrations deployed along embryonic Wnt (x) & BMP (y) axes
        let lig_ephrin_a = (g.wnt_x_slope * p.0 + g.wnt_x_offset).clamp(0.0, 1.0);
        let lig_ephrin_b = (g.bmp_y_slope * p.1 + g.bmp_y_offset).clamp(0.0, 1.0);
        // Eph/ephrin matching equilibrium + Shh laminar attraction
        let da = rec_epha - lig_ephrin_a;
        let db = rec_ephb - lig_ephrin_b;
        let dz = p.2 - target_z;
        0.5 * (g.epha_gain * da * da + g.ephb_gain * db * db + g.shh_gain * dz * dz)
    }

    /// Simulate one autonomous axonal growth cone navigating strictly by local finite-difference
    /// chemical gradients (filopodia sensing at `p ± eps`). Returns `(terminal_pos, path_len)`.
    pub fn navigate_growth_cone(
        start: (f32, f32, f32),
        rec_epha: f32,
        rec_ephb: f32,
        target_z: f32,
        g: &MorphogenGenome,
        rng: &mut Xorshift64,
    ) -> ((f32, f32, f32), f32) {
        let mut p = start;
        let mut path_len = 0.0f32;
        let eps = 0.01f32;
        let max_steps = (g.cone_max_steps.round() as usize).clamp(2, 24);
        let lr = g.cone_step_size;

        for _ in 0..max_steps {
            let gx = (Self::chemical_energy((p.0 + eps, p.1, p.2), rec_epha, rec_ephb, target_z, g)
                - Self::chemical_energy((p.0 - eps, p.1, p.2), rec_epha, rec_ephb, target_z, g))
                / (2.0 * eps);
            let gy = (Self::chemical_energy((p.0, p.1 + eps, p.2), rec_epha, rec_ephb, target_z, g)
                - Self::chemical_energy((p.0, p.1 - eps, p.2), rec_epha, rec_ephb, target_z, g))
                / (2.0 * eps);
            let gz = (Self::chemical_energy((p.0, p.1, p.2 + eps), rec_epha, rec_ephb, target_z, g)
                - Self::chemical_energy((p.0, p.1, p.2 - eps), rec_epha, rec_ephb, target_z, g))
                / (2.0 * eps);

            let dx = (-lr * gx + rng.next_signed_f32() * g.branch_jitter * 0.25).clamp(-0.22, 0.22);
            let dy = (-lr * gy + rng.next_signed_f32() * g.branch_jitter * 0.25).clamp(-0.22, 0.22);
            let dz = (-lr * gz).clamp(-0.25, 0.25);

            let nx = (p.0 + dx).clamp(0.0, 1.0);
            let ny = (p.1 + dy).clamp(0.0, 1.0);
            let nz = (p.2 + dz).clamp(0.0, 1.0);
            path_len += ((nx - p.0).powi(2) + (ny - p.1).powi(2) + (nz - p.2).powi(2)).sqrt();
            p = (nx, ny, nz);
            if gx * gx + gy * gy + gz * gz < 1e-4 {
                break;
            }
        }
        (p, path_len)
    }

    /// Assign 3D embryonic coordinates on a 2D retinotopic/cortical sheet × 3 laminar bands.
    pub fn assign_embryonic_positions(n_neurons: usize, g: &MorphogenGenome) -> Vec<EmbryoNeuronPos> {
        let sensory_end = (n_neurons / 4).max(16);
        let beta_end = (n_neurons * 3) / 4;
        let mut pos = Vec::with_capacity(n_neurons);

        let place_band = |idx_in_band: usize, band_len: usize, z: f32, band: u8| -> EmbryoNeuronPos {
            let side = ((band_len as f32).sqrt().ceil() as usize).max(1);
            let gx = idx_in_band % side;
            let gy = idx_in_band / side;
            let rows = band_len.div_ceil(side).max(1);
            let x = if side > 1 {
                (gx as f32) / ((side - 1) as f32)
            } else {
                0.5
            };
            let y = if rows > 1 {
                (gy as f32) / ((rows - 1) as f32)
            } else {
                0.5
            };
            EmbryoNeuronPos { x, y, z, band }
        };

        for i in 0..n_neurons {
            if i < sensory_end {
                pos.push(place_band(i, sensory_end, g.shh_z_sensory, 0));
            } else if i < beta_end {
                pos.push(place_band(i - sensory_end, beta_end - sensory_end, g.shh_z_assoc, 1));
            } else {
                pos.push(place_band(i - beta_end, n_neurons - beta_end, g.shh_z_apical, 2));
            }
        }
        pos
    }

    /// Decompress a full `CorticalFabric` directly from the 144-byte `MorphogenGenome`
    /// using ONLY local chemical growth-cone navigation (zero explicit edge addresses!).
    pub fn decompress_fabric(
        n_neurons: usize,
        genome: &MorphogenGenome,
        fanout: usize,
        seed: u64,
        cfg: FabricConfig,
    ) -> (CorticalFabric, Vec<EmbryoNeuronPos>) {
        let mut rng = Xorshift64::new(seed ^ 0x4D4F_5250_484F_0001);
        let positions = Self::assign_embryonic_positions(n_neurons, genome);
        let fanout = fanout.max(4).min(n_neurons / 2);

        // Spatial lookup grid per band (8x8 cells) to resolve local physical synaptic contacts fast
        const GRID: usize = 8;
        let mut band_grid: [Vec<Vec<u32>>; 3] = [
            vec![Vec::new(); GRID * GRID],
            vec![Vec::new(); GRID * GRID],
            vec![Vec::new(); GRID * GRID],
        ];
        for (i, p) in positions.iter().enumerate() {
            let cx = ((p.x * (GRID as f32 - 0.001)) as usize).min(GRID - 1);
            let cy = ((p.y * (GRID as f32 - 0.001)) as usize).min(GRID - 1);
            band_grid[p.band as usize][cy * GRID + cx].push(i as u32);
        }

        let mut freqs = vec![0u8; n_neurons];
        for (i, p) in positions.iter().enumerate() {
            freqs[i] = match p.band {
                0 => (genome.freq_gamma_base as u8) + ((i as u8) & 3),
                1 => (genome.freq_beta_base as u8) + ((i as u8) & 3),
                _ => (genome.freq_theta_base as u8) + ((i as u8) & 1),
            };
        }

        let mut row_ptr = Vec::with_capacity(n_neurons + 1);
        let mut synapses = Vec::with_capacity(n_neurons * fanout);
        let mut axon_delay = vec![1u8; n_neurons];
        row_ptr.push(0);

        let mut cand_buf: Vec<(f32, u32)> = Vec::with_capacity(128);

        for src in 0..n_neurons {
            let sp = positions[src];
            let mut local_edges = Vec::with_capacity(fanout);

            // Navigate one primary growth cone per target projection band for neuron `src`
            let (primary_target_band, primary_target_z) = match sp.band {
                0 => (1usize, genome.shh_z_assoc),
                1 => (2usize, genome.shh_z_apical),
                _ => (1usize, genome.shh_z_assoc),
            };
            let (p_pri, len_pri) = Self::navigate_growth_cone(
                (sp.x, sp.y, sp.z),
                sp.x,
                sp.y,
                primary_target_z,
                genome,
                &mut rng,
            );

            // Physical unmyelinated axonal conduction delay proportional to 3D path length
            let raw_delay = (genome.delay_base_unmyelinated + len_pri * genome.delay_per_dist)
                .round()
                .clamp(2.0, 6.0) as u8;
            axon_delay[src] = raw_delay;

            // Secondary feedback cone for Assoc (Beta -> Gamma apical) and Apical recurrent
            let (p_sec, sec_band, sec_kind) = match sp.band {
                0 => (p_pri, 1usize, SynapseKind::BasalFeedforward),
                1 => {
                    let (pf, _) = Self::navigate_growth_cone(
                        (sp.x, sp.y, sp.z),
                        sp.x,
                        sp.y,
                        genome.shh_z_sensory,
                        genome,
                        &mut rng,
                    );
                    (pf, 0usize, SynapseKind::ApicalFeedback)
                }
                _ => {
                    let (pr, _) = Self::navigate_growth_cone(
                        (sp.x, sp.y, sp.z),
                        (sp.x + rng.next_signed_f32() * genome.lateral_cohesion).clamp(0.0, 1.0),
                        (sp.y + rng.next_signed_f32() * genome.lateral_cohesion).clamp(0.0, 1.0),
                        genome.shh_z_apical,
                        genome,
                        &mut rng,
                    );
                    (pr, 2usize, SynapseKind::BasalFeedforward)
                }
            };

            let n_pri = match sp.band {
                0 => fanout,
                1 => ((fanout as f32) * genome.ff_frac_assoc).round() as usize,
                _ => ((fanout as f32) * genome.fb_frac_apical).round() as usize,
            }
            .clamp(2, fanout);

            let gather_local = |term: (f32, f32, f32),
                                band: usize,
                                need: usize,
                                cbuf: &mut Vec<(f32, u32)>,
                                r: &mut Xorshift64| {
                cbuf.clear();
                let gx = ((term.0 * (GRID as f32 - 0.001)) as isize).clamp(0, GRID as isize - 1);
                let gy = ((term.1 * (GRID as f32 - 0.001)) as isize).clamp(0, GRID as isize - 1);
                for radius in [1isize, 2, 4, 8] {
                    cbuf.clear();
                    for dy in -radius..=radius {
                        let ny = gy + dy;
                        if !(0..GRID as isize).contains(&ny) {
                            continue;
                        }
                        for dx in -radius..=radius {
                            let nx = gx + dx;
                            if !(0..GRID as isize).contains(&nx) {
                                continue;
                            }
                            for &cand in &band_grid[band][(ny as usize) * GRID + (nx as usize)] {
                                if cand as usize == src {
                                    continue;
                                }
                                let cp = positions[cand as usize];
                                let d2 = (cp.x - term.0).powi(2)
                                    + (cp.y - term.1).powi(2)
                                    + r.next_f32() * genome.branch_jitter * genome.branch_jitter;
                                cbuf.push((d2, cand));
                            }
                        }
                    }
                    if cbuf.len() >= need {
                        break;
                    }
                }
                cbuf.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
            };

            let pri_kind = if sp.band == 2 {
                SynapseKind::ApicalFeedback
            } else {
                SynapseKind::BasalFeedforward
            };
            gather_local(p_pri, primary_target_band, n_pri, &mut cand_buf, &mut rng);
            for k in 0..n_pri {
                let target = cand_buf[k % cand_buf.len().max(1)].1;
                let tp = positions[target as usize];
                let r_u = rng.next_f32();
                let init_w = if pri_kind == SynapseKind::ApicalFeedback {
                    if r_u < genome.apical_exc_bias {
                        1i8
                    } else if r_u < genome.apical_exc_bias + 0.15 {
                        -1i8
                    } else {
                        0i8
                    }
                } else if r_u < genome.exc_init_prob {
                    1i8
                } else if r_u < genome.exc_init_prob + genome.inh_init_prob {
                    -1i8
                } else {
                    0i8
                };
                // Smooth topographically aligned U(1) phase gauge!
                let g_phase = (((tp.x - sp.x) * genome.phase_spatial_slope_x
                    + (tp.y - sp.y) * genome.phase_spatial_slope_y)
                    .round() as i32) as u8;
                local_edges.push(Synapse {
                    target,
                    weight: init_w,
                    latent_acc: (init_w as i16) * 80,
                    phase_gauge: g_phase,
                    gain_shift: (genome.gain_ff_shift as u8).clamp(1, 2),
                    delay: raw_delay,
                    kind: pri_kind,
                });
            }

            let n_sec = fanout.saturating_sub(n_pri);
            if n_sec > 0 {
                gather_local(p_sec, sec_band, n_sec, &mut cand_buf, &mut rng);
                for k in 0..n_sec {
                    let target = cand_buf[k % cand_buf.len().max(1)].1;
                    let tp = positions[target as usize];
                    let r_u = rng.next_f32();
                    let init_w = if sec_kind == SynapseKind::ApicalFeedback {
                        if r_u < genome.apical_exc_bias {
                            1i8
                        } else if r_u < genome.apical_exc_bias + 0.15 {
                            -1i8
                        } else {
                            0i8
                        }
                    } else if r_u < genome.exc_init_prob {
                        1i8
                    } else if r_u < genome.exc_init_prob + genome.inh_init_prob {
                        -1i8
                    } else {
                        0i8
                    };
                    let g_phase = (((tp.x - sp.x) * genome.phase_spatial_slope_x
                        + (tp.y - sp.y) * genome.phase_spatial_slope_y)
                        .round() as i32) as u8;
                    local_edges.push(Synapse {
                        target,
                        weight: init_w,
                        latent_acc: (init_w as i16) * 80,
                        phase_gauge: g_phase,
                        gain_shift: (genome.gain_fb_shift as u8).clamp(1, 2),
                        delay: raw_delay,
                        kind: sec_kind,
                    });
                }
            }

            if cfg.cache_tiled_csr {
                local_edges.sort_unstable_by_key(|s| (s.target / (CORTEX_TILE as u32), s.target));
            }
            synapses.extend(local_edges);
            row_ptr.push(synapses.len());
        }

        let mut fabric =
            CorticalFabric::from_custom_graph(n_neurons, freqs, row_ptr, synapses, axon_delay, seed, cfg);
        // Spatially coherent initial oscillator phases across the 2D sheet (retinotopic synchrony)
        for (i, p) in positions.iter().enumerate() {
            fabric.phase[i] = ((p.x * 16.0 + p.y * 16.0).round() as u8).wrapping_mul(2);
        }
        (fabric, positions)
    }

    /// Measure exact retinotopic preservation correlation between sensory `(x, y)` coordinates
    /// and the 2D centroid of each sensory neuron's outgoing cortical targets.
    /// Returns mean Pearson `r` across X and Y axes (`~0.95+` morphogenetic vs `~0.0` random).
    pub fn measure_retinotopic_correlation(
        fabric: &CorticalFabric,
        positions: &[EmbryoNeuronPos],
    ) -> f64 {
        let sensory_end = (fabric.n_neurons / 4).max(16);
        let mut sx = Vec::with_capacity(sensory_end);
        let mut sy = Vec::with_capacity(sensory_end);
        let mut tx = Vec::with_capacity(sensory_end);
        let mut ty = Vec::with_capacity(sensory_end);

        for s in 0..sensory_end {
            let start = fabric.row_ptr[s];
            let end = fabric.row_ptr[s + 1];
            if start >= end {
                continue;
            }
            let mut cx = 0.0f64;
            let mut cy = 0.0f64;
            let mut cnt = 0usize;
            for syn in &fabric.synapses[start..end] {
                let tp = positions[syn.target as usize];
                cx += tp.x as f64;
                cy += tp.y as f64;
                cnt += 1;
            }
            if cnt > 0 {
                sx.push(positions[s].x as f64);
                sy.push(positions[s].y as f64);
                tx.push(cx / (cnt as f64));
                ty.push(cy / (cnt as f64));
            }
        }
        0.5 * (pearson_corr(&sx, &tx) + pearson_corr(&sy, &ty))
    }
}

fn pearson_corr(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n < 2 {
        return 0.0;
    }
    let ma = a.iter().sum::<f64>() / (n as f64);
    let mb = b.iter().sum::<f64>() / (n as f64);
    let mut num = 0.0;
    let mut da = 0.0;
    let mut db = 0.0;
    for i in 0..n {
        let xa = a[i] - ma;
        let xb = b[i] - mb;
        num += xa * xb;
        da += xa * xa;
        db += xb * xb;
    }
    let denom = (da * db).sqrt();
    if denom < 1e-12 {
        0.0
    } else {
        num / denom
    }
}

// ============================================================================
// SUBROUTINE 2: Canonical 6-Layer Cortical Microcircuit (Mountcastle + M. Sur)
// ============================================================================

/// Universal 6-layer cortical minicolumn template (~64 bytes).
/// Identical blueprint replicated across V1, A1, Lang — ZERO modality-specific logic!
#[derive(Clone, Copy, Debug)]
pub struct CanonicalColumnTemplate {
    pub l4_dim: usize,         // Sensory granular input (64 = 8x8 patch/spectrogram/byte map)
    pub l23_exc: usize,        // Superficial excitatory feature detectors per column
    pub l5_out: usize,         // Deep pyramidal output units per column
    pub k_winners: usize,      // Local E/I lateral inhibition (k-WTA winners per column)
    pub hebb_lr: f32,          // Local unsupervised Oja/Hebbian plasticity in L4 -> L2/3
    pub lateral_gain: f32,     // Morphogen-guided L2/3 <-> L2/3 horizontal coupling
    pub l6_feedback_gain: f32, // L6 -> L4 predictive cancellation strength
    pub readout_lr: f32,       // Local supervised/reward-modulated L5 readout plasticity
}

impl Default for CanonicalColumnTemplate {
    fn default() -> Self {
        Self {
            l4_dim: 64,
            l23_exc: 16,
            l5_out: 4,
            k_winners: 3,
            hebb_lr: 0.045,
            lateral_gain: 0.15,
            l6_feedback_gain: 0.22,
            readout_lr: 0.06,
        }
    }
}

impl CanonicalColumnTemplate {
    pub const TEMPLATE_BYTES: usize = 64;
}

/// One instantiated cortical minicolumn (identical structure across all cortical areas).
#[derive(Clone, Debug)]
pub struct CorticalMinicolumn {
    pub col_id: usize,
    pub pos_x: f32,
    pub pos_y: f32,
    /// L4 -> L2/3 excitatory receptive fields `[l23_exc * l4_dim]` (normalized competitive Oja)
    pub w_l4_l23: Vec<f32>,
    /// L2/3 -> L5 pyramidal projection `[l5_out * l23_exc]`
    pub w_l23_l5: Vec<f32>,
    /// L5 -> L6 -> L4 top-down predictive weights `[l4_dim * l5_out]`
    pub w_l6_l4: Vec<f32>,
    /// Running firing rate for homeostatic threshold adaptation (ensures every neuron participates)
    pub l23_homeo: Vec<f32>,
    pub l23_act: Vec<f32>,
    pub l5_act: Vec<f32>,
    pub l6_pred: Vec<f32>,
}

impl CorticalMinicolumn {
    pub fn instantiate(col_id: usize, pos_x: f32, pos_y: f32, tpl: &CanonicalColumnTemplate, rng: &mut Xorshift64) -> Self {
        let mut w_l4_l23 = vec![0.0f32; tpl.l23_exc * tpl.l4_dim];
        for r in 0..tpl.l23_exc {
            let mut norm2 = 0.0f32;
            for c in 0..tpl.l4_dim {
                let v = rng.next_normal_f32() * 0.25;
                w_l4_l23[r * tpl.l4_dim + c] = v;
                norm2 += v * v;
            }
            let inv = 1.0 / norm2.sqrt().max(1e-5);
            for c in 0..tpl.l4_dim {
                w_l4_l23[r * tpl.l4_dim + c] *= inv;
            }
        }
        let mut w_l23_l5 = vec![0.0f32; tpl.l5_out * tpl.l23_exc];
        for v in &mut w_l23_l5 {
            *v = rng.next_normal_f32() * 0.15;
        }
        Self {
            col_id,
            pos_x,
            pos_y,
            w_l4_l23,
            w_l23_l5,
            w_l6_l4: vec![0.0f32; tpl.l4_dim * tpl.l5_out],
            l23_homeo: vec![0.1; tpl.l23_exc],
            l23_act: vec![0.0; tpl.l23_exc],
            l5_act: vec![0.0; tpl.l5_out],
            l6_pred: vec![0.0; tpl.l4_dim],
        }
    }
}

/// Massive Instantiation of the Single `CanonicalColumnTemplate` into a multi-column cortical
/// region (`V1`, `A1`, `Lang`, or `A1_rewired_visual` — ZERO modality-specific code!).
#[derive(Clone, Debug)]
pub struct CanonicalCortex {
    pub tpl: CanonicalColumnTemplate,
    pub columns: Vec<CorticalMinicolumn>,
    /// Morphogen-derived horizontal neighbor weights `[col_i][col_j]` (Eph/ephrin proximity)
    pub lateral_adj: Vec<Vec<(usize, f32)>>,
    /// Unified L5 population readout `[n_classes * total_l23_plus_l5]`
    pub n_classes: usize,
    pub readout_w: Vec<f32>,
    pub readout_b: Vec<f32>,
    /// True = canonical 6-layer local inhibition + Oja; False = unstructured random ablation
    pub canonical_enabled: bool,
}

impl CanonicalCortex {
    pub fn instantiate(
        n_cols: usize,
        n_classes: usize,
        tpl: CanonicalColumnTemplate,
        seed: u64,
        canonical_enabled: bool,
    ) -> Self {
        let mut rng = Xorshift64::new(seed ^ 0xCA40_41CA_0002);
        let side = ((n_cols as f32).sqrt().ceil() as usize).max(1);
        let mut columns = Vec::with_capacity(n_cols);
        for c in 0..n_cols {
            let x = ((c % side) as f32) / ((side.saturating_sub(1)).max(1) as f32);
            let y = ((c / side) as f32) / ((n_cols.div_ceil(side).saturating_sub(1)).max(1) as f32);
            columns.push(CorticalMinicolumn::instantiate(c, x, y, &tpl, &mut rng));
        }

        // Morphogenetic horizontal coupling between neighboring minicolumns
        let mut lateral_adj = vec![Vec::new(); n_cols];
        for i in 0..n_cols {
            for j in 0..n_cols {
                if i == j {
                    continue;
                }
                let d2 = (columns[i].pos_x - columns[j].pos_x).powi(2)
                    + (columns[i].pos_y - columns[j].pos_y).powi(2);
                if d2 <= 0.30 {
                    lateral_adj[i].push((j, (-d2 * 4.0).exp() * tpl.lateral_gain));
                }
            }
        }

        let feat_dim = n_cols * tpl.l23_exc;
        Self {
            tpl,
            columns,
            lateral_adj,
            n_classes,
            readout_w: vec![0.0f32; n_classes * feat_dim],
            readout_b: vec![0.0f32; n_classes],
            canonical_enabled,
        }
    }

    /// Forward pass through L4 -> L2/3 (with E/I k-WTA lateral inhibition) -> L5 -> L6 feedback.
    /// Takes ANY normalized 64D sensory vector — completely modality-agnostic!
    pub fn forward_sensory(&mut self, input_64: &[f32; 64], update_unsupervised: bool) -> Vec<f32> {
        let d = self.tpl.l4_dim;
        let m = self.tpl.l23_exc;

        // Normalize sensory input gently (zero-mean unit-norm L4 thalamic preprocessing)
        let mean = input_64.iter().sum::<f32>() / (d as f32);
        let mut x = [0.0f32; 64];
        let mut n2 = 0.0f32;
        for i in 0..d {
            x[i] = input_64[i] - mean;
            n2 += x[i] * x[i];
        }
        let inv_n = 1.0 / n2.sqrt().max(1e-4);
        for v in &mut x {
            *v *= inv_n;
        }

        let mut pop_feat = Vec::with_capacity(self.columns.len() * m);

        for col in &mut self.columns {
            // L4 receives thalamic input modulated by L6 apical predictive cancellation
            let mut l4_drive = [0.0f32; 64];
            for i in 0..d {
                l4_drive[i] = x[i] - self.tpl.l6_feedback_gain * col.l6_pred[i];
            }

            // L4 -> L2/3 linear drive
            let mut raw = vec![0.0f32; m];
            for r in 0..m {
                let row = &col.w_l4_l23[r * d..(r + 1) * d];
                let mut acc = 0.0f32;
                for i in 0..d {
                    acc += row[i] * l4_drive[i];
                }
                raw[r] = acc;
            }

            if self.canonical_enabled {
                // Local inhibitory interneurons (k-WTA competitive sparsity + ON/OFF simple-cell
                // polarity invariance, standard in cortical L2/3 across V1/A1/S1!)
                let mut ranked: Vec<(f32, usize)> = raw
                    .iter()
                    .enumerate()
                    .map(|(idx, &val)| (val.abs() - 0.30 * (col.l23_homeo[idx] - 0.18), idx))
                    .collect();
                ranked.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));

                col.l23_act.fill(0.0);
                for rank in 0..self.tpl.k_winners.min(m) {
                    let idx = ranked[rank].1;
                    col.l23_act[idx] = raw[idx].abs();
                }

                // Unsupervised local competitive Hebbian/Oja plasticity (sign-aligned for ON/OFF bars!)
                if update_unsupervised {
                    let winner = ranked[0].1;
                    let sgn = if raw[winner] >= 0.0 { 1.0f32 } else { -1.0f32 };
                    let lr = self.tpl.hebb_lr;
                    let row = &mut col.w_l4_l23[winner * d..(winner + 1) * d];
                    let mut rn2 = 0.0f32;
                    for i in 0..d {
                        row[i] += lr * (sgn * x[i] - row[i]);
                        rn2 += row[i] * row[i];
                    }
                    let inv = 1.0 / rn2.sqrt().max(1e-5);
                    for i in 0..d {
                        row[i] *= inv;
                    }
                    if m > 1 {
                        let second = ranked[1].1;
                        let sgn2 = if raw[second] >= 0.0 { 1.0f32 } else { -1.0f32 };
                        let row2 = &mut col.w_l4_l23[second * d..(second + 1) * d];
                        let mut rn2b = 0.0f32;
                        for i in 0..d {
                            row2[i] += (lr * 0.25) * (sgn2 * x[i] - row2[i]);
                            rn2b += row2[i] * row2[i];
                        }
                        let inv2 = 1.0 / rn2b.sqrt().max(1e-5);
                        for i in 0..d {
                            row2[i] *= inv2;
                        }
                    }
                    for r in 0..m {
                        let fired = if r == winner { 1.0 } else { 0.0 };
                        col.l23_homeo[r] = 0.96 * col.l23_homeo[r] + 0.04 * fired;
                    }
                    for i in 0..d {
                        col.l6_pred[i] = 0.95 * col.l6_pred[i] + 0.05 * x[i];
                    }
                }
            } else {
                // Unstructured random ablation: no laminar lateral inhibition, no competitive Oja
                for r in 0..m {
                    col.l23_act[r] = raw[r].max(0.0);
                }
            }
        }

        // Morphogen-guided horizontal L2/3 <-> L2/3 lateral contextual integration
        let prev_acts: Vec<Vec<f32>> = self.columns.iter().map(|c| c.l23_act.clone()).collect();
        for (i, col) in self.columns.iter_mut().enumerate() {
            for r in 0..m {
                let mut lat = 0.0f32;
                if self.canonical_enabled {
                    for &(nb, w) in &self.lateral_adj[i] {
                        lat += w * prev_acts[nb][r];
                    }
                }
                pop_feat.push(col.l23_act[r] + lat);
            }
        }
        pop_feat
    }

    /// Predict class logits from the canonical L2/3->L5 population code.
    pub fn predict_class(&self, feat: &[f32]) -> (usize, Vec<f32>) {
        let mut logits = vec![0.0f32; self.n_classes];
        let fd = feat.len();
        for c in 0..self.n_classes {
            let mut acc = self.readout_b[c];
            let off = c * fd;
            for j in 0..fd {
                acc += self.readout_w[off + j] * feat[j];
            }
            logits[c] = acc;
        }
        let mut best = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (c, &v) in logits.iter().enumerate() {
            if v > best_v {
                best_v = v;
                best = c;
            }
        }
        (best, logits)
    }

    /// Local softmax delta-rule update on L5 readout weights ONLY (keeps L4->L2/3 100% unsupervised!).
    pub fn train_readout_step(&mut self, feat: &[f32], target: usize) {
        let (_, logits) = self.predict_class(feat);
        let max_l = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut probs = vec![0.0f32; self.n_classes];
        let mut sum_e = 0.0f32;
        for c in 0..self.n_classes {
            let e = (logits[c] - max_l).exp();
            probs[c] = e;
            sum_e += e;
        }
        let inv = 1.0 / sum_e.max(1e-12);
        for p in &mut probs {
            *p *= inv;
        }
        let lr = self.tpl.readout_lr;
        let fd = feat.len();
        for c in 0..self.n_classes {
            let err = (if c == target { 1.0 } else { 0.0 }) - probs[c];
            self.readout_b[c] += lr * err;
            let off = c * fd;
            for j in 0..fd {
                self.readout_w[off + j] =
                    self.readout_w[off + j] * 0.9995 + lr * err * feat[j];
            }
        }
    }

    /// Reset ONLY the L5 readout head (used when testing zero-shot prenatal representations or
    /// switching modalities cleanly).
    pub fn reset_readout(&mut self) {
        self.readout_w.fill(0.0);
        self.readout_b.fill(0.0);
    }

    /// Neurophysiological Orientation Selectivity Index (OSI) across all L2/3 excitatory neurons:
    /// `OSI_i = (R_pref - R_orth) / (R_pref + R_orth + eps)` measured directly on un-clipped
    /// receptive-field tuning curves `R_i(theta) = |w_i · x_norm|` with plasticity FROZEN!
    /// Input `orient_patterns`: 4 sets of 8x8 visual patterns at `[0°, 45°, 90°, 135°]`.
    pub fn measure_osi(&mut self, orient_patterns: &[Vec<[f32; 64]>; 4]) -> f64 {
        let n_cols = self.columns.len();
        let m = self.tpl.l23_exc;
        let d = self.tpl.l4_dim;
        let total = n_cols * m;
        let mut mean_resp = vec![[0.0f64; 4]; total];

        for (ori, pats) in orient_patterns.iter().enumerate() {
            for pat in pats {
                let mean = pat.iter().sum::<f32>() / (d as f32);
                let mut x = [0.0f32; 64];
                let mut n2 = 0.0f32;
                for i in 0..d {
                    x[i] = pat[i] - mean;
                    n2 += x[i] * x[i];
                }
                let inv_n = 1.0 / n2.sqrt().max(1e-4);
                for v in &mut x {
                    *v *= inv_n;
                }

                for (ci, col) in self.columns.iter().enumerate() {
                    for r in 0..m {
                        let row = &col.w_l4_l23[r * d..(r + 1) * d];
                        let mut acc = 0.0f32;
                        for i in 0..d {
                            acc += row[i] * x[i];
                        }
                        mean_resp[ci * m + r][ori] += acc.abs() as f64;
                    }
                }
            }
            let denom = pats.len().max(1) as f64;
            for r in &mut mean_resp {
                r[ori] /= denom;
            }
        }

        let mut osi_sum = 0.0f64;
        for r in &mean_resp {
            let mut pref = 0usize;
            for o in 1..4 {
                if r[o] > r[pref] {
                    pref = o;
                }
            }
            let orth = (pref + 2) % 4; // 0° <-> 90°, 45° <-> 135°
            let r_pref = r[pref];
            let r_orth = r[orth];
            if r_pref > 1e-4 {
                osi_sum += (r_pref - r_orth) / (r_pref + r_orth + 1e-6);
            }
        }
        osi_sum / (total.max(1) as f64)
    }

    /// Top-down L2/3 -> L4 Predictive Coding Laplace Free Energy (`||x_norm - x_hat||_1 * 300.0`)
    /// averaged across all instantiated canonical minicolumns, using each column's winning
    /// sparse L2/3 excitatory code to generate its top-down prediction `x_hat = (w_best · x_norm) w_best`.
    /// Zero-mean unit-norm input eliminates DC brightness confounds!
    pub fn predictive_free_energy(&self, sensory: &[f32; 64]) -> f64 {
        let d = self.tpl.l4_dim;
        let m = self.tpl.l23_exc;
        let mean = sensory.iter().sum::<f32>() / (d as f32);
        let mut x = [0.0f32; 64];
        let mut n2 = 0.0f32;
        for i in 0..d {
            x[i] = sensory[i] - mean;
            n2 += x[i] * x[i];
        }
        let inv_n = 1.0 / n2.sqrt().max(1e-4);
        for v in &mut x {
            *v *= inv_n;
        }

        let mut fe_sum = 0.0f64;
        for col in &self.columns {
            // Select the winning sparse L2/3 neuron (largest |w_r · x|)
            let mut best_r = 0usize;
            let mut best_proj = 0.0f32;
            for r in 0..m {
                let row = &col.w_l4_l23[r * d..(r + 1) * d];
                let mut acc = 0.0f32;
                for i in 0..d {
                    acc += row[i] * x[i];
                }
                if acc.abs() > best_proj.abs() {
                    best_proj = acc;
                    best_r = r;
                }
            }
            let w_best = &col.w_l4_l23[best_r * d..(best_r + 1) * d];
            let mut l1_res = 0.0f64;
            for i in 0..d {
                let x_hat = best_proj * w_best[i];
                l1_res += (x[i] - x_hat).abs() as f64;
            }
            fe_sum += l1_res * 300.0;
        }
        fe_sum / (self.columns.len().max(1) as f64)
    }
}

// ============================================================================
// SUBROUTINE 3: Overproduction + Microglia Pruning + Physical Myelination
// ============================================================================

/// Summary of the Jesse Conel developmental compilation cycle (`Newborn -> Peak -> Compiled`).
#[derive(Clone, Copy, Debug)]
pub struct DevelopmentalReport {
    pub synapses_newborn: usize,
    pub synapses_peak: usize,
    pub synapses_compiled: usize,
    pub csr_bytes_peak: usize,
    pub csr_bytes_compiled: usize,
    pub prune_frac: f64,
    pub myelinated_neurons: usize,
    pub mean_delay_before: f64,
    pub mean_delay_after: f64,
}

/// Biological developmental compiler executing Overproduction -> Microglia Pruning -> Myelination.
pub struct DevelopmentalPruner;

impl DevelopmentalPruner {
    /// Execute physical Microglia garbage collection + Oligodendrocyte saltatory myelination
    /// on an experienced `CorticalFabric`.
    ///
    /// - Physically deletes the lowest-utility `(1 - keep_frac)` synapses from `fabric.synapses`,
    ///   rebuilds `fabric.row_ptr` compactly, and re-tiles by `CORTEX_TILE` for L1 cache locality.
    /// - Prioritizes active predictive feedback pathways and balanced sparse feedforward routes
    ///   (never resurrecting dead zero synapses or saturating basal gain!).
    /// - Myelinate the top `myelin_top_frac` most active axons: reduces `fabric.axon_delay[src]`
    ///   from unmyelinated `3..6` ticks down to **`1` tick** (saltatory conduction!).
    pub fn microglia_prune_and_myelinate(
        fabric: &mut CorticalFabric,
        synapses_newborn: usize,
        keep_frac: f32,
        myelin_top_frac: f32,
    ) -> DevelopmentalReport {
        let synapses_peak = fabric.synapses.len();
        let csr_bytes_peak = fabric.csr_memory_bytes();
        let mean_delay_before = fabric.axon_delay.iter().map(|&d| d as f64).sum::<f64>()
            / (fabric.n_neurons.max(1) as f64);

        let keep_frac = keep_frac.clamp(0.15, 0.85);
        let has_usage = fabric.syn_usage.len() == fabric.synapses.len();

        let mut new_row_ptr = Vec::with_capacity(fabric.n_neurons + 1);
        let mut new_synapses = Vec::with_capacity(
            ((synapses_peak as f32) * keep_frac) as usize + fabric.n_neurons * 4,
        );
        let mut neuron_activity = vec![(0u64, 0usize); fabric.n_neurons];
        new_row_ptr.push(0);

        let mut scored_ff: Vec<(u64, Synapse)> = Vec::with_capacity(64);
        let mut scored_fb: Vec<(u64, Synapse)> = Vec::with_capacity(64);
        for src in 0..fabric.n_neurons {
            let start = fabric.row_ptr[src];
            let end = fabric.row_ptr[src + 1];
            scored_ff.clear();
            scored_fb.clear();

            for e_idx in start..end {
                let syn = fabric.synapses[e_idx];
                let u = if has_usage { fabric.syn_usage[e_idx] as u64 } else { 0 };
                // Microglia C1q/C3 selection:
                // - For ApicalFeedback: ONLY positive aligned weights (`latent_acc > 0`) cancel
                //   sensory Free Energy; negative/anti-predictive feedback is marked for pruning!
                // - For BasalFeedforward: reward consistent non-zero activity.
                let score = match syn.kind {
                    SynapseKind::ApicalFeedback => {
                        let pos_acc = syn.latent_acc.max(0) as u64;
                        let pos_w = if syn.weight > 0 { 128u64 } else { 0u64 };
                        pos_w + pos_acc * 3 + if syn.weight > 0 { u * 4 } else { 0 }
                    }
                    SynapseKind::BasalFeedforward => {
                        let acc_mag = syn.latent_acc.unsigned_abs() as u64;
                        let nz = if syn.weight != 0 { 24u64 } else { 0u64 };
                        nz + acc_mag * 2 + u * 4
                    }
                };
                match syn.kind {
                    SynapseKind::BasalFeedforward => scored_ff.push((score, syn)),
                    SynapseKind::ApicalFeedback => scored_fb.push((score, syn)),
                }
            }

            scored_ff.sort_unstable_by(|a, b| b.0.cmp(&a.0));
            scored_fb.sort_unstable_by(|a, b| b.0.cmp(&a.0));

            let n_total_keep = (((end - start) as f32) * keep_frac)
                .round()
                .max(4.0)
                .min((end - start) as f32) as usize;
            let n_ff_keep = if scored_fb.is_empty() {
                n_total_keep.min(scored_ff.len())
            } else if scored_ff.is_empty() {
                0
            } else {
                (((scored_ff.len() as f32) * keep_frac).round() as usize)
                    .clamp(2, scored_ff.len())
                    .min(n_total_keep.saturating_sub(2))
            };
            let n_fb_keep = n_total_keep.saturating_sub(n_ff_keep).min(scored_fb.len());

            let mut kept_edges: Vec<Synapse> = Vec::with_capacity(n_ff_keep + n_fb_keep);
            let mut act_sum = 0u64;
            for &(sc, syn) in scored_ff.iter().take(n_ff_keep) {
                act_sum += sc;
                kept_edges.push(syn);
            }
            for &(sc, syn) in scored_fb.iter().take(n_fb_keep) {
                act_sum += sc;
                kept_edges.push(syn);
            }
            neuron_activity[src] = (act_sum, src);

            if fabric.cfg.cache_tiled_csr {
                kept_edges.sort_unstable_by_key(|s| (s.target / (CORTEX_TILE as u32), s.target));
            }
            new_synapses.extend(kept_edges);
            new_row_ptr.push(new_synapses.len());
        }

        // Physical replacement & memory shrink (true RAM reduction!)
        new_synapses.shrink_to_fit();
        fabric.row_ptr = new_row_ptr;
        fabric.synapses = new_synapses;
        fabric.syn_usage.clear();

        // Oligodendrocyte Myelination + Turrigiano Homeostatic Synaptic Scaling:
        // Wrap top active neurons' axons -> delay = 1 tick (saltatory conduction!).
        // Homeostatically scale myelinated apical conductance (`gain_shift = 3`) to match the
        // higher spike arrival rate so fast top-down predictions cancel error cleanly without overshoot!
        neuron_activity.sort_unstable_by(|a, b| b.0.cmp(&a.0));
        let n_myelin = ((fabric.n_neurons as f32) * myelin_top_frac.clamp(0.1, 0.95)).round() as usize;
        for &(_, nid) in neuron_activity.iter().take(n_myelin) {
            fabric.axon_delay[nid] = 1;
            let s0 = fabric.row_ptr[nid];
            let s1 = fabric.row_ptr[nid + 1];
            for syn in &mut fabric.synapses[s0..s1] {
                syn.delay = 1;
                syn.gain_shift = if syn.kind == SynapseKind::ApicalFeedback { 3 } else { 2 };
            }
        }

        let synapses_compiled = fabric.synapses.len();
        let csr_bytes_compiled = fabric.csr_memory_bytes();
        let prune_frac = 1.0 - (synapses_compiled as f64) / (synapses_peak.max(1) as f64);
        let mean_delay_after = fabric.axon_delay.iter().map(|&d| d as f64).sum::<f64>()
            / (fabric.n_neurons.max(1) as f64);

        DevelopmentalReport {
            synapses_newborn,
            synapses_peak,
            synapses_compiled,
            csr_bytes_peak,
            csr_bytes_compiled,
            prune_frac,
            myelinated_neurons: n_myelin,
            mean_delay_before,
            mean_delay_after,
        }
    }
}

// ============================================================================
// SUBROUTINE 4: Prenatal Synthetic Curriculum (Retinal Waves & Motor Twitches)
// ============================================================================

/// Generates spontaneous coherent 2D cholinergic **Retinal Waves** (`8x8` traveling wavefronts),
/// fetal **Motor Twitches** (efference copy -> proprioceptive echo), and pure synthetic
/// **Rhythmic Byte Waves** (0 bytes from any real test corpus -> ZERO data leakage!).
pub struct PrenatalCurriculum;

impl PrenatalCurriculum {
    /// Generate one 2D 8x8 traveling retinal wave frame at phase `t_norm ∈ [0, 1]` with
    /// propagation angle `theta` and spatial width `sigma`.
    pub fn retinal_wave_frame(theta: f32, t_norm: f32, sigma: f32) -> [f32; 64] {
        let mut frame = [0.0f32; 64];
        let nx = theta.cos();
        let ny = theta.sin();
        // Wavefront travels across [-1.1, +1.1] along direction (nx, ny)
        let front = -1.1 + 2.2 * t_norm;
        let inv_2s2 = 1.0 / (2.0 * sigma * sigma).max(1e-4);

        for y in 0..8 {
            let fy = ((y as f32) - 3.5) / 3.5;
            for x in 0..8 {
                let fx = ((x as f32) - 3.5) / 3.5;
                let proj = fx * nx + fy * ny;
                let dist = proj - front;
                frame[y * 8 + x] = (-dist * dist * inv_2s2).exp();
            }
        }
        frame
    }

    /// Generate a sequence of `n_waves` spontaneous prenatal retinal waves (each lasting 6 ticks).
    pub fn generate_retinal_waves(n_waves: usize, seed: u64) -> Vec<[f32; 64]> {
        let mut rng = Xorshift64::new(seed ^ 0x5245_5449_4E41_0004);
        let mut frames = Vec::with_capacity(n_waves * 6);
        for _ in 0..n_waves {
            let theta = rng.next_f32() * std::f32::consts::TAU;
            let sigma = 0.24 + 0.12 * rng.next_f32();
            for step in 0..6 {
                let t_norm = (step as f32) / 5.0;
                frames.push(Self::retinal_wave_frame(theta, t_norm, sigma));
            }
        }
        frames
    }

    /// Negative control: same number of frames as `generate_retinal_waves`, but i.i.d. uniform
    /// white noise (ZERO spatial or temporal coherence!).
    pub fn generate_white_noise_frames(n_frames: usize, seed: u64) -> Vec<[f32; 64]> {
        let mut rng = Xorshift64::new(seed ^ 0x4E4F_4953_4500_0004);
        let mut frames = Vec::with_capacity(n_frames);
        for _ in 0..n_frames {
            let mut f = [0.0f32; 64];
            for v in &mut f {
                *v = rng.next_f32();
            }
            frames.push(f);
        }
        frames
    }

    /// Energy- and histogram-matched spatially shuffled negative control: takes the exact pixel
    /// intensities of each retinal wave frame and applies an independent Fisher-Yates permutation
    /// across the 64 pixels per frame. Matches mean brightness, L1/L2 energy, and variance
    /// bit-for-bit while destroying 100% of spatial and temporal coherence!
    pub fn generate_shuffled_noise_frames(reference: &[[f32; 64]], seed: u64) -> Vec<[f32; 64]> {
        let mut rng = Xorshift64::new(seed ^ 0x5348_5546_464C_0004);
        let mut frames = Vec::with_capacity(reference.len());
        for src in reference {
            let mut f = *src;
            for i in (1..64).rev() {
                let j = rng.next_usize(i + 1);
                f.swap(i, j);
            }
            frames.push(f);
        }
        frames
    }

    /// Deterministic biomechanical proprioceptive echo `s_next = Body(s_prev, motor_twitch)` for
    /// 8 fetal muscle groups (pure physics of the musculoskeletal body, no external world yet!).
    pub fn simulate_fetal_body_echo(
        state: &LatentVec,
        action_id: u16,
        noise_amp: i32,
        rng: &mut Xorshift64,
    ) -> LatentVec {
        let a = (action_id as usize) % 8;
        let mut next = LatentVec::zero();
        for d in 0..LATENT_DIM {
            // Smooth muscle-specific proprioceptive topographic activation + gentle decay of prev state
            let center = (a * 32 + 16) as i32;
            let dist = ((d as i32) - center).abs().min(LATENT_DIM as i32 - ((d as i32) - center).abs());
            let muscle_echo = if dist < 24 {
                ((24 - dist) * 4) as i32
            } else if (dist - 64).abs() < 16 {
                // Antagonist muscle stretch (negative proprioception)
                -((16 - (dist - 64).abs()) * 3) as i32
            } else {
                0
            };
            let prev_carry = (state.data[d] as i32) >> 2;
            let jitter = if noise_amp > 0 {
                ((rng.next_u32() % (2 * noise_amp as u32 + 1)) as i32) - noise_amp
            } else {
                0
            };
            next.data[d] = (muscle_echo + prev_carry + jitter).clamp(-127, 127) as i8;
        }
        next
    }

    /// Encode explicit spinal efference copy (`action_id`) combined with proprioceptive posture
    /// into a 256D sensorimotor query vector (faithful to biological corollary discharge).
    pub fn encode_efference_copy(state: &LatentVec, action_id: u16) -> LatentVec {
        let a = (action_id as usize) % 8;
        let mut out = LatentVec::zero();
        for d in 0..LATENT_DIM {
            // Half bandwidth: explicit motor command efference copy (orthogonally coded per muscle)
            // Half bandwidth: decayed current proprioceptive posture
            let motor_band = if d / 32 == a {
                96i32
            } else if ((d / 32) + 4) % 8 == a {
                -48i32
            } else {
                0i32
            };
            let proprio = (state.data[d] as i32) >> 2;
            out.data[d] = (motor_band + proprio).clamp(-127, 127) as i8;
        }
        out
    }

    /// Run fetal motor-twitch pre-training on an `EnsembleWorldModel` (or white-noise control).
    /// Twitches fire in spontaneous random order (REM fetal sleep), pairing motor efference copy
    /// with biomechanical proprioceptive feedback!
    pub fn run_fetal_motor_twitches(
        world: &mut EnsembleWorldModel,
        n_twitches: usize,
        white_noise_control: bool,
        seed: u64,
    ) {
        let mut rng = Xorshift64::new(seed ^ 0x5457_4954_4348_0004);
        let mut state = LatentVec::zero();
        for _ in 0..n_twitches {
            let action = rng.next_usize(8) as u16;
            let eff_key = Self::encode_efference_copy(&state, action);
            let next = if white_noise_control {
                let mut v = LatentVec::zero();
                for d in 0..LATENT_DIM {
                    v.data[d] = ((rng.next_u32() % 200) as i32 - 100) as i8;
                }
                v
            } else {
                Self::simulate_fetal_body_echo(&state, action, 4, &mut rng)
            };
            let ent = if white_noise_control { 7.7 } else { 2.4 };
            let _ = world.observe_and_decompose(&eff_key, action, &next, ent);
            state = next;
        }
    }

    /// Pure synthetic prenatal rhythmic byte waves for pre-calibrating byte-level circuits
    /// (`AuraByteModel` / `CorticalFabric`) BEFORE seeing byte 0 of any real repository file!
    /// Uniformly spans ALL printable ASCII classes (all 26 lowercase/uppercase letters, digits,
    /// spaces, newlines, punctuation, and balanced hierarchical brackets) — ZERO repo vocabulary!
    pub fn generate_prenatal_byte_waves(n_bytes: usize, seed: u64) -> Vec<u8> {
        let mut rng = Xorshift64::new(seed ^ 0x5052_4542_5954_0004);
        let mut out = Vec::with_capacity(n_bytes + 64);

        let vowels: [u8; 5] = *b"aeiou";
        let consonants: [u8; 21] = *b"bcdfghjklmnpqrstvwxyz";
        let delims: [(u8, u8); 4] = [(b'(', b')'), (b'[', b']'), (b'{', b'}'), (b'`', b'`')];

        let push_synth_word = |buf: &mut Vec<u8>, r: &mut Xorshift64, cap: bool| {
            let len = 3 + r.next_usize(5);
            for k in 0..len {
                let ch = if (k & 1) == 0 {
                    consonants[r.next_usize(consonants.len())]
                } else {
                    vowels[r.next_usize(vowels.len())]
                };
                buf.push(if cap && k == 0 { ch - 32 } else { ch });
            }
        };

        let mut line_idx = 0usize;
        while out.len() < n_bytes {
            match line_idx % 4 {
                0 => {
                    // Balanced prose rhythm across entire a..z / A..Z alphabet
                    let n_words = 4 + rng.next_usize(4);
                    for w in 0..n_words {
                        if w > 0 {
                            out.push(b' ');
                        }
                        push_synth_word(&mut out, &mut rng, w == 0);
                    }
                    out.push(if (line_idx & 1) == 0 { b'.' } else { b',' });
                    out.push(b'\n');
                }
                1 => {
                    // Balanced hierarchical bracket/delimiter wave rhythm
                    let (open, close) = delims[rng.next_usize(delims.len())];
                    out.push(b'-');
                    out.push(b' ');
                    out.push(open);
                    push_synth_word(&mut out, &mut rng, false);
                    out.push(b':');
                    out.push(b' ');
                    out.push(b'0' + (rng.next_usize(10) as u8));
                    out.push(b'.');
                    out.push(b'0' + (rng.next_usize(10) as u8));
                    out.push(close);
                    out.push(b'\n');
                }
                2 => {
                    // Structured pipe-delimited numeric/label wave
                    out.push(b'|');
                    out.push(b' ');
                    push_synth_word(&mut out, &mut rng, false);
                    out.push(b' ');
                    out.push(b'|');
                    out.push(b' ');
                    out.push(b'0' + (rng.next_usize(10) as u8));
                    out.push(b'0' + (rng.next_usize(10) as u8));
                    out.push(b'.');
                    out.push(b'0' + (rng.next_usize(10) as u8));
                    out.push(b' ');
                    out.push(b'|');
                    out.push(b'\n');
                }
                _ => {
                    // Indented key_name = value; wave
                    out.extend_from_slice(b"    ");
                    push_synth_word(&mut out, &mut rng, false);
                    out.push(b'_');
                    push_synth_word(&mut out, &mut rng, false);
                    out.extend_from_slice(b" = ");
                    out.push(b'0' + (rng.next_usize(10) as u8));
                    out.push(b';');
                    out.push(b'\n');
                }
            }
            line_idx += 1;
        }
        out.truncate(n_bytes);
        out
    }

    /// Generate 8x8 oriented bar visual stimulus (`ori ∈ 0..4` -> `0°, 45°, 90°, 135°`)
    /// with random spatial shift and noise.
    pub fn visual_oriented_bar(ori: usize, shift: f32, noise_amp: f32, rng: &mut Xorshift64) -> [f32; 64] {
        Self::visual_oriented_bar_polarity(ori, shift, 1.0, noise_amp, rng)
    }

    /// Generate 8x8 oriented bar visual stimulus with explicit ON (+1) or OFF (-1) contrast
    /// polarity around mid-gray 0.5 (tests genuine contrast-invariant cortical edge detection!).
    pub fn visual_oriented_bar_polarity(
        ori: usize,
        shift: f32,
        polarity: f32,
        noise_amp: f32,
        rng: &mut Xorshift64,
    ) -> [f32; 64] {
        let theta = (ori as f32) * (std::f32::consts::PI / 4.0);
        let nx = -theta.sin();
        let ny = theta.cos();
        let mut pat = [0.0f32; 64];
        for y in 0..8 {
            let fy = ((y as f32) - 3.5) / 3.5;
            for x in 0..8 {
                let fx = ((x as f32) - 3.5) / 3.5;
                let d = fx * nx + fy * ny - shift;
                let bar = (-d * d / (2.0 * 0.22 * 0.22)).exp();
                let base = 0.5 + 0.5 * polarity * (bar * 2.0 - 0.35);
                pat[y * 8 + x] = (base + rng.next_signed_f32() * noise_amp).clamp(0.0, 1.2);
            }
        }
        pat
    }

    /// Generate 8x8 time-frequency auditory spectrogram (`cls ∈ 0..4`:
    /// 0 = rising chirp, 1 = falling chirp, 2 = harmonic dual-tone chord, 3 = broadband impulse burst).
    pub fn auditory_spectrogram(cls: usize, shift: f32, noise_amp: f32, rng: &mut Xorshift64) -> [f32; 64] {
        let mut pat = [0.0f32; 64];
        for f_bin in 0..8 {
            let f = (f_bin as f32) / 7.0;
            for t_bin in 0..8 {
                let t = (t_bin as f32) / 7.0;
                let val = match cls % 4 {
                    0 => {
                        let target_f = (0.15 + 0.70 * t + shift * 0.15).clamp(0.0, 1.0);
                        (-(f - target_f).powi(2) / 0.035).exp()
                    }
                    1 => {
                        let target_f = (0.85 - 0.70 * t + shift * 0.15).clamp(0.0, 1.0);
                        (-(f - target_f).powi(2) / 0.035).exp()
                    }
                    2 => {
                        let b1 = (-(f - (0.25 + shift * 0.1)).powi(2) / 0.02).exp();
                        let b2 = (-(f - (0.72 + shift * 0.1)).powi(2) / 0.02).exp();
                        (b1 + b2).min(1.0)
                    }
                    _ => {
                        let center_t = (0.5 + shift * 0.2).clamp(0.15, 0.85);
                        (-(t - center_t).powi(2) / 0.025).exp()
                    }
                };
                pat[f_bin * 8 + t_bin] = (val + rng.next_signed_f32() * noise_amp).clamp(0.0, 1.2);
            }
        }
        pat
    }

    /// Encode a raw byte window into a 64D translation-invariant sensory feature map
    /// (32 fine ASCII class/character bins + 32 hashed bigram-transition texture bins).
    pub fn bytes_to_sensory_64(slice: &[u8]) -> [f32; 64] {
        let mut pat = [0.0f32; 64];
        if slice.is_empty() {
            return pat;
        }
        let inv_n = 1.0 / (slice.len() as f32);
        // Channels 0..16: coarse ASCII histogram; 16..32: fine syntactic character-class detectors
        for &b in slice {
            pat[(b as usize) >> 4] += inv_n;
            let cls = if b == b' ' {
                16
            } else if b == b'\n' || b == b'\t' {
                17
            } else if b.is_ascii_digit() {
                18
            } else if b == b'.' || b == b',' {
                19
            } else if b == b';' || b == b':' {
                20
            } else if b == b'{' || b == b'}' {
                21
            } else if b == b'(' || b == b')' || b == b'[' || b == b']' {
                22
            } else if b == b'#' || b == b'*' || b == b'|' {
                23
            } else if b == b'_' || b == b'-' || b == b'=' || b == b'>' {
                24
            } else if b.is_ascii_uppercase() {
                25
            } else if matches!(b, b'a' | b'e' | b'i' | b'o' | b'u') {
                26
            } else if b.is_ascii_lowercase() {
                27
            } else if b >= 128 {
                28
            } else {
                29
            };
            pat[cls] += inv_n;
        }
        // Channels 32..64: consecutive-byte bigram transition texture (captures `fn`, `::`, `**`, `->`, `| `)
        if slice.len() >= 2 {
            let inv_b = 1.0 / ((slice.len() - 1) as f32);
            for w in slice.windows(2) {
                let h = ((w[0] as usize).wrapping_mul(31) ^ (w[1] as usize).wrapping_mul(17)) & 31;
                pat[32 + h] += inv_b;
            }
        }
        pat
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn morphogen_genome_roundtrip_and_retinotopic_compression() {
        let g = MorphogenGenome::default();
        let bytes = g.to_bytes();
        assert_eq!(bytes.len(), 144);
        assert!(bytes.len() <= 256);
        let g2 = MorphogenGenome::from_bytes(&bytes);
        assert!((g.epha_gain - g2.epha_gain).abs() < 1e-6);

        let (fabric, pos) = MorphogenRouter::decompress_fabric(
            2048,
            &g,
            24,
            0xA600,
            FabricConfig::default(),
        );
        let comp = (fabric.csr_memory_bytes() as f64) / (bytes.len() as f64);
        let topo = MorphogenRouter::measure_retinotopic_correlation(&fabric, &pos);
        assert!(comp > 1000.0, "Expected >1000x genome compression, got {comp:.1}x");
        assert!(topo > 0.90, "Expected retinotopic corr > 0.90, got {topo:.4}");
    }

    #[test]
    fn microglia_prune_follows_conel_curve_and_reduces_delay() {
        let g = MorphogenGenome::default();
        let mut cfg = FabricConfig::default();
        cfg.myelin_delay_enabled = true;
        let (mut fabric, _) = MorphogenRouter::decompress_fabric(512, &g, 38, 0xA601, cfg);
        let syn_newborn = 512 * 10;
        fabric.enable_usage_tracking();
        for t in 0..64 {
            let b = (t % 16) as u8;
            let _ = fabric.settle_window_amortized(b.wrapping_sub(1), b, 2, 8, 12);
        }
        let rep = DevelopmentalPruner::microglia_prune_and_myelinate(&mut fabric, syn_newborn, 0.32, 0.72);
        assert!(rep.synapses_peak > rep.synapses_newborn * 3);
        assert!(rep.prune_frac >= 0.60, "Expected >=60% pruning, got {:.3}", rep.prune_frac);
        assert!(rep.csr_bytes_compiled < rep.csr_bytes_peak / 2);
        assert!(rep.mean_delay_after < rep.mean_delay_before);
    }

    #[test]
    fn prenatal_retinal_waves_calibrate_orientation_selectivity_before_birth() {
        let tpl = CanonicalColumnTemplate::default();
        let mut c_waves = CanonicalCortex::instantiate(8, 4, tpl, 0xA602, true);
        let mut c_none = CanonicalCortex::instantiate(8, 4, tpl, 0xA602, true);

        let waves = PrenatalCurriculum::generate_retinal_waves(80, 0xA602);
        for frame in &waves {
            let _ = c_waves.forward_sensory(frame, true);
        }

        let mut rng = Xorshift64::new(999);
        let mut test_oris: [Vec<[f32; 64]>; 4] = Default::default();
        for o in 0..4 {
            for k in 0..8 {
                let sh = ((k as f32) / 7.0 - 0.5) * 0.5;
                test_oris[o].push(PrenatalCurriculum::visual_oriented_bar(o, sh, 0.05, &mut rng));
            }
        }
        let osi_waves = c_waves.measure_osi(&test_oris);
        let osi_none = c_none.measure_osi(&test_oris);
        assert!(
            osi_waves > osi_none + 0.15,
            "Expected prenatal retinal waves OSI ({osi_waves:.3}) >> blank ({osi_none:.3})"
        );
    }
}
