use std::collections::{HashMap, HashSet};

#[derive(Clone, Copy, Debug)]
pub struct EntanglementConfig {
    pub max_links_per_node: usize,
    pub max_syncs_per_step: usize,
    pub create_threshold: f32,
    pub coherence_gain: f32,
    pub entropy_decay: f32,
    pub contradiction_gain: f32,
    pub max_entropy: f32,
    pub max_heat: f32,
}

impl Default for EntanglementConfig {
    fn default() -> Self {
        Self {
            max_links_per_node: 6,
            max_syncs_per_step: 256,
            create_threshold: 2.0,
            coherence_gain: 0.08,
            entropy_decay: 0.04,
            contradiction_gain: 0.35,
            max_entropy: 1.0,
            max_heat: 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct EntanglementLink {
    pub a: usize,
    pub b: usize,
    pub coherence: f32,
    pub entropy: f32,
    pub heat: f32,
    pub last_sync_tick: u64,
    pub active: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct EntanglementReport {
    pub created: usize,
    pub synced: usize,
    pub pruned: usize,
    pub conflicts: usize,
    pub active_links: usize,
    pub mean_coherence: f32,
    pub mean_entropy: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PredictiveEntanglementReport {
    pub accepted: bool,
    pub link_updated: bool,
    pub conflict: bool,
    pub effective_benefit: f32,
    pub field: EntanglementReport,
}

#[derive(Clone, Debug)]
pub struct EntanglementField {
    pub config: EntanglementConfig,
    links: Vec<EntanglementLink>,
    lookup: HashMap<(usize, usize), usize>,
    adjacency: HashMap<usize, Vec<usize>>,
    active_degree: HashMap<usize, usize>,
    active_count_cache: usize,
    pending_benefit: HashMap<(usize, usize), f32>,
    sync_frontier: Vec<usize>,
    sync_present: HashSet<usize>,
    tick: u64,
}

impl EntanglementField {
    pub fn new(config: EntanglementConfig) -> Self {
        Self {
            config,
            links: Vec::new(),
            lookup: HashMap::new(),
            adjacency: HashMap::new(),
            active_degree: HashMap::new(),
            active_count_cache: 0,
            pending_benefit: HashMap::new(),
            sync_frontier: Vec::new(),
            sync_present: HashSet::new(),
            tick: 0,
        }
    }

    pub fn active_count(&self) -> usize {
        self.active_count_cache
    }

    pub fn link_slots(&self) -> usize {
        self.links.len()
    }

    /// Elimina físicamente enlaces inactivos y reconstruye índices. Se ejecuta
    /// en mantenimiento/sueño, nunca en la ruta realtime.
    pub fn compact_inactive(&mut self) -> usize {
        let before = self.links.len();
        if before == self.active_count_cache {
            return 0;
        }
        self.links.retain(|link| link.active);
        self.lookup.clear();
        self.adjacency.clear();
        self.active_degree.clear();
        self.active_count_cache = 0;
        for (idx, link) in self.links.iter().enumerate() {
            self.lookup.insert(ordered_pair(link.a, link.b), idx);
            self.adjacency.entry(link.a).or_default().push(idx);
            self.adjacency.entry(link.b).or_default().push(idx);
            *self.active_degree.entry(link.a).or_insert(0) += 1;
            *self.active_degree.entry(link.b).or_insert(0) += 1;
            self.active_count_cache += 1;
        }
        before - self.links.len()
    }

    /// Enlaces activos con su utilidad local para políticas de reasignación.
    /// La utilidad penaliza incoherencia, entropía y calor; no representa física literal.
    pub fn active_link_entries(&self) -> impl Iterator<Item = EntanglementLink> + '_ {
        self.links.iter().copied().filter(|link| link.active)
    }

    /// Libera, como máximo, un enlace de baja utilidad que toca `node`.
    /// Se usa para reservar capacidad EPR sin aumentar el límite de grado.
    pub fn evict_lowest_utility_touching(&mut self, node: usize) -> Option<EntanglementLink> {
        let index = self
            .adjacency
            .get(&node)?
            .iter()
            .copied()
            .filter(|&idx| self.links[idx].active)
            .min_by(|&left, &right| {
                link_utility(self.links[left]).total_cmp(&link_utility(self.links[right]))
            })?;
        self.deactivate_index(index);
        Some(self.links[index])
    }

    pub fn max_links_per_node(&self) -> usize {
        self.config.max_links_per_node
    }

    pub fn active_links_for_node(&self, node: usize) -> usize {
        self.node_link_count(node)
    }

    pub fn has_active_link(&self, a: usize, b: usize) -> bool {
        self.lookup
            .get(&ordered_pair(a, b))
            .is_some_and(|&idx| self.links[idx].active)
    }

    /// Reserva una cuota para un par prospectivo desalojando enlaces de menor utilidad.
    /// Devuelve cuántos enlaces fueron desactivados.
    pub fn reserve_pair_capacity(
        &mut self,
        a: usize,
        b: usize,
        prospective_slots_per_node: usize,
    ) -> usize {
        if a == b || self.has_active_link(a, b) {
            return 0;
        }
        let max = self.config.max_links_per_node.max(1);
        let reserve = prospective_slots_per_node.min(max.saturating_sub(1));
        let threshold = max.saturating_sub(reserve);
        let mut evicted = 0;
        for node in [a, b] {
            if self.node_link_count(node) >= threshold
                && self.evict_lowest_utility_touching(node).is_some()
            {
                evicted += 1;
            }
        }
        evicted
    }

    pub fn set_max_links_per_node(&mut self, max_links: usize) {
        self.config.max_links_per_node = max_links.max(1);
    }

    /// Poda estructural de mantenimiento: aplica límite de grado y después
    /// conserva globalmente los enlaces activos de mayor utilidad.
    pub fn prune_to_budget(&mut self, max_active: usize, max_per_node: usize) -> usize {
        let before = self.active_count_cache;
        let max_per_node = max_per_node.max(1);
        self.config.max_links_per_node = max_per_node;
        let nodes = self.active_degree.keys().copied().collect::<Vec<_>>();
        for node in nodes {
            while self.node_link_count(node) > max_per_node {
                if self.evict_lowest_utility_touching(node).is_none() {
                    break;
                }
            }
        }
        let max_active = max_active.max(1);
        if self.active_count_cache > max_active {
            let mut active = self
                .links
                .iter()
                .enumerate()
                .filter(|(_, link)| link.active)
                .map(|(index, link)| (index, link_utility(*link)))
                .collect::<Vec<_>>();
            active.sort_by(|left, right| left.1.total_cmp(&right.1));
            let remove = self.active_count_cache - max_active;
            for (index, _) in active.into_iter().take(remove) {
                self.deactivate_index(index);
            }
        }
        self.compact_inactive();
        before.saturating_sub(self.active_count_cache)
    }

    pub fn set_create_threshold(&mut self, threshold: f32) {
        self.config.create_threshold = threshold.max(0.0);
    }

    pub fn observe_correlation(&mut self, a: usize, b: usize, benefit: f32) -> bool {
        self.observe_correlation_with_reserve(a, b, benefit, 0).0
    }

    /// EPR clásico guiado por utilidad predictiva, no por coincidencia repetida.
    ///
    /// Un error alto inyecta contradicción sobre un enlace existente; un error
    /// bajo convierte la mejora predictiva en beneficio de creación/refuerzo.
    pub fn observe_predictive_correlation(
        &mut self,
        a: usize,
        b: usize,
        predictive_gain: f32,
        prediction_error: f32,
        max_prediction_error: f32,
    ) -> PredictiveEntanglementReport {
        let prediction_error = prediction_error.abs();
        let max_prediction_error = max_prediction_error.abs().max(f32::EPSILON);
        if prediction_error > max_prediction_error {
            let field = self.inject_conflict(a, b);
            return PredictiveEntanglementReport {
                conflict: true,
                field,
                ..PredictiveEntanglementReport::default()
            };
        }
        let reliability = 1.0 - prediction_error / max_prediction_error;
        let effective_benefit = predictive_gain.max(0.0) * reliability.clamp(0.0, 1.0);
        let link_updated = self.observe_correlation(a, b, effective_benefit);
        PredictiveEntanglementReport {
            accepted: true,
            link_updated,
            effective_benefit,
            field: self.summary(),
            ..PredictiveEntanglementReport::default()
        }
    }

    /// Observación EPR para la ruta realtime. Solo reserva capacidad cuando el
    /// beneficio acumulado cruza el umbral de creación.
    pub fn observe_correlation_with_reserve(
        &mut self,
        a: usize,
        b: usize,
        benefit: f32,
        prospective_slots_per_node: usize,
    ) -> (bool, usize) {
        if a == b {
            return (false, 0);
        }
        let key = ordered_pair(a, b);
        let score = self.pending_benefit.entry(key).or_insert(0.0);
        *score += benefit.max(0.0);
        if *score < self.config.create_threshold {
            return (false, 0);
        }
        *score = 0.0;
        let evicted = self.reserve_pair_capacity(a, b, prospective_slots_per_node);
        (self.create_or_reinforce(a, b), evicted)
    }

    pub fn create_or_reinforce(&mut self, a: usize, b: usize) -> bool {
        let key = ordered_pair(a, b);
        if let Some(&idx) = self.lookup.get(&key) {
            if !self.links[idx].active
                && (self.node_link_count(a) >= self.config.max_links_per_node
                    || self.node_link_count(b) >= self.config.max_links_per_node)
            {
                return false;
            }
            let link = &mut self.links[idx];
            let reactivated = !link.active;
            link.active = true;
            link.coherence = (link.coherence + self.config.coherence_gain).min(1.0);
            link.entropy = (link.entropy - self.config.entropy_decay).max(0.0);
            if reactivated {
                *self.active_degree.entry(key.0).or_insert(0) += 1;
                *self.active_degree.entry(key.1).or_insert(0) += 1;
                self.active_count_cache += 1;
            }
            return false;
        }
        if self.node_link_count(a) >= self.config.max_links_per_node
            || self.node_link_count(b) >= self.config.max_links_per_node
        {
            return false;
        }
        let idx = self.links.len();
        self.links.push(EntanglementLink {
            a: key.0,
            b: key.1,
            coherence: 0.35,
            entropy: 0.10,
            heat: 0.0,
            last_sync_tick: self.tick,
            active: true,
        });
        self.lookup.insert(key, idx);
        self.adjacency.entry(key.0).or_default().push(idx);
        self.adjacency.entry(key.1).or_default().push(idx);
        *self.active_degree.entry(key.0).or_insert(0) += 1;
        *self.active_degree.entry(key.1).or_insert(0) += 1;
        self.active_count_cache += 1;
        true
    }

    pub fn synchronize_candidates(
        &mut self,
        seeds: &[usize],
        candidates: &mut Vec<usize>,
    ) -> EntanglementReport {
        self.synchronize_candidates_with_diagnostics(seeds, candidates, true)
    }

    pub fn synchronize_candidates_with_diagnostics(
        &mut self,
        seeds: &[usize],
        candidates: &mut Vec<usize>,
        diagnostics: bool,
    ) -> EntanglementReport {
        self.tick = self.tick.wrapping_add(1);
        let mut report = EntanglementReport::default();
        let mut budget = self.config.max_syncs_per_step;
        self.sync_frontier.clear();
        self.sync_frontier.extend_from_slice(seeds);
        self.sync_frontier.extend_from_slice(candidates);
        self.sync_present.clear();
        self.sync_present.extend(candidates.iter().copied());

        for position in 0..self.sync_frontier.len() {
            if budget == 0 {
                break;
            }
            let seed = self.sync_frontier[position];
            let Some(link_indices) = self.adjacency.get(&seed) else {
                continue;
            };
            for &idx in link_indices {
                if budget == 0 {
                    break;
                }
                if !self.links[idx].active {
                    continue;
                }
                let link = &mut self.links[idx];
                let remote = if link.a == seed { link.b } else { link.a };
                if self.sync_present.insert(remote) {
                    candidates.push(remote);
                }
                link.coherence = (link.coherence + self.config.coherence_gain).min(1.0);
                link.entropy = (link.entropy - self.config.entropy_decay).max(0.0);
                link.heat = (link.heat - self.config.entropy_decay).max(0.0);
                link.last_sync_tick = self.tick;
                report.synced += 1;
                let prune =
                    link.entropy >= self.config.max_entropy || link.heat >= self.config.max_heat;
                if prune {
                    report.pruned += 1;
                }
                let _ = link;
                if prune {
                    let (a, b) = (self.links[idx].a, self.links[idx].b);
                    self.links[idx].active = false;
                    if let Some(degree) = self.active_degree.get_mut(&a) {
                        *degree = degree.saturating_sub(1);
                    }
                    if let Some(degree) = self.active_degree.get_mut(&b) {
                        *degree = degree.saturating_sub(1);
                    }
                    self.active_count_cache = self.active_count_cache.saturating_sub(1);
                }
                budget -= 1;
            }
        }
        if diagnostics {
            self.fill_summary(&mut report);
        } else {
            report.active_links = self.active_count_cache;
        }
        report
    }

    pub fn inject_conflict(&mut self, a: usize, b: usize) -> EntanglementReport {
        let mut report = EntanglementReport::default();
        if let Some(&idx) = self.lookup.get(&ordered_pair(a, b)) {
            if self.links[idx].active {
                let link = &mut self.links[idx];
                link.entropy = (link.entropy + self.config.contradiction_gain * 2.0).min(2.0);
                link.heat = (link.heat + self.config.contradiction_gain * 2.0).min(2.0);
                report.conflicts = 1;
                let prune =
                    link.entropy >= self.config.max_entropy || link.heat >= self.config.max_heat;
                let _ = link;
                if prune {
                    self.deactivate_index(idx);
                    report.pruned = 1;
                }
            }
        }
        self.fill_summary(&mut report);
        report
    }

    pub fn summary(&self) -> EntanglementReport {
        let mut report = EntanglementReport::default();
        self.fill_summary(&mut report);
        report
    }

    pub fn region_entropy(&self, region: &[usize]) -> f32 {
        let region = region.iter().copied().collect::<HashSet<_>>();
        if region.is_empty() {
            return 0.0;
        }
        self.links
            .iter()
            .filter(|link| link.active && region.contains(&link.a) && region.contains(&link.b))
            .map(link_entropy)
            .sum()
    }

    pub fn boundary_area(&self, region: &[usize]) -> usize {
        let region = region.iter().copied().collect::<HashSet<_>>();
        if region.is_empty() {
            return 0;
        }
        self.links
            .iter()
            .filter(|link| link.active && (region.contains(&link.a) ^ region.contains(&link.b)))
            .count()
    }

    pub fn holographic_area_law_ratio(&self, region: &[usize], area_coupling: f32) -> f32 {
        let area = self.boundary_area(region).max(1) as f32;
        self.region_entropy(region) / (area_coupling.max(f32::EPSILON) * area)
    }

    pub fn active_links_touching(&self, region: &[usize]) -> usize {
        let region = region.iter().copied().collect::<HashSet<_>>();
        self.links
            .iter()
            .filter(|link| link.active && (region.contains(&link.a) || region.contains(&link.b)))
            .count()
    }

    pub fn serialize_persistent_state(&self) -> String {
        let mut out = String::new();
        out.push_str("CDT_RQM_EPR_ENTANGLEMENT_STATE_V1\n");
        out.push_str(&format!("tick {}\n", self.tick));
        out.push_str(&format!(
            "config {} {} {:.7} {:.7} {:.7} {:.7} {:.7} {:.7}\n",
            self.config.max_links_per_node,
            self.config.max_syncs_per_step,
            self.config.create_threshold,
            self.config.coherence_gain,
            self.config.entropy_decay,
            self.config.contradiction_gain,
            self.config.max_entropy,
            self.config.max_heat
        ));
        out.push_str(&format!("links {}\n", self.links.len()));
        for (idx, link) in self.links.iter().enumerate() {
            out.push_str(&format!(
                "l {} {} {} {:.7} {:.7} {:.7} {} {}\n",
                idx,
                link.a,
                link.b,
                link.coherence,
                link.entropy,
                link.heat,
                link.last_sync_tick,
                if link.active { 1 } else { 0 }
            ));
        }
        out.push_str("end\n");
        out
    }

    pub fn apply_persistent_state(&mut self, contents: &str) -> Result<(), String> {
        let mut lines = contents.lines();
        let version = lines.next();
        if version != Some("CDT_RQM_EPR_ENTANGLEMENT_STATE_V1") {
            return Err("version EPR invalida".to_string());
        }
        let tick_line = lines.next().ok_or("falta tick EPR")?;
        let parts = tick_line.split_whitespace().collect::<Vec<_>>();
        if parts.len() != 2 || parts[0] != "tick" {
            return Err(format!("tick EPR invalido: {tick_line}"));
        }
        self.tick = parse_u64(parts[1], "tick")?;

        let config_line = lines.next().ok_or("falta config EPR")?;
        let parts = config_line.split_whitespace().collect::<Vec<_>>();
        if parts.len() != 9 || parts[0] != "config" {
            return Err(format!("config EPR invalida: {config_line}"));
        }
        self.config = EntanglementConfig {
            max_links_per_node: parse_usize(parts[1], "max_links")?,
            max_syncs_per_step: parse_usize(parts[2], "max_syncs")?,
            create_threshold: parse_f32(parts[3], "create_threshold")?,
            coherence_gain: parse_f32(parts[4], "coherence_gain")?,
            entropy_decay: parse_f32(parts[5], "entropy_decay")?,
            contradiction_gain: parse_f32(parts[6], "contradiction_gain")?,
            max_entropy: parse_f32(parts[7], "max_entropy")?,
            max_heat: parse_f32(parts[8], "max_heat")?,
        };

        let links_header = lines.next().ok_or("faltan links EPR")?;
        let link_count = parse_count_header(links_header, "links")?;
        self.links.clear();
        self.lookup.clear();
        self.adjacency.clear();
        self.active_degree.clear();
        self.active_count_cache = 0;
        for _ in 0..link_count {
            let line = lines.next().ok_or("faltan lineas EPR")?;
            let parts = line.split_whitespace().collect::<Vec<_>>();
            if parts.len() != 9 || parts[0] != "l" {
                return Err(format!("link EPR invalido: {line}"));
            }
            let link = EntanglementLink {
                a: parse_usize(parts[2], "a")?,
                b: parse_usize(parts[3], "b")?,
                coherence: parse_f32(parts[4], "coherence")?,
                entropy: parse_f32(parts[5], "entropy")?,
                heat: parse_f32(parts[6], "heat")?,
                last_sync_tick: parse_u64(parts[7], "last_tick")?,
                active: parse_flag(parts[8], "active")?,
            };
            let idx = self.links.len();
            self.lookup.insert(ordered_pair(link.a, link.b), idx);
            self.adjacency.entry(link.a).or_default().push(idx);
            self.adjacency.entry(link.b).or_default().push(idx);
            if link.active {
                *self.active_degree.entry(link.a).or_insert(0) += 1;
                *self.active_degree.entry(link.b).or_insert(0) += 1;
                self.active_count_cache += 1;
            }
            self.links.push(link);
        }
        Ok(())
    }

    fn fill_summary(&self, report: &mut EntanglementReport) {
        let mut coherence_sum = 0.0;
        let mut entropy_sum = 0.0;
        let mut count = 0;
        for link in &self.links {
            if link.active {
                report.active_links += 1;
                coherence_sum += link.coherence;
                entropy_sum += link.entropy;
                count += 1;
            }
        }
        if count > 0 {
            report.mean_coherence = coherence_sum / count as f32;
            report.mean_entropy = entropy_sum / count as f32;
        }
    }

    fn node_link_count(&self, node: usize) -> usize {
        self.active_degree.get(&node).copied().unwrap_or(0)
    }

    fn deactivate_index(&mut self, idx: usize) -> bool {
        let Some(link) = self.links.get_mut(idx) else {
            return false;
        };
        if !link.active {
            return false;
        }
        link.active = false;
        let (a, b) = (link.a, link.b);
        if let Some(degree) = self.active_degree.get_mut(&a) {
            *degree = degree.saturating_sub(1);
        }
        if let Some(degree) = self.active_degree.get_mut(&b) {
            *degree = degree.saturating_sub(1);
        }
        self.active_count_cache = self.active_count_cache.saturating_sub(1);
        true
    }
}

fn ordered_pair(a: usize, b: usize) -> (usize, usize) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

fn link_entropy(link: &EntanglementLink) -> f32 {
    let p = (link.coherence * (1.0 - link.entropy)).clamp(0.0, 1.0);
    binary_entropy(p)
}

fn link_utility(link: EntanglementLink) -> f32 {
    link.coherence * (1.0 - link.entropy).clamp(0.0, 1.0) * (1.0 - link.heat).clamp(0.0, 1.0)
}

fn binary_entropy(p: f32) -> f32 {
    if !(0.0..=1.0).contains(&p) || p <= f32::EPSILON || 1.0 - p <= f32::EPSILON {
        return 0.0;
    }
    -p * p.ln() - (1.0 - p) * (1.0 - p).ln()
}

fn parse_count_header(line: &str, label: &str) -> Result<usize, String> {
    let parts = line.split_whitespace().collect::<Vec<_>>();
    if parts.len() != 2 || parts[0] != label {
        return Err(format!("cabecera {label} invalida: {line}"));
    }
    parse_usize(parts[1], label)
}

fn parse_usize(value: &str, label: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|err| format!("{label} invalido: {err}"))
}

fn parse_u64(value: &str, label: &str) -> Result<u64, String> {
    value
        .parse::<u64>()
        .map_err(|err| format!("{label} invalido: {err}"))
}

fn parse_f32(value: &str, label: &str) -> Result<f32, String> {
    value
        .parse::<f32>()
        .map_err(|err| format!("{label} invalido: {err}"))
}

fn parse_flag(value: &str, label: &str) -> Result<bool, String> {
    match value {
        "0" => Ok(false),
        "1" => Ok(true),
        _ => Err(format!("{label} invalido: {value}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn active_degree_cache_tracks_create_evict_and_reload() {
        let mut field = EntanglementField::new(EntanglementConfig {
            max_links_per_node: 3,
            ..EntanglementConfig::default()
        });
        assert!(field.create_or_reinforce(1, 2));
        assert!(field.create_or_reinforce(1, 3));
        assert_eq!(field.active_count(), 2);
        assert_eq!(field.active_links_for_node(1), 2);
        assert!(field.evict_lowest_utility_touching(1).is_some());
        assert_eq!(field.active_count(), 1);
        assert_eq!(field.active_links_for_node(1), 1);
        assert_eq!(field.compact_inactive(), 1);
        assert_eq!(field.link_slots(), 1);

        let state = field.serialize_persistent_state();
        let mut restored = EntanglementField::new(EntanglementConfig::default());
        restored.apply_persistent_state(&state).unwrap();
        assert_eq!(restored.active_count(), field.active_count());
        assert_eq!(restored.active_links_for_node(1), 1);
    }

    #[test]
    fn structural_prune_enforces_global_and_per_node_budgets() {
        let mut field = EntanglementField::new(EntanglementConfig {
            max_links_per_node: 16,
            ..EntanglementConfig::default()
        });
        for node in 1..=10 {
            assert!(field.create_or_reinforce(0, node));
        }
        assert_eq!(field.active_count(), 10);
        let pruned = field.prune_to_budget(4, 3);
        assert_eq!(pruned, 7);
        assert_eq!(field.active_count(), 3);
        assert!(field.active_links_for_node(0) <= 3);
        assert_eq!(field.link_slots(), field.active_count());
    }

    #[test]
    fn predictive_epr_accepts_useful_links_and_rejects_large_error() {
        let mut field = EntanglementField::new(EntanglementConfig {
            create_threshold: 0.5,
            ..EntanglementConfig::default()
        });
        let useful = field.observe_predictive_correlation(1, 2, 1.0, 0.05, 0.25);
        assert!(useful.accepted);
        assert!(useful.link_updated);
        assert!(field.has_active_link(1, 2));

        let conflict = field.observe_predictive_correlation(1, 2, 1.0, 0.9, 0.25);
        assert!(conflict.conflict);
        assert_eq!(conflict.effective_benefit, 0.0);
        assert!(conflict.field.conflicts > 0);
    }
}
