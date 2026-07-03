use std::collections::HashMap;

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

#[derive(Clone, Debug)]
pub struct EntanglementField {
    pub config: EntanglementConfig,
    links: Vec<EntanglementLink>,
    lookup: HashMap<(usize, usize), usize>,
    adjacency: HashMap<usize, Vec<usize>>,
    pending_benefit: HashMap<(usize, usize), f32>,
    tick: u64,
}

impl EntanglementField {
    pub fn new(config: EntanglementConfig) -> Self {
        Self {
            config,
            links: Vec::new(),
            lookup: HashMap::new(),
            adjacency: HashMap::new(),
            pending_benefit: HashMap::new(),
            tick: 0,
        }
    }

    pub fn active_count(&self) -> usize {
        self.links.iter().filter(|link| link.active).count()
    }

    pub fn observe_correlation(&mut self, a: usize, b: usize, benefit: f32) -> bool {
        if a == b {
            return false;
        }
        let key = ordered_pair(a, b);
        let score = self.pending_benefit.entry(key).or_insert(0.0);
        *score += benefit.max(0.0);
        if *score < self.config.create_threshold {
            return false;
        }
        *score = 0.0;
        self.create_or_reinforce(a, b)
    }

    pub fn create_or_reinforce(&mut self, a: usize, b: usize) -> bool {
        let key = ordered_pair(a, b);
        if let Some(&idx) = self.lookup.get(&key) {
            let link = &mut self.links[idx];
            link.active = true;
            link.coherence = (link.coherence + self.config.coherence_gain).min(1.0);
            link.entropy = (link.entropy - self.config.entropy_decay).max(0.0);
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
        true
    }

    pub fn synchronize_candidates(
        &mut self,
        seeds: &[usize],
        candidates: &mut Vec<usize>,
    ) -> EntanglementReport {
        self.tick = self.tick.wrapping_add(1);
        let mut report = EntanglementReport::default();
        let mut budget = self.config.max_syncs_per_step;
        let existing = candidates.clone();

        for &seed in seeds.iter().chain(existing.iter()) {
            if budget == 0 {
                break;
            }
            let Some(link_indices) = self.adjacency.get(&seed).cloned() else {
                continue;
            };
            for idx in link_indices {
                if budget == 0 {
                    break;
                }
                let link = &mut self.links[idx];
                if !link.active {
                    continue;
                }
                let remote = if link.a == seed { link.b } else { link.a };
                let compatible = candidates.contains(&seed) || seeds.contains(&seed);
                if compatible {
                    if !candidates.contains(&remote) {
                        candidates.push(remote);
                    }
                    link.coherence = (link.coherence + self.config.coherence_gain).min(1.0);
                    link.entropy = (link.entropy - self.config.entropy_decay).max(0.0);
                    link.heat = (link.heat - self.config.entropy_decay).max(0.0);
                    link.last_sync_tick = self.tick;
                    report.synced += 1;
                } else {
                    link.entropy = (link.entropy + self.config.contradiction_gain).min(2.0);
                    link.heat = (link.heat + self.config.contradiction_gain).min(2.0);
                    report.conflicts += 1;
                }
                if link.entropy >= self.config.max_entropy || link.heat >= self.config.max_heat {
                    link.active = false;
                    report.pruned += 1;
                }
                budget -= 1;
            }
        }
        self.fill_summary(&mut report);
        report
    }

    pub fn inject_conflict(&mut self, a: usize, b: usize) -> EntanglementReport {
        let mut report = EntanglementReport::default();
        if let Some(&idx) = self.lookup.get(&ordered_pair(a, b)) {
            let link = &mut self.links[idx];
            if link.active {
                link.entropy = (link.entropy + self.config.contradiction_gain * 2.0).min(2.0);
                link.heat = (link.heat + self.config.contradiction_gain * 2.0).min(2.0);
                report.conflicts = 1;
                if link.entropy >= self.config.max_entropy || link.heat >= self.config.max_heat {
                    link.active = false;
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

    pub fn serialize_persistent_state(&self) -> String {
        let mut out = String::new();
        out.push_str("SNGA_EPR_ENTANGLEMENT_STATE_V1\n");
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
        if lines.next() != Some("SNGA_EPR_ENTANGLEMENT_STATE_V1") {
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
        self.adjacency
            .get(&node)
            .map(|links| links.iter().filter(|&&idx| self.links[idx].active).count())
            .unwrap_or(0)
    }
}

fn ordered_pair(a: usize, b: usize) -> (usize, usize) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
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
