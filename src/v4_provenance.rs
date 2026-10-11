//! v4 provenance guard (protocolo §2 `v4_provenance`, plan §20).
//!
//! Every memory-like store (CDT experience store, RQM, tables, NN index,
//! attractor bank, direct memory) must route reads through [`count_query`].
//! Counters are **thread-local** so parallel seeds never mix; a TEST
//! evaluation snapshots the counters before/after and the delta is written
//! into each prediction's provenance record. Any non-zero delta under a
//! FIELD_ONLY scientific condition marks the row `LEAKED`.

use crate::v4_dataset::sha256_hex;
use std::cell::RefCell;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct QueryCounters {
    pub cdt: u64,
    pub rqm: u64,
    pub table: u64,
    pub nn: u64,
    pub attractor: u64,
    pub direct_memory: u64,
    pub decoder_lookup: u64,
}

impl QueryCounters {
    pub fn total(&self) -> u64 {
        self.cdt
            + self.rqm
            + self.table
            + self.nn
            + self.attractor
            + self.direct_memory
            + self.decoder_lookup
    }

    pub fn delta(&self, before: &QueryCounters) -> QueryCounters {
        QueryCounters {
            cdt: self.cdt - before.cdt,
            rqm: self.rqm - before.rqm,
            table: self.table - before.table,
            nn: self.nn - before.nn,
            attractor: self.attractor - before.attractor,
            direct_memory: self.direct_memory - before.direct_memory,
            decoder_lookup: self.decoder_lookup - before.decoder_lookup,
        }
    }

    pub fn add(&mut self, o: &QueryCounters) {
        self.cdt += o.cdt;
        self.rqm += o.rqm;
        self.table += o.table;
        self.nn += o.nn;
        self.attractor += o.attractor;
        self.direct_memory += o.direct_memory;
        self.decoder_lookup += o.decoder_lookup;
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Store {
    Cdt,
    Rqm,
    Table,
    Nn,
    Attractor,
    DirectMemory,
    DecoderLookup,
}

thread_local! {
    static COUNTERS: RefCell<QueryCounters> = RefCell::new(QueryCounters::default());
}

/// Record one read from a memory-like store (called by the store itself).
pub fn count_query(s: Store, n: u64) {
    COUNTERS.with(|c| {
        let mut c = c.borrow_mut();
        match s {
            Store::Cdt => c.cdt += n,
            Store::Rqm => c.rqm += n,
            Store::Table => c.table += n,
            Store::Nn => c.nn += n,
            Store::Attractor => c.attractor += n,
            Store::DirectMemory => c.direct_memory += n,
            Store::DecoderLookup => c.decoder_lookup += n,
        }
    });
}

pub fn snapshot() -> QueryCounters {
    COUNTERS.with(|c| *c.borrow())
}

/// Per-prediction information path (plan §20).
#[derive(Clone, Debug, Default)]
pub struct Provenance {
    pub experiment: String,
    pub seed: u64,
    pub condition: String,
    pub instance_id: u64,
    pub target_seen_training: bool,
    pub target_seen_dev: bool,
    pub target_seen_cdt: bool,
    pub target_seen_rqm: bool,
    pub target_seen_table: bool,
    pub target_seen_nn: bool,
    pub target_seen_attractor: bool,
    pub target_equivalent_seen: bool,
    pub queries: QueryCounters,
    pub retrieval_on: bool,
}

impl Provenance {
    pub fn leaked(&self) -> bool {
        self.target_seen_training
            || self.target_seen_dev
            || self.target_seen_cdt
            || self.target_seen_rqm
            || self.target_seen_table
            || self.target_seen_nn
            || self.target_seen_attractor
            || self.target_equivalent_seen
            || self.queries.total() > 0
    }

    pub fn status(&self) -> &'static str {
        if self.leaked() {
            "LEAKED"
        } else {
            "CLEAN"
        }
    }

    fn body(&self) -> String {
        let q = &self.queries;
        format!(
            "\"experiment\":\"{}\",\"seed\":\"0x{:X}\",\"condition\":\"{}\",\"instance_id\":{},\"target_seen_training\":{},\"target_seen_dev\":{},\"target_seen_cdt\":{},\"target_seen_rqm\":{},\"target_seen_table\":{},\"target_seen_nn\":{},\"target_seen_attractor\":{},\"target_equivalent_seen\":{},\"cdt_queries\":{},\"rqm_queries\":{},\"table_queries\":{},\"nn_queries\":{},\"attractor_queries\":{},\"direct_memory_queries\":{},\"decoder_lookup\":{},\"retrieval_on\":{},\"status\":\"{}\"",
            self.experiment,
            self.seed,
            self.condition,
            self.instance_id,
            self.target_seen_training,
            self.target_seen_dev,
            self.target_seen_cdt,
            self.target_seen_rqm,
            self.target_seen_table,
            self.target_seen_nn,
            self.target_seen_attractor,
            self.target_equivalent_seen,
            q.cdt,
            q.rqm,
            q.table,
            q.nn,
            q.attractor,
            q.direct_memory,
            q.decoder_lookup,
            self.retrieval_on,
            self.status()
        )
    }

    pub fn provenance_hash(&self) -> String {
        sha256_hex(self.body().as_bytes())
    }

    pub fn to_jsonl(&self) -> String {
        format!(
            "{{{},\"provenance_hash\":\"{}\"}}",
            self.body(),
            self.provenance_hash()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v4_counters_are_thread_local_and_mark_leak() {
        let before = snapshot();
        count_query(Store::Cdt, 2);
        let d = snapshot().delta(&before);
        assert_eq!(d.cdt, 2);
        let t = std::thread::spawn(snapshot).join().unwrap();
        assert_eq!(t.total(), 0);
        let p = Provenance {
            queries: d,
            ..Default::default()
        };
        assert_eq!(p.status(), "LEAKED");
        assert_eq!(Provenance::default().status(), "CLEAN");
    }
}
