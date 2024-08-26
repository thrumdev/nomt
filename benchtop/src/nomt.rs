use crate::{backend::Transaction, timer::Timer, workload::Workload};
use fxhash::FxHashMap;
use nomt::{KeyPath, KeyReadWrite, Nomt, Options, Session};
use sha2::Digest;
use std::collections::hash_map::Entry;

const NOMT_DB_FOLDER: &str = "nomt_db";

pub struct NomtDB {
    nomt: Nomt,
}

impl NomtDB {
    pub fn open(
        reset: bool,
        commit_concurrency: usize,
        io_workers: usize,
        hashtable_buckets: Option<u32>,
    ) -> Self {
        if reset {
            // Delete previously existing db
            let _ = std::fs::remove_dir_all(NOMT_DB_FOLDER);
        }

        let mut opts = Options::new();
        opts.path(NOMT_DB_FOLDER);
        opts.commit_concurrency(commit_concurrency);
        opts.io_workers(io_workers);
        opts.metrics(true);
        if let Some(buckets) = hashtable_buckets {
            opts.hashtable_buckets(buckets);
        }

        let nomt = Nomt::open(opts).unwrap();
        Self { nomt }
    }

    pub fn execute(&self, mut timer: Option<&mut Timer>, workload: &mut dyn Workload) {
        let _timer_guard_total = timer.as_mut().map(|t| t.record_span("workload"));

        let mut transaction = Tx {
            session: self.nomt.begin_session(),
            access: FxHashMap::default(),
            timer,
        };

        workload.run_step(&mut transaction);

        let Tx {
            session,
            access,
            mut timer,
        } = transaction;

        let _timer_guard_commit = timer.as_mut().map(|t| t.record_span("commit_and_prove"));
        let mut actual_access: Vec<_> = access.into_iter().collect();
        actual_access.sort_by_key(|(k, _)| *k);
        self.nomt.commit_and_prove(session, actual_access).unwrap();
    }

    pub fn print_metrics(&self) {
        self.nomt.metrics().print()
    }
}

struct Tx<'a> {
    timer: Option<&'a mut Timer>,
    session: Session,
    access: FxHashMap<KeyPath, KeyReadWrite>,
}

impl<'a> Transaction for Tx<'a> {
    fn read(&mut self, key: &[u8]) -> Option<Vec<u8>> {
        let key_path = sha2::Sha256::digest(key).into();
        let _timer_guard_read = self.timer.as_mut().map(|t| t.record_span("read"));

        match self.access.entry(key_path) {
            Entry::Occupied(o) => o.get().last_value().map(|v| v.to_vec()),
            Entry::Vacant(v) => {
                let value = self.session.tentative_read_slot(key_path).unwrap();
                v.insert(KeyReadWrite::Read(value.clone()));
                value.map(|v| v.to_vec())
            }
        }
    }

    fn write(&mut self, key: &[u8], value: Option<&[u8]>) {
        let key_path = sha2::Sha256::digest(key).into();
        let value = value.map(|v| std::rc::Rc::new(v.to_vec()));

        match self.access.entry(key_path) {
            Entry::Occupied(mut o) => {
                o.get_mut().write(value);
            }
            Entry::Vacant(v) => {
                v.insert(KeyReadWrite::Write(value));
            }
        }

        self.session.tentative_write_slot(key_path);
    }
}
