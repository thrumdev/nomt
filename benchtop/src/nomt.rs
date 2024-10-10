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
        let nomt_db_folder =
            std::env::var("NOMT_DB_FOLDER").unwrap_or_else(|_| NOMT_DB_FOLDER.to_string());

        if reset {
            // Delete previously existing db
            let _ = std::fs::remove_dir_all(&nomt_db_folder);
        }

        let mut opts = Options::new();
        opts.path(nomt_db_folder);
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

        let session = self.nomt.begin_session();
        let mut transaction = Tx {
            session: &session,
            access: FxHashMap::default(),
            timer,
        };

        workload.run_step(&mut transaction);

        let Tx {
            access, mut timer, ..
        } = transaction;

        let _timer_guard_commit = timer.as_mut().map(|t| t.record_span("commit_and_prove"));
        let mut actual_access: Vec<_> = access.into_iter().collect();
        actual_access.sort_by_key(|(k, _)| *k);
        self.nomt.commit_and_prove(session, actual_access).unwrap();
    }

    // note: this is only intended to be used with workloads which are disjoint, i.e. no workload
    // writes a key which another workload reads. re-implementing BlockSTM or other OCC methods are
    // beyond the scope of benchtop.
    pub fn parallel_execute(
        &self,
        mut timer: Option<&mut Timer>,
        thread_pool: &rayon::ThreadPool,
        workloads: &mut [Box<dyn Workload>],
    ) {
        let _timer_guard_total = timer.as_mut().map(|t| t.record_span("workload"));

        let session = self.nomt.begin_session();
        let mut results: Vec<Option<_>> = (0..workloads.len()).map(|_| None).collect();

        let use_timer = timer.is_some();
        thread_pool.in_place_scope(|scope| {
            for (workload, result) in workloads.into_iter().zip(results.iter_mut()) {
                let session = &session;
                scope.spawn(move |_| {
                    let mut workload_timer = if use_timer {
                        Some(Timer::new(String::new()))
                    } else {
                        None
                    };
                    let mut transaction = Tx {
                        session,
                        access: FxHashMap::default(),
                        timer: workload_timer.as_mut(),
                    };
                    workload.run_step(&mut transaction);
                    *result = Some((transaction.access, workload_timer.map(|t| t.freeze())));
                })
            }
        });

        // absorb instrumented times from workload timers.
        for (_, ref mut workload_timer) in results.iter_mut().flatten() {
            if let (Some(ref mut t), Some(wt)) = (timer.as_mut(), workload_timer.take()) {
                t.add(wt);
            }
        }

        let _timer_guard_commit = timer.as_mut().map(|t| t.record_span("commit_and_prove"));
        let mut actual_access: Vec<_> = results
            .into_iter()
            .flatten()
            .map(|(access, _)| access)
            .flatten()
            .collect();
        actual_access.sort_by_key(|(k, _)| *k);
        self.nomt.commit_and_prove(session, actual_access).unwrap();
    }

    pub fn print_metrics(&self) {
        self.nomt.metrics().print()
    }
}

struct Tx<'a> {
    timer: Option<&'a mut Timer>,
    session: &'a Session,
    access: FxHashMap<KeyPath, KeyReadWrite>,
}

impl<'a> Transaction for Tx<'a> {
    fn read(&mut self, key: &[u8]) -> Option<Vec<u8>> {
        let key_path = sha2::Sha256::digest(key).into();
        let _timer_guard_read = self.timer.as_mut().map(|t| t.record_span("read"));

        match self.access.entry(key_path) {
            Entry::Occupied(o) => o.get().last_value().map(|v| v.to_vec()),
            Entry::Vacant(v) => {
                let value = self.session.read(key_path).unwrap();
                self.session.warm_up(key_path);

                v.insert(KeyReadWrite::Read(value.clone()));
                value.map(|v| v.to_vec())
            }
        }
    }

    fn write(&mut self, key: &[u8], value: Option<&[u8]>) {
        let key_path = sha2::Sha256::digest(key).into();
        let value = value.map(|v| v.to_vec());

        match self.access.entry(key_path) {
            Entry::Occupied(mut o) => {
                o.get_mut().write(value);
            }
            Entry::Vacant(v) => {
                v.insert(KeyReadWrite::Write(value));
            }
        }

        self.session.warm_up(key_path);
    }
}
