use crate::{backend::Transaction, timer::Timer, workload::Workload};
use fxhash::FxHashMap;
use nomt::{KeyPath, KeyReadWrite, Nomt, Options, Session};
use sha2::Digest;
use std::{collections::hash_map::Entry, path::PathBuf};

const NOMT_DB_FOLDER: &str = "nomt_db";

pub struct NomtDB {
    nomt: Nomt,
}

impl NomtDB {
    pub fn open(reset: bool, fetch_concurrency: usize) -> Self {
        if reset {
            // Delete previously existing db
            let _ = std::fs::remove_dir_all(NOMT_DB_FOLDER);
        }

        let opts = Options {
            path: PathBuf::from(NOMT_DB_FOLDER),
            fetch_concurrency,
            traversal_concurrency: 1,
        };

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

        workload.run(&mut transaction);

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
        let is_delete = value.is_none();

        match self.access.entry(key_path) {
            Entry::Occupied(mut o) => {
                o.get_mut().write(value);
            }
            Entry::Vacant(v) => {
                v.insert(KeyReadWrite::Write(value));
            }
        }

        self.session.tentative_write_slot(key_path, is_delete);
    }
}
