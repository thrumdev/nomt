use crate::{
    backend::{Action, Db},
    timer::Timer,
};
use fxhash::FxHashMap;
use nomt::{KeyPath, KeyReadWrite, Nomt, Options};
use sha2::Digest;
use std::{collections::hash_map::Entry, path::PathBuf};

const NOMT_DB_FOLDER: &str = "nomt_db";
const NOMT_DB_FOLDER_COPY: &str = "nomt_db_copy";

pub struct NomtDB {
    nomt: Nomt,
}

impl NomtDB {
    pub fn open(reset: bool) -> Self {
        if reset {
            // Delete previously existing db
            let _ = std::fs::remove_dir_all(NOMT_DB_FOLDER);
        }

        let opts = Options {
            path: PathBuf::from(NOMT_DB_FOLDER),
            fetch_concurrency: 1,
            traversal_concurrency: 1,
        };

        let nomt = Nomt::open(opts).unwrap();
        Self { nomt }
    }
}

impl Db for NomtDB {
    fn open_copy(&self) -> Box<dyn Db> {
        // Delete any previously existing copy of the db
        let _ = std::fs::remove_dir_all(NOMT_DB_FOLDER_COPY);

        std::process::Command::new("cp")
            .args(["-r", NOMT_DB_FOLDER, NOMT_DB_FOLDER_COPY])
            .output()
            .expect("Impossible make a copy of the nomt db");

        let opts = Options {
            path: PathBuf::from(NOMT_DB_FOLDER_COPY),
            fetch_concurrency: 1,
            traversal_concurrency: 1,
        };

        let nomt = Nomt::open(opts).unwrap();
        Box::new(Self { nomt })
    }

    fn apply_actions(&mut self, actions: Vec<Action>, mut timer: Option<&mut Timer>) {
        let _timer_guard_total = timer.as_mut().map(|t| t.record_span("workload"));

        let mut session = self.nomt.begin_session();
        let mut access: FxHashMap<KeyPath, KeyReadWrite> = FxHashMap::default();

        for action in actions.into_iter() {
            match action {
                Action::Write { key, value } => {
                    let key_path = sha2::Sha256::digest(key).into();
                    let value = value.map(std::rc::Rc::new);
                    let is_delete = value.is_none();

                    match access.entry(key_path) {
                        Entry::Occupied(mut o) => {
                            o.get_mut().write(value);
                        }
                        Entry::Vacant(v) => {
                            v.insert(KeyReadWrite::Write(value));
                        }
                    }

                    session.tentative_write_slot(key_path, is_delete);
                }
                Action::Read { key } => {
                    let key_path = sha2::Sha256::digest(key).into();

                    let _timer_guard_read = timer.as_mut().map(|t| t.record_span("read"));
                    let _value = match access.entry(key_path) {
                        Entry::Occupied(o) => o.get().last_value().cloned(),
                        Entry::Vacant(v) => {
                            let value = session.tentative_read_slot(key_path).unwrap();
                            v.insert(KeyReadWrite::Read(value.clone()));
                            value
                        }
                    };
                }
            }
        }

        let _timer_guard_commit = timer.as_mut().map(|t| t.record_span("commit_and_prove"));
        let mut actual_access: Vec<_> = access.into_iter().collect();
        actual_access.sort_by_key(|(k, _)| *k);
        self.nomt.commit_and_prove(session, actual_access).unwrap();
    }
}
