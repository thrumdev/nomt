use crate::{
    backend::{Action, Db},
    timer::Timer,
};
use fxhash::FxHashMap;
use nomt::{KeyPath, KeyReadWrite, Nomt, Options};
use sha2::Digest;
use std::{collections::hash_map::Entry, path::PathBuf};

pub struct NomtDB {
    nomt: Nomt,
}

impl NomtDB {
    pub fn new() -> Self {
        let _ = std::fs::remove_dir_all("nomt_db");

        let opts = Options {
            path: PathBuf::from("nomt_db"),
            fetch_concurrency: 1,
            traversal_concurrency: 1,
        };

        let nomt = Nomt::open(opts).unwrap();
        Self { nomt }
    }
}

impl Db for NomtDB {
    fn apply_actions(&mut self, actions: Vec<Action>, timer: Option<&mut Timer>) {
        let _timer_guard = timer.and_then(|t| Some(t.record()));

        let mut session = self.nomt.begin_session();
        let mut access: FxHashMap<KeyPath, KeyReadWrite> = FxHashMap::default();

        for action in actions.into_iter() {
            match action {
                Action::Write { key, value } => {
                    let key_path = sha2::Sha256::digest(key).into();
                    let value = value.map(|v| std::rc::Rc::new(v));
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

        let mut actual_access: Vec<_> = access.into_iter().collect();
        actual_access.sort_by_key(|(k, _)| *k);
        self.nomt.commit_and_prove(session, actual_access).unwrap();
    }
}
