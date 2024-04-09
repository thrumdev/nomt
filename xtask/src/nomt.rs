use crate::{Action, DB};
use fxhash::FxHashMap;
use nomt::{KeyPath, KeyReadWrite, Node, Nomt, Options, Session};
use sha2::Digest;
use std::{collections::hash_map::Entry, path::PathBuf, rc::Rc};

pub struct NomtDB {
    nomt: Nomt,
}

impl NomtDB {
    pub fn new() -> Self {
        let _ = std::fs::remove_dir_all("tmp_nomt_db");

        let opts = Options {
            path: PathBuf::from("tmp_nomt_db"),
            fetch_concurrency: 1,
            traversal_concurrency: 1,
        };

        let nomt = Nomt::open(opts).unwrap();
        Self { nomt }
    }
}

impl DB for NomtDB {
    fn apply_actions(&mut self, actions: Vec<Action>) {
        let mut session = self.nomt.begin_session();
        let mut access: FxHashMap<KeyPath, KeyReadWrite> = FxHashMap::default();

        for action in actions.into_iter() {
            match action {
                Action::Writes(writes) => {
                    for (key, val) in writes {
                        // TODO: Should the hash of the key be included in the benchmarking?
                        let key_path = sha2::Sha256::digest(key).into();
                        let value = val.map(|v| std::rc::Rc::new(v));
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
                }
                _ => todo!(),
            }
        }

        let mut actual_access: Vec<_> = access.into_iter().collect();
        actual_access.sort_by_key(|(k, _)| *k);
        self.nomt.commit_and_prove(session, actual_access).unwrap();
    }
}
