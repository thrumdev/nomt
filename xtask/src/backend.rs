use crate::{nomt::NomtDB, sov_db::SovDB, timer::Timer};

type Value = Vec<u8>;
type Key = Vec<u8>;

#[derive(Debug, Clone, clap::ValueEnum)]
pub enum Backend {
    SovDB,
    Nomt,
}

#[derive(Clone, Debug)]
#[allow(unused)]
pub enum Action {
    // Write into the storage, None as value means delete that key
    Write { key: Key, value: Option<Value> },
    // Read the storage
    Read { key: Key },
}

/// Trait implemented by all backends who wants to be benchmarked.
pub trait Db {
    /// Apply the given actions to the storage, committing them
    /// to the database at the end.
    ///
    /// The function also accepts an optional timer that will be used to measure
    /// the relevant parts of the backend that effectively apply the actions.
    fn apply_actions(&mut self, actions: Vec<Action>, timer: Option<&mut Timer>);
}

impl Backend {
    pub fn all_backends() -> Vec<Self> {
        vec![Backend::SovDB, Backend::Nomt]
    }

    pub fn instantiate(&self) -> Box<dyn Db> {
        match self {
            Backend::SovDB => Box::new(SovDB::new()),
            Backend::Nomt => Box::new(NomtDB::new()),
        }
    }
}
