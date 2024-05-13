use crate::{nomt::NomtDB, sov_db::SovDB, sp_trie::SpTrieDB, timer::Timer, workload::Workload};

#[derive(Debug, Clone, clap::ValueEnum)]
pub enum Backend {
    SovDB,
    Nomt,
    SpTrie,
}

impl Backend {
    pub fn all_backends() -> Vec<Self> {
        vec![Backend::SovDB, Backend::SpTrie, Backend::Nomt]
    }

    // If reset is true, then erase any previous backend's database
    // and restart from an empty database.
    // Otherwise, use the already present database.
    pub fn instantiate(&self, reset: bool) -> DB {
        match self {
            Backend::SovDB => DB::Sov(SovDB::open(reset)),
            Backend::Nomt => DB::Nomt(NomtDB::open(reset)),
            Backend::SpTrie => DB::SpTrie(SpTrieDB::open(reset)),
        }
    }
}

/// A transaction over the database which allows reading and writing.
pub trait Transaction {
    /// Read a value from the database. If a value was previously written, return that.
    fn read(&mut self, key: &[u8]) -> Option<Vec<u8>>;

    /// Write a value to the database. `None` means to delete the previous value.
    fn write(&mut self, key: &[u8], value: Option<&[u8]>);
}

/// A wrapper around all databases implemented in this tool.
pub enum DB {
    Sov(SovDB),
    SpTrie(SpTrieDB),
    Nomt(NomtDB),
}

impl DB {
    /// Execute some code against the DB using the given closure.
    pub fn execute(&mut self, timer: Option<&mut Timer>, workload: &mut dyn Workload) {
        match self {
            DB::Sov(db) => db.execute(timer, workload),
            DB::SpTrie(db) => db.execute(timer, workload),
            DB::Nomt(db) => db.execute(timer, workload),
        }
    }
}
