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
    pub fn instantiate(
        &self,
        reset: bool,
        commit_concurrency: usize,
        io_workers: usize,
        hashtable_buckets: Option<u32>,
    ) -> DB {
        match self {
            Backend::SovDB => DB::Sov(SovDB::open(reset)),
            Backend::Nomt => DB::Nomt(NomtDB::open(
                reset,
                commit_concurrency,
                io_workers,
                hashtable_buckets,
            )),
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
    /// Execute a workload repeatedly until done.
    pub fn execute(&mut self, mut timer: Option<&mut Timer>, workload: &mut dyn Workload) {
        while !workload.is_done() {
            let timer = timer.as_mut().map(|x| &mut **x);
            match self {
                DB::Sov(db) => db.execute(timer, workload),
                DB::SpTrie(db) => db.execute(timer, workload),
                DB::Nomt(db) => db.execute(timer, workload),
            }
        }
    }

    /// Execute a workload repeatedly until done or a time limit is reached.
    pub fn execute_with_timeout(
        &mut self,
        mut timer: Option<&mut Timer>,
        workload: &mut dyn Workload,
        timeout: std::time::Instant,
    ) {
        while !workload.is_done() {
            if std::time::Instant::now() > timeout {
                break;
            }
            let timer = timer.as_deref_mut();
            match self {
                DB::Sov(db) => db.execute(timer, workload),
                DB::SpTrie(db) => db.execute(timer, workload),
                DB::Nomt(db) => db.execute(timer, workload),
            }
        }
    }

    /// Print metrics collected by the Backend if it supports metrics collection
    pub fn print_metrics(&self) {
        match self {
            DB::Nomt(db) => db.print_metrics(),
            _ => (),
        }
    }
}
