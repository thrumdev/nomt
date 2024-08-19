use std::path::PathBuf;

/// Options when opening a [`Nomt`] instance.
pub struct Options {
    /// The path to the directory where the trie is stored.
    pub(crate) path: PathBuf,
    /// The number of commit workers. Values over 64 will be rounded down to 64.
    pub(crate) commit_concurrency: usize,
    /// The number of io_uring instances, or I/O threads on non-Linux platforms.
    pub(crate) io_workers: usize,
    /// Enable or disable metrics collection.
    pub(crate) metrics: bool,
    pub(crate) bitbox_num_pages: u32,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            path: PathBuf::from("nomt_db"),
            commit_concurrency: 1,
            io_workers: 3,
            metrics: false,
            bitbox_num_pages: 64_000,
        }
    }
}

impl Options {
    /// Create a new `Options` instance with the default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the path to the directory where the trie is stored.
    pub fn path(&mut self, path: impl Into<PathBuf>) {
        self.path = path.into();
    }

    /// Set the maximum number of concurrent commit workers.
    ///
    /// Values over 64 will be rounded down to 64.
    ///
    /// May not be zero.
    pub fn commit_concurrency(&mut self, commit_concurrency: usize) {
        self.commit_concurrency = commit_concurrency;
    }

    /// Set metrics collection on or off.
    ///
    /// Default: off.
    pub fn metrics(&mut self, metrics: bool) {
        self.metrics = metrics;
    }

    /// Set the number of io_uring instances.
    ///
    /// Must be more than 0
    pub fn io_workers(&mut self, io_workers: usize) {
        assert!(io_workers > 0);
        self.io_workers = io_workers;
    }
}
