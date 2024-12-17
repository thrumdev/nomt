use std::path::PathBuf;

/// Options when opening a [`crate::Nomt`] instance.
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
    pub(crate) bitbox_seed: [u8; 16],
    pub(crate) panic_on_sync: bool,
    pub(crate) rollback: bool,
    /// The maximum number of commits that can be rolled back.
    pub(crate) max_rollback_log_len: u32,
    pub(crate) warm_up: bool,
    /// Whether to preallocate the hashtable file.
    pub(crate) preallocate_ht: bool,
}

impl Options {
    /// Create a new `Options` instance with the default values and a random bitbox seed.
    pub fn new() -> Self {
        use rand::Rng as _;
        let mut bitbox_seed = [0u8; 16];
        rand::rngs::OsRng.fill(&mut bitbox_seed);

        Self {
            path: PathBuf::from("nomt_db"),
            commit_concurrency: 1,
            io_workers: 3,
            metrics: false,
            bitbox_num_pages: 64_000,
            bitbox_seed,
            panic_on_sync: false,
            rollback: false,
            max_rollback_log_len: 100,
            warm_up: false,
            preallocate_ht: true,
        }
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

    /// Set the number of hashtable buckets to use when creating the database.
    pub fn hashtable_buckets(&mut self, hashtable_buckets: u32) {
        self.bitbox_num_pages = hashtable_buckets;
    }

    /// Set the seed for the hash function used by the bitbox store.
    ///
    /// Useful for reproducibility.
    pub fn bitbox_seed(&mut self, bitbox_seed: [u8; 16]) {
        self.bitbox_seed = bitbox_seed;
    }

    /// Set to `true` to panic on sync after writing the WAL file and updating the manifest, but
    /// before the data has been written to the HT file.
    ///
    /// Useful to test WAL recovery.
    pub fn panic_on_sync(&mut self, panic_on_sync: bool) {
        self.panic_on_sync = panic_on_sync;
    }

    /// Set to `true` to enable rolling back committed sessions.
    pub fn rollback(&mut self, rollback: bool) {
        self.rollback = rollback;
    }

    /// Set the maximum number of commits that can be rolled back.
    ///
    /// Only relevant if rollback is enabled.
    ///
    /// Default: 100.
    pub fn max_rollback_log_len(&mut self, max_rollback_log_len: u32) {
        self.max_rollback_log_len = max_rollback_log_len;
    }

    /// Configure whether merkle page fetches should be warmed up while sessions are ongoing.
    ///
    /// Enabling this feature can pessimize performance.
    pub fn warm_up(&mut self, warm_up: bool) {
        self.warm_up = warm_up;
    }

    /// Sets whether to preallocate the hashtable file.
    ///
    /// Many filesystems don't handle sparse files well. If the `preallocate_ht` option is set to
    /// `true`, NOMT will try to make sure that the file is fully allocated.
    ///
    /// If set to `false` this won't allocate the disk space for the hashtable file upfront, but can
    /// lead to fragmentation later.
    ///
    /// Default: `true`.
    pub fn preallocate_ht(&mut self, preallocate_ht: bool) {
        self.preallocate_ht = preallocate_ht;
    }
}
