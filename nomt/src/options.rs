use std::path::PathBuf;

/// Options when opening a [`Nomt`] instance.
pub struct Options {
    /// The path to the directory where the trie is stored.
    pub(crate) path: PathBuf,
    /// The maximum number of concurrent page fetches. Values over 64 will be rounded down to 64.
    /// May not be zero.
    pub(crate) fetch_concurrency: usize,
    /// Enable or disable metrics collection.
    pub(crate) metrics: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            path: PathBuf::from("nomt_db"),
            fetch_concurrency: 1,
            metrics: false,
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

    /// Set the maximum number of concurrent page fetches.
    ///
    /// Values over 64 will be rounded down to 64.
    ///
    /// May not be zero.
    pub fn fetch_concurrency(&mut self, fetch_concurrency: usize) {
        self.fetch_concurrency = fetch_concurrency;
    }

    /// Set metrics collection on or off.
    ///
    /// Default: off.
    pub fn metrics(&mut self, metrics: bool) {
        self.metrics = metrics;
    }
}
