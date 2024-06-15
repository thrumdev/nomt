use std::{
    fs::{File, OpenOptions},
    os::unix::fs::OpenOptionsExt,
    path::PathBuf,
};

pub mod io;
#[cfg(test)]
mod tests;

/// The Store is an on disk array of [`crate::node_pages_map::Page`]
pub struct Store {
    store_file: File,
}

/// Options to open a Store
pub struct StoreOptions {
    /// File path of the storage file.
    file_path: PathBuf,
}

impl Default for StoreOptions {
    fn default() -> Self {
        Self {
            file_path: PathBuf::from("node_pages_store"),
        }
    }
}

impl Store {
    /// Create a new Store given the StoreOptions
    pub fn new(options: StoreOptions) -> Self {
        let store_file = OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(libc::O_DIRECT)
            .open(options.file_path)
            .expect("could not open file");

        Store { store_file }
    }
}
