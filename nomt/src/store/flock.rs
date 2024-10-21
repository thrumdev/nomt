//! This module provides a cross-platform advisory lock on a directory.

use std::{
    fs::{File, OpenOptions},
    path::Path,
};

use fs2::FileExt as _;

/// Represents a cross-platform advisory lock on a directory.
pub struct Flock {
    lock_fd: File,
}

impl Flock {
    pub fn lock(db_dir: &Path, lock_filename: &str) -> anyhow::Result<Self> {
        let lock_path = db_dir.join(lock_filename);

        let lock_fd = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(lock_path)?;

        match lock_fd.try_lock_exclusive() {
            Ok(_) => Ok(Self { lock_fd }),
            Err(_) => {
                let err = fs2::lock_contended_error();
                anyhow::bail!("Failed to lock directory: {err}");
            }
        }
    }
}

impl Drop for Flock {
    fn drop(&mut self) {
        if let Err(e) = self.lock_fd.unlock() {
            eprintln!("Failed to unlock directory lock: {e}");
        }
    }
}
