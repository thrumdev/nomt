use anyhow::bail;

use std::{
    fs::{File, OpenOptions},
    io::Write,
    os::fd::AsRawFd,
    path::PathBuf,
};

mod batch;
mod entry;
mod record;

pub use crate::wal::{batch::Batch, entry::Entry};

// WAL format:
//
// series of 256KB records
// each record contains a sequence number, a checksum, a number of entries, and the entries
// themselves. entries never span records.
//
// Record format:
//   sequence number (8 bytes)
//   checksum (4 bytes) (magic number ++ last ++ data)
//   entry count (2 bytes)
//   LAST (1 byte) | 0xFF if the last record for the sequence, 0x00 otherwise.
//   data: [Entry]
//
// Entry format:
// kind: u8
//   1 => updated bucket
//   2 => cleared bucket
// data: (variable based on kind)
//  bucket update:
//    page ID (16 bytes)
//    diff (16 bytes)
//    changed [[u8; 32]] (len with diff)
//    bucket index (8 bytes)
//  bucket cleared:
//    bucket index (8 bytes)
//
// Multiple records are expected to have the same sequence number if they are part of the same
// transaction.
// e.g. [0 0 0 1 1 2 2 2 2 3 3 4 4 4 4] might be the order of sequence numbers in a WAL.
// this can be used to compare against the sequence number in the database's metadata file to
// determine which WAL entries need to be reapplied.
//
// WAL entries imply writing buckets and the meta-map. when recovering from a crash, do both.
//
//
// ASSUMPTIONS, TO BE DISCUSSED:
// + the file is expected to be contiguous in memory, so the last record starts at position
//   file.len() - WAL_RECORD_SIZE
// + All records must have a sequence number equal to the previous one or the previous one increased by 1,
//   the first record in the file defines the start.
//   (if we decide to use a ring wall, then the start could be somewhere else)
//   No gaps are accepted between records; otherwise, integrity is not maintained
// + We exect that the entries in the Records are locically correct

const WAL_RECORD_SIZE: usize = 256 * 1024;
// walawala
const WAL_CHECKSUM_MAGIC: u32 = 0x00a10a1a;

pub struct Wal {
    wal_file: File,
}

impl Wal {
    /// Open a WAL file if it exists, otherwise, create a new empty one.
    ///
    /// If it already exists, all records will be parsed to verify the
    /// WAL's integrity. If not satisfied, an error will be returned.
    ///
    /// NOTE: This does not perform any checks on the database's consistency (yet),
    /// it needs to be done after creating the WAL and once we are sure it is correct.
    pub fn open(path: PathBuf) -> anyhow::Result<Self> {
        let wal_file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .append(true)
            .open(&path)?;

        // if the wal file is empty no integrity check needs to be done
        let wal_file_len = wal_file.metadata()?.len();

        // remove when we handle corruption.
        if wal_file_len % WAL_RECORD_SIZE as u64 != 0 {
            bail!("WAL corrupted - delete and restart");
        }

        Ok(Self { wal_file })
    }

    /// Get the current length of the file.
    pub fn file_size(&self) -> u64 {
        self.wal_file.metadata().unwrap().len()
    }

    /// Clean up the first n bytes of the file.
    pub fn prune_front(&self, n: u64) {
        unsafe {
            let res = libc::fallocate(
                self.wal_file.as_raw_fd(),
                libc::FALLOC_FL_COLLAPSE_RANGE,
                0,
                n as libc::off_t,
            );

            assert!(res == 0, "WAL Collapse Range Failed");
        }

        self.wal_file.sync_all().unwrap();
    }

    // apply a batch of changes to the WAL file. returns only after FSYNC has completed.
    pub fn apply_batch(&mut self, batch: &Batch) -> anyhow::Result<()> {
        let records: Vec<_> = batch.to_records();

        for record in records {
            let raw_record = record.to_bytes();

            self.wal_file.write_all(&raw_record)?;
        }

        self.wal_file.flush()?;
        self.wal_file.sync_all()?;

        Ok(())
    }
}
