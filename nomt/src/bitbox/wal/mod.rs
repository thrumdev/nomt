use std::{
    fs::{File, OpenOptions},
    io::{Read, Write},
    os::fd::AsRawFd,
    path::PathBuf,
};

mod batch;
mod entry;
mod record;

pub use crate::wal::{batch::Batch, entry::Entry};

use self::record::Record;

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

const WAL_RECORD_SIZE: usize = 256 * 1024;
// walawala
const WAL_CHECKSUM_MAGIC: u32 = 0x00a10a1a;

// Open a WAL file and append encoded batch of writes to the end,
// and prune previous ones if the current succeeded.
//
// It does not perform any check on the state of the WAL file
pub struct WalWriter {
    wal_file: File,
}

impl WalWriter {
    /// Open a WAL file if it exists, otherwise, create a new empty one.
    ///
    /// No check with be performed if the file exists,
    /// it will only happend new batch to the end of the file
    /// and purge from the front
    pub fn open(path: PathBuf) -> anyhow::Result<Self> {
        let wal_file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .append(true)
            .open(&path)?;

        Ok(Self { wal_file })
    }

    /// Get the current length of the file.
    pub fn file_size(&self) -> u64 {
        self.wal_file.metadata().unwrap().len()
    }

    /// Clean up the first n bytes of the file.
    pub fn prune_front(&self, n: u64) {
        // fallocate call with flag FALLOC_FL_COLLAPSE_RANGE
        // returns an error if you're trying to collapse 0 bytes
        if n == 0 {
            return;
        }

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

/// Open a WAL file and check the integrity of the file, it decodes
/// all records and make sure they are coherent with the WAL format.
pub struct WalChecker {
    last_batch: Option<Batch>,
}

pub enum ConsistencyError {
    LastBatchCrashed(Batch),
    NotConsistent(u64),
}

impl WalChecker {
    /// Open a WAL file and check its integrity, if a problem is found then the
    /// wal is restored to its last valid state, thus it updates the WAL file
    /// by removing everything after the last valid record.
    ///
    /// NOTE: Even though currently in the simulation each batch is pruned right after
    /// the success of the next one, the following method needs to be able to handle multiple
    /// records of different batches and validate them
    pub fn open_and_recover(path: PathBuf) -> Self {
        let mut wal_file = match OpenOptions::new().read(true).write(true).open(&path) {
            Ok(wal_file) => wal_file,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                // a non-existing WAL is considered valid, with the assumed sequence number being zero
                return Self { last_batch: None };
            }
            Err(_) => {
                panic!("Error opening or creating the wal file ")
            }
        };

        let (last_records, last_records_position) = validate_records(&mut wal_file);

        // If not all the WAL was parsed correctly, then the recovery procedure means to remove
        // all incorrect records from the last valid one

        let wal_file_len = wal_file
            .metadata()
            .expect("Error extracting wal file len")
            .len();

        if last_records_position as u64 != wal_file_len {
            // cannot use fallocate + FALLOC_FL_COLLAPSE_RANGE
            // because it requires offset and len to be multiple of the
            // filesystem block size, and len could be different from a multiple of a record size
            unsafe {
                let res =
                    libc::ftruncate(wal_file.as_raw_fd(), last_records_position as libc::off_t);
                assert!(res == 0, "WAL Recovery: ftruncate failed");
            }

            wal_file.sync_all().unwrap();
        }

        Self {
            last_batch: Some(Batch::from_records(last_records)),
        }
    }

    pub fn check_consistency(self, store_sequence_number: u64) -> Result<(), ConsistencyError> {
        let wal_sequence_number = self
            .last_batch
            .as_ref()
            .map(|batch| batch.sequence_number())
            .unwrap_or(0);

        if wal_sequence_number == store_sequence_number {
            Ok(())
        } else if wal_sequence_number == store_sequence_number + 1 {
            Err(ConsistencyError::LastBatchCrashed(self.last_batch.unwrap()))
        } else {
            Err(ConsistencyError::NotConsistent(wal_sequence_number))
        }
    }
}

fn validate_records(wal_file: &mut File) -> (Vec<Record>, usize) {
    // contains the last set of valid records
    let mut last_records = vec![];
    // offset in the WAL file to the byte right after the end of the last parsed valid record
    let mut last_records_position = 0;

    let mut curr_records = vec![];
    let mut sequence_number = None;

    // Rules to be respected:
    //  1. all records must stored in chunks of WAL_RECORD_SIZE bytes
    //  2. all records must be well formatted
    //  3. each record sequence number must be equal to the previous one or equal to previous +1
    //  4. only the last record in a batch must have the 'last' flag to 0xFF, others 0x00

    let record_iter = std::iter::from_fn(|| {
        let mut buf = [0; WAL_RECORD_SIZE];

        let read_bytes = wal_file
            .read(&mut buf)
            .expect("Error reading from wal file");

        match read_bytes {
            // ends the iterator if the read bytes are not exactly the expected
            // size of a record entry or if it was not properly encoded
            WAL_RECORD_SIZE => Some(Record::from_bytes(buf).ok()?),
            _ => None,
        }
    });

    for (record_index, record) in record_iter.enumerate() {
        let is_last = record.last();
        let seq_num = record.sequence_number();
        curr_records.push(record);

        // first record
        let Some(expected_seqn) = sequence_number else {
            sequence_number = Some(seq_num);
            continue;
        };

        if expected_seqn != seq_num {
            break;
        }

        if is_last {
            // sequence_number is expected to be increased from the next record
            sequence_number = Some(seq_num + 1);
            last_records = std::mem::replace(&mut curr_records, vec![]);
            last_records_position = WAL_RECORD_SIZE * (record_index + 1);
            continue;
        }
    }

    (last_records, last_records_position)
}
