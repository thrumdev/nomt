//! This module implements the rollback log.
//!
//! The rollback log is implemented as a segmented log. Each segment is a separate file. The log
//! starts with segment 1, 0 is reserved for the initial empty state.
//!
//! The deltas are written into the currently active segment. When the active segment reaches a
//! certain size, it is closed and a new segment is opened. It's a soft limit and the segment files
//! might exceed the limit.
//!
//! The files for the log segments are named `rollback.0000000001.log`. The number is a 32-bit
//! integer ID of the segment. In the file name, the number is padded to be 10 digits long, to
//! fit in 4 bytes.
//!
//! The current active segment is stored in the manifest file. The invariant is that only the
//! specified ID is open for writing at any given time. All the previous segments are sealed and
//! immutable.
//!
//! Besides the active log segment ID, the manifest also stores the size of the active segment.
//! This is used to indicate the valid portion of the segment file. The file length cannot be relied
//! upon because upon recovery there is no guarantee that all the data written correctly, according
//! to our underlying storage assumptions.
//!
//! We write into the segment file in batches and every batch is aligned to 4 KiB. Actual contents
//! of the batch are enveloped by a header that contains the length of the batch, a 32-bit number.
//! This is done so that the batches are always written as full pages.
//!
//! Currently, a batch contains only a single delta.

use std::{
    collections::{HashMap, VecDeque},
    ffi::OsString,
    fs::File,
    io::{Cursor, Read as _, Write as _},
    path::PathBuf,
    str::FromStr,
    sync::Arc,
};

use nomt_core::trie::KeyPath;
use parking_lot::Mutex;
use threadpool::ThreadPool;

use crate::KeyReadWrite;

mod segment;

use segment::mk_rollback_segment;

pub const BATCH_ALIGNMENT: usize = 4096;

// This is just a placeholder, in the future we should probably make this configurable.
const MAX_ACTIVE_SEGMENT_SIZE: u32 = 1 << 26; // 64 MiB

struct ActiveSegment {
    /// The ID of this segment.
    id: u32,
    /// The file that this segment is written to.
    file: File,
    /// The current size of the segment file in bytes. Always a multiple of [`DELTA_ALIGNMENT`] bytes.
    size: u32,
}

// NOTE: the in-memory cache is currently constrainted to a single delta staged by commit for
//       sync. This is going to be fixed once we allow batches that contain multiple deltas.
struct InMemory {
    /// The most recent delta that we have staged.
    head: Option<Delta>,
    /// The log of deltas that we have accumulated so far modulo the [`Self::head`].
    ///
    /// The items are pushed onto the back and popped from the front. When the log reaches
    /// `limit_log_len`, the oldest delta is discarded.
    ///
    /// The deltas are stored in-memory even after they are dumped on disk. Upon restart, the deltas
    /// are re-read from disk and stored here.
    tail: VecDeque<Delta>,
    /// The number of items that we should keep in the log. Deltas that are past this limit are
    /// discarded.
    limit_log_len: usize,
}

struct Shared {
    /// The path that [`Shared::db_dir_fd`] points to.
    db_dir_path: PathBuf,
    /// The file descriptor for the database directory.
    db_dir_fd: File,
    worker_tp: ThreadPool,
    /// The number of bytes that an active segment can grow to before it is closed and a new one
    /// is opened.
    ///
    /// This is a soft limit.
    max_active_segment_size: u32,
    in_memory: Mutex<InMemory>,
    /// The current rollback log file for writing the next time we sync.
    active_segment: Mutex<Option<ActiveSegment>>,
}

impl ActiveSegment {
    fn brand_new(id: u32, file: File) -> Self {
        Self { id, file, size: 0 }
    }

    fn from_existing(id: u32, file: File, size: u32) -> Self {
        Self { id, file, size }
    }

    fn to_writeout_data(&self) -> WriteoutData {
        WriteoutData {
            active_segment_id: self.id,
            active_segment_size: self.size,
        }
    }
}

impl InMemory {
    fn new(limit_log_len: usize) -> Self {
        Self {
            limit_log_len,
            head: None,
            tail: VecDeque::new(),
        }
    }

    /// Push a delta into the in-memory cache.
    ///
    /// `stage` indicates whether we should set up the delta for staging (and thus the eventual
    /// dumping into the active segment file).
    fn push_back(&mut self, delta: Delta, stage: bool) {
        if stage {
            // This is a temporary invariant that should go away once we have a proper implementation.
            // See the notes for the `InMemory` type.
            assert!(self.head.is_none());
            self.head = Some(delta);
        } else {
            self.tail.push_back(delta);
        }

        // So if the total number of deltas
        if self.total_len() > self.limit_log_len {
            self.tail.pop_front();
        }
    }

    fn promote_staging(&mut self) -> Option<&Delta> {
        // Just carry over the staged delta to the tail. This doesn't change the total number of
        // items in the log and hence no pruning is needed.
        let staging = self.head.take();
        if let Some(staging) = staging {
            self.tail.push_back(staging);
        }
        self.tail.back()
    }

    fn pop_back(&mut self) -> Option<Delta> {
        self.head.take().or_else(|| self.tail.pop_back())
    }

    // Returns the total number of deltas, including the staged one.
    fn total_len(&self) -> usize {
        self.head.is_some() as usize + self.tail.len()
    }
}

/// This structure manages the rollback log. Modifications to the rollback log are made using
/// [`RollbackSnapshotBuilder`].
#[derive(Clone)]
pub struct Rollback {
    shared: Arc<Shared>,
}

impl Rollback {
    pub fn read(
        active_segment_id: u32,
        active_segment_size: u32,
        db_dir_path: PathBuf,
        db_dir_fd: File,
    ) -> anyhow::Result<Self> {
        assert!(
            active_segment_size as usize % BATCH_ALIGNMENT == 0,
            "precondition failed: active_segment_size must be a multiple of {}",
            BATCH_ALIGNMENT
        );

        // Recovery logic is something along the lines of:
        //
        // if the active_segment_id is 0, then we don't do anything.
        //
        // if the active_segment_id is greater than 0, then we attempt to read the segment files
        // to restore the in-memory cache. We read all the segment files up to the active one and
        // up to the `active_segment_size`.
        //
        // NB: if the actual file length of the active segment is greater than `active_segment_size`,
        //     then we need to truncate it to that length, so that appending writes will start at
        //     the correct offset.

        // Keep 100 deltas in memory.
        let limit_log_len = 100;
        let mut in_memory = InMemory::new(limit_log_len);
        let active_segment;

        if active_segment_id == 0 {
            active_segment = None;
        } else {
            // List all the segment files and read them into the in-memory log.
            // UNWRAP: unwrapping infaillable
            let active_segment_file_name =
                OsString::from_str(&segment::format_segment_file_name(active_segment_id)).unwrap();

            let mut segment_paths = Vec::new();
            let mut active_segment_path = None;

            for entry in std::fs::read_dir(&db_dir_path)? {
                let entry = entry?;
                let path = entry.path();
                if path
                    .file_name()
                    .map_or(false, |file_name| file_name == active_segment_file_name)
                {
                    active_segment_path = Some(path);
                } else if segment::is_valid_segment_file(&path) {
                    segment_paths.push(path);
                }
            }

            // At this stage, the directory should contain the active segment file.
            let active_segment_path = match active_segment_path {
                None => {
                    anyhow::bail!(
                        "manifest reports active segment id {} but file `{}` not found",
                        active_segment_id,
                        active_segment_file_name.to_string_lossy(),
                    );
                }
                Some(active_segment_path) => active_segment_path,
            };

            // Sorting the segment files by the segment ID works because the segment ID is a 32-bit
            // integer padded by leading zeros.
            segment_paths.sort();

            for segment_path in segment_paths {
                // Read every segment awhole besides the active one.
                let mut segment_file = std::fs::File::open(segment_path)?;
                read_segment(&mut segment_file, &mut in_memory)?;
            }

            // Now, read the active segment file, but only up to `active_segment_size`.
            let mut active_segment_file = std::fs::OpenOptions::new()
                .read(true)
                .append(true)
                .open(active_segment_path)?;

            // As part of the recovery process, we truncate the active segment file to the size
            // reported in the manifest.
            let file_len = active_segment_file.metadata()?.len();
            let file_len: u32 = file_len.try_into().map_err(|e| {
                anyhow::anyhow!("active segment file exists but is too large: {}", e)
            })?;
            if file_len > active_segment_size {
                active_segment_file.set_len(active_segment_size as u64)?;
            }
            read_segment(&mut active_segment_file, &mut in_memory)?;
            // `active_segment_file` is now positioned at the end of the file and is ready for
            // appending.
            active_segment = Some(ActiveSegment::from_existing(
                active_segment_id,
                active_segment_file,
                active_segment_size,
            ));
        }

        let shared = Arc::new(Shared {
            db_dir_path,
            db_dir_fd,
            worker_tp: ThreadPool::new(4),
            max_active_segment_size: MAX_ACTIVE_SEGMENT_SIZE,
            in_memory: Mutex::new(in_memory),
            active_segment: Mutex::new(active_segment),
        });
        Ok(Self { shared })
    }

    /// Begin a rollback delta.
    pub fn detla_builder(&self) -> ReverseDeltaBuilder {
        ReverseDeltaBuilder {
            tp: self.shared.worker_tp.clone(),
            priors: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Saves the delta into the log.
    ///
    /// This function accepts the final list of operations that should be performed sorted by the
    /// key paths in ascending order.
    pub fn commit(
        &self,
        store: impl LoadValue,
        actuals: &[(KeyPath, KeyReadWrite)],
        delta: ReverseDeltaBuilder,
    ) {
        let delta = delta.finalize(store, actuals);
        self.shared.in_memory.lock().push_back(delta, true);
    }

    /// Returns the changes that need to be performed to rollback the last `n` commits.
    pub fn traceback(&self, mut n: usize) -> Option<HashMap<KeyPath, Option<Vec<u8>>>> {
        assert!(n > 0);
        let mut in_memory = self.shared.in_memory.lock();
        if n > in_memory.total_len() {
            return None;
        }

        let mut traceback = HashMap::new();
        while n > 0 {
            // Pop the most recent delta from the log and add its original values to the traceback,
            // potentially overwriting some of the values that were added in previous iterations.
            //
            // UNWRAP: we checked above that `n` is greater or equal to the total number of deltas
            //         and `n` is strictly decreasing.
            let delta = in_memory.pop_back().unwrap();
            for (key, value) in delta.priors {
                traceback.insert(key, value);
            }
            n -= 1;
        }
        Some(traceback)
    }

    pub fn truncate(&self, n: usize) -> anyhow::Result<()> {
        assert!(n > 0);
        let mut in_memory = self.shared.in_memory.lock();
        while in_memory.total_len() > n {
            in_memory.pop_back();
        }
        Ok(())
    }

    /// Dumps the contents of the staging to the rollback.
    pub fn sync_rollback(&self) -> anyhow::Result<WriteoutData> {
        let mut in_memory = self.shared.in_memory.lock();
        // First figure out what to write out. We go over the log starting from the oldest delta
        // to the most recent one, serializing the deltas into a buffer.
        let delta = in_memory.promote_staging().unwrap();
        let to_write = prepare_batch(delta)?;
        if u32::try_from(to_write.len()).is_err() {
            anyhow::bail!("rollback batch is too large");
        }
        drop(in_memory);

        let mut active_segment_guard = self.shared.active_segment.lock();
        // Either get an existing file or create a new one.
        let active_segment = match &mut *active_segment_guard {
            Some(active_segment) => active_segment,
            None => {
                let id: u32 = 1;
                let file = mk_rollback_segment(&self.shared.db_dir_path, id)?;
                self.shared.db_dir_fd.sync_all()?;

                let active_segment = ActiveSegment::brand_new(id, file);
                *active_segment_guard = Some(active_segment);
                active_segment_guard.as_mut().unwrap()
            }
        };

        // Write the commit to the file.
        active_segment.file.write_all(&to_write)?;
        active_segment.size += to_write.len() as u32;
        active_segment.file.sync_data()?;
        // NOTE that syncing could be performed in parallel with switching the active segment.

        if active_segment.size >= self.shared.max_active_segment_size {
            // Close the current segment and open a new one.
            //
            // wrapping_add is a safe option here because the segments with the lower-numbered IDs
            // are supposed to be long gone.
            let new_segment_id = active_segment.id.wrapping_add(1);

            // We have to sync the parent directory to ensure the new file is created.
            //
            // NOTE that we don't really need to.
            let new_segment_file = mk_rollback_segment(&self.shared.db_dir_path, new_segment_id)?;
            self.shared.db_dir_fd.sync_all()?;
            let active_segment = ActiveSegment::brand_new(new_segment_id, new_segment_file);
            let wd = active_segment.to_writeout_data();
            *active_segment_guard = Some(active_segment);
            Ok(wd)
        } else {
            Ok(active_segment.to_writeout_data())
        }
    }
}

pub struct WriteoutData {
    /// The ID of the segment that was most recently being written to.
    pub active_segment_id: u32,
    /// The size of that segment in bytes.
    pub active_segment_size: u32,
}

/// A delta that should be applied to reverse a commit.
#[derive(Debug)]
struct Delta {
    /// This map contains the prior value for each key that was written by the commit this delta
    /// reverses. `None` indicates that the key did not exist before the commit.
    priors: HashMap<KeyPath, Option<Vec<u8>>>,
}

/// Prepares a batch for writing to a rollback segment file.
fn prepare_batch(delta: &Delta) -> anyhow::Result<Vec<u8>> {
    let mut to_write: Vec<u8> = Vec::with_capacity(MAX_ACTIVE_SEGMENT_SIZE as usize);

    let mut cursor = Cursor::new(&mut to_write);

    // Reserve space for the byte length of the commit.
    cursor.write_all(&0u32.to_le_bytes())?;

    let bytes_written = delta.encode(&mut cursor)?;

    // Pad the buffer to the next 4 KiB boundary and patch up the length.
    let padding = (BATCH_ALIGNMENT - to_write.len() % BATCH_ALIGNMENT) % BATCH_ALIGNMENT;
    to_write.extend(vec![0; padding]);

    // Fixup up the length.
    to_write[..4].copy_from_slice(&(bytes_written as u32).to_le_bytes());

    Ok(to_write)
}

/// Reads and decodes the whole segment file. A segment file contains a list of batches.
fn read_segment(mut fd: &File, in_memory: &mut InMemory) -> anyhow::Result<()> {
    let mut active_segment_contents = Vec::new();
    fd.read_to_end(&mut active_segment_contents)?;

    let mut cursor = Cursor::new(&mut active_segment_contents);
    while let Some(delta) = read_batch(&mut cursor)? {
        // The fact that we are reading the delta from the segment file means that the delta has
        // already been synced to a segment file and that we don't need to dump it, therefore
        // we push_back without staging.
        in_memory.push_back(delta, /* stage */ false);
    }

    Ok(())
}

fn read_batch<T>(reader: &mut Cursor<T>) -> anyhow::Result<Option<Delta>>
where
    T: AsMut<[u8]> + AsRef<[u8]>,
{
    let mut buf = [0; 4];
    match reader.read_exact(&mut buf) {
        Ok(()) => (),
        Err(e) => {
            if e.kind() == std::io::ErrorKind::UnexpectedEof {
                return Ok(None);
            }
            Err(e)?;
        }
    }
    let len = u32::from_le_bytes(buf) as usize;

    // Now, we need to constrain the reader to read `len` bytes.
    let start = reader.position() as usize;
    let end = start + len;
    let mut limited_reader = Cursor::new(reader.get_mut().as_mut()[start..end].as_mut());
    let delta = Delta::decode(&mut limited_reader)?;

    // Adjust the position of the original cursor. We read the batch and now we know that the rest
    // of the page is a padding. Position the cursor to the beginning of the next batch.
    let padding = (BATCH_ALIGNMENT - len) % BATCH_ALIGNMENT;
    reader.set_position((end + padding) as u64);

    Ok(Some(delta))
}

impl Delta {
    #[cfg(test)]
    fn empty() -> Self {
        Self {
            priors: HashMap::new(),
        }
    }

    /// Encode the delta into a buffer.
    ///
    /// Returns the number of bytes written.
    pub fn encode(&self, mut writer: impl std::io::Write) -> std::io::Result<usize> {
        // The serialization format has the following layout.
        //
        // The keys are split into two groups and written as separate arrays. Those groups are:
        //
        // 1. erase: The keys that did not exist before the commit.
        // 2. reinstateThe keys that had prior values.
        //
        // The keys that did not exist are written first. The keys that had prior values are
        // written second.
        //
        // For each kind of key, we first write out the length of the array encoded as a u32.
        // This is followed by the keys themselves, written contiguously in little-endian order.
        //
        // The keys are written as 32-byte big-endian values.

        let mut bytes_written = 0;

        // Sort the keys into two groups.
        let mut to_erase = Vec::with_capacity(self.priors.len());
        let mut to_reinstate = Vec::with_capacity(self.priors.len());
        for (key, value) in self.priors.iter() {
            match value {
                None => to_erase.push(key),
                Some(value) => to_reinstate.push((key, value)),
            }
        }

        let to_erase_len = to_erase.len() as u32;
        writer.write_all(&to_erase_len.to_le_bytes())?;
        bytes_written += 4;
        for key in to_erase {
            writer.write_all(&key[..])?;
            bytes_written += 32;
        }

        let to_reinstate_len = to_reinstate.len() as u32;
        writer.write_all(&to_reinstate_len.to_le_bytes())?;
        bytes_written += 4;
        for (key, value) in to_reinstate {
            writer.write_all(&key[..])?;
            bytes_written += 32;
            let value_len = value.len() as u32;
            writer.write_all(&value_len.to_le_bytes())?;
            bytes_written += 4;
            writer.write_all(value)?;
            bytes_written += value.len();
        }

        Ok(bytes_written)
    }

    /// Decodes the delta from a buffer.
    pub fn decode(reader: &mut Cursor<impl AsRef<[u8]>>) -> anyhow::Result<Self> {
        let mut priors = HashMap::new();

        // Read the number of keys to erase.
        let mut buf = [0; 4];
        reader.read_exact(&mut buf)?;
        let to_erase_len = u32::from_le_bytes(buf);
        // Read the keys to erase.
        for _ in 0..to_erase_len {
            let mut key_path = [0; 32];
            reader.read_exact(&mut key_path)?;
            let preemted = priors.insert(key_path, None).is_some();
            if preemted {
                anyhow::bail!("duplicate key path (erase): {:?}", key_path);
            }
        }

        // Read the number of keys to reinstate.
        reader.read_exact(&mut buf)?;
        let to_reinsate_len = u32::from_le_bytes(buf);
        // Read the keys to reinstate along with their values.
        for _ in 0..to_reinsate_len {
            // Read the key path.
            let mut key_path = [0; 32];
            reader.read_exact(&mut key_path)?;
            // Read the value.
            let mut value = Vec::new();
            reader.read_exact(&mut buf)?;
            let value_len = u32::from_le_bytes(buf);
            value.resize(value_len as usize, 0);
            reader.read_exact(&mut value)?;
            let preempted = priors.insert(key_path, Some(value)).is_some();
            if preempted {
                anyhow::bail!("duplicate key path (reinstate): {:?}", key_path);
            }
        }
        Ok(Delta { priors })
    }
}

pub struct ReverseDeltaBuilder {
    tp: ThreadPool,
    /// The values of the keys that should be preserved at commit time for this delta.
    ///
    /// Before the commit takes place, the set contains tentative values.
    priors: Arc<Mutex<HashMap<KeyPath, Option<Vec<u8>>>>>,
}

pub trait LoadValue: Clone + Send + Sync + 'static {
    fn load_value(&self, key_path: KeyPath) -> anyhow::Result<Option<Vec<u8>>>;
}

impl LoadValue for crate::store::Store {
    fn load_value(&self, key_path: KeyPath) -> anyhow::Result<Option<Vec<u8>>> {
        self.load_value(key_path)
    }
}

impl ReverseDeltaBuilder {
    /// Note that a write might be made to a key and that the rollback should preserve the prior
    /// value. This function is speculative; the rollback delta may later be committed with a
    /// different set of operations, and some of the tentative operations may be discarded.
    ///
    /// This function doesn't block.
    pub fn tentative_preserve_prior(&self, store: impl LoadValue, key_path: KeyPath) {
        self.tp.execute({
            let priors = self.priors.clone();
            move || {
                let value = store.load_value(key_path).unwrap();
                priors.lock().insert(key_path, value);
            }
        });
    }

    /// Finalize the delta.
    fn finalize(self, store: impl LoadValue, actuals: &[(KeyPath, KeyReadWrite)]) -> Delta {
        // Wait for all tentative writes issued so far to complete.
        //
        // NB: This doesn't take into account other users of `tp`. If there are any, we will be
        // needlessly blocking on them.
        self.tp.join();

        // Prune the paths from priors that did not end up being written.
        let mut priors = self.priors.lock();
        priors.retain(|path, _| {
            if let Some(index) = actuals.binary_search_by_key(&path, |(p, _)| p).ok() {
                let (_, read_write) = &actuals[index];
                read_write.is_write()
            } else {
                false
            }
        });

        // Collect the paths that weren't fetched.
        let mut unfetched = Vec::with_capacity(actuals.len());
        for (path, read_write) in actuals {
            if let KeyReadWrite::Write(_) | KeyReadWrite::ReadThenWrite(_, _) = read_write {
                if !priors.contains_key(path) {
                    unfetched.push(path);
                }
            }
        }
        drop(priors);

        // Fetch the remaining unfetched values.
        for path in unfetched {
            let store = store.clone();
            let path = path.clone();
            let priors = self.priors.clone();
            self.tp.execute(move || {
                let value = store.load_value(path).unwrap();
                priors.lock().insert(path, value);
            });
        }

        // Wait for all the fetches to complete. After this point, priors contains the final set of
        // values to be preserved.
        self.tp.join();

        Delta {
            priors: Arc::into_inner(self.priors).unwrap().into_inner(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hex_literal::hex;
    use std::rc::Rc;

    #[test]
    fn delta_roundtrip() {
        let mut delta = Delta {
            priors: HashMap::new(),
        };
        delta.priors.insert([1; 32], Some(b"value1".to_vec()));
        delta.priors.insert([2; 32], None);
        delta.priors.insert([3; 32], Some(b"value3".to_vec()));

        let mut buf = Vec::new();
        delta.encode(&mut buf).unwrap();
        let mut cursor = Cursor::new(&mut buf);
        let delta2 = Delta::decode(&mut cursor).unwrap();
        assert_eq!(delta.priors, delta2.priors);
    }

    #[test]
    fn delta_roundtrip_empty() {
        let delta = Delta {
            priors: HashMap::new(),
        };
        let mut buf = Vec::new();
        delta.encode(&mut buf).unwrap();
        let mut cursor = Cursor::new(&mut buf);
        let delta2 = Delta::decode(&mut cursor).unwrap();
        assert_eq!(delta.priors, delta2.priors);
    }

    #[test]
    fn batch_roundtrip() {
        let mut delta = Delta {
            priors: HashMap::new(),
        };
        delta.priors.insert([1; 32], Some(b"value1".to_vec()));
        delta.priors.insert([2; 32], None);
        delta.priors.insert([3; 32], Some(b"value3".to_vec()));

        let mut batch_bytes = prepare_batch(&delta).unwrap();
        let mut cursor = Cursor::new(&mut batch_bytes);
        let delta2 = read_batch(&mut cursor).unwrap().unwrap();
        assert_eq!(delta.priors, delta2.priors);
    }

    #[test]
    fn batch_roundtrip_empty() {
        let delta = Delta {
            priors: HashMap::new(),
        };
        let mut batch_bytes = Vec::new();
        batch_bytes.extend_from_slice(&prepare_batch(&delta).unwrap());

        let mut cursor = Cursor::new(&mut batch_bytes);
        let delta2 = read_batch(&mut cursor).unwrap().unwrap();
        assert_eq!(delta.priors, delta2.priors);
    }

    #[test]
    fn segment_roundtrip() {
        let mut delta = Delta {
            priors: HashMap::new(),
        };
        delta.priors.insert([1; 32], Some(b"value1".to_vec()));
        delta.priors.insert([2; 32], None);
        delta.priors.insert([3; 32], Some(b"value3".to_vec()));

        let mut file = tempfile::tempfile().unwrap();
        let batch_bytes = prepare_batch(&delta).unwrap();
        file.write_all(&batch_bytes).unwrap();
        file.sync_data().unwrap();

        std::io::Seek::seek(&mut file, std::io::SeekFrom::Start(0)).unwrap();
        let mut in_memory = InMemory::new(100);
        read_segment(&file, &mut in_memory).unwrap();
        let delta2 = in_memory.pop_back().unwrap();
        assert_eq!(delta.priors, delta2.priors);
    }

    /// A mock implementation of `LoadValue` for testing. Describes the "current" state of the
    /// database.
    #[derive(Clone)]
    struct MockStore {
        values: HashMap<KeyPath, Option<Vec<u8>>>,
    }

    impl MockStore {
        fn insert(&mut self, key_path: KeyPath, value: Option<Vec<u8>>) {
            self.values.insert(key_path, value);
        }
    }

    impl LoadValue for MockStore {
        fn load_value(&self, key_path: KeyPath) -> anyhow::Result<Option<Vec<u8>>> {
            match self.values.get(&key_path) {
                Some(value) => return Ok(value.clone()),
                None => panic!("the caller requested a value that was not inserted by the test"),
            }
        }
    }

    #[test]
    fn traceback_works() {
        let temp_dir = tempfile::tempdir().unwrap();
        let db_dir_path = temp_dir.path().join("db");
        std::fs::create_dir_all(&db_dir_path).unwrap();
        let db_dir_fd = std::fs::OpenOptions::new()
            .read(true)
            .open(db_dir_path.clone())
            .unwrap();

        let mut store = MockStore {
            values: HashMap::new(),
        };
        store.insert(
            hex!("0101010101010101010101010101010101010101010101010101010101010101"),
            Some(b"old_value1".to_vec()),
        );
        store.insert(
            hex!("0202020202020202020202020202020202020202020202020202020202020202"),
            Some(b"old_value2".to_vec()),
        );
        store.insert(
            hex!("0303030303030303030303030303030303030303030303030303030303030303"),
            Some(b"old_value3".to_vec()),
        );

        let rollback = Rollback::read(0, 0, db_dir_path, db_dir_fd).unwrap();
        let builder = rollback.detla_builder();
        builder.tentative_preserve_prior(store.clone(), [1; 32]);
        builder.tentative_preserve_prior(store.clone(), [2; 32]);
        builder.tentative_preserve_prior(store.clone(), [3; 32]);
        rollback.commit(
            store.clone(),
            &[
                (
                    hex!("0101010101010101010101010101010101010101010101010101010101010101"),
                    KeyReadWrite::Write(Some(Rc::new(b"new_value1".to_vec()))),
                ),
                (
                    hex!("0202020202020202020202020202020202020202020202020202020202020202"),
                    KeyReadWrite::Write(Some(Rc::new(b"new_value2".to_vec()))),
                ),
            ],
            builder,
        );

        // We want to see the old values for all the keys that have been changed during the commit.
        let traceback = rollback.traceback(1).unwrap();
        assert_eq!(traceback.len(), 2);
        assert_eq!(
            traceback
                .get(&hex!(
                    "0101010101010101010101010101010101010101010101010101010101010101"
                ))
                .unwrap()
                .clone(),
            Some(b"old_value1".to_vec())
        );
        assert_eq!(
            traceback
                .get(&hex!(
                    "0202020202020202020202020202020202020202020202020202020202020202"
                ))
                .unwrap()
                .clone(),
            Some(b"old_value2".to_vec())
        );
    }
}
