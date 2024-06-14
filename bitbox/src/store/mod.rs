use crate::node_pages_map::{Page, PAGE_SIZE};
use crossbeam_channel::{bounded, unbounded, Receiver, Sender};
use io_uring::{cqueue, opcode, squeue, types, IoUring};
use std::{fs::File, os::fd::RawFd};
use std::{ops::Range, os::fd::AsRawFd, path::PathBuf, sync::Arc};

mod completion;
#[cfg(test)]
mod tests;

/// The Store is an on disk array of [`crate::node_pages_map::Page`]
// TODO: make the store thread-safe so that it can be accessed from multiple threads
pub struct Store {
    store_file: File,
    capacity: u64,
    ring: Arc<IoUring>,
    ring_capacity: u32,
    tx_new_batches: Sender<(Range<u64>, Option<Vec<Box<Page>>>, Sender<()>)>,
    tx_kill_completion_handler: Sender<()>,
    nonce: u64,
}

/// Options to open a Store
pub struct StoreOptions {
    /// File path of the storage file.
    file_path: PathBuf,
    /// Max number of pages supported
    pages_capacity: u64,
    /// Maximum number of entries supported in the io_uring submission queue
    io_uring_capacity: u32,
    /// Reset the store to an empty state
    reset: bool,
}

impl Default for StoreOptions {
    fn default() -> Self {
        Self {
            file_path: PathBuf::from("node_pages_store"),
            pages_capacity: 1_000,
            io_uring_capacity: 512,
            reset: true,
        }
    }
}

impl Store {
    /// Create a new Store given the StoreOptions
    pub fn new(options: StoreOptions) -> Self {
        if options.reset {
            // TODO: Is there a better alternative? fallocate, trucate or head maybe
            let _ = std::process::Command::new("dd")
                .args([
                    "if=/dev/zero", // fill with zeros
                    format!("of={}", options.file_path.to_string_lossy()).as_str(),
                    format!("bs={}", options.pages_capacity).as_str(),
                    format!("count={}", PAGE_SIZE).as_str(),
                ])
                .output()
                .expect("Error creating empty store file");
        }

        let store_file = std::fs::File::options()
            .write(true)
            .read(true)
            .open(options.file_path)
            .expect("Imp open file");

        let ring = Arc::new(
            IoUring::<squeue::Entry, cqueue::Entry>::builder()
                // TODO: Test those options
                //.setup_sqpoll()
                //.setup_iopoll()
                .build(options.io_uring_capacity)
                .expect("Error building io_uring"),
        );

        let (tx_new_batches, rx_new_batches) =
            unbounded::<(Range<u64>, Option<Vec<Box<Page>>>, Sender<()>)>();
        let (tx_kill_completion_handler, rx_kill_completion_handler) = bounded::<()>(1);

        completion::start_completion_handler(
            ring.clone(),
            options.io_uring_capacity,
            rx_new_batches,
            rx_kill_completion_handler,
        );

        Store {
            store_file,
            capacity: options.pages_capacity,
            ring_capacity: options.io_uring_capacity,
            ring,
            tx_new_batches,
            tx_kill_completion_handler,
            nonce: 0,
        }
    }

    /// Write the pages at the specified index,
    /// returns a WriteHandle with the number of written pages
    /// or None if there was no space in the Submission queue to perform any write.
    ///
    /// Panics if the specified index is greater than the capacity of the storage.
    pub fn write(&mut self, pages: &[(Page, u64)]) -> Option<(WriteHandle, u64)> {
        let init_range = self.nonce;

        let mut submissions = unsafe { self.ring.submission_shared() };

        let max_entries = self.ring_capacity as usize - submissions.len();

        // TODO: Pages are being boxed to prevent deallocation before the write is completed;
        // consider replacing this with a more efficient alternative.
        let (writes_data, writes_entry): (Vec<Box<Page>>, Vec<squeue::Entry>) = pages
            .iter()
            .take(max_entries)
            .map(|(page, page_index)| {
                if *page_index >= self.capacity {
                    panic!("Specified index bigger then the Store capacity");
                }

                let boxed_page = Box::new(*page);

                // TODO: use registered buffer
                let write_entry = opcode::Write::new(
                    types::Fd(self.store_file.as_raw_fd()),
                    // because we need to make sure the area is NOT dropped until
                    // the write requests is finished
                    boxed_page.as_ptr(),
                    boxed_page.len() as _,
                )
                .offset(PAGE_SIZE as u64 * page_index)
                .build()
                .user_data(self.nonce);

                self.nonce += 1;

                (boxed_page, write_entry)
            })
            .unzip();

        if self.nonce - init_range == 0 {
            return None;
        }

        // UNWRAP: The number of entries in `writes_entry` cannot be greater than max_entries,
        // so the submission should never fail
        unsafe {
            submissions
                .push_multiple(&writes_entry)
                .expect("Error submitting write entries to the submission queue");
        }

        // send all the info to the completion_handler
        let (tx, rx) = bounded::<()>(1);
        self.tx_new_batches
            .send((init_range..self.nonce, Some(writes_data), tx))
            .expect("Error sending new batch to completion_handler");

        submissions.sync();
        // TODO: Under what circumstances could this fail?
        self.ring.submit().unwrap();

        Some((WriteHandle { rx }, self.nonce - init_range))
    }

    // TODO: current NOMT's Store read one page at the time, with io_uring we could issue multiple reads
    // at the same time reducing the overhead of iterating over read
    pub fn read(&mut self, page_index: u64) -> Option<ReadHandle> {
        if page_index >= self.capacity {
            panic!("Specified index bigger then the Store capacity");
        }

        let mut submissions = unsafe { self.ring.submission_shared() };
        submissions.sync();

        if submissions.is_full() {
            return None;
        }

        let mut page_buf = Box::new([0; PAGE_SIZE]);
        let read_entry = opcode::Read::new(
            types::Fd(self.store_file.as_raw_fd()),
            page_buf.as_mut_ptr(),
            page_buf.len() as _,
        )
        .offset(PAGE_SIZE as u64 * page_index)
        .build()
        .user_data(self.nonce);

        // UNWRAP: We just checked that the submission queue is not,
        // so the submission should never fail
        unsafe {
            submissions
                .push(&read_entry)
                .expect("Error submitting read entry to the submission queue");
        }

        let (tx, rx) = bounded::<()>(1);
        self.tx_new_batches
            .send((self.nonce..self.nonce + 1, None, tx))
            .expect("Error sending new read to completion_handler");

        submissions.sync();
        self.ring.submit().expect("Imp submit submission entries");

        self.nonce += 1;

        Some(ReadHandle { data: page_buf, rx })
    }
}

impl Drop for Store {
    fn drop(&mut self) {
        self.tx_kill_completion_handler
            .send(())
            .expect("Error attempting to kill completion_handler");
    }
}

pub struct WriteHandle {
    rx: Receiver<()>,
}

impl WriteHandle {
    pub fn wait(&self) {
        self.rx.recv().unwrap()
    }
}

pub struct ReadHandle {
    data: Box<Page>,
    rx: Receiver<()>,
}

impl ReadHandle {
    pub fn wait(self) -> Page {
        self.rx.recv().unwrap();
        *self.data
    }
}
