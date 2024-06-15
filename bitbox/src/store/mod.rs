use crate::node_pages_map::{Page, PAGE_SIZE};
use io_uring::{cqueue, opcode, squeue, types, IoUring};
use slab::Slab;
use std::{
    fs::{File, OpenOptions},
    io,
    ops::Range,
    os::{fd::AsRawFd, unix::fs::OpenOptionsExt},
    path::PathBuf,
    sync::Arc,
};

#[cfg(test)]
mod tests;

const RING_CAPACITY: u32 = 128;

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

/// An I/O handle for a store. This owns an io_uring ring which is used to submit reads and writes.
pub struct StoreIo {
    store: Arc<Store>,
    ring: IoUring,
    pending: Slab<Pending>,
}

impl StoreIo {
    pub fn new(store: Arc<Store>) -> Self {
        StoreIo {
            ring: IoUring::<squeue::Entry, cqueue::Entry>::builder()
                .setup_sqpoll(100)
                // .setup_iopoll
                .build(RING_CAPACITY)
                .expect("Error building io_uring"),
            store,
            pending: Slab::with_capacity(RING_CAPACITY as _),
        }
    }

    pub fn is_submit_full(&mut self) -> bool {
        self.ring.submission().is_full()
    }

    // read a page. panics if the submission queue is full.
    pub fn read(&mut self, page_id: u64, mut buf: Box<Page>) -> io::Result<()> {
        let entry = opcode::Read::new(
            types::Fd(self.store.store_file.as_raw_fd()),
            &mut buf[0] as *mut u8,
            PAGE_SIZE as u32,
        )
        .offset(page_id * (PAGE_SIZE as u64 + 1))
        .build()
        .user_data(self.pending.vacant_key() as u64);
        unsafe { self.ring.submission().push(&entry).unwrap() }
        self.pending.insert(Pending {
            kind: Kind::Read,
            page_id,
            buf,
        });

        self.ring.submit()?;
        Ok(())
    }

    // write a page. panics if the submission queue is full.
    pub fn write<'a>(&mut self, page_id: u64, buf: Box<Page>) -> io::Result<()> {
        let entry = opcode::Write::new(
            types::Fd(self.store.store_file.as_raw_fd()),
            &buf[0] as *const u8,
            PAGE_SIZE as u32,
        )
        .offset(page_id * (PAGE_SIZE as u64 + 1))
        .build()
        .user_data(self.pending.vacant_key() as u64);
        unsafe { self.ring.submission().push(&entry).unwrap() }
        self.pending.insert(Pending {
            kind: Kind::Read,
            page_id,
            buf,
        });
        self.ring.submit()?;
        Ok(())
    }

    // Process I/O completions.
    pub fn complete(&mut self) -> impl Iterator<Item = std::io::Result<CompleteIo>> + '_ {
        self.ring.completion().map(|completion_event| {
            let op = self.pending.remove(completion_event.user_data() as usize);
            if completion_event.result() != 0 {
                Err(std::io::Error::from_raw_os_error(completion_event.result()))
            } else {
                Ok(match op.kind {
                    Kind::Read => CompleteIo::Read(op.page_id, op.buf),
                    Kind::Write => CompleteIo::Write(op.page_id, op.buf),
                })
            }
        })
    }
}

/// A complete I/O operation.
pub enum CompleteIo {
    Read(u64, Box<Page>),
    Write(u64, Box<Page>),
}

enum Kind {
    Read,
    Write,
}

struct Pending {
    kind: Kind,
    page_id: u64,
    buf: Box<Page>,
}
