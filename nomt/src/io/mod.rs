#[cfg(not(target_family = "unix"))]
std::compile_error!("NOMT only supports Unix-based OSs");

use crossbeam_channel::{Receiver, RecvError, SendError, Sender, TryRecvError, TrySendError};
use std::{fs::File, os::fd::RawFd};

#[cfg(target_os = "linux")]
#[path = "linux.rs"]
mod platform;

#[cfg(not(target_os = "linux"))]
#[path = "unix.rs"]
mod platform;

pub mod page_pool;

pub const PAGE_SIZE: usize = 4096;

pub use page_pool::{FatPage, PagePool};

pub enum IoKind {
    Read(RawFd, u64, FatPage),
    Write(RawFd, u64, FatPage),
    ReadRaw(RawFd, u64, *mut u8, usize),
    WriteRaw(RawFd, u64, *const u8, usize),
}

pub enum IoKindResult {
    Ok,
    Err,
    Retry,
}

impl IoKind {
    pub fn get_result(&self, res: isize) -> IoKindResult {
        match self {
            // pread returns 0 if the file has been read till the end of file
            //
            // This could be a failure because the end of the file could be smaller than PAGE_SIZE.
            // However, as each read operation follows a write operation,
            // there should be no unexpected end-of-file that is not aligned with PAGE_SIZE
            // when all previous writes have succeeded.
            IoKind::Read(_, _, _) | IoKind::ReadRaw(_, _, _, _) if res == 0 => IoKindResult::Ok,
            // pread and pwrite return the number of bytes read or written
            _ if res == PAGE_SIZE as isize => IoKindResult::Ok,
            _ if res == -1 => IoKindResult::Err,
            _ => IoKindResult::Retry,
        }
    }
}

unsafe impl Send for IoKind {}

pub struct IoCommand {
    pub kind: IoKind,
    // note: this isn't passed to io_uring, it's higher-level userdata.
    pub user_data: u64,
}

pub struct CompleteIo {
    pub command: IoCommand,
    pub result: std::io::Result<()>,
}

struct IoPacket {
    command: IoCommand,
    completion_sender: Sender<CompleteIo>,
}

/// Create an I/O worker managing an io_uring and sending responses back via channels to a number
/// of handles.
pub fn start_io_pool(io_workers: usize, page_pool: PagePool) -> IoPool {
    let sender = platform::start_io_worker(io_workers);
    IoPool { sender, page_pool }
}

/// A manager for the broader I/O pool. This can be used to create new I/O handles.
///
/// Dropping this does not close any outstanding I/O handles or shut down I/O workers.
pub struct IoPool {
    sender: Sender<IoPacket>,
    page_pool: PagePool,
}

impl IoPool {
    /// Create a new I/O handle.
    pub fn make_handle(&self) -> IoHandle {
        let (completion_sender, completion_receiver) = crossbeam_channel::unbounded();
        IoHandle {
            page_pool: self.page_pool.clone(),
            sender: self.sender.clone(),
            completion_sender,
            completion_receiver,
        }
    }

    pub fn page_pool(&self) -> &PagePool {
        &self.page_pool
    }
}

/// A handle for submitting I/O commands and receiving their completions.
///
/// Only completions for commands submitted on this handle or its clones will be received, and in
/// no guaranteed order.
///
/// Clones, like normal channel clones, do not create a new handle but instead read from the same
/// stream of completions.
///
/// This is safe to use across multiple threads, but care must be taken by the user for correctness.
#[derive(Clone)]
pub struct IoHandle {
    page_pool: PagePool,
    sender: Sender<IoPacket>,
    completion_sender: Sender<CompleteIo>,
    completion_receiver: Receiver<CompleteIo>,
}

impl IoHandle {
    /// Block the current thread to send an I/O command. This fails if the channel has hung up.
    pub fn send(&self, command: IoCommand) -> Result<(), SendError<IoCommand>> {
        self.sender
            .send(IoPacket {
                command,
                completion_sender: self.completion_sender.clone(),
            })
            .map_err(|SendError(packet)| SendError(packet.command))
    }

    /// Try to send an I/O command without blocking. This fails if the channel is full or has
    /// disconnected.
    pub fn try_send(&self, command: IoCommand) -> Result<(), TrySendError<IoCommand>> {
        self.sender
            .try_send(IoPacket {
                command,
                completion_sender: self.completion_sender.clone(),
            })
            .map_err(|err| match err {
                TrySendError::Full(packet) => TrySendError::Full(packet.command),
                TrySendError::Disconnected(packet) => TrySendError::Disconnected(packet.command),
            })
    }

    /// Block the current thread on receiving an I/O completion.
    /// This fails if the channel has hung up.
    pub fn recv(&self) -> Result<CompleteIo, RecvError> {
        self.completion_receiver.recv()
    }

    /// Try to receive an I/O completion without blocking.
    pub fn try_recv(&self) -> Result<CompleteIo, TryRecvError> {
        self.completion_receiver.try_recv()
    }

    /// Get the underlying receiver.
    pub fn receiver(&self) -> &Receiver<CompleteIo> {
        &self.completion_receiver
    }

    pub fn page_pool(&self) -> &PagePool {
        &self.page_pool
    }
}

/// Read a page from the file at the given page number.
pub fn read_page(page_pool: &PagePool, fd: &File, pn: u64) -> std::io::Result<FatPage> {
    let mut page = page_pool.alloc_fat_page();
    read_page_into(&mut *page, fd, pn)?;
    Ok(page)
}

/// Read a page from the file into the provided buffer. The buffer must have size `PAGE_SIZE`.
pub fn read_page_into(buf: &mut [u8], fd: &File, pn: u64) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt as _;
    fd.read_exact_at(&mut buf[..], pn * PAGE_SIZE as u64)?;
    Ok(())
}
