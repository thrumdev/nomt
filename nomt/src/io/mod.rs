#[cfg(not(target_family = "unix"))]
std::compile_error!("NOMT only supports Unix-based OSs");

use crossbeam_channel::{Receiver, RecvError, SendError, Sender, TryRecvError, TrySendError};
use std::{
    fs::File,
    ops::{Deref, DerefMut},
    os::fd::RawFd,
};

#[cfg(target_os = "linux")]
#[path = "linux.rs"]
mod platform;

#[cfg(not(target_os = "linux"))]
#[path = "unix.rs"]
mod platform;

pub const PAGE_SIZE: usize = 4096;

#[derive(Clone)]
#[repr(align(4096))]
pub struct Page(pub [u8; PAGE_SIZE]);

impl Deref for Page {
    type Target = [u8];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Page {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Page {
    pub fn zeroed() -> Self {
        Self([0; PAGE_SIZE])
    }
}

#[derive(Clone)]
pub enum IoKind {
    Read(RawFd, u64, Box<Page>),
    Write(RawFd, u64, Box<Page>),
    WriteRaw(RawFd, u64, *const u8, usize),
}

pub enum IoKindResult {
    Ok,
    Err,
    Retry,
}

impl IoKind {
    pub fn unwrap_buf(self) -> Box<Page> {
        match self {
            IoKind::Read(_, _, buf) | IoKind::Write(_, _, buf) => buf,
            IoKind::WriteRaw(_, _, _, _) => panic!("attempted to extract buf from write_raw"),
        }
    }

    pub fn get_result(&self, res: isize) -> IoKindResult {
        match self {
            // pread returns 0 if the file has been read till the end of file
            //
            // This could be a failure because the end of the file could be smaller than PAGE_SIZE.
            // However, as each read operation follows a write operation,
            // there should be no unexpected end-of-file that is not aligned with PAGE_SIZE
            // when all previous writes have succeeded.
            IoKind::Read(_, _, _) if res == 0 => IoKindResult::Ok,
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
pub fn start_io_pool(io_workers: usize) -> IoPool {
    let sender = platform::start_io_worker(io_workers);
    IoPool { sender }
}

/// A manager for the broader I/O pool. This can be used to create new I/O handles.
///
/// Dropping this does not close any outstanding I/O handles or shut down I/O workers.
pub struct IoPool {
    sender: Sender<IoPacket>,
}

impl IoPool {
    /// Create a new I/O handle.
    pub fn make_handle(&self) -> IoHandle {
        let (completion_sender, completion_receiver) = crossbeam_channel::unbounded();
        IoHandle {
            sender: self.sender.clone(),
            completion_sender,
            completion_receiver,
        }
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
}

/// Read a page from the file at the given page number.
pub fn read_page(fd: &File, pn: u64) -> std::io::Result<Box<Page>> {
    use std::os::unix::fs::FileExt as _;
    let mut page = Box::new(Page::zeroed());
    fd.read_exact_at(&mut page, pn * PAGE_SIZE as u64)?;
    Ok(page)
}
