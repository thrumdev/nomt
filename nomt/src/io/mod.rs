#[cfg(not(target_family = "unix"))]
std::compile_error!("NOMT only supports Unix-based OSs");

#[cfg(not(target_os = "linux"))]
std::compile_error!("temporary until unix support is achieved");

use crossbeam_channel::{Sender, Receiver};
use std::{
    ops::{Deref, DerefMut},
    os::fd::RawFd,
};

#[cfg(target_os = "linux")]
#[path = "linux.rs"]
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


pub type HandleIndex = usize;

#[derive(Clone)]
pub enum IoKind {
    Read(RawFd, u64, Box<Page>),
    Write(RawFd, u64, Box<Page>),
    WriteRaw(RawFd, u64, *const u8, usize),
    Fsync(RawFd),
}

impl IoKind {
    pub fn unwrap_buf(self) -> Box<Page> {
        match self {
            IoKind::Read(_, _, buf) | IoKind::Write(_, _, buf) => buf,
            IoKind::WriteRaw(_, _, _, _) => panic!("attempted to extract buf from write_raw"),
            IoKind::Fsync(_) => panic!("attempted to extract buf from fsync"),
        }
    }
}

unsafe impl Send for IoKind {}

pub struct IoCommand {
    pub kind: IoKind,
    pub handle: HandleIndex,
    // note: this isn't passed to io_uring, it's higher-level userdata.
    pub user_data: u64,
}

pub struct CompleteIo {
    pub command: IoCommand,
    pub result: std::io::Result<()>,
}

/// Create an I/O worker managing an io_uring and sending responses back via channels to a number
/// of handles.
pub fn start_io_worker(
    num_handles: usize,
    num_rings: usize,
) -> (Sender<IoCommand>, Vec<Receiver<CompleteIo>>) {
    platform::start_io_worker(num_handles, num_rings)
}
