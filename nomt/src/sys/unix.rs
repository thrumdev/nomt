//! Common Unix definitions.

use std::{fs::File, os::fd::AsRawFd as _};

pub fn try_lock_exclusive(file: &File) -> std::io::Result<()> {
    unsafe {
        if libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) == -1 {
            Err(std::io::Error::last_os_error())
        } else {
            Ok(())
        }
    }
}

pub fn unlock(file: &File) -> std::io::Result<()> {
    unsafe {
        if libc::flock(file.as_raw_fd(), libc::LOCK_UN) == -1 {
            Err(std::io::Error::last_os_error())
        } else {
            Ok(())
        }
    }
}
