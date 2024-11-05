//! Linux-specific code.

use super::unix::cvt_r;
use std::fs::File;
use std::os::fd::AsRawFd;

/// Returns true if the file is on a tmpfs filesystem.
/// False if it's not or the check fails.
pub fn tmpfs_check(file: &File) -> bool {
    unsafe {
        // SAFETY: unsafe because ffi call. This should be IO-safe because the file is passed
        //         by reference. This should be memory-safe because the `statfs` struct is
        //         zeroed and the `f_type` field should be set by the ffi call.
        let mut stat: libc::statfs = std::mem::zeroed();
        cvt_r(|| libc::fstatfs(file.as_raw_fd(), &mut stat))
            .map(|_| stat.f_type == libc::TMPFS_MAGIC)
            .unwrap_or(false)
    }
}

/// fallocate changes the size of the file to the given length if it's less than the current size.
/// If the file is larger than the given length, the file is not truncated.
///
/// Doesn't work on tmpfs.
pub fn falloc_zero_file(file: &File, len: u64) -> std::io::Result<()> {
    cvt_r(|| unsafe {
        // SAFETY: unsafe because ffi call. This should be IO-safe because the file is passed
        //         by reference.
        libc::fallocate(
            file.as_raw_fd(),
            libc::FALLOC_FL_ZERO_RANGE,
            0 as _,
            len as _,
        )
    })
    .map(drop)
}
