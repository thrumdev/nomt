// A low-level module for spawning a child process and figuring out if we are the parent or the
// child using the same binary.
//
// The parent spawns a child process and passes a socket to it. The socket is passed to the child
// via a predefined file descriptor. The child then uses this file descriptor to communicate with
// the parent.
//
// For a process launched using the common binary, it can check if it is a child by checking if the
// [`CANARY_SOCKET_FD`] is valid.
//
// The main goal of this module is to tuck away the low-level machinery like working with libc and
// nix into a single place.

use anyhow::{bail, Result};
use cfg_if::cfg_if;
use libc::posix_spawn_file_actions_t;
use std::{
    ffi::CString,
    mem,
    os::{
        fd::{AsRawFd as _, FromRawFd as _, RawFd},
        raw::c_char,
        unix::net::UnixStream,
    },
    ptr,
    sync::atomic::{AtomicBool, Ordering},
};

/// A special file descriptor that is used to pass a socket to the child process.
///
/// We pick a high number to avoid conflicts with other file descriptors.
const CANARY_SOCKET_FD: RawFd = 1000;

/// Checks for evidence that this process is a child of a parent process that spawned it.
///
/// Returns a UnixStream if the process is a child, otherwise returns None.
pub fn am_spawned() -> Option<UnixStream> {
    static CALLED: AtomicBool = AtomicBool::new(false);

    // Only take ownership of the fd if we haven't already
    if CALLED.swap(true, Ordering::SeqCst) {
        return None;
    }

    let is_valid_fd = unsafe { libc::fcntl(CANARY_SOCKET_FD, libc::F_GETFD) != -1 };
    if !is_valid_fd {
        return None;
    }

    // Check if it's actually a Unix domain socket
    let mut type_: libc::c_int = 0;
    let mut type_len = std::mem::size_of::<libc::c_int>() as libc::socklen_t;

    let is_unix_socket = unsafe {
        libc::getsockopt(
            CANARY_SOCKET_FD,
            libc::SOL_SOCKET,
            libc::SO_TYPE,
            &mut type_ as *mut _ as *mut _,
            &mut type_len,
        ) == 0
            && type_ == libc::SOCK_STREAM
    };

    if !is_unix_socket {
        return None;
    }

    let stream = unsafe {
        // SAFETY:
        // - The file descriptor is valid (checked above with fcntl)
        // - We verified it's actually a Unix domain socket (checked with getsockopt)
        // - This code can only run once due to the TAKEN atomic bool, ensuring we have exclusive
        //   ownership, passing it down into the UnixStream instance.
        // - No other code could have taken ownership as this is the first access (TAKEN was false)
        UnixStream::from_raw_fd(CANARY_SOCKET_FD)
    };
    Some(stream)
}

pub fn spawn_child() -> Result<(Child, UnixStream)> {
    let (sock1, sock2) = UnixStream::pair()?;

    let child_pid = spawn_child_with_sock(sock2.as_raw_fd())?;
    drop(sock2); // Close parent's end in child

    Ok((Child { pid: child_pid }, sock1))
}

pub struct Child {
    pub pid: libc::pid_t,
}

impl Child {
    /// Wait an event from the child.
    pub fn wait(&self) -> i32 {
        let mut status = 0;
        unsafe {
            println!("Waiting for child to finish");
            libc::waitpid(self.pid, &mut status, 0);
            println!("Child finished");
        }
        status
    }
}

fn spawn_child_with_sock(socket_fd: RawFd) -> Result<libc::pid_t> {
    let mut file_actions = SpawnFileActions::new()?;
    file_actions.add_dup2(socket_fd, CANARY_SOCKET_FD)?;

    // Prepare argv for the child process.
    //
    // Contains only the program binary path and a null terminator.
    cfg_if! {
        if #[cfg(target_os = "linux")] {
            // Nothing beats the simplicity of /proc/self/exe on Linux.
            let program_c = CString::new("/proc/self/exe")?;
        } else {
            let program_c = CString::new(std::env::current_exe()?.to_string_lossy().as_bytes())?;
        }
    }
    let args = &mut [program_c.as_ptr(), ptr::null_mut()];

    let mut pid: libc::pid_t = 0;
    let ret = unsafe {
        // Spawn the child process
        //
        // SAFETY:
        // - `program_c` is freed after calling `posix_spawn`.
        // - `args_raw` is freed after calling `posix_spawn`.
        libc::posix_spawn(
            &mut pid,
            program_c.as_ptr(),
            file_actions.as_inner_ptr(),
            ptr::null(),
            args.as_mut_ptr() as *mut *mut c_char,
            std::ptr::null(),
        )
    };

    // Explicit drop here after `posix_spawn` to avoid dangling pointer.
    drop(program_c);
    drop(file_actions);

    if ret != 0 {
        bail!("Failed to spawn child process");
    }

    Ok(pid)
}

/// A wrapper around `posix_spawn_file_actions_t` that ensures the file actions are destroyed when
/// the wrapper is dropped.
struct SpawnFileActions {
    file_actions: posix_spawn_file_actions_t,
}

impl SpawnFileActions {
    fn new() -> anyhow::Result<Self> {
        let mut file_actions: posix_spawn_file_actions_t = unsafe { mem::zeroed() };
        unsafe {
            if libc::posix_spawn_file_actions_init(&mut file_actions) != 0 {
                bail!("Failed to init file actions");
            }
        }
        Ok(Self { file_actions })
    }

    /// Adds a dup2 action to the file actions.
    ///
    /// This duplicates the `src` file descriptor from the current process to the `dst` file
    /// descriptor into the child process.
    fn add_dup2(&mut self, src: RawFd, dst: RawFd) -> Result<()> {
        unsafe {
            if libc::posix_spawn_file_actions_adddup2(&mut self.file_actions, src, dst) != 0 {
                bail!("Failed to add dup2 action");
            }
        }
        Ok(())
    }

    fn as_inner_ptr(&self) -> *const posix_spawn_file_actions_t {
        &self.file_actions
    }
}

impl Drop for SpawnFileActions {
    fn drop(&mut self) {
        unsafe {
            let _ = libc::posix_spawn_file_actions_destroy(&mut self.file_actions);
        }
    }
}
