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

use anyhow::Result;
use cfg_if::cfg_if;
use std::{
    os::{
        fd::{AsRawFd as _, FromRawFd as _, RawFd},
        unix::{net::UnixStream, process::CommandExt},
    },
    process::Command,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};
use tracing::trace;

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

    let child = spawn_child_with_sock(sock2.as_raw_fd())?;
    // drop(sock2); // Close parent's end in child

    let pid = child.id() as i32;
    let shared = Shared { child, sock2 };

    Ok((
        Child {
            pid,
            shared: Arc::new(shared),
        },
        sock1,
    ))
}

fn spawn_child_with_sock(socket_fd: RawFd) -> Result<std::process::Child> {
    trace!(?socket_fd, "Spawning child process");

    // Prepare argv for the child process.
    //
    // Contains only the program binary path and a null terminator.
    cfg_if! {
        if #[cfg(target_os = "linux")] {
            // Nothing beats the simplicity of /proc/self/exe on Linux.
            let program = std::ffi::OsString::from("/proc/self/exe");
        } else {
            let program = std::env::current_exe()?;
        }
    }

    let mut cmd = Command::new(program);
    unsafe {
        cmd.pre_exec(move || {
            // Duplicate the socket_fd to the CANARY_SOCKET_FD
            libc::dup2(socket_fd, CANARY_SOCKET_FD);
            libc::close(socket_fd);
            Ok(())
        });
    }
    let child = cmd.spawn()?;

    trace!("spawned child process, pid={}", child.id());
    Ok(child)
}

struct Shared {
    child: std::process::Child,
    sock2: UnixStream,
}

#[derive(Clone)]
pub struct Child {
    pub pid: i32,
    shared: Arc<Shared>,
}

impl Child {
    /// Wait an event from the child.
    ///
    /// Blocking.
    pub fn wait(&self) -> i32 {
        let mut status = 0;
        unsafe {
            libc::waitpid(self.pid, &mut status, 0);
        }
        status
    }

    /// Sends a SIGKILL signal to the child process.
    ///
    /// One signal is usually enough.
    ///
    /// NB: This construction is prone to races. The child process may have already exited by the
    /// time the signal is sent, and the PID may have been reused by another process. There is a way
    /// to avoid that with pidfd but nobody cares really, because the probability of that happening
    /// is extremely low.
    pub fn send_sigkill(&self) {
        trace!("sending SIGKILL to the agent, pid={}", self.pid);
        unsafe {
            libc::kill(self.pid, libc::SIGKILL);
        }
    }
}
