use crossbeam_channel::{Receiver, Sender};

use std::sync::Arc;

use super::{CompleteIo, IoCommand, IoKind, PAGE_SIZE};

const IO_THREADS: usize = 16;

// max number of inflight requests is bounded by the threadpool.
const MAX_IN_FLIGHT: usize = IO_THREADS;

pub fn start_io_worker(
    num_handles: usize,
    _num_rings: usize,
) -> (Sender<IoCommand>, Vec<Receiver<CompleteIo>>) {
    let (command_tx, command_rx) = crossbeam_channel::bounded(MAX_IN_FLIGHT);
    let (handle_txs, handle_rxs) = (0..num_handles)
        .map(|_| crossbeam_channel::unbounded())
        .unzip::<_, _, Vec<Sender<CompleteIo>>, _>();

    let handle_txs = Arc::new(handle_txs);

    for _ in 0..IO_THREADS {
        spawn_worker_thread(command_rx.clone(), handle_txs.clone());
    }

    (command_tx, handle_rxs)
}

fn spawn_worker_thread(command_rx: Receiver<IoCommand>, handle_txs: Arc<Vec<Sender<CompleteIo>>>) {
    let work = move || loop {
        let Ok(command) = command_rx.recv() else {
            break;
        };
        let handle_index = command.handle;
        let complete = execute(command);
        let _ = handle_txs[handle_index].send(complete);
    };

    std::thread::Builder::new()
        .name("nomt-io-worker".to_string())
        .spawn(work)
        .unwrap();
}

fn execute(mut command: IoCommand) -> CompleteIo {
    let err = match command.kind {
        IoKind::Read(fd, page_index, ref mut page) => unsafe {
            libc::pread(
                fd,
                page.as_mut_ptr() as *mut libc::c_void,
                PAGE_SIZE as libc::size_t,
                (page_index * PAGE_SIZE as u64) as libc::off_t,
            ) == -1
        },
        IoKind::Write(fd, page_index, ref page) => unsafe {
            libc::pwrite(
                fd,
                page.as_ptr() as *const libc::c_void,
                PAGE_SIZE as libc::size_t,
                (page_index * PAGE_SIZE as u64) as libc::off_t,
            ) == -1
        },
        IoKind::WriteRaw(fd, page_index, ptr, size) => unsafe {
            libc::pwrite(
                fd,
                ptr as *const libc::c_void,
                size as libc::size_t,
                (page_index * PAGE_SIZE as u64) as libc::off_t,
            ) == -1
        },
        IoKind::Fsync(fd) => unsafe { libc::fsync(fd) == -1 },
    };

    let result = if err {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    };

    CompleteIo { command, result }
}
