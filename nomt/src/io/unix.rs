use super::{CompleteIo, IoCommand, IoKind, IoKindResult, IoPacket, PAGE_SIZE};
use crossbeam_channel::{Receiver, Sender};

// max number of inflight requests is bounded by the threadpool.
const MAX_IN_FLIGHT: usize = 64;

pub fn start_io_worker(io_workers: usize) -> Sender<IoPacket> {
    let (command_tx, command_rx) = crossbeam_channel::bounded(MAX_IN_FLIGHT);

    for _ in 0..io_workers {
        spawn_worker_thread(command_rx.clone());
    }

    command_tx
}

fn spawn_worker_thread(command_rx: Receiver<IoPacket>) {
    let work = move || loop {
        let Ok(packet) = command_rx.recv() else {
            break;
        };
        let complete = execute(packet.command);
        let _ = packet.completion_sender.send(complete);
    };

    std::thread::Builder::new()
        .name("nomt-io-worker".to_string())
        .spawn(work)
        .unwrap();
}

fn execute(mut command: IoCommand) -> CompleteIo {
    let result = loop {
        let res = match command.kind {
            IoKind::Read(fd, page_index, ref mut page) => unsafe {
                libc::pread(
                    fd,
                    page.as_mut_ptr() as *mut libc::c_void,
                    PAGE_SIZE as libc::size_t,
                    (page_index * PAGE_SIZE as u64) as libc::off_t,
                )
            },
            IoKind::Write(fd, page_index, ref page) => unsafe {
                libc::pwrite(
                    fd,
                    page.as_ptr() as *const libc::c_void,
                    PAGE_SIZE as libc::size_t,
                    (page_index * PAGE_SIZE as u64) as libc::off_t,
                )
            },
            IoKind::WriteRaw(fd, page_index, ptr, size) => unsafe {
                libc::pwrite(
                    fd,
                    ptr as *const libc::c_void,
                    size as libc::size_t,
                    (page_index * PAGE_SIZE as u64) as libc::off_t,
                )
            },
        };

        match command.kind.get_result(res) {
            IoKindResult::Ok => break Ok(()),
            IoKindResult::Err => break Err(std::io::Error::last_os_error()),
            _ => (),
        }
    };

    CompleteIo { command, result }
}
