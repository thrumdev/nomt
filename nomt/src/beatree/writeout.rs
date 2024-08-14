use std::{
    fs::File,
    os::fd::{FromRawFd, IntoRawFd, RawFd},
};

use crossbeam_channel::{Receiver, Sender, TrySendError};

use crate::{
    io::Page,
    io::{CompleteIo, IoCommand, IoKind},
};

use super::{allocator::PageNumber, branch::BranchNode, meta::Meta};

pub fn run(
    io_sender: Sender<IoCommand>,
    io_handle_index: usize,
    io_receiver: Receiver<CompleteIo>,
    bbn_fd: RawFd,
    ln_fd: RawFd,
    meta_fd: RawFd,
    bbn: Vec<BranchNode>,
    bbn_free_list_pages: Vec<(PageNumber, Box<Page>)>,
    bbn_extend_file_sz: Option<u64>,
    ln: Vec<(PageNumber, Box<Page>)>,
    ln_extend_file_sz: Option<u64>,
    new_meta: Meta,
) {
    let now = std::time::Instant::now();
    let io = IoDmux::new(io_sender, io_handle_index, io_receiver);
    do_run(
        Cx {
            bbn_write_out: BbnWriteOut {
                bbn_fd,
                bbn_extend_file_sz,
                remaining: bbn.len() + bbn_free_list_pages.len(),
                bbn,
                free_list_pages: bbn_free_list_pages,
            },
            ln_write_out: LnWriteOut {
                ln_fd,
                ln_extend_file_sz,
                ln_remaining: ln.len(),
                ln,
            },
            meta_swap: Some(MetaSwap {
                meta_fd,
                new_meta: Some(new_meta),
                should_fsync: false,
            }),
        },
        io,
    );
    println!("writeout {}us", now.elapsed().as_micros());
}

fn do_run(mut cx: Cx, mut io: IoDmux) {
    // This should perform the following actions:
    // - truncate the BBN file to the correct size.
    // - truncate the LN file to the correct size.
    // - write the BBN pages
    // - write the LN pages
    // - fsync the BBN file
    // - fsync the LN file
    // - update the meta file
    // - fsync on meta file

    cx.bbn_write_out.extend_file();
    cx.bbn_write_out.send_writes(&mut io);

    cx.ln_write_out.extend_file();
    cx.ln_write_out.send_writes(&mut io);

    cx.bbn_write_out.wait_writes(&mut io);
    cx.ln_write_out.wait_writes(&mut io);

    cx.bbn_write_out.fsync(&mut io);
    cx.ln_write_out.fsync(&mut io);

    loop {
        // At this point, the BBN and LN files are fully synced. We can now update the meta file.
        if let Some(ref mut meta_swap) = &mut cx.meta_swap {
            if meta_swap.run(&mut io) {
                cx.meta_swap = None;
                return;
            }
        }
    }
}

struct IoDmux {
    io_sender: Sender<IoCommand>,
    io_handle_index: usize,
    io_receiver: Receiver<CompleteIo>,
    bbn_inbox: Vec<CompleteIo>,
    ln_inbox: Vec<CompleteIo>,
    meta_inbox: Vec<CompleteIo>,
}

impl IoDmux {
    const BBN_USER_DATA: u64 = 0;
    const LN_USER_DATA: u64 = 1;
    const META_USER_DATA: u64 = 2;

    fn new(
        io_sender: Sender<IoCommand>,
        io_handle_index: usize,
        io_receiver: Receiver<CompleteIo>,
    ) -> Self {
        Self {
            io_sender,
            io_handle_index,
            io_receiver,
            bbn_inbox: Vec::new(),
            ln_inbox: Vec::new(),
            meta_inbox: Vec::new(),
        }
    }

    fn try_recv_bbn(&mut self) -> Option<CompleteIo> {
        match self.bbn_inbox.pop() {
            Some(io) => Some(io),
            None => self.try_poll(Self::BBN_USER_DATA),
        }
    }

    fn recv_bbn_write(&mut self) {
        loop {
            if let Some(CompleteIo { command, result }) = self.try_recv_bbn() {
                assert!(result.is_ok());
                match command.kind {
                    IoKind::Write(_, _, _) | IoKind::WriteRaw(_, _, _, _) => {}
                    _ => panic!("unexpected completion kind, expected: Write or WriteRaw"),
                }
                break;
            }
        }
    }

    fn recv_bbn_fsync(&mut self) {
        loop {
            if let Some(CompleteIo { command, result }) = self.try_recv_bbn() {
                assert!(result.is_ok());
                match command.kind {
                    IoKind::Fsync(_) => {}
                    _ => panic!("unexpected completion kind, expected: Write or WriteRaw"),
                }
                break;
            }
        }
    }

    fn recv_ln_write(&mut self) {
        loop {
            if let Some(CompleteIo { command, result }) = self.try_recv_ln() {
                assert!(result.is_ok());
                match command.kind {
                    IoKind::Write(_, _, _) => {}
                    _ => panic!("unexpected completion kind, expected: Write or WriteRaw"),
                }
                break;
            }
        }
    }

    fn recv_ln_fsync(&mut self) {
        loop {
            if let Some(CompleteIo { command, result }) = self.try_recv_ln() {
                assert!(result.is_ok());
                match command.kind {
                    IoKind::Fsync(_) => {}
                    _ => panic!("unexpected completion kind, expected: Write or WriteRaw"),
                }
                break;
            }
        }
    }

    fn try_send_bbn(&mut self, kind: IoKind) -> Result<(), TrySendError<IoCommand>> {
        self.io_sender.try_send(IoCommand {
            kind,
            handle: self.io_handle_index,
            user_data: Self::BBN_USER_DATA,
        })
    }

    fn try_send_ln(&mut self, kind: IoKind) -> Result<(), TrySendError<IoCommand>> {
        self.io_sender.try_send(IoCommand {
            kind,
            handle: self.io_handle_index,
            user_data: Self::LN_USER_DATA,
        })
    }

    fn try_recv_ln(&mut self) -> Option<CompleteIo> {
        match self.ln_inbox.pop() {
            Some(io) => Some(io),
            None => self.try_poll(Self::LN_USER_DATA),
        }
    }

    fn try_send_meta(&self, kind: IoKind) -> Result<(), TrySendError<IoCommand>> {
        self.io_sender.try_send(IoCommand {
            kind,
            handle: self.io_handle_index,
            user_data: Self::META_USER_DATA,
        })
    }

    fn try_recv_meta(&mut self) -> Option<CompleteIo> {
        match self.meta_inbox.pop() {
            Some(io) => Some(io),
            None => self.try_poll(Self::META_USER_DATA),
        }
    }

    /// Try to receive a completion from the io channel. If the completion is for the given user
    /// data, it is immediately returned. Otherwise, it is pushed into the appropriate inbox.
    fn try_poll(&mut self, eager_ud: u64) -> Option<CompleteIo> {
        match self.io_receiver.try_recv() {
            Ok(io) => {
                if io.command.user_data == eager_ud {
                    Some(io)
                } else {
                    match io.command.user_data {
                        Self::BBN_USER_DATA => {
                            self.bbn_inbox.push(io);
                            None
                        }
                        Self::LN_USER_DATA => {
                            self.ln_inbox.push(io);
                            None
                        }
                        Self::META_USER_DATA => {
                            self.meta_inbox.push(io);
                            None
                        }
                        _ => panic!("unexpected user data"),
                    }
                }
            }
            Err(_) => {
                // For future, we should seal the channel and don't attemp to poll it again.
                None
            }
        }
    }
}

struct Cx {
    bbn_write_out: BbnWriteOut,
    ln_write_out: LnWriteOut,
    meta_swap: Option<MetaSwap>,
}

struct BbnWriteOut {
    bbn_fd: RawFd,
    bbn_extend_file_sz: Option<u64>,
    bbn: Vec<BranchNode>,
    free_list_pages: Vec<(PageNumber, Box<Page>)>,
    // Initially, set to the len of `bbn`. Each completion will decrement this.
    remaining: usize,
    should_fsync: bool,
}

impl BbnWriteOut {
    fn extend_file(&mut self) {
        // Turns out, as of this writing, io_uring doesn't support ftruncate. So we have to do it
        // via good old ftruncate syscall here.
        //
        // Do it first for now, so we don't have to deal with deferring writes until after the
        // ftruncate. Later on, we can schedule the writes before the bump pointer (size of the
        // file) right away and defer only the writes to the pages higher the bump pointe.
        if let Some(new_len) = self.bbn_extend_file_sz.take() {
            unsafe {
                let f = File::from_raw_fd(self.bbn_fd);
                f.set_len(new_len).unwrap();
                let _ = f.into_raw_fd();
            }
        }
    }

    fn send_writes(&mut self, io: &mut IoDmux) {
        while self.bbn.len() > 0 {
            // UNWRAP: we know the list is not empty.
            let mut branch_node = self.bbn.pop().unwrap();

            {
                // UNWRAP: the branch node cannot be out of bounds because of the requirement of the
                // sync machine.
                let wrt = branch_node.as_mut_slice();
                let (ptr, len) = (wrt.as_ptr(), wrt.len());
                let bbn_pn = branch_node.bbn_pn();

                if let Err(_) =
                    io.try_send_bbn(IoKind::WriteRaw(self.bbn_fd, bbn_pn as u64, ptr, len))
                {
                    // That's alright. We will retry after trying to get something out of the cqueue
                    io.recv_bbn_write();
                    self.remaining = self.remaining.checked_sub(1).unwrap();
                    self.bbn.push(branch_node);
                }
                let _ = branch_node;
            }
        }

        while self.free_list_pages.len() > 0 {
            // UNWRAP: we know the list is not empty.
            let (pn, page) = self.free_list_pages.pop().unwrap();

            {
                // UNWRAP: the branch node cannot be out of bounds because of the requirement of the
                // sync machine.
                if let Err(TrySendError::Full(IoCommand {
                    kind: IoKind::Write(.., page),
                    ..
                })) = io.try_send_bbn(IoKind::Write(self.bbn_fd, pn.0 as u64, page))
                {
                    // That's alright. We will try again after getting something out of the cqueue
                    io.recv_bbn_write();
                    self.remaining = self.remaining.checked_sub(1).unwrap();
                    self.free_list_pages.push((pn, page));
                }
            }
        }
    }

    fn wait_writes(&mut self, io: &mut IoDmux) {
        // wait for all writes
        while self.remaining > 0 {
            io.recv_bbn_write();
            self.remaining = self.remaining.checked_sub(1).unwrap();
        }
    }

    fn fsync(&mut self, io: &mut IoDmux) {
        while !io.try_send_bbn(IoKind::Fsync(self.bbn_fd)).is_ok() {}
        io.recv_bbn_fsync();
    }
}

struct LnWriteOut {
    ln_fd: RawFd,
    ln_extend_file_sz: Option<u64>,
    ln: Vec<(PageNumber, Box<Page>)>,
    ln_remaining: usize,
    should_fsync: bool,
}

impl LnWriteOut {
    fn extend_file(&mut self) {
        // Turns out, as of this writing, io_uring doesn't support ftruncate. So we have to do it
        // via good old ftruncate syscall here.
        //
        // Do it first for now, so we don't have to deal with deferring writes until after the
        // ftruncate. Later on, we can schedule the writes before the bump pointer (size of the
        // file) right away and defer only the writes to the pages higher the bump pointe.
        if let Some(new_len) = self.ln_extend_file_sz.take() {
            unsafe {
                let f = File::from_raw_fd(self.ln_fd);
                f.set_len(new_len).unwrap();
                let _ = f.into_raw_fd();
            }
        }
    }

    fn send_writes(&mut self, io: &mut IoDmux) {
        while self.ln.len() > 0 {
            // UNWRAP: we know the list is not empty.
            let (ln_pn, ln_page) = self.ln.pop().unwrap();

            {
                // UNWRAP: the branch node cannot be out of bounds because of the requirement of the
                // sync machine.
                if let Err(TrySendError::Full(IoCommand {
                    kind: IoKind::Write(.., ln_page),
                    ..
                })) = io.try_send_ln(IoKind::Write(self.ln_fd, ln_pn.0 as u64, ln_page))
                {
                    // That's alright. We will retry after trying to get something out of the cqueue
                    io.recv_ln_write();
                    self.ln_remaining = self.ln_remaining.checked_sub(1).unwrap();
                    self.ln.push((ln_pn, ln_page));
                }
            }
        }
    }

    fn wait_writes(&mut self, io: &mut IoDmux) {
        // wait for all writes
        while self.ln_remaining > 0 {
            io.recv_ln_write();
            self.ln_remaining = self.ln_remaining.checked_sub(1).unwrap();
        }
    }

    fn fsync(&mut self, io: &mut IoDmux) {
        while !io.try_send_ln(IoKind::Fsync(self.ln_fd)).is_ok() {}
        io.recv_ln_fsync();
    }
}

struct MetaSwap {
    meta_fd: RawFd,
    new_meta: Option<Meta>,
    should_fsync: bool,
}

impl MetaSwap {
    fn run(&mut self, io: &mut IoDmux) -> bool {
        if let Some(new_meta) = self.new_meta.take() {
            // Oh god, there is a special place in hell for this. Will do for now though.
            let mut page = Box::new(Page::zeroed());

            new_meta.encode_to(&mut page.as_mut()[..16]);

            if let Err(_) = io.try_send_meta(IoKind::Write(self.meta_fd, 0, page)) {
                self.new_meta = Some(new_meta);
                return false;
            }
        }

        if self.should_fsync {
            if let Err(_) = io.try_send_meta(IoKind::Fsync(self.meta_fd)) {
                return false;
            }
            self.should_fsync = false;
        }

        // Reap the completions.
        while let Some(CompleteIo { command, result }) = io.try_recv_meta() {
            assert!(result.is_ok());
            match command.kind {
                IoKind::Write(_, _, _) => {
                    self.should_fsync = true;
                    continue;
                }
                IoKind::Fsync(_) => {
                    // done
                    return true;
                }
                _ => panic!("unexpected completion kind"),
            }
        }

        false
    }
}
