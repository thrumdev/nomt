use std::{
    fs::File,
    os::fd::{FromRawFd, RawFd},
};

use crossbeam_channel::{Receiver, Sender, TrySendError};

use crate::{
    io::{CompleteIo, IoCommand, IoKind},
    store::Page,
};

use super::{
    branch::{BranchId, BranchNodePool},
    leaf::PageNumber,
    meta::Meta,
};

pub fn run(
    io_sender: Sender<IoCommand>,
    io_handle_index: usize,
    io_receiver: Receiver<CompleteIo>,
    bnp: BranchNodePool,
    bbn_fd: RawFd,
    ln_fd: RawFd,
    meta_fd: RawFd,
    bbn: Vec<BranchId>,
    bbn_extend_file_sz: Option<u64>,
    ln: Vec<(PageNumber, Box<Page>)>,
    ln_extend_file_sz: Option<u64>,
    new_meta: Meta,
) {
    let io = IoDmux::new(io_sender, io_handle_index, io_receiver);
    do_run(
        Cx {
            bbn_write_out: Some(BbnWriteOut {
                bbn_fd,
                bbn_extend_file_sz,
                bbn_remaining: bbn.len(),
                bbn,
            }),
            ln_write_out: Some(LnWriteOut {
                ln_fd,
                ln_extend_file_sz,
                ln_remaining: ln.len(),
                ln,
            }),
            meta_swap: Some(MetaSwap {
                meta_fd,
                new_meta: Some(new_meta),
                should_fsync: false,
            }),
        },
        io,
        bnp,
    );
}

fn do_run(mut cx: Cx, mut io: IoDmux, bnp: BranchNodePool) {
    // This should perform the following actions:
    // - truncate the BBN file to the correct size.
    // - truncate the LN file to the correct size.
    // - write the BBN pages
    // - write the LN pages
    // - fsync the BBN file
    // - fsync the LN file
    // - update the meta file
    // - fsync on meta file
    loop {
        if cx.bbn_write_out.is_some() || cx.ln_write_out.is_some() {
            if let Some(ref mut bbn_write_out) = &mut cx.bbn_write_out {
                if bbn_write_out.run(&bnp, &mut io) {
                    cx.bbn_write_out = None;
                }
            }
            if let Some(ref mut ln_write_out) = &mut cx.ln_write_out {
                if ln_write_out.run(&mut io) {
                    cx.ln_write_out = None;
                }
            }
            continue;
        }

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
    bbn_write_out: Option<BbnWriteOut>,
    ln_write_out: Option<LnWriteOut>,
    meta_swap: Option<MetaSwap>,
}

struct BbnWriteOut {
    bbn_fd: RawFd,
    bbn_extend_file_sz: Option<u64>,
    bbn: Vec<BranchId>,
    // Initially, set to the len of `bbn`. Each completion will decrement this.
    bbn_remaining: usize,
}

impl BbnWriteOut {
    fn run(&mut self, bnp: &BranchNodePool, io: &mut IoDmux) -> bool {
        // Turns out, as of this writing, io_uring doesn't support ftruncate. So we have to do it
        // via good old ftruncate syscall here.
        //
        // Do it first for now, so we don't have to deal with deferring writes until after the
        // ftruncate. Later on, we can schedule the writes before the bump pointer (size of the
        // file) right away and defer only the writes to the pages higher the bump pointe.
        if let Some(new_len) = self.bbn_extend_file_sz.take() {
            unsafe {
                File::from_raw_fd(self.bbn_fd).set_len(new_len).unwrap();
            }
            return false;
        }

        while self.bbn.len() > 0 {
            // UNWRAP: we know the list is not empty.
            let bnid = self.bbn.pop().unwrap();

            {
                // UNWRAP: the branch node cannot be out of bounds because of the requirement of the
                // sync machine.
                let mut page = bnp.checkout(bnid).unwrap();
                let wrt = page.as_mut_slice();
                let (ptr, len) = (wrt.as_ptr(), wrt.len());
                let bbn_pn = page.bbn_pn();

                if let Err(_) =
                    io.try_send_bbn(IoKind::WriteRaw(self.bbn_fd, bbn_pn as u64, ptr, len))
                {
                    // That's alright. We will retry on the next iteration.
                    self.bbn.push(bnid);
                    return false;
                }
                let _ = page;
            }
        }

        // Reap the completions.
        while let Some(CompleteIo { command, result }) = io.try_recv_bbn() {
            assert!(result.is_ok());
            match command.kind {
                IoKind::WriteRaw(_, _, _, _) => {
                    self.bbn_remaining = self.bbn_remaining.checked_sub(1).unwrap();

                    if self.bbn_remaining == 0 {
                        if let Err(_) = io.try_send_bbn(IoKind::Fsync(self.bbn_fd)) {
                            return false;
                        }
                    }

                    continue;
                }
                IoKind::Fsync(_) => {
                    assert!(self.bbn_remaining == 0);
                    return true;
                }
                _ => panic!("unexpected completion kind"),
            }
        }

        // we are not done yet.
        false
    }
}

struct LnWriteOut {
    ln_fd: RawFd,
    ln_extend_file_sz: Option<u64>,
    ln: Vec<(PageNumber, Box<Page>)>,
    ln_remaining: usize,
}

impl LnWriteOut {
    fn run(&mut self, io: &mut IoDmux) -> bool {
        // NOTE: The current process begins by `ftruncating` the file to support writing
        // all pages contained in `self.ln`.
        // This is necessary because the pages created by the LeafStore fall into two categories:
        // - pages that can be safely written to the file
        // - pages that need the file to be extended because their page number is out of range
        //
        // The first category is currently waiting until `ftruncate` is performed before
        // being written with the second group.
        //
        // A possible improvement could be to allow the LeafStore to write all pages
        // in the first category immediately, keep a vector of pending pages,
        // and have the writeout call `ftruncate` to finish issuing all pending pages.

        if let Some(new_len) = self.ln_extend_file_sz.take() {
            unsafe {
                File::from_raw_fd(self.ln_fd).set_len(new_len).unwrap();
            }
        }

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
                    // That's alright. We will retry on the next iteration.
                    self.ln.push((ln_pn, ln_page));
                    return false;
                }
            }
        }

        // Reap the completions.
        while let Some(CompleteIo { command, result }) = io.try_recv_ln() {
            assert!(result.is_ok());
            match command.kind {
                IoKind::WriteRaw(_, _, _, _) => {
                    self.ln_remaining = self.ln_remaining.checked_sub(1).unwrap();

                    if self.ln_remaining == 0 {
                        if let Err(_) = io.try_send_ln(IoKind::Fsync(self.ln_fd)) {
                            return false;
                        }
                    }

                    continue;
                }
                IoKind::Fsync(_) => {
                    assert!(self.ln_remaining == 0);
                    return true;
                }
                _ => panic!("unexpected completion kind"),
            }
        }

        // we are not done yet.
        false
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
            new_meta.encode_to(&mut page.as_mut()[..24].try_into().unwrap());

            if let Err(_) = io.try_send_meta(IoKind::Write(self.meta_fd, 0, page)) {
                self.new_meta = Some(new_meta);
                return false;
            }
        }

        if self.should_fsync {
            if let Err(_) = io.try_send_meta(IoKind::Fsync(self.meta_fd)) {
                return false;
            }
        }

        // Reap the completions.
        while let Some(CompleteIo { command, result }) = io.try_recv_meta() {
            assert!(result.is_ok());
            match command.kind {
                IoKind::WriteRaw(_, _, _, _) => {
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
