use std::{
    fs::File,
    io::{Seek, SeekFrom},
    os::fd::{FromRawFd, IntoRawFd, RawFd},
};

use crossbeam_channel::TrySendError;

use crate::io::{CompleteIo, IoCommand, IoHandle, IoKind, Page, PAGE_SIZE};

use super::{
    beatree::{allocator::PageNumber, branch::BranchNode},
    meta::Meta,
};

pub fn run(
    io_handle: IoHandle,
    wal_fd: RawFd,
    bbn_fd: RawFd,
    ln_fd: RawFd,
    ht_fd: RawFd,
    meta_fd: RawFd,
    wal_blob: (*mut u8, usize),
    bbn: Vec<BranchNode>,
    bbn_free_list_pages: Vec<(PageNumber, Box<Page>)>,
    bbn_extend_file_sz: Option<u64>,
    ln: Vec<(PageNumber, Box<Page>)>,
    ln_extend_file_sz: Option<u64>,
    ht: Vec<(u64, Box<Page>)>,
    new_meta: Meta,
    panic_on_sync: bool,
) {
    let io = IoDmux::new(io_handle);
    do_run(
        Cx {
            wal_write_out: WalWriteOut::new(wal_fd, wal_blob),
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
            ht_write_out: HtWriteOut {
                ht_fd,
                ht_remaining: ht.len(),
                ht,
            },
            meta_swap: Some(MetaSwap {
                meta_fd,
                new_meta: Some(new_meta),
                should_fsync: false,
            }),
            panic_on_sync,
        },
        io,
    );
}

fn do_run(mut cx: Cx, mut io: IoDmux) {
    // This should perform the following actions:
    // - truncate the WAL file to empty.
    // - truncate the BBN file to the correct size.
    // - truncate the LN file to the correct size.
    // - write WAL blob.
    // - write the BBN pages
    // - write the LN pages
    // - fsync the WAL file
    // - fsync the BBN file
    // - fsync the LN file
    // - update the meta file
    // - fsync on meta file
    // - dump the metabits and bucket pages.
    // - truncate the WAL file.

    cx.wal_write_out.truncate_file();
    cx.wal_write_out.send_writes(&mut io);

    cx.bbn_write_out.extend_file();
    cx.bbn_write_out.send_writes(&mut io);

    cx.ln_write_out.extend_file();
    cx.ln_write_out.send_writes(&mut io);

    cx.wal_write_out.wait_writes(&mut io);
    cx.bbn_write_out.wait_writes(&mut io);
    cx.ln_write_out.wait_writes(&mut io);

    cx.wal_write_out.fsync();
    cx.bbn_write_out.fsync();
    cx.ln_write_out.fsync();

    loop {
        // At this point, the BBN and LN files are fully synced. We can now update the meta file.
        if let Some(ref mut meta_swap) = &mut cx.meta_swap {
            if meta_swap.run(&mut io) {
                cx.meta_swap = None;
                break;
            }
        }
    }

    if cx.panic_on_sync {
        panic!("panic_on_sync triggered");
    }

    // Dump the metabits and bucket pages.
    cx.ht_write_out.send_writes(&mut io);
    cx.ht_write_out.wait_writes(&mut io);
    cx.ht_write_out.fsync();

    // Finally, truncate the WAL file signifying that the sync is complete.
    cx.wal_write_out.truncate_file();
    cx.wal_write_out.fsync();
}

struct IoDmux {
    io_handle: IoHandle,
    wal_inbox: Vec<CompleteIo>,
    bbn_inbox: Vec<CompleteIo>,
    ln_inbox: Vec<CompleteIo>,
    ht_inbox: Vec<CompleteIo>,
    meta_inbox: Vec<CompleteIo>,
}

impl IoDmux {
    const WAL_USER_DATA: u64 = 0;
    const BBN_USER_DATA: u64 = 1;
    const LN_USER_DATA: u64 = 2;
    const HT_USER_DATA: u64 = 3;
    const META_USER_DATA: u64 = 4;

    fn new(io_handle: IoHandle) -> Self {
        Self {
            io_handle,
            wal_inbox: Vec::new(),
            bbn_inbox: Vec::new(),
            ln_inbox: Vec::new(),
            ht_inbox: Vec::new(),
            meta_inbox: Vec::new(),
        }
    }

    fn try_recv_bbn(&mut self) -> Option<CompleteIo> {
        match self.bbn_inbox.pop() {
            Some(io) => Some(io),
            None => self.try_poll(Self::BBN_USER_DATA),
        }
    }

    fn try_recv_wal(&mut self) -> Option<CompleteIo> {
        match self.wal_inbox.pop() {
            Some(io) => Some(io),
            None => self.try_poll(Self::WAL_USER_DATA),
        }
    }

    fn recv_wal_write(&mut self) {
        loop {
            if let Some(CompleteIo { command, result }) = self.try_recv_wal() {
                assert!(result.is_ok());
                match command.kind {
                    IoKind::WriteRaw(_, _, _, _) => {}
                    _ => panic!("unexpected completion kind, expected: WriteRaw"),
                }
                break;
            }
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

    fn try_send_wal(&mut self, kind: IoKind) -> Result<(), TrySendError<IoCommand>> {
        self.io_handle.try_send(IoCommand {
            kind,
            user_data: Self::WAL_USER_DATA,
        })
    }

    fn try_send_bbn(&mut self, kind: IoKind) -> Result<(), TrySendError<IoCommand>> {
        self.io_handle.try_send(IoCommand {
            kind,
            user_data: Self::BBN_USER_DATA,
        })
    }

    fn try_send_ln(&mut self, kind: IoKind) -> Result<(), TrySendError<IoCommand>> {
        self.io_handle.try_send(IoCommand {
            kind,
            user_data: Self::LN_USER_DATA,
        })
    }

    fn try_recv_ln(&mut self) -> Option<CompleteIo> {
        match self.ln_inbox.pop() {
            Some(io) => Some(io),
            None => self.try_poll(Self::LN_USER_DATA),
        }
    }

    fn try_send_ht(&self, kind: IoKind) -> Result<(), TrySendError<IoCommand>> {
        self.io_handle.try_send(IoCommand {
            kind,
            user_data: Self::HT_USER_DATA,
        })
    }

    fn try_recv_ht(&mut self) -> Option<CompleteIo> {
        match self.ht_inbox.pop() {
            Some(io) => Some(io),
            None => self.try_poll(Self::HT_USER_DATA),
        }
    }

    fn recv_ht_write(&mut self) {
        loop {
            if let Some(CompleteIo { command, result }) = self.try_recv_ht() {
                assert!(result.is_ok());
                match command.kind {
                    IoKind::Write(_, _, _) => {}
                    _ => panic!("unexpected completion kind, expected: Write"),
                }
                break;
            }
        }
    }

    fn try_send_meta(&self, kind: IoKind) -> Result<(), TrySendError<IoCommand>> {
        self.io_handle.try_send(IoCommand {
            kind,
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
        match self.io_handle.try_recv() {
            Ok(io) => {
                if io.command.user_data == eager_ud {
                    Some(io)
                } else {
                    match io.command.user_data {
                        Self::WAL_USER_DATA => {
                            self.wal_inbox.push(io);
                            None
                        }
                        Self::BBN_USER_DATA => {
                            self.bbn_inbox.push(io);
                            None
                        }
                        Self::LN_USER_DATA => {
                            self.ln_inbox.push(io);
                            None
                        }
                        Self::HT_USER_DATA => {
                            self.ht_inbox.push(io);
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
    wal_write_out: WalWriteOut,
    bbn_write_out: BbnWriteOut,
    ln_write_out: LnWriteOut,
    ht_write_out: HtWriteOut,
    meta_swap: Option<MetaSwap>,
    panic_on_sync: bool,
}

struct WalWriteOut {
    wal_fd: RawFd,
    wal_blob: (*mut u8, usize),
    wal_blob_pns: Vec<u32>,
    remaining: usize,
}

impl WalWriteOut {
    fn new(wal_fd: RawFd, wal_blob: (*mut u8, usize)) -> Self {
        // HACK: we cannot issue one big write, so we split it into multiple pages.
        let wal_blob_len = wal_blob.1;
        assert!(wal_blob_len % PAGE_SIZE == 0);
        let mut wal_blob_parts = Vec::new();
        for i in 0..wal_blob_len / PAGE_SIZE {
            wal_blob_parts.push(i as u32);
        }
        Self {
            wal_fd,
            wal_blob,
            remaining: wal_blob_parts.len(),
            wal_blob_pns: wal_blob_parts,
        }
    }

    fn truncate_file(&mut self) {
        unsafe {
            let mut f = File::from_raw_fd(self.wal_fd);
            f.set_len(0).unwrap();
            f.seek(SeekFrom::Start(0)).unwrap();
            let _ = f.into_raw_fd();
        }
    }

    fn send_writes(&mut self, io: &mut IoDmux) {
        // send all writes
        while self.wal_blob_pns.len() > 0 {
            // UNWRAP: we know the list is not empty.
            let pn = self.wal_blob_pns.pop().unwrap();

            unsafe {
                let ptr = self.wal_blob.0.add(pn as usize * PAGE_SIZE);
                let len = PAGE_SIZE;
                if let Err(_) = io.try_send_wal(IoKind::WriteRaw(self.wal_fd, pn as u64, ptr, len))
                {
                    // That's alright. We will retry after trying to get something out of the cqueue
                    io.recv_wal_write();
                    self.remaining = self.remaining.checked_sub(1).unwrap();
                    self.wal_blob_pns.push(pn);
                }
            }
        }
    }

    fn wait_writes(&mut self, io: &mut IoDmux) {
        // wait for all writes
        while self.remaining > 0 {
            io.recv_wal_write();
            self.remaining = self.remaining.checked_sub(1).unwrap();
        }
    }

    fn fsync(&self) {
        unsafe {
            let f = File::from_raw_fd(self.wal_fd);
            f.sync_all().expect("wal file: error performing fsync");
            let _ = f.into_raw_fd();
        }
    }
}

struct BbnWriteOut {
    bbn_fd: RawFd,
    bbn_extend_file_sz: Option<u64>,
    bbn: Vec<BranchNode>,
    free_list_pages: Vec<(PageNumber, Box<Page>)>,
    // Initially, set to the len of `bbn`. Each completion will decrement this.
    remaining: usize,
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

    fn fsync(&mut self) {
        unsafe {
            let fsync_res = libc::fsync(self.bbn_fd);
            assert!(fsync_res == 0);
        }
    }
}

struct LnWriteOut {
    ln_fd: RawFd,
    ln_extend_file_sz: Option<u64>,
    ln: Vec<(PageNumber, Box<Page>)>,
    ln_remaining: usize,
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

    fn fsync(&mut self) {
        unsafe {
            let fsync_res = libc::fsync(self.ln_fd);
            assert!(fsync_res == 0);
        }
    }
}

struct HtWriteOut {
    ht_fd: RawFd,
    ht: Vec<(u64, Box<Page>)>,
    ht_remaining: usize,
}

impl HtWriteOut {
    fn send_writes(&mut self, io: &mut IoDmux) {
        while self.ht.len() > 0 {
            // UNWRAP: we know the list is not empty.
            let (pn, page) = self.ht.pop().unwrap();

            {
                if let Err(TrySendError::Full(IoCommand {
                    kind: IoKind::Write(.., page),
                    ..
                })) = io.try_send_ht(IoKind::Write(self.ht_fd, pn, page))
                {
                    // That's alright. We will retry after trying to get something out of the cqueue
                    io.recv_ht_write();
                    self.ht_remaining = self.ht_remaining.checked_sub(1).unwrap();
                    self.ht.push((pn, page));
                }
            }
        }
    }

    fn wait_writes(&mut self, io: &mut IoDmux) {
        // wait for all writes
        while self.ht_remaining > 0 {
            io.recv_ht_write();
            self.ht_remaining = self.ht_remaining.checked_sub(1).unwrap();
        }
    }

    fn fsync(&mut self) {
        unsafe {
            let f = File::from_raw_fd(self.ht_fd);
            f.sync_all().expect("ht file: error performing fsync");
            let _ = f.into_raw_fd();
        }
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

            new_meta.encode_to(&mut page.as_mut()[..40]);

            if let Err(_) = io.try_send_meta(IoKind::Write(self.meta_fd, 0, page)) {
                self.new_meta = Some(new_meta);
            }
        }

        if self.should_fsync {
            unsafe {
                let f = File::from_raw_fd(self.meta_fd);
                f.sync_all().unwrap();
                let _ = f.into_raw_fd();
            }
            return true;
        }

        // Reap the completions.
        while let Some(CompleteIo { command, result }) = io.try_recv_meta() {
            assert!(result.is_ok());
            match command.kind {
                IoKind::Write(_, _, _) => {
                    self.should_fsync = true;
                    continue;
                }
                _ => panic!("unexpected completion kind"),
            }
        }

        false
    }
}
