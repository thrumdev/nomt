use std::{
    fs::File,
    io::{Seek, SeekFrom},
    os::fd::{FromRawFd, IntoRawFd, RawFd},
};

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
            meta_swap: Some(MetaSwap { meta_fd, new_meta }),
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

    // At this point, the BBN and LN files are fully synced. We can now update the meta file.
    if let Some(ref mut meta_swap) = cx.meta_swap {
        meta_swap.run(&mut io);
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

    fn recv_bbn(&mut self) -> Option<CompleteIo> {
        match self.bbn_inbox.pop() {
            Some(io) => Some(io),
            None => self.recv(Self::BBN_USER_DATA),
        }
    }

    fn recv_wal(&mut self) -> Option<CompleteIo> {
        match self.wal_inbox.pop() {
            Some(io) => Some(io),
            None => self.recv(Self::WAL_USER_DATA),
        }
    }

    fn recv_ln(&mut self) -> Option<CompleteIo> {
        match self.ln_inbox.pop() {
            Some(io) => Some(io),
            None => self.recv(Self::LN_USER_DATA),
        }
    }

    fn recv_ht(&mut self) -> Option<CompleteIo> {
        match self.ht_inbox.pop() {
            Some(io) => Some(io),
            None => self.recv(Self::HT_USER_DATA),
        }
    }

    fn recv_meta(&mut self) -> Option<CompleteIo> {
        match self.meta_inbox.pop() {
            Some(io) => Some(io),
            None => self.recv(Self::META_USER_DATA),
        }
    }

    fn recv_wal_write(&mut self) {
        loop {
            if let Some(CompleteIo { command, result }) = self.recv_wal() {
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
            if let Some(CompleteIo { command, result }) = self.recv_bbn() {
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
            if let Some(CompleteIo { command, result }) = self.recv_ln() {
                assert!(result.is_ok());
                match command.kind {
                    IoKind::Write(_, _, _) => {}
                    _ => panic!("unexpected completion kind, expected: Write or WriteRaw"),
                }
                break;
            }
        }
    }

    fn recv_ht_write(&mut self) {
        loop {
            if let Some(CompleteIo { command, result }) = self.recv_ht() {
                assert!(result.is_ok());
                match command.kind {
                    IoKind::Write(_, _, _) => {}
                    _ => panic!("unexpected completion kind, expected: Write"),
                }
                break;
            }
        }
    }

    fn send_wal(&mut self, kind: IoKind) {
        self.io_handle
            .send(IoCommand {
                kind,
                user_data: Self::WAL_USER_DATA,
            })
            .expect("I/O pool down");
    }

    fn send_bbn(&mut self, kind: IoKind) {
        self.io_handle
            .send(IoCommand {
                kind,
                user_data: Self::BBN_USER_DATA,
            })
            .expect("I/O pool down");
    }

    fn send_ln(&mut self, kind: IoKind) {
        self.io_handle
            .send(IoCommand {
                kind,
                user_data: Self::LN_USER_DATA,
            })
            .expect("I/O pool down");
    }

    fn send_ht(&self, kind: IoKind) {
        self.io_handle
            .send(IoCommand {
                kind,
                user_data: Self::HT_USER_DATA,
            })
            .expect("I/O pool down");
    }

    fn send_meta(&self, kind: IoKind) {
        self.io_handle
            .send(IoCommand {
                kind,
                user_data: Self::META_USER_DATA,
            })
            .expect("I/O pool down");
    }

    /// Receive a completion from the io channel. If the completion is for the given user
    /// data, it is immediately returned. Otherwise, it is pushed into the appropriate inbox.
    fn recv(&mut self, eager_ud: u64) -> Option<CompleteIo> {
        match self.io_handle.recv() {
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
                // For future, we should bubble this error up, rather than panicking.
                panic!("I/O pool down");
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
        for pn in self.wal_blob_pns.drain(..) {
            let len = PAGE_SIZE;
            unsafe {
                let ptr = self.wal_blob.0.add(pn as usize * PAGE_SIZE);
                io.send_wal(IoKind::WriteRaw(self.wal_fd, pn as u64, ptr, len))
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
        for mut branch_node in self.bbn.drain(..) {
            // UNWRAP: the branch node cannot be out of bounds because of the requirement of the
            // sync machine.
            let wrt = branch_node.as_mut_slice();
            let (ptr, len) = (wrt.as_ptr(), wrt.len());
            let bbn_pn = branch_node.bbn_pn();

            // SAFETY: BBNs are kept alive by the branch node pool. It is safe to drop the view
            // into the buffer without invalidating the pointer.
            io.send_bbn(IoKind::WriteRaw(self.bbn_fd, bbn_pn as u64, ptr, len));
        }

        for (pn, page) in self.free_list_pages.drain(..) {
            io.send_bbn(IoKind::Write(self.bbn_fd, pn.0 as u64, page));
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
        for (ln_pn, ln_page) in self.ln.drain(..) {
            io.send_ln(IoKind::Write(self.ln_fd, ln_pn.0 as u64, ln_page));
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
        for (pn, page) in self.ht.drain(..) {
            io.send_ht(IoKind::Write(self.ht_fd, pn, page))
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
    new_meta: Meta,
}

impl MetaSwap {
    fn run(&mut self, io: &mut IoDmux) {
        // Oh god, there is a special place in hell for this. Will do for now though.
        let mut page = Box::new(Page::zeroed());
        self.new_meta.encode_to(&mut page.as_mut()[..40]);

        io.send_meta(IoKind::Write(self.meta_fd, 0, page));

        // Reap the completions.
        loop {
            if let Some(CompleteIo { command, result }) = io.recv_meta() {
                assert!(result.is_ok());
                match command.kind {
                    IoKind::Write(_, _, _) => break,
                    _ => panic!("unexpected completion kind"),
                }
            }
        }

        unsafe {
            let f = File::from_raw_fd(self.meta_fd);
            f.sync_all().unwrap();
            let _ = f.into_raw_fd();
        }
    }
}
