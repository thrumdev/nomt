use super::{meta::Meta, Shared, Transaction};
use crate::{
    beatree::{self, allocator::PageNumber, branch::BranchNode},
    bitbox,
    io::{FatPage, IoPool, PagePool},
};

use crossbeam::channel::{self, Receiver};
use std::{fs::File, mem, sync::Arc};
use threadpool::ThreadPool;

pub struct Sync {
    pub(crate) tp: ThreadPool,
    pub(crate) sync_seqn: u32,
    pub(crate) bitbox_num_pages: u32,
    pub(crate) bitbox_seed: [u8; 16],
    pub(crate) panic_on_sync: bool,
}

impl Sync {
    pub fn new(
        sync_seqn: u32,
        bitbox_num_pages: u32,
        bitbox_seed: [u8; 16],
        panic_on_sync: bool,
    ) -> Self {
        Self {
            tp: ThreadPool::with_name("store-sync".into(), 5),
            sync_seqn,
            bitbox_num_pages,
            bitbox_seed,
            panic_on_sync,
        }
    }

    pub fn sync(
        &mut self,
        shared: &Shared,
        mut tx: Transaction,
        bitbox: bitbox::DB,
        beatree: beatree::Tree,
    ) -> anyhow::Result<()> {
        self.sync_seqn += 1;
        let sync_seqn = self.sync_seqn;

        let (bitbox_ht_wd, bitbox_wal_wd) = spawn_prepare_sync_bitbox(
            &self.tp,
            shared.io_pool.page_pool().clone(),
            &mut tx,
            bitbox,
        );
        let (beatree_bbn_wd, beatree_ln_wd, meta_wd) =
            spawn_prepare_sync_beatree(&self.tp, &mut tx, beatree.clone());

        let (bbn_writeout_done, ln_writeout_done) = spawn_bbn_ln_writeout(
            &self.tp,
            &shared.io_pool,
            &shared.bbn_fd,
            &shared.ln_fd,
            beatree_bbn_wd,
            beatree_ln_wd,
        );
        let bitbox_writeout_done = spawn_wal_writeout(&self.tp, &shared.wal_fd, bitbox_wal_wd);

        bbn_writeout_done.recv().unwrap();
        ln_writeout_done.recv().unwrap();
        bitbox_writeout_done.recv().unwrap();

        let beatree_meta_wd = meta_wd.recv().unwrap();

        let new_meta = Meta {
            ln_freelist_pn: beatree_meta_wd.ln_freelist_pn,
            ln_bump: beatree_meta_wd.ln_bump,
            bbn_freelist_pn: beatree_meta_wd.bbn_freelist_pn,
            bbn_bump: beatree_meta_wd.bbn_bump,
            sync_seqn,
            bitbox_num_pages: self.bitbox_num_pages,
            bitbox_seed: self.bitbox_seed,
        };
        Meta::write(&shared.io_pool.page_pool(), &shared.meta_fd, &new_meta)?;

        if self.panic_on_sync {
            panic!("panic_on_sync is true");
        }

        let HtWriteoutData { ht_pages } = bitbox_ht_wd.recv().unwrap();
        bitbox::writeout::write_ht(shared.io_pool.make_handle(), &shared.ht_fd, ht_pages)?;
        bitbox::writeout::truncate_wal(&shared.wal_fd)?;

        beatree.finish_sync(beatree_meta_wd.bbn_index);

        Ok(())
    }
}

struct WalWriteoutData {
    wal_blob: (*mut u8, usize),
}
unsafe impl Send for WalWriteoutData {}

struct HtWriteoutData {
    ht_pages: Vec<(u64, FatPage)>,
}

fn spawn_prepare_sync_bitbox(
    tp: &ThreadPool,
    page_pool: PagePool,
    tx: &mut Transaction,
    bitbox: bitbox::DB,
) -> (Receiver<HtWriteoutData>, Receiver<WalWriteoutData>) {
    let (ht_result_tx, ht_result_rx) = channel::bounded(1);
    let (wal_result_tx, wal_result_rx) = channel::bounded(1);
    let new_pages = mem::take(&mut tx.new_pages);
    tp.execute(move || {
        let bitbox::WriteoutData { ht_pages, wal_blob } = bitbox
            .prepare_sync(&page_pool, new_pages)
            // TODO: handle error.
            .unwrap();
        let _ = ht_result_tx.send(HtWriteoutData { ht_pages });
        let _ = wal_result_tx.send(WalWriteoutData { wal_blob });
    });
    (ht_result_rx, wal_result_rx)
}

struct BbnWriteoutData {
    bbn: Vec<Arc<BranchNode>>,
    bbn_freelist_pages: Vec<(PageNumber, FatPage)>,
    bbn_extend_file_sz: Option<u64>,
}

struct LnWriteoutData {
    ln: Vec<(PageNumber, FatPage)>,
    ln_freelist_pages: Vec<(PageNumber, FatPage)>,
    ln_extend_file_sz: Option<u64>,
}

struct BeatreePostWriteout {
    ln_freelist_pn: u32,
    ln_bump: u32,
    bbn_freelist_pn: u32,
    bbn_bump: u32,
    bbn_index: beatree::Index,
}

fn spawn_prepare_sync_beatree(
    tp: &ThreadPool,
    tx: &mut Transaction,
    beatree: beatree::Tree,
) -> (
    Receiver<BbnWriteoutData>,
    Receiver<LnWriteoutData>,
    Receiver<BeatreePostWriteout>,
) {
    let batch = mem::take(&mut tx.batch);
    let (bbn_result_tx, bbn_result_rx) = channel::bounded(1);
    let (ln_result_tx, ln_result_rx) = channel::bounded(1);
    let (meta_result_tx, meta_result_rx) = channel::bounded(1);
    let tp = tp.clone();
    tp.execute(move || {
        beatree.commit(batch);
        let beatree::WriteoutData {
            bbn,
            bbn_freelist_pages,
            bbn_extend_file_sz,
            ln,
            ln_freelist_pages,
            ln_extend_file_sz,
            ln_freelist_pn,
            ln_bump,
            bbn_freelist_pn,
            bbn_bump,
            bbn_index,
        } = beatree.prepare_sync();

        let _ = bbn_result_tx.send(BbnWriteoutData {
            bbn,
            bbn_freelist_pages,
            bbn_extend_file_sz,
        });
        let _ = ln_result_tx.send(LnWriteoutData {
            ln,
            ln_freelist_pages,
            ln_extend_file_sz,
        });
        let _ = meta_result_tx.send(BeatreePostWriteout {
            ln_freelist_pn,
            ln_bump,
            bbn_freelist_pn,
            bbn_bump,
            bbn_index,
        });
    });
    (bbn_result_rx, ln_result_rx, meta_result_rx)
}

fn spawn_bbn_ln_writeout(
    tp: &ThreadPool,
    io_pool: &IoPool,
    bbn_fd: &File,
    ln_fd: &File,
    beatree_bbn_wd: Receiver<BbnWriteoutData>,
    beatree_ln_wd: Receiver<LnWriteoutData>,
) -> (Receiver<()>, Receiver<()>) {
    let (bbn_result_tx, bbn_result_rx) = channel::bounded(1);
    tp.execute({
        let io_handle = io_pool.make_handle();
        let bbn_fd = bbn_fd.try_clone().unwrap();
        move || {
            let BbnWriteoutData {
                bbn,
                bbn_freelist_pages,
                bbn_extend_file_sz,
            } = beatree_bbn_wd.recv().unwrap();
            beatree::writeout::write_bbn(
                io_handle,
                &bbn_fd,
                bbn,
                bbn_freelist_pages,
                bbn_extend_file_sz,
            )
            .unwrap();
            let _ = bbn_result_tx.send(());
        }
    });

    let (ln_result_tx, ln_result_rx) = channel::bounded(1);
    tp.execute({
        let io_handle = io_pool.make_handle();
        let ln_fd = ln_fd.try_clone().unwrap();
        move || {
            let LnWriteoutData {
                ln,
                ln_freelist_pages,
                ln_extend_file_sz,
            } = beatree_ln_wd.recv().unwrap();
            beatree::writeout::write_ln(
                io_handle,
                &ln_fd,
                ln,
                ln_freelist_pages,
                ln_extend_file_sz,
            )
            .unwrap();
            let _ = ln_result_tx.send(());
        }
    });
    (bbn_result_rx, ln_result_rx)
}

fn spawn_wal_writeout(
    tp: &ThreadPool,
    wal_fd: &File,
    wal_wd: Receiver<WalWriteoutData>,
) -> Receiver<()> {
    let (result_tx, result_rx) = channel::bounded(1);
    let mut wal_fd = wal_fd.try_clone().unwrap();
    tp.execute({
        let WalWriteoutData { wal_blob } = wal_wd.recv().unwrap();
        let (data, len) = wal_blob;
        let wal_blob = unsafe { std::slice::from_raw_parts(data, len) };
        move || {
            bitbox::writeout::write_wal(&mut wal_fd, wal_blob).unwrap();
            let _ = result_tx.send(());
        }
    });
    result_rx
}
