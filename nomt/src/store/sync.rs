use super::{
    meta::{self, Meta},
    MerkleTransaction, Shared, ValueTransaction,
};
use crate::{beatree, bitbox, merkle, page_cache::PageCache, rollback};

use crossbeam::channel::{self, Receiver};
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
            tp: ThreadPool::with_name("store-sync".into(), 6),
            sync_seqn,
            bitbox_num_pages,
            bitbox_seed,
            panic_on_sync,
        }
    }

    pub fn sync(
        &mut self,
        shared: &Shared,
        value_tx: ValueTransaction,
        bitbox: bitbox::DB,
        beatree: beatree::Tree,
        rollback: Option<rollback::Rollback>,
        page_cache: PageCache,
        page_diffs: merkle::PageDiffs,
    ) -> anyhow::Result<()> {
        self.sync_seqn += 1;
        let sync_seqn = self.sync_seqn;

        let rollback_writeout_wd_rx = spawn_rollback_writeout_start(&self.tp, &rollback);

        let mut bitbox_sync = bitbox.sync();
        let mut beatree_sync = beatree.sync();

        let merkle_tx = MerkleTransaction {
            page_pool: shared.page_pool.clone(),
            bucket_allocator: bitbox.bucket_allocator(),
            new_pages: Vec::new(),
        };
        bitbox_sync.begin_sync(page_cache, merkle_tx, page_diffs);
        beatree_sync.begin_sync(value_tx.batch);

        let rollback_writeout_wd = rollback_writeout_wd_rx
            .map(|rollback_writeout_wd| rollback_writeout_wd.recv().unwrap());

        let rollback_start_live;
        let rollback_end_live;
        let rollback_prune_to_new_start_live;
        let rollback_prune_to_new_end_live;
        if let Some(rollback_writeout_wd) = &rollback_writeout_wd {
            rollback_start_live = rollback_writeout_wd.rollback_start_live;
            rollback_end_live = rollback_writeout_wd.rollback_end_live;
            rollback_prune_to_new_start_live = rollback_writeout_wd.prune_to_new_start_live;
            rollback_prune_to_new_end_live = rollback_writeout_wd.prune_to_new_end_live;
        } else {
            rollback_start_live = 0;
            rollback_end_live = 0;
            rollback_prune_to_new_start_live = None;
            rollback_prune_to_new_end_live = None;
        }

        // TODO: comprehensive error handling is coming later.
        bitbox_sync.wait_pre_meta().unwrap();
        let beatree_meta_wd = beatree_sync.wait_pre_meta().unwrap();

        let new_meta = Meta {
            magic: meta::MAGIC,
            version: meta::VERSION,
            ln_freelist_pn: beatree_meta_wd.ln_freelist_pn,
            ln_bump: beatree_meta_wd.ln_bump,
            bbn_freelist_pn: beatree_meta_wd.bbn_freelist_pn,
            bbn_bump: beatree_meta_wd.bbn_bump,
            sync_seqn,
            bitbox_num_pages: self.bitbox_num_pages,
            bitbox_seed: self.bitbox_seed,
            rollback_start_live,
            rollback_end_live,
        };
        Meta::write(&shared.io_pool.page_pool(), &shared.meta_fd, &new_meta)?;

        if self.panic_on_sync {
            panic!("panic_on_sync is true");
        }

        // Spawn a task to finish off the rollback writeout, if required.
        let rollback_writeout_end_rx = if let Some(rollback) = rollback {
            spawn_rollback_writeout_end(
                &self.tp,
                &rollback,
                rollback_prune_to_new_start_live,
                rollback_prune_to_new_end_live,
            )
        } else {
            let (tx, rx) = channel::bounded(1);
            tx.send(()).unwrap();
            rx
        };

        bitbox_sync.post_meta(shared.io_pool.make_handle())?;
        beatree_sync.post_meta();

        rollback_writeout_end_rx.recv().unwrap();

        Ok(())
    }
}

fn spawn_rollback_writeout_start(
    tp: &ThreadPool,
    rollback: &Option<rollback::Rollback>,
) -> Option<Receiver<rollback::WriteoutData>> {
    match rollback {
        None => None,
        Some(rollback) => {
            let (result_tx, result_rx) = channel::bounded(1);
            let rollback = rollback.clone();
            tp.execute(move || {
                let writeout_data = rollback.writeout_start().unwrap();
                let _ = result_tx.send(writeout_data);
            });
            Some(result_rx)
        }
    }
}

fn spawn_rollback_writeout_end(
    tp: &ThreadPool,
    rollback: &rollback::Rollback,
    new_start_live: Option<u64>,
    new_end_live: Option<u64>,
) -> Receiver<()> {
    let (result_tx, result_rx) = channel::bounded(1);
    let rollback = rollback.clone();
    tp.execute(move || {
        rollback.writeout_end(new_start_live, new_end_live).unwrap();
        let _ = result_tx.send(());
    });
    result_rx
}
