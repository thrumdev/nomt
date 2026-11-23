use nomt_core::page_id::PageId;

use super::{
    meta::{self, Meta},
    value_log, DirtyPage, Shared,
};
use crate::{
    beatree, bitbox, options::PanicOnSyncMode, page_cache::PageCache, rollback, task::join_task,
};

pub struct Sync {
    pub(crate) sync_seqn: u32,
    pub(crate) bitbox_num_pages: u32,
    pub(crate) bitbox_seed: [u8; 16],
    pub(crate) panic_on_sync: Option<PanicOnSyncMode>,
    /// The beatree state recorded in the last manifest.
    pub(crate) beatree: beatree::SyncData,
}

impl Sync {
    pub fn new(
        sync_seqn: u32,
        bitbox_num_pages: u32,
        bitbox_seed: [u8; 16],
        panic_on_sync: Option<PanicOnSyncMode>,
        beatree: beatree::SyncData,
    ) -> Self {
        Self {
            sync_seqn,
            bitbox_num_pages,
            bitbox_seed,
            panic_on_sync,
            beatree,
        }
    }

    /// Commits the changes.
    ///
    /// The manifest is written as soon as the hash-table WAL, the rollback delta and the value log
    /// are durable, without waiting for the beatree pages. It records the previous beatree state
    /// and marks the value log as pending. The hash-table pages are then written while the beatree
    /// update finishes, and a second manifest write records the new beatree state.
    pub fn sync(
        &mut self,
        shared: &Shared,
        value_tx: impl IntoIterator<Item = (beatree::Key, beatree::ValueChange)> + Send + 'static,
        bitbox: bitbox::DB,
        beatree: beatree::Tree,
        rollback: Option<rollback::Rollback>,
        rollback_delta: Option<rollback::Delta>,
        page_cache: PageCache,
        updated_pages: impl IntoIterator<Item = (PageId, DirtyPage)> + Send + 'static,
    ) -> anyhow::Result<()> {
        let sync_seqn = self.sync_seqn + 1;

        let mut bitbox_sync = bitbox.sync();
        let mut beatree_sync = beatree.sync();
        let mut rollback_sync = rollback.as_ref().map(|rollback| rollback.sync());

        bitbox_sync.begin_sync(sync_seqn, page_cache, updated_pages);

        let changeset: Vec<_> = value_tx.into_iter().collect();
        let value_log = value_log::encode(sync_seqn, &changeset);
        beatree_sync.begin_sync(changeset);

        std::thread::scope(|s| -> anyhow::Result<()> {
            let value_log_write = s.spawn(|| value_log::write(&shared.value_log_fd, &value_log));

            // Persist the rollback delta while the trees are being prepared. All of them must
            // succeed before the manifest references their new state.
            let rollback_result = match rollback_delta {
                // UNWRAP: a rollback delta is only produced when rollback is enabled.
                Some(delta) => rollback.as_ref().unwrap().commit(delta),
                None => Ok(()),
            };
            let bitbox_result = bitbox_sync.wait_pre_meta();
            // UNWRAP: the writer does not panic.
            let value_log_result = value_log_write.join().unwrap();
            rollback_result?;
            bitbox_result?;
            value_log_result?;

            let (rollback_start_live, rollback_end_live) = match rollback_sync {
                Some(ref mut rollback) => rollback.begin_sync(),
                None => (0, 0),
            };

            if let Some(PanicOnSyncMode::PostWal) = self.panic_on_sync {
                panic!("panic_on_sync is true (post-wal)")
            }

            let mut new_meta = Meta {
                magic: meta::MAGIC,
                version: meta::VERSION,
                ln_freelist_pn: self.beatree.ln_freelist_pn,
                ln_bump: self.beatree.ln_bump,
                bbn_freelist_pn: self.beatree.bbn_freelist_pn,
                bbn_bump: self.beatree.bbn_bump,
                sync_seqn,
                bitbox_num_pages: self.bitbox_num_pages,
                bitbox_seed: self.bitbox_seed,
                rollback_start_live,
                rollback_end_live,
            };
            let page_pool = shared.io_pool.page_pool();
            Meta::write_with_pending(&page_pool, &shared.meta_fd, &new_meta, sync_seqn)?;
            self.sync_seqn += 1;

            if let Some(ref mut rollback) = rollback_sync {
                rollback.post_meta();
            }

            // Write out the hash-table pages while the beatree update completes.
            let ht_writeout = s.spawn(|| bitbox_sync.post_meta(shared.io_pool.make_handle()));

            let beatree_result = beatree_sync.wait_pre_meta();
            if let Some(PanicOnSyncMode::PostMeta) = self.panic_on_sync {
                panic!("panic_on_sync is true (post-meta)");
            }
            let beatree_meta = beatree_result?;
            new_meta.ln_freelist_pn = beatree_meta.ln_freelist_pn;
            new_meta.ln_bump = beatree_meta.ln_bump;
            new_meta.bbn_freelist_pn = beatree_meta.bbn_freelist_pn;
            new_meta.bbn_bump = beatree_meta.bbn_bump;
            Meta::write(&page_pool, &shared.meta_fd, &new_meta)?;
            self.beatree = beatree_meta;

            join_task(&beatree_sync.post_meta());
            // UNWRAP: the writeout does not panic.
            ht_writeout.join().unwrap()?;

            if let Some(ref rollback) = rollback_sync {
                rollback.wait_post_meta()?;
            }
            Ok(())
        })
    }
}
