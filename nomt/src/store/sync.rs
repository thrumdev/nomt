use super::{
    meta::{self, Meta},
    MerkleTransaction, Shared, ValueTransaction,
};
use crate::{beatree, bitbox, merkle, page_cache::PageCache, rollback};

pub struct Sync {
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

        let mut bitbox_sync = bitbox.sync();
        let mut beatree_sync = beatree.sync();
        let mut rollback_sync = rollback.map(|rollback| rollback.sync());

        let merkle_tx = MerkleTransaction {
            page_pool: shared.page_pool.clone(),
            bucket_allocator: bitbox.bucket_allocator(),
            new_pages: Vec::new(),
        };
        bitbox_sync.begin_sync(page_cache, merkle_tx, page_diffs);
        beatree_sync.begin_sync(value_tx.batch);
        if let Some(ref mut rollback) = rollback_sync {
            rollback.begin_sync();
        }

        // TODO: comprehensive error handling is coming later.
        bitbox_sync.wait_pre_meta().unwrap();
        let beatree_meta_wd = beatree_sync.wait_pre_meta().unwrap();
        let (rollback_start_live, rollback_end_live) = match rollback_sync {
            Some(ref rollback) => rollback.wait_pre_meta(),
            None => (0, 0),
        };

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

        if let Some(ref rollback) = rollback_sync {
            rollback.post_meta();
        }

        bitbox_sync.post_meta(shared.io_pool.make_handle())?;
        beatree_sync.post_meta();

        if let Some(ref rollback) = rollback_sync {
            // TODO: comprehensive error handling is coming later.
            rollback.wait_post_meta().unwrap();
        }

        Ok(())
    }
}
