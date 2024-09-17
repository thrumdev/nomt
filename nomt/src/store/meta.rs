/// The utility functions for handling the metadata file.
use anyhow::Result;
use std::fs::File;
use std::os::unix::fs::FileExt as _;

use crate::io::{self, PagePool};

/// This data structure describes the state of the btree.
#[derive(Clone)]
pub struct Meta {
    /// The page number of the head of the freelist of the leaf storage file. 0 means the freelist
    /// is empty.
    pub ln_freelist_pn: u32,
    /// The next page available for allocation in the LN storage file.
    ///
    /// Since the first page is reserved, this is always more than 1.
    pub ln_bump: u32,
    /// The page number of the head of the freelist of the bbn storage file. 0 means the freelist
    /// is empty.
    pub bbn_freelist_pn: u32,
    /// The next page available for allocation in the BBN storage file.
    ///
    /// Since the first page is reserved, this is always more than 1.
    pub bbn_bump: u32,
    /// The sequence number of the last sync.
    ///
    /// 0 means there were no syncs and the DB is empty.
    pub sync_seqn: u32,
    /// The number of pages in the bitbox store.
    pub bitbox_num_pages: u32,
    /// The random seed used for populating the hash-table in a unique way.
    pub bitbox_seed: [u8; 16],
    /// The first live record ID in the rollback seglog.
    pub rollback_start_live: u64,
    /// The last live record ID in the rollback seglog.
    pub rollback_end_live: u64,
}

impl Meta {
    pub fn encode_to(&self, buf: &mut [u8]) {
        assert_eq!(buf.len(), 56);
        buf[0..4].copy_from_slice(&self.ln_freelist_pn.to_le_bytes());
        buf[4..8].copy_from_slice(&self.ln_bump.to_le_bytes());
        buf[8..12].copy_from_slice(&self.bbn_freelist_pn.to_le_bytes());
        buf[12..16].copy_from_slice(&self.bbn_bump.to_le_bytes());
        buf[16..20].copy_from_slice(&self.sync_seqn.to_le_bytes());
        buf[20..24].copy_from_slice(&self.bitbox_num_pages.to_le_bytes());
        buf[24..40].copy_from_slice(&self.bitbox_seed);
        buf[40..48].copy_from_slice(&self.rollback_start_live.to_le_bytes());
        buf[48..56].copy_from_slice(&self.rollback_end_live.to_le_bytes());
    }

    pub fn decode(buf: &[u8]) -> Self {
        let ln_freelist_pn = u32::from_le_bytes(buf[0..4].try_into().unwrap());
        let ln_bump = u32::from_le_bytes(buf[4..8].try_into().unwrap());
        let bbn_freelist_pn = u32::from_le_bytes(buf[8..12].try_into().unwrap());
        let bbn_bump = u32::from_le_bytes(buf[12..16].try_into().unwrap());
        let sync_seqn = u32::from_le_bytes(buf[16..20].try_into().unwrap());
        let bitbox_num_pages = u32::from_le_bytes(buf[20..24].try_into().unwrap());
        let bitbox_seed = buf[24..40].try_into().unwrap();
        let rollback_start_live = u64::from_le_bytes(buf[40..48].try_into().unwrap());
        let rollback_end_live = u64::from_le_bytes(buf[48..56].try_into().unwrap());
        Self {
            ln_freelist_pn,
            ln_bump,
            bbn_freelist_pn,
            bbn_bump,
            sync_seqn,
            bitbox_num_pages,
            bitbox_seed,
            rollback_start_live,
            rollback_end_live,
        }
    }

    pub fn validate(&self) -> Result<()> {
        let mut errors = Vec::new();
        let is_rollback_start_live_nil = self.rollback_start_live == 0;
        let is_rollback_end_live_nil = self.rollback_end_live == 0;
        if is_rollback_start_live_nil ^ is_rollback_end_live_nil {
            errors.push(format!(
                "rollback_start_live and rollback_end_live must both be nil or both be non-nil, got start: {}, end: {}",
                self.rollback_start_live,
                self.rollback_end_live,
            ));
        }

        if errors.is_empty() {
            Ok(())
        } else {
            // Collect all the errors and return them in a single anyhow error.
            Err(anyhow::anyhow!(errors.join("\n")))
        }
    }

    pub fn read(page_pool: &PagePool, fd: &File) -> Result<Self> {
        let page = io::read_page(page_pool, fd, 0)?;
        let meta = Meta::decode(&page[..56]);
        Ok(meta)
    }

    pub fn write(page_pool: &PagePool, fd: &File, meta: &Meta) -> Result<()> {
        let mut page = page_pool.alloc_fat_page();
        meta.encode_to(&mut page.as_mut()[..56]);
        fd.write_all_at(&page[..], 0)?;
        fd.sync_all()?;
        Ok(())
    }
}
