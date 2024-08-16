/// The utility functions for handling the metadata file.
use anyhow::Result;
use std::{fs::File, io::Read as _};

use crate::io::Page;

/// This data structure describes the state of the btree.
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
}

impl Meta {
    pub fn encode_to(&self, buf: &mut [u8]) {
        assert_eq!(buf.len(), 24);
        buf[0..4].copy_from_slice(&self.ln_freelist_pn.to_le_bytes());
        buf[4..8].copy_from_slice(&self.ln_bump.to_le_bytes());
        buf[8..12].copy_from_slice(&self.bbn_freelist_pn.to_le_bytes());
        buf[12..16].copy_from_slice(&self.bbn_bump.to_le_bytes());
        buf[16..20].copy_from_slice(&self.sync_seqn.to_le_bytes());
        buf[20..24].copy_from_slice(&self.bitbox_num_pages.to_le_bytes());
    }

    pub fn decode(buf: &[u8]) -> Self {
        let ln_freelist_pn = u32::from_le_bytes(buf[0..4].try_into().unwrap());
        let ln_bump = u32::from_le_bytes(buf[4..8].try_into().unwrap());
        let bbn_freelist_pn = u32::from_le_bytes(buf[8..12].try_into().unwrap());
        let bbn_bump = u32::from_le_bytes(buf[12..16].try_into().unwrap());
        let sync_seqn = u32::from_le_bytes(buf[16..20].try_into().unwrap());
        let bitbox_num_pages = u32::from_le_bytes(buf[20..24].try_into().unwrap());
        Self {
            ln_freelist_pn,
            ln_bump,
            bbn_freelist_pn,
            bbn_bump,
            sync_seqn,
            bitbox_num_pages,
        }
    }

    pub fn read(mut fd: &File) -> Result<Self> {
        // The fd is expected to be opened with O_DIRECT and thus we have to satisfy the alignment
        // requirements for the buffer. Use a Page for that.
        let mut buf = Page::zeroed();
        fd.read_exact(&mut buf)?;
        Ok(Self::decode(&buf))
    }
}
