/// This data structure describes the state of the btree.
pub struct Meta {
    /// The sequence number of the last commit.
    pub commit_seqn: u64,
    /// The next allocated BBN will have this sequence number.
    pub next_bbn_seqn: u32,
    /// The page number of the head of the freelist of the leaf storage file. 0 means the freelist
    /// is empty.
    pub ln_freelist_pn: u32,
    /// The number of pages in the LN storage file. Since the first page is reserved, this is always
    /// more than 1.
    pub ln_sz: u32,
    /// The number of pages in the BBN storage file. Since the first page is reserved, this is
    /// always more than 1.
    pub bbn_sz: u32,
}

impl Meta {
    pub fn encode_to(&self, buf: &mut [u8; 24]) {
        buf[..8].copy_from_slice(&self.commit_seqn.to_le_bytes());
        buf[8..12].copy_from_slice(&self.next_bbn_seqn.to_le_bytes());
        buf[12..16].copy_from_slice(&self.ln_freelist_pn.to_le_bytes());
        buf[16..20].copy_from_slice(&self.ln_sz.to_le_bytes());
        buf[20..24].copy_from_slice(&self.bbn_sz.to_le_bytes());
    }

    pub fn decode(buf: &[u8]) -> Self {
        let commit_seqn = u64::from_le_bytes(buf[..8].try_into().unwrap());
        let next_bbn_seqn = u32::from_le_bytes(buf[8..12].try_into().unwrap());
        let ln_freelist_pn = u32::from_le_bytes(buf[12..16].try_into().unwrap());
        let ln_sz = u32::from_le_bytes(buf[16..20].try_into().unwrap());
        let bbn_sz = u32::from_le_bytes(buf[20..24].try_into().unwrap());
        Self {
            commit_seqn,
            next_bbn_seqn,
            ln_freelist_pn,
            ln_sz,
            bbn_sz,
        }
    }
}
