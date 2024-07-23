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
}

impl Meta {
    pub fn encode_to(&self, buf: &mut [u8; 20]) {
        buf[0..4].copy_from_slice(&self.ln_freelist_pn.to_le_bytes());
        buf[4..8].copy_from_slice(&self.ln_bump.to_le_bytes());
        buf[8..12].copy_from_slice(&self.bbn_freelist_pn.to_le_bytes());
        buf[12..16].copy_from_slice(&self.bbn_bump.to_le_bytes());
    }

    pub fn decode(buf: &[u8]) -> Self {
        let ln_freelist_pn = u32::from_le_bytes(buf[0..4].try_into().unwrap());
        let ln_bump = u32::from_le_bytes(buf[4..8].try_into().unwrap());
        let bbn_freelist_pn = u32::from_le_bytes(buf[8..12].try_into().unwrap());
        let bbn_bump = u32::from_le_bytes(buf[12..16].try_into().unwrap());
        Self {
            ln_freelist_pn,
            ln_bump,
            bbn_freelist_pn,
            bbn_bump,
        }
    }
}
