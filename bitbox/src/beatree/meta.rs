/// This data structure describes the state of the btree.
pub struct Meta {
    /// The sequence number of the last commit.
    pub commit_seqn: u64,
    /// The next BBN sequence number.
    pub next_bbn_seqn: u32,
    /// The page number of the head of the freelist of the leaf storage file. 0 means the freelist
    /// is empty.
    pub ln_freelist_pn: u32,
    /// The number of pages in the LN storage file.
    pub ln_sz: u32,
    /// The number of pages in the BBN storage file.
    pub bbn_sz: u32,
}
