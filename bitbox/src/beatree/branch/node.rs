use bitvec::prelude::*;
use std::sync::{Arc, Mutex};

use super::{BranchId, BranchNodePoolInner, BRANCH_NODE_SIZE};

// Here is the layout of a branch node:
//
// ```rust,ignore
// bbn_seqn: u64          // The sequence number of this BBN.
//                        // On disk, two nodes with the same seqn
//                        // will be considered the same, and
//                        // and the one with the latest valid
//                        // sync_seqn wins.
//
// sync_seqn: u32       // the sequence number of the commit under
//                        // which this node was created.
//                        // Important for BBNs only.
//
// bbn_pn: u32            // the page number this is stored under,
//                        // if this is a BBN, 0 otherwise.
//
// n: u16                 // item count
// prefix_len: u8         // bits
// separator_len: u8      // bits
//
// # Then the varbits follow. To avoid two padding bytes, the varbits are stored in a single bitvec.
//
// prefix: bitvec[prefix_len]
// separators: bitvec[(n + 1) * separator_len]
//
// # Node pointers follow. The list is aligned to the end of the node, with the last item in the
// # list occupying the last 4 bytes of the node.
//
// node_pointers: LNPN or BNID[n]
// ```

/// A branch node, regardless of its level.
pub struct BranchNode {
    pub(super) pool: Arc<Mutex<BranchNodePoolInner>>,
    pub(super) id: BranchId,
    pub(super) ptr: *mut (),
}

impl BranchNode {
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr as *const u8, BRANCH_NODE_SIZE) }
    }

    pub fn view(&self) -> BranchNodeView {
        BranchNodeView { inner: self.as_slice() }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut u8, BRANCH_NODE_SIZE) }
    }

    pub fn bbn_seqn(&self) -> u64 {
        self.view().bbn_seqn()
    }

    pub fn set_bbn_seqn(&mut self, seqn: u64) {
        let slice = self.as_mut_slice();
        slice[0..8].copy_from_slice(&seqn.to_le_bytes());
    }

    pub fn sync_seqn(&self) -> u32 {
        self.view().sync_seqn()
    }

    pub fn set_sync_seqn(&mut self, seqn: u32) {
        let slice = self.as_mut_slice();
        slice[8..12].copy_from_slice(&seqn.to_le_bytes());
    }

    pub fn bbn_pn(&self) -> u32 {
        self.view().bbn_pn()
    }

    pub fn set_bbn_pn(&mut self, pn: u32) {
        let slice = self.as_mut_slice();
        slice[12..16].copy_from_slice(&pn.to_le_bytes());
    }

    pub fn is_bbn(&self) -> bool {
        self.view().is_bbn()
    }

    pub fn n(&self) -> u16 {
        self.view().n()
    }

    pub fn set_n(&mut self, n: u16) {
        let slice = self.as_mut_slice();
        slice[16..18].copy_from_slice(&n.to_le_bytes());
    }

    pub fn prefix_len(&self) -> u8 {
        self.view().prefix_len()
    }

    pub fn set_prefix_len(&mut self, len: u8) {
        let slice = self.as_mut_slice();
        slice[18] = len;
    }

    pub fn separator_len(&self) -> u8 {
        self.view().separator_len()
    }

    pub fn set_separator_len(&mut self, len: u8) {
        let slice = self.as_mut_slice();
        slice[19] = len;
    }

    fn varbits(&self) -> &BitSlice<u8> {
        self.view().varbits()
    }

    fn varbits_mut(&mut self) -> &mut BitSlice<u8> {
        let bit_cnt =
            self.prefix_len() as usize + (self.separator_len() as usize) * (self.n() as usize + 1);
        self.as_mut_slice()[20..(20 + bit_cnt)].view_bits_mut()
    }

    pub fn prefix(&self) -> &BitSlice<u8> {
        self.view().prefix()
    }

    pub fn separator(&self, i: usize) -> &BitSlice<u8> {
        self.view().separator(i)
    }

    pub fn node_pointer(&self, i: usize) -> u32 {
        self.view().node_pointer(i)
    }

    // TODO: modification.
    //
    // Coming up with the right API for this is tricky:
    // - The offsets depend on `n`, `prefix_len` and `separator_len`, which suggests that it should
    //   be supplied all at once.
    // - At the same time, we want to avoid materializing the structure in memory.
    //
    // It all depends on how the caller wants to use this. Ideally, the caller would be able to
    // build the new node in a single pass.
}

impl Drop for BranchNode {
    fn drop(&mut self) {
        // Remove the node from the checked out list.
        let mut inner = self.pool.lock().unwrap();
        inner.checked_out.retain(|id| *id != self.id);
    }
}

pub struct BranchNodeView<'a> {
    inner: &'a [u8]
}

impl<'a> BranchNodeView<'a> {
    pub fn from_slice(slice: &'a [u8]) -> Self {
        assert_eq!(slice.len(), BRANCH_NODE_SIZE);
        BranchNodeView {
            inner: slice,
        }
    }

    pub fn bbn_seqn(&self) -> u64 {
        u64::from_le_bytes(self.inner[0..8].try_into().unwrap())
    }

    pub fn sync_seqn(&self) -> u32 {
        u32::from_le_bytes(self.inner[8..12].try_into().unwrap())
    }

    pub fn bbn_pn(&self) -> u32 {
        u32::from_le_bytes(self.inner[12..16].try_into().unwrap())
    }

    pub fn is_bbn(&self) -> bool {
        self.bbn_pn() == 0
    }

    pub fn n(&self) -> u16 {
        u16::from_le_bytes(self.inner[16..18].try_into().unwrap())
    }

    pub fn prefix_len(&self) -> u8 {
        self.inner[18]
    }

    pub fn separator_len(&self) -> u8 {
        self.inner[19]
    }

    fn varbits(&self) -> &'a BitSlice<u8> {
        let bit_cnt =
            self.prefix_len() as usize + (self.separator_len() as usize) * (self.n() as usize + 1);
        self.inner[20..(20 + bit_cnt)].view_bits()
    }

    pub fn prefix(&self) -> &'a BitSlice<u8> {
        &self.varbits()[..self.prefix_len() as usize]
    }

    pub fn separator(&self, i: usize) -> &'a BitSlice<u8> {
        let offset = self.prefix_len() as usize + (i + 1) * self.separator_len() as usize;
        &self.varbits()[offset..offset + self.separator_len() as usize]
    }

    pub fn node_pointer(&self, i: usize) -> u32 {
        let offset = BRANCH_NODE_SIZE - (self.n() as usize - i) * 4;
        u32::from_le_bytes(self.inner[offset..offset + 4].try_into().unwrap())
    }
}
