use bitvec::prelude::*;
use std::sync::{Arc, Mutex};

use super::{BranchId, BranchNodePoolInner, BRANCH_NODE_SIZE};
use crate::beatree::Key;

// Here is the layout of a branch node:
//
// ```rust,ignore
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
// separators: bitvec[n * separator_len]
//
// # Node pointers follow. The list is aligned to the end of the node, with the last item in the
// # list occupying the last 4 bytes of the node.
//
// node_pointers: LNPN or BNID[n]
// ```

pub const BRANCH_NODE_BODY_SIZE: usize = BRANCH_NODE_SIZE - (4 + 2 + 1 + 1);

/// A branch node, regardless of its level.
pub struct BranchNode {
    pub(super) pool: Arc<Mutex<BranchNodePoolInner>>,
    pub(super) id: BranchId,
    pub(super) ptr: *mut u8,
}

impl BranchNode {
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr as *const u8, BRANCH_NODE_SIZE) }
    }

    pub fn view(&self) -> BranchNodeView {
        BranchNodeView {
            inner: self.as_slice(),
        }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut u8, BRANCH_NODE_SIZE) }
    }

    pub fn bbn_pn(&self) -> u32 {
        self.view().bbn_pn()
    }

    pub fn set_bbn_pn(&mut self, pn: u32) {
        let slice = self.as_mut_slice();
        slice[0..4].copy_from_slice(&pn.to_le_bytes());
    }

    pub fn is_bbn(&self) -> bool {
        self.view().is_bbn()
    }

    pub fn n(&self) -> u16 {
        self.view().n()
    }

    pub fn set_n(&mut self, n: u16) {
        let slice = self.as_mut_slice();
        slice[4..6].copy_from_slice(&n.to_le_bytes());
    }

    pub fn prefix_len(&self) -> u8 {
        self.view().prefix_len()
    }

    pub fn set_prefix_len(&mut self, len: u8) {
        let slice = self.as_mut_slice();
        slice[6] = len;
    }

    pub fn separator_len(&self) -> u8 {
        self.view().separator_len()
    }

    pub fn set_separator_len(&mut self, len: u8) {
        let slice = self.as_mut_slice();
        slice[7] = len;
    }

    fn varbits(&self) -> &BitSlice<u8, Msb0> {
        self.view().varbits()
    }

    fn varbits_mut(&mut self) -> &mut BitSlice<u8, Msb0> {
        let body_end = body_size(
            self.prefix_len() as _,
            self.separator_len() as _,
            self.n() as _,
        ) + 8;
        self.as_mut_slice()[8..body_end].view_bits_mut()
    }

    pub fn prefix(&self) -> &BitSlice<u8, Msb0> {
        self.view().prefix()
    }

    pub fn separator(&self, i: usize) -> &BitSlice<u8, Msb0> {
        self.view().separator(i)
    }

    pub fn node_pointer(&self, i: usize) -> u32 {
        self.view().node_pointer(i)
    }

    fn set_node_pointer(&mut self, i: usize, node_pointer: u32) {
        let offset = BRANCH_NODE_SIZE - (self.n() as usize - i) * 4;
        self.as_mut_slice()[offset..offset + 4].copy_from_slice(&node_pointer.to_le_bytes());
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
    inner: &'a [u8],
}

impl<'a> BranchNodeView<'a> {
    pub fn from_slice(slice: &'a [u8]) -> Self {
        assert_eq!(slice.len(), BRANCH_NODE_SIZE);
        BranchNodeView { inner: slice }
    }

    pub fn bbn_pn(&self) -> u32 {
        u32::from_le_bytes(self.inner[0..4].try_into().unwrap())
    }

    pub fn is_bbn(&self) -> bool {
        self.bbn_pn() == 0
    }

    pub fn n(&self) -> u16 {
        u16::from_le_bytes(self.inner[4..6].try_into().unwrap())
    }

    pub fn prefix_len(&self) -> u8 {
        self.inner[6]
    }

    pub fn separator_len(&self) -> u8 {
        self.inner[7]
    }

    fn varbits(&self) -> &'a BitSlice<u8, Msb0> {
        let body_end = body_size(
            self.prefix_len() as _,
            self.separator_len() as _,
            self.n() as _,
        ) + 8;
        self.inner[8..body_end].view_bits()
    }

    pub fn prefix(&self) -> &'a BitSlice<u8, Msb0> {
        &self.varbits()[..self.prefix_len() as usize]
    }

    pub fn separator(&self, i: usize) -> &'a BitSlice<u8, Msb0> {
        let offset = self.prefix_len() as usize + i * self.separator_len() as usize;
        &self.varbits()[offset..offset + self.separator_len() as usize]
    }

    pub fn node_pointer(&self, i: usize) -> u32 {
        let offset = BRANCH_NODE_SIZE - (self.n() as usize - i) * 4;
        u32::from_le_bytes(self.inner[offset..offset + 4].try_into().unwrap())
    }
}

unsafe impl Send for BranchNode {}

pub fn body_size(prefix_len: usize, separator_len: usize, n: usize) -> usize {
    // prefix plus separator lengths are measured in bits, which we round
    // up to the next byte boundary and then follow by the node pointers.
    (prefix_len + (separator_len * n) + 7) / 8 + (4 * n)
}

pub fn body_fullness(prefix_len: usize, separator_len: usize, n: usize) -> f32 {
    body_size(prefix_len, separator_len, n) as f32 / BRANCH_NODE_BODY_SIZE as f32
}

pub struct BranchNodeBuilder {
    branch: BranchNode,
    index: usize,
    prefix_len: usize,
    separator_len: usize,
}

impl BranchNodeBuilder {
    pub fn new(
        mut branch: BranchNode,
        n: usize,
        prefix_len: usize,
        total_separator_len: usize,
    ) -> Self {
        let separator_len = total_separator_len - prefix_len;

        branch.set_n(n as u16);
        branch.set_prefix_len(prefix_len as u8);
        branch.set_separator_len(separator_len as u8);

        BranchNodeBuilder {
            branch,
            index: 0,
            prefix_len,
            separator_len,
        }
    }

    pub fn push(&mut self, key: Key, pn: u32) {
        assert!(self.index < self.branch.n() as usize);

        let varbits = self.branch.varbits_mut();
        if self.index == 0 {
            let prefix = &key.view_bits::<Msb0>()[..self.prefix_len];
            varbits[..self.prefix_len].copy_from_bitslice(prefix);
        }

        let separator = &key.view_bits::<Msb0>()[self.prefix_len..][..self.separator_len];

        let cell_start = self.prefix_len + self.index * self.separator_len;
        let cell_end = cell_start + self.separator_len;
        varbits[cell_start..cell_end].copy_from_bitslice(separator);

        self.branch.set_node_pointer(self.index, pn);

        self.index += 1;
    }

    pub fn finish(self) -> BranchNode {
        self.branch
    }
}
