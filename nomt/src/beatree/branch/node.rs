use bitvec::prelude::*;
use std::sync::{Arc, Mutex};

use super::{BranchId, BranchNodePoolInner, BRANCH_NODE_SIZE};
use crate::beatree::Key;

// Here is the layout of a branch node:
//
// ```rust,ignore
// bbn_pn: u32          // the page number this is stored under,
//                      // if this is a BBN, 0 otherwise.
//
// n: u16               // item count
// prefix_len: u16      // bits
// cells: u16[n]        // bit offsets of the end of separators within the separators bitvec
//
// # To avoid two padding bytes, prefix and separators are stored in a single bitvec.
//
// prefix: bitvec[prefix_len]
// separators: bitvec
//
// # Node pointers follow. The list is aligned to the end of the node, with the last item in the
// # list occupying the last 4 bytes of the node.
//
// node_pointers: LNPN or BNID[n]
// ```

const BRANCH_NODE_HEADER_SIZE: usize = 4 + 2 + 2;
pub const BRANCH_NODE_BODY_SIZE: usize = BRANCH_NODE_SIZE - BRANCH_NODE_HEADER_SIZE;

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

    pub fn n(&self) -> u16 {
        self.view().n()
    }

    pub fn set_n(&mut self, n: u16) {
        let slice = self.as_mut_slice();
        slice[4..6].copy_from_slice(&n.to_le_bytes());
    }

    pub fn prefix_len(&self) -> u16 {
        self.view().prefix_len()
    }

    pub fn set_prefix_len(&mut self, len: u16) {
        let slice = self.as_mut_slice();
        slice[6..8].copy_from_slice(&len.to_le_bytes());
    }

    pub fn prefix(&self) -> &BitSlice<u8, Msb0> {
        self.view().prefix()
    }

    fn set_prefix(&mut self, prefix: &BitSlice<u8, Msb0>) {
        let start = BRANCH_NODE_HEADER_SIZE + self.n() as usize * 2;
        let prefix_len = self.prefix_len() as usize;
        self.as_mut_slice()[start..].view_bits_mut()[..prefix_len].copy_from_bitslice(prefix);
    }

    pub fn separator(&self, i: usize) -> &BitSlice<u8, Msb0> {
        self.view().separator(i)
    }

    fn set_separators(&mut self, cells: Vec<u8>, separators: BitVec<u8, Msb0>) {
        let n = self.n() as usize;
        let prefix_len = self.prefix_len() as usize;
        let slice = self.as_mut_slice();

        let cells_start = BRANCH_NODE_HEADER_SIZE;
        let cells_end = cells_start + n * 2;
        slice[cells_start..cells_end].copy_from_slice(cells.as_slice());

        slice[cells_end..].view_bits_mut()[prefix_len..][..separators.len()]
            .copy_from_bitslice(&separators);
    }

    pub fn node_pointer(&self, i: usize) -> u32 {
        self.view().node_pointer(i)
    }

    fn set_node_pointer(&mut self, i: usize, node_pointer: u32) {
        let offset = BRANCH_NODE_SIZE - (self.n() as usize - i) * 4;
        self.as_mut_slice()[offset..offset + 4].copy_from_slice(&node_pointer.to_le_bytes());
    }
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

    pub fn n(&self) -> u16 {
        u16::from_le_bytes(self.inner[4..6].try_into().unwrap())
    }

    pub fn prefix_len(&self) -> u16 {
        u16::from_le_bytes(self.inner[6..8].try_into().unwrap())
    }

    pub fn cell(&self, i: usize) -> usize {
        let cell_offset = BRANCH_NODE_HEADER_SIZE + (i * 2);
        u16::from_le_bytes(self.inner[cell_offset..][..2].try_into().unwrap()) as usize
    }

    pub fn prefix(&self) -> &'a BitSlice<u8, Msb0> {
        let start = BRANCH_NODE_HEADER_SIZE + self.n() as usize * 2;
        &self.inner[start..].view_bits()[..self.prefix_len() as usize]
    }

    pub fn separator(&self, i: usize) -> &'a BitSlice<u8, Msb0> {
        let start_separators = BRANCH_NODE_HEADER_SIZE + self.n() as usize * 2;

        let mut bit_offset_start = self.prefix_len() as usize;
        bit_offset_start += if i != 0 { self.cell(i - 1) } else { 0 };
        let bit_offset_end = self.prefix_len() as usize + self.cell(i);

        &self.inner[start_separators..].view_bits()[bit_offset_start..bit_offset_end]
    }

    pub fn node_pointer(&self, i: usize) -> u32 {
        let offset = BRANCH_NODE_SIZE - (self.n() as usize - i) * 4;
        u32::from_le_bytes(self.inner[offset..offset + 4].try_into().unwrap())
    }
}

unsafe impl Send for BranchNode {}

pub fn body_size(prefix_len: usize, tot_separators_len: usize, n: usize) -> usize {
    // prefix plus separator lengths are measured in bits, which we round
    // up to the next byte boundary. They are preceded by cells and followed by node pointers
    (n * 2) + (prefix_len + tot_separators_len + 7) / 8 + (n * 4)
}

pub struct BranchNodeBuilder {
    branch: BranchNode,
    index: usize,
    prefix_len: usize,
    cells: Vec<u8>,
    separator_bit_offset: u16,
    separators: BitVec<u8, Msb0>,
}

impl BranchNodeBuilder {
    pub fn new(mut branch: BranchNode, n: usize, prefix_len: usize) -> Self {
        branch.set_n(n as u16);
        branch.set_prefix_len(prefix_len as u16);

        BranchNodeBuilder {
            branch,
            index: 0,
            prefix_len,
            cells: Vec::new(),
            separator_bit_offset: 0,
            separators: BitVec::new(),
        }
    }

    pub fn push(&mut self, key: Key, separator_len: usize, pn: u32) {
        assert!(self.index < self.branch.n() as usize);

        if self.index == 0 {
            let prefix = &key.view_bits::<Msb0>()[..self.prefix_len];
            self.branch.set_prefix(prefix);
        }

        // There are cases, for example a separator made by all zeros, where the
        // prefix_len could be bigger then the separator_len
        let separator_len = separator_len.saturating_sub(self.prefix_len);

        let separator = &key.view_bits::<Msb0>()[self.prefix_len..][..separator_len];

        self.separator_bit_offset += separator_len as u16;
        self.cells.extend(self.separator_bit_offset.to_le_bytes());

        self.separators.extend_from_bitslice(separator);

        self.branch.set_node_pointer(self.index, pn);

        self.index += 1;
    }

    pub fn finish(mut self) -> BranchNode {
        self.branch.set_separators(self.cells, self.separators);
        self.branch
    }
}

#[cfg(feature = "benchmarks")]
pub mod benches {
    use crate::{
        beatree::{
            benches::get_keys,
            branch::{BranchNodeBuilder, BranchNodePool},
        },
        io::PAGE_SIZE,
    };
    use criterion::{BenchmarkId, Criterion};

    pub fn branch_builder_benchmark(c: &mut Criterion) {
        let mut group = c.benchmark_group("branch_builder");

        // benchmark the branch builder creating an almost full branch node
        // given different prefix sizes

        let branch_node_pool = BranchNodePool::new();

        for prefix_len_bytes in [1, 4, 8, 12, 16] {
            // body_size = (prefix_len_bits + (separator_len_bits * n) + 7)/8 + 4 * n
            // n = (8 * body_size - prefix_len_bits) / (separator_len_bits + 8*4)
            let body_size_target = PAGE_SIZE - 8;
            let prefix_len_bits = prefix_len_bytes * 8;
            let separator_len_bits = (32 - prefix_len_bytes) * 8;
            let n = (8 * body_size_target - prefix_len_bits) / (separator_len_bits + 8 * 4);

            let mut separators = get_keys(prefix_len_bytes, n);
            separators.sort();

            group.bench_function(
                BenchmarkId::new("prefix_len_bytes", prefix_len_bytes),
                |b| {
                    b.iter_batched(
                        || {
                            let branch_id = branch_node_pool.allocate();
                            branch_node_pool.checkout(branch_id).unwrap()
                        },
                        |branch_node| {
                            let mut branch_node_builder =
                                BranchNodeBuilder::new(branch_node, n, prefix_len_bits, 256);

                            for (index, separator) in separators.iter().enumerate() {
                                branch_node_builder.push(separator.clone(), index as u32);
                            }

                            branch_node_builder.finish();
                        },
                        criterion::BatchSize::SmallInput,
                    )
                },
            );
        }

        group.finish();
    }
}
