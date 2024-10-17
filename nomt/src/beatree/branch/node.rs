use bitvec::prelude::*;

use super::BRANCH_NODE_SIZE;
use crate::beatree::Key;
use crate::io::{FatPage, PAGE_SIZE};

use std::ops::{Deref, DerefMut};

// Here is the layout of a branch node:
//
// ```rust,ignore
// bbn_pn: u32            // the page number this is stored under,
//                        // if this is a BBN, 0 otherwise.
//
// n: u16                 // item count
// prefix_compressed: u16 // number of consecutive separators sharing the prefix,
//                        // starting from the first one in the separators list,
//                        // followed by entirely encoded separators
// prefix_len: u16        // bits
// cells: u16[n]          // bit offsets of the end of separators within the separators bitvec
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

const BRANCH_NODE_HEADER_SIZE: usize = 4 + 2 + 2 + 2;
pub const BRANCH_NODE_BODY_SIZE: usize = BRANCH_NODE_SIZE - BRANCH_NODE_HEADER_SIZE;

/// A branch node.
#[derive(Clone)]
pub struct BranchNode<T = FatPage> {
    pub(super) page: T,
}

impl<T> BranchNode<T> {
    /// Take the underlying page from the branch node wrapper.
    pub fn into_inner(self) -> T {
        self.page
    }
}

impl<T: Deref<Target = [u8]>> BranchNode<T> {
    /// Create a new read-only branch node.
    ///
    /// ## Panics
    ///
    /// This panics at runtime if the buffer size is not equal to the expected page size.
    pub fn new(page: T) -> BranchNode<T> {
        assert_eq!(page.len(), PAGE_SIZE);
        BranchNode { page }
    }

    pub fn as_slice(&self) -> &[u8] {
        &*self.page
    }

    pub fn bbn_pn(&self) -> u32 {
        u32::from_le_bytes(self.as_slice()[0..4].try_into().unwrap())
    }

    pub fn n(&self) -> u16 {
        u16::from_le_bytes(self.as_slice()[4..6].try_into().unwrap())
    }
    pub fn prefix_compressed(&self) -> u16 {
        u16::from_le_bytes(self.as_slice()[6..8].try_into().unwrap())
    }

    pub fn prefix_len(&self) -> u16 {
        u16::from_le_bytes(self.as_slice()[8..10].try_into().unwrap())
    }

    pub fn prefix<'a>(&'a self) -> &'a BitSlice<u8, Msb0> {
        let start = BRANCH_NODE_HEADER_SIZE + self.n() as usize * 2;
        &self.as_slice()[start..].view_bits()[..self.prefix_len() as usize]
    }

    pub fn cell(&self, i: usize) -> usize {
        let cell_offset = BRANCH_NODE_HEADER_SIZE + (i * 2);
        u16::from_le_bytes(self.as_slice()[cell_offset..][..2].try_into().unwrap()) as usize
    }

    pub fn raw_prefix<'a>(&'a self) -> RawPrefix<'a> {
        let bit_len = self.prefix_len() as usize;

        let start = BRANCH_NODE_HEADER_SIZE + self.n() as usize * 2;
        let end = start + ((bit_len + 7) / 8);

        (&self.as_slice()[start..end], bit_len)
    }

    pub fn separator<'a>(&'a self, i: usize) -> &'a BitSlice<u8, Msb0> {
        let start_separators = BRANCH_NODE_HEADER_SIZE + self.n() as usize * 2;
        let mut bit_offset_start = self.prefix_len() as usize;
        bit_offset_start += if i != 0 { self.cell(i - 1) } else { 0 };
        let bit_offset_end = self.prefix_len() as usize + self.cell(i);
        &self.as_slice()[start_separators..].view_bits()[bit_offset_start..bit_offset_end]
    }

    pub fn raw_separator<'a>(&'a self, i: usize) -> RawSeparator<'a> {
        let mut bit_offset_start = self.prefix_len() as usize;
        bit_offset_start += if i != 0 { self.cell(i - 1) } else { 0 };
        let bit_offset_end = self.prefix_len() as usize + self.cell(i);

        let bit_len = bit_offset_end - bit_offset_start;

        if bit_len == 0 {
            return (&[], 0, bit_len);
        }

        let bit_init = bit_offset_start % 8;
        let start_separators = BRANCH_NODE_HEADER_SIZE + self.n() as usize * 2;
        let start = start_separators + (bit_offset_start / 8);
        // load only slices into RawSeparator that have a length multiple of 8 bytes
        let byte_len = (((bit_init + bit_len) + 7) / 8).next_multiple_of(8);

        (&self.as_slice()[start..start + byte_len], bit_init, bit_len)
    }

    pub fn node_pointer(&self, i: usize) -> u32 {
        let offset = BRANCH_NODE_SIZE - (self.n() as usize - i) * 4;
        u32::from_le_bytes(self.as_slice()[offset..offset + 4].try_into().unwrap())
    }
}

impl<T: DerefMut<Target = [u8]>> BranchNode<T> {
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut *self.page
    }

    pub fn set_bbn_pn(&mut self, pn: u32) {
        let slice = self.as_mut_slice();
        slice[0..4].copy_from_slice(&pn.to_le_bytes());
    }

    pub fn set_n(&mut self, n: u16) {
        let slice = self.as_mut_slice();
        slice[4..6].copy_from_slice(&n.to_le_bytes());
    }

    pub fn set_prefix_compressed(&mut self, prefix_compressed: u16) {
        let slice = self.as_mut_slice();
        slice[6..8].copy_from_slice(&prefix_compressed.to_le_bytes());
    }

    pub fn set_prefix_len(&mut self, len: u16) {
        let slice = self.as_mut_slice();
        slice[8..10].copy_from_slice(&len.to_le_bytes());
    }

    fn set_prefix(&mut self, prefix: &BitSlice<u8, Msb0>) {
        let start = BRANCH_NODE_HEADER_SIZE + self.n() as usize * 2;
        let prefix_len = self.prefix_len() as usize;
        self.as_mut_slice()[start..].view_bits_mut()[..prefix_len].copy_from_bitslice(prefix);
    }

    fn set_separator(
        &mut self,
        i: usize,
        separator: &BitSlice<u8, Msb0>,
        bit_offset_start: usize,
        bit_offset_end: usize,
    ) {
        let n = self.n() as usize;
        let prefix_len = self.prefix_len() as usize;
        let slice = self.as_mut_slice();

        let cells_start = BRANCH_NODE_HEADER_SIZE + (i * 2);
        slice[cells_start..][..2].copy_from_slice(&(bit_offset_end as u16).to_le_bytes());

        let separators_start = BRANCH_NODE_HEADER_SIZE + (n * 2);
        slice[separators_start..].view_bits_mut()[prefix_len..][bit_offset_start..bit_offset_end]
            .copy_from_bitslice(&separator);
    }

    fn set_node_pointer(&mut self, i: usize, node_pointer: u32) {
        let offset = BRANCH_NODE_SIZE - (self.n() as usize - i) * 4;
        self.as_mut_slice()[offset..offset + 4].copy_from_slice(&node_pointer.to_le_bytes());
    }
}

#[cfg(feature = "benchmarks")]
impl BranchNode<FatPage> {
    pub fn new_fat(page_pool: &crate::io::PagePool) -> Self {
        BranchNode {
            page: page_pool.alloc_fat_page(),
        }
    }
}

// A RawPrefix is made by a tuple of raw bytes and the relative bit length
pub type RawPrefix<'a> = (&'a [u8], usize);
// A RawSeparator is made by a triple, the raw bytes, the bit-offset
// at which the separator starts to be encoded in the first byte
// and the relative bit length
//
// The raw bytes are always a multiple of 8 bytes in length
pub type RawSeparator<'a> = (&'a [u8], usize, usize);

pub fn body_size(prefix_len: usize, total_separator_lengths: usize, n: usize) -> usize {
    // prefix plus separator lengths are measured in bits, which we round
    // up to the next byte boundary. They are preceded by cells and followed by node pointers
    (n * 2) + (prefix_len + total_separator_lengths + 7) / 8 + (n * 4)
}

pub struct BranchNodeBuilder<T> {
    branch: BranchNode<T>,
    index: usize,
    prefix_len: usize,
    prefix_compressed: usize,
    separator_bit_offset: usize,
}

impl<T: DerefMut<Target = [u8]>> BranchNodeBuilder<T> {
    pub fn new(
        mut branch: BranchNode<T>,
        n: usize,
        prefix_compressed: usize,
        prefix_len: usize,
    ) -> Self {
        branch.set_n(n as u16);
        branch.set_prefix_compressed(prefix_compressed as u16);
        branch.set_prefix_len(prefix_len as u16);

        BranchNodeBuilder {
            branch,
            index: 0,
            prefix_len,
            prefix_compressed,
            separator_bit_offset: 0,
        }
    }

    pub fn push(&mut self, key: Key, mut separator_len: usize, pn: u32) {
        assert!(self.index < self.branch.n() as usize);

        if self.index == 0 {
            let prefix = &key.view_bits::<Msb0>()[..self.prefix_len];
            self.branch.set_prefix(prefix);
        }

        let separator = if self.index < self.prefix_compressed {
            // The first separator can have length less than prefix due to trailing zero
            // compression.
            separator_len = separator_len.saturating_sub(self.prefix_len);
            &key.view_bits::<Msb0>()[self.prefix_len..][..separator_len]
        } else {
            &key.view_bits::<Msb0>()[..separator_len]
        };

        let offset_start = self.separator_bit_offset;
        let offset_end = self.separator_bit_offset + separator_len;

        self.branch
            .set_separator(self.index, separator, offset_start, offset_end);

        self.separator_bit_offset = offset_end;

        self.branch.set_node_pointer(self.index, pn);

        self.index += 1;
    }

    pub fn finish(self) -> BranchNode<T> {
        self.branch
    }
}

#[cfg(feature = "benchmarks")]
pub mod benches {
    use crate::{
        beatree::{
            ops::bit_ops::separator_len,
            Key,
            {
                benches::get_keys,
                branch::{BranchNode, BranchNodeBuilder},
            },
        },
        io::{PagePool, PAGE_SIZE},
    };
    use criterion::{BenchmarkId, Criterion};

    pub fn branch_builder_benchmark(c: &mut Criterion) {
        let mut group = c.benchmark_group("branch_builder");

        // benchmark the branch builder creating an almost full branch node
        // given different prefix sizes

        let page_pool = PagePool::new();

        for prefix_len_bytes in [1, 4, 8, 12, 16] {
            // body_size = (2 * n) + (prefix_len_bits + (separator_len_bits * n) + 7)/8 + (4 * n)
            // n = (8 * body_size - prefix_len_bits) / (separator_len_bits + 8*6)
            let body_size_target = PAGE_SIZE - 8;
            let prefix_len_bits = prefix_len_bytes * 8;
            let separator_len_bits = (32 - prefix_len_bytes) * 8;
            let n = (8 * body_size_target - prefix_len_bits) / (separator_len_bits + 8 * 6);

            let mut separators: Vec<(usize, Key)> = get_keys(prefix_len_bytes, n)
                .into_iter()
                .map(|s| (separator_len(&s), s))
                .collect();
            separators.sort_by(|a, b| a.1.cmp(&b.1));

            group.bench_function(
                BenchmarkId::new("prefix_len_bytes", prefix_len_bytes),
                |b| {
                    b.iter_batched(
                        || (BranchNode::new_fat(&page_pool), separators.clone()),
                        |(branch_node, separators)| {
                            let mut branch_node_builder =
                                BranchNodeBuilder::new(branch_node, n, n, prefix_len_bits);

                            for (index, (separator_len, separator)) in
                                separators.into_iter().enumerate()
                            {
                                branch_node_builder.push(separator, separator_len, index as u32);
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
