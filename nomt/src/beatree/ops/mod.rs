//! BTree Operations.

use anyhow::Result;
use bitvec::prelude::*;

use std::cmp::Ordering;

use super::{
    allocator::PageNumber,
    branch::BranchNode,
    index::Index,
    leaf::{self, node::LeafNode},
    Key,
};

pub(crate) mod bit_ops;
mod reconstruction;
mod update;

pub use reconstruction::reconstruct;
pub use update::update;

/// Lookup a key in the btree.
pub fn lookup(
    key: Key,
    bbn_index: &Index,
    leaf_store: &leaf::store::LeafStoreReader,
) -> Result<Option<Vec<u8>>> {
    let branch = match bbn_index.lookup(key) {
        None => return Ok(None),
        Some((_, branch)) => branch,
    };

    let leaf_pn = match search_branch(&branch, key.clone()) {
        None => return Ok(None),
        Some((_, leaf_pn)) => leaf_pn,
    };

    let leaf = LeafNode {
        inner: leaf_store.query(leaf_pn),
    };

    let maybe_value = leaf.get(&key).map(|(v, is_overflow)| {
        if is_overflow {
            leaf::overflow::read(v, leaf_store)
        } else {
            v.to_vec()
        }
    });

    Ok(maybe_value)
}

/// Binary search a branch node for the child node containing the key. This returns the last child
/// node pointer whose separator is less than or equal to the given key.
fn search_branch(branch: &BranchNode, key: Key) -> Option<(usize, PageNumber)> {
    let prefix = branch.prefix();
    let n = branch.n() as usize;
    let prefix_compressed = branch.prefix_compressed() as usize;

    match key.view_bits::<Msb0>()[..prefix.len()].cmp(prefix) {
        Ordering::Less => return None,
        Ordering::Greater if n == prefix_compressed => {
            let i = n - 1;
            return Some((i, branch.node_pointer(i).into()));
        }
        Ordering::Equal | Ordering::Greater => {}
    }

    let mut low = 0;
    let mut high = branch.n() as usize;

    while low < high {
        let mid = low + (high - low) / 2;

        if key < get_key(branch, mid) {
            high = mid;
        } else {
            low = mid + 1;
        }
    }

    // sanity: this only happens if `key` is less than separator 0.
    if high == 0 {
        return None;
    }
    let node_pointer = branch.node_pointer(high - 1);
    Some((high - 1, node_pointer.into()))
}

// Extract the key at a given index from a BranchNode, taking into account prefix compression.
fn get_key(node: &BranchNode, index: usize) -> Key {
    let prefix = if index < node.prefix_compressed() as usize {
        Some(node.raw_prefix())
    } else {
        None
    };
    bit_ops::reconstruct_key(prefix, node.raw_separator(index))
}

#[cfg(feature = "benchmarks")]
pub mod benches {
    use crate::{
        beatree::{
            benches::get_keys,
            branch::{node::BranchNodeBuilder, BranchNode},
            ops::bit_ops::separator_len,
            Key,
        },
        io::{PagePool, PAGE_SIZE},
    };
    use criterion::{BenchmarkId, Criterion};
    use rand::Rng;

    pub fn search_branch_benchmark(c: &mut Criterion) {
        let mut group = c.benchmark_group("search_branch");
        let mut rand = rand::thread_rng();
        let page_pool = PagePool::new();

        for prefix_len_bytes in [1, 4, 8, 12, 16] {
            // fill the branch node with as many separators as possible
            //
            // body_size = (prefix_len_bits + (separator_len_bits * n) + 7)/8 + 4 * n
            // n = (8 * body_size - prefix_len_bits) / (separator_len_bits + 8*4)
            let body_size_target = PAGE_SIZE - 8;
            let prefix_len_bits = prefix_len_bytes * 8;
            let separator_len_bits = (32 - prefix_len_bytes) * 8;
            let n = (8 * body_size_target - prefix_len_bits) / (separator_len_bits + 8 * 4);

            let mut separators: Vec<(usize, Key)> = get_keys(prefix_len_bytes, n)
                .into_iter()
                .map(|s| (separator_len(&s), s))
                .collect();
            separators.sort_by(|a, b| a.1.cmp(&b.1));

            let branch_node = BranchNode::new_in(&page_pool);
            let mut branch_node_builder =
                BranchNodeBuilder::new(branch_node, n, prefix_len_bits, 256);

            for (index, (separator_len, separator)) in separators.iter().enumerate() {
                branch_node_builder.push(separator.clone(), *separator_len, index as u32);
            }

            let branch = branch_node_builder.finish();

            group.bench_function(
                BenchmarkId::new("prefix_len_bytes", prefix_len_bytes),
                |b| {
                    b.iter_batched(
                        || {
                            let index = rand.gen_range(0..separators.len());
                            separators[index].1.clone()
                        },
                        |separator| super::search_branch(&branch, separator),
                        criterion::BatchSize::SmallInput,
                    )
                },
            );
        }

        group.finish();
    }
}
