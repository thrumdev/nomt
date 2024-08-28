//! BTree Operations.

use anyhow::Result;
use bitvec::prelude::*;

use std::cmp::Ordering;

use super::{
    allocator::PageNumber,
    branch::{self, BranchId},
    index::Index,
    leaf::{self, node::LeafNode},
    Key,
};

mod reconstruction;
pub(crate) mod update;

pub use reconstruction::reconstruct;
pub use update::update;

/// Lookup a key in the btree.
pub fn lookup(
    key: Key,
    bbn_index: &Index,
    branch_node_pool: &branch::BranchNodePool,
    leaf_store: &leaf::store::LeafStoreReader,
) -> Result<Option<Vec<u8>>> {
    let branch_id = match bbn_index.lookup(key) {
        None => return Ok(None),
        Some(branch_id) => branch_id,
    };

    let branch = branch_node_pool
        .checkout(branch_id)
        .expect("missing branch node in pool");

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
fn search_branch(branch: &branch::BranchNode, key: Key) -> Option<(usize, PageNumber)> {
    let prefix = branch.prefix();

    match key.view_bits::<Msb0>()[..prefix.len()].cmp(prefix) {
        Ordering::Equal => {}
        Ordering::Less => return None,
        Ordering::Greater => {
            let i = branch.n() as usize - 1;
            return Some((i, branch.node_pointer(i).into()));
        }
    }

    let total_separator_len = prefix.len() + branch.separator_len() as usize;
    let post_key = &key.view_bits::<Msb0>()[prefix.len()..total_separator_len];

    let mut low = 0;
    let mut high = branch.n() as usize;

    while low < high {
        let mid = low + (high - low) / 2;
        if post_key < branch.separator(mid) {
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

#[cfg(feature = "benchmarks")]
pub mod benches {
    use crate::{
        beatree::{
            benches::get_keys,
            branch::{node::BranchNodeBuilder, BranchNodePool},
        },
        io::PAGE_SIZE,
    };
    use criterion::{BenchmarkId, Criterion};
    use rand::Rng;

    pub fn search_branch_benchmark(c: &mut Criterion) {
        let mut group = c.benchmark_group("search_branch");
        let mut rand = rand::thread_rng();

        let branch_node_pool = BranchNodePool::new();

        for prefix_len_bytes in [1, 4, 8, 12, 16] {
            // fill the branch node with as many separators as possible
            //
            // body_size = (prefix_len_bits + (separator_len_bits * n) + 7)/8 + 4 * n
            // n = (8 * body_size - prefix_len_bits) / (separator_len_bits + 8*4)
            let body_size_target = PAGE_SIZE - 8;
            let prefix_len_bits = prefix_len_bytes * 8;
            let separator_len_bits = (32 - prefix_len_bytes) * 8;
            let n = (8 * body_size_target - prefix_len_bits) / (separator_len_bits + 8 * 4);

            let mut separators = get_keys(prefix_len_bytes, n);
            separators.sort();

            let branch_id = branch_node_pool.allocate();
            let branch_node = branch_node_pool.checkout(branch_id).unwrap();
            let mut branch_node_builder =
                BranchNodeBuilder::new(branch_node, n, prefix_len_bits, 256);

            for (index, separator) in separators.iter().enumerate() {
                branch_node_builder.push(separator.clone(), index as u32);
            }

            let branch = branch_node_builder.finish();

            group.bench_function(
                BenchmarkId::new("prefix_len_bytes", prefix_len_bytes),
                |b| {
                    b.iter_batched(
                        || {
                            let index = rand.gen_range(0..separators.len());
                            separators[index].clone()
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
