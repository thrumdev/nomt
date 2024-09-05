use anyhow::Result;
use bitvec::prelude::*;
use crossbeam_channel::{Receiver, Sender};

use std::collections::{BTreeMap, HashMap};

use crate::beatree::{
    allocator::PageNumber,
    bbn,
    branch::{BranchNodePool, BRANCH_NODE_BODY_SIZE},
    index::Index,
    leaf::{
        node::{LeafNode, LEAF_NODE_BODY_SIZE, MAX_LEAF_VALUE_SIZE},
        overflow,
        store::{LeafStoreReader, LeafStoreWriter},
    },
    Key,
};

use super::BranchId;

mod branch_stage;
mod branch_updater;
mod leaf_stage;
mod leaf_updater;

// All nodes less than this body size will be merged with a neighboring node.
const BRANCH_MERGE_THRESHOLD: usize = BRANCH_NODE_BODY_SIZE / 2;

// At 180% of the branch size, we perform a 'bulk split' which follows a different algorithm
// than a simple split. Bulk splits are encountered when there are a large number of insertions
// on a single node, typically when inserting into a fresh database.
const BRANCH_BULK_SPLIT_THRESHOLD: usize = (BRANCH_NODE_BODY_SIZE * 9) / 5;
// When performing a bulk split, we target 75% fullness for all of the nodes we create except the
// last.
const BRANCH_BULK_SPLIT_TARGET: usize = (BRANCH_NODE_BODY_SIZE * 3) / 4;

const LEAF_MERGE_THRESHOLD: usize = LEAF_NODE_BODY_SIZE / 2;
const LEAF_BULK_SPLIT_THRESHOLD: usize = (LEAF_NODE_BODY_SIZE * 9) / 5;
const LEAF_BULK_SPLIT_TARGET: usize = (LEAF_NODE_BODY_SIZE * 3) / 4;

/// Change the btree in the specified way. Updates the branch index in-place and returns
/// a list of branches which have become obsolete.
///
/// The changeset is a list of key value pairs to be added or removed from the btree.
pub fn update(
    changeset: &BTreeMap<Key, Option<Vec<u8>>>,
    bbn_index: &mut Index,
    bnp: &mut BranchNodePool,
    leaf_reader: &LeafStoreReader,
    leaf_writer: &mut LeafStoreWriter,
    bbn_writer: &mut bbn::BbnStoreWriter,
) -> Result<Vec<BranchId>> {
    let leaf_cache = preload_leaves(leaf_reader, &bbn_index, &bnp, changeset.keys().cloned())?;

    let changeset = changeset
        .iter()
        .map(|(k, v)| match v {
            Some(v) if v.len() <= MAX_LEAF_VALUE_SIZE => (*k, Some((v.clone(), false))),
            Some(large_value) => {
                let pages = overflow::chunk(&large_value, leaf_writer);
                let cell = overflow::encode_cell(large_value.len(), &pages);
                (*k, Some((cell, true)))
            }
            None => (*k, None),
        })
        .collect::<_>();

    let (leaf_changes, overflow_deleted) =
        leaf_stage::run(&bbn_index, &bnp, leaf_cache, leaf_reader, changeset);

    let branch_changeset = leaf_changes
        .into_iter()
        .map(|(key, leaf_entry)| {
            let leaf_pn = leaf_entry.inserted.map(|leaf| leaf_writer.write(leaf));
            if let Some(prev_pn) = leaf_entry.deleted {
                leaf_writer.release(prev_pn);
            }

            (key, leaf_pn)
        })
        .collect::<Vec<_>>();

    for overflow_cell in overflow_deleted {
        overflow::delete(&overflow_cell, leaf_reader, leaf_writer);
    }

    let (branch_changes, fresh_released) = branch_stage::run(&bbn_index, &bnp, branch_changeset);

    let mut removed_branches = Vec::new();
    for (key, changed_branch) in branch_changes {
        match changed_branch.inserted {
            Some((branch_id, node)) => {
                bbn_index.insert(key, branch_id);
                bbn_writer.allocate(node);
            }
            None => {
                bbn_index.remove(&key);
            }
        }

        if let Some((deleted_branch_id, deleted_pn)) = changed_branch.deleted {
            removed_branches.push(deleted_branch_id);
            bbn_writer.release(deleted_pn);
        }
    }

    for branch_id in fresh_released {
        bnp.release(branch_id);
    }

    Ok(removed_branches)
}

fn preload_leaves(
    leaf_reader: &LeafStoreReader,
    bbn_index: &Index,
    bnp: &BranchNodePool,
    keys: impl IntoIterator<Item = Key>,
) -> Result<HashMap<PageNumber, LeafNode>> {
    let mut leaf_pages = HashMap::new();
    let mut last_pn = None;

    let mut submissions = 0;
    for key in keys {
        let Some((_, branch_id)) = bbn_index.lookup(key) else {
            continue;
        };
        // UNWRAP: all branches in index exist.
        let branch = bnp.checkout(branch_id).unwrap();
        let Some((_, leaf_pn)) = super::search_branch(&branch, key) else {
            continue;
        };
        if last_pn == Some(leaf_pn) {
            continue;
        }
        last_pn = Some(leaf_pn);
        leaf_reader
            .io_handle()
            .send(leaf_reader.io_command(leaf_pn, leaf_pn.0 as u64))
            .expect("I/O Pool Disconnected");

        submissions += 1;
    }

    for _ in 0..submissions {
        let completion = leaf_reader
            .io_handle()
            .recv()
            .expect("I/O Pool Disconnected");
        completion.result?;
        let pn = PageNumber(completion.command.user_data as u32);
        let page = completion.command.kind.unwrap_buf();
        leaf_pages.insert(pn, LeafNode { inner: page });
    }

    Ok(leaf_pages)
}

// A half-open range [low, high), where each key corresponds to a known separator of a node.
struct SeparatorRange {
    low: Option<Key>,
    high: Option<Key>,
}

struct LeftNeighbor<T> {
    rx: Receiver<ExtendRangeRequest<T>>,
}

struct RightNeighbor<T> {
    tx: Sender<ExtendRangeRequest<T>>,
}

// a request to extend the range to the next node following the high bound of the range.
struct ExtendRangeRequest<T> {
    tx: Sender<T>,
}

pub fn reconstruct_key(prefix: &BitSlice<u8, Msb0>, separator: &BitSlice<u8, Msb0>) -> Key {
    let mut key = [0u8; 32];
    key.view_bits_mut::<Msb0>()[..prefix.len()].copy_from_bitslice(prefix);
    key.view_bits_mut::<Msb0>()[prefix.len()..][..separator.len()].copy_from_bitslice(separator);
    key
}

// separate two keys a and b where b > a
pub fn separate(a: &Key, b: &Key) -> Key {
    // if b > a at some point b must have a 1 where a has a 0 and they are equal up to that point.
    let len = a
        .view_bits::<Msb0>()
        .iter()
        .zip(b.view_bits::<Msb0>().iter())
        .take_while(|(a, b)| a == b)
        .count()
        + 1;

    let mut separator = [0u8; 32];
    separator.view_bits_mut::<Msb0>()[..len].copy_from_bitslice(&b.view_bits::<Msb0>()[..len]);
    separator
}

pub fn prefix_len(key_a: &Key, key_b: &Key) -> usize {
    key_a
        .view_bits::<Msb0>()
        .iter()
        .zip(key_b.view_bits::<Msb0>().iter())
        .take_while(|(a, b)| a == b)
        .count()
}

pub fn separator_len(key: &Key) -> usize {
    if key == &[0u8; 32] {
        return 1;
    }
    let key = &key.view_bits::<Msb0>();
    key.len() - key.trailing_zeros()
}

#[cfg(feature = "benchmarks")]
pub mod benches {
    use bitvec::{prelude::Msb0, view::BitView};
    use criterion::{BenchmarkId, Criterion};
    use rand::RngCore;

    pub fn separate_benchmark(c: &mut Criterion) {
        let mut group = c.benchmark_group("separate");

        for shared_bytes in [0, 4, 8, 12, 16] {
            let (key1, key2) = get_key_pair(shared_bytes);
            group.bench_function(BenchmarkId::new("shared_bytes", shared_bytes), |b| {
                b.iter(|| super::separate(&key1, &key2));
            });
        }

        group.finish();
    }

    pub fn reconstruct_key_benchmark(c: &mut Criterion) {
        let mut group = c.benchmark_group("reconstruct_key");

        for prefix_bytes in [0, 4, 8, 12, 16, 20] {
            let mut rand = rand::thread_rng();
            let mut key = [0; 32];
            rand.fill_bytes(&mut key);

            let prefix = &key.view_bits::<Msb0>()[0..prefix_bytes * 8];
            let separator = &key.view_bits::<Msb0>()[prefix_bytes * 8..];

            group.bench_function(BenchmarkId::new("prefix_len_bytes", prefix_bytes), |b| {
                b.iter(|| super::reconstruct_key(prefix, separator))
            });
        }

        group.finish();
    }

    pub fn separator_len_benchmark(c: &mut Criterion) {
        let mut group = c.benchmark_group("separator_len");

        // n_bytes represents the amount of bytes set to one
        // from the beginning of the key
        for n_bytes in [16, 20, 24, 28, 31].into_iter().rev() {
            let mut separator = [0; 32];
            for byte in separator.iter_mut().take(n_bytes) {
                *byte = 255;
            }

            group.bench_function(BenchmarkId::new("zero_bytes", 32 - n_bytes), |b| {
                b.iter(|| super::separator_len(&separator));
            });
        }

        group.finish();
    }

    pub fn prefix_len_benchmark(c: &mut Criterion) {
        let mut group = c.benchmark_group("prefix_len");

        for prefix_len_bytes in [0, 4, 8, 12, 16] {
            let (key1, key2) = get_key_pair(prefix_len_bytes);
            group.bench_function(BenchmarkId::new("shared_bytes", prefix_len_bytes), |b| {
                b.iter(|| super::prefix_len(&key1, &key2));
            });
        }

        group.finish();
    }
}
