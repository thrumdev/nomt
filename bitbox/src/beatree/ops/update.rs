use anyhow::{bail, ensure, Result};
use bitvec::prelude::*;
use crossbeam_channel::Receiver;
use itertools::{Itertools, EitherOrBoth};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{ErrorKind, Read, Seek},
};

use crate::beatree::{
    allocator::PageNumber,
    bbn,
    branch::{self, BRANCH_NODE_SIZE, BRANCH_NODE_BODY_SIZE},
    index::Index, 
    leaf, Key,
};

use super::BranchId;

const BRANCH_MERGE_THRESHOLD: usize = BRANCH_NODE_BODY_SIZE / 2;

// At 180% of the branch size, we perform a 'bulk split' which follows a different algorithm
// than a simple split.
const BRANCH_BULK_SPLIT_THRESHOLD: usize = (BRANCH_NODE_BODY_SIZE * 9) / 5;

/// Change the btree in the specified way. Updates the branch index in-place and returns
/// a list of branches which have become obsolete.
///
/// The changeset is a list of key value pairs to be added or removed from the btree.
pub fn update(
    changeset: &BTreeMap<Key, Option<Vec<u8>>>,
    bbn_index: &mut Index,
    bnp: &mut branch::BranchNodePool,
    leaf_reader: &leaf::store::LeafStoreReader,
    leaf_writer: &mut leaf::store::LeafStoreWriter,
    bbn_store_writer: &mut bbn::BbnStoreWriter,
) -> Result<Vec<BranchId>> {
    let mut cur_branch: Option<BranchId> = None;
    let mut cur_leaf: Option<PageNumber> = None;

    for (key, value_change) in changeset {
        // 1. find branch for key. if not equal to the branch for the last key,
        //    apply final changes to last leaf and branch.
        // 2. find leaf for key. if not equal to leaf for last key, apply final changes
        //    to last leaf
    }

    Ok(todo!())
}

struct PrevBranch {
    node: branch::BranchNode,
    id: branch::BranchId,
}

struct BranchChanges {
    prev: Option<PrevBranch>,
    changes: Vec<(Key, Option<PageNumber>)>,
}

impl BranchChanges {
    fn is_empty(&self) -> bool {
        self.changes.is_empty()
    }

    // Apply the changes. Returns `Some` if there are leftover changes to be applied
    // to the first branch following the returned key. 
    fn apply(
        self, 
        bbn_index: &mut Index,
        bnp: &mut branch::BranchNodePool,
        bbn_store_writer: &mut bbn::BbnStoreWriter,
    ) -> Option<(Key, BranchChanges)> {
        if self.is_empty() { return None }

        let existing_keys = self.prev.as_ref().into_iter().flat_map(|p| {
            let prefix = p.node.prefix();
            (0..p.node.n()).map(|i| p.node.separator(i as usize)).map(move |separator| reconstruct_key(prefix, separator))
        });

        let merge = existing_keys.merge_join_by(
            self.changes.iter(),
            |existing_key, change| existing_key.cmp(&change.0)
        );

        let mut gauge = BranchGauge::new();

        for merge_item in merge.clone() {
            match merge_item {
                EitherOrBoth::Left(key) => {
                    gauge.ingest(key);
                }
                EitherOrBoth::Right((key, Some(_))) => {
                    gauge.ingest(*key);
                }
                EitherOrBoth::Right((_, None)) => panic!("removed nonexistent branch"),
                EitherOrBoth::Both(key, (_, Some(branch_id))) => {
                    gauge.ingest(key);
                }
                EitherOrBoth::Both(key, (_, None)) => {
                    gauge.ingest(key);
                }
            }

            if gauge.body_size() > BRANCH_BULK_SPLIT_THRESHOLD {
                // TODO: initiate bulk split.
            }
        }

        let new_size = gauge.body_size();

        if new_size > BRANCH_NODE_BODY_SIZE {
            // TODO: split
            todo!()
        } else if new_size <= BRANCH_MERGE_THRESHOLD {
            // TODO: collect all existing keys/updates/inserts into branch changes
            // and pass to next branch
            todo!()
        } else {
            // TODO: just create a replacement branch
            todo!()
        }

        // TODO: delete previous branch.
    }
}

struct BranchGauge {
    first_separator: Option<Key>,
    prefix_len: usize,
    // total separator len, not counting prefix.
    separator_len: usize,
    n: usize,
}

impl BranchGauge {
    fn new() -> Self {
        BranchGauge {
            first_separator: None,
            prefix_len: 0,
            separator_len: 0,
            n: 0,
        }
    }

    fn ingest(&mut self, key: Key) {
        let Some(ref first) = self.first_separator else {
            self.first_separator = Some(key);
            self.separator_len = separator_len(&key);
            self.prefix_len = self.separator_len;

            self.n = 1;
            return
        };

        self.separator_len = std::cmp::max(separator_len(&key), self.separator_len);
        self.prefix_len = prefix_len(first, &key);
        self.n += 1;
    }

    fn body_size(&self) -> usize {
        branch::body_size(self.prefix_len, self.separator_len - self.prefix_len, self.n)
    }
}

fn prefix_len(key_a: &Key, key_b: &Key) -> usize {
    key_a.view_bits::<Lsb0>()
        .iter()
        .zip(key_b.view_bits::<Lsb0>().iter())
        .take_while(|(a, b)| a == b)
        .count()
}

fn separator_len(key: &Key) -> usize {
    std::cmp::min(1, key.view_bits::<Lsb0>().last_one().unwrap_or(1))
}

fn reconstruct_key(
    prefix: &BitSlice<u8>,
    separator: &BitSlice<u8>,
) -> Key {
    let mut key = [0u8; 32];
    key.view_bits_mut::<Lsb0>()[..prefix.len()].copy_from_bitslice(prefix);
    key.view_bits_mut::<Lsb0>()[prefix.len()..][..separator.len()].copy_from_bitslice(prefix);
    key
}
