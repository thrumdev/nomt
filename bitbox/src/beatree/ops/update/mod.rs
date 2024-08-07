use anyhow::{bail, ensure, Result};
use bitvec::prelude::*;
use crossbeam_channel::Receiver;
use std::{
    cmp::Ordering,
    collections::BTreeMap,
    fs::File,
    io::{ErrorKind, Read, Seek},
};

use crate::beatree::{
    allocator::PageNumber,
    bbn,
    branch::{BranchNodePool, BRANCH_NODE_SIZE, BRANCH_NODE_BODY_SIZE},
    index::Index, 
    leaf::{
        node::{LeafNode, LeafBuilder, LEAF_NODE_BODY_SIZE},
        store::{LeafStoreReader, LeafStoreWriter},
    }, Key,
};

use super::BranchId;
use branch::{BaseBranch, ActiveBranch};
use leaf::{BaseLeaf, ActiveLeaf};

mod branch;
mod leaf;

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
    bbn_store_writer: &mut bbn::BbnStoreWriter,
) -> Result<Vec<BranchId>> {
    let mut updater = Updater::new(&*bbn_index, &*bnp, leaf_reader);
    for (key, value_change) in changeset {
        updater.ingest(*key, value_change.clone(), leaf_writer);
    }

    updater.complete(leaf_writer);

    Ok(todo!())
}

struct Updater {
    active_branch: ActiveBranch,
    active_leaf: ActiveLeaf,
}

impl Updater {
    fn new(
        bbn_index: &Index,
        bnp: &BranchNodePool,
        leaf_reader: &LeafStoreReader,
    ) -> Self {
        let first = bbn_index.first();

        // UNWRAP: all nodes in index must exist.
        let first_branch = first.as_ref().map(|(_, id)| bnp.checkout(*id).unwrap());
        let first_branch_cutoff = first.as_ref()
            .and_then(|(k, _)| bbn_index.next_after(*k))
            .map(|(k, _)| k);

        // first leaf cutoff is the separator of the second leaf _or_ the separator of the next
        // branch if there is only 1 leaf, or nothing.
        let first_leaf_cutoff = first_branch.as_ref().and_then(|node| if node.n() > 1 {
            Some(reconstruct_key(node.prefix(), node.separator(1)))
        } else {
            None
        }).or(first_branch_cutoff);

        let first_leaf = first_branch.as_ref()
            .map(|node| (
                PageNumber::from(node.node_pointer(0)), 
                reconstruct_key(node.prefix(), node.separator(0)),
            ))
            .map(|(id, separator)| BaseLeaf {
                id,
                node: leaf_reader.query(id),
                iter_pos: 0,
                separator,
            });

        // active branch: first branch, cut-off second branch key.
        let active_branch = ActiveBranch::new(
            first_branch.map(|node| BaseBranch {
                // UNWRAP: node can only exist if ID does.
                id: first.as_ref().unwrap().1,
                node,
                iter_pos: 0,
            }),
           first_branch_cutoff,
        );

        // active leaf: first leaf in first branch, cut-off second leaf key.
        let active_leaf = ActiveLeaf::new(first_leaf, first_leaf_cutoff);

        Updater {
            active_branch,
            active_leaf,
        }
    }

    fn ingest(
        &mut self, 
        key: Key, 
        value_change: Option<Vec<u8>>,
        leaf_writer: &mut LeafStoreWriter,
    ) {
        // This is a while loop because merges may require multiple iterations.
        while !self.active_leaf.is_in_scope(&key) {
            self.active_leaf.complete(&mut self.active_branch, leaf_writer);
            // TODO: determine where the active leaf should be set next. if merging, it's just the
            // 'next' leaf. If done, it's the leaf for the key. Set active branch to the branch
            // for the active leaf.

            if self.active_branch.is_in_scope(&key) {
                continue
            }
            // TODO: complete branch. set active to relevant leaf and branch.
        }

        self.active_leaf.ingest(key, value_change);
    }

    fn complete(&mut self, leaf_writer: &mut LeafStoreWriter) {
        self.active_leaf.complete(&mut self.active_branch, leaf_writer);
        self.active_branch.complete();
    }
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
