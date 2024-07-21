//! BTree Operations.

use anyhow::Result;
use bitvec::prelude::*;
use std::{collections::BTreeMap, fs::File};

use super::{
    branch::{self, BranchId},
    index::Index,
    leaf, Key,
};

mod reconstruction;

pub use reconstruction::reconstruct;

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
        Some(leaf_pn) => leaf_pn,
    };

    let _ = leaf_store;
    let _ = leaf_pn;
    todo!();
}

/// Change the btree in the specified way. Updates the branch index in-place and returns 
/// a list of branches which have become obsolete.
///
/// The changeset is a list of key value pairs to be added or removed from the btree.
pub fn update(
    sync_seqn: u32,
    next_bbn_seqn: &mut u32,
    changeset: BTreeMap<Key, Option<Vec<u8>>>,
    bbn_index: &mut Index,
    bnp: &mut branch::BranchNodePool,
    leaf_store: &mut leaf::store::LeafStoreTx,
) -> Result<Vec<BranchId>> {
    let _ = (sync_seqn, next_bbn_seqn, changeset, bbn_index, bnp, leaf_store);
    todo!();
}

/// Binary search a branch node for the child node containing the key. This returns the last child
/// node pointer whose separator is less than or equal to the given key.
fn search_branch(branch: &branch::BranchNode, key: Key) -> Option<leaf::PageNumber> {
    let prefix = branch.prefix();

    if key.view_bits::<Lsb0>()[..prefix.len()] != prefix {
        return None;
    }

    let post_key =
        &key.view_bits::<Lsb0>()[prefix.len()..prefix.len() + branch.separator_len() as usize];

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
        return None
    }
    let node_pointer = branch.node_pointer(high - 1);
    Some(node_pointer.into())
}
