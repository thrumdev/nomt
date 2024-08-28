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
