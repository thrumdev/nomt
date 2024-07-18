use anyhow::Result;
use bitvec::prelude::*;
use std::{cmp::Ordering, collections::BTreeMap, fs::File};

use super::{
    branch::{self, BranchId},
    leaf::{self, LeafPn},
};

// TODO: this should be [u8; 32] or (even better) [u8; KEY_LEN] but leaving as-is for now.
pub type Key = Vec<u8>;

/// Lookup a key in the btree.
pub fn lookup(
    key: Key,
    root: BranchId,
    branch_node_pool: &branch::BranchNodePool,
    leaf_store: &leaf::LeafStore,
) -> Result<Option<Vec<u8>>> {
    let mut branch_id = root;
    let leaf_pn = loop {
        let branch = branch_node_pool.checkout(branch_id).expect("missing branch node in pool");
        match search_branch(&branch, key.clone()) {
            None => return Ok(None),
            Some(NodePointer::Branch(id)) => {
                branch_id = id;
                continue
            }
            Some(NodePointer::Leaf(leaf_pn)) => break leaf_pn,
        }
    };

    let _ = leaf_store;
    todo!();
}

/// Change the btree in the specified way. Returns the root of the new btree.
///
/// The changeset is a list of key value pairs to be added or removed from the btree.
pub fn update(
    commit_seqn: u32,
    changeset: BTreeMap<Key, Option<Vec<u8>>>,
    root: BranchId,
    bnp: &mut branch::BranchNodePool,
    leaf_store: &mut leaf::LeafStoreTx,
) -> Result<BranchId> {
    let _ = (commit_seqn, changeset, root, bnp, leaf_store);
    todo!();
}

/// Reconstruct the upper branch nodes of the btree from the bottom branch nodes and the leaf nodes.
/// Returns the root of the reconstructed btree.
pub fn reconstruct(bn_fd: File, bnp: &mut branch::BranchNodePool) -> Result<BranchId> {
    let _ = (bn_fd, bnp);
    todo!();
}

enum NodePointer {
    Branch(BranchId),
    Leaf(LeafPn),
}

/// Binary search a branch node for the child node containing the key.
fn search_branch(branch: &branch::BranchNode, key: Key) -> Option<NodePointer> {
    let prefix = branch.prefix();

    if key.view_bits::<Lsb0>()[..prefix.len()] != prefix {
        return None;
    }

    let post_key =
        &key.view_bits::<Lsb0>()[prefix.len()..prefix.len() + branch.separator_len() as usize];

    // ignore two endpoint separators, which are special and only used to indicate the key range of
    // BBNs.
    let mut low = 1;
    let mut high = branch.n() as usize;
    while low < high {
        let mid = low + (high - low) / 2;
        if post_key <= branch.separator(mid) {
            high = mid;
        } else {
            low = mid + 1;
        }
    }

    let node_pointer = branch.node_pointer(high - 1);
    Some(if branch.is_bbn() {
        NodePointer::Leaf(node_pointer.into())
    } else {
        NodePointer::Branch(node_pointer.into())
    })
}
