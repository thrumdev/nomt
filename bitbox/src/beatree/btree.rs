use std::fs::File;
use anyhow::Result;

use super::{
    branch::{self, BranchId},
    leaf,
};

/// Lookup a key in the btree.
pub fn lookup(
    key: Vec<u8>,
    root: BranchId,
    branch_node_pool: &branch::BranchNodePool,
    leaf_store: &leaf::LeafStore,
) -> Result<Option<Vec<u8>>> {
    let _ = (key, root, branch_node_pool, leaf_store);
    todo!();
}

/// Change the btree in the specified way. Returns the root of the new btree.
///
/// The changeset is a list of key value pairs to be added or removed from the btree.
pub fn update(
    commit_seqn: u32,
    changeset: &[(Vec<u8>, Option<Vec<u8>>)],
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
