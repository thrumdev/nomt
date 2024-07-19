//! Reconstruction of the in-memory B-Tree from a file.
//!
//! Algorithm sketch: 
//!   1. Read in all BBNs from the file. Order them in an in-memory std BTree by the first separator.
//!     Skip future BBNs according to the commit sequence number.
//!   2. Iterate over BBNs in order, building next level of branches with targeted 75% fullness.
//!     Create a sorted list of this next layer of BNs as we go.
//!   3. Repeat step 2 until the root node is created.

use anyhow::Result;
use crossbeam_channel::{Sender, Receiver};
use bitvec::prelude::*;
use std::{collections::{BTreeMap, HashMap}, fs::File};

use crate::beatree::{
    branch::{self, BRANCH_NODE_SIZE, BranchId},
    leaf,
};

// 256K
type SeqReadChunk = [u8; BRANCH_NODE_SIZE * 64];

/// Reconstruct the upper branch nodes of the btree from the bottom branch nodes and the leaf nodes.
/// Returns the root of the reconstructed btree.
pub fn reconstruct(
    sync_seqn: u32,
    bn_fd: File, 
    bnp: &mut branch::BranchNodePool,
) -> Result<BranchId> {
    let bbns = read_bbns(sync_seqn, bn_fd, bnp);
    todo!()
}

fn read_bbns(
    sync_seqn: u32,
    bn_fd: File,
    bnp: &mut branch::BranchNodePool,
) -> Vec<(BitVec<u8>, BranchId)> {
    let mut branch_seqns: HashMap<u64, (u32, Option<usize>)> = HashMap::new();
    let mut branch_meta: Vec<Option<(BitVec<u8>, BranchId)>> = Vec::new();

    for seq_chunk in read_sequential(bn_fd) {
        let nodes = seq_chunk.chunks(BRANCH_NODE_SIZE);
        for node in nodes {
            // TODO: handle empty

            let view = branch::BranchNodeView::from_slice(node);
            if view.sync_seqn() > sync_seqn { continue }

            if let Some((existing_sync_seqn, branch_meta_index)) = branch_seqns.get(&view.bbn_seqn()) {
                if *existing_sync_seqn < view.sync_seqn() {
                    if let Some(branch_meta_index) = *branch_meta_index {
                        // UNWRAP: indices always correspond to `Some` entries.
                        let (_, branch_id) = branch_meta[branch_meta_index].take().unwrap();
                        bnp.release(branch_id);
                    }
                } else {
                    continue
                }
            }

            let branch_meta_index = if view.n() == 0 {
                // tombstone
                None
            } else {
                let new_branch_id = bnp.allocate();

                // UNWRAP: just allocated
                bnp.checkout(new_branch_id).unwrap().as_mut_slice().copy_from_slice(node);
                branch_meta.push(Some((todo!(), new_branch_id))); // TODO: first separator.
                Some(branch_meta.len())
            };

            branch_seqns.insert(view.bbn_seqn(), (view.sync_seqn(), branch_meta_index));
        }
    }

    // note: we expect the vector here to have a couple of million entries here at maximum
    // (2.5M * 4096) = 10G . With 1B leaves (4TB) this is a branching factor of 400 expected.
    // This takes a couple of seconds when the database is at absolute maximum capacity.
    let mut branch_meta = branch_meta.into_iter().flatten().collect::<Vec<_>>();
    branch_meta.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    branch_meta
}

fn read_sequential(
    bn_fd: File,
) -> Receiver<Box<SeqReadChunk>> {
    let (tx, rx) = crossbeam_channel::unbounded();
    std::thread::spawn(move || {
        // TODO read entire file sequentially while sending chunks out.
    });
    rx
}
