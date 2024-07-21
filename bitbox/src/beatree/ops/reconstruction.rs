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
use std::{collections::{BTreeMap, HashMap}, fs::File, io::{ErrorKind, Read, Seek}};

use crate::beatree::{
    branch::{self, BRANCH_NODE_SIZE, BranchId},
    index::Index,
    leaf,
    Key,
};

// 256K
const SEQ_READ_CHUNK_LEN: usize = BRANCH_NODE_SIZE * 64;
type SeqReadChunk = [u8; SEQ_READ_CHUNK_LEN];

/// Reconstruct the upper branch nodes of the btree from the bottom branch nodes and the leaf nodes.
/// This places all branches into the BNP and returns an index into all BBNs. 
pub fn reconstruct(
    sync_seqn: u32,
    bn_fd: File, 
    bnp: &mut branch::BranchNodePool,
) -> Result<Index> {
    let mut branch_seqns: HashMap<u64, (u32, Option<(Key, BranchId)>)> = HashMap::new();
    let mut index = Index::default();
    let mut displaced = Vec::new();

    for seq_chunk in read_sequential(bn_fd)? {
        let seq_chunk = seq_chunk?;
        let nodes = seq_chunk.chunks(BRANCH_NODE_SIZE);
        for node in nodes {
            let view = branch::BranchNodeView::from_slice(node);
            if view.sync_seqn() > sync_seqn { continue }

            // handle empty.
            if view.n() == 0 && node == [0; BRANCH_NODE_SIZE] {
                continue
            }

            if let Some((existing_sync_seqn, separator)) = branch_seqns.get(&view.bbn_seqn()) {
                if *existing_sync_seqn < view.sync_seqn() {
                    if let Some((separator, expected_branch)) = *separator {
                        // remove the previous branch, handling a corner case where it's already
                        // been displaced by some other branch that shares the exact same
                        // separator.
                        // UNWRAP: indices always correspond to `Some` entries.
                        let branch = index.remove(&separator).unwrap();
                        if branch != expected_branch {
                            index.insert(separator, branch);
                        } else {
                            bnp.release(branch);
                        }
                    }
                } else {
                    continue
                }
            }

            let branch_info = if view.n() == 0 {
                // tombstone
                None
            } else {
                let new_branch_id = bnp.allocate();

                // UNWRAP: just allocated
                bnp.checkout(new_branch_id).unwrap().as_mut_slice().copy_from_slice(node);

                let mut separator = [0u8; 32];
                {
                    let prefix = view.prefix();
                    let mut separator = separator.view_bits_mut::<Lsb0>();
                    separator[..prefix.len()].copy_from_bitslice(prefix);
                    let first = view.separator(0);
                    separator[prefix.len() .. prefix.len() + first.len()].copy_from_bitslice(first);
                }

                if let Some(displaced_branch_id) = index.insert(separator, new_branch_id) {
                    // UNWRAP: previously allocated, never released.
                    let displaced_branch = bnp.checkout(displaced_branch_id).unwrap();
                    displaced.push((
                        displaced_branch.bbn_seqn(), 
                        displaced_branch_id,
                    ));
                }

                Some((separator, new_branch_id))
            };

            branch_seqns.insert(view.bbn_seqn(), (view.sync_seqn(), branch_info));
        }
    }

    // after iterating all BBNs, reinstate any displaced BBNs which are still relevant.
    for (displaced_bbn_seqn, displaced_branch_id) in displaced {
        if let Some((separator, _)) = branch_seqns
            .get(&displaced_bbn_seqn)
            .and_then(|x| x.1)
            .filter(|x| x.1 == displaced_branch_id)
        {
            let _ = index.insert(separator, displaced_branch_id);
        } else {
            bnp.release(displaced_branch_id);
        }
    }

    Ok(index)
}

fn read_sequential(
    mut bn_fd: File,
) -> Result<Receiver<Result<Box<SeqReadChunk>>>> {
    let (tx, rx) = crossbeam_channel::unbounded();
    bn_fd.seek(std::io::SeekFrom::Start(BRANCH_NODE_SIZE as u64))?;
    std::thread::spawn(move || loop {
        let mut buf = Box::new([0; SEQ_READ_CHUNK_LEN]);
        match bn_fd.read_exact(&mut *buf) {
            Ok(()) => { let _ = tx.send(Ok(buf)); },
            Err(e) if e.kind() == ErrorKind::UnexpectedEof => {
                let _ = tx.send(Ok(buf));
                break
            }
            Err(e) => {
                let _ = tx.send(Err(anyhow::Error::from(e)));
                break
            }
        }
    });
    Ok(rx)
}
