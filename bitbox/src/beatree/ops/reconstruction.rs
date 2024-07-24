//! Reconstruction of the in-memory B-Tree from a file.
//!
//! Algorithm sketch:
//!   1. Read in all BBNs from the file. Order them in an in-memory std BTree by the first separator.
//!     Skip future BBNs according to the commit sequence number.
//!   2. Iterate over BBNs in order, building next level of branches with targeted 75% fullness.
//!     Create a sorted list of this next layer of BNs as we go.
//!   3. Repeat step 2 until the root node is created.

use anyhow::{bail, ensure, Result};
use bitvec::prelude::*;
use crossbeam_channel::Receiver;
use std::{
    collections::BTreeSet,
    fs::File,
    io::{ErrorKind, Read, Seek},
};

use crate::beatree::{
    allocator::PageNumber,
    branch::{self, BRANCH_NODE_SIZE},
    index::Index,
};

// 256K
const SEQ_READ_CHUNK_LEN: usize = BRANCH_NODE_SIZE * 64;
type SeqReadChunk = [u8; SEQ_READ_CHUNK_LEN];

/// Reconstruct the upper branch nodes of the btree from the bottom branch nodes and the leaf nodes.
/// This places all branches into the BNP and returns an index into all BBNs.
pub fn reconstruct(
    bn_fd: File,
    bnp: &mut branch::BranchNodePool,
    bbn_freelist: &BTreeSet<PageNumber>,
    bump: PageNumber,
) -> Result<Index> {
    let mut index = Index::default();

    let mut pn = 0u32;
    for seq_chunk in read_sequential(bn_fd)? {
        let seq_chunk = seq_chunk?;
        let nodes = seq_chunk.chunks(BRANCH_NODE_SIZE);
        for node in nodes {
            if pn >= bump.0 {
                // Exceeded last possible valid page
                return Ok(index);
            }

            let view = branch::BranchNodeView::from_slice(node);
            ensure!(view.bbn_pn() == pn, "pn mismatch");
            if bbn_freelist.contains(&view.bbn_pn().into()) {
                continue;
            }

            if view.n() == 0 && node == [0; BRANCH_NODE_SIZE] {
                bail!("zero-length branch node")
            }

            let new_branch_id = bnp.allocate();

            // UNWRAP: just allocated
            bnp.checkout(new_branch_id)
                .unwrap()
                .as_mut_slice()
                .copy_from_slice(node);

            let mut separator = [0u8; 32];
            {
                let prefix = view.prefix();
                let separator = separator.view_bits_mut::<Lsb0>();
                separator[..prefix.len()].copy_from_bitslice(prefix);
                let first = view.separator(0);
                separator[prefix.len()..prefix.len() + first.len()].copy_from_bitslice(first);
            }

            if let Some(_) = index.insert(separator, new_branch_id) {
                bail!("2 branch nodes with same separator")
            }

            pn += 1;
        }
    }

    Ok(index)
}

fn read_sequential(mut bn_fd: File) -> Result<Receiver<Result<Box<SeqReadChunk>>>> {
    let (tx, rx) = crossbeam_channel::unbounded();
    bn_fd.seek(std::io::SeekFrom::Start(BRANCH_NODE_SIZE as u64))?;
    std::thread::spawn(move || loop {
        let mut buf = Box::new([0; SEQ_READ_CHUNK_LEN]);
        match bn_fd.read_exact(&mut *buf) {
            Ok(()) => {
                let _ = tx.send(Ok(buf));
            }
            Err(e) if e.kind() == ErrorKind::UnexpectedEof => {
                let _ = tx.send(Ok(buf));
                break;
            }
            Err(e) => {
                let _ = tx.send(Err(anyhow::Error::from(e)));
                break;
            }
        }
    });
    Ok(rx)
}
