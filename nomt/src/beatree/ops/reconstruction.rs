//! Reconstruction of the in-memory B-Tree from a file.
//!
//! Algorithm sketch:
//!   1. Read in all BBNs from the file. Order them in an in-memory std BTree by the first separator.
//!     Skip future BBNs according to the commit sequence number.
//!   2. Iterate over BBNs in order, building next level of branches with targeted 75% fullness.
//!     Create a sorted list of this next layer of BNs as we go.
//!   3. Repeat step 2 until the root node is created.

use anyhow::{bail, ensure, Ok, Result};
use bitvec::prelude::*;
use std::{collections::BTreeSet, fs::File, os::fd::AsRawFd, ptr};

use crate::beatree::{
    allocator::PageNumber,
    branch::{self, BranchNodeView, BRANCH_NODE_SIZE},
    index::Index,
};

/// Reconstruct the upper branch nodes of the btree from the bottom branch nodes and the leaf nodes.
/// This places all branches into the BNP and returns an index into all BBNs.
pub fn reconstruct(
    bn_fd: File,
    bnp: &mut branch::BranchNodePool,
    bbn_freelist_tracked: &BTreeSet<PageNumber>,
    bump: PageNumber,
) -> Result<Index> {
    let mut index = Index::default();

    let mut chunker = SeqFileReader::new(bn_fd, bump.0)?;
    while let Some((pn, node)) = chunker.next()? {
        let view = BranchNodeView::from_slice(node);

        if view.n() == 0 && node == [0; BRANCH_NODE_SIZE] {
            // Just skip empty nodes.
            continue;
        }

        if bbn_freelist_tracked.contains(&PageNumber(pn)) {
            continue;
        }

        ensure!(
            view.bbn_pn() == pn,
            "pn mismatch {} != {}",
            view.bbn_pn(),
            pn
        );

        let new_branch_id = bnp.allocate();

        // UNWRAP: just allocated
        bnp.checkout(new_branch_id)
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(node);

        let mut separator = [0u8; 32];
        {
            let prefix = view.prefix();
            let separator = separator.view_bits_mut::<Msb0>();
            separator[..prefix.len()].copy_from_bitslice(prefix);
            let first = view.separator(0);
            separator[prefix.len()..prefix.len() + first.len()].copy_from_bitslice(first);
        }

        if let Some(_) = index.insert(separator, new_branch_id) {
            bail!(
                "2 branch nodes with same separator, separator={:?}",
                separator
            );
        }
    }
    Ok(index)
}

/// An utility to read sequentially from a file.
///
/// This is backed by an mmap of the file. The kernel is instructed that the contents of the file
/// should be read sequentially. This will make the kernel to read ahead the file sequentially.
struct SeqFileReader {
    ptr: *mut u8,
    len: u64,
    pn: u32,
    bump: u32,
}

impl SeqFileReader {
    fn new(bbn_fd: File, bump: u32) -> Result<Self> {
        let len = bbn_fd.metadata()?.len();
        ensure!(
            len % BRANCH_NODE_SIZE as u64 == 0,
            "file size is not a multiple of 4KiB page"
        );
        ensure!(
            len >= BRANCH_NODE_SIZE as u64,
            "file is too small for BBN store"
        );

        let pn = 0u32;
        let ptr = unsafe {
            // MAP_PRIVATE
            //
            //     PRIVATE vs. SHARED should not matter much since we are only reading. However, opt
            //     for a private mapping because it would create a private mapping undisturbed from
            //     the rest of the system. Not that this matters much since we take the assumption that
            //     the file is under exclusive access of this process.
            let flags = libc::MAP_PRIVATE;
            let addr = libc::mmap(
                ptr::null_mut(),
                len as usize,
                libc::PROT_READ,
                flags,
                bbn_fd.as_raw_fd(),
                0,
            );
            if addr == libc::MAP_FAILED {
                bail!("mmap failed");
            }
            if libc::madvise(addr, len as usize, libc::MADV_SEQUENTIAL) != 0 {
                bail!("madvise failed"); // although this should not be fatal
            }
            addr as *mut u8
        };

        Ok(Self { ptr, len, pn, bump })
    }

    pub fn next<'a>(&'a mut self) -> Result<Option<(u32, &'a [u8])>> {
        if self.pn >= self.bump {
            return Ok(None);
        }

        let node = unsafe {
            let offset = self.pn as usize * BRANCH_NODE_SIZE;
            let ptr = self.ptr.offset(offset as isize);
            std::slice::from_raw_parts(ptr, BRANCH_NODE_SIZE)
        };

        let pn = self.pn;
        self.pn += 1;
        Ok(Some((pn, node)))
    }
}

impl Drop for SeqFileReader {
    fn drop(&mut self) {
        unsafe {
            let _ = libc::munmap(self.ptr as *mut _, self.len as usize);
        }
    }
}
