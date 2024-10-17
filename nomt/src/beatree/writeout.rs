//! The writeout logic for beatree.

// As part of beatree writeout, we need to write BBN and LN files, resizing them to the correct
// size beforehand. After the writes are completed (fsync'd), we wait for the MANIFEST to be
// updated and then perform some cleanup.

use super::{allocator::PageNumber, branch::BranchNode};
use crate::io::{
    page_pool::{FatPage, UnsafePageView},
    IoHandle,
};
use std::{fs::File, os::fd::AsRawFd as _};

pub fn write_bbn(
    io_handle: IoHandle,
    bbn_fd: &File,
    bbn: Vec<BranchNode<UnsafePageView>>,
    bbn_free_list_pages: Vec<(PageNumber, FatPage)>,
    bbn_extend_file_sz: Option<u64>,
) -> anyhow::Result<()> {
    if let Some(new_len) = bbn_extend_file_sz {
        bbn_fd.set_len(new_len)?;
    }

    let mut sent = 0;
    for branch_node in bbn {
        let bbn_pn = branch_node.bbn_pn();
        let page_view = branch_node.into_inner();
        let ptr = page_view.as_ptr();
        let len = page_view.len();

        io_handle
            .send(crate::io::IoCommand {
                kind: crate::io::IoKind::WriteRaw(bbn_fd.as_raw_fd(), bbn_pn as u64, ptr, len),
                user_data: 0,
            })
            .unwrap();
        sent += 1;
    }

    for (pn, page) in bbn_free_list_pages {
        io_handle
            .send(crate::io::IoCommand {
                kind: crate::io::IoKind::Write(bbn_fd.as_raw_fd(), pn.0 as u64, page),
                user_data: 0,
            })
            .unwrap();
        sent += 1;
    }

    while sent > 0 {
        io_handle.recv().unwrap();
        sent -= 1;
    }

    bbn_fd.sync_all()?;
    Ok(())
}

pub fn write_ln(
    io_handle: IoHandle,
    ln_fd: &File,
    ln: Vec<(PageNumber, FatPage)>,
    ln_free_list_pages: Vec<(PageNumber, FatPage)>,
    ln_extend_file_sz: Option<u64>,
) -> anyhow::Result<()> {
    if let Some(new_len) = ln_extend_file_sz {
        ln_fd.set_len(new_len)?;
    }

    let mut sent = 0;
    for (pn, page) in ln {
        io_handle
            .send(crate::io::IoCommand {
                kind: crate::io::IoKind::Write(ln_fd.as_raw_fd(), pn.0 as u64, page),
                user_data: 0,
            })
            .unwrap();
        sent += 1;
    }

    for (pn, page) in ln_free_list_pages {
        io_handle
            .send(crate::io::IoCommand {
                kind: crate::io::IoKind::Write(ln_fd.as_raw_fd(), pn.0 as u64, page),
                user_data: 0,
            })
            .unwrap();
        sent += 1;
    }

    while sent > 0 {
        io_handle.recv().unwrap();
        sent -= 1;
    }

    ln_fd.sync_all()?;
    Ok(())
}
