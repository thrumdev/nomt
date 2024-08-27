//! This module defines the branch node pool.
//!
//! This is an in-memory structure that stores branch nodes irrespective of their level: it stores
//! both bottom-level and top-level branch nodes.
//!
//! The branch nodes are addressed by a branch ID, which is a 32-bit integer. Each branch node
//! is backed by a 4 KiB page.

use std::{
    ptr,
    sync::{Arc, Mutex},
};

pub use node::{body_size, BranchNode, BranchNodeBuilder, BranchNodeView, BRANCH_NODE_BODY_SIZE};

pub mod node;

/// The ID of a branch node in the node pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BranchId(u32);

impl From<u32> for BranchId {
    fn from(x: u32) -> Self {
        BranchId(x)
    }
}

pub const BRANCH_NODE_SIZE: usize = 4096;

/// Handle. Cheap to clone.
pub struct BranchNodePool {
    inner: Arc<Mutex<BranchNodePoolInner>>,
}

struct BranchNodePoolInner {
    /// The pointer to the beginning of the pool allocation.
    pool_base_ptr: *mut u8,

    /// The next index in the pool available for allocation.
    bump: u32,

    /// The list of the branch nodes that are ready to be reused.
    freelist: Vec<BranchId>,

    /// The list of the branch nodes that are currently checked out.
    checked_out: Vec<BranchId>,
}

impl Drop for BranchNodePoolInner {
    fn drop(&mut self) {
        unsafe {
            let res = libc::munmap(self.pool_base_ptr as *mut _, BNP_MMAP_SIZE as usize);
            if res != 0 {
                panic!("munmap failed");
            }
        }
    }
}

// Should be 16 TiB.
const BNP_MMAP_SIZE: u64 = BRANCH_NODE_SIZE as u64 * u32::MAX as u64;

impl BranchNodePool {
    pub fn new() -> BranchNodePool {
        let pool_base_ptr = unsafe {
            // On Linux this is ignored for MAP_ANON, but on macOS it's used to specify flags for
            // Mach VM, so be explicit here.
            let fd = -1;
            // MAP_NORESRVE is Linux only. Not sure what would happen on macOS. Using MAP_NORESERVE
            // is not a good idea anyway since any access to a page could cause a SEGFAULT.
            let flags = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE;
            libc::mmap(
                ptr::null_mut(),
                BNP_MMAP_SIZE as usize,
                libc::PROT_READ | libc::PROT_WRITE,
                flags,
                fd,
                0,
            )
        };
        if pool_base_ptr == libc::MAP_FAILED {
            panic!("mmap failed");
        }
        let pool_base_ptr = pool_base_ptr as *mut u8;
        BranchNodePool {
            inner: Arc::new(Mutex::new(BranchNodePoolInner {
                pool_base_ptr,
                bump: 0,
                freelist: Vec::new(),
                checked_out: Vec::new(),
            })),
        }
    }

    pub fn allocate(&self) -> BranchId {
        let mut inner = self.inner.lock().unwrap();
        if let Some(id) = inner.freelist.pop() {
            id
        } else {
            // allocate the index for the new node.
            let ix = inner.bump;
            inner.bump += 1;
            BranchId(ix)
        }
    }

    pub fn release(&self, id: BranchId) {
        let mut inner = self.inner.lock().unwrap();
        inner.freelist.push(id);
    }

    /// Returns the branch node with the given ID, if it exists.
    ///
    /// Note that this function expects the node to be allocated. If the node is not allocated,
    /// the behavior is unspecified.
    ///
    /// # Panics
    ///
    /// Panics if the node is already checked out.
    pub fn checkout(&self, id: BranchId) -> Option<BranchNode> {
        let mut inner = self.inner.lock().unwrap();
        if id.0 >= inner.bump {
            return None;
        }
        if inner.checked_out.contains(&id) {
            panic!();
        }
        inner.checked_out.push(id);
        unsafe {
            let offset = id.0 as usize * BRANCH_NODE_SIZE;
            let ptr = inner.pool_base_ptr.offset(offset as isize);
            Some(BranchNode {
                ptr,
                pool: self.inner.clone(),
                id,
            })
        }
    }
}

impl Clone for BranchNodePool {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

unsafe impl Send for BranchNodePool {}
unsafe impl Sync for BranchNodePool {}

#[test]
fn test_branch_node_pool() {
    let pool = BranchNodePool::new();
    let id = pool.allocate();
    let mut node = pool.checkout(id).unwrap();
    node.set_separator_len(10);
}

#[test]
fn test_branch_node_pool_release() {
    let pool = BranchNodePool::new();
    let id = pool.allocate();
    let node = pool.checkout(id).unwrap();
    drop(node);
    let _node = pool.checkout(id).unwrap();
}
