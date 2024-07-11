//! This module defines the branch node pool.
//! 
//! This is an in-memory structure that stores branch nodes irrespective of their level: it stores
//! both bottom-level and top-level branch nodes.
//! 
//! The branch nodes are addressed by a branch ID, which is a 32-bit integer. Each branch node
//! is backed by a 4 KiB page.


/// The ID of a branch node in the node pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BranchId(u32);

/// A branch node, regardless of its level.
pub struct BranchNode {
}

impl BranchNode {
}

pub struct BranchNodePool {
}

impl BranchNodePool {
    pub fn allocate(&self) -> BranchId {
        todo!()
    }

    pub fn release(&self, id: BranchId) {
        todo!()
    }

    /// Returns the branch node with the given ID, if it exists.
    pub fn query(&self, id: BranchId) -> Option<BranchNode> {
        todo!()
    }
}
