//! In-memory index tracking bottom level branch nodes. This is an immutable data structure,
//! which is cheaply cloneable in O(1) and performs COW operations.

use std::iter::DoubleEndedIterator;

use im::OrdMap;

use super::{
    branch::{self, BranchId},
    Key
};

#[derive(Default, Clone)]
pub struct Index {
    first_key_map: OrdMap<Key, BranchId>,
}

impl Index {
    /// Look up the branch that would store the given key.
    ///
    /// This is either a branch whose separator is exactly equal to this key or the branch with the
    /// highest separator less than the key.
    pub fn lookup(&self, key: Key) -> Option<BranchId> {
        self.first_key_map.range(std::ops::RangeToInclusive { end: key })
            .next_back()
            .map(|(_sep, b)| b.clone())
    } 

    /// Remove the branch with the given separator key.
    pub fn remove(&mut self, separator: &Key) -> Option<BranchId> {
        self.first_key_map.remove(separator)
    }

    /// Insert a branch with the given separator key.
    pub fn insert(&mut self, separator: Key, branch: BranchId) -> Option<BranchId> {
        self.first_key_map.insert(separator, branch)
    }
}
