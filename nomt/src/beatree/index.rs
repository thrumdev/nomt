//! In-memory index tracking bottom level branch nodes. This is an immutable data structure,
//! which is cheaply cloneable in O(1) and performs COW operations.

use std::iter::DoubleEndedIterator;
use std::ops::{Bound, RangeBounds, RangeToInclusive};

use im::OrdMap;

use super::{branch::BranchId, Key};

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
        self.first_key_map
            .range(RangeToInclusive { end: key })
            .next_back()
            .map(|(_sep, b)| b.clone())
    }

    /// Get the first branch in the index by key.
    pub fn first(&self) -> Option<(Key, BranchId)> {
        self.first_key_map.iter().next().map(|(k, b)| (*k, *b))
    }

    /// Get the first branch with separator greater than the given key.
    pub fn next_after(&self, key: Key) -> Option<(Key, BranchId)> {
        self.first_key_map
            .range(RangeFromExclusive { start: key })
            .next()
            .map(|(k, b)| (*k, *b))
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

struct RangeFromExclusive {
    start: Key,
}

impl RangeBounds<Key> for RangeFromExclusive {
    fn start_bound(&self) -> Bound<&Key> {
        Bound::Excluded(&self.start)
    }

    fn end_bound(&self) -> Bound<&Key> {
        Bound::Unbounded
    }

    fn contains<U>(&self, item: &U) -> bool
    where
        U: PartialOrd<Key> + ?Sized,
    {
        item > &self.start
    }
}
