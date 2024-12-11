//! Database iterators over the Beatree.

// TODO: remove this once used.
#![allow(unused)]

use std::{
    cmp::Ordering,
    ops::{Range, RangeFrom},
    sync::Arc,
};

use imbl::ordmap::{Iter as OrdMapIter, OrdMap};
use nomt_core::trie::ValueHash;

use super::{
    allocator::PageNumber,
    branch::node::{get_key, BranchNode},
    index::Index,
    leaf::node::LeafNode,
    Key, ReadTransaction, ValueChange,
};

/// An iterator over the state of the beatree at some particular point.
///
/// This combines the in-memory overlays with the state of the leaf pages on the disk.
/// This iterator does not handle the fetching of pages internally, but instead provides the needed
/// page numbers. The [`ReadTransaction`]` will provide facilities for dispatching I/O with a
/// provided handle.
///
/// This is not a normal Rust iterator, due to its need to block. Furthermore, it is a streaming
/// iterator which does not clone or copy its outputs, rather returning them as borrowed. This means
/// that the standard iterator combinators can't be used with it, making it less versatile than a
/// typical Rust iterator.
pub struct BeatreeIterator {
    memory_values: StagingIterator,
    leaf_values: LeafIterator,
}

impl BeatreeIterator {
    pub(super) fn new(tx: &ReadTransaction, start: Key, end: Option<Key>) -> Self {
        BeatreeIterator {
            memory_values: StagingIterator::new(tx, start, end),
            leaf_values: LeafIterator::new(tx.bbn_index.clone(), start, end),
        }
    }

    /// Get the next value from the iterator.
    ///
    /// This may either return a value or a marker indicating that it is blocked on a page load.
    /// The page to load can be determined with [`BeatreeIterator::next_pns`].
    /// If `None` is returned, it indicates that the iterator is exhausted.
    ///
    /// Values can take the form of regular values (stored in-line in the leaf) or overflow values,
    /// which only store metadata about the value.
    pub fn next<'a>(&'a mut self) -> Option<IterOutput<'a>> {
        enum Action {
            TakeLeaf,
            TakeMemory,
        }

        let action = loop {
            match (self.leaf_values.peek_key(), self.memory_values.peek()) {
                (None, None) => return None,
                (Some(_), None) => break Action::TakeLeaf,
                (None, Some((_, ValueChange::Delete))) => {
                    // skip deleted in-memory values unless they correspond to a leaf value.
                    let _ = self.memory_values.next();
                    continue;
                }
                (None, Some(_)) => break Action::TakeMemory,
                (Some((leaves_key, leaf_pending)), Some((memory_key, memory_value))) => {
                    match (memory_key.cmp(&leaves_key), memory_value) {
                        (Ordering::Less, ValueChange::Delete) => {
                            // skip deleted in-memory values that don't correspond to a leaf value.
                            // this can happen when the item was inserted in the earlier overlay,
                            // having never existed on-disk, and then deleted also in memory.
                            // we can just skip these safely.
                            let _ = self.memory_values.next();
                            continue;
                        }
                        (Ordering::Less, _) => {
                            // the next memory key is before the next leaf key. take the memory
                            // value.
                            break Action::TakeMemory;
                        }
                        (Ordering::Equal, _) if leaf_pending => {
                            // the keys are equal, but the leaf is pending so we are blocked
                            // until the next page is supplied.
                            return Some(IterOutput::Blocked);
                        }
                        (Ordering::Equal, ValueChange::Delete) => {
                            // skip both values if they are equal but the in-memory version has
                            // been deleted.
                            let _ = self.leaf_values.next();
                            let _ = self.memory_values.next();
                            continue;
                        }
                        (Ordering::Equal, _) => {
                            // skip the leaf value if they are equal but the in-memory version
                            // exists.
                            let _ = self.leaf_values.next();
                            break Action::TakeMemory;
                        }
                        (Ordering::Greater, _) => {
                            // the next leaf key is before the next memory key. take the leaf value.
                            break Action::TakeLeaf;
                        }
                    }
                }
            }
        };

        match action {
            Action::TakeLeaf => self.leaf_values.next(),
            Action::TakeMemory => match self.memory_values.next().unwrap() {
                // PANIC: this case is checked previously.
                (_, ValueChange::Delete) => panic!(),
                (k, ValueChange::Insert(val)) => return Some(IterOutput::Item(*k, val)),
                (k, ValueChange::InsertOverflow(ref overflow_cell, ref value_hash)) => {
                    return Some(IterOutput::OverflowItem(*k, *value_hash, overflow_cell))
                }
            },
        }
    }

    /// Get an iterator over the next leaf page numbers needed by the iterator.
    ///
    /// You can call this at any point during the iterator's lifetime and it will always return a
    /// valid iterator of all pages which may need to be loaded, in order. Any pages which have
    /// been supplied with [`BeatreeIterator::provide_leaf`] will not be present in this iterator.
    ///
    /// Care must be taken when loading pages; if the read transaction used to create this iterator
    /// is no longer live, it is likely that garbage data will be read from the disk.
    pub fn needed_leaves(&self) -> NeededLeavesIter {
        self.leaf_values.needed_leaves()
    }

    /// Provide a leaf to the iterator. This will panic if the iterator is not waiting on a leaf,
    /// i.e. if it has not returned [`IterOutput::Blocked`].
    ///
    /// This does no checking of whether the provided page is actually the correct one. GIGO.
    pub fn provide_leaf(&mut self, leaf: Arc<LeafNode>) {
        self.leaf_values.provide_leaf(leaf);
    }
}

/// The output of the iterator.
pub enum IterOutput<'a> {
    // The iterator is blocked and needs a new leaf to be supplied.
    Blocked,
    // The iterator has produced a new item.
    Item(Key, &'a [u8]),
    // The iterator has produced a new overflow item. The slice here is the entire overflow cell.
    OverflowItem(Key, ValueHash, &'a [u8]),
}

struct CurrentLeaf {
    index: usize,
    leaf: Arc<LeafNode>,
    separator: Key,
}

impl CurrentLeaf {
    fn is_consumed(&self) -> bool {
        self.index == self.leaf.n()
    }

    fn is_in_range(&self, end: Option<&Key>) -> bool {
        end.map_or(true, |end| &self.leaf.key(self.index) < end)
    }

    fn last_output(&self) -> IterOutput {
        let index = self.index - 1;
        let key = self.leaf.key(index);
        let (cell, overflow) = self.leaf.value(index);
        let (_, value_hash, _) = super::leaf::overflow::decode_cell(cell);
        if overflow {
            IterOutput::OverflowItem(key, value_hash, cell)
        } else {
            IterOutput::Item(key, cell)
        }
    }
}

struct LeafIterator {
    index: Index,
    state: LeafIteratorState,
    // `None` means "past the start key".
    start: Option<Key>,
    end: Option<Key>,
}

impl LeafIterator {
    fn new(index: Index, start: Key, end: Option<Key>) -> Self {
        let mut iter = LeafIterator {
            index,
            state: LeafIteratorState::Done { last: None },
            start: Some(start),
            end,
        };

        let Some((_, branch)) = iter.index.lookup(start) else {
            return iter;
        };

        let Some((index_in_branch, pn)) = super::ops::search_branch(&branch, start) else {
            return iter;
        };

        let separator = get_key(&branch, index_in_branch);
        iter.state = LeafIteratorState::Blocked {
            branch,
            pn,
            index_in_branch,
            separator,
            last: None,
        };
        iter
    }

    // the bool, if `true`, indicates that the key is pending (a new leaf is needed).
    fn peek_key(&mut self) -> Option<(Key, bool)> {
        match self.state {
            LeafIteratorState::Done { .. } => None,
            LeafIteratorState::Blocked { ref separator, .. } => Some((*separator, true)),
            LeafIteratorState::Proceeding { ref current, .. } => {
                Some((current.leaf.key(current.index), false))
            }
        }
    }

    fn next<'a>(&'a mut self) -> Option<IterOutput<'a>> {
        // This is a so-called "Streaming Iterator", where the items borrow the lifetime
        // of the callsite rather than consuming items from the object - these are somewhat awkward
        // because we first have to advance the state of the iterator and then return a borrow
        // from the new state.
        //
        // we need to first point the state at the next item and then borrow it out of the advanced
        // state.
        //
        // result: None => None, Some(false) => blocked, Some(true) => take last.
        let (new_state, result) =
            match std::mem::replace(&mut self.state, LeafIteratorState::Done { last: None }) {
                s @ LeafIteratorState::Done { .. } => (s, None),
                s @ LeafIteratorState::Blocked { .. } => (s, Some(false)),
                LeafIteratorState::Proceeding {
                    branch,
                    index_in_branch,
                    mut current,
                } => {
                    current.index += 1;
                    let next_state = if current.is_consumed() {
                        // move to next leaf.
                        self.new_state_leaf_consumed(branch, index_in_branch + 1, current)
                    } else if !current.is_in_range(self.end.as_ref()) {
                        // iterator is done
                        LeafIteratorState::Done {
                            last: Some(current),
                        }
                    } else {
                        // iterator is still proceeding
                        LeafIteratorState::Proceeding {
                            branch,
                            index_in_branch,
                            current,
                        }
                    };

                    (next_state, Some(true))
                }
            };

        // Now that we've advanced the state, we borrow the current value out of it.
        self.state = new_state;
        match result {
            None => None,
            Some(false) => Some(IterOutput::Blocked),
            Some(true) => match self.state {
                LeafIteratorState::Done { ref last } => last.as_ref().map(|x| x.last_output()),
                LeafIteratorState::Blocked { ref last, .. } => {
                    last.as_ref().map(|x| x.last_output())
                }
                LeafIteratorState::Proceeding { ref current, .. } => Some(current.last_output()),
            },
        }
    }

    fn new_state_leaf_consumed(
        &self,
        branch: Arc<BranchNode>,
        next_index_in_branch: usize,
        last: CurrentLeaf,
    ) -> LeafIteratorState {
        if branch.n() as usize == next_index_in_branch {
            // out of range. look up next.
            let next_key = self
                .index
                .next_key(get_key(&*branch, next_index_in_branch - 1));
            match next_key {
                None => LeafIteratorState::Done { last: Some(last) },
                Some(k) if self.end.as_ref().map_or(false, |end| end < &k) => {
                    LeafIteratorState::Done { last: Some(last) }
                }
                Some(k) => {
                    // UNWRAP: items returned in `next_key` always exist in index.
                    let (separator, branch) = self.index.lookup(k).unwrap();
                    LeafIteratorState::Blocked {
                        index_in_branch: 0,
                        pn: branch.node_pointer(0).into(),
                        separator,
                        branch,
                        last: Some(last),
                    }
                }
            }
        } else {
            LeafIteratorState::Blocked {
                index_in_branch: next_index_in_branch,
                pn: branch.node_pointer(next_index_in_branch).into(),
                separator: get_key(&branch, next_index_in_branch),
                branch,
                last: Some(last),
            }
        }
    }

    // Provide the next needed leaf. This does no verification of whether the leaf is actually
    // the one requested. Panics if the iterator is not expecting a leaf (has returned `Blocked`).
    fn provide_leaf(&mut self, leaf: Arc<LeafNode>) {
        // If this is the first leaf requested, we need to skip all the items that are less than
        // the iterator's range.
        let index = self.start.take().map_or(0, |start| {
            let cell_pointers = leaf.cell_pointers();
            let res = cell_pointers.binary_search_by(|cell_pointer| {
                let k = super::leaf::node::extract_key(cell_pointer);
                k.cmp(&start)
            });

            res.unwrap_or_else(|i| i)
        });

        let prev_state = std::mem::replace(&mut self.state, LeafIteratorState::Done { last: None });
        let LeafIteratorState::Blocked {
            branch,
            pn,
            index_in_branch,
            separator,
            ..
        } = prev_state
        else {
            // PANIC: part of the function's contract.
            panic!("No leaf expected in iterator")
        };

        let leaf = CurrentLeaf {
            index,
            leaf,
            separator,
        };

        self.state = if leaf.is_consumed() {
            self.new_state_leaf_consumed(branch, index_in_branch + 1, leaf)
        } else {
            LeafIteratorState::Proceeding {
                branch,
                index_in_branch,
                current: leaf,
            }
        };
    }

    fn needed_leaves(&self) -> NeededLeavesIter {
        let iter_state = match self.state {
            LeafIteratorState::Blocked {
                ref branch,
                index_in_branch,
                ..
            } => Some((branch.clone(), index_in_branch)),
            LeafIteratorState::Proceeding {
                ref branch,
                index_in_branch,
                ..
            } => Some((branch.clone(), index_in_branch + 1)),
            LeafIteratorState::Done { .. } => None,
        };

        let iter_state = iter_state.map(|(branch, start_index)| {
            let range = start_index..branch.n() as usize;
            (branch, range)
        });

        NeededLeavesIter {
            index: self.index.clone(),
            state: iter_state,
            end: self.end,
        }
    }
}

/// An iterator over the leaf page numbers needed by the DB iterator.
pub struct NeededLeavesIter {
    index: Index,
    state: Option<(Arc<BranchNode>, Range<usize>)>,
    end: Option<Key>,
}

impl Iterator for NeededLeavesIter {
    type Item = PageNumber;

    fn next(&mut self) -> Option<PageNumber> {
        let Some((branch, mut range)) = self.state.take() else {
            return None;
        };
        if let Some(i) = range.next() {
            let key = get_key(&branch, i);
            let pn = branch.node_pointer(i).into();
            if self.end.as_ref().map_or(true, |end| &key < end) {
                self.state = Some((branch, range));
                Some(pn)
            } else {
                None
            }
        } else {
            let last_separator = get_key(&branch, branch.n() as usize - 1);
            let next_separator = match self.index.next_key(last_separator) {
                None => return None,
                Some(k) => k,
            };

            // UNWRAP: keys returned by `next_key` always exist; qed.
            let next_branch = self.index.lookup(next_separator).unwrap().1;
            let range = 0..next_branch.n() as usize;
            self.state = Some((next_branch, range));
            self.next()
        }
    }
}

enum LeafIteratorState {
    Blocked {
        branch: Arc<BranchNode>,
        pn: PageNumber,
        index_in_branch: usize,
        separator: Key,
        // this ensures that borrows can be kept valid.
        last: Option<CurrentLeaf>,
    },
    Proceeding {
        branch: Arc<BranchNode>,
        index_in_branch: usize,
        current: CurrentLeaf,
    },
    Done {
        last: Option<CurrentLeaf>,
    },
}

struct StagingIterator {
    primary: OrdMapOwnedIter,
    secondary: Option<OrdMapOwnedIter>,
}

impl StagingIterator {
    fn new(tx: &ReadTransaction, start: Key, end: Option<Key>) -> Self {
        StagingIterator {
            primary: OrdMapOwnedIter::new(tx.primary_staging.clone(), start, end),
            secondary: tx
                .secondary_staging
                .clone()
                .map(|s| OrdMapOwnedIter::new(s, start, end)),
        }
    }

    fn peek<'a>(&'a mut self) -> Option<(&'a Key, &'a ValueChange)> {
        let primary_peek = self.primary.peek();
        let secondary_peek = self.secondary.as_mut().and_then(|s| s.peek());

        match (primary_peek, secondary_peek) {
            (None, None) => None,
            (Some(x), None) | (None, Some(x)) => Some(x),
            (Some(primary), Some(secondary)) => {
                if primary.0 <= secondary.0 {
                    // if equal, favor the primary (more recent) staging map.
                    Some(primary)
                } else {
                    Some(secondary)
                }
            }
        }
    }

    fn next<'a>(&'a mut self) -> Option<(&'a Key, &'a ValueChange)> {
        let primary_peek = self.primary.next();
        let secondary_peek = self.secondary.as_mut().and_then(|s| s.next());

        match (primary_peek, secondary_peek) {
            (None, None) => None,
            (Some(x), None) | (None, Some(x)) => Some(x),
            (Some(primary), Some(secondary)) => {
                if primary.0 <= secondary.0 {
                    Some(primary)
                } else {
                    Some(secondary)
                }
            }
        }
    }
}

// This lets us do a range iteration over an `OrdMap` in an owned manner, as a streaming iterator.
struct OrdMapOwnedIter {
    _map: OrdMap<Key, ValueChange>,
    iter: std::iter::Peekable<OrdMapIter<'static, Key, ValueChange>>,
}

impl OrdMapOwnedIter {
    fn peek<'a>(&'a mut self) -> Option<(&'a Key, &'a ValueChange)> {
        self.iter.peek().map(|x| (x.0, x.1))
    }

    fn next<'a>(&'a mut self) -> Option<(&'a Key, &'a ValueChange)> {
        self.iter.peek().map(|x| (x.0, x.1))
    }
}

impl OrdMapOwnedIter {
    fn new(map: OrdMap<Key, ValueChange>, start: Key, end: Option<Key>) -> Self {
        let iter = if let Some(end) = end {
            map.range(Range { start, end })
        } else {
            map.range(RangeFrom { start })
        };

        // hack: an owned cursor would be very, very useful instead of doing this.
        // SAFETY: the OrdMap is kept alive and not used mutably during the lifetime of this
        // struct.
        let iter: OrdMapIter<'static, Key, ValueChange> = unsafe { std::mem::transmute(iter) };
        OrdMapOwnedIter {
            _map: map,
            iter: iter.peekable(),
        }
    }
}
