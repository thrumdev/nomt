//! A utility for managing in-memory overlays.
//!
//! This module exposes two types. The user-facing [`Overlay`] is opaque and frozen. The internal
//! [`LiveOverlay`] is meant to be used within a session only.
//!
//! Overlays contain weak references to all their ancestors. This allows ancestors to be dropped or
//! committed during the lifetime of the overlay. Importantly, this means that memory is cleaned
//! up gracefully as overlays are dropped and committed.
//!
//! However, creating a new [`LiveOverlay`] requires the user to provide strong references to each
//! of the ancestors which are still alive. It is still the user's responsibility to ensure
//! that all live ancestors are provided, or else data will go silently missing.
//!
//! Looking up a value in an overlay does not have a varying cost. First there is a look-up in an
//! index to see which ancestor has the data, and then a query on the ancestor's storage is done.
//!
//! Creating a new overlay is an O(n) operation in the amount of changes relative to the parent,
//! both in terms of new changes and outdated ancestors.

use crate::{beatree::ValueChange, store::DirtyPage};
use nomt_core::{
    page_id::PageId,
    trie::{KeyPath, Node},
};

use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Weak,
};

/// An in-memory overlay of merkle tree and b-tree changes.
pub struct Overlay {
    inner: Arc<OverlayInner>,
}

impl Overlay {
    /// Get the merkle root at this overlay.
    pub fn root(&self) -> Node {
        self.inner.root
    }

    /// Check whether the parent of this overlay matches the provided marker.
    /// If the provided marker is `None`, then this checks that this overlay doesn't have a parent.
    pub(super) fn parent_matches_marker(&self, marker: Option<&OverlayMarker>) -> bool {
        match (self.inner.data.parent_status.as_ref(), marker) {
            (None, None) => true,
            (Some(parent), Some(marker)) => parent.ptr_eq(&marker.0),
            _ => false,
        }
    }

    /// Get the merkle page changes associated uniquely with this overlay.
    pub(super) fn page_changes(&self) -> &HashMap<PageId, DirtyPage> {
        &self.inner.data.pages
    }

    /// Get the merkle page changes associated uniquely with this overlay.
    pub(super) fn value_changes(&self) -> &HashMap<KeyPath, ValueChange> {
        &self.inner.data.values
    }

    /// Get the rollback delta associated uniquely with this overlay.
    pub(super) fn rollback_delta(&self) -> Option<&crate::rollback::Delta> {
        self.inner.rollback_delta.as_ref()
    }

    /// Mark the overlay as committed and return a marker.
    pub(super) fn commit(&self) -> OverlayMarker {
        let status = self.inner.data.status.clone();
        status.commit();
        OverlayMarker(status)
    }
}

struct OverlayInner {
    root: Node,
    index: Index,
    data: Arc<Data>,
    seqn: u64,
    // ordered by recency.
    ancestor_data: Vec<Weak<Data>>,
    rollback_delta: Option<crate::rollback::Delta>,
}

/// A marker indicating the overlay uniquely, until dropped. Used to enforce commit order.
pub(super) struct OverlayMarker(OverlayStatus);

#[derive(Clone)]
struct OverlayStatus(Arc<AtomicUsize>);

impl OverlayStatus {
    const LIVE: usize = 0;
    const DROPPED: usize = 1;
    const COMMITTED: usize = 2;

    fn new_live() -> Self {
        OverlayStatus(Arc::new(AtomicUsize::new(Self::LIVE)))
    }

    fn commit(&self) {
        self.0.store(Self::COMMITTED, Ordering::Relaxed);
    }

    fn drop(&self) {
        // If the overlay has not been committed, then we will mark it as dead.
        let _ = self.0.compare_exchange(
            Self::LIVE,
            Self::DROPPED,
            Ordering::Relaxed,
            Ordering::Relaxed,
        );
    }

    fn is_committed(&self) -> bool {
        self.0.load(Ordering::Relaxed) == Self::COMMITTED
    }

    fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

// Maps changes to sequence number.
#[derive(Default, Clone)]
struct Index {
    pages: imbl::HashMap<PageId, u64>,
    values: imbl::OrdMap<KeyPath, u64>,

    // sorted ascending by seqn.
    pages_by_seqn: imbl::Vector<(u64, PageId)>,
    values_by_seqn: imbl::Vector<(u64, KeyPath)>,
}

impl Index {
    // Prune all items with a sequence number less than the minimum.
    // O(n) in number of pruned items.
    fn prune_below(&mut self, min: u64) {
        loop {
            match self.pages_by_seqn.pop_front() {
                None => break,
                Some((seqn, key)) if seqn >= min => {
                    self.pages_by_seqn.push_front((seqn, key));
                    break;
                }
                Some((seqn, page_id)) => {
                    if let Some(got_seqn) = self
                        .pages
                        .remove(&page_id)
                        .filter(|&got_seqn| got_seqn != seqn && got_seqn >= min)
                    {
                        // page_id has been updated since this point. reinsert.
                        self.pages.insert(page_id, got_seqn);
                    }
                }
            }
        }

        loop {
            match self.values_by_seqn.pop_front() {
                None => break,
                Some((seqn, key)) if seqn >= min => {
                    self.values_by_seqn.push_front((seqn, key));
                    break;
                }
                Some((seqn, key)) => {
                    if let Some(got_seqn) = self
                        .values
                        .remove(&key)
                        .filter(|&got_seqn| got_seqn != seqn && got_seqn >= min)
                    {
                        // key has been updated since this point. reinsert.
                        self.values.insert(key, got_seqn);
                    }
                }
            }
        }
    }

    /// Insert all the value keys in the iterator with the given sequence number.
    ///
    /// The sequence number is assumed to be greater than or equal to the maximum in the vector.
    fn insert_pages(&mut self, seqn: u64, page_ids: impl IntoIterator<Item = PageId>) {
        for page_id in page_ids {
            self.pages_by_seqn.push_back((seqn, page_id.clone()));
            self.pages.insert(page_id, seqn);
        }
    }

    /// Insert all the value keys in the iterator with the given sequence number.
    ///
    /// The sequence number is assumed to be greater than or equal to the maximum in the vector.
    fn insert_values(&mut self, seqn: u64, value_keys: impl IntoIterator<Item = KeyPath>) {
        for key in value_keys {
            self.values_by_seqn.push_back((seqn, key));
            self.values.insert(key, seqn);
        }
    }
}

/// Data associated with a single overlay.
struct Data {
    pages: HashMap<PageId, DirtyPage>,
    values: HashMap<KeyPath, ValueChange>,
    status: OverlayStatus,
    parent_status: Option<OverlayStatus>,
}

impl Drop for Data {
    fn drop(&mut self) {
        self.status.drop();
    }
}

/// An error type indicating that the ancestors provided did not match.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InvalidAncestors {
    /// One of the provided ancestors was not actually an ancestor.
    NotAncestor,
    /// The ancestor chain is incomplete.
    Incomplete,
}

/// A live overlay which is being used as a child. This can be queried for all value/page changes in
/// any of the relevant ancestors.
#[derive(Clone)]
pub(super) struct LiveOverlay {
    parent: Option<Arc<OverlayInner>>,
    ancestor_data: Vec<Arc<Data>>,
    min_seqn: u64,
}

impl LiveOverlay {
    /// Create a new live overlay based on this iterator of ancestors.
    pub(super) fn new<'a>(
        live_ancestors: impl IntoIterator<Item = &'a Overlay>,
    ) -> Result<Self, InvalidAncestors> {
        let mut live_ancestors = live_ancestors.into_iter();
        let Some(parent) = live_ancestors.next().map(|p| p.inner.clone()) else {
            return Ok(LiveOverlay {
                parent: None,
                ancestor_data: Vec::new(),
                min_seqn: 0,
            });
        };

        let mut ancestor_data = Vec::new();
        for (supposed_ancestor, actual_ancestor) in live_ancestors.zip(parent.ancestor_data.iter())
        {
            let Some(actual_ancestor) = actual_ancestor.upgrade() else {
                return Err(InvalidAncestors::Incomplete);
            };

            if !Arc::ptr_eq(&supposed_ancestor.inner.data, &actual_ancestor) {
                return Err(InvalidAncestors::NotAncestor);
            }

            ancestor_data.push(actual_ancestor);
        }

        // verify that the chain is complete. The last ancestor's parent must either be `None` or
        // committed.
        if ancestor_data
            .last()
            .and_then(|a| a.parent_status.as_ref())
            .map_or(false, |status| !status.is_committed())
        {
            return Err(InvalidAncestors::Incomplete);
        }

        let min_seqn = parent.seqn - ancestor_data.len() as u64;

        Ok(LiveOverlay {
            parent: Some(parent),
            ancestor_data,
            min_seqn,
        })
    }

    /// Get a page by ID.
    ///
    /// `None` indicates that the page is not present in the overlay, not that the page doesn't
    /// exist.
    pub(super) fn page(&self, page_id: &PageId) -> Option<&DirtyPage> {
        self.parent
            .as_ref()
            .and_then(|parent| parent.index.pages.get(&page_id))
            .and_then(|seqn| seqn.checked_sub(self.min_seqn))
            .map(|seqn_diff| {
                if seqn_diff == 0 {
                    self.parent
                        .as_ref()
                        .unwrap() // UNWRAP: parent existence checked above
                        .data
                        .pages
                        .get(page_id)
                        .unwrap() // UNWRAP: index indicates that data exists.
                } else {
                    self.ancestor_data[seqn_diff as usize - 1]
                        .pages
                        .get(page_id)
                        .unwrap() // UNWRAP: index indicates that data exists.
                }
            })
    }

    /// Get a value change by ID.
    ///
    /// `None` indicates that the value has not changed in the overlay, not that the value doesn't
    /// exist.
    pub(super) fn value(&self, key: &KeyPath) -> Option<ValueChange> {
        self.parent
            .as_ref()
            .and_then(|parent| parent.index.values.get(key))
            .and_then(|seqn| seqn.checked_sub(self.min_seqn))
            .map(|seqn_diff| self.value_inner(key, seqn_diff))
            .cloned()
    }

    fn value_inner(&self, key: &KeyPath, seqn_diff: u64) -> &ValueChange {
        if seqn_diff == 0 {
            // UNWRAP: parent existence checked above
            // UNWRAP: index indicates that data exists.
            self.parent.as_ref().unwrap().data.values.get(key).unwrap()
        } else {
            // UNWRAP: index indicates that data exists.
            self.ancestor_data[seqn_diff as usize - 1]
                .values
                .get(key)
                .unwrap()
        }
    }

    /// Iterate all value changes within the given key bounds.
    pub(super) fn value_iter<'a>(
        &'a self,
        start: KeyPath,
        end: Option<KeyPath>,
    ) -> impl Iterator<Item = (KeyPath, &'a ValueChange)> {
        self.parent
            .as_ref()
            .map(move |parent| {
                parent
                    .index
                    .values
                    .range(start..)
                    .take_while(move |(k, _)| end.as_ref().map_or(true, |end| end > k))
                    .filter_map(|(k, seqn)| seqn.checked_sub(self.min_seqn).map(|s| (k, s)))
                    .map(|(k, seqn_diff)| (*k, self.value_inner(k, seqn_diff)))
            })
            .into_iter()
            .flatten()
    }

    /// Finish this overlay and transform it into a frozen [`Overlay`].
    pub(super) fn finish(
        self,
        root: Node,
        page_changes: HashMap<PageId, DirtyPage>,
        value_changes: HashMap<KeyPath, ValueChange>,
        rollback_delta: Option<crate::rollback::Delta>,
    ) -> Overlay {
        let new_seqn = self.parent.as_ref().map_or(0, |p| p.seqn + 1);

        // rebuild the index, including the new stuff, and excluding stuff from dead overlays.
        let mut index = self
            .parent
            .as_ref()
            .map_or_else(Default::default, |p| p.index.clone());
        index.prune_below(self.min_seqn);

        index.insert_pages(new_seqn, page_changes.keys().cloned());
        index.insert_values(new_seqn, value_changes.keys().cloned());

        let parent_status = self.parent.as_ref().map(|p| p.data.status.clone());
        let ancestor_data = self
            .parent
            .map(|parent| {
                std::iter::once(Arc::downgrade(&parent.data))
                    .chain(self.ancestor_data.into_iter().map(|d| Arc::downgrade(&d)))
                    .collect()
            })
            .unwrap_or_default();

        Overlay {
            inner: Arc::new(OverlayInner {
                index,
                root,
                data: Arc::new(Data {
                    pages: page_changes,
                    values: value_changes,
                    status: OverlayStatus::new_live(),
                    parent_status,
                }),
                seqn: new_seqn,
                ancestor_data,
                rollback_delta,
            }),
        }
    }
}
