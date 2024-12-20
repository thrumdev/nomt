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

#![allow(dead_code)]

use crate::{beatree::ValueChange, page_cache::Page, page_diff::PageDiff};
use nomt_core::{page_id::PageId, trie::KeyPath};

use std::collections::HashMap;
use std::sync::{Arc, Weak};

/// An in-memory overlay of merkle tree and b-tree changes.
pub struct Overlay {
    inner: Arc<OverlayInner>,
}

struct OverlayInner {
    index: Index,
    data: Arc<Data>,
    seqn: u64,
    // ordered by recency.
    ancestor_data: Vec<Weak<Data>>,
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
    pages: HashMap<PageId, (Page, PageDiff)>,
    values: HashMap<KeyPath, ValueChange>,
}

/// An error type indicating that the ancestors provided did not match.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InvalidAncestors;

/// A live overlay which is being used as a child. This can be queried for all value/page changes in
/// any of the relevant ancestors.
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
                return Err(InvalidAncestors);
            };

            if !Arc::ptr_eq(&supposed_ancestor.inner.data, &actual_ancestor) {
                return Err(InvalidAncestors);
            }

            ancestor_data.push(actual_ancestor);
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
    pub(super) fn page(&self, page_id: &PageId) -> Option<(Page, PageDiff)> {
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
            .cloned()
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
            .map(|seqn_diff| {
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
            })
            .cloned()
    }

    /// Finish this overlay and transform it into a frozen [`Overlay`].
    pub(super) fn finish(
        self,
        page_changes: HashMap<PageId, (Page, PageDiff)>,
        value_changes: HashMap<KeyPath, ValueChange>,
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
                data: Arc::new(Data {
                    pages: page_changes,
                    values: value_changes,
                }),
                seqn: new_seqn,
                ancestor_data,
            }),
        }
    }
}
