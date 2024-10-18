use anyhow::Result;
use dashmap::DashMap;
use threadpool::ThreadPool;

use std::{collections::BTreeMap, ops::Deref};

use crate::beatree::{
    allocator::PageNumber,
    bbn,
    branch::{node::BranchNode, BRANCH_NODE_BODY_SIZE},
    index::Index,
    leaf::{
        node::{LEAF_NODE_BODY_SIZE, MAX_LEAF_VALUE_SIZE},
        overflow,
        store::{LeafStoreReader, LeafStoreWriter},
    },
    ops::bit_ops::reconstruct_key,
    Key,
};
use crate::io::page_pool::{Page, UnsafePageView};

mod branch_stage;
mod branch_updater;
mod extend_range_protocol;
mod leaf_stage;
mod leaf_updater;

// All nodes less than this body size will be merged with a neighboring node.
const BRANCH_MERGE_THRESHOLD: usize = BRANCH_NODE_BODY_SIZE / 2;

// At 180% of the branch size, we perform a 'bulk split' which follows a different algorithm
// than a simple split. Bulk splits are encountered when there are a large number of insertions
// on a single node, typically when inserting into a fresh database.
const BRANCH_BULK_SPLIT_THRESHOLD: usize = (BRANCH_NODE_BODY_SIZE * 9) / 5;
// When performing a bulk split, we target 75% fullness for all of the nodes we create except the
// last.
const BRANCH_BULK_SPLIT_TARGET: usize = (BRANCH_NODE_BODY_SIZE * 3) / 4;

const LEAF_MERGE_THRESHOLD: usize = LEAF_NODE_BODY_SIZE / 2;
const LEAF_BULK_SPLIT_THRESHOLD: usize = (LEAF_NODE_BODY_SIZE * 9) / 5;
const LEAF_BULK_SPLIT_TARGET: usize = (LEAF_NODE_BODY_SIZE * 3) / 4;

/// Change the btree in the specified way. Updates the branch index in-place.
///
/// The changeset is a list of key value pairs to be added or removed from the btree.
pub fn update(
    changeset: &BTreeMap<Key, Option<Vec<u8>>>,
    bbn_index: &mut Index,
    leaf_reader: &LeafStoreReader,
    leaf_writer: &mut LeafStoreWriter,
    bbn_writer: &mut bbn::BbnStoreWriter,
    thread_pool: ThreadPool,
    workers: usize,
) -> Result<Vec<Vec<Page>>> {
    let leaf_cache = preload_leaves(leaf_reader, &bbn_index, changeset.keys().cloned())?;

    let changeset = changeset
        .iter()
        .map(|(k, v)| match v {
            Some(v) if v.len() <= MAX_LEAF_VALUE_SIZE => (*k, Some((v.clone(), false))),
            Some(large_value) => {
                let pages = overflow::chunk(&large_value, leaf_writer);
                let cell = overflow::encode_cell(large_value.len(), &pages);
                (*k, Some((cell, true)))
            }
            None => (*k, None),
        })
        .collect::<_>();

    let (leaf_changes, overflow_deleted) = leaf_stage::run(
        &bbn_index,
        leaf_cache,
        leaf_reader,
        leaf_writer.page_pool().clone(),
        changeset,
        thread_pool.clone(),
        workers,
    );

    let branch_changeset = leaf_changes
        .into_iter()
        .map(|(key, leaf_entry)| {
            let leaf_pn = leaf_entry.inserted.map(|leaf| leaf_writer.write(leaf));
            if let Some(prev_pn) = leaf_entry.deleted {
                leaf_writer.release(prev_pn);
            }

            (key, leaf_pn)
        })
        .collect::<Vec<_>>();

    for overflow_cell in overflow_deleted {
        overflow::delete(&overflow_cell, leaf_reader, leaf_writer);
    }

    let (branch_changes, bbn_outdated_pages) = branch_stage::run(
        &bbn_index,
        leaf_writer.page_pool().clone(),
        branch_changeset,
        thread_pool,
        workers,
    );

    for (key, changed_branch) in branch_changes {
        match changed_branch.inserted {
            Some(mut node) => {
                bbn_writer.allocate(&mut node);
                let read_only_node = BranchNode::new(node.into_inner().into_shared());
                bbn_writer.write(read_only_node.clone());
                bbn_index.insert(key, read_only_node.into_inner().into_inner());
            }
            None => {
                bbn_index.remove(&key);
            }
        }

        if let Some(deleted_pn) = changed_branch.deleted {
            bbn_writer.release(deleted_pn);
        }
    }

    Ok(bbn_outdated_pages)
}

/// Extract the key at a given index from a BranchNode
pub fn get_key<T: Deref<Target = [u8]>>(node: &BranchNode<T>, index: usize) -> Key {
    let prefix = if index < node.prefix_compressed() as usize {
        Some(node.raw_prefix())
    } else {
        None
    };
    reconstruct_key(prefix, node.raw_separator(index))
}

// TODO: this should not be necessary with proper warm-ups.
fn preload_leaves(
    leaf_reader: &LeafStoreReader,
    bbn_index: &Index,
    keys: impl IntoIterator<Item = Key>,
) -> Result<DashMap<PageNumber, Page>> {
    let leaf_pages = DashMap::new();
    let mut last_pn = None;

    let mut submissions = 0;
    for key in keys {
        let Some((_, branch)) = bbn_index.lookup(key) else {
            continue;
        };
        // SAFETY: page pool is alive, pages in index are live and frozen.
        let view = unsafe { UnsafePageView::new(branch) };
        let branch = BranchNode::new(view);

        let Some((_, leaf_pn)) = super::search_branch(&branch, key) else {
            continue;
        };
        if last_pn == Some(leaf_pn) {
            continue;
        }
        last_pn = Some(leaf_pn);

        let page = leaf_reader.page_pool().alloc();

        // SAFETY: the page is kept alive and unaliased until the submission is reaped.
        let command = unsafe { leaf_reader.io_command(leaf_pn, 0, &page) };
        leaf_reader
            .io_handle()
            .send(command)
            .expect("I/O Pool Disconnected");

        leaf_pages.insert(leaf_pn, page);
        submissions += 1;
    }

    for _ in 0..submissions {
        let completion = leaf_reader
            .io_handle()
            .recv()
            .expect("I/O Pool Disconnected");
        completion.result?;
    }

    Ok(leaf_pages)
}

/// Container of possible changes made to a node
pub struct ChangedNodeEntry<Node> {
    /// PageNumber of the Node that is being replaced by the current entry
    pub deleted: Option<PageNumber>,
    /// New or modified Node that will be written
    pub inserted: Option<Node>,
    /// Separator of the next node.
    pub next_separator: Option<Key>,
}

/// Tracker of all changes that happen to the nodes during an update
pub struct NodesTracker<Node> {
    /// Elements being tracked by the NodesTracker, each Separator
    /// is associated with a ChangedNodeEntry
    pub inner: BTreeMap<Key, ChangedNodeEntry<Node>>,
    /// Pending base received from the right worker which will be used as new base
    pub pending_base: Option<(Key, Node, Option<Key>)>,
}

impl<Node> NodesTracker<Node> {
    /// Create a new NodesTracker
    pub fn new() -> Self {
        Self {
            inner: BTreeMap::new(),
            pending_base: None,
        }
    }

    /// Add or modify a ChangedNodeEntry specifying a deleted PageNumber.
    /// If the entry is already present, it cannot be associated with another deleted PageNumber.
    pub fn delete(&mut self, key: Key, pn: PageNumber, next_separator: Option<Key>) {
        let entry = self.inner.entry(key).or_insert(ChangedNodeEntry {
            deleted: None,
            inserted: None,
            next_separator,
        });

        // we can only delete a node once.
        assert!(entry.deleted.is_none());

        entry.deleted.replace(pn);
        entry.next_separator = next_separator;
    }

    /// Add or modify a ChangedNodeEntry specifying an inserted Node.
    pub fn insert(&mut self, key: Key, node: Node, next_separator: Option<Key>) {
        let entry = self.inner.entry(key).or_insert(ChangedNodeEntry {
            deleted: None,
            inserted: None,
            next_separator,
        });

        entry.next_separator = next_separator;
        entry.inserted.replace(node);
    }

    #[cfg(test)]
    pub fn get(&self, key: Key) -> Option<&ChangedNodeEntry<Node>> {
        self.inner.get(&key)
    }
}
