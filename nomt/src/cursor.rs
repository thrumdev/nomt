use std::{cell::RefCell, cmp, collections::HashSet};

use crate::{
    page_cache::{Page, PageCache},
    rw_pass_cell::{ReadPass, WritePass},
};
use bitvec::prelude::*;
use nomt_core::{
    page::DEPTH,
    page_id::{PageId, PageIdsIterator, ROOT_PAGE_ID},
    trie::{self, KeyPath, Node, NodeKind},
};

/// The breadth of the prefetch request.
///
/// The number of items we want to request in a single batch.
const PREFETCH_N: usize = 7;

enum Mode {
    Read(ReadPass),
    Write {
        write_pass: RefCell<WritePass>,
        dirtied: HashSet<PageId>,
    },
}

impl Mode {
    fn with_read_pass<R, F>(&self, f: F) -> R
    where
        F: FnOnce(&ReadPass) -> R,
    {
        match self {
            Mode::Read(ref read_pass) => f(read_pass),
            Mode::Write { write_pass, .. } => {
                let mut write_pass = write_pass.borrow_mut();
                f(write_pass.downgrade())
            }
        }
    }
}

/// Modes for seeking to a key path.
#[derive(Debug, Clone, Copy)]
pub enum SeekMode {
    /// Retrieve the pages with the child location of any sibling nodes which are also leaves.
    ///
    /// This should be used when preparing to delete a key, which can cause leaf nodes to be
    /// relocated.
    RetrieveSiblingLeafChildren,
    /// Retrieve the pages along the path to the key path's corresponding terminal node only.
    PathOnly,
}

/// A cursor wrapping a [`PageCache`].
///
/// This performs I/O internally.
pub struct PageCacheCursor {
    pages: PageCache,
    root: Node,
    path: KeyPath,
    depth: u8,
    node_index: usize,
    // Invariant: this is always set when not at the root and never
    // set at the root.
    cached_page: Option<(PageId, Page)>,
    mode: Mode,
}

impl PageCacheCursor {
    /// Create a new [`PageCacheCursor`] configured for reading.
    ///
    /// Subsequent calls to `modify` or `finish_write` will panic.
    pub fn new_read(root: Node, pages: PageCache, read_pass: ReadPass) -> Self {
        Self::new(root, pages, Mode::Read(read_pass))
    }

    /// Create a new [`PageCacheCursor`] configured for writing.
    pub fn new_write(root: Node, pages: PageCache, write_pass: WritePass) -> Self {
        Self::new(
            root,
            pages,
            Mode::Write {
                write_pass: RefCell::new(write_pass),
                dirtied: HashSet::new(),
            },
        )
    }

    /// Creates the cursor pointing at the given root of the trie.
    fn new(root: Node, pages: PageCache, mode: Mode) -> Self {
        Self {
            root,
            path: KeyPath::default(),
            depth: 0,
            node_index: 0,
            pages,
            cached_page: None,
            mode,
        }
    }

    /// Moves the cursor back to the root.
    pub fn rewind(&mut self) {
        self.depth = 0;
        self.cached_page = None;
        self.node_index = 0;
    }

    /// Get the current position of the cursor expressed as a bit-path and length. Bits after the
    /// length are irrelevant.
    pub fn position(&self) -> (KeyPath, u8) {
        (self.path, self.depth)
    }

    /// Jump to the node at the given path. Only the first `depth` bits are relevant.
    /// It is possible to jump out of bounds, that is, to a node whose parent is a terminal.
    pub fn jump(&mut self, path: KeyPath, depth: u8) {
        self.path = path;
        self.depth = depth;

        if depth == 0 {
            self.rewind();
            return;
        }

        let n_pages = (self.depth - 1) as usize / DEPTH;
        let page_id = PageIdsIterator::new(self.path)
            .nth(n_pages)
            .expect("all keys with <= 256 bits have pages; qed");
        self.node_index = {
            let path = last_page_path(&self.path, self.depth);
            node_index(path)
        };

        self.cached_page = Some((page_id.clone(), self.retrieve(page_id)));
    }

    /// Moves the cursor to the given [`KeyPath`].
    ///
    /// Moving the cursor using this function would be more efficient than using the navigation
    /// functions such as [`Self::down_left`] and [`Self::down_right`] due to leveraging warmup
    /// hints.
    ///
    /// After returning of this function, the cursor is positioned either at the given key path or
    /// at the closest key path that is on the way to the given key path.
    ///
    /// If the terminal node is a leaf, this returns the leaf data associated with that leaf.
    pub fn seek(&mut self, dest: KeyPath, seek_mode: SeekMode) -> Option<trie::LeafData> {
        self.rewind();
        if !trie::is_internal(&self.root) {
            // The root either a leaf or the terminator. In either case, we can't navigate anywhere.
            return if trie::is_leaf(&self.root) {
                Some(self.read_leaf_children())
            } else {
                None
            };
        }

        let mut ppf = PageIdsIterator::new(dest);
        for bit in dest.view_bits::<Msb0>().iter().by_vals() {
            if !trie::is_internal(&self.node()) {
                return if trie::is_leaf(&self.node()) {
                    let leaf_data = self.read_leaf_children();
                    assert!(leaf_data
                        .key_path
                        .view_bits::<Msb0>()
                        .starts_with(&dest.view_bits::<Msb0>()[..self.depth as usize]));
                    Some(leaf_data)
                } else {
                    None
                };
            }
            if self.depth as usize % DEPTH == 0 {
                if self.depth as usize % PREFETCH_N == 0 {
                    for _ in 0..PREFETCH_N {
                        let page_id = match ppf.next() {
                            Some(page) => page,
                            None => break,
                        };
                        self.pages.prepopulate(page_id);
                    }
                }

                if let (&Some((ref id, _)), SeekMode::RetrieveSiblingLeafChildren, true) = (
                    &self.cached_page,
                    seek_mode,
                    trie::is_leaf(&self.peek_sibling()),
                ) {
                    // sibling is a leaf and at the end of the (non-root) page.
                    // initiate a load of the sibling's page.
                    let page_idx = bottom_node_index(sibling_index(self.node_index));
                    let child_page_id = id.child_page_id(page_idx).expect(
                        "Child index is 6 bits and Pages do not go deeper than the maximum layer, 42"
                    );

                    // async; just warm up.
                    let _ = self.pages.prepopulate(child_page_id);
                }
            }
            // page is loaded, so this won't block.
            self.down(bit);
        }

        // sanity: should be impossible not to encounter a node.
        None
    }

    /// Go up the trie by `d` levels.
    ///
    /// The cursor will not go beyond the root. If the cursor's location was sound before the call,
    /// it will be sound after the call.
    pub fn up(&mut self, d: u8) {
        let d = cmp::min(self.depth, d);
        let prev_depth = self.depth;
        self.depth -= d;

        if self.depth == 0 {
            self.rewind();
            return;
        }

        let prev_page_depth = (prev_depth as usize + DEPTH - 1) / DEPTH;
        let new_page_depth = (self.depth as usize + DEPTH - 1) / DEPTH;

        // sanity: always not root unless depth is zero, checked above.
        let mut cur_page_id = self.cached_page.as_ref().expect("not root; qed").0.clone();

        for _ in new_page_depth..prev_page_depth {
            cur_page_id = cur_page_id.parent_page_id();
        }

        if prev_page_depth == new_page_depth {
            let depth_in_page = self.depth_in_page();
            for depth in (depth_in_page..depth_in_page + d as usize).rev().map(|x| x + 1) {
                self.node_index = parent_node_index(self.node_index, depth);
            }
        } else {
            let path = last_page_path(&self.path, self.depth);
            self.node_index = node_index(path);
        }

        self.cached_page = Some((cur_page_id.clone(), self.retrieve(cur_page_id)));
    }

    /// Traverse to the child of this node indicated by the given bit.
    ///
    /// The assumption is that the cursor is pointing to the internal node and it's the
    /// responsibility of the caller to ensure that. If violated,the cursor's location will be
    /// invalid.
    pub fn down(&mut self, bit: bool) {
        if self.depth as usize % DEPTH == 0 {
            // attempt to load next page if we are at the end of our previous page or the root.
            // UNWRAP: page index is valid, nodes never fall beyond the 42nd page.
            let page_id = match self.cached_page {
                None => ROOT_PAGE_ID,
                Some((ref id, _)) => id.child_page_id(bottom_node_index(self.node_index)).unwrap(),
            };

            self.cached_page = Some((page_id.clone(), self.retrieve(page_id)));
            self.node_index = bit as usize;
        } else {
            let (l, r) = child_node_indices(self.node_index, self.depth_in_page());
            self.node_index = if bit { r } else { l };
        }

        // Update the cursor's lookup path.
        self.path
            .view_bits_mut::<Msb0>()
            .set(self.depth as usize, bit);
        self.depth += 1;
    }

    /// Traverse to the sibling node of the current position. No-op at the root.
    pub fn sibling(&mut self) {
        if self.depth == 0 {
            return;
        }

        let bits = self.path.view_bits_mut::<Msb0>();
        let i = self.depth as usize - 1;
        bits.set(i, !bits[i]);
        self.node_index = sibling_index(self.node_index);
    }

    /// Returns the node at the current location of the cursor.
    pub fn node(&self) -> Node {
        self.mode
            .with_read_pass(|read_pass| match self.cached_page {
                None => self.root,
                Some((_, ref page)) => page.node(&read_pass, self.node_index),
            })
    }

    fn read_leaf_children(&self) -> trie::LeafData {
        let fetched_page;
        let (page, (left_idx, right_idx)) = match self.cached_page {
            None => {
                fetched_page = self.pages.retrieve_sync(ROOT_PAGE_ID);
                (&fetched_page, (0, 1))
            }
            Some((page_id, ref page)) => {
                let depth_in_page = self.depth_in_page();
                if depth_in_page == DEPTH {
                    let child_page_idx = bottom_node_index(self.node_index);

                    // UNWRAP: page index is valid, leaf children never fall beyond the 42nd
                    // page.
                    let child_page_id = page_id.child_page_id(child_page_idx).unwrap();
                    fetched_page = self.pages.retrieve_sync(child_page_id);
                    (&fetched_page, (0, 1))
                } else {
                    (page, child_node_indices(self.node_index, depth_in_page))
                }
            }
        };

        self.mode.with_read_pass(|read_pass| trie::LeafData {
            key_path: page.node(&read_pass, left_idx),
            value_hash: page.node(&read_pass, right_idx),
        })
    }

    fn depth_in_page(&self) -> usize {
        if self.depth == 0 {
            0
        } else {
            self.depth as usize - ((self.depth as usize - 1) / DEPTH) * DEPTH
        }
    }

    fn write_leaf_children(&mut self, leaf_data: Option<trie::LeafData>) {
        let depth_in_page = self.depth_in_page();
        let (write_pass, dirtied) = match self.mode {
            Mode::Read(_) => panic!("attempted to call modify on a read-only cursor"),
            Mode::Write {
                ref mut write_pass,
                ref mut dirtied,
            } => (write_pass, dirtied),
        };
        let fetched_page;

        let (page_id, page, left_idx) = match self.cached_page {
            None => {
                fetched_page = self.pages.retrieve_sync(ROOT_PAGE_ID);
                (ROOT_PAGE_ID, &fetched_page, 0)
            }
            Some((page_id, ref page)) => {
                if depth_in_page == DEPTH {
                    let child_page_idx = bottom_node_index(self.node_index);

                    // UNWRAP: page index is valid, leaf children never fall beyond the 42nd
                    // page.
                    let child_page_id = page_id.child_page_id(child_page_idx).unwrap();
                    fetched_page = self.pages.retrieve_sync(child_page_id);
                    (child_page_id, &fetched_page, 0)
                } else {
                    (page_id, page, child_node_indices(self.node_index, depth_in_page).0)
                }
            }
        };

        match leaf_data {
            None => page.clear_leaf_data(&mut *write_pass.borrow_mut(), left_idx),
            Some(leaf) => page.set_leaf_data(&mut *write_pass.borrow_mut(), left_idx, leaf),
        }
        dirtied.insert(page_id);
    }

    /// Place a non-leaf node at the current location.
    pub fn place_non_leaf(&mut self, node: Node) {
        assert!(!trie::is_leaf(&node));

        if trie::is_leaf(&self.node()) {
            self.write_leaf_children(None);
        }

        let (write_pass, dirtied) = match self.mode {
            Mode::Read(_) => panic!("attempted to call modify on a read-only cursor"),
            Mode::Write {
                ref mut write_pass,
                ref mut dirtied,
            } => (write_pass, dirtied),
        };
        match self.cached_page {
            None => {
                self.root = node;
            }
            Some((page_id, ref mut page)) => {
                page.set_node(&mut *write_pass.borrow_mut(), self.node_index, node);
                dirtied.insert(page_id);
            }
        }
    }

    /// Place a leaf node at the current location.
    pub fn place_leaf(&mut self, node: Node, leaf: trie::LeafData) {
        assert!(trie::is_leaf(&node));

        self.write_leaf_children(Some(leaf));

        let (write_pass, dirtied) = match self.mode {
            Mode::Read(_) => panic!("attempted to call modify on a read-only cursor"),
            Mode::Write {
                ref mut write_pass,
                ref mut dirtied,
            } => (write_pass, dirtied),
        };
        match self.cached_page {
            None => {
                self.root = node;
            }
            Some((page_id, ref mut page)) => {
                page.set_node(&mut *write_pass.borrow_mut(), self.node_index, node);
                dirtied.insert(page_id);
            }
        }
    }

    /// Attempt to compact this node with its sibling. There are four possible outcomes.
    ///
    /// 1. If both this and the sibling are terminators, this moves the cursor up one position
    ///    and replaces the parent with a terminator.
    /// 2. If one of this and the sibling is a leaf, and the other is a terminator, this deletes
    ///    the leaf, moves the cursor up one position, and replaces the parent with the deleted
    ///    leaf.
    /// 3. If either or both is an internal node, this moves the cursor up one position and
    ///    return an internal node data structure comprised of this and this sibling.
    /// 4. This is the root - return.
    pub fn compact_up(&mut self) -> Option<trie::InternalData> {
        if self.depth == 0 {
            return None;
        }

        let node = self.node();
        let sibling = self.peek_sibling();

        let this_bit_idx = self.depth as usize - 1;
        // unwrap: depth != 0 above
        let bit = *self.path.view_bits::<Msb0>().get(this_bit_idx).unwrap();

        match (NodeKind::of(&node), NodeKind::of(&sibling)) {
            (NodeKind::Terminator, NodeKind::Terminator) => {
                // compact terminators.
                self.up(1);
                self.place_non_leaf(trie::TERMINATOR);
                None
            }
            (NodeKind::Leaf, NodeKind::Terminator) => {
                // compact: clear this node, move leaf up.

                let prev = self.read_leaf_children();
                self.write_leaf_children(None);
                self.place_non_leaf(trie::TERMINATOR);
                self.up(1);
                self.place_leaf(node, prev);

                None
            }
            (NodeKind::Terminator, NodeKind::Leaf) => {
                // compact: clear sibling node, move leaf up.
                self.sibling();

                let prev = self.read_leaf_children();
                self.write_leaf_children(None);
                self.place_non_leaf(trie::TERMINATOR);
                self.up(1);
                self.place_leaf(sibling, prev);

                None
            }
            _ => {
                // otherwise, internal
                let node_data = if bit {
                    trie::InternalData {
                        left: sibling,
                        right: node,
                    }
                } else {
                    trie::InternalData {
                        left: node,
                        right: sibling,
                    }
                };
                self.up(1);
                Some(node_data)
            }
        }
    }

    /// Peek at the sibling node of the current position without moving the cursor. At the root,
    /// gives the terminator.
    pub fn peek_sibling(&self) -> Node {
        self.mode
            .with_read_pass(|read_pass| match self.cached_page {
                None => trie::TERMINATOR,
                Some((_, ref page)) => page.node(&read_pass, sibling_index(self.node_index)),
            })
    }

    fn retrieve(&mut self, page_id: PageId) -> Page {
        self.pages.retrieve_sync(page_id)
    }

    /// Called when the write is finished.
    pub fn finish_write(self) -> (HashSet<PageId>, WritePass) {
        match self.mode {
            Mode::Read(_) => panic!("attempted to call dirtied_pages on a read-only cursor"),
            Mode::Write {
                dirtied,
                write_pass,
            } => (dirtied, write_pass.into_inner()),
        }
    }
}

impl nomt_core::Cursor for PageCacheCursor {
    fn position(&self) -> (KeyPath, u8) {
        PageCacheCursor::position(self)
    }

    fn node(&self) -> Node {
        PageCacheCursor::node(self)
    }

    fn peek_sibling(&self) -> Node {
        PageCacheCursor::peek_sibling(self)
    }

    fn rewind(&mut self) {
        PageCacheCursor::rewind(self)
    }

    fn jump(&mut self, path: KeyPath, depth: u8) {
        PageCacheCursor::jump(self, path, depth)
    }

    fn seek(&mut self, path: KeyPath) {
        let _ = PageCacheCursor::seek(self, path, SeekMode::PathOnly);
    }

    fn sibling(&mut self) {
        PageCacheCursor::sibling(self)
    }

    fn down(&mut self, bit: bool) {
        PageCacheCursor::down(self, bit)
    }

    fn up(&mut self, d: u8) {
        PageCacheCursor::up(self, d)
    }

    fn place_non_leaf(&mut self, node: Node) {
        PageCacheCursor::place_non_leaf(self, node)
    }

    fn place_leaf(&mut self, node: Node, leaf: trie::LeafData) {
        PageCacheCursor::place_leaf(self, node, leaf)
    }

    fn compact_up(&mut self) -> Option<trie::InternalData> {
        PageCacheCursor::compact_up(self)
    }
}

// extract the relevant portion of the key path to the last page. panics on empty path.
fn last_page_path(total_path: &KeyPath, total_depth: u8) -> &BitSlice<u8, Msb0> {
    let prev_page_end = ((total_depth as usize - 1) / DEPTH) * DEPTH;
    &total_path.view_bits::<Msb0>()[prev_page_end..total_depth as usize]
}

/// Given a node index, get the index of the sibling.
fn sibling_index(node_index: usize) -> usize {
    if node_index % 2 == 0 {
        node_index + 1
    } else {
        node_index - 1
    }
}

// Transform a node index and depth in the current page to the indices where the child nodes are
// stored.
//
// The expected range of `depth` is between 1 and `DEPTH` - 1, inclusive.
// A depth out of range panics.
fn child_node_indices(node_index: usize, depth: usize) -> (usize, usize) {
    let left = match depth {
        1 => 2 + node_index * 2,
        2 => 6 + (node_index - 2) * 2,
        3 => 14 + (node_index - 6) * 2,
        4 => 30 + (node_index - 14) * 2,
        5 => 62 + (node_index - 30) * 2,
        _ => panic!("{depth} out of bounds 1..{}", DEPTH - 1),
    };

    (left, left + 1)
}

// Transform a bit-path to an index in a page.
//
// The expected length of the page path is between 1 and `DEPTH`, inclusive. A length of 0 returns
// 0 and all bits beyond `DEPTH` are ignored.
fn node_index(page_path: &BitSlice<u8, Msb0>) -> usize {
    let depth = core::cmp::min(DEPTH, page_path.len());

    if depth == 0 {
        0
    } else {
        // each node is stored at (2^depth - 2) + as_uint(path)
        (1 << depth) - 2 + page_path[..depth].load_be::<usize>()
    }
}

fn bottom_node_index(node_index: usize) -> u8 {
    node_index as u8 - 62
}

// Transform a node index and depth in the current page to the indices where the parent node is
// stored.
//
// The expected range of `depth` is between 2 and `DEPTH`, inclusive.
// A depth out of range panics.
fn parent_node_index(node_index: usize, depth: usize) -> usize {
    match depth {
        2 => (node_index - 2) / 2,
        3 => 2 + (node_index - 6) / 2,
        4 => 6 + (node_index - 14) / 2,
        5 => 14 + (node_index - 30) / 2,
        6 => 30 + (node_index - 62) / 2,
        _ => panic!("depth {depth} out of bounds 2..=6"),
    }
}
