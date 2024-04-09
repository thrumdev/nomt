use std::{cell::RefCell, collections::HashSet};

use crate::{
    page_cache::{Page, PageCache},
    rw_pass_cell::{ReadPass, WritePass},
};
use nomt_core::{
    page::DEPTH,
    page_id::{PageId, PageIdsIterator, ROOT_PAGE_ID},
    trie::{self, KeyPath, Node, NodeKind},
    trie_pos::TriePosition,
};

#[allow(unused)]
// TODO: this wil disappear in follow-up
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

/// A cursor wrapping a [`PageCache`].
///
/// This performs I/O internally.
pub struct PageCacheCursor {
    pages: PageCache,
    root: Node,
    pos: TriePosition,
    // Invariant: this is always set when not at the root and never
    // set at the root.
    cached_page: Option<(PageId, Page)>,
    mode: Mode,
}

impl PageCacheCursor {
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
            pos: TriePosition::new(),
            pages,
            cached_page: None,
            mode,
        }
    }

    /// Moves the cursor back to the root.
    pub fn rewind(&mut self) {
        self.pos = TriePosition::new();
        self.cached_page = None;
    }

    /// Get the current position of the cursor expressed as a bit-path and length. Bits after the
    /// length are irrelevant.
    pub fn position(&self) -> TriePosition {
        self.pos.clone()
    }

    /// Jump to the node at the given path. Only the first `depth` bits are relevant.
    /// It is possible to jump out of bounds, that is, to a node whose parent is a terminal.
    pub fn jump(&mut self, path: KeyPath, depth: u8) {
        if depth == 0 {
            self.rewind();
            return;
        }

        self.pos = TriePosition::from_path_and_depth(path, depth);

        let n_pages = (self.pos.depth() - 1) as usize / DEPTH;
        let page_id = PageIdsIterator::new(path)
            .nth(n_pages)
            .expect("all keys with <= 256 bits have pages; qed");
        self.cached_page = Some((page_id.clone(), self.retrieve(page_id)));
    }

    /// Go up the trie by `d` levels.
    ///
    /// The cursor will not go beyond the root. If the cursor's location was sound before the call,
    /// it will be sound after the call.
    pub fn up(&mut self, d: u8) {
        let prev_depth = self.pos.depth();
        self.pos.up(d);

        if self.pos.depth() == 0 {
            self.rewind();
            return;
        }

        let prev_page_depth = (prev_depth as usize + DEPTH - 1) / DEPTH;
        let new_page_depth = (self.pos.depth() as usize + DEPTH - 1) / DEPTH;

        // sanity: always not root unless depth is zero, checked above.
        let mut cur_page_id = self.cached_page.as_ref().expect("not root; qed").0.clone();

        for _ in new_page_depth..prev_page_depth {
            cur_page_id = cur_page_id.parent_page_id();
        }

        self.cached_page = Some((cur_page_id.clone(), self.retrieve(cur_page_id)));
    }

    /// Traverse to the child of this node indicated by the given bit.
    ///
    /// The assumption is that the cursor is pointing to the internal node and it's the
    /// responsibility of the caller to ensure that. If violated,the cursor's location will be
    /// invalid.
    ///
    /// hint_fresh indicates that any new page which needs to be fetched should be freshly
    /// allocated. Improper use of this function will cause deletion of existing tree data.
    pub fn down(&mut self, bit: bool, hint_fresh: bool) {
        if self.pos.depth() as usize % DEPTH == 0 {
            // attempt to load next page if we are at the end of our previous page or the root.
            // UNWRAP: page index is valid, nodes never fall beyond the 42nd page.
            let page_id = match self.cached_page {
                None => ROOT_PAGE_ID,
                Some((ref id, _)) => id
                    .child_page_id(self.pos.child_page_index())
                    .expect("Pages do not go deeper than the maximum layer, 42"),
            };

            self.cached_page = Some((
                page_id.clone(),
                self.pages.retrieve_sync(page_id, hint_fresh),
            ));
        }
        self.pos.down(bit);
    }

    /// Traverse to the sibling node of the current position. No-op at the root.
    pub fn sibling(&mut self) {
        self.pos.sibling();
    }

    /// Returns the node at the current location of the cursor.
    pub fn node(&self) -> Node {
        self.mode
            .with_read_pass(|read_pass| match self.cached_page {
                None => self.root,
                Some((_, ref page)) => page.node(&read_pass, self.pos.node_index()),
            })
    }

    fn read_leaf_children(&self) -> trie::LeafData {
        let (page, _page_id, children) =
            crate::page_cache::locate_leaf_data(&self.pos, self.cached_page.as_ref(), |page_id| {
                self.pages.retrieve_sync(page_id, false)
            });

        self.mode.with_read_pass(|read_pass| trie::LeafData {
            key_path: page.node(&read_pass, children.left()),
            value_hash: page.node(&read_pass, children.right()),
        })
    }

    fn write_leaf_children(&mut self, leaf_data: Option<trie::LeafData>, hint_fresh: bool) {
        let (page, page_id, children) =
            crate::page_cache::locate_leaf_data(&self.pos, self.cached_page.as_ref(), |page_id| {
                self.pages.retrieve_sync(page_id, hint_fresh)
            });

        let (write_pass, dirtied) = match self.mode {
            Mode::Read(_) => panic!("attempted to call modify on a read-only cursor"),
            Mode::Write {
                ref mut write_pass,
                ref mut dirtied,
            } => (write_pass, dirtied),
        };

        match leaf_data {
            None => page.clear_leaf_data(&mut *write_pass.borrow_mut(), children),
            Some(leaf) => page.set_leaf_data(&mut *write_pass.borrow_mut(), children, leaf),
        }
        dirtied.insert(page_id.clone());
    }

    /// Place a non-leaf node at the current location.
    pub fn place_non_leaf(&mut self, node: Node) {
        assert!(!trie::is_leaf(&node));

        if trie::is_leaf(&self.node()) {
            self.write_leaf_children(None, false);
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
            Some((ref page_id, ref mut page)) => {
                page.set_node(&mut *write_pass.borrow_mut(), self.pos.node_index(), node);
                dirtied.insert(page_id.clone());
            }
        }
    }

    /// Place a leaf node at the current location.
    pub fn place_leaf(&mut self, node: Node, leaf: trie::LeafData) {
        assert!(trie::is_leaf(&node));

        self.write_leaf_children(Some(leaf), true);

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
            Some((ref page_id, ref mut page)) => {
                page.set_node(&mut *write_pass.borrow_mut(), self.pos.node_index(), node);
                dirtied.insert(page_id.clone());
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
        if self.pos.depth() == 0 {
            return None;
        }

        let node = self.node();
        let sibling = self.peek_sibling();

        let bit = self.pos.peek_last_bit();

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
                self.write_leaf_children(None, false);
                self.place_non_leaf(trie::TERMINATOR);
                self.up(1);
                self.place_leaf(node, prev);

                None
            }
            (NodeKind::Terminator, NodeKind::Leaf) => {
                // compact: clear sibling node, move leaf up.
                self.sibling();

                let prev = self.read_leaf_children();
                self.write_leaf_children(None, false);
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
                Some((_, ref page)) => page.node(&read_pass, self.pos.sibling_index()),
            })
    }

    fn retrieve(&mut self, page_id: PageId) -> Page {
        self.pages.retrieve_sync(page_id, false)
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
    fn position(&self) -> TriePosition {
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

    fn sibling(&mut self) {
        PageCacheCursor::sibling(self)
    }

    fn down(&mut self, bit: bool, hint_fresh: bool) {
        PageCacheCursor::down(self, bit, hint_fresh)
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
