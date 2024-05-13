#![allow(unused)] // TODO: remove altogether?

use crate::{
    page_cache::{Page, PageCache, PageDiff},
    page_region::PageRegion,
    rw_pass_cell::{ReadPass, WritePass},
};
use nomt_core::{
    page::DEPTH,
    page_id::{PageId, PageIdsIterator, ROOT_PAGE_ID},
    trie::{self, KeyPath, Node, NodeKind},
    trie_pos::TriePosition,
};

/// A cursor wrapping a [`PageCache`] for read-only access to the page tree.
pub struct PageCacheCursor {
    pages: PageCache,
    root: Node,
    pos: TriePosition,
    // Invariant: this is always set when not at the root and never
    // set at the root.
    cached_page: Option<(PageId, Page)>,
    read_pass: ReadPass<PageRegion>,
}

impl PageCacheCursor {
    /// Creates the cursor pointing at the given root of the trie.
    fn new(root: Node, pages: PageCache, read_pass: ReadPass<PageRegion>) -> Self {
        Self {
            root,
            pos: TriePosition::new(),
            pages,
            cached_page: None,
            read_pass,
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
        match self.cached_page {
            None => self.root,
            Some((_, ref page)) => page.node(&self.read_pass, self.pos.node_index()),
        }
    }

    fn read_leaf_children(&self) -> trie::LeafData {
        let (page, _page_id, children) =
            crate::page_cache::locate_leaf_data(&self.pos, self.cached_page.as_ref(), |page_id| {
                self.pages.retrieve_sync(page_id, false)
            });

        trie::LeafData {
            key_path: page.node(&self.read_pass, children.left()),
            value_hash: page.node(&self.read_pass, children.right()),
        }
    }

    /// Peek at the sibling node of the current position without moving the cursor. At the root,
    /// gives the terminator.
    pub fn peek_sibling(&self) -> Node {
        match self.cached_page {
            None => trie::TERMINATOR,
            Some((_, ref page)) => page.node(&self.read_pass, self.pos.sibling_index()),
        }
    }

    fn retrieve(&mut self, page_id: PageId) -> Page {
        self.pages.retrieve_sync(page_id, false)
    }
}
