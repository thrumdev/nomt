use std::cmp;

use crate::page_cache::{Page, PageCache, PagePromise};
use bitvec::prelude::*;
use nomt_core::{
    page::DEPTH,
    page_id::{PageId, PageIdsIterator, ROOT_PAGE_ID},
    trie::{self, KeyPath, Node},
};

/// The breadth of the prefetch request.
///
/// The number of items we want to request in a single batch.
const PREFETCH_N: usize = 7;

/// A cursor wrapping a [`PageCache`].
///
/// This performs I/O internally.
pub struct PageCacheCursor {
    pages: PageCache,
    root: Node,
    path: KeyPath,
    depth: u8,
    // Invariant: this is always set when not at the root and never
    // set at the root.
    cached_page: Option<(PageId, Page)>,
}

impl PageCacheCursor {
    /// Creates the cursor pointing at the given root of the trie.
    pub fn at_root(root: Node, pages: PageCache) -> Self {
        Self {
            root,
            path: KeyPath::default(),
            depth: 0,
            pages,
            cached_page: None,
        }
    }

    /// Moves the cursor back to the root.
    pub fn rewind(&mut self) {
        self.depth = 0;
        self.cached_page = None;
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

        let n_pages = self.depth as usize / DEPTH;
        let page_id = PageIdsIterator::new(&self.path)
            .nth(n_pages)
            .expect("all keys with <= 256 bits have pages; qed");

        self.cached_page = Some((page_id.clone(), self.pages.retrieve(page_id).wait()));
    }

    /// Moves the cursor to the given [`KeyPath`].
    ///
    /// Moving the cursor using this function would be more efficient than using the navigation
    /// functions such as [`Self::down_left`] and [`Self::down_right`] due to leveraging warmup
    /// hints.
    ///
    /// After returning of this function, the cursor is positioned either at the given key path or
    /// at the closest key path that is on the way to the given key path.
    pub fn seek(&mut self, dest: KeyPath) {
        self.rewind();
        if !trie::is_internal(&self.root) {
            // The root either a leaf or the terminator. In either case, we can't navigate anywhere.
            return;
        }
        let mut ppf = PagePrefetcher::new(dest);

        for bit in dest.view_bits::<Msb0>().iter().by_vals() {
            if !trie::is_internal(&self.node()) {
                break;
            }
            if self.depth as usize % DEPTH == 0 {
                // attempt to load next page if we are at the end of our previous page or the root.
                match ppf.next(&self.pages) {
                    None => {
                        panic!("reached the end of the prefetcher without finding the terminal")
                    }
                    Some(p) => p.wait(),
                };
            }
            // page is loaded, so this won't block.
            self.down(bit);
        }
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
            self.cached_page = None;
            return;
        }

        let prev_page_depth = (prev_depth as usize + DEPTH - 1) / DEPTH;
        let new_page_depth = (self.depth as usize + DEPTH - 1) / DEPTH;

        // sanity: always not root unless depth is zero, checked above.
        let mut cur_page_id = self.cached_page.as_ref().expect("not root; qed").0.clone();

        for _ in new_page_depth..prev_page_depth {
            cur_page_id = cur_page_id.parent_page_id();
        }

        self.cached_page = Some((cur_page_id.clone(), self.pages.retrieve(cur_page_id).wait()));
    }

    /// Traverse to the child of this node indicated by the given bit.
    ///
    /// The assumption is that the cursor is pointing to the internal node and it's the
    /// responsibility of the caller to ensure that. If violated,the cursor's location will be
    /// invalid.
    pub fn down(&mut self, bit: bool) {
        if self.depth as usize % DEPTH == 0 {
            // attempt to load next page if we are at the end of our previous page or the root.
            let page_id = match self.cached_page {
                None => ROOT_PAGE_ID,
                Some((ref id, _)) => {
                    let child_page_idx = last_page_path(&self.path, self.depth).load_be::<u8>();
                    id.child_page_id(child_page_idx).expect(
                        "Child index is 6 bits and Pages do not go deeper than the maximum layer, 42"
                    )
                }
            };

            self.cached_page = Some((page_id.clone(), self.pages.retrieve(page_id).wait()));
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
    }

    /// Returns the node at the current location of the cursor.
    pub fn node(&self) -> Node {
        match self.cached_page {
            None => self.root,
            Some((_, ref page)) => {
                let path = last_page_path(&self.path, self.depth);
                page.node(node_index(path))
            }
        }
    }

    /// Modify the node at the current location of the cursor.
    pub fn modify(&mut self, node: Node) {
        match self.cached_page {
            None => {
                self.root = node;
            }
            Some((page_id, ref mut page)) => {
                let path = last_page_path(&self.path, self.depth);
                let index = node_index(path);
                page.set_node(index, node);
                self.pages.mark_dirty(page_id);
            }
        }
    }

    /// Peek at the sibling node of the current position without moving the cursor. At the root,
    /// gives the terminator.
    pub fn peek_sibling(&self) -> Node {
        match self.cached_page {
            None => trie::TERMINATOR,
            Some((_, ref page)) => {
                let path = last_page_path(&self.path, self.depth);
                page.node(sibling_index(path))
            }
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
        PageCacheCursor::seek(self, path)
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

    fn modify(&mut self, node: Node) {
        PageCacheCursor::modify(self, node)
    }
}

// extract the relevant portion of the key path to the last page. panics on empty path.
fn last_page_path(total_path: &KeyPath, total_depth: u8) -> &BitSlice<u8, Msb0> {
    let prev_page_end = ((total_depth as usize - 1) / DEPTH) * DEPTH;
    &total_path.view_bits::<Msb0>()[prev_page_end..total_depth as usize]
}

// Transform a bit-path to the index in a page corresponding to the sibling node.
//
// The expected length of the page path is between 1 and `DEPTH`, inclusive. A length of 0 returns
// 0 and all bits beyond `DEPTH` are ignored.
fn sibling_index(page_path: &BitSlice<u8, Msb0>) -> usize {
    let index = node_index(page_path);
    if page_path.is_empty() {
        0
    } else if index % 2 == 0 {
        index + 1
    } else {
        index - 1
    }
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

struct PagePrefetcher {
    /// The page IDs of the destination key path in reverse order.
    rev_page_ids: Vec<PageId>,
    /// The promises of the pages that are currently being fetched. The promises are laid out in
    /// such a way, that [`Vec::pop`] would return the next page to be processed.
    ///
    /// This vector is at most of size [`PREFETCH_N`].
    page_promises: Vec<PagePromise>,
}

impl PagePrefetcher {
    fn new(dest: KeyPath) -> Self {
        let rev_page_ids = {
            // Chop the destination key path into the corresponding page IDs and reverse them so
            // that we can `pop`.
            let mut x = PageIdsIterator::new(&dest).collect::<Vec<_>>();
            x.reverse();
            x
        };
        Self {
            rev_page_ids,
            page_promises: Vec::new(),
        }
    }

    fn next(&mut self, page_cache: &PageCache) -> Option<PagePromise> {
        if self.page_promises.is_empty() {
            // The prefetch hasn't started yet or we consumed all the promises. We need to initiate
            // a new prefetch.
            if self.rev_page_ids.is_empty() {
                // No pages left to prefetch.
                return None;
            }
            let n = cmp::min(PREFETCH_N, self.rev_page_ids.len());
            for _ in 0..n {
                // Unwrap: pop must succeed since the n is clamped to the length of the vector.
                let page_id = self.rev_page_ids.pop().unwrap();
                self.page_promises.push(page_cache.retrieve(page_id));
            }
            self.page_promises.reverse();
        }
        self.page_promises.pop()
    }
}
