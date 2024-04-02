//! Pages: efficient node storage.
//!
//! Because each node in the trie is exactly 32 bytes, we can easily pack groups of nodes into
//! a predictable paged representation regardless of the information in the trie.
//!
//! Each page is 4096 bytes and stores up to 126 nodes plus a unique 32-byte page identifier,
//! with 32 bytes left over.
//!
//! A page stores a rootless sub-tree with depth 6: that is, it stores up to
//! 2 + 4 + 8 + 16 + 32 + 64 nodes at known positions.
//! Semantically, all nodes within the page should descend from the layer above, and the
//! top two nodes are expected to be siblings. Each page logically has up to 64 child pages, which
//! correspond to the rootless sub-tree descending from each of the 64 child nodes on the bottom
//! layer.
//!
//! Every page is referred to by a unique ID, given by `parent_id * 2^6 + child_index + 1`, where
//! the root page has ID `0x00..00`. The child index ranges from 0 to 63 and therefore can be
//! represented as a 6 bit string. This module exposes functions for manipulating page IDs.
//!
//! The [`Page`] structure wraps a borrowed slice of 32-byte data and treats it as a page.
//! A [`PageSet`] is a simple index of pages which are loaded in memory. It can be used to build a
//! [`PageSetCursor`] for traversing the nodes stored within pages.

use crate::page_id::{PageId, ROOT_PAGE_ID};
use crate::trie::{KeyPath, Node, TERMINATOR};

use alloc::collections::BTreeMap;
use bitvec::prelude::*;
use core::borrow::Borrow;

/// Depth of the rootless sub-binary tree stored in a page
pub const DEPTH: usize = 6;

// Total number of nodes stored in one Page. It depends on the `DEPTH`
// of the rootless sub-binary tree stored in a page following this formula:
// (2^(DEPTH + 1)) - 2
pub const NODES_PER_PAGE: usize = (1 << DEPTH + 1) - 2;

/// A raw, unsized page data slice.
pub type RawPage = [[u8; 32]];

/// A read-only view over borrowed page data.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PageView<'a> {
    /// The underlying slice of nodes in memory.
    /// This is expected to have a length equal to a power of two.
    ///
    /// The first `len - 2` slots are reserved for nodes. The last slot is the page ID.
    /// The second-to-last slot data does not matter.
    data: &'a RawPage,
}

impl<'a> PageView<'a> {
    /// Create a new view over a page.
    /// Fails if RawPage has an incorrect length or if the contained PageId is invalid.
    pub fn new(data: &'a RawPage) -> Result<Self, InvalidPage> {
        if data.len() != NODES_PER_PAGE + 2 {
            return Err(InvalidPage::Length);
        }

        PageId::from_bytes(data[NODES_PER_PAGE + 1]).map_err(|_| InvalidPage::Id)?;

        Ok(PageView { data })
    }

    /// Get the node storage section of the page.
    pub fn nodes(&self) -> &'a [Node] {
        &self.data[..NODES_PER_PAGE]
    }

    /// Get the page ID of the page.
    pub fn id(&self) -> PageId {
        PageId::from_bytes(self.data[NODES_PER_PAGE + 1])
            .expect("PageView is being created checking the validity of its PageId")
    }
}

/// Error related to the creation of a PageView over a RawPage
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InvalidPage {
    /// Error indicating that a page has an invalid length.
    Length,
    /// Error indicating that a page has an invalid PageId.
    Id,
}

/// A simple and inefficient [`PageSet`] implementation.
#[derive(Debug, Clone, PartialEq)]
pub struct SimplePageSet {
    page_map: BTreeMap<PageId, Vec<[u8; 32]>>,
}

impl SimplePageSet {
    /// Create a new page set.
    pub fn new() -> Self {
        SimplePageSet {
            page_map: BTreeMap::new(),
        }
    }

    /// Insert a page. Returns the previous page stored under that ID.
    pub fn insert(&mut self, page: Vec<[u8; 32]>) -> Result<Option<Vec<[u8; 32]>>, InvalidPage> {
        let id = PageView::new(&page[..])?.id().clone();
        Ok(self.page_map.insert(id, page))
    }

    /// Get a page by ID.
    pub fn get(&self, id: &PageId) -> Option<PageView> {
        self.page_map
            .get(id)
            .map(|p| PageView::new(p.borrow()).expect("checked on insertion"))
    }
}

#[derive(Debug, Clone, PartialEq)]
enum CursorLocation<'a> {
    Root,
    Page(PageView<'a>),
    Missing(PageId),
}

impl<'a> CursorLocation<'a> {
    fn page_id(&self) -> Option<PageId> {
        match *self {
            CursorLocation::Root => None,
            CursorLocation::Page(p) => Some(p.id()),
            CursorLocation::Missing(ref p) => Some(p.clone()),
        }
    }
}

/// A cursor used to inspect the binary trie nodes expected to be stored.
///
/// This cursor operates over a provided root node and a [`SimplePageSet`].
///
/// The [`PageSetCursor`] simply reads what is stored in the underlying pages, even if they are not
/// a valid binary trie. It also does not prevent the user from traversing beyond the end of the
/// trie. It provides enough information for higher level logic to avoid those mistakes.
///
/// If a page is missing, this cursor interprets its data as all-zero.
#[derive(Debug, Clone, PartialEq)]
pub struct PageSetCursor<'a> {
    root: Node,
    path: KeyPath,
    depth: u8,
    pages: &'a SimplePageSet,
    location: CursorLocation<'a>,
}

impl<'a> PageSetCursor<'a> {
    /// Create a new cursor at the given trie root, reading out of the
    /// provided pages.
    pub fn new(root: Node, pages: &'a SimplePageSet) -> Self {
        PageSetCursor {
            root,
            path: [0u8; 32],
            depth: 0,
            pages,
            location: CursorLocation::Root,
        }
    }
}

impl<'a> crate::cursor::Cursor for PageSetCursor<'a> {
    /// The current position of the cursor, expressed as a bit-path and length. Bits after the
    /// length are irrelevant.
    fn position(&self) -> (KeyPath, u8) {
        (self.path, self.depth)
    }

    /// The current node.
    fn node(&self) -> &Node {
        match self.location {
            CursorLocation::Root => &self.root,
            CursorLocation::Missing(_) => &TERMINATOR,
            CursorLocation::Page(p) => {
                let path = last_page_path(&self.path, self.depth);
                &p.nodes()[node_index(path)]
            }
        }
    }

    /// Traverse upwards by `d` bits. If d is greater than or equal to the current position length,
    /// move to the root.
    fn up(&mut self, d: u8) {
        let d = core::cmp::min(self.depth, d);
        let prev_depth = self.depth;
        self.depth -= d;

        if self.depth == 0 {
            self.location = CursorLocation::Root;
            return;
        }

        let prev_page_depth = (prev_depth as usize + DEPTH - 1) / DEPTH;
        let new_page_depth = (self.depth as usize + DEPTH - 1) / DEPTH;

        // sanity: always not root unless depth is zero, checked above.
        let mut cur_page_id = self.location.page_id().expect("not root; qed");

        for _ in new_page_depth..prev_page_depth {
            cur_page_id = cur_page_id.parent_page_id();
        }

        self.location = match self.pages.get(&cur_page_id) {
            Some(p) => CursorLocation::Page(p),
            None => CursorLocation::Missing(cur_page_id),
        };
    }

    fn jump(&mut self, path: KeyPath, depth: u8) {
        self.path = path;
        self.depth = depth;

        if depth == 0 {
            self.rewind();
            return;
        }

        let n_pages = self.depth as usize / DEPTH;
        let page_id = crate::page_id::PageIdsIterator::new(&self.path)
            .nth(n_pages)
            .expect("all keys with <= 256 bits have pages; qed");

        self.location = match self.pages.get(&page_id) {
            None => CursorLocation::Missing(page_id),
            Some(p) => CursorLocation::Page(p),
        };
    }

    fn down(&mut self, bit: bool) {
        if self.depth as usize % DEPTH == 0 {
            // attempt to load next page if we are at the end of our previous page or the root.
            let page_id = match self.location.page_id() {
                None => ROOT_PAGE_ID,
                Some(p) => {
                    let child_page_idx = last_page_path(&self.path, self.depth).load_be::<u8>();
                    p.child_page_id(child_page_idx).expect(
                        "Child index is 6 bits and Pages do not go deeper than the maximum layer, 42"
                    )
                }
            };

            self.location = match self.pages.get(&page_id) {
                None => CursorLocation::Missing(page_id),
                Some(p) => CursorLocation::Page(p),
            };
        }

        // Update the cursor's lookup path.
        self.path
            .view_bits_mut::<Msb0>()
            .set(self.depth as usize, bit);
        self.depth += 1;
    }

    fn peek_sibling(&self) -> &Node {
        match self.location {
            CursorLocation::Root => &TERMINATOR,
            CursorLocation::Missing(_) => &TERMINATOR,
            CursorLocation::Page(p) => {
                let path = last_page_path(&self.path, self.depth);
                &p.nodes()[sibling_index(path)]
            }
        }
    }

    fn sibling(&mut self) {
        if self.depth == 0 {
            return;
        }

        let bits = self.path.view_bits_mut::<Msb0>();
        let i = self.depth as usize - 1;
        bits.set(i, !bits[i]);
    }

    fn seek(&mut self, path: KeyPath) -> Option<(Node, u8)> {
        let mut res = None;

        for (i, bit) in path.view_bits::<Msb0>().iter().by_vals().enumerate() {
            if crate::trie::is_internal(self.node()) {
                self.down(bit);
            } else {
                res = Some((*self.node(), i as u8));
                break;
            }
        }

        res
    }

    /// Rewind the cursor to the root.
    fn rewind(&mut self) {
        self.location = CursorLocation::Root;
        self.depth = 0;
    }
}

// extract the relevant portion of the key path to the last page. panics on empty path.
fn last_page_path(total_path: &KeyPath, total_depth: u8) -> &BitSlice<u8, Msb0> {
    let prev_page_end = ((total_depth as usize - 1) / DEPTH) * DEPTH;
    &total_path.view_bits::<Msb0>()[prev_page_end..total_depth as usize]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_index_mapping_is_complete() {
        let mut positions = vec![None; NODES_PER_PAGE];

        check_recursive(BitVec::<u8, Msb0>::new(), &mut positions);
        fn check_recursive(b: BitVec<u8, Msb0>, positions: &mut [Option<BitVec<u8, Msb0>>]) {
            let pos = node_index(&b[..]);
            if !b.is_empty() {
                if let Some(other) = &positions[pos] {
                    panic!("{} and {} both map to {}", other, b, pos);
                }
                positions[pos] = Some(b.clone());
            }
            if b.len() == DEPTH {
                return;
            }

            let mut left = b.clone();
            let mut right = b;
            left.push(false);
            right.push(true);
            check_recursive(left, positions);
            check_recursive(right, positions);
        }

        assert!(positions.iter().all(|p| p.is_some()));
    }

    #[test]
    fn sibling_node_index_mapping_is_complete() {
        let mut positions = vec![None; NODES_PER_PAGE];

        check_recursive(BitVec::<u8, Msb0>::new(), &mut positions);
        fn check_recursive(b: BitVec<u8, Msb0>, positions: &mut [Option<BitVec<u8, Msb0>>]) {
            let pos = sibling_index(&b[..]);
            if !b.is_empty() {
                if let Some(other) = &positions[pos] {
                    panic!("{} and {} both map to {}", other, b, pos);
                }
                positions[pos] = Some(b.clone());
            }
            if b.len() == DEPTH {
                return;
            }

            let mut left = b.clone();
            let mut right = b;
            left.push(false);
            right.push(true);
            check_recursive(left, positions);
            check_recursive(right, positions);
        }

        assert!(positions.iter().all(|p| p.is_some()));
    }

    #[test]
    #[should_panic]
    fn last_page_path_empty_panic() {
        last_page_path(&[0u8; 32], 0);
    }

    #[test]
    fn last_page_path_works() {
        let key = [0b0101_0101u8; 32];
        let mut expected = bitvec![0];
        for i in 1..256 {
            assert_eq!(last_page_path(&key, i as u8), expected);
            if i % DEPTH == 0 {
                expected.clear();
            }
            expected.push(i % 2 == 1);
        }
    }
}
