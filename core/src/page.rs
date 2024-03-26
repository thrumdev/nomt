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

use crate::page_id::{child_page_id, parent_page_id, PageId, ROOT_PAGE_ID};
use crate::trie::{KeyPath, Node};

use alloc::collections::BTreeMap;
use bitvec::prelude::*;

/// Depth of the rootless sub-binary tree stored in a page
pub const DEPTH: usize = 6;

// Total number of nodes stored in one Page. It depends on the `DEPTH`
// of the rootless sub-binary tree stored in a page following this formula:
// (2^DEPTH) - 2
pub const NODES_PER_PAGE: usize = (1 << DEPTH + 1) - 2;

/// A read-only, borrowed page.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Page<'a> {
    /// The underlying slice of nodes in memory.
    /// This is expected to have a length equal to a power of two.
    ///
    /// The first `len - 2` slots are reserved for nodes. The last slot is the page ID.
    /// The second-to-last slot data does not matter.
    data: &'a [[u8; 32]],
}

impl<'a> Page<'a> {
    /// Create a new view over a page.
    pub fn new(data: &'a [[u8; 32]]) -> Result<Self, InvalidPageLength> {
        if data.len() != NODES_PER_PAGE + 2 {
            Err(InvalidPageLength)
        } else {
            Ok(Page { data })
        }
    }

    /// Get the node storage section of the page.
    pub fn nodes(&self) -> &'a [Node] {
        &self.data[..NODES_PER_PAGE]
    }

    /// Get the page ID of the page.
    pub fn id(&self) -> &PageId {
        &self.data[NODES_PER_PAGE + 1]
    }
}

/// Error indicating that a page has an invalid length.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InvalidPageLength;

/// A read-only set of borrowed pages.
#[derive(Debug, Clone, PartialEq)]
pub struct PageSet<'a> {
    page_map: BTreeMap<PageId, Page<'a>>,
}

impl<'a> PageSet<'a> {
    /// Create a new page set.
    pub fn new() -> Self {
        PageSet {
            page_map: BTreeMap::new(),
        }
    }

    /// Insert a page. Returns the previous page stored under that ID.
    pub fn insert(&mut self, page: Page<'a>) -> Option<Page<'a>> {
        let id = page.id().clone();
        self.page_map.insert(id, page)
    }

    /// Get a page by ID.
    pub fn get(&self, id: &PageId) -> Option<Page<'a>> {
        self.page_map.get(id).map(|p| *p)
    }
}

/// Error indicating a page missing.
#[derive(Debug, Clone, PartialEq)]
pub struct MissingPage;

#[derive(Debug, Clone, PartialEq)]
enum CursorLocation<'a> {
    Root,
    Page(Page<'a>),
}

/// A cursor used to inspect the binary trie nodes expected to be stored within pages while
/// abstracting over the underlying page structure itself.
///
/// This cursor operates over a provided root node and [`PageSet`].
///
/// The [`PageSetCursor`] simply reads what is stored in the underlying pages, even if they are not
/// a valid binary trie. It also does not prevent the user from traversing beyond the end of the
/// trie. It provides enough information for higher level logic to avoid those mistakes.
#[derive(Debug, Clone, PartialEq)]
pub struct PageSetCursor<'a> {
    root: Node,
    path: KeyPath,
    depth: u8,
    pages: &'a PageSet<'a>,
    location: CursorLocation<'a>,
}

impl<'a> PageSetCursor<'a> {
    /// Create a new cursor at the given trie root, reading out of the
    /// provided pages.
    pub fn new(root: Node, pages: &'a PageSet<'a>) -> Self {
        PageSetCursor {
            root,
            path: [0u8; 32],
            depth: 0,
            pages,
            location: CursorLocation::Root,
        }
    }

    /// The current position of the cursor, expressed as a bit-path and length. Bits after the
    /// length are irrelevant.
    pub fn position(&self) -> (KeyPath, u8) {
        (self.path, self.depth)
    }

    /// The current node.
    pub fn node(&self) -> &Node {
        match self.location {
            CursorLocation::Root => &self.root,
            CursorLocation::Page(p) => {
                let path = last_page_path(&self.path, self.depth);
                &p.nodes()[node_index(path)]
            }
        }
    }

    /// Traverse upwards by `d` bits. If d is greater than or equal to the current position length,
    /// move to the root.
    pub fn traverse_parents(&mut self, d: u8) {
        let d = core::cmp::min(self.depth, d);
        let prev_depth = self.depth;
        self.depth -= d;

        if self.depth == 0 {
            self.location = CursorLocation::Root;
            return;
        }

        let prev_page_depth = (prev_depth as usize + DEPTH - 1) / DEPTH;
        let new_page_depth = (self.depth as usize + DEPTH - 1) / DEPTH;

        for _ in new_page_depth..prev_page_depth {
            // sanity: always `Page` unless depth is zero, checked above.
            if let CursorLocation::Page(p) = self.location {
                let parent_id = parent_page_id(*p.id());
                let parent_page = self.pages.get(&parent_id).expect(
                    "parent pages must have been traversed through, therefore are stored; qed",
                );

                self.location = CursorLocation::Page(parent_page);
            }
        }
    }

    /// Traverse into the two locations below the current node.
    ///
    /// Provide a closure which peeks at both values and informs the node which path to take:
    /// `false` is left and `true` is right.
    ///
    /// It is the caller's responsibility to ensure that the current node is an internal node. If
    /// it is not, then the cursor will traverse into the empty locations below without complaining.
    ///
    /// This may attempt to load a page and fail, if the current location is at the end of the
    /// current page.
    pub fn traverse_children(
        &mut self,
        f: impl FnOnce(&Node, &Node) -> bool,
    ) -> Result<(), MissingPage> {
        let (left_idx, right_idx) = if self.depth as usize % DEPTH == 0 {
            // attempt to load next page if we are at the end of our previous page or the root.
            let page_id = match self.location {
                CursorLocation::Root => ROOT_PAGE_ID,
                CursorLocation::Page(p) => {
                    let child_page_idx = last_page_path(&self.path, self.depth).load_be::<u8>();
                    child_page_id(*p.id(), child_page_idx)
                }
            };

            let page = self.pages.get(&page_id).ok_or(MissingPage)?;
            self.location = CursorLocation::Page(page);

            child_node_indices(BitSlice::empty())
        } else {
            let path = last_page_path(&self.path, self.depth);
            child_node_indices(path)
        };

        // invariant: cursor location is always `Page` at this point onwards.
        let bit = match self.location {
            CursorLocation::Root => panic!("Root means depth = 0, we loaded next page; qed"),
            CursorLocation::Page(p) => f(&p.nodes()[left_idx], &p.nodes()[right_idx]),
        };

        // Update the cursor's lookup path.
        self.path
            .view_bits_mut::<Msb0>()
            .set(self.depth as usize, bit);
        self.depth += 1;

        Ok(())
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

// Transform a bit-path to the indices of the two child positions in a page.
//
// The expected length of the page path is between 0 and `DEPTH - 1`, inclusive. All bits beyond
// `DEPTH - 1` are ignored.
fn child_node_indices(page_path: &BitSlice<u8, Msb0>) -> (usize, usize) {
    if page_path.is_empty() {
        return (0, 1);
    }
    let depth = core::cmp::min(DEPTH - 1, page_path.len());

    // parent is at (2^depth - 2) + as_uint(parent)
    // children are at (2^(depth+1) - 2) + as_uint(parent)*2 + (0 or 1)
    let base = (1 << depth + 1) - 2 + 2 * page_path[..depth].load_be::<usize>();
    (base, base + 1)
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
    fn child_node_index_mapping_is_complete() {
        let mut positions = vec![None; NODES_PER_PAGE];

        check_recursive(BitVec::<u8, Msb0>::new(), &mut positions);
        fn check_recursive(b: BitVec<u8, Msb0>, positions: &mut [Option<BitVec<u8, Msb0>>]) {
            let (left_pos, right_pos) = child_node_indices(&b[..]);

            if let Some(other) = &positions[left_pos] {
                panic!("{} and {} both map to {}", other, b, left_pos);
            }
            if let Some(other) = &positions[right_pos] {
                panic!("{} and {} both map to {}", other, b, right_pos);
            }
            positions[left_pos] = Some(b.clone());
            positions[right_pos] = Some(b.clone());

            if b.len() == DEPTH - 1 {
                return;
            }

            let mut left = b.clone();
            let mut right = b;
            left.push(false);
            right.push(true);

            // check consistency.
            assert_eq!(left_pos, node_index(&left));
            assert_eq!(right_pos, node_index(&right));

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

    // TODO: after path id functions, test cursor
}
