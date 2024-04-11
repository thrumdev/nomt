use crate::{page::DEPTH, page_id::ChildPageIndex, trie::KeyPath};
use bitvec::prelude::*;

// /// The path to a key. All paths have a 256 bit fixed length.
// #[derive(Clone, Copy, PartialEq, Eq)]
// pub struct KeyPath(pub [u8; 32]);

// impl fmt::Display for KeyPath {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         write!(f, "0x{}", hex::encode(&self.0))
//     }
// }

// impl fmt::Debug for KeyPath {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         write!(f, "0x{}", hex::encode(&self.0))
//     }
// }

/// Encapsulates logic for moving around in paged storage for a binary trie.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TriePosition {
    // The bits after depth are irrelevant.
    path: [u8; 32],
    depth: u8,
    node_index: usize,
}

impl TriePosition {
    /// Create a new `TriePosition` at the root.
    pub fn new() -> Self {
        TriePosition {
            path: [0; 32],
            depth: 0,
            node_index: 0,
        }
    }

    /// Create a new `TriePosition` based on the first `depth` bits of `path`.
    ///
    /// Panics if depth is zero.
    pub fn from_path_and_depth(path: KeyPath, depth: u8) -> Self {
        let page_path = last_page_path(&path, depth);
        TriePosition {
            path,
            depth,
            node_index: node_index(&page_path),
        }
    }

    /// Get the current `depth` of the position.
    pub fn depth(&self) -> u8 {
        self.depth
    }

    /// Get the path to the current position.
    pub fn path(&self) -> &BitSlice<u8, Msb0> {
        &self.path.view_bits::<Msb0>()[..self.depth as usize]
    }

    /// Move the position down by 1, towards either the left or right child.
    /// Panics on depth out of range.
    pub fn down(&mut self, bit: bool) {
        if self.depth == 255 {
            panic!();
        }
        if self.depth as usize % DEPTH == 0 {
            self.node_index = bit as usize;
        } else {
            let (l, r) = self.child_node_indices();
            self.node_index = if bit { r } else { l };
        }
        self.path
            .view_bits_mut::<Msb0>()
            .set(self.depth as usize, bit);
        self.depth += 1;
    }

    /// Move the position up by `d` bits. This panics if `d` is greater than the current depth.
    pub fn up(&mut self, d: u8) {
        let prev_depth = self.depth;
        let Some(new_depth) = self.depth.checked_sub(d) else {
            panic!();
        };
        if new_depth == 0 {
            *self = TriePosition::new();
            return;
        }

        self.depth = new_depth;
        let prev_page_depth = (prev_depth as usize + DEPTH - 1) / DEPTH;
        let new_page_depth = (self.depth as usize + DEPTH - 1) / DEPTH;
        if prev_page_depth == new_page_depth {
            let depth_in_page = self.depth_in_page();
            for depth in (depth_in_page..depth_in_page + d as usize)
                .rev()
                .map(|x| x + 1)
            {
                self.node_index = parent_node_index(self.node_index, depth);
            }
        } else {
            let path = last_page_path(&self.path, self.depth);
            self.node_index = node_index(path);
        }
    }

    /// Move the position to the sibling node.
    pub fn sibling(&mut self) {
        if self.depth == 0 {
            panic!();
        }
        let bits = self.path.view_bits_mut::<Msb0>();
        let i = self.depth as usize - 1;
        bits.set(i, !bits[i]);
        self.node_index = sibling_index(self.node_index);
    }

    /// Peek at the last bit of the path. Panics if at the root.
    pub fn peek_last_bit(&self) -> bool {
        if self.depth == 0 {
            panic!();
        }
        let this_bit_idx = self.depth as usize - 1;
        // unwrap: depth != 0 above
        let bit = *self.path.view_bits::<Msb0>().get(this_bit_idx).unwrap();
        bit
    }

    /// Get the child page index, relative to the current page,
    /// where the children of the current node are stored.
    ///
    /// Panics if the position is not in the last layer of the page.
    pub fn child_page_index(&self) -> ChildPageIndex {
        assert!(self.node_index >= 62);
        ChildPageIndex::new(bottom_node_index(self.node_index)).unwrap()
    }

    /// Get the child page index, relative to the current page,
    /// where the children of the sibling node are stored.
    ///
    /// Panics if the position is not in the last layer of the page.
    pub fn sibling_child_page_index(&self) -> ChildPageIndex {
        ChildPageIndex::new(bottom_node_index(sibling_index(self.node_index))).unwrap()
    }

    /// Transform a bit-path to the index in a page corresponding to the child node indices.
    ///
    /// The expected length of the page path is between 0 and `DEPTH` - 1, inclusive.
    /// A length out of range returns `None`.
    pub fn child_node_indices(&self) -> (usize, usize) {
        let node_index = self.node_index;
        let depth = self.depth_in_page();
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

    /// Get the index of the sibling node within a page.
    pub fn sibling_index(&self) -> usize {
        sibling_index(self.node_index)
    }

    /// Get the index of the current node within a page.
    pub fn node_index(&self) -> usize {
        self.node_index
    }

    /// Get the depth of the position within the current page.
    ///
    /// Returns `0` at the root. Otherwise, returns a value in range 1..=DEPTH.
    pub fn depth_in_page(&self) -> usize {
        if self.depth == 0 {
            0
        } else {
            self.depth as usize - ((self.depth as usize - 1) / DEPTH) * DEPTH
        }
    }
}

// extract the relevant portion of the key path to the last page. panics on empty path.
fn last_page_path(path: &[u8; 32], depth: u8) -> &BitSlice<u8, Msb0> {
    if depth == 0 {
        panic!();
    }
    let prev_page_end = ((depth as usize - 1) / DEPTH) * DEPTH;
    &path.view_bits::<Msb0>()[prev_page_end..depth as usize]
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

/// Given a node index, get the index of the sibling.
fn sibling_index(node_index: usize) -> usize {
    if node_index % 2 == 0 {
        node_index + 1
    } else {
        node_index - 1
    }
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
