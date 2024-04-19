use crate::{page::DEPTH, page_id::ChildPageIndex, trie::KeyPath};
use alloc::fmt;
use bitvec::prelude::*;

/// Encapsulates logic for moving around in paged storage for a binary trie.
#[derive(Clone, PartialEq, Eq)]
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
        assert_ne!(depth, 0, "depth must be non-zero");
        let page_path = last_page_path(&path, depth);
        TriePosition {
            path,
            depth,
            node_index: node_index(&page_path),
        }
    }

    /// Parse a `TriePosition` from a bit string.
    #[cfg(test)]
    pub fn from_str(s: &str) -> Self {
        let mut bitvec = BitVec::<u8, Msb0>::new();
        if s.len() > 256 {
            panic!("bit string too long");
        }
        for ch in s.chars() {
            match ch {
                '0' => bitvec.push(false),
                '1' => bitvec.push(true),
                _ => panic!("invalid character in bit string"),
            }
        }
        let node_index = node_index(&bitvec);
        let depth = bitvec.len() as u8;
        bitvec.resize(256, false);
        // Unwrap: resized to 256 bit, or 32 bytes, above.
        let path = bitvec.as_raw_slice().try_into().unwrap();
        Self {
            path,
            depth,
            node_index,
        }
    }

    /// Whether the position is at the root.
    pub fn is_root(&self) -> bool {
        self.depth == 0
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
    ///
    /// Panics on depth out of range.
    pub fn down(&mut self, bit: bool) {
        assert_ne!(self.depth, 255, "can't descend past 255 bits");
        if self.depth as usize % DEPTH == 0 {
            self.node_index = bit as usize;
        } else {
            let children = self.child_node_indices();
            self.node_index = if bit {
                children.right()
            } else {
                children.left()
            };
        }
        self.path
            .view_bits_mut::<Msb0>()
            .set(self.depth as usize, bit);
        self.depth += 1;
    }

    /// Move the position up by `d` bits.
    ///
    /// Panics if `d` is greater than the current depth.
    pub fn up(&mut self, d: u8) {
        let prev_depth = self.depth;
        let Some(new_depth) = self.depth.checked_sub(d) else {
            panic!("can't move up by {} bits from depth {}", d, prev_depth)
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
    ///
    /// Panic if at the root.
    pub fn sibling(&mut self) {
        assert_ne!(self.depth, 0, "can't move to sibling of root node");
        let bits = self.path.view_bits_mut::<Msb0>();
        let i = self.depth as usize - 1;
        bits.set(i, !bits[i]);
        self.node_index = sibling_index(self.node_index);
    }

    /// Peek at the last bit of the path.
    ///
    /// Panics if at the root.
    pub fn peek_last_bit(&self) -> bool {
        assert_ne!(self.depth, 0, "can't peek at root node");
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
    pub fn child_node_indices(&self) -> ChildNodeIndices {
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
        ChildNodeIndices(left)
    }

    /// Get the index of the sibling node within a page.
    pub fn sibling_index(&self) -> usize {
        sibling_index(self.node_index)
    }

    /// Get the index of the current node within a page.
    pub fn node_index(&self) -> usize {
        self.node_index
    }

    /// Get the number of bits traversed in the current page.
    ///
    /// Note that every page has traversed at least 1 bit, therefore the return value would be
    /// between 1 and `DEPTH`, with the exception of the root node, which has traversed 0 bits.
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

impl fmt::Debug for TriePosition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let cropped_path = &self.path[..self.depth as usize];
        if cropped_path.len() == 0 {
            write!(f, "TriePosition(root)")
        } else {
            write!(
                f,
                "TriePosition({})",
                hex::encode(&self.path[..self.depth as usize]),
            )
        }
    }
}

/// Child node indices.
#[derive(Debug, Clone, Copy)]
pub struct ChildNodeIndices(usize);

impl ChildNodeIndices {
    /// Create from a left child index.
    pub fn from_left(left: usize) -> Self {
        ChildNodeIndices(left)
    }

    /// Get the index of the left child.
    pub fn left(&self) -> usize {
        self.0
    }
    /// Get the index of the right child.
    pub fn right(&self) -> usize {
        self.0 + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[ignore = "to be addressed in followups"]
    #[test]
    fn path_can_go_deeper_255_bit() {
        let mut p = TriePosition::from_str(
            "1010101010101010101010101010101010101010101010101010101010101010\
            1010101010101010101010101010101010101010101010101010101010101010\
            1010101010101010101010101010101010101010101010101010101010101010\
            101010101010101010101010101010101010101010101010101010101010101",
        );
        assert_eq!(p.depth as usize, 255);
        p.down(false);
    }
}
