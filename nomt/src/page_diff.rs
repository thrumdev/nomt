use crate::{io, page_cache::NODES_PER_PAGE};
use bitvec::prelude::*;

/// A bitfield tracking which nodes have changed within a page.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct PageDiff {
    /// Each bit indicates whether the node at the corresponding index has changed.
    /// Later bits are unused.
    changed_nodes: BitArray<[u8; 16], Lsb0>,
}

impl PageDiff {
    /// Create a new page diff from bytes.
    ///
    /// Returns `None` if any of unused bits are set to 1.
    pub fn from_bytes(bytes: [u8; 16]) -> Option<Self> {
        let changed_nodes = BitArray::<[u8; 16], Lsb0>::new(bytes);
        // Check if the two last bits are set to 1
        assert!(changed_nodes.len() == 128);
        if changed_nodes[126] || changed_nodes[127] {
            return None;
        }
        Some(Self { changed_nodes })
    }

    /// Note that some 32-byte slot in the page data has changed.
    /// The acceptable range is 0..NODES_PER_PAGE
    pub fn set_changed(&mut self, slot_index: usize) {
        assert!(slot_index < NODES_PER_PAGE);
        self.changed_nodes.set(slot_index, true);
    }

    /// Given the page data, collect the nodes that have changed according to this diff.
    pub fn pack_changed_nodes<'a, 'b: 'a>(
        &'b self,
        page: &'a io::Page,
    ) -> impl Iterator<Item = [u8; 32]> + 'a {
        self.changed_nodes.iter_ones().map(|node_index| {
            let start = node_index * 32;
            let end = start + 32;
            page[start..end].try_into().unwrap()
        })
    }

    /// Given the changed nodes, apply them to the given page according to the diff.
    ///
    /// Panics if the number of changed nodes doesn't equal to the number of nodes
    /// this diff recorded.
    pub fn unpack_changed_nodes(&self, nodes: &[[u8; 32]], page: &mut io::Page) {
        assert!(self.changed_nodes.count_ones() == nodes.len());
        for (node_index, node) in self.changed_nodes.iter_ones().zip(nodes) {
            let start = node_index * 32;
            let end = start + 32;
            page[start..end].copy_from_slice(&node[..]);
        }
    }

    /// Returns the number of changed nodes. Capped at [NODES_PER_PAGE].
    pub fn count(&self) -> usize {
        self.changed_nodes.count_ones()
    }

    /// Get raw bytes representing the PageDiff
    pub fn as_bytes(&self) -> [u8; 16] {
        self.changed_nodes.data
    }
}

#[test]
fn ensure_cap() {
    assert_eq!(NODES_PER_PAGE, 126);
}
