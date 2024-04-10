//! This module contains all the relevant methods to work with PageIds.
//!
//! A PageId is an unique identifier for a Page in a tree of pages with branching factor 2^6 and
//! a maximum depth of 42, with the root page counted as depth 0.
//!
//! Each PageId consists of a list of numbers between 0 and 2^6 - 1, which encodes a path through
//! the tree. The list may have between 0 and 42 (inclusive) items.
//!
//! Page IDs also have a disambiguated 256-bit representation which is given by starting with a
//! blank bit pattern, and then repeatedly shifting it to the left by 6 bits, then adding the next
//! child index, then adding 1. This disambiguated representation uniquely encodes all the page IDs
//! in a fixed-width bit pattern as, essentially, a base-64 integer.

use crate::{page::DEPTH, trie::KeyPath};
use bitvec::prelude::*;
use ruint::Uint;

/// A unique ID for a page.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialOrd, Ord)]
pub struct PageId {
    depth: usize,
    limbs: [u8; 42],
}

impl PartialEq for PageId {
    fn eq(&self, other: &Self) -> bool {
        if self.depth == other.depth {
            self.limbs[..self.depth] == other.limbs[..self.depth]
        } else {
            false
        }
    }
}

/// The root page is the one containing the sub-trie directly descending from the root node.
///
/// It has an ID consisting of all zeros.
pub const ROOT_PAGE_ID: PageId = PageId {
    depth: 0,
    limbs: [0; 42],
};

#[allow(unused)]
const LOWEST_ENCODED_42: Uint<256, 4> = Uint::from_be_bytes([
    0, 65, 4, 16, 65, 4, 16, 65, 4, 16, 65, 4, 16, 65, 4, 16, 65, 4, 16, 65, 4, 16, 65, 4, 16, 65,
    4, 16, 65, 4, 16, 65,
]);

const HIGHEST_ENCODED_42: Uint<256, 4> = Uint::from_be_bytes([
    16, 65, 4, 16, 65, 4, 16, 65, 4, 16, 65, 4, 16, 65, 4, 16, 65, 4, 16, 65, 4, 16, 65, 4, 16, 65,
    4, 16, 65, 4, 16, 64,
]);

/// The PageId of the leftmost page of the last layer of the page tree.
/// It is the lowest PageId beyond which all pages with a PageId equal to
/// or higher overflow if a child is attempted to be accessed
pub const LOWEST_42: PageId = PageId {
    depth: 42,
    limbs: [0; 42],
};

/// The PageId of the rightmost page of the last layer of the page tree.
/// It is the highest possible PageId present in the page tree.
pub const HIGHEST_42: PageId = PageId {
    depth: 42,
    limbs: [0b111111; 42],
};

const MAX_CHILD_INDEX: u8 = (1 << DEPTH) - 1;

impl PageId {
    /// Decode a page ID from its disambiguated representation.
    ///
    /// This can fall out of bounds.
    pub fn decode(bytes: [u8; 32]) -> Result<Self, InvalidPageIdBytes> {
        let uint = Uint::from_be_bytes(bytes);
        if uint > HIGHEST_ENCODED_42 {
            return Err(InvalidPageIdBytes);
        }

        let leading_zeros = uint.leading_zeros();
        let bit_count = 256 - leading_zeros;
        let sextets = (bit_count + 5) / 6;
        let first_sextet_start = 256 - sextets * 6;

        if bit_count == 0 {
            return Ok(PageId {
                depth: 0,
                limbs: [0u8; 42]
            });
        }

        // first we normalize the page ID: any sextet which is all zeros
        // is an artifact from adding 1 to a sextet which was 2^6 - 1.
        // by iterating the sextets from least significant to most significant and subtracting out
        // the additional '1', we can restore the bit pattern to an
        // ambiguous form and then iterate that. although the bit pattern will now be in an
        // ambiguous form, we know where the MSB was initially and can use that to determine the
        // correct number of limbs.
        let normalized_page_id = {
            let mut uint = uint;
            for i in (0..sextets).rev() {
                let bit_off = (sextets - i - 1) * 6;

                let sub = if i == 0 {
                    // check if the last sextet is now totally zeroed.
                    let new_bit_count = 256 - uint.leading_zeros();
                    let new_sextets = (new_bit_count + 5) / 6;
                    if new_sextets < sextets {
                        Uint::<256, 4>::from(0)
                    } else {
                        Uint::<256, 4>::from(1) << bit_off
                    }
                } else {
                    Uint::<256, 4>::from(1) << bit_off
                };
                uint -= sub;
            }
            uint.to_be_bytes::<32>()
        };

        let mut real_depth = None;
        let mut page_id = PageId {
            depth: 0,
            limbs: [0u8; 42],
        };
        for (i, sextet) in normalized_page_id.view_bits::<Msb0>()[first_sextet_start..]
            .rchunks(6)
            .enumerate()
        {
            if !sextet.is_empty() {
                let depth = sextets - i;
                if real_depth.is_none() {
                    real_depth = Some(depth);
                }
                page_id.limbs[depth - 1] = sextet.load_be::<u8>();
            }
        }

        page_id.depth = real_depth.unwrap_or(0);
        Ok(page_id)
    }

    /// Encode this page ID to its disambiguated representation.
    pub fn encode(&self) -> [u8; 32] {
        let mut uint = Uint::<256, 4>::default();
        for child in self.limbs[0..self.depth].iter().rev() {
            uint <<= 6;
            uint += Uint::<256,4>::from(child + 1);
        }

        uint.to_be_bytes::<32>()
    }

    /// Construct the Child PageId given the previous PageId and the child index.
    ///
    /// Child index must be a 6 bit integer, two most significant bits must be zero.
    /// Passed PageId must be a valid PageId and be located in a layer below 42 otherwise
    /// `PageIdOverflow` will be returned.
    pub fn child_page_id(&self, child_index: u8) -> Result<Self, ChildPageIdError> {
        if child_index > MAX_CHILD_INDEX {
            return Err(ChildPageIdError::InvalidChildIndex);
        }

        if self.depth >= 42 {
            return Err(ChildPageIdError::PageIdOverflow);
        }

        let mut new_page_id = self.clone();
        new_page_id.limbs[self.depth] = child_index;
        new_page_id.depth += 1;

        Ok(new_page_id)
    }

    /// Extract the Parent PageId given a PageId.
    ///
    /// If the provided PageId is the one pointing to the root,
    /// then itself is returned.
    pub fn parent_page_id(&self) -> Self {
        if *self == ROOT_PAGE_ID {
            return ROOT_PAGE_ID;
        }

        let mut new_page_id = self.clone();

        new_page_id.depth -= 1;
        new_page_id.limbs[new_page_id.depth] = 0;

        new_page_id
    }
}

/// The bytes cannot form a valid PageId because they define
/// a PageId bigger than the biggest valid one, the rightmost Page in the last layer.
#[derive(Debug, PartialEq)]
pub struct InvalidPageIdBytes;

/// Errors related to the construction of a Child PageId
#[derive(Debug, PartialEq)]
pub enum ChildPageIdError {
    /// Child index must be made by only the first 6 bits in a byte
    InvalidChildIndex,
    /// PageId was at the last layer of the page tree
    /// or it was too big to represent a valid page
    PageIdOverflow,
}

/// Iterator of PageIds over a KeyPath,
/// PageIds will be lazily constructed as needed
pub struct PageIdsIterator {
    key_path: Uint<256, 4>,
    page_id: Option<PageId>,
}

impl PageIdsIterator {
    /// Create a PageIds Iterator over a KeyPath
    pub fn new(key_path: KeyPath) -> Self {
        Self {
            key_path: Uint::from_be_bytes(key_path),
            page_id: Some(ROOT_PAGE_ID),
        }
    }
}

impl Iterator for PageIdsIterator {
    type Item = PageId;

    fn next(&mut self) -> Option<Self::Item> {
        let prev = self.page_id.take()?;

        let child_index = self.key_path.byte(31) >> 2;
        self.key_path <<= 6;
        self.page_id = prev.child_page_id(child_index).ok();
        Some(prev)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::page_id::PageIdsIterator;

    #[test]
    fn test_child_and_parent_page_id() {
        let mut page_id_1 = [0u8; 32]; // child index 6
        page_id_1[31] = 0b00000111;
        let page_id_1 = PageId::decode(page_id_1).unwrap();

        assert_eq!(Ok(page_id_1.clone()), ROOT_PAGE_ID.child_page_id(6));
        assert_eq!(ROOT_PAGE_ID, page_id_1.parent_page_id());

        let mut page_id_2 = [0u8; 32]; // child index 4
        page_id_2[31] = 0b11000101;
        page_id_2[30] = 0b00000001;
        let page_id_2 = PageId::decode(page_id_2).unwrap();

        assert_eq!(Ok(page_id_2.clone()), page_id_1.child_page_id(4));
        assert_eq!(page_id_1, page_id_2.parent_page_id());

        let mut page_id_3 = [0u8; 32]; // child index 63
        page_id_3[31] = 0b10000000;
        page_id_3[30] = 0b01110001;
        let page_id_3 = PageId::decode(page_id_3).unwrap();

        assert_eq!(
            Ok(page_id_3.clone()),
            page_id_2.child_page_id(MAX_CHILD_INDEX)
        );
        assert_eq!(page_id_2, page_id_3.parent_page_id());
    }

    #[test]
    fn test_page_ids_iterator() {
        // key_path = 0b000001|000010|0...
        let mut key_path = [0u8; 32];
        key_path[0] = 0b00000100;
        key_path[1] = 0b00100000;

        let mut page_id_1 = [0u8; 32];
        page_id_1[31] = 0b00000010; // 0b000001 + 1
        let page_id_1 = PageId::decode(page_id_1).unwrap();
        let mut page_id_2 = [0u8; 32];
        page_id_2[31] = 0b10000011; // (0b000001 + 1 << 6) + 0b000010 + 1
        let page_id_2 = PageId::decode(page_id_2).unwrap();

        let mut page_ids = PageIdsIterator::new(key_path);
        assert_eq!(page_ids.next(), Some(ROOT_PAGE_ID));
        assert_eq!(page_ids.next(), Some(page_id_1));
        assert_eq!(page_ids.next(), Some(page_id_2));

        // key_path = 0b000010|111111|0...
        let mut key_path = [0u8; 32];
        key_path[0] = 0b00001011;
        key_path[1] = 0b11110000;

        let mut page_id_1 = [0u8; 32];
        page_id_1[31] = 0b00000011; // 0b000010 + 1
        let page_id_1 = PageId::decode(page_id_1).unwrap();
        let mut page_id_2 = [0u8; 32];
        page_id_2[31] = 0b0000000;
        page_id_2[30] = 0b0000001; // (0b00000011 << 6) + 0b111111 + 1 = (0b00000011 + 1) << 6
        let page_id_2 = PageId::decode(page_id_2).unwrap();

        let mut page_ids = PageIdsIterator::new(key_path);
        assert_eq!(page_ids.next(), Some(ROOT_PAGE_ID));
        assert_eq!(page_ids.next(), Some(page_id_1));
        assert_eq!(page_ids.next(), Some(page_id_2));
    }

    #[test]
    fn invalid_child_index() {
        assert_eq!(
            Err(ChildPageIdError::InvalidChildIndex),
            ROOT_PAGE_ID.child_page_id(0b01010000)
        );

        assert_eq!(
            Err(ChildPageIdError::InvalidChildIndex),
            ROOT_PAGE_ID.child_page_id(0b10000100)
        );

        assert_eq!(
            Err(ChildPageIdError::InvalidChildIndex),
            ROOT_PAGE_ID.child_page_id(0b11000101)
        );
    }

    #[test]
    fn test_invalid_page_id() {
        // position 255
        let mut page_id = [0u8; 32];
        page_id[0] = 128;
        assert_eq!(Err(InvalidPageIdBytes), PageId::decode(page_id));

        // position 252
        let mut page_id = [0u8; 32];
        page_id[0] = 128;
        assert_eq!(Err(InvalidPageIdBytes), PageId::decode(page_id));
    }

    #[test]
    fn test_page_id_overflow() {
        let first_page_last_layer = PageIdsIterator::new([0u8; 32]).last().unwrap();
        let last_page_last_layer = PageIdsIterator::new([255; 32]).last().unwrap();
        assert_eq!(
            Err(ChildPageIdError::PageIdOverflow),
            first_page_last_layer.child_page_id(0)
        );
        assert_eq!(
            Err(ChildPageIdError::PageIdOverflow),
            last_page_last_layer.child_page_id(0)
        );

        // position 255
        let page_id = HIGHEST_42;
        assert_eq!(
            Err(ChildPageIdError::PageIdOverflow),
            page_id.child_page_id(0)
        );

        // any PageId bigger than LOWEST_42 must overflow
        let mut page_id = LOWEST_ENCODED_42.to_be_bytes();
        page_id[31] = 255;
        let page_id = PageId::decode(page_id).unwrap();
        assert_eq!(
            Err(ChildPageIdError::PageIdOverflow),
            page_id.child_page_id(0)
        );

        // position 245
        let mut page_id = [0u8; 32];
        page_id[1] = 32;
        let page_id = PageId::decode(page_id).unwrap();
        assert!(page_id.child_page_id(0).is_ok());

        // neither of those two have to panic if called at most 41 times
        let mut low = ROOT_PAGE_ID;
        let mut high = ROOT_PAGE_ID;
        for _ in 0..42 {
            low = low.child_page_id(0).unwrap();
            high = high.child_page_id(MAX_CHILD_INDEX).unwrap();
        }
    }
}
