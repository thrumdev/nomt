//! This module contains all the relevant methods to work with PageIds.
//!
//! A PageId is an unique identifier for a Page,
//! Root Page has id 0, its leftmost child 1 and so on.
//!
//! There are 2^250 possible pages, so 250 bits are needed to represent all of them.
//!
//! An array of bytes will be used as a container for the PageId,
//! bytes will be treated as a 256-bit big-endian integer.
//!
//! Given a KeyPath, it will be divided into chunks of 6 bits (sextet), forming
//! an array of sextets. Each sextet will act as a child_index for the next page.
//! Each PageId will be constructed in the following manner:
//!
//! page_id[i] = prev_page_id * 2^6 + sextet[i] + 1
//!
//! Only the first 42 sextets represent valid child indexes in a KeyPath,
//! and the last four bits are discarded.

use crate::{page::DEPTH, trie::KeyPath};
use bitvec::{prelude::*, slice::ChunksExact};

/// A unique ID for a page.
pub type PageId = [u8; 32];

/// The root page is the one containing the sub-trie directly descending from the root node.
///
/// It has an ID consisting of all zeros.
pub const ROOT_PAGE_ID: [u8; 32] = [0; 32];

/// The PageId of the leftmost page of the last layer of the page tree.
/// It is the lowest PageId beyond which all pages with a PageId equal to
/// or higher overflow if a child is attempted to be accessed
pub const LOWEST_42: [u8; 32] = [
    0, 65, 4, 16, 65, 4, 16, 65, 4, 16, 65, 4, 16, 65, 4, 16, 65, 4, 16, 65, 4, 16, 65, 4, 16, 65,
    4, 16, 65, 4, 16, 65,
];

const MAX_CHILD_INDEX: u8 = (1 << DEPTH) - 1;

/// Construct the Child PageId given the previous PageId and the child index.
///
/// Child index must be a 6 bit integer, two most significant bits must be zero.
/// Passed PageId must be a valid PageId and be located in a layer below 42 otherwise
/// `PageIdOverflow` will be returned.
pub fn child_page_id(mut page_id: PageId, child_index: u8) -> Result<PageId, ChildPageIdError> {
    if child_index > MAX_CHILD_INDEX {
        return Err(ChildPageIdError::InvalidChildIndex);
    }

    // RootPageId is at layer zero, the second layer has index one
    // and so on. There are 43 layers.
    //
    // For each layer between 1 and 42 the lowest id for layer I is:
    //
    // L(I) = sum_i(2^(6 * i)) where i goes from 0 to I - 1
    //
    // while the highest id for each layer is:
    //
    // H(I) = L(I) + 2^(6 * I) - 1 = sum_i(2^(6 * i)) where i goes from 1 to I
    //
    // So for each layer bigger than 0 the range of PageIds are:
    //
    // [L(I)..=H(I)]
    //
    // And L(I) = H(I - 1) + 1
    //
    // Any PageId larger than or equal to L(42) will overflow.
    // L(42) and H(41) have the same most significant bit
    // set to one at position 246, resulting in 9 leading zeros.
    // If the PageId passed as an argument has fewer than 9 leading zeros,
    // it will definitely overflow. If it has exactly 9 leading zeros,
    // it needs to be compared to L(42). If it is higher than L(42),
    // it will overflow.
    let leading_zeros = page_id.view_bits::<Msb0>().leading_zeros();
    if leading_zeros < 9 {
        return Err(ChildPageIdError::PageIdOverflow);
    } else if leading_zeros == 9 {
        let mut lower = false;

        for (byte, byte_lowest) in page_id.iter().zip(LOWEST_42.iter()) {
            if byte < byte_lowest {
                lower = true;
                break;
            }
        }

        if !lower {
            return Err(ChildPageIdError::PageIdOverflow);
        }
    }

    let page_id_bits = page_id.view_bits_mut::<Msb0>();

    if child_index == MAX_CHILD_INDEX {
        // if the child_index is the maximum then adding one would overlow
        // becoming 2^6
        //
        // The formula then becomes: next_page_id = (prev_page_id + 1) << d
        //
        // Adding one to prev_page_id meas starting from the lsb changing all 1 to 0
        // and as soon as a zero is found making it 1 and stop

        // add 1
        let trailing_ones = page_id_bits.trailing_ones();
        page_id_bits[256 - trailing_ones..256].fill(false);
        page_id_bits.set(256 - trailing_ones - 1, true);

        page_id_bits.shift_left(DEPTH);
    } else {
        // If the child_index is less than MAX_CHILD_INDEX,
        // after shifting the previous page id, it can be ORed with child_index + 1

        // next_page_id = (prev_page_id << d) | (child_index + 1)
        page_id_bits.shift_left(DEPTH);
        page_id_bits[256 - DEPTH..256]
            .copy_from_bitslice(&(child_index + 1).view_bits::<Msb0>()[2..]);
    };
    Ok(page_id)
}

/// Extract the Parent PageId given a PageId.
///
/// If the provided PageId is the one pointing to the root,
/// then itself is returned.
pub fn parent_page_id(mut page_id: PageId) -> PageId {
    if page_id == ROOT_PAGE_ID {
        return ROOT_PAGE_ID;
    }

    let page_id_bits = page_id.view_bits_mut::<Msb0>();

    // sub 1
    let trailing_zeros = page_id_bits.trailing_zeros();
    page_id_bits[256 - trailing_zeros..256].fill(true);
    page_id_bits.set(256 - trailing_zeros - 1, false);

    page_id_bits.shift_right(DEPTH);
    page_id
}

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
pub struct PageIdsIterator<'a> {
    sextets: ChunksExact<'a, u8, Msb0>,
    page_id: Option<PageId>,
}

impl<'a> PageIdsIterator<'a> {
    /// Create a PageIds Iterator over a KeyPath
    pub fn new(key_path: &'a KeyPath) -> Self {
        Self {
            // Last bits will not be used as a sextet to identify a page
            sextets: key_path.view_bits::<Msb0>().chunks_exact(DEPTH),
            page_id: None,
        }
    }
}

impl<'a> Iterator for PageIdsIterator<'a> {
    type Item = PageId;

    fn next(&mut self) -> Option<Self::Item> {
        let new_page_id = match self.page_id {
            None => Some(ROOT_PAGE_ID),
            Some(prev_page_id) => {
                // If sextets are finished, then there are no more pages to load
                let child_index = self.sextets.next()?.load_be::<u8>();
                Some(child_page_id(prev_page_id, child_index).expect(
                    "Child index is 6 bits and Pages do not go deeper than the maximum layer, 42",
                ))
            }
        };
        self.page_id = new_page_id;
        new_page_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::page_id::PageIdsIterator;

    #[test]
    fn test_child_and_parent_page_id() {
        let page_id_0 = [0u8; 32];

        let mut page_id_1 = [0u8; 32]; // child index 6
        page_id_1[31] = 0b00000111;

        assert_eq!(Ok(page_id_1), child_page_id(page_id_0, 6));
        assert_eq!(page_id_0, parent_page_id(page_id_1));

        let mut page_id_2 = page_id_0; // child index 4
        page_id_2[31] = 0b11000101;
        page_id_2[30] = 0b00000001;

        assert_eq!(Ok(page_id_2), child_page_id(page_id_1, 4));
        assert_eq!(page_id_1, parent_page_id(page_id_2));

        let mut page_id_3 = page_id_0; // child index 63
        page_id_3[31] = 0b10000000;
        page_id_3[30] = 0b01110001;

        assert_eq!(Ok(page_id_3), child_page_id(page_id_2, MAX_CHILD_INDEX));
        assert_eq!(page_id_2, parent_page_id(page_id_3));
    }

    #[test]
    fn test_page_ids_iterator() {
        // key_path = 0b000001|000010|0...
        let mut key_path = [0u8; 32];
        key_path[0] = 0b00000100;
        key_path[1] = 0b00100000;

        let page_id_0 = [0u8; 32];
        let mut page_id_1 = [0u8; 32];
        page_id_1[31] = 0b00000010; // 0b000001 + 1
        let mut page_id_2 = [0u8; 32];
        page_id_2[31] = 0b10000011; // (0b000001 + 1 << 6) + 0b000010 + 1

        let mut page_ids = PageIdsIterator::new(&key_path);
        assert_eq!(page_ids.next(), Some(page_id_0));
        assert_eq!(page_ids.next(), Some(page_id_1));
        assert_eq!(page_ids.next(), Some(page_id_2));

        // key_path = 0b000010|111111|0...
        let mut key_path = [0u8; 32];
        key_path[0] = 0b00001011;
        key_path[1] = 0b11110000;

        let page_id_0 = [0u8; 32];
        let mut page_id_1 = [0u8; 32];
        page_id_1[31] = 0b00000011; // 0b000010 + 1
        let mut page_id_2 = [0u8; 32];
        page_id_2[31] = 0b0000000;
        page_id_2[30] = 0b0000001; // (0b00000011 << 6) + 0b111111 + 1 = (0b00000011 + 1) << 6

        let mut page_ids = PageIdsIterator::new(&key_path);
        assert_eq!(page_ids.next(), Some(page_id_0));
        assert_eq!(page_ids.next(), Some(page_id_1));
        assert_eq!(page_ids.next(), Some(page_id_2));
    }

    #[test]
    fn invalid_child_index() {
        assert_eq!(
            Err(ChildPageIdError::InvalidChildIndex),
            child_page_id(ROOT_PAGE_ID, 0b01010000)
        );

        assert_eq!(
            Err(ChildPageIdError::InvalidChildIndex),
            child_page_id(ROOT_PAGE_ID, 0b10000100)
        );

        assert_eq!(
            Err(ChildPageIdError::InvalidChildIndex),
            child_page_id(ROOT_PAGE_ID, 0b11000101)
        );
    }

    #[test]
    fn test_page_id_overflow() {
        let first_page_last_layer = PageIdsIterator::new(&[0u8; 32]).last().unwrap();
        let last_page_last_layer = PageIdsIterator::new(&[255; 32]).last().unwrap();
        assert_eq!(
            Err(ChildPageIdError::PageIdOverflow),
            child_page_id(first_page_last_layer, 0)
        );
        assert_eq!(
            Err(ChildPageIdError::PageIdOverflow),
            child_page_id(last_page_last_layer, 0)
        );

        // position 255
        let mut page_id = ROOT_PAGE_ID;
        page_id[0] = 128;
        assert_eq!(
            Err(ChildPageIdError::PageIdOverflow),
            child_page_id(page_id, 0)
        );

        // any PageId bigger than LOWEST_42 must overflow
        page_id = LOWEST_42;
        page_id[31] = 255;
        assert_eq!(
            Err(ChildPageIdError::PageIdOverflow),
            child_page_id(page_id, 0)
        );

        // position 245
        page_id = ROOT_PAGE_ID;
        page_id[1] = 32;
        assert!(child_page_id(page_id, 0).is_ok());

        // neither of those two have to panic if called at most 41 times
        let mut low = ROOT_PAGE_ID;
        let mut high = ROOT_PAGE_ID;
        for _ in 0..42 {
            low = child_page_id(low, 0).unwrap();
            high = child_page_id(high, MAX_CHILD_INDEX).unwrap();
        }
    }
}
