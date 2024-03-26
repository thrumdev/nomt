//! TODO: add docs

use crate::{page::DEPTH, trie::KeyPath};
use bitvec::{prelude::*, slice::ChunksExact};

/// A unique ID for a page.
pub type PageId = [u8; 32];

/// The root page is the one containing the sub-trie directly descending from the root node.
///
/// It has an ID consisting of all zeros.
pub const ROOT_PAGE_ID: [u8; 32] = [0; 32];

const MAX_CHILD_INDEX: u8 = (1 << DEPTH) - 1;

pub fn child_page_id(mut page_id: PageId, child_index: u8) -> PageId {
    // TODO: proper error handling
    assert!(child_index & 0b11000000 == 0);

    let page_id_bits = page_id.view_bits_mut::<Msb0>();

    if child_index ^ MAX_CHILD_INDEX == 0 {
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
    page_id
}

pub fn parent_page_id(mut page_id: PageId) -> PageId {
    let page_id_bits = page_id.view_bits_mut::<Msb0>();

    // sub 1
    let trailing_zeros = page_id_bits.trailing_zeros();
    page_id_bits[256 - trailing_zeros..256].fill(true);
    page_id_bits.set(256 - trailing_zeros - 1, false);

    page_id_bits.shift_right(DEPTH);
    page_id
}

/// Each Page is identified by a unique PageId,
/// Root Page has id 0, its leftmost child 1 and so on.
///
/// There are 2^250 possible pages, so 250 bits are needed to represent all of them.
///
/// An array of bytes will be used as a container for the PageId,
/// bytes will be treated as a 256-bit big-endian integer.
///
/// Given a KeyPath, it will be divided into chunks of 6 bits (sextet), forming
/// an array of sextets. Each PageId will be constructed in the following manner:
///
/// page_ids[i] = (prev_page_id << d) + int(sextets[i]) + 1
///
/// The following struct implements an Iterator of PageIds over a KeyPath
/// to lazily construct them as needed
struct PageIdsIterator<'a> {
    sextets: ChunksExact<'a, u8, Msb0>,
    page_id: Option<PageId>,
}

impl<'a> PageIdsIterator<'a> {
    fn new(key_path: &'a KeyPath) -> Self {
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
                // TODO: could be nice to modify prev_page_id in place
                // to avoid cloning twice
                Some(child_page_id(prev_page_id, child_index))
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

        assert_eq!(page_id_1, child_page_id(page_id_0, 6));
        assert_eq!(page_id_0, parent_page_id(page_id_1));

        let mut page_id_2 = page_id_0; // child index 4
        page_id_2[31] = 0b11000101;
        page_id_2[30] = 0b00000001;

        assert_eq!(page_id_2, child_page_id(page_id_1, 4));
        assert_eq!(page_id_1, parent_page_id(page_id_2));

        let mut page_id_3 = page_id_0; // child index 63
        page_id_3[31] = 0b10000000;
        page_id_3[30] = 0b01110001;

        assert_eq!(page_id_3, child_page_id(page_id_2, MAX_CHILD_INDEX));
        assert_eq!(page_id_2, parent_page_id(page_id_3));
    }
    #[test]
    fn test_get_page_ids() {
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
}
