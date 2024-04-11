//! This module contains all the relevant methods to work with PageIds.
//!
//! A PageId is an unique identifier for a Page,
//! Root Page has id 0, its leftmost child 1 and so on.
//!
//! PageId is a 256-bit big-endian integer because there are 2^250 possible pages,
//! requiring at least 250 bits for representation
//!
//! Given a KeyPath, it will be divided into chunks of 6 bits (sextet), forming
//! an array of sextets. Each sextet will act as a child_index for the next page.
//! Each PageId will be constructed in the following manner:
//!
//! page_id[i] = prev_page_id * 2^6 + sextet[i] + 1
//!
//! Only the first 42 sextets represent valid child indexes in a KeyPath,
//! and the last four bits are discarded.
//!
//! Root PageId is at layer zero of the page tree,
//! the second layer has index one and so on. There are 43 layers.
//!
//! For each layer between 1 and 42 the lowest id for layer I is:
//!
//! L(I) = sum_i(2^(6 * i)) where i goes from 0 to I - 1
//!
//! while the highest id for each layer is:
//!
//! H(I) = L(I) + 2^(6 * I) - 1 = sum_i(2^(6 * i)) where i goes from 1 to I
//!
//! So for each layer bigger than 0 the range of PageIds are:
//!
//! [L(I)..=H(I)]
//!
//! And L(I) = H(I - 1) + 1

use crate::{page::DEPTH, trie::KeyPath};
use core::ops::{AddAssign, ShlAssign, ShrAssign, SubAssign};
use ruint::Uint;

/// A unique ID for a page.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct PageId(Uint<256, 4>);

/// The root page is the one containing the sub-trie directly descending from the root node.
///
/// It has an ID consisting of all zeros.
pub const ROOT_PAGE_ID: PageId = PageId(Uint::from_be_bytes([0; 32]));

/// The PageId of the leftmost page of the last layer of the page tree.
/// It is the lowest PageId beyond which all pages with a PageId equal to
/// or higher overflow if a child is attempted to be accessed
pub const LOWEST_42: PageId = PageId(Uint::from_be_bytes([
    0, 65, 4, 16, 65, 4, 16, 65, 4, 16, 65, 4, 16, 65, 4, 16, 65, 4, 16, 65, 4, 16, 65, 4, 16, 65,
    4, 16, 65, 4, 16, 65,
]));

/// The PageId of the rightmost page of the last layer of the page tree.
/// It is the highest possible PageId present in the page tree.
pub const HIGHEST_42: PageId = PageId(Uint::from_be_bytes([
    16, 65, 4, 16, 65, 4, 16, 65, 4, 16, 65, 4, 16, 65, 4, 16, 65, 4, 16, 65, 4, 16, 65, 4, 16, 65,
    4, 16, 65, 4, 16, 64,
]));

pub const MAX_CHILD_INDEX: u8 = (1 << DEPTH) - 1;

/// The index of a children of a page.
///
/// Each page can be thought of a root-less binary tree. The leaves of that tree are roots of
/// subtrees stored in subsequent pages. There are 64 (2^[`DEPTH`]) children in each page.
#[derive(Debug, PartialEq, Eq)]
pub struct ChildPageIndex(u8);

impl ChildPageIndex {
    pub fn new(index: u8) -> Option<Self> {
        if index > MAX_CHILD_INDEX {
            return None;
        }
        Some(Self(index))
    }

    pub fn to_u8(self) -> u8 {
        self.0
    }
}

impl PageId {
    /// Construct a PageId performing a bound check.
    /// The given bytes cannot represent a PageId bigger than the biggest valid one.
    pub fn from_bytes(bytes: [u8; 32]) -> Result<Self, InvalidPageIdBytes> {
        let page_id = Uint::from_be_bytes(bytes);
        if page_id > HIGHEST_42.0 {
            return Err(InvalidPageIdBytes);
        }
        Ok(Self(page_id))
    }

    /// Get raw bytes of the PageId.
    pub fn to_bytes(self) -> [u8; 32] {
        self.0.to_be_bytes()
    }

    /// Construct the Child PageId given the previous PageId and the child index.
    ///
    /// Child index must be a 6 bit integer, two most significant bits must be zero.
    /// Passed PageId must be a valid PageId and be located in a layer below 42 otherwise
    /// `PageIdOverflow` will be returned.
    pub fn child_page_id(&self, child_index: ChildPageIndex) -> Result<Self, ChildPageIdError> {
        // Any PageId larger than or equal to LOWEST_42 will overflow.
        if *self >= LOWEST_42 {
            return Err(ChildPageIdError::PageIdOverflow);
        }

        let mut new_page_id = self.clone();
        // next_page_id = (prev_page_id << d) + (child_index + 1)
        // there is no need to do any check on the following operations
        // because the previous checks make them impossible to overflow
        new_page_id.0.shl_assign(DEPTH);
        new_page_id
            .0
            .add_assign(Uint::<256, 4>::from(child_index.0 + 1));

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

        // prev_page_id = (next_page_id - 1) >> d
        new_page_id.0.sub_assign(Uint::<256, 4>::from(1));
        new_page_id.0.shr_assign(DEPTH);

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

        // unwrap: `new` can't return an error because the key_path is shifted.
        let child_index = ChildPageIndex::new(self.key_path.byte(31) >> 2).unwrap();
        self.key_path <<= 6;
        self.page_id = prev.child_page_id(child_index).ok();
        Some(prev)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn child_page_id(page_id: PageId, child_index: u8) -> Result<PageId, ChildPageIdError> {
        page_id.child_page_id(ChildPageIndex::new(child_index).unwrap())
    }

    #[test]
    fn test_child_and_parent_page_id() {
        let page_id_0 = ROOT_PAGE_ID;

        let mut page_id_1 = [0u8; 32]; // child index 6
        page_id_1[31] = 0b00000111;
        let page_id_1 = PageId::from_bytes(page_id_1).unwrap();

        assert_eq!(Ok(page_id_1.clone()), child_page_id(page_id_0, 6));
        assert_eq!(page_id_0, page_id_1.parent_page_id());

        let mut page_id_2 = [0u8; 32]; // child index 4
        page_id_2[31] = 0b11000101;
        page_id_2[30] = 0b00000001;
        let page_id_2 = PageId::from_bytes(page_id_2).unwrap();

        assert_eq!(Ok(page_id_2.clone()), child_page_id(page_id_1, 4));
        assert_eq!(page_id_1, page_id_2.parent_page_id());

        let mut page_id_3 = [0u8; 32]; // child index 63
        page_id_3[31] = 0b10000000;
        page_id_3[30] = 0b01110001;
        let page_id_3 = PageId::from_bytes(page_id_3).unwrap();

        assert_eq!(
            Ok(page_id_3.clone()),
            child_page_id(page_id_2, MAX_CHILD_INDEX),
        );
        assert_eq!(page_id_2, page_id_3.parent_page_id());
    }

    #[test]
    fn test_page_ids_iterator() {
        // key_path = 0b000001|000010|0...
        let mut key_path = [0u8; 32];
        key_path[0] = 0b00000100;
        key_path[1] = 0b00100000;

        let page_id_0 = ROOT_PAGE_ID;
        let mut page_id_1 = [0u8; 32];
        page_id_1[31] = 0b00000010; // 0b000001 + 1
        let page_id_1 = PageId::from_bytes(page_id_1).unwrap();
        let mut page_id_2 = [0u8; 32];
        page_id_2[31] = 0b10000011; // (0b000001 + 1 << 6) + 0b000010 + 1
        let page_id_2 = PageId::from_bytes(page_id_2).unwrap();

        let mut page_ids = PageIdsIterator::new(key_path);
        assert_eq!(page_ids.next(), Some(page_id_0));
        assert_eq!(page_ids.next(), Some(page_id_1));
        assert_eq!(page_ids.next(), Some(page_id_2));

        // key_path = 0b000010|111111|0...
        let mut key_path = [0u8; 32];
        key_path[0] = 0b00001011;
        key_path[1] = 0b11110000;

        let page_id_0 = ROOT_PAGE_ID;
        let mut page_id_1 = [0u8; 32];
        page_id_1[31] = 0b00000011; // 0b000010 + 1
        let page_id_1 = PageId::from_bytes(page_id_1).unwrap();
        let mut page_id_2 = [0u8; 32];
        page_id_2[31] = 0b0000000;
        page_id_2[30] = 0b0000001; // (0b00000011 << 6) + 0b111111 + 1 = (0b00000011 + 1) << 6
        let page_id_2 = PageId::from_bytes(page_id_2).unwrap();

        let mut page_ids = PageIdsIterator::new(key_path);
        assert_eq!(page_ids.next(), Some(page_id_0));
        assert_eq!(page_ids.next(), Some(page_id_1));
        assert_eq!(page_ids.next(), Some(page_id_2));
    }

    #[test]
    fn invalid_child_index() {
        assert_eq!(None, ChildPageIndex::new(0b01010000));
        assert_eq!(None, ChildPageIndex::new(0b10000100));
        assert_eq!(None, ChildPageIndex::new(0b11000101));
    }

    #[test]
    fn test_invalid_page_id() {
        // position 255
        let mut page_id = [0u8; 32];
        page_id[0] = 128;
        assert_eq!(Err(InvalidPageIdBytes), PageId::from_bytes(page_id));

        // position 252
        let mut page_id = [0u8; 32];
        page_id[0] = 128;
        assert_eq!(Err(InvalidPageIdBytes), PageId::from_bytes(page_id));
    }

    #[test]
    fn test_page_id_overflow() {
        let first_page_last_layer = PageIdsIterator::new([0u8; 32]).last().unwrap();
        let last_page_last_layer = PageIdsIterator::new([255; 32]).last().unwrap();
        assert_eq!(
            Err(ChildPageIdError::PageIdOverflow),
            child_page_id(first_page_last_layer, 0),
        );
        assert_eq!(
            Err(ChildPageIdError::PageIdOverflow),
            child_page_id(last_page_last_layer, 0),
        );

        // position 255
        let mut page_id = [0u8; 32];
        page_id[0] = 128;
        let page_id = PageId(Uint::<256, 4>::from_be_bytes(page_id));
        assert_eq!(
            Err(ChildPageIdError::PageIdOverflow),
            child_page_id(page_id, 0),
        );

        // any PageId bigger than LOWEST_42 must overflow
        let mut page_id = LOWEST_42.to_bytes();
        page_id[31] = 255;
        let page_id = PageId::from_bytes(page_id).unwrap();
        assert_eq!(
            Err(ChildPageIdError::PageIdOverflow),
            child_page_id(page_id, 0),
        );

        // position 245
        let mut page_id = [0u8; 32];
        page_id[1] = 32;
        let page_id = PageId::from_bytes(page_id).unwrap();
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
