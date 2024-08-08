// Here is the layout of a leaf node:
//
// ```rust,ignore
// n: u16
// cell_pointers: [(key ++ offset); n]
// padding: [u8] // empty space between cell_pointers and cells
// cells: [[u8]; n]
// ```
//
// | n | [(key ++ offset); n] | ----  | [[u8]; n] |
//
// Where key is an [u8; 32], and offset is the byte offset in the node
// to the beginning of the value.
//
// Cell pointers are saved in order of the key, and consequently, so are the cells.
// The length of the value is determined by the difference between the start offsets
// of this value and the next.
//
// Cells are left-aligned and thus the last value is always attached to the end.
//
// The offset of the first cell also serves to detect potential overlap
// between the growth of cell_pointers and cells.
//
// Overflow pages: TODO

use std::ops::Range;

use crate::{
    beatree::Key,
    store::{Page, PAGE_SIZE},
};

pub const LEAF_NODE_BODY_SIZE: usize = PAGE_SIZE - 2;

pub struct LeafNode {
    pub inner: Box<Page>,
}

#[derive(PartialEq, Debug)]
pub enum LeafInsertResult {
    Ok,
    NoSpaceLeft,
}

impl LeafNode {
    pub fn zeroed() -> Self {
        LeafNode {
            inner: Box::new(Page::zeroed()),
        }
    }

    pub fn n(&self) -> usize {
        u16::from_le_bytes(self.inner[0..2].try_into().unwrap()) as usize
    }

    pub fn set_n(&mut self, n: u16) {
        self.inner[0..2].copy_from_slice(&n.to_le_bytes());
    }

    pub fn key(&self, i: usize) -> Key {
        let mut key = [0u8; 32];
        key.copy_from_slice(&self.cell_pointers()[i][..32]);
        key
    }

    pub fn value(&self, i: usize) -> &[u8] {
        let range = self.value_range(self.cell_pointers(), i);
        &self.inner[range]
    }

    pub fn get(&self, key: &Key) -> Option<&[u8]> {
        let cell_pointers = self.cell_pointers();

        search(cell_pointers, key).ok()
            .map(|index| &self.inner[self.value_range(cell_pointers, index)])
    }

    // returns the range at which the value of a cell is stored
    fn value_range(&self, cell_pointers: &[[u8; 34]], index: usize) -> Range<usize> {
        let start = cell_offset(cell_pointers, index);
        let end = if index == cell_pointers.len() - 1 {
            PAGE_SIZE
        } else {
            cell_offset(cell_pointers, index + 1)
        };

        start..end
    }

    fn cell_pointers(&self) -> &[[u8; 34]] {
        unsafe {
            std::slice::from_raw_parts(self.inner[2..36].as_ptr() as *const [u8; 34], self.n())
        }
    }

    fn cell_pointers_mut(&mut self) -> &mut [[u8; 34]] {
        unsafe {
            std::slice::from_raw_parts_mut(self.inner[2..36].as_mut_ptr() as *mut [u8; 34], self.n())
        }
    }
}

pub struct LeafBuilder {
    leaf: LeafNode,
    index: usize,
    total_value_size: usize,
}

impl LeafBuilder {
    pub fn new(n: usize) -> Self {
        let mut leaf = LeafNode { inner: Box::new(Page::zeroed()) };
        leaf.set_n(n as u16);
        LeafBuilder {
            leaf,
            index: 0,
            total_value_size: 0,
        }
    } 

    pub fn push(&mut self, key: Key, value: &[u8]) {
        assert!(self.index < self.leaf.n());

        let offset = (self.leaf.n() as usize) * 34 + self.total_value_size + 2;
        let mut cell_pointer = self.leaf.cell_pointers_mut()[self.index];

        encode_cell(&mut cell_pointer[..], key, offset);
        self.leaf.inner[offset..][..value.len()].copy_from_slice(value);

        self.index += 1;
        self.total_value_size += value.len();
    }

    pub fn finish(self) -> LeafNode {
        self.leaf
    }
}

pub fn body_size(n: usize, value_size_sum: usize) -> usize {
    n * 34 + value_size_sum
}

fn cell_offset(cell_pointers: &[[u8; 34]], index: usize) -> usize {
    let mut buf = [0; 2];
    buf.copy_from_slice(&cell_pointers[index][32..34]);
    u16::from_le_bytes(buf) as usize
}

// panics if offset is bigger then u16 or `cell` length is less than 34.
fn encode_cell(cell: &mut [u8], key: [u8; 32], offset: usize) {
    cell[0..32].copy_from_slice(&key);
    cell[32..34].copy_from_slice(&(u16::try_from(offset).unwrap()).to_le_bytes());
}

// look for key in the node. the return value has the same semantics as std binary_search*.
fn search(cell_pointers: &[[u8; 34]], key: &Key) -> Result<usize, usize> {
    cell_pointers.binary_search_by(|cell| cell[0..32].cmp(key))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_to_empty_leaf() {
        let key = [1; 32];
        let val = vec![5; 10];
        let leaf = LeafNode::new(key, val.clone());
        assert_eq!(leaf.get(&[0; 32]), None);
        assert_eq!(leaf.get(&key), Some(&val[..]));
    }

    #[test]
    fn add_to_leaf() {
        let key = [0; 32];
        let val = vec![5; 10];
        let mut leaf = LeafNode::new(key, val.clone());

        let ks = [3, 1, 2].map(|i| [i; 32]);

        for k in ks.iter() {
            leaf.insert(k.clone(), k[0..11].to_vec());
        }

        for k in ks.iter() {
            assert_eq!(leaf.get(&k), Some(&k[0..11]));
        }
    }

    #[test]
    fn update_to_bigger_leaf_value() {
        let key = [0; 32];
        let val = vec![5; 10];
        let mut leaf = LeafNode::new(key, val.clone());

        let ks = [3, 1, 2].map(|i| [i; 32]);

        for k in ks.iter() {
            leaf.insert(k.clone(), k[0..11].to_vec());
        }

        // update to bigger value
        let key = [2; 32];
        leaf.insert(key.clone(), key[0..20].to_vec());
        for k in ks.iter() {
            if *k != key {
                assert_eq!(leaf.get(&k), Some(&k[0..11]));
            } else {
                assert_eq!(leaf.get(&key), Some(&key[0..20]));
            }
        }
    }

    #[test]
    fn update_to_smaller_leaf_value() {
        let key = [0; 32];
        let val = vec![5; 10];
        let mut leaf = LeafNode::new(key, val.clone());

        let ks = [3, 1, 2].map(|i| [i; 32]);

        for k in ks.iter() {
            leaf.insert(k.clone(), k[0..11].to_vec());
        }

        // update to smaller value
        let key = [1; 32];
        leaf.insert(key.clone(), key[0..3].to_vec());
        for k in ks.iter() {
            if *k != key {
                assert_eq!(leaf.get(&k), Some(&k[0..11]));
            } else {
                assert_eq!(leaf.get(&key), Some(&key[0..3]));
            }
        }
    }

    #[test]
    fn remove_leaf_value() {
        let key = [0; 32];
        let val = vec![5; 10];
        let mut leaf = LeafNode::new(key, val.clone());

        let ks = [3, 1, 2].map(|i| [i; 32]);

        for k in ks.iter() {
            leaf.insert(k.clone(), k[0..11].to_vec());
        }

        // update to bigger value
        let key = [2; 32];
        leaf.remove(key.clone());
        for k in ks.iter() {
            if *k != key {
                assert_eq!(leaf.get(&k), Some(&k[0..11]));
            } else {
                assert_eq!(leaf.get(&key), None);
            }
        }
    }

    #[test]
    fn insert_overflow_leaf() {
        let key = [0; 32];
        let val = vec![0; 1024];
        let mut leaf = LeafNode::new(key, val.clone());

        assert_eq!(LeafInsertResult::Ok, leaf.insert([1; 32], vec![1; 1024]));
        assert_eq!(LeafInsertResult::Ok, leaf.insert([2; 32], vec![2; 1024]));

        assert_eq!(
            LeafInsertResult::NoSpaceLeft,
            leaf.insert([3; 32], vec![3; 1024])
        );
    }
}
