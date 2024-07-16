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

pub struct LeafNode {
    pub inner: Box<Page>,
}

#[derive(PartialEq, Debug)]
pub enum LeafInsertResult {
    Ok,
    NoSpaceLeft,
}

impl LeafNode {
    pub fn new(key: Key, value: Vec<u8>) -> Self {
        let mut leaf = Self {
            inner: Box::new(Page::zeroed()),
        };

        assert_eq!(leaf.insert(key, value), LeafInsertResult::Ok);
        leaf
    }

    pub fn n(&self) -> usize {
        u16::from_le_bytes(self.inner[0..2].try_into().unwrap()) as usize
    }

    pub fn set_n(&mut self, n: u16) {
        self.inner[0..2].copy_from_slice(&n.to_le_bytes());
    }

    pub fn insert(&mut self, key: Key, value: Vec<u8>) -> LeafInsertResult {
        assert!(!value.is_empty());

        let n_items = self.n();
        let cell_pointers = self.cell_pointers();
        let index = search(cell_pointers, &key);

        // if the key is the same then that's an update
        if index < n_items && cell_key(cell_pointers, index) == &key {
            self.update(index, key, value);
            return LeafInsertResult::Ok;
        }

        // make sure the new byte fits
        if self.space_left(cell_pointers) < 34 + value.len() {
            return LeafInsertResult::NoSpaceLeft;
        }

        // When inserting a new element, the cell_pointers starting from the index must be shifted
        // to the right by 34 bytes. The values prior to the one being inserted must be shifted to the left
        // by the size of the inserted value.
        //
        // This adjustment also applies to all cell_pointers preceding the one related to the newly inserted value,
        // where their offsets should be reduced by the length of the inserted value.

        // create space
        let new_offset = self.shift_left_cells(index, value.len());
        self.shift_right_cell_pointers(index);

        // update n + 1 items
        self.set_n(n_items as u16 + 1);

        // update cell_pointers offset
        let cell_pointers = self.cell_pointers_mut();
        for i in 0..index {
            decrease_cell_offset(cell_pointers, i, value.len());
        }

        // store new cell_pointer offset
        cell_pointers[index] = encode_cell(key, new_offset);

        // store value
        self.inner[new_offset..new_offset + value.len()].copy_from_slice(&value);

        LeafInsertResult::Ok
    }

    pub fn remove(&mut self, key: Key) {
        let cell_pointers = self.cell_pointers();
        let index = search(cell_pointers, &key);

        if cell_key(cell_pointers, index) != &key {
            return;
        }

        let prev_value_len = self.value_range(self.cell_pointers(), index).len();

        // shift cell and cells_pointers
        self.shift_right_cells(index, prev_value_len);
        self.shift_left_cell_pointers(index);

        // update n - 1 items
        self.set_n(self.n() as u16 - 1);

        // update cell_pointers offset
        let cell_pointers = self.cell_pointers_mut();
        for i in 0..index {
            increase_cell_offset(cell_pointers, i, prev_value_len);
        }
    }

    pub fn get(&self, key: &Key) -> Option<&[u8]> {
        let cell_pointers = self.cell_pointers();
        let index = search(cell_pointers, key);

        if cell_key(cell_pointers, index) == key {
            Some(&self.inner[self.value_range(cell_pointers, index)])
        } else {
            None
        }
    }

    // Updates an existing value
    //
    // panics if the value is empty
    fn update(&mut self, index: usize, key: Key, new_value: Vec<u8>) {
        // create space
        let prev_value_len = self.value_range(self.cell_pointers(), index).len();

        let delta = new_value.len() as i64 - prev_value_len as i64;

        let new_offset = match delta {
            // space value should decrease and the cell offset should increase
            d if d < 0 => Some(self.shift_right_cells(index, delta.abs() as usize)),
            // space value should increase and the cell offset should decrease
            d if d > 0 => Some(self.shift_left_cells(index, delta as usize)),
            // do nothing if the size is not changes
            _ => None,
        };

        if let Some(new_offset) = new_offset {
            // update cell_pointers offset
            let cell_pointers = self.cell_pointers_mut();
            for i in 0..index {
                if delta < 0 {
                    increase_cell_offset(cell_pointers, i, delta.abs() as usize);
                } else if delta > 0 {
                    decrease_cell_offset(cell_pointers, i, delta as usize);
                }
            }
            // store new cell_pointer offset
            let cell_pointers = self.cell_pointers_mut();
            cell_pointers[index] = encode_cell(key, new_offset);
        }

        let value_range = self.value_range(self.cell_pointers(), index);
        // store value
        self.inner[value_range].copy_from_slice(&new_value);
    }

    fn space_left(&self, cell_pointers: &[[u8; 34]]) -> usize {
        if cell_pointers.is_empty() {
            return PAGE_SIZE - 2;
        }

        let end_cell_pointers = 2 + cell_pointers.len() * 34;
        let start_cells = self.value_range(cell_pointers, 0).start;

        start_cells - end_cell_pointers
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
            std::slice::from_raw_parts_mut(self.inner[2..36].as_ptr() as *mut [u8; 34], self.n())
        }
    }

    fn shift_right_cell_pointers(&mut self, index: usize) {
        if index == self.n() {
            return;
        }

        let range = 2 + index * 34..2 + self.n() * 34;

        for byte in range.rev() {
            self.inner[byte + 34] = self.inner[byte];
        }
    }

    fn shift_left_cell_pointers(&mut self, index: usize) {
        let range = 2 + (index + 1) * 34..2 + self.n() * 34;

        for byte in range {
            self.inner[byte - 34] = self.inner[byte];
        }
    }

    // returns the offset at which the new free space is avaible
    fn shift_left_cells(&mut self, index: usize, size: usize) -> usize {
        let cell_pointers = self.cell_pointers();

        if cell_pointers.is_empty() {
            return PAGE_SIZE - size;
        }

        let start = self.value_range(cell_pointers, 0).start;
        let end = self.value_range(cell_pointers, index - 1).end;

        for byte in start..end {
            self.inner[byte - size] = self.inner[byte];
        }

        end - size
    }

    // returns the new start of the shrink space
    fn shift_right_cells(&mut self, index: usize, size: usize) -> usize {
        let cell_pointers = self.cell_pointers();
        let start = self.value_range(cell_pointers, 0).start;
        let end = self.value_range(cell_pointers, index - 1).end;

        for byte in (start..end).rev() {
            self.inner[byte + size] = self.inner[byte];
        }

        end + size
    }
}

fn cell_offset(cell_pointers: &[[u8; 34]], index: usize) -> usize {
    let mut buf = [0; 2];
    buf.copy_from_slice(&cell_pointers[index][32..34]);
    u16::from_le_bytes(buf) as usize
}

fn increase_cell_offset(cell_pointers: &mut [[u8; 34]], index: usize, amount: usize) {
    let mut buf = [0; 2];
    buf.copy_from_slice(&cell_pointers[index][32..34]);
    let new_offset = u16::from_le_bytes(buf) + amount as u16;
    cell_pointers[index][32..34].copy_from_slice(&new_offset.to_le_bytes());
}

fn decrease_cell_offset(cell_pointers: &mut [[u8; 34]], index: usize, amount: usize) {
    let mut buf = [0; 2];
    buf.copy_from_slice(&cell_pointers[index][32..34]);
    let new_offset = u16::from_le_bytes(buf) - u16::try_from(amount).unwrap();
    cell_pointers[index][32..34].copy_from_slice(&new_offset.to_le_bytes());
}

fn cell_key<'a>(cell_pointers: &'a [[u8; 34]], index: usize) -> &'a [u8] {
    &cell_pointers[index][0..32]
}

// panics if offset is bigger then u16
fn encode_cell(key: [u8; 32], offset: usize) -> [u8; 34] {
    let mut buf = [0; 34];
    buf[0..32].copy_from_slice(&key);
    buf[32..34].copy_from_slice(&(u16::try_from(offset).unwrap()).to_le_bytes());
    buf
}

// look for key in the node, returns the index of
// the cell_pointer which contains the key or the index
// of the cell_pointer just after where the key should be
fn search(cell_pointers: &[[u8; 34]], key: &Key) -> usize {
    cell_pointers
        .binary_search_by(|cell| {
            if &cell[0..32] < key {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            }
        })
        .unwrap_err()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_to_emtpy_leaf() {
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
