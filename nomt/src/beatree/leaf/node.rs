/// Here is the layout of a leaf node:
///
/// ```rust,ignore
/// n: u16
/// cell_pointers: [(key ++ offset); n]
/// padding: [u8] // empty space between cell_pointers and cells
/// cells: [Cell; n]
/// value cell: [u8]
/// overflow cell: (u64, [NodePointer]) | semantically, (value_size, [NodePointer]).
/// ```
///
/// | n | [(key ++ offset); n] | ----  | [[u8]; n] |
///
/// Where key is an [u8; 32], and offset is the byte offset in the node
/// to the beginning of the value.
///
/// Cell pointers are saved in order of the key, and consequently, so are the cells.
/// The length of the value is determined by the difference between the start offsets
/// of this value and the next.
///
/// When a cell is an overflow cell, the high bit in the offset is set to `1`. Only the low
/// 15 bits should count when considering the offset.
///
/// Cells are left-aligned and thus the last value is always attached to the end.
///
/// The offset of the first cell also serves to detect potential overlap
/// between the growth of cell_pointers and cells.
use std::ops::Range;

use crate::{
    beatree::Key,
    io::{Page, PAGE_SIZE},
};

/// The size of the leaf node body: everything excluding the mandatory header.
pub const LEAF_NODE_BODY_SIZE: usize = PAGE_SIZE - 2;

/// The maximum value size before overflow pages are used.
pub const MAX_LEAF_VALUE_SIZE: usize = (LEAF_NODE_BODY_SIZE / 3) - 32;

/// The maximum number of node pointers which may appear directly in an overflow cell.
///
/// Note that this gives an overflow value cell maximum size of 100 bytes.
pub const MAX_OVERFLOW_CELL_NODE_POINTERS: usize = 23;

/// We use the high bit to encode whether a cell is an overflow cell.
const OVERFLOW_BIT: u16 = 1 << 15;

pub struct LeafNode {
    pub inner: Box<Page>,
}

impl LeafNode {
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

    pub fn value(&self, i: usize) -> (&[u8], bool) {
        let (range, overflow) = self.value_range(self.cell_pointers(), i);
        (&self.inner[range], overflow)
    }

    pub fn get(&self, key: &Key) -> Option<(&[u8], bool)> {
        let cell_pointers = self.cell_pointers();

        search(cell_pointers, key)
            .ok()
            .map(|index| self.value_range(cell_pointers, index))
            .map(|(range, overflow)| (&self.inner[range], overflow))
    }

    // returns the range at which the value of a cell is stored
    fn value_range(&self, cell_pointers: &[[u8; 34]], index: usize) -> (Range<usize>, bool) {
        let (start, overflow) = cell_offset(cell_pointers, index);
        let end = if index == cell_pointers.len() - 1 {
            PAGE_SIZE
        } else {
            cell_offset(cell_pointers, index + 1).0
        };

        (start..end, overflow)
    }

    fn cell_pointers(&self) -> &[[u8; 34]] {
        unsafe {
            std::slice::from_raw_parts(self.inner[2..36].as_ptr() as *const [u8; 34], self.n())
        }
    }

    fn cell_pointers_mut(&mut self) -> &mut [[u8; 34]] {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.inner[2..36].as_mut_ptr() as *mut [u8; 34],
                self.n(),
            )
        }
    }
}

pub struct LeafBuilder {
    leaf: LeafNode,
    index: usize,
    remaining_value_size: usize,
}

impl LeafBuilder {
    pub fn new(n: usize, total_value_size: usize) -> Self {
        let mut leaf = LeafNode {
            inner: Box::new(Page::zeroed()),
        };
        leaf.set_n(n as u16);
        LeafBuilder {
            leaf,
            index: 0,
            remaining_value_size: total_value_size,
        }
    }

    pub fn push_cell(&mut self, key: Key, value: &[u8], overflow: bool) {
        assert!(self.index < self.leaf.n());

        let offset = PAGE_SIZE - self.remaining_value_size;
        let cell_pointer = &mut self.leaf.cell_pointers_mut()[self.index];

        encode_cell_pointer(&mut cell_pointer[..], key, offset, overflow);
        self.leaf.inner[offset..][..value.len()].copy_from_slice(value);

        self.index += 1;
        self.remaining_value_size -= value.len();
    }

    pub fn finish(self) -> LeafNode {
        assert!(self.remaining_value_size == 0);
        self.leaf
    }
}

pub fn body_size(n: usize, value_size_sum: usize) -> usize {
    n * 34 + value_size_sum
}

// get the cell offset and whether the cell is an overflow cell.
fn cell_offset(cell_pointers: &[[u8; 34]], index: usize) -> (usize, bool) {
    let mut buf = [0; 2];
    buf.copy_from_slice(&cell_pointers[index][32..34]);
    let val = u16::from_le_bytes(buf);
    (
        (val & !OVERFLOW_BIT) as usize,
        val & OVERFLOW_BIT == OVERFLOW_BIT,
    )
}

// panics if offset is bigger than 2^15 - 1.
fn encode_cell_pointer(cell: &mut [u8], key: [u8; 32], offset: usize, overflow: bool) {
    let mut val = u16::try_from(offset).unwrap();
    assert!(val < OVERFLOW_BIT);

    if overflow {
        val |= OVERFLOW_BIT;
    }

    cell[0..32].copy_from_slice(&key);
    cell[32..34].copy_from_slice(&val.to_le_bytes());
}

// look for key in the node. the return value has the same semantics as std binary_search*.
fn search(cell_pointers: &[[u8; 34]], key: &Key) -> Result<usize, usize> {
    cell_pointers.binary_search_by(|cell| cell[0..32].cmp(key))
}

#[cfg(feature = "benchmarks")]
pub mod benches {

    use crate::beatree::{
        benches::get_keys,
        leaf::node::{LeafBuilder, LEAF_NODE_BODY_SIZE},
    };
    use criterion::{BatchSize, BenchmarkId, Criterion};
    use rand::Rng;

    pub fn leaf_search_benchmark(c: &mut Criterion) {
        let mut group = c.benchmark_group("search_leaf");
        let mut rand = rand::thread_rng();

        // we fill the leaf with as much as possible 4B values
        // leaf_body_size = b = n * 34 + value_size_sum
        //                    = n * 34 + (n * 4)
        //                  n = b / 38

        let n = LEAF_NODE_BODY_SIZE / 38;
        let mut leaf_builder = LeafBuilder::new(n, n * 4);

        let mut keys = get_keys(0, n);
        keys.sort();
        for (index, k) in keys.iter().enumerate() {
            leaf_builder.push_cell(k.clone(), &(index as u32).to_le_bytes()[..], false);
        }
        let leaf = leaf_builder.finish();

        group.bench_function(BenchmarkId::new("full_leaf", format!("{}-keys", n)), |b| {
            b.iter_batched(
                || {
                    let index = rand.gen_range(0..keys.len());
                    keys[index].clone()
                },
                |key| leaf.get(&key),
                BatchSize::SmallInput,
            )
        });

        group.finish();
    }

    pub fn leaf_builder_benchmark(c: &mut Criterion) {
        let mut group = c.benchmark_group("leaf_builder");

        // benchmark the leaf builder creating an almost full leaf node
        // given different value sizes

        for value_size in [4, 8, 16, 32, 64, 128] {
            // leaf_body_size = b = n * 34 + value_size_sum
            //                  b = n * 34 + (n * value_size)
            //                  n = b / (34 + value_size)

            let n = (LEAF_NODE_BODY_SIZE as f64 / (34 + value_size) as f64).floor() as usize;
            let mut keys = get_keys(0, n);
            keys.sort();

            group.bench_function(BenchmarkId::new("value_len_bytes", value_size), |b| {
                b.iter_batched(
                    || {
                        (
                            keys.clone(),
                            std::iter::repeat(12).take(value_size).collect::<Vec<u8>>(),
                        )
                    },
                    |(keys, value)| {
                        let mut leaf_builder = LeafBuilder::new(n, n * value_size);
                        for k in keys.into_iter() {
                            leaf_builder.push_cell(k, &value[..], false);
                        }
                        leaf_builder.finish();
                    },
                    criterion::BatchSize::SmallInput,
                )
            });
        }

        group.finish();
    }
}
