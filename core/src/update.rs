//! Trie update logic.
//!
//! The core of this module is the Multi-Update Algorithm, by which we can cause the same effect
//! to a trie as a series of standalone put and delete operations, but with a minimal amount of
//! hashing and revisiting of nodes.
//!
//! ## Multi-Update Algorithm
//!
//! The algorithm is based off of the observation that a set of put and delete operations can be
//! partitioned into groups based on which terminal node their keys currently look up to.
//! Each terminal node is then replaced with the sub-trie resulting from the set of given updates,
//! and the trie is compacted into its smallest possible form, and hashed.
//!
//! For example,
//!   - Replacing a single leaf node with another leaf node in the case of the previous leaf
//!     being deleted and a new one with the same key or at least key prefix being put.
//!   - Replacing a single leaf node with a terminator, in the case of deleting the leaf which was
//!     there prior.
//!   - Replacing a terminator with a leaf, in the case of a single put operation with that prefix
//!   - Replacing a leaf node or terminator with a larger sub-trie in the case of multiple puts for
//!     keys beginning with that prefix, possibly preserving the initial leaf.
//!
//! We refer to this as sub-trie replacement.
//!
//! Any newly created terminator nodes must be "compacted" upwards as long as their sibling is a
//! terminator or a leaf node to create the most tractable representation. We combine this operation
//! with hashing up towards the root, which will be described in a moment.
//!
//! Any changes in the trie must be reflected in the hashes of the nodes above them, all the way
//! up to the root. When we replace a terminal node with a new sub-trie, we apply the compaction
//! and hashing operations up to the point where no subsequently altered terminal will affect its
//! result. The last terminal finishes hashing to the root. We refer to this as partial compaction.

use crate::cursor::Cursor;
use crate::trie::{self, KeyPath, LeafData, Node, NodeHasher, NodeHasherExt, ValueHash};

use bitvec::prelude::*;

/// A visited terminal node.
#[derive(Clone)]
pub struct VisitedTerminal {
    /// the path to the terminal node. only the first `depth` bits matter.
    pub path: KeyPath,
    /// the depth of the terminal node.
    pub depth: u8,
    /// the leaf at the terminal. `None` if it's a terminator.
    pub leaf: Option<LeafData>,
}

/// Apply an update, represented as a sequence of operations, to the trie
/// navigated by the given cursor.
///
/// The `ops` provided should be sorted ascending, lexicographically by key. This bails early when
/// encountering unsorted keys. The `visited_leaves` consists of all of the preimages to leaf nodes
/// encountered when looking up any of the keys in `ops` and should also be sorted
/// lexicographically ascending by key.
///
/// In short, the API accepts a sorted set of operations, which may be partitioned into contiguous
/// slices by the terminal nodes that they look up to in the trie.
pub fn update<H: NodeHasher>(
    cursor: &mut impl Cursor,
    ops: &[(KeyPath, Option<ValueHash>)],
    visited_terminals: &[VisitedTerminal],
) {
    if ops.is_empty() {
        return;
    }

    cursor.rewind();

    let mut batch_start = 0;
    let mut visited_terminals = visited_terminals.iter().cloned();
    while batch_start < ops.len() {
        let terminal = match visited_terminals.next() {
            Some(t) => t,
            None => {
                // sanity: should never occur in correct API usage.
                return;
            }
        };
        let leaf_data = terminal.leaf.clone();

        cursor.jump(terminal.path, terminal.depth);

        // note: cursor position does not move during these
        // comparisons.
        let batch_len = ops[batch_start..]
            .iter()
            .skip(1)
            .take_while(|op| shares_prefix(&*cursor, op.0))
            .count();

        let next_batch_start = batch_start + 1 + batch_len;
        let compact_layers = if next_batch_start < ops.len() {
            let current_depth = cursor.position().depth() as usize;
            let shared_depth = shared_with_cursor(&*cursor, ops[next_batch_start].0);

            // shared_depth is guaranteed less than current_depth because the full prefix isn't
            // shared.
            // we want to compact up (inclusive) to the depth `shared_depth + 1`
            current_depth - (shared_depth + 1)
        } else {
            // last batch: all the way to root
            cursor.position().depth() as usize
        };

        let batch = &ops[batch_start..next_batch_start];

        replace_subtrie::<H>(cursor, leaf_data, batch);
        for _ in 0..compact_layers {
            if let Some(internal_data) = cursor.compact_up() {
                cursor.place_non_leaf(H::hash_internal(&internal_data));
            }
        }

        batch_start = next_batch_start;
    }
}

fn shares_prefix(cursor: &impl Cursor, key: KeyPath) -> bool {
    let pos = cursor.position();
    let bits = pos.depth() as usize;
    &pos.path() == &key.view_bits::<Msb0>()[..bits as usize]
}

fn shared_with_cursor(cursor: &impl Cursor, key: KeyPath) -> usize {
    // TODO: check if this is correct
    let pos = cursor.position();
    shared_bits(&pos.path(), key.view_bits::<Msb0>())
}

fn shared_bits(a: &BitSlice<u8, Msb0>, b: &BitSlice<u8, Msb0>) -> usize {
    a.iter().zip(b.iter()).take_while(|(a, b)| a == b).count()
}

/// Creates an iterator of all provided operations, with the leaf value spliced in if its key
/// does not appear in the original ops list. Then filters out all `None`s.
pub fn leaf_ops_spliced(
    leaf: Option<LeafData>,
    ops: &[(KeyPath, Option<ValueHash>)],
) -> impl Iterator<Item = (KeyPath, ValueHash)> + '_ {
    let splice_index = leaf
        .as_ref()
        .and_then(|leaf| ops.binary_search_by_key(&leaf.key_path, |x| x.0).err());
    let preserve_value = splice_index
        .zip(leaf)
        .map(|(_, leaf)| (leaf.key_path, Some(leaf.value_hash)));
    let splice_index = splice_index.unwrap_or(0);

    // splice: before / item / after
    // skip deleted.
    ops[..splice_index]
        .into_iter()
        .cloned()
        .chain(preserve_value)
        .chain(ops[splice_index..].into_iter().cloned())
        .filter_map(|(k, o)| o.map(move |value| (k, value)))
}

// replaces a terminal node with a new sub-trie in place. `None` keys are ignored.
// cursor is returned at the same point it started at.
fn replace_subtrie<H: NodeHasher>(
    cursor: &mut impl Cursor,
    leaf_data: Option<LeafData>,
    ops: &[(KeyPath, Option<ValueHash>)],
) {
    let start_pos = cursor.position();
    let skip = start_pos.depth() as usize;

    let ops = leaf_ops_spliced(leaf_data, ops);

    build_trie::<H>(skip, ops, |visit_control, node, leaf_data| {
        cursor.up(visit_control.up);
        for bit in visit_control.down.iter().by_vals() {
            cursor.down(bit, true);
        }

        match leaf_data {
            None => cursor.place_non_leaf(node),
            Some(leaf_data) => cursor.place_leaf(node, leaf_data),
        }
    });
}

/// Indicates the relative position of the next node.
pub struct VisitControl<'a> {
    /// How many levels up.
    pub up: u8,
    /// Path to follow down.
    pub down: &'a BitSlice<u8, Msb0>,
}

// Build a trie out of the given prior terminal and operations. Operations should all start
// with the same prefix of len `skip` and be ordered lexicographically. The root node of the
// generated trie is the one residing at path `prefix[..skip]`. When skip=0, this is the actual
// root.
//
// Provide a visitor which will be called for each computed node of the trie.
//
// The visitor is assumed to have a default position at the root of the trie and from
// there will be controlled with `VisitControl` to indicate the position of the next node.
// The root is always visited at the end. If the written node is a leaf, the leaf-data preimage
// will be provided.
pub fn build_trie<H: NodeHasher>(
    skip: usize,
    ops: impl IntoIterator<Item = (KeyPath, ValueHash)>,
    mut visit: impl FnMut(VisitControl, Node, Option<LeafData>),
) -> Node {
    // we build a compact addressable sub-trie in-place based on the given set of ordered keys,
    // ignoring deletions as they are implicit in a fresh sub-trie.
    //
    // an algorithm for building the compact sub-trie follows:
    //
    // consider any three leaves, A, B, C in sorted order by key, with different keys.
    // A and B have some number of shared bits n1
    // B and C have some number of shared bits n2
    //
    // We can make an accurate statement about the position of B regardless of what other material
    // appears in the trie, as long as there is no A' s.t. A < A' < B and no C' s.t. B < C' < C.
    //
    // A is a leaf somewhere to the left of B, which is in turn somewhere to the left of C
    // A and B share an internal node at depth n1, while B and C share an internal node at depth n2.
    // n1 cannot equal n2, as there are only 2 keys with shared prefix n and a != b != c.
    // If n1 is less than n2, then B is a leaf at depth n2+1 along its path (always left)
    // If n2 is less than n1, then B is a leaf at depth n1+1 along its path (always right)
    // QED
    //
    // A similar process applies to the first leaf in the list: it is a leaf on the left of an
    // internal node at depth n, where n is the number of shared bits with the following key.
    //
    // Same for the last leaf in the list: it is on the right of an internal node at depth n,
    // where n is the number of shared bits with the previous key.
    //
    // If the list has a single item, the sub-trie is a single leaf.
    // And if the list is empty, the sub-trie is a terminator.

    // A left-frontier: all modified nodes are to the left of
    // `b`, so this stores their layers.
    let mut pending_siblings: Vec<(Node, usize)> = Vec::new();

    let mut leaf_ops = ops.into_iter();

    let mut a = None;
    let mut b = leaf_ops.next();
    let mut c = leaf_ops.next();

    if b.is_none() {
        visit(
            VisitControl {
                up: 0,
                down: BitSlice::empty(),
            },
            trie::TERMINATOR,
            None,
        );
        return trie::TERMINATOR;
    }

    let common_after_prefix = |k1: &KeyPath, k2: &KeyPath| {
        let x = &k1.view_bits::<Msb0>()[skip..];
        let y = &k2.view_bits::<Msb0>()[skip..];
        shared_bits(x, y)
    };

    while let Some((this_key, this_val)) = b {
        let n1 = a.as_ref().map(|(k, _)| common_after_prefix(k, &this_key));
        let n2 = c.as_ref().map(|(k, _)| common_after_prefix(k, &this_key));

        let leaf_data = trie::LeafData {
            key_path: this_key,
            value_hash: this_val,
        };
        let leaf = H::hash_leaf(&leaf_data);
        let (leaf_depth, hash_up_layers) = match (n1, n2) {
            (None, None) => {
                // single value - no hashing required.
                (0, 0)
            }
            (None, Some(n2)) => {
                // first value, n2 ancestor will be affected by next.
                (n2 + 1, 0)
            }
            (Some(n1), None) => {
                // last value, hash up to sub-trie root.
                (n1 + 1, n1 + 1)
            }
            (Some(n1), Some(n2)) => {
                // middle value, hash up to incoming ancestor + 1.
                (core::cmp::max(n1, n2) + 1, n1.saturating_sub(n2))
            }
        };

        let mut layer = leaf_depth;
        let mut last_node = leaf;
        let down_start = skip + n1.unwrap_or(0);
        let leaf_end_bit = skip + leaf_depth;

        visit(
            VisitControl {
                up: n1.map_or(0, |_| 1), // previous iterations always get to current layer + 1
                down: &this_key.view_bits::<Msb0>()[down_start..leaf_end_bit],
            },
            leaf,
            Some(leaf_data),
        );

        for bit in this_key.view_bits::<Msb0>()[skip..leaf_end_bit]
            .iter()
            .by_vals()
            .rev()
            .take(hash_up_layers)
        {
            layer -= 1;
            let sibling = if pending_siblings.last().map_or(false, |l| l.1 == layer + 1) {
                // unwrap: just checked
                pending_siblings.pop().unwrap().0
            } else {
                trie::TERMINATOR
            };

            let node_data = if bit {
                trie::InternalData {
                    left: sibling,
                    right: last_node,
                }
            } else {
                trie::InternalData {
                    left: last_node,
                    right: sibling,
                }
            };

            last_node = H::hash_internal(&node_data);
            visit(
                VisitControl {
                    up: 1,
                    down: BitSlice::empty(),
                },
                last_node,
                None,
            );
        }
        pending_siblings.push((last_node, layer));

        a = Some((this_key, this_val));
        b = c;
        c = leaf_ops.next();
    }

    let new_root = pending_siblings
        .pop()
        .map(|n| n.0)
        .unwrap_or(trie::TERMINATOR);
    new_root
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyNodeHasher;

    impl NodeHasher for DummyNodeHasher {
        fn hash_node(data: &trie::NodePreimage) -> [u8; 32] {
            blake3::hash(data).into()
        }
    }

    fn leaf(key: u8) -> (LeafData, [u8; 32]) {
        let key = [key; 32];
        let leaf = trie::LeafData {
            key_path: key.clone(),
            value_hash: key.clone(),
        };

        let hash = DummyNodeHasher::hash_leaf(&leaf);
        (leaf, hash)
    }

    fn branch_hash(left: [u8; 32], right: [u8; 32]) -> [u8; 32] {
        let data = trie::InternalData { left, right };

        let hash = DummyNodeHasher::hash_internal(&data);
        hash
    }

    #[derive(Default)]
    struct Visited {
        key: BitVec<u8, Msb0>,
        visited: Vec<(BitVec<u8, Msb0>, Node)>,
    }

    impl Visited {
        fn at(key: BitVec<u8, Msb0>) -> Self {
            Visited {
                key,
                visited: Vec::new(),
            }
        }

        fn visit(&mut self, control: VisitControl, node: Node) {
            let n = self.key.len() - control.up as usize;
            self.key.truncate(n);
            self.key.extend_from_bitslice(&control.down);
            self.visited.push((self.key.clone(), node));
        }
    }

    #[test]
    fn build_empty_trie() {
        let mut visited = Visited::default();
        let root = build_trie::<DummyNodeHasher>(0, vec![], |control, node, _| {
            visited.visit(control, node)
        });

        let visited = visited.visited;

        assert_eq!(visited, vec![(bitvec![u8, Msb0;], [0u8; 32]),],);

        assert_eq!(root, [0u8; 32]);
    }

    #[test]
    fn build_single_value_trie() {
        let mut visited = Visited::default();

        let (leaf, leaf_hash) = leaf(0xff);
        let root = build_trie::<DummyNodeHasher>(
            0,
            vec![(leaf.key_path, leaf.value_hash)],
            |control, node, _| visited.visit(control, node),
        );

        let visited = visited.visited;

        assert_eq!(visited, vec![(bitvec![u8, Msb0;], leaf_hash),],);

        assert_eq!(root, leaf_hash);
    }

    #[test]
    fn sub_trie() {
        let (leaf_a, leaf_hash_a) = leaf(0b0001_0001);
        let (leaf_b, leaf_hash_b) = leaf(0b0001_0010);
        let (leaf_c, leaf_hash_c) = leaf(0b0001_0100);

        let mut visited = Visited::at(bitvec![u8, Msb0; 0, 0, 0, 1]);

        let ops = [leaf_a, leaf_b, leaf_c]
            .iter()
            .map(|l| (l.key_path, l.value_hash))
            .collect::<Vec<_>>();

        let root =
            build_trie::<DummyNodeHasher>(4, ops, |control, node, _| visited.visit(control, node));

        let visited = visited.visited;

        let branch_ab_hash = branch_hash(leaf_hash_a, leaf_hash_b);
        let branch_abc_hash = branch_hash(branch_ab_hash, leaf_hash_c);
        let root_branch_hash = branch_hash(branch_abc_hash, [0u8; 32]);

        assert_eq!(
            visited,
            vec![
                (bitvec![u8, Msb0; 0, 0, 0, 1, 0, 0, 0], leaf_hash_a),
                (bitvec![u8, Msb0; 0, 0, 0, 1, 0, 0, 1], leaf_hash_b),
                (bitvec![u8, Msb0; 0, 0, 0, 1, 0, 0], branch_ab_hash),
                (bitvec![u8, Msb0; 0, 0, 0, 1, 0, 1], leaf_hash_c),
                (bitvec![u8, Msb0; 0, 0, 0, 1, 0], branch_abc_hash),
                (bitvec![u8, Msb0; 0, 0, 0, 1], root_branch_hash),
            ],
        );

        assert_eq!(root, root_branch_hash);
    }

    #[test]
    fn multi_value() {
        let (leaf_a, leaf_hash_a) = leaf(0b0001_0000);
        let (leaf_b, leaf_hash_b) = leaf(0b0010_0000);
        let (leaf_c, leaf_hash_c) = leaf(0b0100_0000);
        let (leaf_d, leaf_hash_d) = leaf(0b1010_0000);
        let (leaf_e, leaf_hash_e) = leaf(0b1011_0000);

        let mut visited = Visited::default();

        let ops = [leaf_a, leaf_b, leaf_c, leaf_d, leaf_e]
            .iter()
            .map(|l| (l.key_path, l.value_hash))
            .collect::<Vec<_>>();

        let root =
            build_trie::<DummyNodeHasher>(0, ops, |control, node, _| visited.visit(control, node));

        let visited = visited.visited;

        let branch_ab_hash = branch_hash(leaf_hash_a, leaf_hash_b);
        let branch_abc_hash = branch_hash(branch_ab_hash, leaf_hash_c);

        let branch_de_hash_1 = branch_hash(leaf_hash_d, leaf_hash_e);
        let branch_de_hash_2 = branch_hash([0u8; 32], branch_de_hash_1);
        let branch_de_hash_3 = branch_hash(branch_de_hash_2, [0u8; 32]);

        let branch_abc_de_hash = branch_hash(branch_abc_hash, branch_de_hash_3);

        assert_eq!(
            visited,
            vec![
                (bitvec![u8, Msb0; 0, 0, 0], leaf_hash_a),
                (bitvec![u8, Msb0; 0, 0, 1], leaf_hash_b),
                (bitvec![u8, Msb0; 0, 0], branch_ab_hash),
                (bitvec![u8, Msb0; 0, 1], leaf_hash_c),
                (bitvec![u8, Msb0; 0], branch_abc_hash),
                (bitvec![u8, Msb0; 1, 0, 1, 0], leaf_hash_d),
                (bitvec![u8, Msb0; 1, 0, 1, 1], leaf_hash_e),
                (bitvec![u8, Msb0; 1, 0, 1], branch_de_hash_1),
                (bitvec![u8, Msb0; 1, 0], branch_de_hash_2),
                (bitvec![u8, Msb0; 1], branch_de_hash_3),
                (bitvec![u8, Msb0;], branch_abc_de_hash),
            ],
        );

        assert_eq!(root, branch_abc_de_hash);
    }
}
