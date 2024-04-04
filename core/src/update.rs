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
use crate::trie::{self, KeyPath, LeafData, Node, NodeHasher, NodeHasherExt, NodeKind, ValueHash};

use bitvec::prelude::*;

/// Apply an update, represented as a sequence of operations, to the trie
/// navigated by the given cursor.
///
/// The `ops` provided should be sorted ascending, lexicographically by key. This bails early when
/// encountering unsorted keys. The `visited_leaves` consists of all of the preimages to leaf nodes
/// encountered when looking up any of the keys in `ops` and should also be sorted
/// lexicographically ascending by key.
///
/// In short, the API accepts a sorted set of operations, which may be partitioned into contiguous
/// slices by the terminal nodes that they look up to in the trie. When that node is a leaf, it
/// should be present in the `visited_leaves` slice.
pub fn update<H: NodeHasher>(
    cursor: &mut impl Cursor,
    ops: &[(KeyPath, Option<ValueHash>)],
    visited_leaves: &[LeafData],
) {
    if ops.is_empty() {
        return;
    }

    cursor.rewind();

    let mut batch_start = 0;
    let mut terminal_leaves = visited_leaves.iter().cloned();
    while batch_start < ops.len() {
        cursor.seek(ops[batch_start].0);
        let leaf_data = if trie::is_leaf(&cursor.node()) {
            // note: this should always be `Some` in correct
            // API usage.
            terminal_leaves.next()
        } else {
            None
        };

        // note: cursor position does not move during these
        // comparisons.
        let batch_len = ops[batch_start..]
            .iter()
            .skip(1)
            .take_while(|op| shares_prefix(&*cursor, op.0))
            .count();

        let next_batch_start = batch_start + 1 + batch_len;
        let compact_layers = if next_batch_start < ops.len() {
            let current_depth = cursor.position().1 as usize;
            let shared_depth = shared_with_cursor(&*cursor, ops[next_batch_start].0) + 1;

            // shared_depth is guaranteed less than current_depth because the full prefix isn't
            // shared.
            // we don't want to compute the node at `shared_depth` before that terminal is replaced
            // therefore:
            current_depth - shared_depth - 1
        } else {
            // last batch: all the way to root
            cursor.position().1 as usize
        };

        let batch = &ops[batch_start..next_batch_start];

        replace_subtrie::<H>(cursor, leaf_data, batch);
        compact_and_hash::<H>(cursor, compact_layers);

        batch_start = next_batch_start;
    }
}

fn shares_prefix(cursor: &impl Cursor, key: KeyPath) -> bool {
    let (k, bits) = cursor.position();
    &k.view_bits::<Msb0>()[..bits as usize] == &key.view_bits::<Msb0>()[..bits as usize]
}

fn shared_with_cursor(cursor: &impl Cursor, key: KeyPath) -> usize {
    let (k, _) = cursor.position();
    shared_bits(&k.view_bits::<Msb0>(), key.view_bits::<Msb0>())
}

fn shared_bits(a: &BitSlice<u8, Msb0>, b: &BitSlice<u8, Msb0>) -> usize {
    a.iter().zip(b.iter()).take_while(|(a, b)| a == b).count()
}

/// Creates an iterator of all provided operations, with the leaf value spliced in if its key
/// does not appear in the original ops list. Then filters out all `None`s.
fn leaf_ops_spliced(
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
    let skip = start_pos.1 as usize;

    build_sub_trie::<H>(skip, leaf_data, ops, |key, depth, node| {
        cursor.jump(key, depth);
        cursor.modify(node);
    });
}

// Build a sub-trie out of the given prior terminal and operations. Operations should all start
// with the same prefix of len `skip` and be ordered lexicographically.
//
// Provide a visitor which will be called for each computed node of the sub-trie.
// The root is always visited at the end.
pub fn build_sub_trie<H: NodeHasher>(
    skip: usize,
    leaf: Option<LeafData>,
    ops: &[(KeyPath, Option<ValueHash>)],
    mut visit: impl FnMut(KeyPath, u8, Node),
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

    let mut leaf_ops = crate::update::leaf_ops_spliced(leaf.clone(), ops);

    let mut a = None;
    let mut b = leaf_ops.next();
    let mut c = leaf_ops.next();

    let make_leaf = |(k, value)| {
        H::hash_leaf(&trie::LeafData {
            key_path: k,
            value_hash: value,
        })
    };
    let common_after_prefix = |k1: &KeyPath, k2: &KeyPath| {
        let x = &k1.view_bits::<Msb0>()[skip..];
        let y = &k2.view_bits::<Msb0>()[skip..];
        shared_bits(x, y)
    };

    while let Some((this_key, this_val)) = b {
        let n1 = a.as_ref().map(|(k, _)| common_after_prefix(k, &this_key));
        let n2 = c.as_ref().map(|(k, _)| common_after_prefix(k, &this_key));

        let leaf = make_leaf((this_key, this_val));
        let (depth_after_skip, hash_up_layers) = match (n1, n2) {
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

        let mut layer = depth_after_skip;
        let mut last_node = leaf;
        for bit in this_key.view_bits::<Msb0>()[..depth_after_skip]
            .iter()
            .by_vals()
            .rev()
            .take(hash_up_layers)
        {
            layer -= 1;
            let sibling = if pending_siblings.last().map_or(false, |l| l.1 == layer) {
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
            visit(this_key, layer as u8, last_node);
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
    if let Some(leaf) = leaf {
        visit(leaf.key_path, skip as u8, new_root);
    }
    new_root
}

fn compact_and_hash<H: NodeHasher>(cursor: &mut impl Cursor, layers: usize) {
    if layers == 0 {
        return;
    }

    let (key, depth) = cursor.position();
    for bit in key.view_bits::<Msb0>()[..depth as usize]
        .iter()
        .by_vals()
        .rev()
        .take(layers)
    {
        let sibling = cursor.peek_sibling();
        let node = cursor.node();
        match (NodeKind::of(&node), NodeKind::of(&sibling)) {
            (NodeKind::Terminator, NodeKind::Terminator) => {
                // compact terminators.
                cursor.up(1);
                cursor.modify(trie::TERMINATOR);
            }
            (NodeKind::Leaf, NodeKind::Terminator) => {
                // compact: clear this node, move leaf up.
                cursor.modify(trie::TERMINATOR);
                cursor.up(1);
                cursor.modify(node);
            }
            (NodeKind::Terminator, NodeKind::Leaf) => {
                // compact: clear sibling node, move leaf up.
                cursor.sibling();
                cursor.modify(trie::TERMINATOR);
                cursor.up(1);
                cursor.modify(sibling);
            }
            _ => {
                // otherwise, internal
                let node_data = if bit {
                    trie::InternalData {
                        left: sibling,
                        right: node,
                    }
                } else {
                    trie::InternalData {
                        left: node,
                        right: sibling,
                    }
                };
                cursor.up(1);
                cursor.modify(H::hash_internal(&node_data));
            }
        }
    }
}
