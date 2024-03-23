//! Proving and verifying inclusion, non-inclusion, and updates to the trie.

use crate::trie::{InternalData, KeyPath, LeafData, Node, NodeHasher, NodeHasherExt, TERMINATOR};

use alloc::vec::Vec;
use bitvec::prelude::*;

/// Generic cursor over binary trie storage.
pub trait Cursor {
    /// The current position of the cursor, expressed as a bit-path and length. Bits after the
    /// length are irrelevant.
    fn position(&self) -> (KeyPath, u8);
    /// The current node.
    fn node(&self) -> Node;

    /// Peek at the children of this node, if it is an internal node.
    fn peek_children(&self) -> Option<(&Node, &Node)>;

    /// Traverse to the left child of this node, if it is an internal node. No-op otherwise.
    fn traverse_left_child(&mut self);
    /// Traverse to the right child of this node, if it is an internal node. No-op otherwise.
    fn traverse_right_child(&mut self);
    /// Traverse upwards by d bits. No-op if d is greater than the current position length.
    fn traverse_parents(&mut self, d: u8);
}

/// Sibling nodes recorded while looking up some path, in ascending order by depth.
#[derive(Debug, Clone)]
pub struct Siblings(pub Vec<Node>);

/// A proof of some particular path through the trie.
#[derive(Debug, Clone)]
pub struct PathProof {
    /// The terminal node encountered when looking up a key. This is always either a terminator or
    /// leaf.
    pub terminal: Option<LeafData>,
    /// Sibling nodes encountered during lookup, in ascending order by depth.
    pub siblings: Siblings,
}

impl PathProof {
    /// Verify this path proof.
    ///
    /// Provide the root node and a key path. The key path can be any key that results in the
    /// lookup of the terminal node.
    pub fn verify<H: NodeHasher>(
        &self,
        key_path: &KeyPath,
        root: Node,
    ) -> Result<VerifiedPathProof, ()> {
        // TODO, proper error
        let mut cur_node = match &self.terminal {
            None => TERMINATOR,
            Some(leaf_data) => H::hash_leaf(&leaf_data),
        };

        if self.siblings.0.len() > 256 {
            return Err(());
        }

        let relevant_path = &key_path.view_bits::<Msb0>()[..self.siblings.0.len()];
        for (bit, &sibling) in relevant_path.iter().by_vals().rev().zip(&self.siblings.0) {
            let (left, right) = if bit {
                (sibling, cur_node)
            } else {
                (cur_node, sibling)
            };

            let next = InternalData {
                left: left.clone(),
                right: right.clone(),
            };
            cur_node = H::hash_internal(&next);
        }

        if cur_node == root {
            Ok(VerifiedPathProof {
                key_path: key_path.clone(),
                depth: self.siblings.0.len() as _,
                terminal: self.terminal.clone(),
            })
        } else {
            Err(())
        }
    }
}

/// A verified path through the trie.
///
/// Each verified path can be used to check up to two kinds of statements:
///   1. That a single key has a specific value.
///   2. That a single or multiple keys do not have a value.
///
/// Statement (1) is true when the path leads to a leaf node and the leaf has the provided key and
/// value.
///
/// Statement (2) is true for any key which begins with the proven path, where the terminal node is
/// either not a leaf or contains a value for a different key.
#[derive(Clone)]
pub struct VerifiedPathProof {
    key_path: KeyPath,
    depth: u8,
    terminal: Option<LeafData>,
}

/// Record a query path through the trie. This does no sanity-checking of the underlying
/// cursor's results, so a faulty cursor will lead to a faulty path.
///
/// This returns the sibling nodes and the terminal node terminal node encountered when looking up
/// a key. This is always either a terminator or leaf.
pub fn record_path(cursor: &mut impl Cursor, key: &KeyPath) -> (Node, Siblings) {
    // reset to root
    let (_, d) = cursor.position();
    if d != 0 {
        cursor.traverse_parents(d);
    }

    let mut siblings = Vec::new();
    let mut terminal = TERMINATOR;

    for bit in key.view_bits::<Msb0>().iter().by_vals() {
        if let Some((left, right)) = cursor.peek_children() {
            if !bit {
                siblings.push(right.clone());
                cursor.traverse_left_child();
            } else {
                siblings.push(left.clone());
                cursor.traverse_right_child();
            }
        } else {
            terminal = cursor.node();
            break;
        }
    }

    let siblings: Vec<_> = siblings.into_iter().rev().collect();
    (terminal, Siblings(siblings))
}
