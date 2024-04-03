//! Proving and verifying inclusion, non-inclusion, and updates to the trie.

use crate::cursor::Cursor;
use crate::trie::{InternalData, KeyPath, LeafData, Node, NodeHasher, NodeHasherExt, TERMINATOR};

use alloc::vec::Vec;
use bitvec::prelude::*;

/// A proof of some particular path through the trie.
#[derive(Debug, Clone)]
pub struct PathProof {
    /// The terminal node encountered when looking up a key. This is always either a terminator or
    /// leaf.
    pub terminal: Option<LeafData>,
    /// Sibling nodes encountered during lookup, in ascending order by depth.
    pub siblings: Vec<Node>,
}

impl PathProof {
    /// Verify this path proof.
    ///
    /// Provide the root node and a key path. The key path can be any key that results in the
    /// lookup of the terminal node and must be at least as long as the siblings vector.
    pub fn verify<H: NodeHasher>(
        &self,
        key_path: &BitSlice<u8, Msb0>,
        root: Node,
    ) -> Result<VerifiedPathProof, PathProofVerificationError> {
        if self.siblings.len() > 256 {
            return Err(PathProofVerificationError::TooManySiblings);
        }

        let mut cur_node = match &self.terminal {
            None => TERMINATOR,
            Some(leaf_data) => H::hash_leaf(&leaf_data),
        };

        let relevant_path = &key_path[..self.siblings.len()];
        for (bit, &sibling) in relevant_path.iter().by_vals().rev().zip(&self.siblings) {
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
                key_path: relevant_path.into(),
                terminal: self.terminal.clone(),
            })
        } else {
            Err(PathProofVerificationError::RootMismatch)
        }
    }
}

/// An error type indicating that a key is out of scope of a path proof.
#[derive(Debug, Clone, Copy)]
pub struct KeyOutOfScope;

/// Errors in path proof verification.
pub enum PathProofVerificationError {
    /// Amount of provided siblings is impossible for the expected trie depth.
    TooManySiblings,
    /// Root hash mismatched at the end of the verification.
    RootMismatch,
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
    key_path: BitVec<u8, Msb0>,
    terminal: Option<LeafData>,
}

impl VerifiedPathProof {
    /// Get the terminal node. `None` signifies that this path concludes with a [`TERMINATOR`].
    pub fn terminal(&self) -> Option<&LeafData> {
        self.terminal.as_ref()
    }

    /// Get the proven path.
    pub fn path(&self) -> &BitSlice<u8, Msb0> {
        &self.key_path[..]
    }

    /// Check whether this path resolves to the given leaf.
    ///
    /// A return value of `Ok(true)` confirms that the key indeed has this value in the trie.
    /// `Ok(false)` confirms that this key has a different value.
    ///
    /// Fails if the key is out of the scope of this path.
    pub fn confirm_value(&self, expected_leaf: &LeafData) -> Result<bool, KeyOutOfScope> {
        self.in_scope(&expected_leaf.key_path)
            .map(|_| self.terminal() == Some(expected_leaf))
    }

    /// Check whether this proves that a key has no value in the trie.
    ///
    /// A return value of `Ok(true)` confirms that the key indeed has no value in the trie.
    /// A return value of `Ok(false)` means that the key definitely exists within the trie.
    ///
    /// Fails if the key is out of the scope of this path.
    pub fn confirm_nonexistence(&self, key_path: &KeyPath) -> Result<bool, KeyOutOfScope> {
        self.in_scope(key_path).map(|_| {
            self.terminal()
                .as_ref()
                .map_or(true, |d| &d.key_path != key_path)
        })
    }

    fn in_scope(&self, key_path: &KeyPath) -> Result<(), KeyOutOfScope> {
        let this_path = self.path();
        let other_path = &key_path.view_bits::<Msb0>()[..self.key_path.len()];
        if this_path == other_path {
            Ok(())
        } else {
            Err(KeyOutOfScope)
        }
    }
}

/// Record a query path through the trie. This does no sanity-checking of the underlying
/// cursor's results, so a faulty cursor will lead to a faulty path.
///
/// This returns the sibling nodes and the terminal node encountered when looking up
/// a key.
pub fn record_path(cursor: &mut impl Cursor, key: &BitSlice<u8, Msb0>) -> (Node, Vec<Node>) {
    let mut siblings = Vec::new();
    let mut terminal = TERMINATOR;

    cursor.rewind();

    for bit in key.iter().by_vals() {
        terminal = cursor.node();

        if crate::trie::is_internal(&cursor.node()) {
            cursor.down(bit);
            siblings.push(cursor.peek_sibling());
        } else {
            break;
        }
    }

    let siblings: Vec<_> = siblings.into_iter().rev().collect();
    (terminal, siblings)
}
