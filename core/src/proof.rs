//! Proving and verifying inclusion, non-inclusion, and updates to the trie.

use crate::page::{MissingPage, PageSet, PageSetCursor};
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
    ) -> Result<VerifiedPathProof, PathProofVerificationError> {
        if self.siblings.0.len() > 256 {
            return Err(PathProofVerificationError::TooManySiblings);
        }

        let mut cur_node = match &self.terminal {
            None => TERMINATOR,
            Some(leaf_data) => H::hash_leaf(&leaf_data),
        };

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
    key_path: KeyPath,
    depth: u8,
    terminal: Option<LeafData>,
}

impl VerifiedPathProof {
    /// Get the terminal node. `None` signifies that this path concludes with a [`TERMINATOR`].
    pub fn terminal(&self) -> Option<&LeafData> {
        self.terminal.as_ref()
    }

    /// Get the proven path.
    pub fn path(&self) -> &BitSlice<u8, Msb0> {
        &self.key_path.view_bits()[..self.depth as usize]
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
        let other_path = &key_path.view_bits::<Msb0>()[..self.depth as usize];
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
/// a key. This is always either a terminator or leaf.
pub fn record_path(
    root: Node,
    pages: &impl PageSet,
    key: &KeyPath,
) -> Result<(Node, Siblings), MissingPage> {
    let mut cursor = PageSetCursor::new(root, pages);

    let mut siblings = Vec::new();
    let mut terminal = TERMINATOR;

    for bit in key.view_bits::<Msb0>().iter().by_vals() {
        if crate::trie::is_internal(cursor.node()) {
            let inspector = |left: &Node, right: &Node| {
                siblings.push(if !bit { right.clone() } else { left.clone() });
                bit
            };
            cursor.traverse_children(inspector)?;
        } else {
            terminal = *cursor.node();
            break;
        }
    }

    let siblings: Vec<_> = siblings.into_iter().rev().collect();
    Ok((terminal, Siblings(siblings)))
}
