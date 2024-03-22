//! Utilities for proving the results of operations on the trie.

#![cfg(feature = "std")]

use crate::trie::{KeyPath, LeafData, Node};

/// A path to a node in the trie.
pub struct BitPath {
    /// The path through the trie. Only the first `len` bits matter.
    pub path: [u8; 32],
    /// The length of the path.
    pub len: u8,
}

/// A sparse merkle multiproof adapted for NOMT.
pub struct Proof {
    /// The leaves and terminators required to prove the values of the proven keys, tagged with
    /// their depths.
    ///
    /// If the `LeafData` is `Some`, this is a leaf. Otherwise, it's a terminal.
    /// Leaves can be used either to prove the value of a key or to prove that a key doesn't exist
    /// in the trie by contradiction. Terminals can be used to prove that a key doesn't exist in the
    /// trie.
    ///
    /// These should be sorted by depth (descending) and then lexicographically (ascending) within
    /// thesame depth.
    pub relevant_terminals: Vec<(BitPath, Option<LeafData>)>,
    /// All additional nodes required as part of this proof. These are ordered positionally.
    pub siblings: Vec<Node>,
    /// Each proven key, tagged with the index of the relevant terminal node that proves its value
    /// or non-existence.
    pub proven_keys: Vec<(KeyPath, u32)>,
}
