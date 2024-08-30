//! Proving and verifying inclusion, non-inclusion, and updates to the trie.

use crate::trie::{
    self, InternalData, KeyPath, LeafData, Node, NodeHasher, NodeHasherExt, NodeKind, TERMINATOR,
};
use crate::trie_pos::TriePosition;

use bitvec::prelude::*;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Wrapper for a terminal node, it will store the LeafData if it is a leaf node,
/// and just the KeyPath to that terminal if it is a terminator node
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PathProofTerminal {
    Leaf(LeafData),
    Terminator(TriePosition),
}

impl PathProofTerminal {
    /// Return the bit-path to the Terminal Node
    pub fn path(&self) -> &BitSlice<u8, Msb0> {
        match self {
            Self::Leaf(leaf_data) => &leaf_data.key_path.view_bits(),
            Self::Terminator(key_path) => key_path.path(),
        }
    }

    pub fn node<H: NodeHasher>(&self) -> Node {
        match self {
            Self::Leaf(leaf_data) => H::hash_leaf(leaf_data),
            Self::Terminator(_key_path) => TERMINATOR,
        }
    }
}

/// A proof of some particular path through the trie.
#[derive(Debug, Clone)]
pub struct PathProof {
    /// The terminal node encountered when looking up a key. This is always either a terminator or
    /// leaf.
    pub terminal: PathProofTerminal,
    /// Sibling nodes encountered during lookup, in descending order by depth.
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
        if self.siblings.len() > core::cmp::min(key_path.len(), 256) {
            return Err(PathProofVerificationError::TooManySiblings);
        }
        let relevant_path = &key_path[..self.siblings.len()];

        let cur_node = self.terminal.node::<H>();

        let new_root = hash_path::<H>(cur_node, relevant_path, self.siblings.iter().rev().cloned());

        if new_root == root {
            Ok(VerifiedPathProof {
                key_path: relevant_path.into(),
                terminal: match &self.terminal {
                    PathProofTerminal::Leaf(leaf_data) => Some(leaf_data.clone()),
                    PathProofTerminal::Terminator(_) => None,
                },
                siblings: self.siblings.clone(),
                root,
            })
        } else {
            Err(PathProofVerificationError::RootMismatch)
        }
    }
}

/// Given a node, a path, and a set of siblings, hash up to the root and return it.
/// This only consumes the last `siblings.len()` bits of the path, or the whole path.
/// Siblings are in ascending order from the last bit of `path`.
pub fn hash_path<H: NodeHasher>(
    mut node: Node,
    path: &BitSlice<u8, Msb0>,
    siblings: impl IntoIterator<Item = Node>,
) -> Node {
    for (bit, sibling) in path.iter().by_vals().rev().zip(siblings) {
        let (left, right) = if bit {
            (sibling, node)
        } else {
            (node, sibling)
        };

        let next = InternalData {
            left: left.clone(),
            right: right.clone(),
        };
        node = H::hash_internal(&next);
    }

    node
}

/// An error type indicating that a key is out of scope of a path proof.
#[derive(Debug, Clone, Copy)]
pub struct KeyOutOfScope;

/// Errors in path proof verification.
#[derive(Debug, Clone, Copy)]
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
    siblings: Vec<Node>,
    root: Node,
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

    /// Get the proven root.
    pub fn root(&self) -> Node {
        self.root
    }

    /// Check whether this path resolves to the given leaf.
    ///
    /// A return value of `Ok(true)` confirms that the key indeed has this value in the trie.
    /// `Ok(false)` confirms that this key has a different value or does not exist.
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

#[derive(Debug, Clone, Copy)]
pub enum VerifyUpdateError {
    PathsOutOfOrder,
    OpsOutOfOrder,
    OpOutOfScope,
    PathWithoutOps,
    /// Paths were verified against different state-roots.
    RootMismatch,
}

/// An update to the node at some path.
pub struct PathUpdate {
    /// The proven path.
    pub inner: VerifiedPathProof,
    /// Update operations to perform on keys that all start with the path.
    pub ops: Vec<(KeyPath, Option<trie::ValueHash>)>,
}

/// Verify an update operation against the root node. This follows a similar algorithm to the
/// multi-item update, but without altering any backing storage.
///
/// Paths should be ascending, ops should be ascending, all ops should look up to one of the
/// paths in `paths`.
///
/// All paths should share the same root.
pub fn verify_update<H: NodeHasher>(
    prev_root: Node,
    paths: &[PathUpdate],
) -> Result<Node, VerifyUpdateError> {
    if paths.iter().any(|p| p.inner.root() != prev_root) {
        return Err(VerifyUpdateError::RootMismatch);
    }

    // left frontier
    let mut pending_siblings: Vec<(Node, usize)> = Vec::new();
    for (i, path) in paths.iter().enumerate() {
        if i != 0 && paths[i - 1].inner.path() >= path.inner.path() {
            return Err(VerifyUpdateError::PathsOutOfOrder);
        }

        for (j, (key, _value)) in path.ops.iter().enumerate() {
            if j != 0 && &path.ops[j - 1].0 >= key {
                return Err(VerifyUpdateError::OpsOutOfOrder);
            }

            if !key.view_bits::<Msb0>().starts_with(path.inner.path()) {
                return Err(VerifyUpdateError::OpOutOfScope);
            }
        }

        if path.ops.is_empty() {
            return Err(VerifyUpdateError::PathWithoutOps);
        }

        let leaf = path.inner.terminal().map(|x| x.clone());
        let ops = &path.ops;
        let skip = path.inner.path().len();

        let up_layers = match paths.get(i + 1) {
            None => skip, // go to root
            Some(p) => {
                let n = shared_bits(p.inner.path(), path.inner.path());
                // n always < skip
                // we want to end at layer n + 1
                skip - (n + 1)
            }
        };

        let ops = crate::update::leaf_ops_spliced(leaf, ops);
        let sub_root = crate::update::build_trie::<H>(skip, ops, |_, _, _| {});

        let mut cur_node = sub_root;
        let mut cur_layer = skip;
        let end_layer = skip - up_layers;
        // iterate siblings up to the point of collision with next path, replacing with pending
        // siblings, and compacting where possible.
        // push (node, end_layer) to pending siblings when done.
        for (bit, sibling) in path
            .inner
            .path()
            .iter()
            .by_vals()
            .rev()
            .take(up_layers)
            .zip(path.inner.siblings.iter().rev())
        {
            let sibling = if pending_siblings.last().map_or(false, |p| p.1 == cur_layer) {
                // unwrap: checked above
                pending_siblings.pop().unwrap().0
            } else {
                *sibling
            };

            match (NodeKind::of(&cur_node), NodeKind::of(&sibling)) {
                (NodeKind::Terminator, NodeKind::Terminator) => {}
                (NodeKind::Leaf, NodeKind::Terminator) => {}
                (NodeKind::Terminator, NodeKind::Leaf) => {
                    // relocate sibling upwards.
                    cur_node = sibling;
                }
                _ => {
                    // otherwise, internal
                    let node_data = if bit {
                        trie::InternalData {
                            left: sibling,
                            right: cur_node,
                        }
                    } else {
                        trie::InternalData {
                            left: cur_node,
                            right: sibling,
                        }
                    };
                    cur_node = H::hash_internal(&node_data);
                }
            }
            cur_layer -= 1;
        }
        pending_siblings.push((cur_node, end_layer));
    }

    // The last `path` iterates up to layer 0, so this always stores the root if paths nonempty.
    Ok(pending_siblings.pop().map(|n| n.0).unwrap_or(TERMINATOR))
}

// TODO: dedup, this appears in `update` as well.
pub fn shared_bits(a: &BitSlice<u8, Msb0>, b: &BitSlice<u8, Msb0>) -> usize {
    a.iter().zip(b.iter()).take_while(|(a, b)| a == b).count()
}
