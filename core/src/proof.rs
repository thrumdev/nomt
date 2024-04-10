//! Proving and verifying inclusion, non-inclusion, and updates to the trie.

use crate::cursor::Cursor;
use crate::trie::{
    self, InternalData, KeyPath, LeafData, Node, NodeHasher, NodeHasherExt, NodeKind, TERMINATOR,
};

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
        if self.siblings.len() > core::cmp::min(key_path.len(), 256) {
            return Err(PathProofVerificationError::TooManySiblings);
        }
        let relevant_path = &key_path[..self.siblings.len()];

        let cur_node = match &self.terminal {
            None => TERMINATOR,
            Some(leaf_data) => H::hash_leaf(&leaf_data),
        };

        let new_root = hash_path::<H>(cur_node, relevant_path, self.siblings.iter().cloned());

        if new_root == root {
            Ok(VerifiedPathProof {
                key_path: relevant_path.into(),
                terminal: self.terminal.clone(),
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
    paths: &[&PathUpdate],
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
            if j != 0 && &path.ops[i - 1].0 >= key {
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

        let sub_root = crate::update::build_sub_trie::<H>(skip, leaf, ops, |_, _| {});

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
            .zip(&path.inner.siblings)
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
fn shared_bits(a: &BitSlice<u8, Msb0>, b: &BitSlice<u8, Msb0>) -> usize {
    a.iter().zip(b.iter()).take_while(|(a, b)| a == b).count()
}
