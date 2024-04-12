//! A generic cursor through a binary merkle trie.
//!
//! This is not intended so much for abstraction as it is for dependency injection and testability
//! and should not be considered stable.

use crate::trie::{InternalData, KeyPath, LeafData, Node};

/// Generic cursor over binary trie storage.
///
/// This is not required to give results that make sense; higher level code is required to ensure
/// that the nodes actually hash up to a common root.
///
/// The API here allows moving out-of-bounds, that is, to a node whose "parent" is a terminal node
/// or a leaf node. In these cases the results are implementation-defined. It is safe to write to
/// out-of-bounds space, but reading from out-of-bounds space does not give defined results.
pub trait Cursor {
    /// The current position of the cursor, expressed as a bit-path and length. Bits after the
    /// length are irrelevant.
    fn position(&self) -> (KeyPath, u8);
    /// The current node.
    fn node(&self) -> Node;
    /// Peek at the sibling node of the current position. At the root, gives the terminator.
    fn peek_sibling(&self) -> Node;

    /// Rewind to the root.
    fn rewind(&mut self);
    /// Jump to the node at the given path. Only the first `depth` bits are relevant.
    /// It is possible to jump out of bounds, that is, to a node whose parent is a terminal.
    fn jump(&mut self, path: KeyPath, depth: u8);

    /// Traverse to the sibling node of the current position. No-op at the root.
    fn sibling(&mut self);
    /// Traverse to a child of the current position. Provide a bit that indicates whether
    /// the left or right child should be taken.
    fn down(&mut self, bit: bool);
    /// Traverse upwards by d bits. No-op if d is greater than the current position length.
    fn up(&mut self, d: u8);

    /// Place a non-leaf node at the current location.
    ///
    /// Writing a node into out-of-bounds space may invalidate the terminal node in the path above
    /// it. This previous terminator must be eventually recalculated or re-submitted.
    ///
    /// It is illegal to write a terminator node into out-of-bounds space.
    fn place_non_leaf(&mut self, node: Node);

    /// Place a leaf node at the current location.
    ///
    /// Writing a node into out-of-bounds space may invalidate the terminal node in the path above
    /// it. This previous terminator must be eventually recalculated or re-submitted.
    fn place_leaf(&mut self, node: Node, leaf: LeafData);

    /// Attempt to compact this node with its sibling.
    ///
    /// Returns an optional `InternalData` if the node and sibling can't be compacted. In all cases,
    /// after returning, the position is equivalent to having called `up(1)`.
    ///
    /// 1. If both this and the sibling are terminators, this moves the cursor up one position
    ///    and replaces the parent with a terminator.
    /// 2. If one of this and the sibling is a leaf, and the other is a terminator, this deletes
    ///    the leaf, moves the cursor up one position, and replaces the parent with the deleted
    ///    leaf.
    /// 3. If either or both is an internal node, this moves the cursor up one position and
    ///    return an internal node data structure comprised of this and this sibling.
    /// 4. This is the root - None is returned.
    fn compact_up(&mut self) -> Option<InternalData>;
}
