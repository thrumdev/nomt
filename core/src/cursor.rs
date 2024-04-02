//! A generic cursor through a binary merkle trie.
//!
//! This is not intended so much for abstraction as it is for dependency injection and testability
//! and should not be considered stable.

use crate::trie::{KeyPath, Node};

/// Generic cursor over binary trie storage.
///
/// This is not required to give results that make sense; higher level code is required to ensure
/// that the nodes actually hash up to a common root.
///
/// The API here allows moving out-of-bounds, that is, to a node whose parent is a terminal node.
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
    /// Seek to the terminal of the given path. Returns the terminal node and its depth.
    ///
    /// This can be more efficient than using repeated calls to `down` in the case that I/O may
    /// be predicted based on the key-path.
    fn seek(&mut self, path: KeyPath);

    /// Traverse to the sibling node of the current position. No-op at the root.
    fn sibling(&mut self);
    /// Traverse to a child of the current position. Provide a bit that indicates whether
    /// the left or right child should be taken.
    fn down(&mut self, bit: bool);
    /// Traverse upwards by d bits. No-op if d is greater than the current position length.
    fn up(&mut self, d: u8);

    /// Modify the node at the given position.
    fn modify(&mut self, node: Node);
}
