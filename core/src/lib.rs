//! Core operations and types within the Nearly Optimal Merkle Trie.
//!
//! This crate defines the schema and basic operations over the merkle trie in a backend-agnostic manner.
//! 
//! Nothing within this crate relies on the standard library.

#![no_std]

/// The hash of a node. In this schema, it is always 256 bits.
pub type NodeHash = [u8; 32];

/// Denotes an empty sub-tree beyond this point.
pub const NODE_EMPTY: NodeHash = [0u8; 32];

/// Whether the node hash indicates the node is a leaf.
pub fn is_leaf(hash: &NodeHash) -> bool {
    hash[0] >> 7 == 1
}

/// Whether the node hash indicates the node is an internal node.
pub fn is_internal(hash: &NodeHash) -> bool {
    hash[0] >> 7 == 0 && !is_empty(&hash)
}

/// Whether the node holds the special `NODE_EMPTY` value.
pub fn is_empty(hash: &NodeHash) -> bool {
    hash == &NODE_EMPTY
}