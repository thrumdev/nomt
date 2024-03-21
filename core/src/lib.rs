//! Core operations and types within the Nearly Optimal Merkle Trie.
//!
//! This crate defines the schema and basic operations over the merkle trie in a backend-agnostic
//! manner.
//!
//! The core types and proof verification routines of this crate do not require the
//! standard library. Generating proofs does require the standard library.
//!
//! ## Schema
//!
//! This crate defines a binary merkle trie, generalized over a 256 bit hash function. All lookup
//! paths in the trie are 256 bits.
//!
//! All nodes are 256 bits. There are three kinds of nodes.
//!   1. Internal nodes, which each have two children. The value of an internal node is
//!       given by hashing the concatenation of the two child nodes and setting the MSB to 0.
//!   2. Leaf nodes, which have zero children. The value of a leaf node is given by hashing
//!       the concatenation of the 256-bit lookup path and the hash of the value stored at the leaf,
//!       and setting the MSB to 1.
//!   3. [`TERMINATOR`] nodes, which have the special value of all 0s. These nodes have no children
//!      and serve as a stand-in for an empty sub-trie at any height. Terminator nodes enable the
//!      trie to be tractably represented.
//!
//! All node preimages are 512 bits.

#![cfg_attr(not(feature = "std"), no_std)]

/// A node in the binary trie. In this schema, it is always 256 bits and is the hash of an
/// underlying structure, represented as [`NodePreimage`].
///
/// The first bit of the [`Node`] is used to indicate the kind of node
pub type Node = [u8; 32];

/// The path to a key. All paths have a 256 bit fixed length.
pub type KeyPath = [u8; 32];

/// The hash of a value. In this schema, it is always 256 bits.
pub type ValueHash = [u8; 32];

/// The preimage of a node. In this schema, it is always 512 bits.
///
/// Note that this encoding itself does not contain information about
/// whether the node is a leaf or internal node.
pub type NodePreimage = [u8; 64];

/// The terminator hash is a special node hash value denoting an empty sub-tree.
/// Concretely, when this appears at a given location in the trie,
/// it implies that no key with a path beginning with the location has a value.
///
/// This value may appear at any height.
pub const TERMINATOR: Node = [0u8; 32];

/// Whether the node hash indicates the node is a leaf.
pub fn is_leaf(hash: &Node) -> bool {
    hash[0] >> 7 == 1
}

/// Whether the node hash indicates the node is an internal node.
pub fn is_internal(hash: &Node) -> bool {
    hash[0] >> 7 == 0 && !is_terminator(&hash)
}

/// Whether the node holds the special `EMPTY_SUBTREE` value.
pub fn is_terminator(hash: &Node) -> bool {
    hash == &TERMINATOR
}

/// The data of an internal (branch) node.
pub struct InternalData {
    /// The hash of the left child of this node.
    pub left: Node,
    /// The hash of the right child of this node.
    pub right: Node,
}

impl InternalData {
    /// Encode the internal node.
    pub fn encode(&self) -> NodePreimage {
        let mut node = [0u8; 64];
        node[0..32].copy_from_slice(&self.left[..]);
        node[32..64].copy_from_slice(&self.right[..]);
        node
    }
}

/// The data of a leaf node.
pub struct LeafData {
    /// The total path to this value within the trie.
    ///
    /// The actual location of this node may be anywhere along this path, depending on the other
    /// data within the trie.
    pub key_path: KeyPath,
    /// The hash of the value carried in this leaf.
    pub value_hash: ValueHash,
}

impl LeafData {
    /// Encode the leaf node.
    pub fn encode(&self) -> NodePreimage {
        let mut node = [0u8; 64];
        node[0..32].copy_from_slice(&self.key_path[..]);
        node[32..64].copy_from_slice(&self.value_hash[..]);
        node
    }
}

/// Either kind of node.
pub enum NodeData {
    /// An internal node.
    Internal(InternalData),
    /// A leaf node.
    Leaf(LeafData),
}

/// A trie node hash function specialized for 64 bytes of data.
pub trait NodeHasher {
    /// Hash a node, encoded as exactly 64 bytes of data. This should not
    /// domain-separate the hash.
    fn hash_node(data: &NodePreimage) -> [u8; 32];
}

pub trait NodeHasherExt: NodeHasher {
    /// Hash an internal node. This returns a domain-separated hash.
    fn hash_internal(internal: &InternalData) -> Node {
        let data = internal.encode();
        let mut hash = Self::hash_node(&data);

        // set msb to 0.
        hash[0] &= 0b01111111;
        hash
    }

    /// Hash a leaf node. This returns a domain-separated hash.
    fn hash_leaf(leaf: &LeafData) -> Node {
        let data = leaf.encode();
        let mut hash = Self::hash_node(&data);

        // set msb to 1
        hash[0] |= 0b10000000;
        hash
    }
}
