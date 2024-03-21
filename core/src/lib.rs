//! Core operations and types within the Nearly Optimal Merkle Trie.
//!
//! This crate defines the schema and basic operations over the merkle trie in a backend-agnostic manner.
//!
//! Nothing within this crate relies on the standard library.
//!
//! ## Schema
//!
//! This crate defines a binary merkle trie, generalized over a 256 bit hash function. All keys have 256 bits.
//!
//! There are two kinds of  nodes: [`Internal`] nodes, which each have two children, 
//! and [`Leaf`] nodes which store value hashes.
//! The trie itself does not store values. Sub-tries which contain no values are referenced with 
//! the placeholder [`EMPTY_SUBTREE`] hash. All nodes can be encoded as 512 bit values.
//!
//! Node hashes are domain-separated: the most significant bit is reserved
//! as an indication of the type of node. A node hash with an MSB of 0 refers to an internal node, 
//! and a node hash with an MSB of 1 refers to a leaf node.

#![no_std]

/// The hash of a node. In this schema, it is always 256 bits.
pub type NodeHash = [u8; 32];

/// The path to a key. All paths have a 256 bit fixed length.
pub type KeyPath = [u8; 32];

/// The hash of a value. In this schema, it is always 256 bits.
pub type ValueHash = [u8; 32];

/// The encoded state of a node. In this schema, it is always 512 bits.
///
/// Note that this encoding itself does not contain information about
/// whether the node is a leaf or internal node.
pub type NodeEncoding = [u8; 64];

/// Special node hash value denoting an empty sub-tree. Concretely, when this appears within
/// an internal node, it implies that no keys beginning with the path to the internal node plus the child's bit
/// will have values within the trie. This value may appear within an internal node at any height, including the root.
pub const EMPTY_SUBTREE: NodeHash = [0u8; 32];

/// Whether the node hash indicates the node is a leaf.
pub fn is_leaf(hash: &NodeHash) -> bool {
    hash[0] >> 7 == 1
}

/// Whether the node hash indicates the node is an internal node.
pub fn is_internal(hash: &NodeHash) -> bool {
    hash[0] >> 7 == 0 && !is_empty(&hash)
}

/// Whether the node holds the special `EMPTY_SUBTREE` value.
pub fn is_empty(hash: &NodeHash) -> bool {
    hash == &EMPTY_SUBTREE
}

/// An internal (branch) node. This carries a left and right child.
pub struct Internal {
    /// The hash of the left child of this node.
    pub left: NodeHash,
    /// The hash of the right child of this node.
    pub right: NodeHash,
}

impl Internal {
    /// Encode the internal node.
    pub fn encode(&self) -> NodeEncoding {
        let mut node = [0u8; 64];
        node[0..32].copy_from_slice(&self.left[..]);
        node[32..64].copy_from_slice(&self.right[..]);
        node
    }
}

/// A leaf node carrying a value.
pub struct Leaf {
    /// The total path to this value within the trie.
    ///
    /// The actual location of this node may be anywhere along this path, depending on the other data
    /// within the trie.
    pub key_path: KeyPath,
    /// The hash of the value carried in this leaf.
    pub value_hash: ValueHash,
}

impl Leaf {
    /// Encode the leaf node.
    pub fn encode(&self) -> NodeEncoding {
        let mut node = [0u8; 64];
        node[0..32].copy_from_slice(&self.key_path[..]);
        node[32..64].copy_from_slice(&self.value_hash[..]);
        node
    }
}

/// Either kind of node.
pub enum Node {
    /// An internal node.
    Internal(Internal),
    /// A leaf node.
    Leaf(Leaf),
}

/// A trie node hash function specialized for 64 bytes of data.
pub trait NodeHasher {
    /// Hash a node, encoded as exactly 64 bytes of data. This should not
    /// domain-separate the hash.
    fn hash_node(data: &[u8; 64]) -> NodeHash;
}

pub trait NodeHasherExt: NodeHasher {
    /// Hash an internal node. This returns a domain-separated hash.
    fn hash_internal(internal: &Internal) -> NodeHash {
        let data = internal.encode();
        let mut hash = Self::hash_node(&data);

        // set msb to 0.
        hash[0] &= 0b01111111;
        hash
    }

    /// Hash a leaf node. This returns a domain-separated hash.
    fn hash_leaf(leaf: &Leaf) -> NodeHash {
        let data = leaf.encode();
        let mut hash = Self::hash_node(&data);

        // set msb to 1
        hash[0] |= 0b10000000;
        hash
    }
}
