//! A reference trie model shared by the fuzz targets, independent of the production trie builder.

use arbitrary::Unstructured;
use bitvec::{order::Msb0, view::BitView};
use nomt_core::{
    hasher::{Blake3Hasher, NodeHasher},
    trie::{InternalData, KeyPath, LeafData, Node, ValueHash, TERMINATOR},
};

pub fn bit(key: &KeyPath, depth: usize) -> bool {
    key.view_bits::<Msb0>()[depth]
}

/// Reference hash of the sub-trie holding `leaves`: sorted, distinct, sharing the first `depth` bits.
pub fn node(leaves: &[(KeyPath, ValueHash)], depth: usize) -> Node {
    match leaves {
        [] => TERMINATOR,
        [(key_path, value_hash)] => Blake3Hasher::hash_leaf(&LeafData {
            key_path: *key_path,
            value_hash: *value_hash,
        }),
        _ => {
            let mid = leaves.partition_point(|(k, _)| !bit(k, depth));
            Blake3Hasher::hash_internal(&InternalData {
                left: node(&leaves[..mid], depth + 1),
                right: node(&leaves[mid..], depth + 1),
            })
        }
    }
}

/// Mostly derive keys from existing ones, diverging at a chosen bit, so the trie gets deep.
pub fn new_key(input: &mut Unstructured<'_>, existing: &[KeyPath]) -> arbitrary::Result<KeyPath> {
    let mut key: KeyPath = input.arbitrary()?;
    if !existing.is_empty() && input.ratio(3, 4)? {
        let base = existing[input.choose_index(existing.len())?];
        let diverge_at = input.int_in_range(0..=255)?;
        let bits = key.view_bits_mut::<Msb0>();
        bits[..diverge_at].copy_from_bitslice(&base.view_bits::<Msb0>()[..diverge_at]);
        bits.set(diverge_at, !bit(&base, diverge_at));
    }
    Ok(key)
}
