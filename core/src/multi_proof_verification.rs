use crate::{
    multi_proof::{MultiPathProof, MultiProof},
    proof::{hash_path, KeyOutOfScope, PathProofTerminal},
    trie::{InternalData, KeyPath, LeafData, Node, NodeHasher, NodeHasherExt, TERMINATOR},
};
use bitvec::prelude::*;
use core::cmp::Ordering;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Errors in multi-proof verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MultiProofVerificationError {
    /// Root hash mismatched at the end of the verification.
    RootMismatch,
    /// Extra siblings were provided.
    TooManySiblings,
}

#[derive(Debug, Clone)]
struct VerifiedMultiPath {
    terminal: PathProofTerminal,
    depth: usize,
}

/// A verified multi-proof.
#[derive(Debug, Clone)]
pub struct VerifiedMultiProof {
    inner: Vec<VerifiedMultiPath>,
}

impl VerifiedMultiProof {
    /// Find the index of the path contained in this multi-proof, if any, which would prove
    /// the given key.
    ///
    /// Runtime is O(log(n)) in the number of paths this multi-proof contains.
    pub fn find_index_for(&self, key_path: &KeyPath) -> Result<usize, KeyOutOfScope> {
        let search_result = self.inner.binary_search_by(|v| {
            v.terminal.path()[..v.depth].cmp(&key_path.view_bits::<Msb0>()[..v.depth])
        });

        search_result.map_err(|_| KeyOutOfScope)
    }

    /// Check whether this proves that a key has no value in the trie.
    /// Runtime is O(log(n)) in the number of paths this multi-proof contains.
    ///
    /// A return value of `Ok(true)` confirms that the key indeed has no value in the trie.
    /// A return value of `Ok(false)` means that the key definitely exists within the trie.
    ///
    /// Fails if the key is out of the scope of this proof.
    pub fn confirm_nonexistence(&self, key_path: &KeyPath) -> Result<bool, KeyOutOfScope> {
        let index = self.find_index_for(key_path)?;
        Ok(self.confirm_nonexistence_inner(key_path, index))
    }

    /// Check whether this proves that a key has no value in the trie.
    /// Runtime is O(log(n)) in the number of paths this multi-proof contains.
    ///
    /// A return value of `Ok(true)` confirms that this key indeed has this value in the trie.
    /// A return value of `Ok(false)` means that this key has a different value or does not exist.
    ///
    /// Fails if the key is out of the scope of this proof.
    pub fn confirm_value(&self, expected_leaf: &LeafData) -> Result<bool, KeyOutOfScope> {
        let index = self.find_index_for(&expected_leaf.key_path)?;
        Ok(self.confirm_value_inner(&expected_leaf, index))
    }

    /// Check whether the specific path with index `index` proves that a key has no value in
    /// the trie.
    /// Runtime is O(1).
    ///
    /// A return value of `Ok(true)` confirms that the key indeed has no value in the trie.
    /// A return value of `Ok(false)` means that the key definitely exists within the trie.
    ///
    /// Fails if the key is out of the scope of this path.
    ///
    /// # Panics
    ///
    /// Panics if the index is out-of-bounds.
    pub fn confirm_nonexistence_with_index(
        &self,
        key_path: &KeyPath,
        index: usize,
    ) -> Result<bool, KeyOutOfScope> {
        let path = &self.inner[index];
        let depth = path.depth;
        let in_scope = path.terminal.path()[..depth] == key_path.view_bits::<Msb0>()[..depth];

        if in_scope {
            Ok(self.confirm_nonexistence_inner(key_path, index))
        } else {
            Err(KeyOutOfScope)
        }
    }

    /// Check whether this proves that a key has no value in the trie.
    /// Runtime is O(1).
    ///
    /// A return value of `Ok(true)` confirms that this key indeed has this value in the trie.
    /// A return value of `Ok(false)` means that this key has a different value or does not exist
    /// in the trie.
    ///
    /// Fails if the key is out of the scope of this path.
    ///
    /// # Panics
    ///
    /// Panics if the index is out-of-bounds.
    pub fn confirm_value_with_index(
        &self,
        expected_leaf: &LeafData,
        index: usize,
    ) -> Result<bool, KeyOutOfScope> {
        let path = &self.inner[index];
        let depth = path.depth;
        let in_scope =
            path.terminal.path()[..depth] == expected_leaf.key_path.view_bits::<Msb0>()[..depth];

        if in_scope {
            Ok(self.confirm_value_inner(&expected_leaf, index))
        } else {
            Err(KeyOutOfScope)
        }
    }

    // assume in-scope
    fn confirm_nonexistence_inner(&self, key_path: &KeyPath, index: usize) -> bool {
        match self.inner[index].terminal {
            PathProofTerminal::Terminator(_) => true,
            PathProofTerminal::Leaf(ref leaf_data) => &leaf_data.key_path != key_path,
        }
    }

    // assume in-scope
    fn confirm_value_inner(&self, expected_leaf: &LeafData, index: usize) -> bool {
        match self.inner[index].terminal {
            PathProofTerminal::Terminator(_) => false,
            PathProofTerminal::Leaf(ref leaf_data) => leaf_data == expected_leaf,
        }
    }
}

/// Verify a multi-proof against an expected root.
pub fn verify<H: NodeHasher>(
    multi_proof: &MultiProof,
    root: Node,
) -> Result<VerifiedMultiProof, MultiProofVerificationError> {
    let (new_root, siblings_used) =
        verify_range::<H>(0, &multi_proof.paths, &multi_proof.siblings)?;

    if root != new_root {
        return Err(MultiProofVerificationError::RootMismatch);
    }

    if siblings_used != multi_proof.siblings.len() {
        return Err(MultiProofVerificationError::TooManySiblings);
    }

    let paths = multi_proof
        .paths
        .iter()
        .map(|path| VerifiedMultiPath {
            terminal: path.terminal.clone(),
            depth: path.depth,
        })
        .collect::<Vec<_>>();
    Ok(VerifiedMultiProof { inner: paths })
}

// returns the the node made by verifying this range along with the number of siblings used.
fn verify_range<H: NodeHasher>(
    start_depth: usize,
    paths: &[MultiPathProof],
    siblings: &[Node],
) -> Result<(Node, usize), MultiProofVerificationError> {
    // the range should never be empty except in the first call, if the entire multi-proof is
    // empty.
    if paths.is_empty() {
        return Ok((TERMINATOR, 0));
    }
    if paths.len() == 1 {
        // at a terminal node, 'siblings' will contain all unique
        // nodes, hash them up, and return that
        let terminal_path = &paths[0];
        let unique_len = terminal_path.depth - start_depth;

        let node = hash_path::<H>(
            terminal_path.terminal.node::<H>(),
            &terminal_path.terminal.path()[start_depth..start_depth + unique_len],
            siblings[..unique_len].iter().rev().copied(),
        );

        return Ok((node, unique_len));
    }

    let start_path = &paths[0];
    let end_path = &paths[paths.len() - 1];

    let common_bits = crate::proof::shared_bits(
        &start_path.terminal.path()[start_depth..],
        &end_path.terminal.path()[start_depth..],
    );

    let common_len = start_depth + common_bits;
    // TODO: if `common_len` == 256 the multi-proof is malformed. error

    let uncommon_start_len = common_len + 1;

    // bisect `paths` by finding the first path which starts with the right bit set.
    let search_result = paths.binary_search_by(|item| {
        if !item.terminal.path()[uncommon_start_len - 1] {
            Ordering::Less
        } else {
            Ordering::Greater
        }
    });

    // UNWRAP: always `Err` because we never return Ordering::Equal
    // index always in-bounds because `end_path` has the significant bit set to 1
    // furthermore, the left and right slices must be non-empty because start/end exist and the
    // bisection is based off of them.
    let bisect_idx = search_result.unwrap_err();

    // recurse into the left bisection.
    let (left_node, left_siblings_used) = verify_range::<H>(
        uncommon_start_len,
        &paths[..bisect_idx],
        &siblings[common_bits..],
    )?;

    // now that we know how many siblings were used on the left, we can recurse into the right.
    let (right_node, right_siblings_used) = verify_range::<H>(
        uncommon_start_len,
        &paths[bisect_idx..],
        &siblings[common_bits + left_siblings_used..],
    )?;

    let total_siblings_used = common_bits + left_siblings_used + right_siblings_used;
    // hash up the internal node composed of left/right, then repeatedly apply common siblings.
    let node = hash_path::<H>(
        H::hash_internal(&InternalData {
            left: left_node,
            right: right_node,
        }),
        &start_path.terminal.path()[start_depth..common_len], // == last_path.same...
        siblings[..common_bits].iter().rev().copied(),
    );
    Ok((node, total_siblings_used))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{proof::PathProof, trie};

    /// Hash nodes with blake3.
    pub struct Blake3Hasher;

    impl NodeHasher for Blake3Hasher {
        fn hash_node(data: &trie::NodePreimage) -> [u8; 32] {
            blake3::hash(data).into()
        }
    }

    #[test]
    pub fn test_verify_multiproof_two_leafs() {
        //     root
        //     /  \
        //    s3   v1
        //   / \
        //  v0  v2

        let mut key_path_0 = [0; 32];
        key_path_0[0] = 0b00000000;

        let mut key_path_1 = [0; 32];
        key_path_1[0] = 0b10000000;

        let mut key_path_2 = [0; 32];
        key_path_2[0] = 0b01000000;

        let leaf_0 = LeafData {
            key_path: key_path_0,
            value_hash: [0; 32],
        };

        let leaf_1 = LeafData {
            key_path: key_path_1,
            value_hash: [1; 32],
        };

        let leaf_2 = LeafData {
            key_path: key_path_2,
            value_hash: [2; 32],
        };

        // this is the
        let v0 = Blake3Hasher::hash_leaf(&leaf_0);
        let v1 = Blake3Hasher::hash_leaf(&leaf_1);
        let v2 = Blake3Hasher::hash_leaf(&leaf_2);
        let s3 = Blake3Hasher::hash_internal(&InternalData {
            left: v0.clone(),
            right: v2,
        });
        let root = Blake3Hasher::hash_internal(&InternalData {
            left: s3,
            right: v1,
        });

        let path_proof_0 = PathProof {
            terminal: PathProofTerminal::Leaf(leaf_0.clone()),
            siblings: vec![v1, v2],
        };
        let path_proof_1 = PathProof {
            terminal: PathProofTerminal::Leaf(leaf_1.clone()),
            siblings: vec![s3],
        };

        let multi_proof =
            MultiProof::from_path_proofs(vec![path_proof_0.clone(), path_proof_1.clone()]);

        let verified = verify::<Blake3Hasher>(&multi_proof, root).unwrap();

        assert!(verified.confirm_value(&leaf_0).unwrap());
        assert!(verified.confirm_value(&leaf_1).unwrap());
    }

    #[test]
    pub fn test_verify_multiproof_multiple_leafs() {
        //                       root
        //                 /            \
        //                i3            i6
        //             /     \       /      \
        //            i2      T     v3      i5
        //          /     \                /   \
        //         i1     v2              i4   v5
        //        / \                    /  \
        //       v0  v1                 v4   T

        let path = |byte| [byte; 32];

        let k0 = path(0b00000000);
        let k1 = path(0b00011000);
        let k2 = path(0b00101101);
        let k3 = path(0b10101010);
        let k4 = path(0b11000011);
        let k5 = path(0b11100010);

        let make_leaf = |key_path| {
            let leaf_data = LeafData {
                key_path,
                value_hash: [key_path[0]; 32],
            };

            let hash = Blake3Hasher::hash_leaf(&leaf_data);
            (leaf_data, hash)
        };
        let internal_hash =
            |left, right| Blake3Hasher::hash_internal(&InternalData { left, right });

        let (l0, v0) = make_leaf(k0);
        let (l1, v1) = make_leaf(k1);
        let (l2, v2) = make_leaf(k2);
        let (l3, v3) = make_leaf(k3);
        let (l4, v4) = make_leaf(k4);
        let (l5, v5) = make_leaf(k5);

        let i1 = internal_hash(v0, v1);
        let i2 = internal_hash(i1, v2);
        let i3 = internal_hash(i2, TERMINATOR);

        let i4 = internal_hash(v4, TERMINATOR);
        let i5 = internal_hash(i4, v5);
        let i6 = internal_hash(v3, i5);

        let root = internal_hash(i3, i6);

        let leaf_proof = |leaf, siblings| PathProof {
            terminal: PathProofTerminal::Leaf(leaf),
            siblings,
        };

        let path_proof_0 = leaf_proof(l0.clone(), vec![i6, TERMINATOR, v2, v1]);
        let path_proof_1 = leaf_proof(l1.clone(), vec![i6, TERMINATOR, v2, v0]);
        let path_proof_2 = leaf_proof(l2.clone(), vec![i6, TERMINATOR, i1]);
        let path_proof_3 = leaf_proof(l3.clone(), vec![i3, i5]);
        let path_proof_4 = leaf_proof(l4.clone(), vec![i3, v3, v5, TERMINATOR]);
        let path_proof_5 = leaf_proof(l5.clone(), vec![i3, v3, i4]);

        let multi_proof = MultiProof::from_path_proofs(vec![
            path_proof_0.clone(),
            path_proof_1.clone(),
            path_proof_2.clone(),
            path_proof_3.clone(),
            path_proof_4.clone(),
            path_proof_5.clone(),
        ]);

        let verified = verify::<Blake3Hasher>(&multi_proof, root).unwrap();
        assert!(verified.confirm_value(&l0).unwrap());
        assert!(verified.confirm_value(&l1).unwrap());
        assert!(verified.confirm_value(&l2).unwrap());
        assert!(verified.confirm_value(&l3).unwrap());
        assert!(verified.confirm_value(&l4).unwrap());
        assert!(verified.confirm_value(&l5).unwrap());
    }

    #[test]
    pub fn test_verify_multiproof_siblings_structure() {
        //                           root
        //                     /            \
        //                    v0           i10
        //                               /     \
        //                              i9     e7
        //                            /     \
        //                           i8     e6
        //                       /        \
        //                      i4        i7
        //                     /  \      /  \
        //                    i3   e3   e5   i6
        //                   /  \           /  \
        //                  i2   e2        e4   i5
        //                 /  \               /  \
        //                i1   e1            v3   v4
        //               /  \
        //              v1  v2

        let path = |byte| [byte; 32];

        let k0 = path(0b00000000);
        let k1 = path(0b10000000);
        let k2 = path(0b10000001);
        let k3 = path(0b10011100);
        let k4 = path(0b10011110);

        let make_leaf = |key_path| {
            let leaf_data = LeafData {
                key_path,
                value_hash: [key_path[0]; 32],
            };

            let hash = Blake3Hasher::hash_leaf(&leaf_data);
            (leaf_data, hash)
        };
        let internal_hash =
            |left, right| Blake3Hasher::hash_internal(&InternalData { left, right });

        let (_l0, v0) = make_leaf(k0);
        let (l1, v1) = make_leaf(k1);
        let (l2, v2) = make_leaf(k2);
        let (l3, v3) = make_leaf(k3);
        let (l4, v4) = make_leaf(k4);

        let e1 = [1; 32];
        let e2 = [2; 32];
        let e3 = [3; 32];
        let e4 = [4; 32];
        let e5 = [5; 32];
        let e6 = [6; 32];
        let e7 = TERMINATOR;

        let i1 = internal_hash(v1, v2);
        let i2 = internal_hash(i1, e1);
        let i3 = internal_hash(i2, e2);
        let i4 = internal_hash(i3, e3);

        let i5 = internal_hash(v3, v4);
        let i6 = internal_hash(e4, i5);
        let i7 = internal_hash(e5, i6);

        let i8 = internal_hash(i4, i7);
        let i9 = internal_hash(i8, e6);
        let i10 = internal_hash(i9, e7);

        let root = internal_hash(v0, i10);

        let leaf_proof = |leaf, siblings| PathProof {
            terminal: PathProofTerminal::Leaf(leaf),
            siblings,
        };

        let path_proof_1 = leaf_proof(l1.clone(), vec![v0, e7, e6, i7, e3, e2, e1, v2]);
        let path_proof_2 = leaf_proof(l2.clone(), vec![v0, e7, e6, i7, e3, e2, e1, v1]);
        let path_proof_3 = leaf_proof(l3.clone(), vec![v0, e7, e6, i4, e5, e4, v4]);
        let path_proof_4 = leaf_proof(l4.clone(), vec![v0, e7, e6, i4, e5, e4, v3]);

        let multi_proof = MultiProof::from_path_proofs(vec![
            path_proof_1.clone(),
            path_proof_2.clone(),
            path_proof_3.clone(),
            path_proof_4.clone(),
        ]);

        assert_eq!(multi_proof.siblings, vec![v0, e7, e6, e3, e2, e1, e5, e4]);

        let verified = verify::<Blake3Hasher>(&multi_proof, root).unwrap();
        assert!(verified.confirm_value(&l1).unwrap());
        assert!(verified.confirm_value(&l2).unwrap());
        assert!(verified.confirm_value(&l3).unwrap());
        assert!(verified.confirm_value(&l4).unwrap());
    }
}
