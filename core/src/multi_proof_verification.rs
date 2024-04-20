use crate::{
    multi_proof::{MultiProof, SubPathProof},
    proof::{
        hash_path, shared_bits, PathProofTerminal, PathProofVerificationError, VerifiedPathProof,
    },
    trie::{InternalData, KeyPath, LeafData, Node, NodeHasher, NodeHasherExt, TERMINATOR},
};
use alloc::collections::BTreeMap;
use bitvec::{order::Msb0, vec::BitVec, view::BitView};

/// A verifiable proof of multiple path through the trie.
#[derive(Debug, Clone)]
pub struct VerifiableMultiProof<H: NodeHasher> {
    nodes: BTreeMap<BitVec<u8, Msb0>, Node>,
    terminals: BTreeMap<KeyPath, Option<LeafData>>,
    root: Node,
    _phantom: core::marker::PhantomData<H>,
}

struct SubTrieRoot {
    root: Node,
    key_path: BitVec<u8, Msb0>,
}

#[derive(Debug, Clone)]
pub enum MultiProofVerificationError {
    IncorrectMultiProofFormat,
}

impl<H: NodeHasher> TryFrom<MultiProof> for VerifiableMultiProof<H> {
    type Error = MultiProofVerificationError;

    fn try_from(multi_proof: MultiProof) -> Result<Self, Self::Error> {
        // collections of all nodes that will be used to verify each key_path
        let mut nodes_map: BTreeMap<BitVec<u8, Msb0>, Node> = BTreeMap::new();
        // collection of all proven terminal nodes
        let mut terminals_map: BTreeMap<KeyPath, Option<LeafData>> = BTreeMap::new();

        // 1. Create sub_tree A (using the first available sub_tree).
        // 2. Create sub_tree B (using the next available sub_tree).
        // 3. Are A and B siblings?
        //    - If yes: Create a new internal node with A and B as children.
        //      This node will replace A and jump to step 2
        //    - If no: Proceed to step 4.
        // 4. A attempts to reach B by utilizing `n` nodes as siblings
        //    to hash up the tree and generate new internal nodes.
        //    Where n is `A.depth - shared_bits(A, B) - 1`.
        //    Choose siblings from the vector of ext_siblings or
        //    the last node in the stack by considering:
        //    - if the last node in the stack is a sibling used it
        //    (and remove it from the stack)
        //    - if not then pop an external sibling
        //    If n equals zero, proceed to step 5, otherwise, return to step 3
        // 5. If A and B are not siblings, A is already the root of its sub_tree
        //    and cannot ascend further without constructing B's subtree.
        //    Push A into the stack and B becomes A. Return to step 2.
        //
        // If there are no more sub_trees available in step 2,
        // reach the root using `A.depth` as `n` in step 4.
        //
        // During the entire traversal, save each node in a map
        // with its corresponding key_path.
        //
        // A and B are denoted as curr_sub_tree and next_sub_tree.

        let mut sub_paths_iter = multi_proof.sub_paths.into_iter();
        let mut ext_siblings = multi_proof.external_siblings.into_iter().rev();

        let Some(sub_path) = sub_paths_iter.next() else {
            // a multiproof with no terminal to prove is not meaningful
            return Err(MultiProofVerificationError::IncorrectMultiProofFormat);
        };

        // Construct the first sub_tree and save into nodes all the sibilings
        insert_sub_path::<H>(&sub_path, &mut nodes_map, &mut terminals_map);
        let mut curr_sub_tree = SubTrieRoot {
            key_path: relevant_key_path(&sub_path),
            root: hash_sub_tree::<H>(sub_path, &mut nodes_map),
        };

        let mut sub_trie_stack: Vec<SubTrieRoot> = vec![];

        'sub_path_loop: while let Some(next_sub_path) = sub_paths_iter.next() {
            // Construct next sub_tree and save into nodes all the sibilings
            insert_sub_path::<H>(&next_sub_path, &mut nodes_map, &mut terminals_map);
            let next_sub_tree = SubTrieRoot {
                key_path: relevant_key_path(&next_sub_path),
                root: hash_sub_tree::<H>(next_sub_path, &mut nodes_map),
            };

            'compact_loop: loop {
                if are_siblings(&curr_sub_tree.key_path, &next_sub_tree.key_path) {
                    // Join two subtrees and save the newly create internal node
                    curr_sub_tree = join_sub_trees::<H>(curr_sub_tree, next_sub_tree);

                    // There could be the possibility that the just created node
                    // is the root so let's avoid adding it to the map
                    if !curr_sub_tree.key_path.is_empty() {
                        nodes_map.insert(curr_sub_tree.key_path.clone(), curr_sub_tree.root);
                    }

                    continue 'sub_path_loop;
                }

                let n_nodes_up = curr_sub_tree.key_path.len()
                    - shared_bits(&curr_sub_tree.key_path, &next_sub_tree.key_path)
                    - 1;

                // if curr_sub_trie does not need to hash up the trie it's time to
                // move to the next sub tree
                if n_nodes_up == 0 {
                    break 'compact_loop;
                }

                hash_up_n_nodes::<H>(
                    &mut curr_sub_tree,
                    n_nodes_up,
                    &mut sub_trie_stack,
                    &mut ext_siblings,
                    &mut nodes_map,
                )?;
            }

            sub_trie_stack.push(curr_sub_tree);
            curr_sub_tree = next_sub_tree;
        }

        let depth = curr_sub_tree.key_path.len();
        hash_up_n_nodes::<H>(
            &mut curr_sub_tree,
            depth,
            &mut sub_trie_stack,
            &mut ext_siblings,
            &mut nodes_map,
        )?;

        // curr_sub_tree is now the root, its key_path must be empty.
        // Additionally, sub_trie_stack and ext_siblings must be empty.
        if !curr_sub_tree.key_path.is_empty() || !sub_trie_stack.is_empty() {
            return Err(MultiProofVerificationError::IncorrectMultiProofFormat);
        }
        if let Some(_) = ext_siblings.next() {
            return Err(MultiProofVerificationError::IncorrectMultiProofFormat);
        }

        // nodes_map is filled with all necessary siblings in the trie.
        Ok(Self {
            nodes: nodes_map,
            terminals: terminals_map,
            root: curr_sub_tree.root,
            _phantom: Default::default(),
        })
    }
}

// Helper function used to save all siblings found in a SubPathProof
// into the nodes map, along with its terminal node into terminals_map
fn insert_sub_path<H: NodeHasher>(
    sub_path: &SubPathProof,
    nodes_map: &mut BTreeMap<BitVec<u8, Msb0>, Node>,
    terminals_map: &mut BTreeMap<KeyPath, Option<LeafData>>,
) {
    let terminal = match &sub_path.terminal {
        PathProofTerminal::Leaf(leaf_data) => Some(leaf_data.clone()),
        PathProofTerminal::Terminator(_) => None,
    };

    terminals_map.insert(*sub_path.terminal.key_path(), terminal);

    let sub_path_depth = sub_path.depth as usize;
    let relevant_sub_path = &sub_path.terminal.key_path().view_bits::<Msb0>()
        [..sub_path_depth + sub_path.inner_siblings.len()];

    // There are some edge cases where the terminals are used as siglings
    // of other nodes let's insert them in both map
    nodes_map.insert(relevant_sub_path.into(), sub_path.terminal.node::<H>());

    // save all sibling with their path to construct the map of nodes
    for (index, sibling) in sub_path.inner_siblings.iter().enumerate() {
        let mut curr_sibling_path =
            BitVec::<u8, Msb0>::from(&relevant_sub_path[0..sub_path_depth + index + 1]);
        {
            let mut last_bit = curr_sibling_path
                .last_mut()
                .expect("There must be at least one bit in the key_path");
            *last_bit = !last_bit.clone();
        }

        nodes_map.insert(curr_sibling_path, sibling.clone());
    }
}

// Helper function used to hash a sub_tree described by a sub_path
fn hash_sub_tree<H: NodeHasher>(
    sub_path: SubPathProof,
    map: &mut BTreeMap<BitVec<u8, Msb0>, Node>,
) -> Node {
    let curr_node = sub_path.terminal.node::<H>();
    let sub_path_depth = sub_path.depth as usize;
    let relevant_path = &sub_path.terminal.key_path().view_bits::<Msb0>()
        [sub_path_depth..sub_path_depth + sub_path.inner_siblings.len()];

    let sub_tree_root = hash_path::<H>(
        curr_node,
        relevant_path,
        sub_path.inner_siblings.into_iter().rev(),
    );

    if relevant_path.len() > 0 {
        println!(
            "inserting sub tree root {:?}",
            BitVec::<u8, Msb0>::from(
                &sub_path.terminal.key_path().view_bits::<Msb0>()[..sub_path.depth as usize]
            )
        );
        map.insert(
            sub_path.terminal.key_path().view_bits::<Msb0>()[..sub_path.depth as usize].into(),
            sub_tree_root,
        );
    }
    sub_tree_root
}

// Extract only the key_path required to reach the root of the sub_tree
// defined by the SubPathProof
fn relevant_key_path(sub_path: &SubPathProof) -> BitVec<u8, Msb0> {
    BitVec::<u8, Msb0>::from(
        &sub_path.terminal.key_path().view_bits::<Msb0>()[..sub_path.depth as usize],
    )
}

// Check if two bitvecs represent two siblings node
//
// Giving the same key twice is considered unexpected and will return true
fn are_siblings(key_path1: &BitVec<u8, Msb0>, key_path2: &BitVec<u8, Msb0>) -> bool {
    if key_path1.len() != key_path2.len() {
        return false;
    }

    let d = key_path1.len() - 1;
    key_path1[..d] == key_path2[..d]
}

// Accept two SubTrieRoots representing the left (a) and right (b) child nodes
// that will be used to create a new internal node
fn join_sub_trees<H: NodeHasher>(mut a: SubTrieRoot, b: SubTrieRoot) -> SubTrieRoot {
    let parent = InternalData {
        left: a.root.clone(),
        right: b.root.clone(),
    };
    // remove last bit
    let _ = a.key_path.pop();
    SubTrieRoot {
        root: H::hash_internal(&parent),
        key_path: a.key_path,
    }
}

// This helper function hashes up the tree starting from `curr_node` by `n_nodes`,
// using the stack of previously formed sub_trees or an external sibling.
// Newly created nodes are saved in `nodes_map`.
fn hash_up_n_nodes<H: NodeHasher>(
    curr_node: &mut SubTrieRoot,
    n_nodes: usize,
    sub_trie_stack: &mut Vec<SubTrieRoot>,
    ext_siblings: &mut impl Iterator<Item = Node>,
    nodes_map: &mut BTreeMap<BitVec<u8, Msb0>, Node>,
) -> Result<(), MultiProofVerificationError> {
    for bit in curr_node
        .key_path
        .clone()
        .iter()
        .by_vals()
        .rev()
        .take(n_nodes)
    {
        let sibling = match sub_trie_stack.last() {
            Some(sub_tree_in_stack)
                if are_siblings(&curr_node.key_path, &sub_tree_in_stack.key_path) =>
            {
                sub_trie_stack
                    .pop()
                    .expect("The existence of the last element in the stack has just been verified")
                    .root
            }
            Some(_) | None => {
                let ext_sibiling = ext_siblings
                    .next()
                    .ok_or(MultiProofVerificationError::IncorrectMultiProofFormat)?;

                let mut sibling_path = curr_node.key_path.clone();
                {
                    let mut last_bit = sibling_path
                        .last_mut()
                        .expect("There must be at least one bit in the key_path");
                    *last_bit = !last_bit.clone();
                }

                nodes_map.insert(sibling_path, ext_sibiling);
                ext_sibiling
            }
        };

        let parent = InternalData {
            left: if bit { sibling } else { curr_node.root },
            right: if bit { curr_node.root } else { sibling },
        };

        curr_node.root = H::hash_internal(&parent);
        curr_node.key_path.pop();

        nodes_map.insert(curr_node.key_path.clone(), curr_node.root);
    }
    Ok(())
}

impl<H: NodeHasher> VerifiableMultiProof<H> {
    pub fn verify(
        &self,
        key_path: KeyPath,
    ) -> Result<VerifiedPathProof, PathProofVerificationError> {
        // all sibilings following the provided key_path are extracted form the nodes map,
        // sibilings are finished when a fetch of a sibling fails
        let mut key_path_bits = key_path.view_bits::<Msb0>().iter().by_vals().enumerate();
        let siblings: Vec<Node> = core::iter::from_fn(|| {
            let (index, bit) = key_path_bits.next()?;
            let mut sibling_key = BitVec::<u8, Msb0>::from(&key_path.view_bits::<Msb0>()[0..index]);
            sibling_key.push(!bit);

            self.nodes.get(&sibling_key).map(|node| node.clone())
        })
        .collect();

        let relevant_path = &key_path.view_bits::<Msb0>()[..siblings.len()];
        let terminal = self
            .terminals
            .get(&key_path)
            .ok_or(PathProofVerificationError::NotVerifiableKeyPath)?;

        let terminal_node = match terminal {
            Some(leaf_data) => H::hash_leaf(leaf_data),
            None => TERMINATOR,
        };

        let new_root = hash_path::<H>(terminal_node, relevant_path, siblings.iter().rev().cloned());

        if new_root == self.root {
            Ok(VerifiedPathProof {
                key_path: relevant_path.into(),
                terminal: terminal.clone(),
                siblings: siblings.clone(),
                root: self.root,
            })
        } else {
            Err(PathProofVerificationError::RootMismatch)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{proof::PathProof, trie};
    use bitvec::bits;

    /// Hash nodes with blake3.
    pub struct Blake3Hasher;

    impl NodeHasher for Blake3Hasher {
        fn hash_node(data: &trie::NodePreimage) -> [u8; 32] {
            blake3::hash(data).into()
        }
    }

    #[test]
    pub fn test_verifialbe_multiproof_creation() {
        let mut key_path_0 = [0; 32];
        key_path_0[0] = 0b00000000;

        let mut key_path_1 = [0; 32];
        key_path_1[0] = 0b00110000;

        let mut key_path_2 = [0; 32];
        key_path_2[0] = 0b00111000;

        let mut key_path_3 = [0; 32];
        key_path_3[0] = 0b11000000;

        let sibiling1 = [1; 32];
        let sibiling2 = [2; 32];
        let sibiling3 = [3; 32];
        let sibiling4 = [4; 32];
        let sibiling5 = [5; 32];
        let sibiling6 = [6; 32];
        let sibiling7 = [7; 32];
        let sibiling8 = [8; 32];
        let sibiling9 = [9; 32];
        let sibiling10 = [10; 32];
        let sibiling11 = [11; 32];

        let path_proof_0 = PathProof {
            terminal: PathProofTerminal::Terminator(key_path_0),
            siblings: vec![sibiling1, sibiling2, sibiling3, sibiling4],
        };
        let path_proof_1 = PathProof {
            terminal: PathProofTerminal::Terminator(key_path_1),
            siblings: vec![sibiling1, sibiling2, sibiling5, sibiling6, sibiling7],
        };
        let path_proof_2 = PathProof {
            terminal: PathProofTerminal::Terminator(key_path_2),
            siblings: vec![sibiling1, sibiling2, sibiling5, sibiling6, sibiling8],
        };
        let path_proof_3 = PathProof {
            terminal: PathProofTerminal::Terminator(key_path_3),
            siblings: vec![sibiling9, sibiling10, sibiling11],
        };

        let multi_proof = MultiProof::from_path_proofs(vec![
            path_proof_0.clone(),
            path_proof_1.clone(),
            path_proof_2.clone(),
            path_proof_3.clone(),
        ]);

        let verifiable_multi_proof =
            VerifiableMultiProof::<Blake3Hasher>::try_from(multi_proof).unwrap();

        assert_eq!(verifiable_multi_proof.nodes.len(), 15);
        assert!(verifiable_multi_proof
            .nodes
            .contains_key(bits![u8, Msb0; 0]));
        assert!(verifiable_multi_proof
            .nodes
            .contains_key(bits![u8, Msb0; 0, 1]));
        assert!(verifiable_multi_proof
            .nodes
            .contains_key(bits![u8, Msb0; 0, 0, 1]));
        assert!(verifiable_multi_proof
            .nodes
            .contains_key(bits![u8, Msb0; 0, 0, 1, 0]));
        assert!(verifiable_multi_proof
            .nodes
            .contains_key(bits![u8, Msb0; 0, 0, 0]));
        assert!(verifiable_multi_proof
            .nodes
            .contains_key(bits![u8, Msb0; 0, 0, 0, 1]));
        assert!(verifiable_multi_proof
            .nodes
            .contains_key(bits![u8, Msb0; 1]));
        assert!(verifiable_multi_proof
            .nodes
            .contains_key(bits![u8, Msb0; 1, 0]));
        assert!(verifiable_multi_proof
            .nodes
            .contains_key(bits![u8, Msb0; 1, 1, 1]));
    }

    #[test]
    pub fn test_verify_multiproof_two_leafs() {
        //     root
        //     /  \
        //    s3   v1
        //   / \
        //  v0  s2

        let mut key_path_0 = [0; 32];
        key_path_0[0] = 0b00000000;

        let mut key_path_1 = [0; 32];
        key_path_1[0] = 0b10000000;

        let leaf_0 = LeafData {
            key_path: key_path_0,
            value_hash: [0; 32],
        };

        let leaf_1 = LeafData {
            key_path: key_path_1,
            value_hash: [1; 32],
        };

        // this is the
        let v0 = Blake3Hasher::hash_leaf(&leaf_0);
        let v1 = Blake3Hasher::hash_leaf(&leaf_1);
        let s2 = [0; 32];
        let s3 = Blake3Hasher::hash_internal(&InternalData {
            left: v0.clone(),
            right: s2.clone(),
        });
        //let root = Blake3Hasher::hash_internal(&InternalData {
        //    left: s3.clone(),
        //    right: v1.clone(),
        //});

        let path_proof_0 = PathProof {
            terminal: PathProofTerminal::Leaf(leaf_0),
            siblings: vec![v1, s2],
        };
        let path_proof_1 = PathProof {
            terminal: PathProofTerminal::Leaf(leaf_1),
            siblings: vec![s3],
        };

        let multi_proof =
            MultiProof::from_path_proofs(vec![path_proof_0.clone(), path_proof_1.clone()]);

        let verifiable_multi_proof =
            VerifiableMultiProof::<Blake3Hasher>::try_from(multi_proof).unwrap();

        let proof = verifiable_multi_proof.verify(key_path_0).unwrap();
        assert_eq!(proof.siblings, path_proof_0.siblings);
        let proof = verifiable_multi_proof.verify(key_path_1).unwrap();
        assert_eq!(proof.siblings, path_proof_1.siblings);
    }

    #[test]
    pub fn test_verify_multiproof_multiple_leafs() {
        //                       root
        //                 /            \
        //                i7            i8
        //             /     \       /      \
        //            i4     s6     i5      i6
        //          /     \        /  \    /   \
        //         i1     i2      v2  s4  i3   s5
        //        / \     / \            /  \
        //       v0  s1  s2  v1         v3  s3

        let mut key_path_0 = [0; 32];
        key_path_0[0] = 0b00000000;
        let mut key_path_1 = [0; 32];
        key_path_1[0] = 0b00110000;
        let mut key_path_2 = [0; 32];
        key_path_2[0] = 0b10000000;
        let mut key_path_3 = [0; 32];
        key_path_3[0] = 0b11000000;

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
        let leaf_3 = LeafData {
            key_path: key_path_3,
            value_hash: [3; 32],
        };

        let v0 = Blake3Hasher::hash_leaf(&leaf_0);
        let v1 = Blake3Hasher::hash_leaf(&leaf_1);
        let v2 = Blake3Hasher::hash_leaf(&leaf_2);
        let v3 = Blake3Hasher::hash_leaf(&leaf_3);

        let s1 = [1; 32];
        let s2 = [2; 32];
        let s3 = [3; 32];
        let s4 = [4; 32];
        let s5 = [5; 32];
        let s6 = [6; 32];

        let i1 = Blake3Hasher::hash_internal(&InternalData {
            left: v0.clone(),
            right: s1.clone(),
        });
        let i2 = Blake3Hasher::hash_internal(&InternalData {
            left: s2.clone(),
            right: v1.clone(),
        });
        let i3 = Blake3Hasher::hash_internal(&InternalData {
            left: v3.clone(),
            right: s3.clone(),
        });
        let i4 = Blake3Hasher::hash_internal(&InternalData {
            left: i1.clone(),
            right: i2.clone(),
        });
        let i5 = Blake3Hasher::hash_internal(&InternalData {
            left: v2.clone(),
            right: s4.clone(),
        });
        let i6 = Blake3Hasher::hash_internal(&InternalData {
            left: i3.clone(),
            right: s5.clone(),
        });
        let i7 = Blake3Hasher::hash_internal(&InternalData {
            left: i4.clone(),
            right: s6.clone(),
        });
        let i8 = Blake3Hasher::hash_internal(&InternalData {
            left: i5.clone(),
            right: i6.clone(),
        });

        let path_proof_0 = PathProof {
            terminal: PathProofTerminal::Leaf(leaf_0),
            siblings: vec![i8, s6, i2, s1],
        };
        let path_proof_1 = PathProof {
            terminal: PathProofTerminal::Leaf(leaf_1),
            siblings: vec![i8, s6, i1, s2],
        };
        let path_proof_2 = PathProof {
            terminal: PathProofTerminal::Leaf(leaf_2),
            siblings: vec![i7, i6, s4],
        };
        let path_proof_3 = PathProof {
            terminal: PathProofTerminal::Leaf(leaf_3),
            siblings: vec![i7, i5, s5, s3],
        };

        let multi_proof = MultiProof::from_path_proofs(vec![
            path_proof_0.clone(),
            path_proof_1.clone(),
            path_proof_2.clone(),
            path_proof_3.clone(),
        ]);

        let verifiable_multi_proof =
            VerifiableMultiProof::<Blake3Hasher>::try_from(multi_proof).unwrap();
        //assert_eq!(verifiable_multi_proof.nodes.len(), 16);

        let proof = verifiable_multi_proof.verify(key_path_0).unwrap();
        assert_eq!(proof.siblings, path_proof_0.siblings);
        let proof = verifiable_multi_proof.verify(key_path_1).unwrap();
        assert_eq!(proof.siblings, path_proof_1.siblings);
        let proof = verifiable_multi_proof.verify(key_path_2).unwrap();
        assert_eq!(proof.siblings, path_proof_2.siblings);
        let proof = verifiable_multi_proof.verify(key_path_3).unwrap();
        assert_eq!(proof.siblings, path_proof_3.siblings);
    }
}
