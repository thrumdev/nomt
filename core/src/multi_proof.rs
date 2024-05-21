//! Generate a multiproof from a vector of path proofs.
//! The multiproof will contain the minimum information needed to verify
//! the inclusion of all provided path proofs.

use crate::{
    proof::{PathProof, PathProofTerminal},
    trie::Node,
};

/// Each terminal node may have a unique set of sibling nodes
/// that are exclusively used in its verification process.
/// This struct includes the terminal node, the depth from which
/// the siblings are uniquely associated with that terminal,
/// and the siblings themselves.
#[derive(Debug, Clone)]
pub struct SubPathProof {
    /// Terminal node
    pub terminal: PathProofTerminal,
    /// Depth after which sibligns are collected
    pub depth: usize,
    /// Siblings uniquely associated with the terminal node,
    /// stored in the order in which they are encountered.
    pub inner_siblings: Vec<Node>,
}

/// A proof of multiple paths through the trie.
#[derive(Debug, Clone)]
pub struct MultiProof {
    /// All subpaths related to a single terminal node. These are sorted, ascending, by bit-path.
    pub sub_paths: Vec<SubPathProof>,
    /// Vector containing the minimum number of nodes required
    /// to reconstruct all other nodes later.
    ///
    /// The format is a recursive bisection:
    /// [common_siblings ++ right_siblings ++ left_siblings]
    ///
    /// common_siblings are the siblings shared among all paths in each bisection or all path proofs
    /// at the root.
    ///
    /// right_siblings is the same format, but applied to all the nodes in the right bisection.
    /// left_siblings is the same format, but applied to all the nodes in the left bisection.
    pub external_siblings: Vec<Node>,
}

impl MultiProof {
    /// Construct a MultiProof from a vector of *ordered* PathProof
    pub fn from_path_proofs(path_proofs: Vec<PathProof>) -> Self {
        // A multi-proof can be viewed by associating each terminal node
        // with its first n uniquely related siblings from its path proof,
        // followed by all necessary siblings that are not derivable from
        // the previously mentioned siblings. Those siblings will be called
        // `external_siblings`.
        let mut sub_paths: Vec<SubPathProof> = vec![];
        let mut external_siblings: Vec<Node> = vec![];

        // The goal is to traverse the entire tree
        // formed by all path proofs and only collect the siblings
        // that cannot be reconstructed in the future
        //
        // The traversal does not occur on the siblings themselves,
        // but on the ordered set of key_paths within PathProofs.
        //
        // For example, take two key_paths starting from the first bit.
        // If two keys share bits up to index i, they will have the same sibling at index i.
        // If the bit at index i differs, their paths diverge, and no sibling is needed at that index
        // because it can be reconstructed from siblings collected later in the other key.
        // When a differing bit is found, each key_path needs separate scanning.
        //
        // When there is a single key_path, all siblings are necessary.
        //
        // When there are more than two key_paths, the same logic is applied,
        // but if two key_paths have different bits at an index, the key_paths are split
        // based on the value at that index.
        //
        // If all key_paths share a bit at an index, that sibling is required and it is one
        // of the `external_siblings` mentioned earlier.
        // If at least two key_paths differ, no sibling is needed, but the key_paths must be divided
        // based on having bit 0 or 1 at that index.
        //
        // Iterate this algorithm on the bisection to determine the minimum necessary siblings.
        //
        // `external_siblings` will follow this structure for each bisection
        // |common siblings| ext siblings in the right bisection | ext siblings in the left bisection |

        // it represent a slice of the vector of key_paths with the notion of
        // path_index, at which bit in the key_path that slice is pointing at

        struct TraverseInfo {
            // lower bound of the slice, index of `path_proofs`
            lower: usize,
            // inclusive upper bound of the slice, index of `path_proofs`
            upper: usize,
            // bit index in the key path where this struct is pointing at,
            // this index will be increased or used to create a new bisection
            //
            // each TraverseInfo covers a slice of `path_proofs`
            // with key_paths that have common bits up to path_bit_index
            path_bit_index: usize,
            // TODO
            ext_sibling_index: usize,
        }
        let mut stack_traverse_info: Vec<TraverseInfo> = vec![];

        // External siblings encountered during a subtree traversal
        let mut ext_sibling_in_sub_tree: Vec<Node> = vec![];

        // initially we're looking at all the key_paths
        let mut curr_traverse_info = TraverseInfo {
            path_bit_index: 0,
            lower: 0,
            upper: path_proofs.len() - 1,
            ext_sibling_index: 0,
        };

        loop {
            // If there is no longer a common prefix, there is only one key in the TraverseInfo,
            // and then all remaining siblings are required for the multiproof.
            if curr_traverse_info.lower == curr_traverse_info.upper {
                let path_proof = &path_proofs[curr_traverse_info.lower];
                let sub_path_proof = SubPathProof {
                    terminal: path_proof.terminal.clone(),
                    // The depth at which the terminal starts not sharing any siblings
                    depth: curr_traverse_info.path_bit_index,
                    inner_siblings: path_proof
                        .siblings
                        .iter()
                        .skip(curr_traverse_info.path_bit_index)
                        .copied()
                        .collect(),
                };
                sub_paths.push(sub_path_proof);

                // terminal always immediately follows a bisection in a well-formed trie.
                assert!(ext_sibling_in_sub_tree.is_empty());

                // skip to the next bisection in the stack, if empty we're finished
                curr_traverse_info = match stack_traverse_info.pop() {
                    Some(i) => i,
                    None => break,
                };
                continue;
            }

            // check if at least two key_path in the key_paths slice
            // has two different bits in position path_index
            //
            // they are ordered and all the bits up to path_index
            // are shared so we can just check the first and the last one differs
            let path_lower = path_proofs[curr_traverse_info.lower].terminal.key_path();
            let path_upper = path_proofs[curr_traverse_info.upper].terminal.key_path();

            if path_lower[curr_traverse_info.path_bit_index]
                != path_upper[curr_traverse_info.path_bit_index]
            {
                // if they differ we can skip their siblings but we need to bisect the slice
                //
                // binary search between key_paths in the slice to see where to
                // perform the bisection

                // path_slice_upper and path_slice_lower will never be the same otherwise
                // bits at path_index would have been the same

                // We have just checked that path_lower and path_upper differ at path_bit_index,
                // thus, with the vector of path_proofs ordered, we're sure there is at least one
                // path with its key_path containing a value of 1 at path_bit_index
                let mid = curr_traverse_info.lower
                    + path_proofs[curr_traverse_info.lower..=curr_traverse_info.upper]
                        .binary_search_by(|path_proof| {
                            if !path_proof.terminal.key_path()[curr_traverse_info.path_bit_index] {
                                std::cmp::Ordering::Less
                            } else {
                                std::cmp::Ordering::Greater
                            }
                        })
                        .expect_err("There must be at least one bit set to 1");

                // insert all collected siblings (those are the shared ones)
                // in this sub tree in ext_sibling at the position this bisection is pointing at
                let collected_siblings = ext_sibling_in_sub_tree.len();
                for sibling in ext_sibling_in_sub_tree.drain(..).rev() {
                    external_siblings.insert(curr_traverse_info.ext_sibling_index, sibling);
                }

                // push into the stack the right Bisection and work on the left one
                stack_traverse_info.push(TraverseInfo {
                    path_bit_index: curr_traverse_info.path_bit_index + 1,
                    lower: mid,
                    upper: curr_traverse_info.upper,
                    ext_sibling_index: curr_traverse_info.ext_sibling_index + collected_siblings,
                });

                curr_traverse_info = TraverseInfo {
                    path_bit_index: curr_traverse_info.path_bit_index + 1,
                    lower: curr_traverse_info.lower,
                    upper: mid - 1,
                    ext_sibling_index: curr_traverse_info.ext_sibling_index + collected_siblings,
                };
            } else {
                // if they are the same then a sibling must be inserted in the external node list
                let sibling = path_proofs[curr_traverse_info.lower].siblings
                    [curr_traverse_info.path_bit_index];

                ext_sibling_in_sub_tree.push(sibling);

                curr_traverse_info.path_bit_index += 1;
            }
        }

        Self {
            sub_paths,
            external_siblings,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::MultiProof;

    use crate::proof::{PathProof, PathProofTerminal};
    use bitvec::{order::Msb0, view::BitView};

    #[test]
    pub fn test_multiproof_creation_single_path_proof() {
        let mut key_path = [0; 32];
        key_path[0] = 0b10000000;
        let sibling1 = [1; 32];
        let sibling2 = [2; 32];
        let path_proof = PathProof {
            terminal: PathProofTerminal::Terminator(key_path.view_bits::<Msb0>().into()),
            siblings: vec![sibling1, sibling2],
        };

        let multi_proof = MultiProof::from_path_proofs(vec![path_proof]);
        assert_eq!(multi_proof.sub_paths.len(), 1);
        assert_eq!(multi_proof.external_siblings.len(), 0);
        assert_eq!(
            multi_proof.sub_paths[0].terminal,
            PathProofTerminal::Terminator(key_path.view_bits::<Msb0>().into())
        );
        assert_eq!(
            multi_proof.sub_paths[0].inner_siblings,
            vec![sibling1, sibling2]
        );
        assert_eq!(multi_proof.sub_paths[0].depth, 0);
    }

    #[test]
    pub fn test_multiproof_creation_two_path_proofs() {
        let mut key_path_1 = [0; 32];
        key_path_1[0] = 0b00000000;

        let mut key_path_2 = [0; 32];
        key_path_2[0] = 0b00111000;

        let sibling1 = [1; 32];
        let sibling2 = [2; 32];
        let sibling3 = [3; 32];
        let sibling4 = [4; 32];
        let sibling5 = [5; 32];
        let sibling6 = [6; 32];

        let sibling_x = [b'x'; 32];

        let path_proof_1 = PathProof {
            terminal: PathProofTerminal::Terminator(key_path_1.view_bits::<Msb0>().into()),
            siblings: vec![sibling1, sibling2, sibling_x, sibling3, sibling4],
        };
        let path_proof_2 = PathProof {
            terminal: PathProofTerminal::Terminator(key_path_2.view_bits::<Msb0>().into()),
            siblings: vec![sibling1, sibling2, sibling_x, sibling5, sibling6],
        };

        let multi_proof = MultiProof::from_path_proofs(vec![path_proof_1, path_proof_2]);

        assert_eq!(multi_proof.sub_paths.len(), 2);
        assert_eq!(multi_proof.external_siblings.len(), 2);

        assert_eq!(
            multi_proof.sub_paths[0].terminal,
            PathProofTerminal::Terminator(key_path_1.view_bits::<Msb0>().into())
        );
        assert_eq!(
            multi_proof.sub_paths[1].terminal,
            PathProofTerminal::Terminator(key_path_2.view_bits::<Msb0>().into())
        );

        assert_eq!(
            multi_proof.sub_paths[0].inner_siblings,
            vec![sibling3, sibling4]
        );
        assert_eq!(
            multi_proof.sub_paths[1].inner_siblings,
            vec![sibling5, sibling6]
        );

        assert_eq!(multi_proof.sub_paths[0].depth, 3);
        assert_eq!(multi_proof.sub_paths[1].depth, 3);

        assert_eq!(multi_proof.external_siblings, vec![sibling1, sibling2]);
    }

    #[test]
    pub fn test_multiproof_creation_two_path_proofs_256_depth() {
        let mut key_path_1 = [0; 32];
        key_path_1[31] = 0b00000000;

        let mut key_path_2 = [0; 32];
        key_path_2[31] = 0b00000001;

        let mut siblings_1: Vec<[u8; 32]> = (0..255).map(|i| [i; 32]).collect();
        let mut siblings_2 = siblings_1.clone();
        siblings_1.push([b'2'; 32]);
        siblings_2.push([b'1'; 32]);

        let path_proof_1 = PathProof {
            terminal: PathProofTerminal::Terminator(key_path_1.view_bits::<Msb0>().into()),
            siblings: siblings_1.clone(),
        };
        let path_proof_2 = PathProof {
            terminal: PathProofTerminal::Terminator(key_path_2.view_bits::<Msb0>().into()),
            siblings: siblings_2,
        };

        let multi_proof = MultiProof::from_path_proofs(vec![path_proof_1, path_proof_2]);

        assert_eq!(multi_proof.sub_paths.len(), 2);
        assert_eq!(multi_proof.external_siblings.len(), 255);

        assert_eq!(
            multi_proof.sub_paths[0].terminal,
            PathProofTerminal::Terminator(key_path_1.view_bits::<Msb0>().into())
        );
        assert_eq!(
            multi_proof.sub_paths[1].terminal,
            PathProofTerminal::Terminator(key_path_2.view_bits::<Msb0>().into())
        );

        assert!(multi_proof.sub_paths[0].inner_siblings.is_empty());
        assert!(multi_proof.sub_paths[1].inner_siblings.is_empty());

        assert_eq!(multi_proof.sub_paths[0].depth, 256);
        assert_eq!(multi_proof.sub_paths[1].depth, 256);

        siblings_1.pop();
        assert_eq!(multi_proof.external_siblings, siblings_1);
    }

    #[test]
    pub fn test_multiproof_creation_multiple_path_proofs() {
        let mut key_path_1 = [0; 32];
        key_path_1[0] = 0b00000000;

        let mut key_path_2 = [0; 32];
        key_path_2[0] = 0b01000000;

        let mut key_path_3 = [0; 32];
        key_path_3[0] = 0b01001100;

        let mut key_path_4 = [0; 32];
        key_path_4[0] = 0b11101100;

        let mut key_path_5 = [0; 32];
        key_path_5[0] = 0b11110100;

        let mut key_path_6 = [0; 32];
        key_path_6[0] = 0b11111000;

        let sibling1 = [1; 32];
        let sibling2 = [2; 32];
        let sibling3 = [3; 32];
        let sibling4 = [4; 32];
        let sibling5 = [5; 32];
        let sibling6 = [6; 32];
        let sibling7 = [7; 32];
        let sibling8 = [8; 32];
        let sibling9 = [9; 32];
        let sibling10 = [10; 32];
        let sibling11 = [11; 32];
        let sibling12 = [12; 32];
        let sibling13 = [13; 32];
        let sibling14 = [14; 32];
        let sibling15 = [15; 32];
        let sibling16 = [16; 32];
        let sibling17 = [17; 32];
        let sibling18 = [18; 32];
        let sibling19 = [19; 32];

        let path_proof_1 = PathProof {
            terminal: PathProofTerminal::Terminator(key_path_1.view_bits::<Msb0>().into()),
            siblings: vec![sibling1, sibling2],
        };

        let path_proof_2 = PathProof {
            terminal: PathProofTerminal::Terminator(key_path_2.view_bits::<Msb0>().into()),
            siblings: vec![sibling1, sibling3, sibling4, sibling5, sibling6, sibling7],
        };

        let path_proof_3 = PathProof {
            terminal: PathProofTerminal::Terminator(key_path_3.view_bits::<Msb0>().into()),
            siblings: vec![sibling1, sibling3, sibling4, sibling5, sibling8, sibling9],
        };

        let path_proof_4 = PathProof {
            terminal: PathProofTerminal::Terminator(key_path_4.view_bits::<Msb0>().into()),
            siblings: vec![
                sibling10, sibling11, sibling12, sibling13, sibling14, sibling15,
            ],
        };

        let path_proof_5 = PathProof {
            terminal: PathProofTerminal::Terminator(key_path_5.view_bits::<Msb0>().into()),
            siblings: vec![
                sibling10, sibling11, sibling12, sibling16, sibling17, sibling18,
            ],
        };

        let path_proof_6 = PathProof {
            terminal: PathProofTerminal::Terminator(key_path_6.view_bits::<Msb0>().into()),
            siblings: vec![sibling10, sibling11, sibling12, sibling16, sibling19],
        };

        let multi_proof = MultiProof::from_path_proofs(vec![
            path_proof_1,
            path_proof_2,
            path_proof_3,
            path_proof_4,
            path_proof_5,
            path_proof_6,
        ]);

        assert_eq!(multi_proof.sub_paths.len(), 6);
        assert_eq!(multi_proof.external_siblings.len(), 4);

        assert_eq!(
            multi_proof.sub_paths[0].terminal,
            PathProofTerminal::Terminator(key_path_1.view_bits::<Msb0>().into())
        );
        assert_eq!(
            multi_proof.sub_paths[1].terminal,
            PathProofTerminal::Terminator(key_path_2.view_bits::<Msb0>().into())
        );
        assert_eq!(
            multi_proof.sub_paths[2].terminal,
            PathProofTerminal::Terminator(key_path_3.view_bits::<Msb0>().into())
        );
        assert_eq!(
            multi_proof.sub_paths[3].terminal,
            PathProofTerminal::Terminator(key_path_4.view_bits::<Msb0>().into())
        );
        assert_eq!(
            multi_proof.sub_paths[4].terminal,
            PathProofTerminal::Terminator(key_path_5.view_bits::<Msb0>().into())
        );
        assert_eq!(
            multi_proof.sub_paths[5].terminal,
            PathProofTerminal::Terminator(key_path_6.view_bits::<Msb0>().into())
        );

        assert_eq!(
            multi_proof.sub_paths[0].inner_siblings,
            Vec::<[u8; 32]>::new()
        );
        assert_eq!(multi_proof.sub_paths[1].inner_siblings, vec![sibling7]);
        assert_eq!(multi_proof.sub_paths[2].inner_siblings, vec![sibling9]);
        assert_eq!(
            multi_proof.sub_paths[3].inner_siblings,
            vec![sibling14, sibling15]
        );
        assert_eq!(multi_proof.sub_paths[4].inner_siblings, vec![sibling18]);
        assert_eq!(
            multi_proof.sub_paths[5].inner_siblings,
            Vec::<[u8; 32]>::new()
        );

        assert_eq!(multi_proof.sub_paths[0].depth, 2);
        assert_eq!(multi_proof.sub_paths[1].depth, 5);
        assert_eq!(multi_proof.sub_paths[2].depth, 5);
        assert_eq!(multi_proof.sub_paths[3].depth, 4);
        assert_eq!(multi_proof.sub_paths[4].depth, 5);
        assert_eq!(multi_proof.sub_paths[5].depth, 5);

        assert_eq!(
            multi_proof.external_siblings,
            vec![sibling11, sibling12, sibling4, sibling5]
        );
    }

    #[test]
    pub fn test_multiproof_creation_ext_siblings_order() {
        let mut key_path_0 = [0; 32];
        key_path_0[0] = 0b00001000;

        let mut key_path_1 = [0; 32];
        key_path_1[0] = 0b00010000;

        let mut key_path_2 = [0; 32];
        key_path_2[0] = 0b10000000;

        let mut key_path_3 = [0; 32];
        key_path_3[0] = 0b10000010;

        let mut key_path_4 = [0; 32];
        key_path_4[0] = 0b10010001;

        let mut key_path_5 = [0; 32];
        key_path_5[0] = 0b10010011;

        let sibling1 = [1; 32];
        let sibling2 = [2; 32];
        let sibling3 = [3; 32];
        let sibling4 = [4; 32];
        let sibling5 = [5; 32];
        let sibling6 = [6; 32];
        let sibling7 = [7; 32];
        let sibling8 = [8; 32];
        let sibling9 = [9; 32];
        let sibling10 = [10; 32];
        let sibling11 = [11; 32];
        let sibling12 = [12; 32];
        let sibling13 = [13; 32];
        let sibling14 = [14; 32];
        let sibling15 = [15; 32];
        let sibling16 = [16; 32];
        let sibling17 = [17; 32];
        let sibling18 = [18; 32];
        let sibling19 = [19; 32];
        let sibling20 = [20; 32];
        let sibling21 = [21; 32];
        let sibling22 = [22; 32];
        let sibling23 = [23; 32];
        let sibling24 = [24; 32];

        let path_proof_0 = PathProof {
            terminal: PathProofTerminal::Terminator(key_path_0.view_bits::<Msb0>().into()),
            siblings: vec![sibling1, sibling2, sibling3, sibling4, sibling5],
        };

        let path_proof_1 = PathProof {
            terminal: PathProofTerminal::Terminator(key_path_1.view_bits::<Msb0>().into()),
            siblings: vec![sibling1, sibling2, sibling3, sibling6, sibling7],
        };

        let path_proof_2 = PathProof {
            terminal: PathProofTerminal::Terminator(key_path_2.view_bits::<Msb0>().into()),
            siblings: vec![
                sibling8, sibling9, sibling10, sibling11, sibling12, sibling13, sibling14,
                sibling15,
            ],
        };
        let path_proof_3 = PathProof {
            terminal: PathProofTerminal::Terminator(key_path_3.view_bits::<Msb0>().into()),
            siblings: vec![
                sibling8, sibling9, sibling10, sibling11, sibling12, sibling13, sibling16,
                sibling17,
            ],
        };

        let path_proof_4 = PathProof {
            terminal: PathProofTerminal::Terminator(key_path_4.view_bits::<Msb0>().into()),
            siblings: vec![
                sibling8, sibling9, sibling10, sibling18, sibling19, sibling20, sibling21,
                sibling22,
            ],
        };

        let path_proof_5 = PathProof {
            terminal: PathProofTerminal::Terminator(key_path_5.view_bits::<Msb0>().into()),
            siblings: vec![
                sibling8, sibling9, sibling10, sibling18, sibling19, sibling20, sibling23,
                sibling24,
            ],
        };

        let multi_proof = MultiProof::from_path_proofs(vec![
            path_proof_0,
            path_proof_1,
            path_proof_2,
            path_proof_3,
            path_proof_4,
            path_proof_5,
        ]);

        assert_eq!(multi_proof.sub_paths.len(), 6);
        assert_eq!(multi_proof.external_siblings.len(), 8);

        assert_eq!(
            multi_proof.external_siblings,
            vec![
                sibling9, sibling10, sibling19, sibling20, sibling12, sibling13, sibling2, sibling3
            ]
        );
    }
}
