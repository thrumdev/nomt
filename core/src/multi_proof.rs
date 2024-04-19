//! Generate a multiproof from a vector of path proofs.
//! The multiproof will contain the minimum information needed to verify
//! the inclusion of all provided path proofs.

use crate::{
    proof::{PathProof, PathProofTerminal},
    trie::Node,
};
use bitvec::{order::Msb0, view::BitView};

// Each terminal node may have a set of siblings that are uniquely
// used during the verification with its terminal node.
// This struct contains the terminal node itself,
// the depth at which the siblings start to be uniquely associated to that terminal,
// and the siblings themselves.
#[derive(Debug, Clone)]
struct SubPathProof {
    terminal: PathProofTerminal,
    depth: u8,
    inner_siblings: Vec<Node>,
}

/// A proof of multiple path through the trie.
#[derive(Debug, Clone)]
pub struct MultiProof {
    // All subpaths related to a single terminal node
    sub_paths: Vec<SubPathProof>,
    // Vector containing the minimum number of nodes required
    // to reconstruct all other nodes later.
    external_siblings: Vec<Node>,
}

impl MultiProof {
    /// Construct a MultiProof from a vector of *ordered* PathProof
    pub fn from_path_proofs(path_proofs: Vec<PathProof>) -> Self {
        // A multi-proof can be viewed by associating each terminal node
        // with its first n uniquely related siblings from its path proof,
        // followed by all necessary siblings that are not derivable from
        // the previously mentioned siblings.
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
        // If two keys share the same bit at index i, they will have the same sibling at index i.
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
        // If all key_paths share a bit at an index, that sibling is required;
        // if at least two key_paths differ, no sibling is needed, but the key_paths must be divided
        // based on having bit 0 or 1 at that index.
        //
        // Iterate this algorithm on the bisection to determine the minimum necessary siblings.
        //
        // `external_siblings` whill follow this structure for each bicection:
        // |commons siblings| ext siblings in the right bisection | ext siblings in the left bisection |

        // it represent a slice of the vector of key_paths with the notion of
        // path_index, at which bit in the key_path that slice is pointing at
        struct TraverseInfo {
            path_index: usize,
            path_slice_lower: usize,
            path_slice_upper: usize,
            ext_sibling_index: usize,
        }
        let mut stack_traverse_info: Vec<TraverseInfo> = vec![];

        // External siblings encountered during a subtree traversal
        let mut ext_sibling_in_sub_tree: Vec<Node> = vec![];

        // initially we're looking at all the key_paths
        let mut curr_traverse_info = TraverseInfo {
            path_index: 0,
            path_slice_lower: 0,
            path_slice_upper: path_proofs.len() - 1,
            ext_sibling_index: 0,
        };

        loop {
            // Based on how we bisect the key_path slice, we can see that each TraverseInfo
            // will represent a slice where all key_paths will have same bits before
            // TraverseInfo.path_index.

            // If there is no longer a common prefix, there is only one key in the TraverseInfo,
            // and then all remaining siblings are required for the multiproof.
            if curr_traverse_info.path_slice_lower == curr_traverse_info.path_slice_upper {
                let path_proof = &path_proofs[curr_traverse_info.path_slice_lower];
                let sub_path_proof = SubPathProof {
                    terminal: path_proof.terminal.clone(),
                    // The depth at which the terminal starts not sharing any siblings
                    depth: curr_traverse_info.path_index as u8,
                    inner_siblings: path_proof
                        .siblings
                        .iter()
                        .skip(curr_traverse_info.path_index)
                        .copied()
                        .collect(),
                };
                sub_paths.push(sub_path_proof);

                // insert all collected siblings in this sub tree in ext_sibling
                // at the position this bisection is pointing at
                for sibling in ext_sibling_in_sub_tree.into_iter().rev() {
                    external_siblings.insert(curr_traverse_info.ext_sibling_index, sibling);
                }
                ext_sibling_in_sub_tree = vec![];

                // skip to the next bisection in the stak, if empty we're finished
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
            let path_lower = path_proofs[curr_traverse_info.path_slice_lower]
                .terminal
                .key_path()
                .view_bits::<Msb0>();
            let path_upper = path_proofs[curr_traverse_info.path_slice_upper]
                .terminal
                .key_path()
                .view_bits::<Msb0>();

            if path_lower[curr_traverse_info.path_index]
                != path_upper[curr_traverse_info.path_index]
            {
                // if they differ we can skip their siblings but we need to bisec the slice
                //
                // binary search between key_paths in the slice to see where to
                // perform the bisection

                // path_slice_upper and path_slice_lower will never be the same otherwise
                // bits at path_index would have been the same

                let mut low = curr_traverse_info.path_slice_lower;
                let mut up = curr_traverse_info.path_slice_upper;
                let mut mid;
                loop {
                    mid = (low + up) / 2;

                    let bit = path_proofs[mid].terminal.key_path().view_bits::<Msb0>()
                        [curr_traverse_info.path_index];

                    if !bit {
                        low = mid + 1;
                    } else {
                        // we need the first one, so mid - 1 needs to have a 0
                        // at position curr_traverse_info.path_index
                        let prev_bit = path_proofs[mid - 1].terminal.key_path().view_bits::<Msb0>()
                            [curr_traverse_info.path_index];
                        if !prev_bit {
                            break;
                        } else {
                            up = mid - 1;
                        }
                    }
                }

                // insert all collected siblings (those are the shared ones)
                // in this sub tree in ext_sibling at the position this bisection is pointing at
                let collected_siblings = ext_sibling_in_sub_tree.len();
                for sibling in ext_sibling_in_sub_tree.into_iter().rev() {
                    external_siblings.insert(curr_traverse_info.ext_sibling_index, sibling);
                }
                ext_sibling_in_sub_tree = vec![];

                // push into the stack the right Bisection and work on the left one
                stack_traverse_info.push(TraverseInfo {
                    path_index: curr_traverse_info.path_index + 1,
                    path_slice_lower: mid,
                    path_slice_upper: curr_traverse_info.path_slice_upper,
                    ext_sibling_index: curr_traverse_info.ext_sibling_index + collected_siblings,
                });

                curr_traverse_info = TraverseInfo {
                    path_index: curr_traverse_info.path_index + 1,
                    path_slice_lower: curr_traverse_info.path_slice_lower,
                    path_slice_upper: mid - 1,
                    ext_sibling_index: curr_traverse_info.ext_sibling_index + collected_siblings,
                };
            } else {
                // if they are the same then a sibling must be inserted in the external node list
                let sibling = path_proofs[curr_traverse_info.path_slice_lower].siblings
                    [curr_traverse_info.path_index];

                ext_sibling_in_sub_tree.push(sibling);

                curr_traverse_info.path_index += 1;
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

    #[test]
    pub fn test_multiproof_creation_single_path_proof() {
        let mut key_path = [0; 32];
        key_path[0] = 0b10000000;
        let sibling1 = [1; 32];
        let sibling2 = [2; 32];
        let path_proof = PathProof {
            terminal: PathProofTerminal::Terminator(key_path),
            siblings: vec![sibling1, sibling2],
        };

        let multi_proof = MultiProof::from_path_proofs(vec![path_proof]);
        assert_eq!(multi_proof.sub_paths.len(), 1);
        assert_eq!(multi_proof.external_siblings.len(), 0);
        assert_eq!(
            multi_proof.sub_paths[0].terminal,
            PathProofTerminal::Terminator(key_path)
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
            terminal: PathProofTerminal::Terminator(key_path_1),
            siblings: vec![sibling1, sibling2, sibling_x, sibling3, sibling4],
        };
        let path_proof_2 = PathProof {
            terminal: PathProofTerminal::Terminator(key_path_2),
            siblings: vec![sibling1, sibling2, sibling_x, sibling5, sibling6],
        };

        let multi_proof = MultiProof::from_path_proofs(vec![path_proof_1, path_proof_2]);

        assert_eq!(multi_proof.sub_paths.len(), 2);
        assert_eq!(multi_proof.external_siblings.len(), 2);

        assert_eq!(
            multi_proof.sub_paths[0].terminal,
            PathProofTerminal::Terminator(key_path_1)
        );
        assert_eq!(
            multi_proof.sub_paths[1].terminal,
            PathProofTerminal::Terminator(key_path_2)
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
            terminal: PathProofTerminal::Terminator(key_path_1),
            siblings: vec![sibling1, sibling2],
        };

        let path_proof_2 = PathProof {
            terminal: PathProofTerminal::Terminator(key_path_2),
            siblings: vec![sibling1, sibling3, sibling4, sibling5, sibling6, sibling7],
        };

        let path_proof_3 = PathProof {
            terminal: PathProofTerminal::Terminator(key_path_3),
            siblings: vec![sibling1, sibling3, sibling4, sibling5, sibling8, sibling9],
        };

        let path_proof_4 = PathProof {
            terminal: PathProofTerminal::Terminator(key_path_4),
            siblings: vec![
                sibling10, sibling11, sibling12, sibling13, sibling14, sibling15,
            ],
        };

        let path_proof_5 = PathProof {
            terminal: PathProofTerminal::Terminator(key_path_5),
            siblings: vec![
                sibling10, sibling11, sibling12, sibling16, sibling17, sibling18,
            ],
        };

        let path_proof_6 = PathProof {
            terminal: PathProofTerminal::Terminator(key_path_6),
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
            PathProofTerminal::Terminator(key_path_1)
        );
        assert_eq!(
            multi_proof.sub_paths[1].terminal,
            PathProofTerminal::Terminator(key_path_2)
        );
        assert_eq!(
            multi_proof.sub_paths[2].terminal,
            PathProofTerminal::Terminator(key_path_3)
        );
        assert_eq!(
            multi_proof.sub_paths[3].terminal,
            PathProofTerminal::Terminator(key_path_4)
        );
        assert_eq!(
            multi_proof.sub_paths[4].terminal,
            PathProofTerminal::Terminator(key_path_5)
        );
        assert_eq!(
            multi_proof.sub_paths[5].terminal,
            PathProofTerminal::Terminator(key_path_6)
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
            terminal: PathProofTerminal::Terminator(key_path_0),
            siblings: vec![sibling1, sibling2, sibling3, sibling4, sibling5],
        };

        let path_proof_1 = PathProof {
            terminal: PathProofTerminal::Terminator(key_path_1),
            siblings: vec![sibling1, sibling2, sibling3, sibling6, sibling7],
        };

        let path_proof_2 = PathProof {
            terminal: PathProofTerminal::Terminator(key_path_2),
            siblings: vec![
                sibling8, sibling9, sibling10, sibling11, sibling12, sibling13, sibling14,
                sibling15,
            ],
        };
        let path_proof_3 = PathProof {
            terminal: PathProofTerminal::Terminator(key_path_3),
            siblings: vec![
                sibling8, sibling9, sibling10, sibling11, sibling12, sibling13, sibling16,
                sibling17,
            ],
        };

        let path_proof_4 = PathProof {
            terminal: PathProofTerminal::Terminator(key_path_4),
            siblings: vec![
                sibling8, sibling9, sibling10, sibling18, sibling19, sibling20, sibling21,
                sibling22,
            ],
        };

        let path_proof_5 = PathProof {
            terminal: PathProofTerminal::Terminator(key_path_5),
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
