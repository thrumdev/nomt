//! Generate a multiproof from a vector of path proofs.
//! The multiproof will contain the minimum information needed to verify
//! the inclusion of all provided path proofs.

use crate::{
    proof::{PathProof, PathProofTerminal},
    trie::Node,
};

// Each terminal node may have a set of siblings that are uniquely
// used during the verification with its terminal node.
// This struct contains the terminal node itself,
// the depth at which the siblings start to be uniquely associated to that terminal,
// and the siblings themselves.
#[derive(Debug, Clone)]
struct SubPathProof {
    terminal: PathProofTerminal,
    depth: usize,
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

// Given a vector of PathProofs ordered by the key_path,
// `PathProofRange` represent a range of that vector
// with key_paths that have common bits up to path_bit_index
struct PathProofRange {
    // lower bound of the range
    lower: usize,
    // upper bound of the range
    upper: usize,
    // bit index in the key path where this struct is pointing at,
    // this index will be increased or used to create a new bisection
    path_bit_index: usize,
}

enum PathProofRangeStep {
    Bisect {
        left: PathProofRange,
        right: PathProofRange,
    },
    Advance {
        sibling: Node,
    },
}

impl PathProofRange {
    fn prove_unique_path_remainder(&self, path_proofs: &Vec<PathProof>) -> Option<SubPathProof> {
        // If there is no longer a common prefix, there is only one key in the PathProofRange,
        // and then all remaining siblings are required for the multiproof
        if self.lower == self.upper - 1 {
            let path_proof = &path_proofs[self.lower];
            Some(SubPathProof {
                terminal: path_proof.terminal.clone(),
                // The depth at which the terminal starts not sharing any siblings
                depth: self.path_bit_index,
                inner_siblings: path_proof
                    .siblings
                    .iter()
                    .skip(self.path_bit_index)
                    .copied()
                    .collect(),
            })
        } else {
            None
        }
    }

    fn step(&mut self, path_proofs: &Vec<PathProof>) -> PathProofRangeStep {
        // check if at least two key_path in the path_proofs range
        // has two different bits in position path_index
        //
        // they are ordered and all the bits up to path_index
        // are shared so we can just check if the first and the last one differs
        let path_lower = path_proofs[self.lower].terminal.path();
        let path_upper = path_proofs[self.upper - 1].terminal.path();

        if path_lower[self.path_bit_index] != path_upper[self.path_bit_index] {
            // if they differ we can skip their siblings but we need to bisect the slice
            //
            // binary search between key_paths in the slice to see where to
            // perform the bisection
            //
            // UNWRAP: We have just checked that path_lower and path_upper differ at path_bit_index.
            // Therefore, with the vector of path_proofs ordered, we can be sure that there is at least
            // one path with its key_path containing a value of 1 at path_bit_index.
            // Since there is at least one 1 bit and std::cmp::Ordering::Equal is never returned,
            // the method binary_search_by will always return an Error containing the index
            // of the first occurrence of one in the key_path.
            let mid = self.lower
                + path_proofs[self.lower..self.upper]
                    .binary_search_by(|path_proof| {
                        if !path_proof.terminal.path()[self.path_bit_index] {
                            std::cmp::Ordering::Less
                        } else {
                            std::cmp::Ordering::Greater
                        }
                    })
                    .unwrap_err();

            let left = PathProofRange {
                path_bit_index: self.path_bit_index + 1,
                lower: self.lower,
                upper: mid,
            };

            let right = PathProofRange {
                path_bit_index: self.path_bit_index + 1,
                lower: mid,
                upper: self.upper,
            };

            PathProofRangeStep::Bisect { left, right }
        } else {
            let sibling = path_proofs[self.lower].siblings[self.path_bit_index];
            self.path_bit_index += 1;
            PathProofRangeStep::Advance { sibling }
        }
    }
}

impl MultiProof {
    /// Construct a MultiProof from a vector of *ordered* PathProof
    pub fn from_path_proofs(path_proofs: Vec<PathProof>) -> Self {
        // A multi-proof can be viewed by associating each terminal node
        // with its first n uniquely related siblings from its path proof,
        // followed by all necessary siblings that are not derivable from
        // the previously mentioned siblings. Those siblings will be called
        // `external_siblings`.

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
        // |common siblings| ext siblings in the left bisection | ext siblings in the right bisection |

        let mut sub_paths: Vec<SubPathProof> = vec![];
        let mut external_siblings: Vec<Node> = vec![];

        // initially we're looking at all the path_proofs
        let mut proof_range = PathProofRange {
            path_bit_index: 0,
            lower: 0,
            upper: path_proofs.len(),
        };

        // Common siblings encountered while stepping through PathProofRange
        let mut common_siblings: Vec<Node> = vec![];

        // stack used to handle bfs through PathProofRanges
        let mut stack: Vec<PathProofRange> = vec![];

        loop {
            // check if proof_range represents a unique path proof
            if let Some(sub_path_proof) = proof_range.prove_unique_path_remainder(&path_proofs) {
                sub_paths.push(sub_path_proof);

                // sub_path_proof always immediately follows a bisection in a well-formed trie
                assert!(common_siblings.is_empty());

                // skip to the next bisection in the stack, if empty we're finished
                proof_range = match stack.pop() {
                    Some(v) => v,
                    None => break,
                };
                continue;
            }

            // Step through the proof_range, it could result in a bisection,
            // or the index of the key_path is moved forward producing a new
            // external sibling of the current sub tree
            match proof_range.step(&path_proofs) {
                PathProofRangeStep::Bisect { left, right } => {
                    // insert collected common siblings
                    external_siblings.extend(common_siblings.drain(..));

                    // push into the stack the right Bisection and work on the left one
                    proof_range = left;
                    stack.push(right);
                }
                PathProofRangeStep::Advance { sibling } => common_siblings.push(sibling),
            };
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
            vec![sibling4, sibling5, sibling11, sibling12]
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
                sibling2, sibling3, sibling9, sibling10, sibling12, sibling13, sibling19, sibling20
            ]
        );
    }
}
