//! Generate a multiproof from a vector of path proofs.
//! The multiproof will contain the minimum information needed to verify
//! the inclusion of all provided path proofs.

use crate::{
    proof::{PathProof, PathProofTerminal},
    trie::Node,
};

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

/// This struct includes the terminal node and its depth
#[derive(Debug, Clone)]
pub struct MultiPathProof {
    /// Terminal node
    pub terminal: PathProofTerminal,
    /// Depth of the terminal node
    pub depth: usize,
}

/// A proof of multiple paths through the trie.
#[derive(Debug, Clone)]
pub struct MultiProof {
    /// List of all provable paths. These are sorted in ascending order by bit-path
    pub paths: Vec<MultiPathProof>,
    /// Vector containing the minimum number of nodes required
    /// to reconstruct all other nodes later.
    ///
    /// The format is a recursive bisection:
    /// [upper_siblings ++ left_siblings ++ right_siblings]
    ///
    /// upper_siblings could be:
    /// + siblings shared among all paths in each bisection
    /// + unique siblings associated with a terminal node
    ///
    /// In the latter case, both left and right bisections will be empty
    ///
    /// left_siblings is the same format, but applied to all the nodes in the left bisection.
    /// right_siblings is the same format, but applied to all the nodes in the right bisection.
    pub siblings: Vec<Node>,
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
    fn prove_unique_path_remainder(
        &self,
        path_proofs: &[PathProof],
    ) -> Option<(MultiPathProof, Vec<Node>)> {
        // If PathProofRange contains only one path_proofs
        // return a MultiPathProof with all its unique siblings
        if self.lower != self.upper - 1 {
            return None;
        }

        let path_proof = &path_proofs[self.lower];
        let unique_siblings: Vec<Node> = path_proof
            .siblings
            .iter()
            .skip(self.path_bit_index)
            .copied()
            .collect();

        Some((
            MultiPathProof {
                terminal: path_proof.terminal.clone(),
                depth: self.path_bit_index + unique_siblings.len(),
            },
            unique_siblings,
        ))
    }

    fn step(&mut self, path_proofs: &[PathProof]) -> PathProofRangeStep {
        // check if at least two key_path in the path_proofs range
        // has two different bits in position path_bit_index
        //
        // they are ordered and all the bits up to path_bit_index
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
            // Since there is at least one 1 bit and Ordering::Equal is never returned,
            // the method binary_search_by will always return an Error containing the index
            // of the first occurrence of one in the key_path.
            let mid = self.lower
                + path_proofs[self.lower..self.upper]
                    .binary_search_by(|path_proof| {
                        if !path_proof.terminal.path()[self.path_bit_index] {
                            core::cmp::Ordering::Less
                        } else {
                            core::cmp::Ordering::Greater
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
            // if they don't differ, we need their sibling
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
        // the previously mentioned siblings.

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
        // of the siblings mentioned earlier.
        // If at least two key_paths differ, no sibling is needed, but the key_paths must be divided
        // based on having bit 0 or 1 at that index.
        //
        // Iterate this algorithm on the bisection to determine the minimum necessary siblings.
        //
        // `siblings` will follow this structure for each bisection
        // |common siblings| ext siblings in the left bisection | ext siblings in the right bisection |

        let mut paths: Vec<MultiPathProof> = vec![];
        let mut siblings: Vec<Node> = vec![];

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
            if let Some((sub_path_proof, unique_siblings)) =
                proof_range.prove_unique_path_remainder(&path_proofs)
            {
                paths.push(sub_path_proof);
                siblings.extend(unique_siblings);

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
            // sibling of the current sub tree
            match proof_range.step(&path_proofs) {
                PathProofRangeStep::Bisect { left, right } => {
                    // insert collected common siblings
                    siblings.extend(common_siblings.drain(..));

                    // push into the stack the right Bisection and work on the left one
                    proof_range = left;
                    stack.push(right);
                }
                PathProofRangeStep::Advance { sibling } => common_siblings.push(sibling),
            };
        }

        Self { paths, siblings }
    }
}

#[cfg(test)]
mod tests {
    use super::MultiProof;

    use crate::{
        proof::{PathProof, PathProofTerminal},
        trie_pos::TriePosition,
    };

    #[test]
    pub fn test_multiproof_creation_single_path_proof() {
        let mut key_path = [0; 32];
        key_path[0] = 0b10000000;
        let sibling1 = [1; 32];
        let sibling2 = [2; 32];
        let path_proof = PathProof {
            terminal: PathProofTerminal::Terminator(TriePosition::from_path_and_depth(
                key_path, 256,
            )),
            siblings: vec![sibling1, sibling2],
        };

        let multi_proof = MultiProof::from_path_proofs(vec![path_proof]);
        assert_eq!(multi_proof.paths.len(), 1);
        assert_eq!(
            multi_proof.paths[0].terminal,
            PathProofTerminal::Terminator(TriePosition::from_path_and_depth(key_path, 256))
        );
        assert_eq!(multi_proof.paths[0].depth, 2);
        assert_eq!(multi_proof.siblings.len(), 2);
        assert_eq!(multi_proof.siblings, vec![sibling1, sibling2]);
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
            terminal: PathProofTerminal::Terminator(TriePosition::from_path_and_depth(
                key_path_1, 256,
            )),
            siblings: vec![sibling1, sibling2, sibling_x, sibling3, sibling4],
        };
        let path_proof_2 = PathProof {
            terminal: PathProofTerminal::Terminator(TriePosition::from_path_and_depth(
                key_path_2, 256,
            )),
            siblings: vec![sibling1, sibling2, sibling_x, sibling5, sibling6],
        };

        let multi_proof = MultiProof::from_path_proofs(vec![path_proof_1, path_proof_2]);

        assert_eq!(multi_proof.paths.len(), 2);
        assert_eq!(multi_proof.siblings.len(), 6);

        assert_eq!(
            multi_proof.paths[0].terminal,
            PathProofTerminal::Terminator(TriePosition::from_path_and_depth(key_path_1, 256))
        );
        assert_eq!(
            multi_proof.paths[1].terminal,
            PathProofTerminal::Terminator(TriePosition::from_path_and_depth(key_path_2, 256))
        );

        assert_eq!(multi_proof.paths[0].depth, 5);
        assert_eq!(multi_proof.paths[1].depth, 5);

        assert_eq!(
            multi_proof.siblings,
            vec![sibling1, sibling2, sibling3, sibling4, sibling5, sibling6]
        );
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
            terminal: PathProofTerminal::Terminator(TriePosition::from_path_and_depth(
                key_path_1, 256,
            )),
            siblings: siblings_1.clone(),
        };
        let path_proof_2 = PathProof {
            terminal: PathProofTerminal::Terminator(TriePosition::from_path_and_depth(
                key_path_2, 256,
            )),
            siblings: siblings_2,
        };

        let multi_proof = MultiProof::from_path_proofs(vec![path_proof_1, path_proof_2]);

        assert_eq!(multi_proof.paths.len(), 2);
        assert_eq!(multi_proof.siblings.len(), 255);

        assert_eq!(
            multi_proof.paths[0].terminal,
            PathProofTerminal::Terminator(TriePosition::from_path_and_depth(key_path_1, 256))
        );
        assert_eq!(
            multi_proof.paths[1].terminal,
            PathProofTerminal::Terminator(TriePosition::from_path_and_depth(key_path_2, 256))
        );

        assert_eq!(multi_proof.paths[0].depth, 256);
        assert_eq!(multi_proof.paths[1].depth, 256);

        siblings_1.pop();
        assert_eq!(multi_proof.siblings, siblings_1);
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
            terminal: PathProofTerminal::Terminator(TriePosition::from_path_and_depth(
                key_path_1, 256,
            )),
            siblings: vec![sibling1, sibling2],
        };

        let path_proof_2 = PathProof {
            terminal: PathProofTerminal::Terminator(TriePosition::from_path_and_depth(
                key_path_2, 256,
            )),
            siblings: vec![sibling1, sibling3, sibling4, sibling5, sibling6, sibling7],
        };

        let path_proof_3 = PathProof {
            terminal: PathProofTerminal::Terminator(TriePosition::from_path_and_depth(
                key_path_3, 256,
            )),
            siblings: vec![sibling1, sibling3, sibling4, sibling5, sibling8, sibling9],
        };

        let path_proof_4 = PathProof {
            terminal: PathProofTerminal::Terminator(TriePosition::from_path_and_depth(
                key_path_4, 256,
            )),
            siblings: vec![
                sibling10, sibling11, sibling12, sibling13, sibling14, sibling15,
            ],
        };

        let path_proof_5 = PathProof {
            terminal: PathProofTerminal::Terminator(TriePosition::from_path_and_depth(
                key_path_5, 256,
            )),
            siblings: vec![
                sibling10, sibling11, sibling12, sibling16, sibling17, sibling18,
            ],
        };

        let path_proof_6 = PathProof {
            terminal: PathProofTerminal::Terminator(TriePosition::from_path_and_depth(
                key_path_6, 256,
            )),
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

        assert_eq!(multi_proof.paths.len(), 6);
        assert_eq!(multi_proof.siblings.len(), 9);

        assert_eq!(
            multi_proof.paths[0].terminal,
            PathProofTerminal::Terminator(TriePosition::from_path_and_depth(key_path_1, 256))
        );
        assert_eq!(
            multi_proof.paths[1].terminal,
            PathProofTerminal::Terminator(TriePosition::from_path_and_depth(key_path_2, 256))
        );
        assert_eq!(
            multi_proof.paths[2].terminal,
            PathProofTerminal::Terminator(TriePosition::from_path_and_depth(key_path_3, 256))
        );
        assert_eq!(
            multi_proof.paths[3].terminal,
            PathProofTerminal::Terminator(TriePosition::from_path_and_depth(key_path_4, 256))
        );
        assert_eq!(
            multi_proof.paths[4].terminal,
            PathProofTerminal::Terminator(TriePosition::from_path_and_depth(key_path_5, 256))
        );
        assert_eq!(
            multi_proof.paths[5].terminal,
            PathProofTerminal::Terminator(TriePosition::from_path_and_depth(key_path_6, 256))
        );

        assert_eq!(multi_proof.paths[0].depth, 2);
        assert_eq!(multi_proof.paths[1].depth, 6);
        assert_eq!(multi_proof.paths[2].depth, 6);
        assert_eq!(multi_proof.paths[3].depth, 6);
        assert_eq!(multi_proof.paths[4].depth, 6);
        assert_eq!(multi_proof.paths[5].depth, 5);

        assert_eq!(
            multi_proof.siblings,
            vec![
                sibling4, sibling5, sibling7, sibling9, sibling11, sibling12, sibling14, sibling15,
                sibling18
            ]
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
            terminal: PathProofTerminal::Terminator(TriePosition::from_path_and_depth(
                key_path_0, 256,
            )),
            siblings: vec![sibling1, sibling2, sibling3, sibling4, sibling5],
        };

        let path_proof_1 = PathProof {
            terminal: PathProofTerminal::Terminator(TriePosition::from_path_and_depth(
                key_path_1, 256,
            )),
            siblings: vec![sibling1, sibling2, sibling3, sibling6, sibling7],
        };

        let path_proof_2 = PathProof {
            terminal: PathProofTerminal::Terminator(TriePosition::from_path_and_depth(
                key_path_2, 256,
            )),
            siblings: vec![
                sibling8, sibling9, sibling10, sibling11, sibling12, sibling13, sibling14,
                sibling15,
            ],
        };
        let path_proof_3 = PathProof {
            terminal: PathProofTerminal::Terminator(TriePosition::from_path_and_depth(
                key_path_3, 256,
            )),
            siblings: vec![
                sibling8, sibling9, sibling10, sibling11, sibling12, sibling13, sibling16,
                sibling17,
            ],
        };

        let path_proof_4 = PathProof {
            terminal: PathProofTerminal::Terminator(TriePosition::from_path_and_depth(
                key_path_4, 256,
            )),
            siblings: vec![
                sibling8, sibling9, sibling10, sibling18, sibling19, sibling20, sibling21,
                sibling22,
            ],
        };

        let path_proof_5 = PathProof {
            terminal: PathProofTerminal::Terminator(TriePosition::from_path_and_depth(
                key_path_5, 256,
            )),
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

        assert_eq!(multi_proof.paths.len(), 6);
        assert_eq!(multi_proof.siblings.len(), 14);

        assert_eq!(
            multi_proof.siblings,
            vec![
                sibling2, sibling3, sibling5, sibling7, sibling9, sibling10, sibling12, sibling13,
                sibling15, sibling17, sibling19, sibling20, sibling22, sibling24
            ]
        );
    }
}
