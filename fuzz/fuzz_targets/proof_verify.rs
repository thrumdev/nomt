#![no_main]

use std::collections::{BTreeMap, BTreeSet};

use arbitrary::{Arbitrary, Unstructured};
use bitvec::{order::Msb0, view::BitView};
use libfuzzer_sys::fuzz_target;
use nomt_core::{
    hasher::{Blake3Hasher, NodeHasher},
    proof::{
        verify_multi_proof, verify_multi_proof_update, verify_update, KeyOutOfScope, MultiProof,
        PathProof, PathProofTerminal, PathUpdate,
    },
    trie::{InternalData, KeyPath, LeafData, Node, ValueHash, TERMINATOR},
    trie_pos::TriePosition,
    update::build_trie,
};

type H = Blake3Hasher;

fuzz_target!(|run: Run| {
    let Run {
        leaves,
        queries,
        ops,
        mutations,
    } = run;

    let sorted_leaves = leaves.iter().map(|(k, v)| (*k, *v)).collect::<Vec<_>>();
    let root = node(&sorted_leaves, 0);
    assert_eq!(
        root,
        build_trie::<H>(0, sorted_leaves.iter().copied(), |_| {})
    );

    let mut updated = leaves.clone();
    for (key, value) in &ops {
        match value {
            Some(value) => updated.insert(*key, *value),
            None => updated.remove(key),
        };
    }
    let updated = updated.into_iter().collect::<Vec<_>>();
    let expected_root = node(&updated, 0);

    // Honest path proofs must verify, make true claims, and produce the right updated root.
    let mut path_proofs = Vec::new();
    let mut path_updates: Vec<PathUpdate> = Vec::new();
    for key in &queries {
        let proof = path_proof(&sorted_leaves, key);
        let verified = proof.verify::<H>(key.view_bits(), root).unwrap();
        check_claims(
            &leaves,
            *key,
            true,
            |k| verified.confirm_nonexistence(k),
            |l| verified.confirm_value(l),
        );
        if let Some(value) = ops.get(key) {
            match path_updates.last_mut() {
                Some(update) if update.inner.path() == verified.path() => {
                    update.ops.push((*key, *value))
                }
                _ => path_updates.push(PathUpdate {
                    inner: verified,
                    ops: vec![(*key, *value)],
                }),
            }
        }
        path_proofs.push(proof);
    }
    assert_eq!(
        verify_update::<H>(root, &path_updates).unwrap(),
        expected_root
    );

    // Tampered path proofs: if one still verifies, everything it claims must be true.
    for (key, proof) in queries.iter().zip(&path_proofs) {
        let mut proof = proof.clone();
        for mutation in &mutations {
            mutation.apply_to_path_proof(&mut proof);
        }
        if let Ok(verified) = proof.verify::<H>(key.view_bits(), root) {
            check_claims(
                &leaves,
                *key,
                false,
                |k| verified.confirm_nonexistence(k),
                |l| verified.confirm_value(l),
            );
        }
    }

    if queries.is_empty() {
        return;
    }

    path_proofs.sort_by(|a, b| a.terminal.path().cmp(b.terminal.path()));
    path_proofs.dedup_by(|a, b| a.terminal.path() == b.terminal.path());
    let multi_proof = MultiProof::from_path_proofs(path_proofs);
    let ops = ops.into_iter().collect::<Vec<_>>();

    // An honest multi-proof must verify, make true claims, and produce the right updated root.
    let verified = verify_multi_proof::<H>(&multi_proof, root).unwrap();
    for key in &queries {
        check_claims(
            &leaves,
            *key,
            true,
            |k| verified.confirm_nonexistence(k),
            |l| verified.confirm_value(l),
        );
    }
    assert_eq!(
        verify_multi_proof_update::<H>(&verified, ops.clone()).unwrap(),
        expected_root
    );

    // Tampered multi-proof: must not panic, and if it still verifies, it must not lie.
    let mut multi_proof = multi_proof;
    for mutation in &mutations {
        mutation.apply_to_multi_proof(&mut multi_proof);
    }
    if let Ok(verified) = verify_multi_proof::<H>(&multi_proof, root) {
        for key in &queries {
            check_claims(
                &leaves,
                *key,
                false,
                |k| verified.confirm_nonexistence(k),
                |l| verified.confirm_value(l),
            );
        }
        if let Ok(new_root) = verify_multi_proof_update::<H>(&verified, ops) {
            assert_eq!(new_root, expected_root);
        }
    }
});

/// Asserts that what a verified proof claims about `key` matches `model`.
/// `in_scope` is set when the proof was built for `key`, so it must not report it out of scope.
fn check_claims(
    model: &BTreeMap<KeyPath, ValueHash>,
    key: KeyPath,
    in_scope: bool,
    confirm_nonexistence: impl Fn(&KeyPath) -> Result<bool, KeyOutOfScope>,
    confirm_value: impl Fn(&LeafData) -> Result<bool, KeyOutOfScope>,
) {
    let value_hash = model.get(&key).copied();
    match confirm_nonexistence(&key) {
        Ok(absent) => assert_eq!(absent, value_hash.is_none()),
        Err(KeyOutOfScope) => assert!(!in_scope),
    }
    if let Some(value_hash) = value_hash {
        match confirm_value(&LeafData {
            key_path: key,
            value_hash,
        }) {
            Ok(found) => assert!(found),
            Err(KeyOutOfScope) => assert!(!in_scope),
        }
    }
    let mut wrong = value_hash.unwrap_or_default();
    wrong[0] ^= 1;
    assert!(!matches!(
        confirm_value(&LeafData {
            key_path: key,
            value_hash: wrong,
        }),
        Ok(true)
    ));
}

fn bit(key: &KeyPath, depth: usize) -> bool {
    key.view_bits::<Msb0>()[depth]
}

/// Reference hash of the sub-trie holding `leaves`: sorted, distinct, sharing the first `depth` bits.
fn node(leaves: &[(KeyPath, ValueHash)], depth: usize) -> Node {
    match leaves {
        [] => TERMINATOR,
        [(key_path, value_hash)] => H::hash_leaf(&LeafData {
            key_path: *key_path,
            value_hash: *value_hash,
        }),
        _ => {
            let mid = leaves.partition_point(|(k, _)| !bit(k, depth));
            H::hash_internal(&InternalData {
                left: node(&leaves[..mid], depth + 1),
                right: node(&leaves[mid..], depth + 1),
            })
        }
    }
}

fn path_proof(mut leaves: &[(KeyPath, ValueHash)], key: &KeyPath) -> PathProof {
    let mut siblings = Vec::new();
    let mut depth = 0;
    while leaves.len() > 1 {
        let mid = leaves.partition_point(|(k, _)| !bit(k, depth));
        let (here, other) = if bit(key, depth) {
            (&leaves[mid..], &leaves[..mid])
        } else {
            (&leaves[..mid], &leaves[mid..])
        };
        siblings.push(node(other, depth + 1));
        leaves = here;
        depth += 1;
    }
    let terminal = match leaves {
        [(key_path, value_hash)] => PathProofTerminal::Leaf(LeafData {
            key_path: *key_path,
            value_hash: *value_hash,
        }),
        _ => PathProofTerminal::Terminator(position(*key, depth as u16)),
    };
    PathProof { terminal, siblings }
}

fn position(path: KeyPath, depth: u16) -> TriePosition {
    match depth % 257 {
        0 => TriePosition::new(),
        depth => TriePosition::from_path_and_depth(path, depth),
    }
}

#[derive(Debug)]
struct Run {
    leaves: BTreeMap<KeyPath, ValueHash>,
    queries: BTreeSet<KeyPath>,
    ops: BTreeMap<KeyPath, Option<ValueHash>>,
    mutations: Vec<Mutation>,
}

impl<'a> Arbitrary<'a> for Run {
    fn arbitrary(input: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let mut keys = Vec::new();
        let mut leaves = BTreeMap::new();
        for _ in 0..input.int_in_range(0..=64)? {
            let key = new_key(input, &keys)?;
            keys.push(key);
            leaves.insert(key, input.arbitrary()?);
        }

        let mut queries = BTreeSet::new();
        let mut ops = BTreeMap::new();
        for _ in 0..input.int_in_range(0..=16)? {
            let key = if !keys.is_empty() && input.ratio(1, 2)? {
                keys[input.choose_index(keys.len())?]
            } else {
                new_key(input, &keys)?
            };
            queries.insert(key);
            if input.ratio(1, 2)? {
                ops.insert(key, input.arbitrary()?);
            }
        }

        let mut mutations = Vec::new();
        for _ in 0..input.int_in_range(0..=4)? {
            mutations.push(input.arbitrary()?);
        }

        Ok(Self {
            leaves,
            queries,
            ops,
            mutations,
        })
    }
}

/// Mostly derive keys from existing ones, diverging at a chosen bit, so the trie gets deep.
fn new_key(input: &mut Unstructured<'_>, existing: &[KeyPath]) -> arbitrary::Result<KeyPath> {
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

#[derive(Debug, Arbitrary)]
enum Mutation {
    FlipSiblingBit { index: usize, bit: u8 },
    RemoveSibling { index: usize },
    InsertSibling { index: usize, node: Node },
    SetTerminal { index: usize, terminal: Terminal },
    SetDepth { index: usize, depth: u16 },
    RemovePath { index: usize },
    DuplicatePath { index: usize },
    SwapPaths { a: usize, b: usize },
}

#[derive(Debug, Arbitrary)]
enum Terminal {
    Leaf {
        key_path: KeyPath,
        value_hash: ValueHash,
    },
    Terminator {
        path: KeyPath,
        depth: u16,
    },
}

impl Terminal {
    fn build(&self) -> PathProofTerminal {
        match *self {
            Terminal::Leaf {
                key_path,
                value_hash,
            } => PathProofTerminal::Leaf(LeafData {
                key_path,
                value_hash,
            }),
            Terminal::Terminator { path, depth } => {
                PathProofTerminal::Terminator(position(path, depth))
            }
        }
    }
}

impl Mutation {
    fn apply_to_siblings(&self, siblings: &mut Vec<Node>) {
        match *self {
            Mutation::FlipSiblingBit { index, bit } if !siblings.is_empty() => {
                let len = siblings.len();
                siblings[index % len][bit as usize / 8] ^= 1 << (bit % 8);
            }
            Mutation::RemoveSibling { index } if !siblings.is_empty() => {
                siblings.remove(index % siblings.len());
            }
            Mutation::InsertSibling { index, node } => {
                siblings.insert(index % (siblings.len() + 1), node);
            }
            _ => {}
        }
    }

    fn apply_to_path_proof(&self, proof: &mut PathProof) {
        self.apply_to_siblings(&mut proof.siblings);
        if let Mutation::SetTerminal { terminal, .. } = self {
            proof.terminal = terminal.build();
        }
    }

    fn apply_to_multi_proof(&self, proof: &mut MultiProof) {
        self.apply_to_siblings(&mut proof.siblings);
        let paths = &mut proof.paths;
        if paths.is_empty() {
            return;
        }
        let len = paths.len();
        match *self {
            Mutation::SetTerminal {
                index,
                ref terminal,
            } => paths[index % len].terminal = terminal.build(),
            Mutation::SetDepth { index, depth } => paths[index % len].depth = depth as usize,
            Mutation::RemovePath { index } => {
                paths.remove(index % len);
            }
            Mutation::DuplicatePath { index } => {
                let path = paths[index % len].clone();
                paths.insert(index % len, path);
            }
            Mutation::SwapPaths { a, b } => paths.swap(a % len, b % len),
            _ => {}
        }
    }
}
