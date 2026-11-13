//! Untrusted proof bytes: decode a `PathProof`, `MultiProof` or `Witness` with borsh or bincode
//! (serde), then verify it the way a stateless verifier does.
//!
//! Inputs are either raw fuzzer bytes or the encoding of an honest proof with byte-level edits.
//! Witness and multi-proof operations can also be edited before encoding, so the update verifiers
//! see unsorted, duplicated and out-of-scope ops.
//!
//! Oracles:
//! - decoding never panics, and a successful decode re-encodes to exactly the same bytes;
//! - an unedited honest proof decodes, verifies and yields the model's updated root;
//! - whatever a decoded proof verifies to, every claim it confirms and every updated root it
//!   returns matches an independent reference trie.

#![no_main]

use std::collections::{BTreeMap, BTreeSet};

use arbitrary::{Arbitrary, Unstructured};
use bincode::Options as _;
use bitvec::{order::Msb0, view::BitView};
use borsh::{BorshDeserialize, BorshSerialize};
use libfuzzer_sys::fuzz_target;
use nomt_core::{
    hasher::{Blake3Hasher, NodeHasher},
    proof::{
        verify_multi_proof, verify_multi_proof_update, verify_update, KeyOutOfScope, MultiProof,
        PathProof, PathProofTerminal, PathUpdate,
    },
    trie::{InternalData, KeyPath, LeafData, Node, ValueHash, TERMINATOR},
    trie_pos::TriePosition,
    witness::{Witness, WitnessedOperations, WitnessedPath, WitnessedRead, WitnessedWrite},
};
use serde::{de::DeserializeOwned, Serialize};

type H = Blake3Hasher;
type Model = BTreeMap<KeyPath, ValueHash>;

fuzz_target!(|run: Run| {
    let root = root_of(&run.leaves);
    match run.kind {
        Kind::PathProof => path_proof_case(&run, root),
        Kind::MultiProof => multi_proof_case(&run, root),
        Kind::Witness => witness_case(&run, root),
    }
});

/// A single `PathProof`, checked against one of the queried keys.
fn path_proof_case(run: &Run, root: Node) {
    let key = match run.queries.len() {
        0 => [0; 32],
        len => run.queries[run.query_index as usize % len],
    };
    let honest = path_proof(&run.leaves, &key);
    let Some(proof) = run.transmit(&honest) else {
        return;
    };

    match proof.verify::<H>(key.view_bits(), root) {
        Ok(verified) => {
            for query in run.queries.iter().chain([&key]) {
                // The proven path is a prefix of `key`, so `key` is always in scope.
                check_claims(
                    &run.leaves,
                    *query,
                    *query == key,
                    |k| verified.confirm_nonexistence(k),
                    |l| verified.confirm_value(l),
                );
            }
        }
        Err(e) => assert!(!run.pristine(), "honest path proof rejected: {e:?}"),
    }
}

/// A `MultiProof` over all queried keys, then `verify_multi_proof_update` with the (edited) ops.
fn multi_proof_case(run: &Run, root: Node) {
    let mut path_proofs = run
        .queries
        .iter()
        .map(|key| path_proof(&run.leaves, key))
        .collect::<Vec<_>>();
    path_proofs.dedup_by(|a, b| a.terminal.path() == b.terminal.path());
    let honest = MultiProof::from_path_proofs(path_proofs);
    let Some(multi_proof) = run.transmit(&honest) else {
        return;
    };

    let mut ops = run.ops.clone();
    for edit in &run.struct_edits {
        if let StructEdit::Write(edit) = edit {
            edit.apply(&mut ops);
        }
    }

    // An empty multi-proof proves nothing, so only demand success when something was queried.
    let must_verify = run.pristine() && !run.queries.is_empty();
    let verified = match verify_multi_proof::<H>(&multi_proof, root) {
        Ok(verified) => verified,
        Err(e) => {
            assert!(!must_verify, "honest multi-proof rejected: {e:?}");
            return;
        }
    };
    for key in run.queries.iter().chain(ops.iter().map(|op| &op.key)) {
        check_claims(
            &run.leaves,
            *key,
            must_verify,
            |k| verified.confirm_nonexistence(k),
            |l| verified.confirm_value(l),
        );
    }

    let ops = ops.iter().map(|op| (op.key, op.value)).collect::<Vec<_>>();
    match verify_multi_proof_update::<H>(&verified, ops.clone()) {
        Ok(new_root) => assert_eq!(
            new_root,
            root_after(&run.leaves, &ops),
            "multi-proof update accepted with a wrong root"
        ),
        Err(e) => assert!(!must_verify, "honest multi-proof update rejected: {e:?}"),
    }
}

/// A full `Witness`, consumed as in `examples/witness_verification` but without `unwrap`.
fn witness_case(run: &Run, root: Node) {
    let honest = honest_witness(run);
    let Some(witness) = run.transmit(&honest) else {
        return;
    };
    let reads = &witness.operations.reads;
    let writes = &witness.operations.writes;

    let mut updates = Vec::new();
    let mut applied_ops = Vec::new();
    for (i, witnessed_path) in witness.path_proofs.iter().enumerate() {
        let verified = match witnessed_path
            .inner
            .verify::<H>(witnessed_path.path.path(), root)
        {
            Ok(verified) => verified,
            Err(e) => {
                assert!(!run.pristine(), "honest witness path rejected: {e:?}");
                continue;
            }
        };

        for read in reads
            .iter()
            .skip_while(|r| r.path_index != i)
            .take_while(|r| r.path_index == i)
        {
            check_claims(
                &run.leaves,
                read.key,
                run.pristine(),
                |k| verified.confirm_nonexistence(k),
                |l| verified.confirm_value(l),
            );
            // The value the witness claims was read.
            let confirmed = match read.value {
                None => verified.confirm_nonexistence(&read.key),
                Some(value_hash) => verified.confirm_value(&LeafData {
                    key_path: read.key,
                    value_hash,
                }),
            };
            match confirmed {
                Ok(true) => assert_eq!(read.value, run.leaves.get(&read.key).copied()),
                Ok(false) => assert_ne!(read.value, run.leaves.get(&read.key).copied()),
                Err(KeyOutOfScope) => assert!(!run.pristine()),
            }
        }

        let ops = writes
            .iter()
            .skip_while(|w| w.path_index != i)
            .take_while(|w| w.path_index == i)
            .map(|w| (w.key, w.value))
            .collect::<Vec<_>>();
        if !ops.is_empty() || run.keep_empty_updates {
            applied_ops.extend_from_slice(&ops);
            updates.push(PathUpdate {
                inner: verified,
                ops,
            });
        }
    }

    match verify_update::<H>(root, &updates) {
        Ok(new_root) => assert_eq!(
            new_root,
            root_after(&run.leaves, &applied_ops),
            "witness update accepted with a wrong root"
        ),
        Err(e) => assert!(
            !run.pristine() || updates.iter().any(|u| u.ops.is_empty()),
            "honest witness update rejected: {e:?}"
        ),
    }
}

/// The witness nomt would produce for reading every query and writing `ops`: one path per
/// distinct terminal, in key order.
fn honest_witness(run: &Run) -> Witness {
    let mut paths: Vec<(PathProof, TriePosition)> = Vec::new();
    let mut reads = Vec::new();
    let mut writes = Vec::new();
    for key in &run.queries {
        let proof = path_proof(&run.leaves, key);
        let position = position(*key, proof.siblings.len() as u16);
        if paths
            .last()
            .map_or(true, |(_, p)| p.path() != position.path())
        {
            paths.push((proof, position));
        }
        let path_index = paths.len() - 1;
        reads.push(Op {
            key: *key,
            value: run.leaves.get(key).copied(),
            path_index,
        });
        if let Some(op) = run.ops.iter().find(|op| op.key == *key) {
            writes.push(Op { path_index, ..*op });
        }
    }

    for edit in &run.struct_edits {
        match edit {
            StructEdit::Read(edit) => edit.apply(&mut reads),
            StructEdit::Write(edit) => edit.apply(&mut writes),
            StructEdit::Path(edit) => edit.apply(&mut paths),
        }
    }

    Witness {
        path_proofs: paths
            .into_iter()
            .map(|(inner, path)| WitnessedPath { inner, path })
            .collect(),
        operations: WitnessedOperations {
            reads: reads
                .iter()
                .map(|op| WitnessedRead {
                    key: op.key,
                    value: op.value,
                    path_index: op.path_index,
                })
                .collect(),
            writes: writes
                .iter()
                .map(|op| WitnessedWrite {
                    key: op.key,
                    value: op.value,
                    path_index: op.path_index,
                })
                .collect(),
        },
    }
}

// --- encoding ---------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Arbitrary)]
enum Codec {
    Borsh,
    /// serde, with bincode's fixed-width integers.
    Bincode,
}

fn bincode() -> impl bincode::Options {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .reject_trailing_bytes()
}

impl Codec {
    fn encode<T: BorshSerialize + Serialize>(self, value: &T) -> Vec<u8> {
        match self {
            Codec::Borsh => borsh::to_vec(value).unwrap(),
            Codec::Bincode => bincode().serialize(value).unwrap(),
        }
    }

    fn decode<T: BorshDeserialize + DeserializeOwned>(self, bytes: &[u8]) -> Option<T> {
        match self {
            Codec::Borsh => borsh::from_slice(bytes).ok(),
            Codec::Bincode => bincode().deserialize(bytes).ok(),
        }
    }
}

impl Run {
    /// Whether the proof reaches the verifier exactly as an honest prover built it.
    fn pristine(&self) -> bool {
        matches!(&self.source, Source::Edited(edits) if edits.is_empty())
            && self.struct_edits.is_empty()
    }

    /// Sends `honest` (or the raw bytes) through the codec, applying the byte edits on the way.
    fn transmit<T>(&self, honest: &T) -> Option<T>
    where
        T: BorshSerialize + BorshDeserialize + Serialize + DeserializeOwned,
    {
        let bytes = match &self.source {
            Source::Raw(bytes) => bytes.clone(),
            Source::Edited(edits) => {
                let mut bytes = self.codec.encode(honest);
                for edit in edits {
                    edit.apply(&mut bytes);
                }
                bytes
            }
        };
        let Some(decoded) = self.codec.decode::<T>(&bytes) else {
            assert!(!self.pristine(), "honest encoding does not decode");
            return None;
        };
        assert!(
            self.codec.encode(&decoded) == bytes,
            "decoded value does not re-encode to its input"
        );
        Some(decoded)
    }
}

#[derive(Debug)]
enum Source {
    /// Decode the fuzzer's bytes as they are.
    Raw(Vec<u8>),
    /// Encode the honest proof, then apply these edits.
    Edited(Vec<ByteEdit>),
}

#[derive(Debug, Arbitrary)]
enum ByteEdit {
    Overwrite {
        at: u16,
        bytes: Vec<u8>,
    },
    Insert {
        at: u16,
        bytes: Vec<u8>,
    },
    Delete {
        at: u16,
        len: u16,
    },
    Truncate {
        len: u16,
    },
    /// Little-endian, as both codecs encode integers: hits lengths, depths and indices.
    Int {
        at: u16,
        value: Int,
    },
}

#[derive(Debug, Arbitrary)]
enum Int {
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
}

impl ByteEdit {
    fn apply(&self, bytes: &mut Vec<u8>) {
        let pos = |at: u16, bytes: &Vec<u8>| at as usize % (bytes.len() + 1);
        let overwrite = |at: usize, new: &[u8], bytes: &mut Vec<u8>| {
            let end = (at + new.len()).min(bytes.len());
            bytes[at..end].copy_from_slice(&new[..end - at]);
        };
        match self {
            ByteEdit::Overwrite { at, bytes: new } => overwrite(pos(*at, bytes), new, bytes),
            ByteEdit::Insert { at, bytes: new } => {
                let at = pos(*at, bytes);
                bytes.splice(at..at, new.iter().copied());
            }
            ByteEdit::Delete { at, len } => {
                let at = pos(*at, bytes);
                let end = (at + *len as usize).min(bytes.len());
                bytes.drain(at..end);
            }
            ByteEdit::Truncate { len } => bytes.truncate(pos(*len, bytes)),
            ByteEdit::Int { at, value } => {
                let new = match *value {
                    Int::U8(v) => v.to_le_bytes().to_vec(),
                    Int::U16(v) => v.to_le_bytes().to_vec(),
                    Int::U32(v) => v.to_le_bytes().to_vec(),
                    Int::U64(v) => v.to_le_bytes().to_vec(),
                };
                overwrite(pos(*at, bytes), &new, bytes);
            }
        }
    }
}

// --- structural edits to the operations -------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct Op {
    key: KeyPath,
    value: Option<ValueHash>,
    /// Ignored outside witnesses.
    path_index: usize,
}

#[derive(Debug, Arbitrary)]
enum StructEdit {
    /// Witness reads.
    Read(OpEdit),
    /// Witness writes, or the multi-proof update ops.
    Write(OpEdit),
    /// Witness paths.
    Path(ListEdit),
}

#[derive(Debug, Arbitrary)]
enum ListEdit {
    Swap { a: u8, b: u8 },
    Duplicate { index: u8 },
    Remove { index: u8 },
}

#[derive(Debug, Arbitrary)]
enum OpEdit {
    List(ListEdit),
    SetKey {
        index: u8,
        key: KeyPath,
    },
    FlipKeyBit {
        index: u8,
        bit: u8,
    },
    SetValue {
        index: u8,
        value: Option<ValueHash>,
    },
    SetPathIndex {
        index: u8,
        path_index: u8,
    },
    Push {
        op: (KeyPath, Option<ValueHash>, u8),
    },
}

impl ListEdit {
    fn apply<T: Clone>(&self, list: &mut Vec<T>) {
        if list.is_empty() {
            return;
        }
        let len = list.len();
        match *self {
            ListEdit::Swap { a, b } => list.swap(a as usize % len, b as usize % len),
            ListEdit::Duplicate { index } => {
                let item = list[index as usize % len].clone();
                list.insert(index as usize % len, item);
            }
            ListEdit::Remove { index } => {
                list.remove(index as usize % len);
            }
        }
    }
}

impl OpEdit {
    fn apply(&self, ops: &mut Vec<Op>) {
        if let OpEdit::Push {
            op: (key, value, path_index),
        } = *self
        {
            ops.push(Op {
                key,
                value,
                path_index: path_index as usize,
            });
            return;
        }
        if ops.is_empty() {
            return;
        }
        let len = ops.len();
        match *self {
            OpEdit::List(ref edit) => edit.apply(ops),
            OpEdit::SetKey { index, key } => ops[index as usize % len].key = key,
            OpEdit::FlipKeyBit { index, bit } => {
                ops[index as usize % len].key[bit as usize / 8] ^= 0x80 >> (bit % 8)
            }
            OpEdit::SetValue { index, value } => ops[index as usize % len].value = value,
            OpEdit::SetPathIndex { index, path_index } => {
                ops[index as usize % len].path_index = path_index as usize
            }
            OpEdit::Push { .. } => unreachable!(),
        }
    }
}

// --- reference trie (as in proof_verify) ------------------------------------------------------

/// Asserts that what a verified proof claims about `key` matches `model`.
/// `in_scope` is set when the proof must cover `key`, so it must not report it out of scope.
fn check_claims(
    model: &Model,
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
    match confirm_value(&LeafData {
        key_path: key,
        value_hash: wrong,
    }) {
        Ok(found) => assert!(!found),
        Err(KeyOutOfScope) => assert!(!in_scope),
    }
}

fn root_of(model: &Model) -> Node {
    node(&model.iter().map(|(k, v)| (*k, *v)).collect::<Vec<_>>(), 0)
}

/// The model's root after applying `ops` in order.
fn root_after(model: &Model, ops: &[(KeyPath, Option<ValueHash>)]) -> Node {
    let mut model = model.clone();
    for (key, value) in ops {
        match value {
            Some(value) => model.insert(*key, *value),
            None => model.remove(key),
        };
    }
    root_of(&model)
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

fn path_proof(model: &Model, key: &KeyPath) -> PathProof {
    let leaves = model.iter().map(|(k, v)| (*k, *v)).collect::<Vec<_>>();
    let mut leaves = &leaves[..];
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
    match depth {
        0 => TriePosition::new(),
        depth => TriePosition::from_path_and_depth(path, depth),
    }
}

// --- input ------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Arbitrary)]
enum Kind {
    PathProof,
    MultiProof,
    Witness,
}

#[derive(Debug)]
struct Run {
    kind: Kind,
    codec: Codec,
    leaves: Model,
    /// Sorted and distinct.
    queries: Vec<KeyPath>,
    /// Sorted and distinct, each on a queried key, before any edits.
    ops: Vec<Op>,
    query_index: u8,
    source: Source,
    struct_edits: Vec<StructEdit>,
    /// Pass witness paths without writes to `verify_update` instead of skipping them.
    keep_empty_updates: bool,
}

impl<'a> Arbitrary<'a> for Run {
    fn arbitrary(input: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let kind = input.arbitrary()?;
        let codec = input.arbitrary()?;

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
                ops.insert(key, input.arbitrary::<Option<ValueHash>>()?);
            }
        }

        let query_index = input.arbitrary()?;
        let keep_empty_updates = input.ratio(1, 8)?;
        let source = if input.ratio(1, 8)? {
            Source::Raw(input.arbitrary()?)
        } else {
            let mut edits = Vec::new();
            for _ in 0..input.int_in_range(0..=4)? {
                edits.push(input.arbitrary()?);
            }
            Source::Edited(edits)
        };
        let mut struct_edits = Vec::new();
        for _ in 0..input.int_in_range(0..=4)? {
            struct_edits.push(input.arbitrary()?);
        }

        Ok(Self {
            kind,
            codec,
            leaves,
            queries: queries.into_iter().collect(),
            ops: ops
                .into_iter()
                .map(|(key, value)| Op {
                    key,
                    value,
                    path_index: 0,
                })
                .collect(),
            query_index,
            source,
            struct_edits,
            keep_empty_updates,
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
