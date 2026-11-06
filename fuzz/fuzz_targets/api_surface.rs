#![no_main]

mod reference;

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    path::{Path, PathBuf},
};

use arbitrary::Arbitrary;
use bitvec::view::BitView;
use libfuzzer_sys::fuzz_target;

use nomt::{
    hasher::{Blake3Hasher, ValueHasher},
    proof::{verify_update, PathUpdate},
    trie::{KeyPath, LeafData, Node, ValueHash},
    KeyReadWrite, Nomt, Options, SessionParams, Value, Witness, WitnessMode,
};
use nomt_core::update::build_trie;

fuzz_target!(|run: Run| {
    let tempdir = tempfile::tempdir().unwrap();
    let path = tempdir.path().join("db");
    let mut db = Nomt::<Blake3Hasher>::open(run.open.options(&path)).unwrap();
    let mut model = BTreeMap::<KeyPath, ValueHash>::new();

    for call in run.calls.calls {
        match call {
            NomtCall::Reopen => {
                drop(db);
                db = Nomt::open(run.open.options(&path)).unwrap();
                assert_eq!(db.root().into_inner(), reference_root(&model));
                for (key_path, value_hash) in &model {
                    let value = db.read(*key_path).unwrap().unwrap();
                    assert_eq!(Blake3Hasher::hash_value(&value), *value_hash);
                }
            }
            NomtCall::BeginSession { session_calls } => {
                let session = db.begin_session(
                    SessionParams::default().witness_mode(WitnessMode::read_write()),
                );
                for session_call in session_calls {
                    match session_call {
                        SessionCall::TentativeRead {
                            key_path,
                            expected_value,
                        } => {
                            let actual_value = session.read(key_path).unwrap();
                            session.warm_up(key_path);
                            assert_eq!(actual_value, expected_value);
                        }
                        SessionCall::TentativeWrite { key_path } => {
                            session.warm_up(key_path);
                        }
                        SessionCall::Prove { key_path } => {
                            let verified = session
                                .prove(key_path)
                                .unwrap()
                                .verify::<Blake3Hasher>(
                                    key_path.view_bits(),
                                    session.prev_root().into_inner(),
                                )
                                .unwrap();
                            match model.get(&key_path) {
                                Some(value_hash) => assert!(verified
                                    .confirm_value(&LeafData {
                                        key_path,
                                        value_hash: *value_hash,
                                    })
                                    .unwrap()),
                                None => {
                                    assert!(verified.confirm_nonexistence(&key_path).unwrap())
                                }
                            }
                        }
                        SessionCall::CommitAndProve { keys } => {
                            let prev_root = db.root().into_inner();
                            let mut finished = session.finish(keys.clone()).unwrap();
                            let witness = finished.take_witness().unwrap();
                            check_witness(witness, &keys, &model, prev_root, finished.root());

                            for (key_path, key_rw) in &keys {
                                match key_rw {
                                    KeyReadWrite::Write(Some(value))
                                    | KeyReadWrite::ReadThenWrite(_, Some(value)) => {
                                        model.insert(*key_path, Blake3Hasher::hash_value(value));
                                    }
                                    KeyReadWrite::Write(None)
                                    | KeyReadWrite::ReadThenWrite(_, None) => {
                                        model.remove(key_path);
                                    }
                                    KeyReadWrite::Read(_) => {}
                                }
                            }
                            finished.commit(&db).unwrap();
                            let expected_root = reference_root(&model);
                            assert_eq!(
                                build_trie::<Blake3Hasher>(
                                    0,
                                    model.iter().map(|(k, v)| (*k, *v)),
                                    |_| {},
                                ),
                                expected_root
                            );
                            assert_eq!(db.root().into_inner(), expected_root);
                            break;
                        }
                        SessionCall::Drop => {
                            drop(session);
                            break;
                        }
                    }
                }
            }
        }
    }
});

fn reference_root(model: &BTreeMap<KeyPath, ValueHash>) -> Node {
    let leaves = model.iter().map(|(k, v)| (*k, *v)).collect::<Vec<_>>();
    reference::node(&leaves, 0)
}

/// Checks that the witness proves exactly the session's reads and writes against `prev_root`, with
/// the values in `model`, and that applying its writes yields `new_root`.
fn check_witness(
    witness: Witness,
    keys: &[(KeyPath, KeyReadWrite)],
    model: &BTreeMap<KeyPath, ValueHash>,
    prev_root: Node,
    new_root: nomt::Root,
) {
    let verified = witness
        .path_proofs
        .iter()
        .map(|p| {
            p.inner
                .verify::<Blake3Hasher>(p.path.path(), prev_root)
                .unwrap()
        })
        .collect::<Vec<_>>();

    let mut reads = Vec::new();
    for read in &witness.operations.reads {
        assert_eq!(read.value, model.get(&read.key).copied());
        let proof = &verified[read.path_index];
        match read.value {
            Some(value_hash) => assert!(proof
                .confirm_value(&LeafData {
                    key_path: read.key,
                    value_hash,
                })
                .unwrap()),
            None => assert!(proof.confirm_nonexistence(&read.key).unwrap()),
        }
        reads.push(read.key);
    }
    reads.sort_unstable();
    let expected_reads = keys
        .iter()
        .filter(|(_, rw)| matches!(rw, KeyReadWrite::Read(_) | KeyReadWrite::ReadThenWrite(..)))
        .map(|(k, _)| *k)
        .collect::<Vec<_>>();
    assert_eq!(reads, expected_reads);

    let mut writes = Vec::new();
    let mut ops_by_path = BTreeMap::<usize, Vec<(KeyPath, Option<ValueHash>)>>::new();
    for write in &witness.operations.writes {
        writes.push((write.key, write.value));
        ops_by_path
            .entry(write.path_index)
            .or_default()
            .push((write.key, write.value));
    }
    writes.sort_unstable();
    let expected_writes = keys
        .iter()
        .filter_map(|(k, rw)| match rw {
            KeyReadWrite::Write(v) | KeyReadWrite::ReadThenWrite(_, v) => {
                Some((*k, v.as_ref().map(|v| Blake3Hasher::hash_value(v))))
            }
            KeyReadWrite::Read(_) => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(writes, expected_writes);

    let mut updates = ops_by_path
        .into_iter()
        .map(|(i, ops)| PathUpdate {
            inner: verified[i].clone(),
            ops,
        })
        .collect::<Vec<_>>();
    // Witness path proofs may be emitted in worker order rather than trie path order.
    updates.sort_by(|a, b| a.inner.path().cmp(b.inner.path()));
    assert_eq!(
        verify_update::<Blake3Hasher>(prev_root, &updates).unwrap(),
        new_root.into_inner()
    );
}

struct Context {
    /// All the key paths that have been created, irregardless of whether they have been committed
    /// or not.
    key_paths: HashSet<KeyPath>,
    /// The same key paths in creation order, so that picking one is deterministic.
    key_path_list: Vec<KeyPath>,
    /// The key value pairs that have been committed into the store.
    ///
    /// The u32 is the index of the key path in `key_paths`.
    committed: HashMap<KeyPath, Value>,
}

impl Context {
    fn new_key_path(
        &mut self,
        input: &mut arbitrary::Unstructured<'_>,
    ) -> arbitrary::Result<Option<KeyPath>> {
        let mut attempts = 0;
        loop {
            let key_path = reference::new_key(input, &self.key_path_list)?;
            if self.key_paths.insert(key_path) {
                self.key_path_list.push(key_path);
                return Ok(Some(key_path));
            }
            attempts += 1;
            if attempts > 10 {
                return Ok(None);
            }
        }
    }

    fn new_value(
        &mut self,
        input: &mut arbitrary::Unstructured<'_>,
    ) -> arbitrary::Result<Option<Value>> {
        if input.arbitrary()? {
            // Inline leaf values, both sides of the overflow threshold (MAX_LEAF_VALUE_SIZE = 1332)
            // and overflow values spanning several pages.
            let size = match input.int_in_range(0..=6)? {
                0 => 0,
                1 => input.int_in_range(1..=32)?,
                2 => input.int_in_range(33..=1331)?,
                3 => input.int_in_range(1332..=1333)?,
                4 => input.int_in_range(1334..=4096)?,
                5 => input.int_in_range(4097..=8192)?,
                _ => input.int_in_range(8193..=32 * 1024)?,
            };
            let mut buf = vec![0; size];
            input.fill_buffer(&mut buf)?;
            Ok(Some(buf.into()))
        } else {
            Ok(None)
        }
    }
}

#[derive(Debug)]
struct Run {
    open: OpenParams,
    calls: NomtCalls,
}

impl<'a> Arbitrary<'a> for Run {
    fn arbitrary(input: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        let run = Run {
            open: OpenParams {
                commit_concurrency: input.int_in_range(1..=4)?,
                page_cache_size: input.int_in_range(1..=32)?,
                leaf_cache_size: input.int_in_range(1..=32)?,
                page_cache_upper_levels: input.int_in_range(0..=2)?,
                prepopulate_page_cache: input.arbitrary()?,
            },
            calls: input.arbitrary()?,
        };
        Ok(run)
    }
}

#[derive(Debug)]
struct OpenParams {
    commit_concurrency: usize,
    page_cache_size: usize,
    leaf_cache_size: usize,
    page_cache_upper_levels: usize,
    prepopulate_page_cache: bool,
}

impl OpenParams {
    fn options(&self, path: &Path) -> Options {
        let mut options = Options::new();
        options.path(PathBuf::from(path));
        options.commit_concurrency(self.commit_concurrency);
        options.page_cache_size(self.page_cache_size);
        options.leaf_cache_size(self.leaf_cache_size);
        options.page_cache_upper_levels(self.page_cache_upper_levels);
        options.prepopulate_page_cache(self.prepopulate_page_cache);
        options.preallocate_ht(false);
        options
    }
}

#[derive(Debug)]
enum NomtCall {
    BeginSession {
        session_calls: Vec<SessionCall>,
    },
    /// Close and reopen the database, then check every committed value.
    Reopen,
}

#[derive(Debug)]
enum SessionCall {
    TentativeRead {
        key_path: KeyPath,
        expected_value: Option<Value>,
    },
    TentativeWrite {
        key_path: KeyPath,
    },
    /// Prove the key against the session's base root.
    Prove {
        key_path: KeyPath,
    },
    /// Commit the given keys and check the witness.
    CommitAndProve {
        keys: Vec<(KeyPath, KeyReadWrite)>,
    },
    /// Instructs to drop the session without committing.
    Drop,
}

#[derive(Debug)]
struct NomtCalls {
    pub calls: Vec<NomtCall>,
}

impl<'a> Arbitrary<'a> for NomtCalls {
    fn arbitrary(input: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        let call_cnt = input.int_in_range(0..=10)?;
        let mut calls = Vec::with_capacity(call_cnt);

        let mut cx = Context {
            key_paths: HashSet::new(),
            key_path_list: Vec::new(),
            committed: HashMap::new(),
        };

        for _ in 0..call_cnt {
            if input.ratio(1, 8)? {
                calls.push(NomtCall::Reopen);
            }

            let session_sz = input.int_in_range(0..=1000)?;

            // Collect all the available keys into a vector and destroy the non-determinism
            // by sorting them.
            let mut available_keys = cx.key_paths.iter().copied().collect::<Vec<_>>();
            available_keys.sort_unstable();

            let mut session: Vec<(KeyPath, KeyReadWrite)> = Vec::new();
            for _ in 0..session_sz {
                // Select a key: either an already used key or a brand new one.
                let key_path: KeyPath = if !available_keys.is_empty() && input.ratio(9, 10)? {
                    let idx = input.choose_index(available_keys.len())?;
                    available_keys.swap_remove(idx)
                } else {
                    // Create a brand new key.
                    match cx.new_key_path(input)? {
                        Some(key_path) => key_path,
                        None => {
                            // No new key path was created, so we need to skip this session.
                            break;
                        }
                    }
                };

                // Now, we need to choose if we want to read, write, or read-then-write with each
                // value.
                match *input.choose(&[0, 1, 2])? {
                    0 => {
                        // Check if the key has been committed, because we need our session ops be
                        // consistent with the committed state.
                        let value = cx.committed.get(&key_path).cloned();
                        session.push((key_path, KeyReadWrite::Read(value)));
                    }
                    1 => {
                        session.push((key_path, KeyReadWrite::Write(cx.new_value(input)?)));
                    }
                    2 => {
                        // Check if the key has been committed, because we need our session ops be
                        // consistent with the committed state.
                        let old_value = cx.committed.get(&key_path).cloned();
                        let new_value = cx.new_value(input)?;
                        session.push((key_path, KeyReadWrite::ReadThenWrite(old_value, new_value)));
                    }
                    _ => unreachable!(),
                }
            }

            session.sort_unstable_by(|(a, _), (b, _)| a.cmp(b));

            // Now having an idea what the session would look like we should create the calls to the
            // api.
            let mut session_calls = Vec::new();
            for (key_path, key_rw) in &session {
                if input.ratio(1, 8)? {
                    session_calls.push(SessionCall::Prove {
                        key_path: *key_path,
                    });
                }
                match key_rw {
                    KeyReadWrite::ReadThenWrite(v, _) | KeyReadWrite::Read(v) => {
                        session_calls.push(SessionCall::TentativeRead {
                            key_path: *key_path,
                            expected_value: v.clone(),
                        });
                    }
                    KeyReadWrite::Write(_) => {
                        session_calls.push(SessionCall::TentativeWrite {
                            key_path: *key_path,
                        });
                    }
                }
            }

            let drop_session = input.ratio(1, 5)?;
            if drop_session {
                session_calls.push(SessionCall::Drop);
            } else {
                // Update committed state.
                for (key_path, key_rw) in &session {
                    match key_rw {
                        KeyReadWrite::ReadThenWrite(_, new_value)
                        | KeyReadWrite::Write(new_value) => {
                            if let Some(new_value) = new_value {
                                cx.committed.insert(*key_path, new_value.clone());
                            } else {
                                cx.committed.remove(key_path);
                            }
                        }
                        KeyReadWrite::Read(_) => {}
                    }
                }
                session_calls.push(SessionCall::CommitAndProve { keys: session });
            }

            calls.push(NomtCall::BeginSession { session_calls });
        }
        Ok(NomtCalls { calls })
    }
}
