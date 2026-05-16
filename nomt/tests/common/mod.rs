use nomt::{
    trie::{KeyPath, Node},
    KeyReadWrite, Nomt, Options, Overlay, PanicOnSyncMode, Root, Session, SessionParams, Value,
    Witness, WitnessMode,
};
use nomt_core::proof::PathProof;
use nomt_test_utils::{DivergingPair, SharedPrefixCluster, TestKeyPath};
use quickcheck::{Arbitrary, Gen};
use std::{
    collections::{hash_map::Entry, BTreeMap, BTreeSet, HashMap},
    mem,
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
};

static NEXT_TEST_ID: AtomicUsize = AtomicUsize::new(0);

#[allow(unused_imports)]
pub use nomt_test_utils::{account_path, key_diverging_at};

#[allow(dead_code)]
pub fn expected_root(accounts: u64) -> Node {
    let mut ops = (0..accounts)
        .map(account_path)
        .map(|a| (a, *blake3::hash(&1000u64.to_le_bytes()).as_bytes()))
        .collect::<Vec<_>>();
    ops.sort_unstable_by_key(|(a, _)| *a);
    nomt_core::update::build_trie::<nomt::hasher::Blake3Hasher>(0, ops, |_| {})
}

#[allow(dead_code)]
pub fn fresh_test_name(prefix: &str) -> String {
    format!("{prefix}_{}", NEXT_TEST_ID.fetch_add(1, Ordering::Relaxed))
}

#[allow(dead_code)]
fn fill_missing_keys(g: &mut Gen, excluded: &mut BTreeSet<KeyPath>, target: usize) -> Vec<KeyPath> {
    let mut out = Vec::with_capacity(target);
    let attempt_budget = target.saturating_mul(8) + 8;
    let mut attempts = 0;
    while out.len() < target && attempts < attempt_budget {
        let key = TestKeyPath::arbitrary(g).into_inner();
        if excluded.insert(key) {
            out.push(key);
        }
        attempts += 1;
    }
    let mut fill_seed = 0u64;
    while out.len() < target {
        let key = account_path(fill_seed);
        if excluded.insert(key) {
            out.push(key);
        }
        fill_seed = fill_seed.wrapping_add(1);
    }
    out
}

#[allow(dead_code)]
pub fn apply_accesses(t: &mut Test, accesses: &[(KeyPath, KeyReadWrite)]) {
    for (key, access) in accesses {
        match access {
            KeyReadWrite::Read(expected) => {
                assert_eq!(t.read(*key), *expected);
            }
            KeyReadWrite::Write(value) => t.write(*key, value.clone()),
            KeyReadWrite::ReadThenWrite(expected, value) => {
                assert_eq!(t.read(*key), *expected);
                t.write(*key, value.clone());
            }
        }
    }
}

#[allow(dead_code)]
pub fn arbitrary_small_value(g: &mut Gen) -> Value {
    let len = usize::arbitrary(g) % 9;
    let mut value = Vec::with_capacity(len);
    for _ in 0..len {
        value.push(u8::arbitrary(g));
    }
    value
}

#[allow(dead_code)]
pub fn arbitrary_optional_small_value(g: &mut Gen) -> Option<Value> {
    if bool::arbitrary(g) {
        Some(arbitrary_small_value(g))
    } else {
        None
    }
}

#[allow(dead_code)]
pub fn arbitrary_interesting_keys(g: &mut Gen, max_len: usize) -> Vec<KeyPath> {
    let target_len = usize::arbitrary(g) % (max_len + 1);
    let mut keys = BTreeSet::new();

    if target_len >= 2 && bool::arbitrary(g) {
        let pair = DivergingPair::arbitrary(g);
        keys.insert(pair.left);
        keys.insert(pair.right);
    }

    if keys.len() < target_len && target_len >= 2 && bool::arbitrary(g) {
        let cluster = SharedPrefixCluster::arbitrary(g);
        for member in cluster.members {
            if keys.len() >= target_len {
                break;
            }
            keys.insert(member);
        }
    }

    while keys.len() < target_len {
        keys.insert(TestKeyPath::arbitrary(g).into_inner());
    }

    keys.into_iter().collect()
}

#[allow(dead_code)]
pub fn shuffle<T>(items: &mut [T], g: &mut Gen) {
    if items.len() < 2 {
        return;
    }

    for i in (1..items.len()).rev() {
        let j = usize::arbitrary(g) % (i + 1);
        items.swap(i, j);
    }
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct SessionAccessCase {
    pub prev_data: Vec<(KeyPath, Value)>,
    pub accesses: Vec<(KeyPath, KeyReadWrite)>,
}

#[allow(dead_code)]
impl SessionAccessCase {
    pub fn prev_data_with_options(&self) -> Vec<(KeyPath, Option<Value>)> {
        self.prev_data
            .iter()
            .map(|(key, value)| (*key, Some(value.clone())))
            .collect()
    }

    pub fn expected_final_state(&self) -> BTreeMap<KeyPath, Value> {
        let mut state = self
            .prev_data
            .iter()
            .map(|(key, value)| (*key, value.clone()))
            .collect::<BTreeMap<_, _>>();

        for (key, access) in &self.accesses {
            match access {
                KeyReadWrite::Read(_) => {}
                KeyReadWrite::Write(value) | KeyReadWrite::ReadThenWrite(_, value) => {
                    if let Some(value) = value {
                        state.insert(*key, value.clone());
                    } else {
                        state.remove(key);
                    }
                }
            }
        }

        state
    }
}

impl Arbitrary for SessionAccessCase {
    fn arbitrary(g: &mut Gen) -> Self {
        let mut prev_state = BTreeMap::new();
        for key in arbitrary_interesting_keys(g, 6) {
            prev_state.insert(key, arbitrary_small_value(g));
        }

        let mut all_keys = prev_state.keys().copied().collect::<BTreeSet<_>>();
        let missing_target = 1 + (usize::arbitrary(g) % 4);
        let missing_keys = fill_missing_keys(g, &mut all_keys, missing_target);

        let mut accesses = Vec::new();
        for (key, value) in &prev_state {
            if bool::arbitrary(g) {
                accesses.push((*key, arbitrary_present_access(g, value.clone())));
            }
        }
        let allow_missing_read_then_write = !prev_state.is_empty();
        for key in missing_keys {
            if bool::arbitrary(g) {
                accesses.push((
                    key,
                    arbitrary_missing_access(g, allow_missing_read_then_write),
                ));
            }
        }

        if accesses.is_empty() {
            if let Some((key, value)) = prev_state.iter().next() {
                accesses.push((*key, arbitrary_present_access(g, value.clone())));
            } else {
                accesses.push((
                    TestKeyPath::arbitrary(g).into_inner(),
                    arbitrary_missing_access(g, false),
                ));
            }
        }

        shuffle(&mut accesses, g);

        Self {
            prev_data: prev_state.into_iter().collect(),
            accesses,
        }
    }
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct ProofCase {
    pub state: Vec<(KeyPath, Value)>,
    pub present_samples: Vec<KeyPath>,
    pub missing_samples: Vec<KeyPath>,
}

impl Arbitrary for ProofCase {
    fn arbitrary(g: &mut Gen) -> Self {
        let state_keys = arbitrary_interesting_keys(g, 8);
        let mut state = state_keys
            .iter()
            .map(|key| (*key, arbitrary_small_value(g)))
            .collect::<Vec<_>>();
        state.sort_by_key(|(key, _)| *key);

        let mut present_samples = state_keys
            .iter()
            .copied()
            .filter(|_| bool::arbitrary(g))
            .collect::<Vec<_>>();
        if present_samples.is_empty() && !state_keys.is_empty() {
            present_samples.push(state_keys[0]);
        }

        let mut used_keys = state_keys.into_iter().collect::<BTreeSet<_>>();
        let missing_target = 1 + (usize::arbitrary(g) % 4);
        let mut missing_samples = fill_missing_keys(g, &mut used_keys, missing_target);

        shuffle(&mut present_samples, g);
        shuffle(&mut missing_samples, g);

        Self {
            state,
            present_samples,
            missing_samples,
        }
    }
}

fn arbitrary_present_access(g: &mut Gen, prior: Value) -> KeyReadWrite {
    match u8::arbitrary(g) % 3 {
        0 => KeyReadWrite::Read(Some(prior)),
        1 => KeyReadWrite::Write(arbitrary_optional_small_value(g)),
        _ => KeyReadWrite::ReadThenWrite(Some(prior), arbitrary_optional_small_value(g)),
    }
}

fn arbitrary_missing_access(g: &mut Gen, allow_read_then_write: bool) -> KeyReadWrite {
    match if allow_read_then_write {
        u8::arbitrary(g) % 3
    } else {
        u8::arbitrary(g) % 2
    } {
        0 => KeyReadWrite::Read(None),
        1 => KeyReadWrite::Write(Some(arbitrary_small_value(g))),
        _ => KeyReadWrite::ReadThenWrite(None, Some(arbitrary_small_value(g))),
    }
}

fn opts(path: PathBuf) -> Options {
    let mut opts = Options::new();
    opts.path(path);
    opts.commit_concurrency(1);
    opts
}

pub struct Test {
    nomt: Nomt<nomt::hasher::Blake3Hasher>,
    session: Option<Session<nomt::hasher::Blake3Hasher>>,
    access: HashMap<KeyPath, KeyReadWrite>,
}

#[allow(dead_code)]
impl Test {
    pub fn new(name: impl AsRef<Path>) -> Self {
        Self::new_with_params(name, 1, 64_000, None, true)
    }

    pub fn new_with_params(
        name: impl AsRef<Path>,
        commit_concurrency: usize,
        hashtable_buckets: u32,
        panic_on_sync: Option<PanicOnSyncMode>,
        cleanup_dir: bool,
    ) -> Self {
        let path = {
            let mut p = PathBuf::from("test");
            p.push(name);
            p
        };
        if cleanup_dir {
            let _ = std::fs::remove_dir_all(&path);
        }
        let mut o = opts(path);
        if let Some(mode) = panic_on_sync {
            o.panic_on_sync(mode);
        }
        o.bitbox_seed([0; 16]);
        o.hashtable_buckets(hashtable_buckets);
        o.commit_concurrency(commit_concurrency);
        o.io_workers(1); // Too many IO workers can run into system rlimits in tests.
        let nomt = Nomt::open(o).unwrap();
        let session =
            nomt.begin_session(SessionParams::default().witness_mode(WitnessMode::read_write()));
        Self {
            nomt,
            session: Some(session),
            access: HashMap::default(),
        }
    }

    pub fn write_id(&mut self, id: u64, value: Option<Vec<u8>>) {
        self.write(account_path(id), value);
    }

    pub fn write(&mut self, key: KeyPath, value: Option<Vec<u8>>) {
        match self.access.entry(key) {
            Entry::Occupied(mut o) => {
                o.get_mut().write(value);
            }
            Entry::Vacant(v) => {
                v.insert(KeyReadWrite::Write(value));
            }
        }
        self.session.as_mut().unwrap().warm_up(key);
    }

    pub fn read_id(&mut self, id: u64) -> Option<Vec<u8>> {
        self.read(account_path(id))
    }

    pub fn read(&mut self, key: KeyPath) -> Option<Vec<u8>> {
        match self.access.entry(key) {
            Entry::Occupied(o) => o.get().last_value().map(|v| v.to_vec()),
            Entry::Vacant(v) => {
                let session = self.session.as_mut().unwrap();
                let value = session.read(key).unwrap();
                session.warm_up(key);
                v.insert(KeyReadWrite::Read(value.clone()));
                value
            }
        }
    }

    pub fn commit(&mut self) -> (Root, Witness) {
        let session = mem::take(&mut self.session).unwrap();
        let mut actual_access: Vec<_> = mem::take(&mut self.access).into_iter().collect();
        actual_access.sort_by_key(|(k, _)| *k);
        let mut finished = session.finish(actual_access).unwrap();
        let root = finished.root();
        let witness = finished.take_witness().unwrap();
        finished.commit(&self.nomt).unwrap();
        self.session = Some(
            self.nomt
                .begin_session(SessionParams::default().witness_mode(WitnessMode::read_write())),
        );
        (root, witness)
    }

    pub fn update(&mut self) -> (Overlay, Witness) {
        let session = mem::take(&mut self.session).unwrap();
        let mut actual_access: Vec<_> = mem::take(&mut self.access).into_iter().collect();
        actual_access.sort_by_key(|(k, _)| *k);
        let mut finished = session.finish(actual_access).unwrap();
        let witness = finished.take_witness().unwrap();

        self.session = Some(
            self.nomt
                .begin_session(SessionParams::default().witness_mode(WitnessMode::read_write())),
        );

        (finished.into_overlay(), witness)
    }

    pub fn commit_overlay(&mut self, overlay: Overlay) {
        // force drop of live session before committing.
        self.access.clear();
        self.session = None;
        overlay.commit(&self.nomt).unwrap();
        self.session = Some(
            self.nomt
                .begin_session(SessionParams::default().witness_mode(WitnessMode::read_write())),
        );
    }

    pub fn start_overlay_session<'a>(&mut self, ancestors: impl IntoIterator<Item = &'a Overlay>) {
        // force drop of live session before creating a new one.
        self.access.clear();
        self.session = None;
        let params = SessionParams::default()
            .witness_mode(WitnessMode::read_write())
            .overlay(ancestors)
            .unwrap();
        self.session = Some(self.nomt.begin_session(params));
    }

    pub fn prove(&self, key: KeyPath) -> PathProof {
        self.session.as_ref().unwrap().prove(key).unwrap()
    }

    pub fn prove_id(&self, id: u64) -> PathProof {
        self.prove(account_path(id))
    }

    pub fn root(&self) -> Root {
        self.nomt.root()
    }
}

pub fn read_balance(t: &mut Test, id: u64) -> Option<u64> {
    t.read_id(id)
        .map(|v| u64::from_le_bytes(v[..].try_into().unwrap()))
}

pub fn set_balance(t: &mut Test, id: u64, balance: u64) {
    t.write_id(id, Some(balance.to_le_bytes().to_vec()));
}

#[allow(unused)]
pub fn transfer(t: &mut Test, from: u64, to: u64, amount: u64) {
    let from_balance = read_balance(t, from).unwrap_or(0);
    let to_balance = read_balance(t, to).unwrap_or(0);
    if from_balance < amount {
        return;
    }
    set_balance(t, from, from_balance - amount);
    set_balance(t, to, to_balance + amount);
}

#[allow(unused)]
pub fn kill(t: &mut Test, from: u64) {
    t.write_id(from, None);
}
