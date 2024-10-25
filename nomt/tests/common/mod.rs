use nomt::{KeyPath, KeyReadWrite, Node, Nomt, Options, Session, Witness, WitnessedOperations};
use std::{
    collections::{hash_map::Entry, HashMap},
    mem,
    path::{Path, PathBuf},
};

pub fn account_path(id: u64) -> KeyPath {
    // KeyPaths must be uniformly distributed, but we don't want to spend time on a good hash. So
    // the next best option is to use a PRNG seeded with the id.
    use rand::{RngCore as _, SeedableRng as _};
    let mut seed = [0; 16];
    seed[0..8].copy_from_slice(&id.to_le_bytes());
    let mut rng = rand_pcg::Lcg64Xsh32::from_seed(seed);
    let mut path = KeyPath::default();
    for i in 0..4 {
        path[i * 4..][..4].copy_from_slice(&rng.next_u32().to_le_bytes());
    }
    path
}

#[allow(dead_code)]
pub fn expected_root(accounts: u64) -> Node {
    let mut ops = (0..accounts)
        .map(account_path)
        .map(|a| (a, *blake3::hash(&1000u64.to_le_bytes()).as_bytes()))
        .collect::<Vec<_>>();
    ops.sort_unstable_by_key(|(a, _)| *a);
    nomt_core::update::build_trie::<nomt::Blake3Hasher>(0, ops, |_| {})
}

fn opts(path: PathBuf) -> Options {
    let mut opts = Options::new();
    opts.path(path);
    opts.commit_concurrency(1);
    opts
}

pub struct Test {
    nomt: Nomt<nomt::Blake3Hasher>,
    session: Option<Session>,
    access: HashMap<KeyPath, KeyReadWrite>,
}

impl Test {
    pub fn new(name: impl AsRef<Path>) -> Self {
        Self::new_with_params(name, 1, 64_000, false, true)
    }

    pub fn new_with_params(
        name: impl AsRef<Path>,
        commit_concurrency: usize,
        hashtable_buckets: u32,
        panic_on_sync: bool,
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
        o.panic_on_sync(panic_on_sync);
        o.bitbox_seed([0; 16]);
        o.hashtable_buckets(hashtable_buckets);
        o.commit_concurrency(commit_concurrency);
        let nomt = Nomt::open(o).unwrap();
        let session = nomt.begin_session();
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

    pub fn commit(&mut self) -> (Node, Witness, WitnessedOperations) {
        let session = mem::take(&mut self.session).unwrap();
        let mut actual_access: Vec<_> = mem::take(&mut self.access).into_iter().collect();
        actual_access.sort_by_key(|(k, _)| *k);
        let x = self.nomt.commit_and_prove(session, actual_access).unwrap();
        self.session = Some(self.nomt.begin_session());
        x
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
