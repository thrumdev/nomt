use nomt::{KeyPath, KeyReadWrite, Node, Nomt, Options, Session};
use std::{
    collections::{hash_map::Entry, HashMap},
    mem,
    rc::Rc,
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

fn opts() -> Options {
    Options {
        path: "test".into(),
        fetch_concurrency: 1,
        traversal_concurrency: 1,
    }
}

pub struct Test {
    nomt: Nomt,
    session: Option<Session>,
    access: HashMap<KeyPath, KeyReadWrite>,
}

impl Test {
    pub fn new() -> Self {
        let _ = std::fs::remove_dir_all("test");
        let nomt = Nomt::open(opts()).unwrap();
        let session = nomt.begin_session();
        Self {
            nomt,
            session: Some(session),
            access: HashMap::default(),
        }
    }

    pub fn write(&mut self, id: u64, value: Option<u64>) {
        let path = account_path(id);
        let value = value.map(|v| Rc::new(v.to_le_bytes().to_vec()));
        let is_delete = value.is_none();
        match self.access.entry(path) {
            Entry::Occupied(mut o) => {
                o.get_mut().write(value);
            }
            Entry::Vacant(v) => {
                v.insert(KeyReadWrite::Write(value));
            }
        }
        self.session
            .as_mut()
            .unwrap()
            .tentative_write_slot(path, is_delete);
    }

    #[allow(unused)]
    pub fn read(&mut self, id: u64) -> Option<u64> {
        let path = account_path(id);
        let to_u64 = |v: Option<&[u8]>| v.map(|v| u64::from_le_bytes(v.try_into().unwrap()));
        let value = match self.access.entry(path) {
            Entry::Occupied(o) => o.get().last_value().cloned(),
            Entry::Vacant(v) => {
                let value = self
                    .session
                    .as_mut()
                    .unwrap()
                    .tentative_read_slot(path)
                    .unwrap();
                v.insert(KeyReadWrite::Read(value.clone()));
                value
            }
        };
        to_u64(value.as_ref().map(|v| &v[..]))
    }

    pub fn commit(&mut self) -> Node {
        let session = mem::take(&mut self.session).unwrap();
        let mut actual_access: Vec<_> = mem::take(&mut self.access).into_iter().collect();
        actual_access.sort_by_key(|(k, _)| *k);
        self.nomt.commit_and_prove(session, actual_access).unwrap();
        self.session = Some(self.nomt.begin_session());
        self.nomt.root()
    }
}

pub fn set_balance(t: &mut Test, id: u64, balance: u64) {
    t.write(id, Some(balance));
}

#[allow(unused)]
pub fn transfer(t: &mut Test, from: u64, to: u64, amount: u64) {
    let from = t.read(from).unwrap_or(0);
    let to = t.read(to).unwrap_or(0);
    if from < amount {
        return;
    }
    t.write(from, Some(from - amount));
    t.write(to, Some(to + amount));
}

pub fn kill(t: &mut Test, from: u64) {
    t.write(from, None);
}
