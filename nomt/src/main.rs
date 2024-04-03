use std::{collections::HashMap, mem, path::PathBuf};

use nomt::{CommitSpec, KeyPath, Node, Nomt, Options, Value};


fn path(id: u64) -> KeyPath {
    blake3::hash(&id.to_le_bytes()).into()
}

fn opts() -> Options {
    Options {
        path: PathBuf::from("test"),
        fetch_concurrency: 4,
        traversal_concurrency: 2,
    }
}

struct Test {
    nomt: Nomt,
    cached: HashMap<KeyPath, Option<Value>>,
    read_set: HashMap<KeyPath, Option<Value>>,
    write_set: HashMap<KeyPath, Option<Value>>,
}

impl Test {
    fn new() -> Self {
        let nomt = Nomt::open(opts()).unwrap();
        Self {
            nomt,
            read_set: HashMap::new(),
            write_set: HashMap::new(),
            cached: HashMap::new(),
        }
    }

    fn write(&mut self, id: u64, value: Option<u64>) {
        let path = path(id);
        let value = value.map(|v| v.to_le_bytes().to_vec());
        self.cached.insert(path, value.clone());
        self.write_set.insert(path, value);
        self.nomt.hint_write_slot(path);
    }

    fn read(&mut self, id: u64) -> Option<u64> {
        let path = path(id);
        let raw = if let Some(value) = self.cached.get(&path) {
            value.clone()
        } else {
            let value = self.nomt.read_slot(path).unwrap();
            self.read_set.insert(path, value.clone());
            value
        };
        raw.map(|v| u64::from_le_bytes(v.as_slice().try_into().unwrap()))
    }

    fn commit(&mut self) -> Node {
        let read_set = mem::take(&mut self.read_set);
        let write_set = mem::take(&mut self.write_set);
        self.nomt
            .commit_and_prove(CommitSpec {
                read_set: read_set.into_iter().collect(),
                write_set: write_set.into_iter().collect(),
            })
            .unwrap();
        self.nomt.root()
    }
}

fn set_balance(t: &mut Test, id: u64, balance: u64) {
    t.write(id, Some(balance));
}

fn transfer(t: &mut Test, from: u64, to: u64, amount: u64) {
    let from = t.read(from).unwrap_or(0);
    let to = t.read(to).unwrap_or(0);
    if from < amount {
        return;
    }
    t.write(from, Some(from - amount));
    t.write(to, Some(to + amount));
}

fn main() {
    let mut t = Test::new();
    let mut accounts = 0;
    let mut cur_account = 0;
    loop {
        for _ in 0..100 {
            set_balance(&mut t, accounts, 1000);
            accounts += 1;
        }
        for _ in 0..1000 {
            for i in 0..accounts {
                transfer(&mut t, cur_account, (cur_account + 1) % accounts, 1);
                cur_account = (cur_account + 1) % accounts;
            }
        }
        t.commit();
    }
}
