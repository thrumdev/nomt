use fxhash::FxHashMap;
use nomt::{KeyPath, KeyReadWrite, Node, Nomt, Options, Session};
use std::{collections::hash_map::Entry, mem, path::PathBuf, rc::Rc};

fn path(id: u64) -> KeyPath {
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
        path: PathBuf::from("test"),
        fetch_concurrency: 1,
        traversal_concurrency: 1,
    }
}

struct Test {
    nomt: Nomt,
    session: Option<Session>,
    access: FxHashMap<KeyPath, KeyReadWrite>,
}

impl Test {
    fn new() -> Self {
        let _ = std::fs::remove_dir_all("test");
        let nomt = Nomt::open(opts()).unwrap();
        let session = nomt.begin_session();
        Self {
            nomt,
            session: Some(session),
            access: FxHashMap::default(),
        }
    }

    fn write(&mut self, id: u64, value: Option<u64>) {
        let path = path(id);
        let value = value.map(|v| Rc::new(v.to_le_bytes().to_vec()));
        let delete = value.is_none();
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
            .tentative_write_slot(path, delete);
    }

    fn read(&mut self, id: u64) -> Option<u64> {
        let path = path(id);
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

    fn commit(&mut self) -> Node {
        let session = mem::take(&mut self.session).unwrap();
        let mut actual_access: Vec<_> = mem::take(&mut self.access).into_iter().collect();
        actual_access.sort_by_key(|(k, _)| *k);
        self.nomt.commit_and_prove(session, actual_access).unwrap();
        self.session = Some(self.nomt.begin_session());
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

struct Timer {
    name: &'static str,
    h: hdrhistogram::Histogram<u64>,
    ops: u64,
}

impl Timer {
    fn new(name: &'static str) -> Self {
        Self {
            name,
            h: hdrhistogram::Histogram::<u64>::new(3).unwrap(),
            ops: 0,
        }
    }

    fn record<'a>(&'a mut self) -> impl Drop + 'a {
        struct Record<'a> {
            h: &'a mut hdrhistogram::Histogram<u64>,
            ops: &'a mut u64,
            start: std::time::Instant,
        }
        impl Drop for Record<'_> {
            fn drop(&mut self) {
                let elapsed = self.start.elapsed().as_nanos() as u64;
                self.h.record(elapsed).unwrap();
                *self.ops += 1;
            }
        }
        Record {
            h: &mut self.h,
            ops: &mut self.ops,
            start: std::time::Instant::now(),
        }
    }

    fn print(&mut self) {
        println!("{}:", self.name);
        println!("  ops={}", self.ops);
        for q in [0.001, 0.01, 0.25, 0.50, 0.75, 0.95, 0.99, 0.999] {
            let lat = self.h.value_at_quantile(q);
            println!("  {}th: {}", q * 100.0, pretty_display_ns(lat));
        }
        println!("  mean={}", pretty_display_ns(self.h.mean() as u64));
        println!();
        self.ops = 0;
    }
}

fn pretty_display_ns(ns: u64) -> String {
    // preserve 3 sig figs at minimum.
    let (val, unit) = if ns > 100 * 1_000_000_000 {
        (ns / 1_000_000_000, "s")
    } else if ns > 100 * 1_000_000 {
        (ns / 1_000_000, "ms")
    } else if ns > 100 * 1_000 {
        (ns / 1_000, "us")
    } else {
        (ns, "ns")
    };

    format!("{val} {unit}")
}

fn main() {
    let mut t = Test::new();
    let mut accounts = 0;
    let mut cur_account = 0;
    let mut t_transfer = Timer::new("transfer");
    let mut t_commit = Timer::new("commit");

    let mut now = std::time::Instant::now();
    loop {
        for _ in 0..100 {
            set_balance(&mut t, accounts, 1000);
            accounts += 1;
        }
        for _ in 0..1000 {
            for _ in 0..accounts {
                {
                    let _guard = t_transfer.record();
                    transfer(&mut t, cur_account, (cur_account + 1) % accounts, 1);
                }
                cur_account = (cur_account + 1) % accounts;
            }
        }
        {
            let _guard = t_commit.record();
            t.commit();
        }

        // Every second print the latency stats.
        if now.elapsed().as_secs() >= 1 {
            t_transfer.print();
            t_commit.print();
            now = std::time::Instant::now();
        }
    }
}
