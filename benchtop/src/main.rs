use nomt::{KeyPath, Node, Nomt, Options, Session};
use std::{mem, path::PathBuf, rc::Rc};

fn path(id: u64) -> KeyPath {
    // KeyPaths must be uniformly distributed, but we don't want to spend time on a good hash. So
    // the next best option is to use a PRNG seeded with the id.
    use rand::{RngCore as _, SeedableRng as _};
    let mut seed = [0; 16];
    seed[0..8].copy_from_slice(&id.to_le_bytes());
    let mut rng = rand_pcg::Lcg64Xsh32::from_seed(seed);
    let mut path = KeyPath::default();
    for i in 0..4 {
        path[i] = rng.next_u32() as u8;
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
}

impl Test {
    fn new() -> Self {
        let _ = std::fs::remove_dir_all("test");
        let nomt = Nomt::open(opts()).unwrap();
        let session = nomt.begin_session();
        Self {
            nomt,
            session: Some(session),
        }
    }

    fn write(&mut self, id: u64, value: Option<u64>) {
        let path = path(id);
        let value = value.map(|v| Rc::new(v.to_le_bytes().to_vec()));
        self.session.as_mut().unwrap().write_slot(path, value);
    }

    fn read(&mut self, id: u64) -> Option<u64> {
        let path = path(id);
        let to_u64 = |v: Option<&[u8]>| v.map(|v| u64::from_le_bytes(v.try_into().unwrap()));
        let value = self.session.as_mut().unwrap().read_slot(path).unwrap();
        to_u64(value.as_ref().map(|v| &v[..]))
    }

    fn commit(&mut self) -> Node {
        let session = mem::take(&mut self.session).unwrap();
        self.nomt.commit_and_prove(session).unwrap();
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
            h: hdrhistogram::Histogram::<u64>::new_with_bounds(1, 1_000_000, 3).unwrap(),
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
                let elapsed = self.start.elapsed().as_micros() as u64;
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
            println!("  {}th: {} ns", q * 100.0, lat);
        }
        println!("  mean={} ns", self.h.mean());
        println!();
        self.ops = 0;
    }
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
