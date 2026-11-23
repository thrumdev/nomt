//! Shared harness for fuzz targets that drive one database from several threads at once.
//!
//! Each round releases a few threads together. Every thread runs its own script of sessions
//! (reads, nested sessions, `Nomt::read`, proofs and writes) that end in a commit, an overlay
//! commit or a drop, plus rollbacks and reads outside any session. NOMT lets any number of
//! sessions live together while a commit or rollback waits for all of them, so these interleave
//! across threads. A thread never commits or rolls back while holding a live session itself: that
//! would wait on its own session forever.
//!
//! Nothing can change under a live session, so all of its reads and proofs are checked against
//! the state its `prev_root` names. A commit must succeed exactly when the root hasn't moved
//! since its session began, so the successful commits and rollbacks form one chain of roots from
//! the empty trie to the current root.

use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    path::Path,
    sync::{Arc, Barrier, Mutex},
    time::Duration,
};

use arbitrary::{Result, Unstructured};
use bitvec::{order::Msb0, view::BitView};
use nomt::{
    hasher::{Blake3Hasher, ValueHasher},
    trie::{KeyPath, LeafData, TERMINATOR},
    FinishedSession, KeyReadWrite, Nomt, Options, SessionParams, Value, WitnessMode,
};
use nomt_core::update::build_trie;

// Each target uses one mode.
#[allow(dead_code)]
pub enum Mode {
    /// Commits race each other for the write lock. No rollbacks are issued.
    Racing,
    /// Rollbacks are issued too. To know what each one undoes, commits and rollbacks queue on a
    /// harness lock first, which they take only while their thread holds no session.
    Ordered,
}

type Model = BTreeMap<KeyPath, Value>;
type Root = [u8; 32];

pub fn run(data: &[u8], mode: Mode) {
    let mut u = Unstructured::new(data);
    let Ok(plan) = Plan::arbitrary(&mut u, &mode) else {
        return;
    };
    let tempdir = tempfile::tempdir().unwrap();
    let path = tempdir.path().join("db");

    let tracker = Tracker {
        states: Mutex::new(HashMap::from([(TERMINATOR, Arc::new(Model::new()))])),
        transitions: Mutex::new(Vec::new()),
        stale: Mutex::new(Vec::new()),
        history: matches!(mode, Mode::Ordered).then(|| {
            Mutex::new(History {
                roots: VecDeque::from([TERMINATOR]),
                cap: plan.config.rollback_log_len as usize,
            })
        }),
    };

    let mut db = plan.config.open(&path);
    for round in &plan.rounds {
        match round {
            Round::Reopen => {
                let root = db.root();
                drop(db);
                db = plan.config.open(&path);
                assert_eq!(db.root(), root, "root changed across reopen");
            }
            Round::Threads(threads) => {
                let ctx = Ctx {
                    db: &db,
                    keys: &plan.keys,
                    tracker: &tracker,
                };
                let barrier = Barrier::new(threads.len());
                std::thread::scope(|s| {
                    for ops in threads {
                        let (ctx, barrier) = (&ctx, &barrier);
                        s.spawn(move || {
                            barrier.wait();
                            for op in ops {
                                ctx.op(op);
                            }
                        });
                    }
                });
            }
        }
        tracker.check(&db, &plan.keys);
    }
}

/// What the harness knows about the database, shared by all threads.
struct Tracker {
    /// Every state a finished session produced, by root. Includes uncommitted ones.
    states: Mutex<HashMap<Root, Arc<Model>>>,
    /// Successful commits and rollbacks, as (from, to) roots, in no particular order.
    transitions: Mutex<Vec<(Root, Root)>>,
    /// Previous roots of commits rejected as stale.
    stale: Mutex<Vec<Root>>,
    /// `Ordered` mode only: also the commit/rollback lock.
    history: Option<Mutex<History>>,
}

/// The roots the rollback log can still return to, oldest first, ending with the current root.
struct History {
    roots: VecDeque<Root>,
    cap: usize,
}

impl History {
    fn current(&self) -> Root {
        *self.roots.back().unwrap()
    }

    fn logged(&self) -> usize {
        self.roots.len() - 1
    }

    fn push(&mut self, root: Root) {
        self.roots.push_back(root);
        while self.logged() > self.cap {
            self.roots.pop_front();
        }
    }
}

impl Tracker {
    fn state(&self, root: &Root) -> Arc<Model> {
        match self.states.lock().unwrap().get(root) {
            Some(model) => model.clone(),
            None => panic!("database is at root {root:02x?}, which no session produced"),
        }
    }

    fn insert_state(&self, root: Root, model: Model) {
        let mut states = self.states.lock().unwrap();
        if let Some(existing) = states.get(&root) {
            assert_eq!(**existing, model, "two different states share a root");
        } else {
            states.insert(root, Arc::new(model));
        }
    }

    /// Single-threaded checks between rounds.
    fn check(&self, db: &Nomt<Blake3Hasher>, keys: &[KeyPath]) {
        let root = db.root().into_inner();
        let model = self.state(&root);
        for key in keys {
            assert_eq!(
                db.read(*key).unwrap().as_ref(),
                model.get(key),
                "key {key:02x?}"
            );
        }
        if let Some(history) = &self.history {
            assert_eq!(root, history.lock().unwrap().current());
        }

        // The successful commits and rollbacks happened one at a time, each starting from the root
        // the last one left. So they form a single walk from the empty trie to `root` that uses
        // every transition once, and every node's out-degree minus in-degree is fixed.
        let mut balance = HashMap::<Root, i64>::new();
        for (from, to) in self.transitions.lock().unwrap().iter() {
            *balance.entry(*from).or_default() += 1;
            *balance.entry(*to).or_default() -= 1;
        }
        *balance.entry(TERMINATOR).or_default() -= 1;
        *balance.entry(root).or_default() += 1;
        for (node, b) in &balance {
            assert_eq!(
                *b, 0,
                "commits and rollbacks don't chain at root {node:02x?}"
            );
        }

        // A commit is only stale if something moved the root off its previous root.
        let transitions = self.transitions.lock().unwrap();
        for prev in self.stale.lock().unwrap().iter() {
            assert!(
                transitions
                    .iter()
                    .any(|(from, to)| from == prev && to != prev),
                "commit from {prev:02x?} was rejected although the root never moved off it"
            );
        }
    }
}

struct Ctx<'a> {
    db: &'a Nomt<Blake3Hasher>,
    keys: &'a [KeyPath],
    tracker: &'a Tracker,
}

impl Ctx<'_> {
    fn op(&self, op: &Op) {
        match op {
            Op::Session(plan) => self.session(plan),
            Op::Rollback(raw_n) => self.rollback(*raw_n),
            Op::Read(k) => {
                // Without a session the state can move at any time, so only check that the value
                // is one some state had.
                let key = self.keys[*k];
                let value = self.db.read(key).unwrap();
                let states = self.tracker.states.lock().unwrap();
                assert!(
                    states.values().any(|m| m.get(&key) == value.as_ref()),
                    "Nomt::read returned a value no state had"
                );
            }
            Op::Pause(us) => pause(*us),
        }
    }

    fn session(&self, plan: &SessionPlan) {
        let witness = if plan.witness {
            WitnessMode::read_write()
        } else {
            WitnessMode::disabled()
        };
        let session = self
            .db
            .begin_session(SessionParams::default().witness_mode(witness));
        let prev = session.prev_root().into_inner();
        let base = self.tracker.state(&prev);
        let mut actuals = BTreeMap::<KeyPath, KeyReadWrite>::new();

        for step in &plan.steps {
            match step {
                Step::Read(k) => {
                    let key = self.keys[*k];
                    session.warm_up(key);
                    let value = session.read(key).unwrap();
                    assert_eq!(value.as_ref(), base.get(&key), "session read");
                    actuals.entry(key).or_insert(KeyReadWrite::Read(value));
                }
                Step::Write(k, value, preserve) => {
                    let key = self.keys[*k];
                    session.warm_up(key);
                    if *preserve {
                        session.preserve_prior_value(key);
                    }
                    let entry = match actuals.remove(&key) {
                        Some(KeyReadWrite::Read(old) | KeyReadWrite::ReadThenWrite(old, _)) => {
                            KeyReadWrite::ReadThenWrite(old, value.clone())
                        }
                        _ => KeyReadWrite::Write(value.clone()),
                    };
                    actuals.insert(key, entry);
                }
                Step::Prove(k) => {
                    let key = self.keys[*k];
                    session.warm_up(key);
                    let proof = session.prove(key).unwrap();
                    let verified = proof
                        .verify::<Blake3Hasher>(key.view_bits::<Msb0>(), prev)
                        .expect("session proof doesn't verify against its prev root");
                    let confirmed = match base.get(&key) {
                        Some(value) => verified.confirm_value(&LeafData {
                            key_path: key,
                            value_hash: Blake3Hasher::hash_value(value),
                        }),
                        None => verified.confirm_nonexistence(&key),
                    };
                    assert!(confirmed.unwrap(), "session proof disagrees with the state");
                }
                Step::DbRead(k) => {
                    // Takes the read lock recursively, past any queued commit.
                    let key = self.keys[*k];
                    let value = self.db.read(key).unwrap();
                    assert_eq!(value.as_ref(), base.get(&key), "Nomt::read under a session");
                }
                Step::Nested(ks) => {
                    let nested = self.db.begin_session(SessionParams::default());
                    assert_eq!(nested.prev_root().into_inner(), prev, "nested session root");
                    for k in ks {
                        let key = self.keys[*k];
                        assert_eq!(
                            nested.read(key).unwrap().as_ref(),
                            base.get(&key),
                            "nested session read"
                        );
                    }
                }
                Step::Pause(us) => pause(*us),
            }
        }

        let (pause_us, how) = match plan.end {
            End::Drop => return,
            End::FinishAndDrop => {
                let _ = session.finish(actuals.into_iter().collect()).unwrap();
                return;
            }
            End::Commit(pause_us, how) => (pause_us, how),
        };

        let mut model = (*base).clone();
        for (key, rw) in &actuals {
            if let KeyReadWrite::Write(value) | KeyReadWrite::ReadThenWrite(_, value) = rw {
                match value {
                    Some(value) => model.insert(*key, value.clone()),
                    None => model.remove(key),
                };
            }
        }
        let mut finished = session.finish(actuals.into_iter().collect()).unwrap();
        let root = finished.root().into_inner();
        assert_eq!(finished.prev_root().into_inner(), prev);
        assert_eq!(root, root_of(&model), "finished session root");
        if plan.witness {
            assert!(finished.take_witness().is_some(), "witness missing");
        }
        self.tracker.insert_state(root, model);

        // The session's read guard is gone, so other threads can commit in this window.
        pause(pause_us);
        self.commit(prev, root, finished, how);
    }

    fn commit(&self, prev: Root, root: Root, finished: FinishedSession, how: How) {
        let history = self.tracker.history.as_ref().map(|h| h.lock().unwrap());
        let result = match how {
            How::Blocking => finished.commit(self.db),
            How::Nonblocking => match finished.try_commit_nonblocking(self.db) {
                Ok(Some(finished)) => finished.commit(self.db),
                Ok(None) => Ok(()),
                Err(e) => Err(e),
            },
            How::Overlay => finished.into_overlay().commit(self.db),
        };
        if let Some(history) = &history {
            // No one else can commit or roll back while we hold the history lock.
            assert_eq!(
                result.is_ok(),
                history.current() == prev,
                "commit result {result:?} disagrees with whether the root moved"
            );
        }
        match result {
            Ok(()) => {
                self.tracker.transitions.lock().unwrap().push((prev, root));
                if let Some(mut history) = history {
                    assert_eq!(self.db.root().into_inner(), root);
                    history.push(root);
                }
            }
            Err(e) => {
                assert!(
                    e.to_string().contains("no longer valid"),
                    "commit failed: {e:?}"
                );
                self.tracker.stale.lock().unwrap().push(prev);
            }
        }
    }

    fn rollback(&self, raw_n: u8) {
        let Some(history) = &self.tracker.history else {
            return;
        };
        let mut history = history.lock().unwrap();
        let logged = history.logged();
        // 1 past what's logged must fail.
        let n = raw_n as usize % (logged + 1) + 1;
        let result = self.db.rollback(n);
        if n > logged {
            assert!(result.is_err(), "rollback({n}) with only {logged} logged");
            return;
        }
        result.unwrap();
        let from = history.current();
        for _ in 0..n {
            history.roots.pop_back();
        }
        let to = history.current();
        assert_eq!(
            self.db.root().into_inner(),
            to,
            "rollback({n}) landed elsewhere"
        );
        self.tracker.transitions.lock().unwrap().push((from, to));
    }
}

fn root_of(model: &Model) -> Root {
    build_trie::<Blake3Hasher>(
        0,
        model.iter().map(|(k, v)| (*k, Blake3Hasher::hash_value(v))),
        |_| {},
    )
}

fn pause(us: u16) {
    if us == 0 {
        std::thread::yield_now();
    } else {
        std::thread::sleep(Duration::from_micros(us as u64));
    }
}

struct Config {
    commit_concurrency: usize,
    io_workers: usize,
    warm_up: bool,
    rollback: bool,
    rollback_log_len: u32,
}

impl Config {
    fn open(&self, path: &Path) -> Nomt<Blake3Hasher> {
        let mut options = Options::new();
        options.path(path);
        options.commit_concurrency(self.commit_concurrency);
        options.io_workers(self.io_workers);
        options.warm_up(self.warm_up);
        options.rollback(self.rollback);
        options.max_rollback_log_len(self.rollback_log_len);
        options.hashtable_buckets(4096);
        options.preallocate_ht(false);
        Nomt::open(options).unwrap()
    }
}

struct Plan {
    config: Config,
    keys: Vec<KeyPath>,
    rounds: Vec<Round>,
}

enum Round {
    Threads(Vec<Vec<Op>>),
    Reopen,
}

enum Op {
    Session(SessionPlan),
    Rollback(u8),
    /// `Nomt::read` outside any session.
    Read(usize),
    Pause(u16),
}

struct SessionPlan {
    witness: bool,
    steps: Vec<Step>,
    end: End,
}

enum Step {
    Read(usize),
    /// Key, value, whether to call `preserve_prior_value` first.
    Write(usize, Option<Value>, bool),
    Prove(usize),
    /// `Nomt::read` while this thread holds the session.
    DbRead(usize),
    /// Open a second session on this thread and read through it.
    Nested(Vec<usize>),
    Pause(u16),
}

#[derive(Clone, Copy)]
enum End {
    Drop,
    FinishAndDrop,
    /// Pause after finishing, then commit.
    Commit(u16, How),
}

#[derive(Clone, Copy)]
enum How {
    Blocking,
    Nonblocking,
    Overlay,
}

impl Plan {
    fn arbitrary(u: &mut Unstructured, mode: &Mode) -> Result<Self> {
        let ordered = matches!(mode, Mode::Ordered);
        let config = Config {
            commit_concurrency: u.int_in_range(1..=4)?,
            io_workers: u.int_in_range(1..=3)?,
            warm_up: u.arbitrary()?,
            rollback: ordered || u.arbitrary()?,
            rollback_log_len: if ordered { u.int_in_range(1..=8)? } else { 16 },
        };

        let mut keys = Vec::new();
        for _ in 0..u.int_in_range(1..=24)? {
            keys.push(new_key(u, &keys)?);
        }

        let mut rounds = Vec::new();
        for _ in 0..u.int_in_range(1..=6)? {
            if u.ratio(1, 8)? {
                rounds.push(Round::Reopen);
                continue;
            }
            let mut threads = Vec::new();
            for _ in 0..u.int_in_range(2..=4)? {
                let mut ops = Vec::new();
                for _ in 0..u.int_in_range(1..=6)? {
                    ops.push(op(u, keys.len(), ordered)?);
                }
                threads.push(ops);
            }
            rounds.push(Round::Threads(threads));
        }

        Ok(Plan {
            config,
            keys,
            rounds,
        })
    }
}

fn op(u: &mut Unstructured, n_keys: usize, ordered: bool) -> Result<Op> {
    Ok(match u.int_in_range(0..=9)? {
        0 if ordered => Op::Rollback(u.arbitrary()?),
        1 if ordered => Op::Rollback(u.arbitrary()?),
        2 => Op::Read(u.choose_index(n_keys)?),
        3 => Op::Pause(pause_us(u)?),
        _ => Op::Session(session_plan(u, n_keys)?),
    })
}

fn session_plan(u: &mut Unstructured, n_keys: usize) -> Result<SessionPlan> {
    let witness = u.ratio(1, 4)?;
    let mut steps = Vec::new();
    for _ in 0..u.int_in_range(0..=12)? {
        let k = u.choose_index(n_keys)?;
        steps.push(match u.int_in_range(0..=11)? {
            0..=3 => Step::Write(k, value(u)?, u.ratio(1, 4)?),
            4..=6 => Step::Read(k),
            7 => Step::Prove(k),
            8 => Step::DbRead(k),
            9 => {
                let mut ks = vec![k];
                for _ in 0..u.int_in_range(0..=3)? {
                    ks.push(u.choose_index(n_keys)?);
                }
                Step::Nested(ks)
            }
            _ => Step::Pause(pause_us(u)?),
        });
    }
    let end = match u.int_in_range(0..=9)? {
        0 => End::Drop,
        1 => End::FinishAndDrop,
        2 | 3 => End::Commit(pause_us(u)?, How::Nonblocking),
        4 | 5 => End::Commit(pause_us(u)?, How::Overlay),
        _ => End::Commit(pause_us(u)?, How::Blocking),
    };
    Ok(SessionPlan {
        witness,
        steps,
        end,
    })
}

fn pause_us(u: &mut Unstructured) -> Result<u16> {
    Ok(match u.int_in_range(0..=3)? {
        0 => 0,
        1 => u.int_in_range(1..=50)?,
        _ => u.int_in_range(50..=500)?,
    })
}

fn value(u: &mut Unstructured) -> Result<Option<Value>> {
    if u.ratio(1, 5)? {
        return Ok(None);
    }
    // Sizes around the point where values move out of leaf pages into overflow pages.
    let len = match u.int_in_range(0..=4)? {
        0 => u.int_in_range(0..=32)?,
        1 => u.int_in_range(33..=1331)?,
        2 => u.int_in_range(1332..=1334)?,
        3 => u.int_in_range(1335..=4096)?,
        _ => u.int_in_range(4097..=16384)?,
    };
    Ok(Some(vec![u.arbitrary::<u8>()?; len]))
}

/// Mostly derive keys from existing ones, diverging at a chosen bit, so the trie gets deep.
fn new_key(u: &mut Unstructured, existing: &[KeyPath]) -> Result<KeyPath> {
    let mut key: KeyPath = u.arbitrary()?;
    if !existing.is_empty() && u.ratio(3, 4)? {
        let base = existing[u.choose_index(existing.len())?];
        let diverge_at = u.int_in_range(0..=255)?;
        let bits = key.view_bits_mut::<Msb0>();
        bits[..diverge_at].copy_from_bitslice(&base.view_bits::<Msb0>()[..diverge_at]);
        let flipped = !base.view_bits::<Msb0>()[diverge_at];
        bits.set(diverge_at, flipped);
    }
    Ok(key)
}
