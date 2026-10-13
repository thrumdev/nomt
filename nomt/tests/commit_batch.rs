use nomt::{
    hasher::Blake3Hasher, trie::KeyPath, KeyReadWrite, Nomt, Options, Overlay, SessionParams, Value,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
};

type Db = Nomt<Blake3Hasher>;
type Ops = BTreeMap<KeyPath, Option<Value>>;
type Model = BTreeMap<KeyPath, Value>;

fn options(name: &str, rollback: bool) -> Options {
    let mut o = Options::new();
    o.path(PathBuf::from("test").join(name));
    o.commit_concurrency(1);
    o.bitbox_seed([0; 16]);
    o.hashtable_buckets(64_000);
    o.io_workers(1);
    o.rollback(rollback);
    o
}

fn open(name: &str, rollback: bool, clean: bool) -> Db {
    if clean {
        let _ = std::fs::remove_dir_all(PathBuf::from("test").join(name));
    }
    Nomt::open(options(name, rollback)).unwrap()
}

fn cluster_key(prefix: [u8; 2], i: u32) -> KeyPath {
    let mut k = *blake3::hash(&[&prefix[..], &i.to_le_bytes()[..]].concat()).as_bytes();
    k[0] = prefix[0];
    k[1] = prefix[1];
    k
}

fn random_key(i: u32) -> KeyPath {
    *blake3::hash(&[b"r".as_slice(), &i.to_le_bytes()[..]].concat()).as_bytes()
}

fn val(version: u32, i: u32, len: usize) -> Value {
    let seed = blake3::hash(&[version.to_le_bytes(), i.to_le_bytes()].concat());
    seed.as_bytes().iter().copied().cycle().take(len).collect()
}

const C1: [u8; 2] = [0xAA, 0x01];
const C2: [u8; 2] = [0xBB, 0x02];
const C3: [u8; 2] = [0xCC, 0x03];
const C4: [u8; 2] = [0xDD, 0x04];
const K1: [u8; 32] = [0x11; 32];

fn put(ops: &mut Ops, keys: impl IntoIterator<Item = KeyPath>, version: u32, len: usize) {
    for (i, k) in keys.into_iter().enumerate() {
        ops.insert(k, Some(val(version, i as u32, len)));
    }
}

fn del(ops: &mut Ops, keys: impl IntoIterator<Item = KeyPath>) {
    for k in keys {
        ops.insert(k, None);
    }
}

fn cluster(prefix: [u8; 2], range: std::ops::Range<u32>) -> impl Iterator<Item = KeyPath> {
    range.map(move |i| cluster_key(prefix, i))
}

fn randoms(range: std::ops::Range<u32>) -> impl Iterator<Item = KeyPath> {
    range.map(random_key)
}

fn base_ops() -> Ops {
    let mut o = Ops::new();
    put(&mut o, randoms(0..400), 0, 16);
    put(&mut o, cluster(C1, 0..64), 0, 16);
    put(&mut o, cluster(C2, 0..64), 0, 16);
    put(&mut o, cluster(C3, 0..40), 0, 16);
    o
}

/// Each overlay's writes. Covers: create, modify, existing->delete, delete->recreate,
/// delete->recreate->delete->recreate, create->delete (never on disk), create->delete->recreate,
/// elision of a stored subtree and its re-promotion, overflow values.
fn chain_ops() -> Vec<Ops> {
    let mut o1 = Ops::new();
    put(&mut o1, cluster(C4, 0..64), 1, 16);
    put(&mut o1, randoms(0..50), 1, 16);
    del(&mut o1, cluster(C1, 0..64));
    o1.insert(K1, Some(val(1, 0, 8)));
    put(&mut o1, randoms(100..110), 1, 5000);

    let mut o2 = Ops::new();
    put(&mut o2, randoms(0..50), 2, 16);
    put(&mut o2, cluster(C1, 0..64), 2, 16);
    put(&mut o2, cluster(C4, 0..64), 2, 16);
    o2.insert(K1, None);
    del(&mut o2, cluster(C3, 5..40));
    put(&mut o2, randoms(100..105), 2, 16);

    let mut o3 = Ops::new();
    del(&mut o3, cluster(C4, 0..64));
    del(&mut o3, cluster(C1, 0..64));
    put(&mut o3, cluster(C2, 0..64), 3, 16);
    o3.insert(K1, Some(val(3, 0, 8)));
    del(&mut o3, randoms(200..260));

    let mut o4 = Ops::new();
    put(&mut o4, cluster(C4, 0..64), 4, 16);
    put(&mut o4, randoms(0..50), 4, 16);
    put(&mut o4, cluster(C3, 5..40), 4, 16);
    put(&mut o4, randoms(200..230), 4, 16);

    let mut o5 = Ops::new();
    put(&mut o5, cluster(C1, 0..64), 5, 16);
    del(&mut o5, randoms(0..10));
    del(&mut o5, cluster(C2, 0..32));
    put(&mut o5, randoms(100..110), 5, 6000);

    let mut o6 = Ops::new();
    put(&mut o6, randoms(0..10), 6, 16);
    put(&mut o6, cluster(C2, 0..32), 6, 16);
    del(&mut o6, cluster(C4, 0..64));

    vec![o1, o2, o3, o4, o5, o6]
}

fn after_ops() -> Ops {
    let mut o = Ops::new();
    del(&mut o, cluster(C1, 0..64));
    put(&mut o, randoms(400..450), 7, 16);
    put(&mut o, cluster(C4, 0..64), 7, 16);
    o
}

fn universe() -> BTreeSet<KeyPath> {
    let mut u = BTreeSet::new();
    u.extend(randoms(0..450));
    for c in [C1, C2, C3, C4] {
        u.extend(cluster(c, 0..64));
    }
    u.insert(K1);
    u
}

fn apply(model: &mut Model, ops: &Ops) {
    for (k, v) in ops {
        match v {
            Some(v) => model.insert(*k, v.clone()),
            None => model.remove(k),
        };
    }
}

fn expected_root(model: &Model) -> [u8; 32] {
    nomt_core::update::build_trie::<Blake3Hasher>(
        0,
        model.iter().map(|(k, v)| (*k, *blake3::hash(v).as_bytes())),
        |_| {},
    )
}

fn actuals(ops: &Ops) -> Vec<(KeyPath, KeyReadWrite)> {
    ops.iter()
        .map(|(k, v)| (*k, KeyReadWrite::Write(v.clone())))
        .collect()
}

fn commit_session(db: &Db, ops: &Ops) {
    let session = db.begin_session(SessionParams::default());
    for k in ops.keys() {
        session.warm_up(*k);
    }
    session.finish(actuals(ops)).unwrap().commit(db).unwrap();
}

fn build_chain(db: &Db, chain: &[Ops]) -> Vec<Overlay> {
    let mut overlays: Vec<Overlay> = Vec::new();
    for ops in chain {
        let params = SessionParams::default()
            .overlay(overlays.iter().rev())
            .unwrap();
        let session = db.begin_session(params);
        for k in ops.keys() {
            session.warm_up(*k);
        }
        overlays.push(session.finish(actuals(ops)).unwrap().into_overlay());
    }
    overlays
}

fn check(db: &Db, model: &Model, label: &str) {
    assert_eq!(
        db.root().into_inner(),
        expected_root(model),
        "{label}: root"
    );
    for k in universe() {
        assert_eq!(
            db.read(k).unwrap(),
            model.get(&k).cloned(),
            "{label}: key {k:?}"
        );
    }
}

fn occupied(db: &Db) -> usize {
    db.hash_table_utilization().occupied
}

/// Returns (db1 = per-overlay commit, db2 = commit_batch, final model, occupancy trace of db1).
fn setup_pair(prefix: &str, rollback: bool) -> (Db, Db, Model, Vec<usize>) {
    let chain = chain_ops();
    let db1 = open(&format!("{prefix}_single"), rollback, true);
    let db2 = open(&format!("{prefix}_batch"), rollback, true);
    let mut model = Model::new();
    apply(&mut model, &base_ops());
    commit_session(&db1, &base_ops());
    commit_session(&db2, &base_ops());
    assert_eq!(db1.root(), db2.root());
    assert_eq!(occupied(&db1), occupied(&db2));

    let chain1 = build_chain(&db1, &chain);
    let chain2 = build_chain(&db2, &chain);
    for (a, b) in chain1.iter().zip(chain2.iter()) {
        assert_eq!(a.root(), b.root());
    }

    let mut trace = vec![occupied(&db1)];
    for overlay in chain1 {
        overlay.commit(&db1).unwrap();
        trace.push(occupied(&db1));
    }
    Overlay::commit_batch(chain2, &db2).unwrap();

    for ops in &chain {
        apply(&mut model, ops);
    }
    (db1, db2, model, trace)
}

#[test]
fn commit_batch_matches_sequential_commits() {
    let (db1, db2, mut model, trace) = setup_pair("commit_batch_eq", false);

    // The workload must actually clear and recreate stored pages.
    assert!(
        trace.windows(2).any(|w| w[1] < w[0]),
        "no page was cleared: {trace:?}"
    );
    assert!(
        trace.windows(2).skip(1).any(|w| w[1] > w[0]),
        "no page recreated: {trace:?}"
    );

    check(&db1, &model, "db1");
    check(&db2, &model, "db2");
    assert_eq!(occupied(&db1), occupied(&db2), "occupied buckets");

    drop(db1);
    drop(db2);
    let v1 = nomt::validate_hashtable(&options("commit_batch_eq_single", false)).unwrap();
    let v2 = nomt::validate_hashtable(&options("commit_batch_eq_batch", false)).unwrap();
    assert_eq!(v1.occupied, v2.occupied, "validated occupancy");

    let db1 = open("commit_batch_eq_single", false, false);
    let db2 = open("commit_batch_eq_batch", false, false);
    check(&db1, &model, "db1 reopened");
    check(&db2, &model, "db2 reopened");
    assert_eq!(occupied(&db1), occupied(&db2));

    let after = after_ops();
    commit_session(&db1, &after);
    commit_session(&db2, &after);
    apply(&mut model, &after);
    check(&db1, &model, "db1 after");
    check(&db2, &model, "db2 after");
    assert_eq!(occupied(&db1), occupied(&db2));

    drop(db1);
    drop(db2);
    let db1 = open("commit_batch_eq_single", false, false);
    let db2 = open("commit_batch_eq_batch", false, false);
    check(&db1, &model, "db1 after reopened");
    check(&db2, &model, "db2 after reopened");
}

#[test]
fn commit_batch_rollback_matches_sequential_commits() {
    let chain = chain_ops();
    let mut models = vec![Model::new()];
    apply(&mut models[0], &base_ops());
    for ops in &chain {
        let mut m = models.last().unwrap().clone();
        apply(&mut m, ops);
        models.push(m);
    }

    for n in 1..=chain.len() {
        let prefix = format!("commit_batch_rb_{n}");
        let (db1, db2, model, _) = setup_pair(&prefix, true);
        check(&db1, &model, "db1 pre-rollback");
        check(&db2, &model, "db2 pre-rollback");

        db1.rollback(n).unwrap();
        db2.rollback(n).unwrap();
        assert_eq!(db1.root(), db2.root(), "rollback {n}: roots differ");
        let target = &models[chain.len() - n];
        check(&db1, target, &format!("db1 rollback {n}"));
        check(&db2, target, &format!("db2 rollback {n}"));
        assert_eq!(occupied(&db1), occupied(&db2), "rollback {n}: occupancy");

        drop(db1);
        drop(db2);
        let db1 = open(&format!("{prefix}_single"), true, false);
        let db2 = open(&format!("{prefix}_batch"), true, false);
        check(&db1, target, &format!("db1 rollback {n} reopened"));
        check(&db2, target, &format!("db2 rollback {n} reopened"));
    }

    // Rollback of the full chain after a reopen uses the persisted log.
    let (db1, db2, _, _) = setup_pair("commit_batch_rb_reopen", true);
    drop(db1);
    drop(db2);
    let db1 = open("commit_batch_rb_reopen_single", true, false);
    let db2 = open("commit_batch_rb_reopen_batch", true, false);
    db1.rollback(chain.len()).unwrap();
    db2.rollback(chain.len()).unwrap();
    check(&db1, &models[0], "db1 reopen+rollback");
    check(&db2, &models[0], "db2 reopen+rollback");
}

#[test]
fn commit_batch_rollback_log_shorter_than_chain() {
    let chain = chain_ops();
    let mut models = vec![Model::new()];
    apply(&mut models[0], &base_ops());
    for ops in &chain {
        let mut m = models.last().unwrap().clone();
        apply(&mut m, ops);
        models.push(m);
    }
    let names = [
        "commit_batch_rb_short_single",
        "commit_batch_rb_short_batch",
    ];
    let dbs: Vec<Db> = names
        .iter()
        .map(|n| {
            let _ = std::fs::remove_dir_all(PathBuf::from("test").join(n));
            let mut o = options(n, true);
            o.max_rollback_log_len(3);
            Nomt::open(o).unwrap()
        })
        .collect();
    for db in &dbs {
        commit_session(db, &base_ops());
    }
    let chain1 = build_chain(&dbs[0], &chain);
    let chain2 = build_chain(&dbs[1], &chain);
    for o in chain1 {
        o.commit(&dbs[0]).unwrap();
    }
    Overlay::commit_batch(chain2, &dbs[1]).unwrap();
    for (db, n) in dbs.iter().zip(names) {
        assert!(db.rollback(4).is_err(), "{n}: live rollback(4) should fail");
    }
    drop(dbs);

    // Expired deltas must not come back after reopening.
    for n in names {
        let mut o = options(n, true);
        o.max_rollback_log_len(3);
        let db = Nomt::open(o).unwrap();
        assert!(
            db.rollback(4).is_err(),
            "{n}: reopened rollback(4) should fail"
        );
        db.rollback(3).unwrap();
        check(&db, &models[chain.len() - 3], n);
    }
}

#[test]
fn commit_batch_rejects_invalid_chain() {
    let db = open("commit_batch_invalid", false, true);
    commit_session(&db, &base_ops());
    let root0 = db.root();
    let chain = chain_ops();

    // Out of order.
    let mut overlays = build_chain(&db, &chain[..3]);
    overlays.swap(1, 2);
    assert!(Overlay::commit_batch(overlays, &db).is_err());
    assert_eq!(db.root(), root0);
    assert!(!db.is_poisoned());

    // Gap: [o1, o3] while o2 is held outside. Nothing may be marked committed:
    // o2's parent (o1) must not appear committed afterwards.
    let mut overlays = build_chain(&db, &chain[..3]);
    let o3 = overlays.pop().unwrap();
    let o2 = overlays.pop().unwrap();
    let o1 = overlays.pop().unwrap();
    assert!(Overlay::commit_batch(vec![o1, o3], &db).is_err());
    assert_eq!(db.root(), root0);
    assert!(!db.is_poisoned());
    assert!(SessionParams::default().overlay([&o2]).is_err());
    assert!(o2.commit(&db).is_err());
    assert_eq!(db.root(), root0);

    // Overlay from a different fork (not a chain): o1 and an unrelated root-based overlay.
    let mut overlays = build_chain(&db, &chain[..1]);
    let unrelated = build_chain(&db, &chain[1..2]).pop().unwrap();
    overlays.push(unrelated);
    assert!(Overlay::commit_batch(overlays, &db).is_err());
    assert_eq!(db.root(), root0);
    assert!(!db.is_poisoned());

    // A valid batch still works afterwards.
    let overlays = build_chain(&db, &chain);
    Overlay::commit_batch(overlays, &db).unwrap();
    let mut model = Model::new();
    apply(&mut model, &base_ops());
    for ops in &chain {
        apply(&mut model, ops);
    }
    check(&db, &model, "after rejected batches");
}
