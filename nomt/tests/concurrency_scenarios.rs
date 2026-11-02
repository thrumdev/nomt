//! Small concurrency scenarios meant to be run one at a time under a schedule explorer:
//! - FFS: `ffs explore -- <test-binary> --ignored --exact <name> --test-threads=1`. All are
//!   `#[ignore]`d so that plain `cargo test` is unaffected.
//! - shuttle: built with `--cfg shuttle` and the shims from `nomt-shuttle-shims`, each scenario
//!   runs under `shuttle::check_pct` (or `check_random` with `SHUTTLE_SCHED=random`) for
//!   `SHUTTLE_ITERS` iterations.

mod common;

use bitvec::prelude::*;
use common::{account_path, Test};
use nomt::{
    hasher::Blake3Hasher, proof, trie::KeyPath, trie::LeafData, KeyReadWrite, Nomt, Options,
    Root, SessionParams,
};
#[cfg(shuttle)]
use shuttle::thread;
use std::{path::Path, sync::Arc};
#[cfg(not(shuttle))]
use std::{sync::mpsc, thread, time::Duration};

type Db = Nomt<Blake3Hasher>;

#[cfg(not(shuttle))]
/// Run `f` on a fresh thread and panic if it doesn't finish within `secs`, so a deadlock
/// surfaces as a failure instead of a hung test binary.
/// `CS_NO_DEADLINE=1` runs `f` inline instead, leaving hang detection to the explorer.
fn with_deadline(secs: u64, f: impl FnOnce() + Send + 'static) {
    if std::env::var_os("CS_NO_DEADLINE").is_some() {
        return f();
    }
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        f();
        let _ = tx.send(());
    });
    match rx.recv_timeout(Duration::from_secs(secs)) {
        Ok(()) => {}
        Err(mpsc::RecvTimeoutError::Timeout) => panic!("scenario did not finish in {secs}s"),
        Err(mpsc::RecvTimeoutError::Disconnected) => panic!("scenario panicked"),
    }
}

fn open(path: &Path, configure: impl FnOnce(&mut Options)) -> Db {
    let mut o = Options::new();
    o.path(path);
    o.bitbox_seed([0; 16]);
    o.hashtable_buckets(4096);
    o.io_workers(1);
    o.commit_concurrency(1);
    configure(&mut o);
    Nomt::open(o).unwrap()
}

fn value(round: u64) -> Vec<u8> {
    round.to_le_bytes().to_vec()
}

fn write_all(keys: &[KeyPath], round: u64) -> Vec<(KeyPath, KeyReadWrite)> {
    let mut actuals: Vec<_> = keys
        .iter()
        .map(|k| (*k, KeyReadWrite::Write(Some(value(round)))))
        .collect();
    actuals.sort_by_key(|(k, _)| *k);
    actuals
}

fn commit(db: &Db, actuals: Vec<(KeyPath, KeyReadWrite)>) -> Root {
    let session = db.begin_session(SessionParams::default());
    for (k, _) in &actuals {
        session.warm_up(*k);
        session.preserve_prior_value(*k);
    }
    let finished = session.finish(actuals).unwrap();
    let root = finished.root();
    finished.commit(db).unwrap();
    root
}

fn keys(n: u64) -> Vec<KeyPath> {
    (0..n).map(account_path).collect()
}

/// Positive control: witness operations must refer to the path proof covering their key when
/// several commit workers finish in arbitrary order (fixed by b6ff34d, not on master).
fn scenario_witness_concurrent_workers() {
    let mut t = Test::new_with_params(common::fresh_test_name("cs_witness"), 4, 4096, None, true);
    for i in 0..200 {
        common::set_balance(&mut t, i, 1000);
    }
    let (mut prev_root, _) = t.commit();

    for round in 0..3 {
        for i in 0..100 {
            t.read_id(i);
        }
        for i in 100..200 {
            common::set_balance(&mut t, i, round);
        }
        let (new_root, witness) = t.commit();

        let verified = witness
            .path_proofs
            .iter()
            .map(|p| {
                p.inner
                    .verify::<Blake3Hasher>(&p.path.path(), prev_root.into_inner())
                    .unwrap()
            })
            .collect::<Vec<_>>();

        for read in &witness.operations.reads {
            let leaf = LeafData {
                key_path: read.key,
                value_hash: read.value.unwrap(),
            };
            assert!(verified[read.path_index].confirm_value(&leaf).unwrap());
        }

        let mut ops_by_path = std::collections::BTreeMap::<usize, Vec<_>>::new();
        for write in &witness.operations.writes {
            ops_by_path
                .entry(write.path_index)
                .or_default()
                .push((write.key, write.value));
        }
        let mut updates = ops_by_path
            .into_iter()
            .map(|(i, ops)| proof::PathUpdate {
                inner: verified[i].clone(),
                ops,
            })
            .collect::<Vec<_>>();
        updates.sort_by(|a, b| a.inner.path().partial_cmp(b.inner.path()).unwrap());
        assert_eq!(
            proof::verify_update::<Blake3Hasher>(prev_root.into_inner(), &updates).unwrap(),
            new_root.into_inner(),
        );
        prev_root = new_root;
    }
}

/// Three coexisting sessions with rollback enabled; the rollback worker pool has 2 threads and
/// each live session's reverse-delta worker occupies one until that session ends.
fn scenario_rollback_three_sessions() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path(), |o| o.rollback(true));
    let ks = keys(8);
    commit(&db, write_all(&ks, 1));

    let s1 = db.begin_session(SessionParams::default());
    let s2 = db.begin_session(SessionParams::default());
    let s3 = db.begin_session(SessionParams::default());
    for k in &ks {
        s3.preserve_prior_value(*k);
    }
    let _ = s3.finish(write_all(&ks, 2)).unwrap();
    drop((s1, s2));
}

/// Two coexisting sessions with warm-up and a single commit worker: the first session's warm-up
/// task holds the only `nomt-commit` thread until that session finishes.
fn scenario_warm_up_two_sessions() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path(), |o| o.warm_up(true));
    let ks = keys(8);
    commit(&db, write_all(&ks, 1));

    let s1 = db.begin_session(SessionParams::default());
    let s2 = db.begin_session(SessionParams::default());
    let _ = s2.finish(write_all(&ks, 2)).unwrap();
    drop(s1);
}

/// Commit an overlay on one thread while another builds a session on top of its child.
/// Either the child is rejected as incomplete, or the session must see both overlays' writes.
fn scenario_overlay_commit_vs_child_session() {
    let dir = tempfile::tempdir().unwrap();
    let db = Arc::new(open(dir.path(), |o| o.commit_concurrency(2)));
    let k1 = account_path(1);
    let k2 = account_path(2);
    commit(&db, write_all(&[k1, k2], 0));

    let o1 = db
        .begin_session(SessionParams::default())
        .finish(write_all(&[k1], 1))
        .unwrap()
        .into_overlay();
    let o2 = Arc::new(
        db.begin_session(SessionParams::default().overlay([&o1]).unwrap())
            .finish(write_all(&[k2], 2))
            .unwrap()
            .into_overlay(),
    );
    let o2_root = o2.root();

    let committer = thread::spawn({
        let db = db.clone();
        move || o1.commit(&db).unwrap()
    });
    let reader = thread::spawn({
        let db = db.clone();
        let o2 = o2.clone();
        move || {
            if let Ok(params) = SessionParams::default().overlay([&*o2]) {
                let s = db.begin_session(params);
                assert_eq!(s.read(k1).unwrap(), Some(value(1)));
                assert_eq!(s.read(k2).unwrap(), Some(value(2)));
                assert_eq!(s.prev_root(), o2_root);
            }
        }
    });
    committer.join().unwrap();
    reader.join().unwrap();

    Arc::into_inner(o2).unwrap().commit(&db).unwrap();
    assert_eq!(db.root(), o2_root);
}

/// One thread commits rounds that rewrite every key; another opens sessions and checks it sees
/// a single round across all keys, and that proofs verify against the session's root.
fn scenario_commit_vs_readers() {
    let dir = tempfile::tempdir().unwrap();
    let db = Arc::new(open(dir.path(), |o| o.commit_concurrency(2)));
    let ks = keys(16);
    commit(&db, write_all(&ks, 0));
    const ROUNDS: u64 = 3;

    let writer = thread::spawn({
        let db = db.clone();
        let ks = ks.clone();
        move || {
            for round in 1..=ROUNDS {
                commit(&db, write_all(&ks, round));
            }
        }
    });
    let reader = thread::spawn({
        let db = db.clone();
        let ks = ks.clone();
        move || {
            for _ in 0..ROUNDS {
                let s = db.begin_session(SessionParams::default());
                let seen = s.read(ks[0]).unwrap();
                for k in &ks {
                    assert_eq!(s.read(*k).unwrap(), seen, "torn snapshot");
                }
                let proof = s.prove(ks[3]).unwrap();
                let verified = proof
                    .verify::<Blake3Hasher>(ks[3].view_bits::<Msb0>(), s.prev_root().into_inner())
                    .unwrap();
                let leaf = LeafData {
                    key_path: ks[3],
                    value_hash: *blake3::hash(seen.as_ref().unwrap()).as_bytes(),
                };
                assert!(verified.confirm_value(&leaf).unwrap());
                assert_eq!(db.read(ks[5]).unwrap().is_some(), true);
            }
        }
    });
    writer.join().unwrap();
    reader.join().unwrap();
    assert_eq!(db.read(ks[0]).unwrap(), Some(value(ROUNDS)));
}

/// A thread holding a session opens a second one while another thread commits. Both sessions take
/// `access_lock` for reading; parking_lot's fair `RwLock` makes the second read wait behind a
/// queued writer, which itself waits for the first session.
fn scenario_nested_session_vs_commit() {
    let dir = tempfile::tempdir().unwrap();
    let db = Arc::new(open(dir.path(), |_| {}));
    let ks = keys(4);
    commit(&db, write_all(&ks, 0));

    let writer = thread::spawn({
        let db = db.clone();
        let ks = ks.clone();
        move || {
            commit(&db, write_all(&ks, 1));
        }
    });
    let reader = thread::spawn({
        let db = db.clone();
        move || {
            let s1 = db.begin_session(SessionParams::default());
            let s2 = db.begin_session(SessionParams::default());
            drop((s1, s2));
        }
    });
    writer.join().unwrap();
    reader.join().unwrap();
}

/// Rollback racing with readers: readers must always see a whole round, and the final root must
/// be the one recorded before the rolled-back commit.
fn scenario_rollback_vs_readers() {
    let dir = tempfile::tempdir().unwrap();
    let db = Arc::new(open(dir.path(), |o| {
        o.rollback(true);
        o.commit_concurrency(2);
    }));
    let ks = keys(16);
    let root0 = commit(&db, write_all(&ks, 0));
    commit(&db, write_all(&ks, 1));

    let roller = thread::spawn({
        let db = db.clone();
        move || db.rollback(1).unwrap()
    });
    let reader = thread::spawn({
        let db = db.clone();
        let ks = ks.clone();
        move || {
            for _ in 0..2 {
                let s = db.begin_session(SessionParams::default());
                let seen = s.read(ks[0]).unwrap();
                assert!(seen == Some(value(0)) || seen == Some(value(1)));
                for k in &ks {
                    assert_eq!(s.read(*k).unwrap(), seen, "torn snapshot");
                }
            }
        }
    });
    roller.join().unwrap();
    reader.join().unwrap();
    assert_eq!(db.root(), root0);
    assert_eq!(db.read(ks[7]).unwrap(), Some(value(0)));
}

#[cfg(not(shuttle))]
macro_rules! scenario_tests {
    ($($name:ident => $scenario:ident),* $(,)?) => {
        $(
            #[test]
            #[ignore]
            fn $name() {
                with_deadline(60, $scenario);
            }
        )*
    };
}

#[cfg(shuttle)]
fn run_shuttle(f: fn()) {
    use shuttle::scheduler::{PctScheduler, RandomScheduler};
    let iters = std::env::var("SHUTTLE_ITERS").map_or(100, |v| v.parse().unwrap());
    let mut config = shuttle::Config::new();
    // NOMT builds large arrays on the stack (e.g. the page pool's region table).
    config.stack_size = 8 << 20;
    match std::env::var("SHUTTLE_SCHED").as_deref() {
        Ok("random") => shuttle::Runner::new(RandomScheduler::new(iters), config).run(f),
        _ => shuttle::Runner::new(PctScheduler::new(3, iters), config).run(f),
    };
}

#[cfg(shuttle)]
macro_rules! scenario_tests {
    ($($name:ident => $scenario:ident),* $(,)?) => {
        $(
            #[test]
            fn $name() {
                run_shuttle($scenario);
            }
        )*
    };
}

scenario_tests! {
    witness_concurrent_workers => scenario_witness_concurrent_workers,
    rollback_three_sessions => scenario_rollback_three_sessions,
    warm_up_two_sessions => scenario_warm_up_two_sessions,
    overlay_commit_vs_child_session => scenario_overlay_commit_vs_child_session,
    commit_vs_readers => scenario_commit_vs_readers,
    rollback_vs_readers => scenario_rollback_vs_readers,
    nested_session_vs_commit => scenario_nested_session_vs_commit,
}
