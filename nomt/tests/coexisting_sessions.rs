use nomt::{hasher::Blake3Hasher, KeyReadWrite, Nomt, Options, SessionParams};
use nomt_test_utils::account_path;

fn open(path: &std::path::Path, configure: impl FnOnce(&mut Options)) -> Nomt<Blake3Hasher> {
    let mut o = Options::new();
    o.path(path);
    o.bitbox_seed([0; 16]);
    o.hashtable_buckets(4096);
    o.io_workers(1);
    o.commit_concurrency(1);
    configure(&mut o);
    Nomt::open(o).unwrap()
}

/// Finishes the newest of `n` coexisting sessions and commits it once the others are gone.
fn finish_newest_of(db: &Nomt<Blake3Hasher>, n: usize) {
    let key = account_path(1);
    let mut sessions: Vec<_> = (0..n)
        .map(|_| db.begin_session(SessionParams::default()))
        .collect();
    let newest = sessions.pop().unwrap();
    newest.warm_up(key);
    newest.preserve_prior_value(key);
    let finished = newest
        .finish(vec![(key, KeyReadWrite::Write(Some(vec![1])))])
        .unwrap();
    drop(sessions);
    finished.commit(db).unwrap();
    assert_eq!(db.read(key).unwrap(), Some(vec![1]));
}

#[test]
fn many_coexisting_sessions_with_rollback() {
    // Each live session has a reverse-delta worker running until the session ends.
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path(), |o| o.rollback(true));
    finish_newest_of(&db, 4);
}
