//! Once a `Nomt` and its sessions are dropped, the database can be reopened right away.

use nomt::{hasher::Blake3Hasher, KeyReadWrite, Nomt, Options, SessionParams};
use nomt_test_utils::account_path;

fn open(path: &std::path::Path, warm_up: bool) -> Nomt<Blake3Hasher> {
    let mut o = Options::new();
    o.path(path);
    o.bitbox_seed([0; 16]);
    o.hashtable_buckets(4096);
    o.io_workers(1);
    o.commit_concurrency(1);
    o.warm_up(warm_up);
    o.rollback(true);
    Nomt::open(o).expect("reopening right after dropping the database")
}

#[test]
fn reopen_right_after_dropping_sessions() {
    let dir = tempfile::tempdir().unwrap();

    // Give the warm-ups some pages to fetch.
    let db = open(dir.path(), false);
    let session = db.begin_session(SessionParams::default());
    let mut writes = (0..1000)
        .map(|i| (account_path(i), KeyReadWrite::Write(Some(vec![1; 64]))))
        .collect::<Vec<_>>();
    writes.sort_by_key(|(key, _)| *key);
    session.finish(writes).unwrap().commit(&db).unwrap();
    drop(db);

    for round in 0..50 {
        let db = open(dir.path(), true);

        let key = account_path(round);
        let session = db.begin_session(SessionParams::default());
        session.warm_up(key);
        session
            .finish(vec![(key, KeyReadWrite::Write(Some(vec![round as u8])))])
            .unwrap()
            .commit(&db)
            .unwrap();

        // Sessions dropped without finishing, with warm-ups in flight and idle, right before the
        // database itself.
        let session = db.begin_session(SessionParams::default());
        for i in 0..100 {
            session.warm_up(account_path(round * 10 + i));
            session.preserve_prior_value(account_path(round * 10 + i));
        }
        drop(session);
        drop(db.begin_session(SessionParams::default()));
        drop(db);
    }

    let db = open(dir.path(), true);
    assert_eq!(db.read(account_path(49)).unwrap(), Some(vec![49]));
}
