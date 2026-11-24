use nomt::{
    hasher::Blake3Hasher, trie::KeyPath, FinishedSession, KeyReadWrite, Nomt, Options, Root,
    SessionParams,
};
use std::path::PathBuf;

fn open(name: &str) -> Nomt<Blake3Hasher> {
    let path = PathBuf::from("test").join(name);
    let _ = std::fs::remove_dir_all(&path);
    let mut o = Options::new();
    o.path(path);
    o.commit_concurrency(1);
    o.bitbox_seed([0; 16]);
    o.hashtable_buckets(4096);
    o.rollback(true);
    Nomt::open(o).unwrap()
}

fn key(first: u8) -> KeyPath {
    let mut key = [0; 32];
    key[0] = first;
    key
}

fn finish(
    nomt: &Nomt<Blake3Hasher>,
    params: SessionParams,
    writes: &[(u8, Option<u8>)],
) -> FinishedSession {
    let actuals = writes
        .iter()
        .map(|(k, v)| (key(*k), KeyReadWrite::Write(v.map(|v| vec![v]))))
        .collect();
    nomt.begin_session(params).finish(actuals).unwrap()
}

fn commit(nomt: &Nomt<Blake3Hasher>, writes: &[(u8, Option<u8>)]) -> Root {
    finish(nomt, SessionParams::default(), writes)
        .commit(nomt)
        .unwrap();
    nomt.root()
}

// A changeset is built against the pages of the database at that moment. After a commit and its
// rollback the root is the same again, but the pages are not.
#[test]
fn session_is_stale_after_commit_and_rollback() {
    let nomt = open("stale_session_after_rollback");
    let base = commit(&nomt, &[(1, Some(1)), (2, Some(2))]);

    let stale = finish(&nomt, SessionParams::default(), &[(3, Some(3))]);
    commit(&nomt, &[(4, Some(4))]);
    nomt.rollback(1).unwrap();
    assert_eq!(nomt.root(), base);

    assert!(stale.commit(&nomt).is_err());
    assert_eq!(nomt.root(), base);
}

#[test]
fn overlay_is_stale_after_root_returns_without_rollback() {
    let nomt = open("stale_overlay_after_reinsert");
    let base = commit(&nomt, &[(1, Some(1)), (2, Some(2))]);

    let stale = finish(&nomt, SessionParams::default(), &[(3, Some(3))]).into_overlay();
    commit(&nomt, &[(1, None)]);
    assert_eq!(commit(&nomt, &[(1, Some(1))]), base);

    assert!(stale.commit(&nomt).is_err());
    assert_eq!(nomt.root(), base);
}

#[test]
fn session_on_an_overlay_commits_after_that_overlay() {
    let nomt = open("session_after_its_overlay");
    commit(&nomt, &[(1, Some(1))]);

    let overlay = finish(&nomt, SessionParams::default(), &[(2, Some(2))]).into_overlay();
    let params = SessionParams::default().overlay([&overlay]).unwrap();
    let child = finish(&nomt, params, &[(3, Some(3))]);
    let child_root = child.root();

    overlay.commit(&nomt).unwrap();
    child.commit(&nomt).unwrap();
    assert_eq!(nomt.root(), child_root);
}

#[test]
fn session_survives_a_commit_that_changes_nothing() {
    let nomt = open("session_after_noop_commit");
    commit(&nomt, &[(1, Some(1))]);

    let pending = finish(&nomt, SessionParams::default(), &[(2, Some(2))]);
    let pending_root = pending.root();
    commit(&nomt, &[]);

    pending.commit(&nomt).unwrap();
    assert_eq!(nomt.root(), pending_root);
}
