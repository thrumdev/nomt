use nomt::{hasher::Blake3Hasher, KeyReadWrite, Nomt, Options, SessionParams};

fn setup_nomt(name: &str) -> (tempfile::TempDir, Nomt<Blake3Hasher>) {
    let dir = tempfile::tempdir().unwrap();
    let mut o = Options::new();
    o.path(dir.path().join(name));
    o.commit_concurrency(1);
    let nomt = Nomt::open(o).unwrap();
    (dir, nomt)
}

#[test]
fn finish_rejects_repeated_key() {
    let (_dir, nomt) = setup_nomt("repeated_key");
    let session = nomt.begin_session(SessionParams::default());
    let actuals = vec![
        ([1; 32], KeyReadWrite::Write(Some(vec![1]))),
        ([1; 32], KeyReadWrite::Write(None)),
    ];
    assert!(session.finish(actuals).is_err());
    assert!(nomt.root().is_empty());
}

#[test]
fn finish_rejects_unsorted_keys() {
    let (_dir, nomt) = setup_nomt("unsorted_keys");
    let session = nomt.begin_session(SessionParams::default());
    let actuals = vec![
        ([2; 32], KeyReadWrite::Write(Some(vec![2]))),
        ([1; 32], KeyReadWrite::Write(Some(vec![1]))),
    ];
    assert!(session.finish(actuals).is_err());
    assert!(nomt.root().is_empty());
}
