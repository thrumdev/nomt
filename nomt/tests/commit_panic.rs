use nomt::{
    hasher::Blake3Hasher, KeyReadWrite, Nomt, Options, PanicOnSyncMode, Root, SessionParams,
};
use std::{path::PathBuf, sync::mpsc, time::Duration};

fn key(first: u8) -> [u8; 32] {
    let mut k = [0u8; 32];
    k[0] = first;
    k
}

fn open(path: &PathBuf, panic_on_sync: Option<PanicOnSyncMode>) -> Nomt<Blake3Hasher> {
    let mut o = Options::new();
    o.path(path);
    o.commit_concurrency(1);
    o.io_workers(1);
    o.bitbox_seed([0; 16]);
    o.hashtable_buckets(64_000);
    if let Some(mode) = panic_on_sync {
        o.panic_on_sync(mode);
    }
    Nomt::open(o).unwrap()
}

fn commit_catching(nomt: &Nomt<Blake3Hasher>, writes: &[(u8, u8)]) -> (Root, bool) {
    let session = nomt.begin_session(SessionParams::default());
    let actuals = writes
        .iter()
        .map(|&(k, v)| (key(k), KeyReadWrite::Write(Some(vec![v]))))
        .collect::<Vec<_>>();
    for (k, _) in &actuals {
        session.warm_up(*k);
    }
    let finished = session.finish(actuals).unwrap();
    let root = finished.root();
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| finished.commit(nomt)));
    (root, r.is_err())
}

// A commit that panicked after the meta write is durable; a later commit attempt on the
// same instance must not destroy it.
#[test]
fn panicked_commit_survives_a_later_commit_attempt() {
    let control = PathBuf::from("test/commit_panic_control");
    let _ = std::fs::remove_dir_all(&control);
    let nomt = open(&control, Some(PanicOnSyncMode::PostMeta));
    let (control_root, panicked) = commit_catching(&nomt, &[(0x00, 1), (0x80, 2)]);
    assert!(panicked);
    drop(nomt);
    let nomt = open(&control, None);
    assert_eq!(
        nomt.root(),
        control_root,
        "control: single panicked commit recovers"
    );
    drop(nomt);

    let path = PathBuf::from("test/commit_panic");
    let _ = std::fs::remove_dir_all(&path);

    let (tx, rx) = mpsc::channel();
    let thread_path = path.clone();
    std::thread::spawn(move || {
        let nomt = open(&thread_path, Some(PanicOnSyncMode::PostMeta));
        let (root1, panicked1) = commit_catching(&nomt, &[(0x00, 1), (0x80, 2)]);
        let poisoned_after_first = nomt.is_poisoned();
        let (_, panicked2) = commit_catching(&nomt, &[(0x40, 3)]);
        let poisoned_after_second = nomt.is_poisoned();
        drop(nomt);
        let _ = tx.send((
            root1,
            panicked1,
            poisoned_after_first,
            panicked2,
            poisoned_after_second,
        ));
    });
    let (root1, panicked1, poisoned_after_first, panicked2, poisoned_after_second) = rx
        .recv_timeout(Duration::from_secs(120))
        .expect("commit after a panicked commit hung");
    assert!(panicked1);
    eprintln!(
        "poisoned after first panic: {poisoned_after_first}; second commit panicked: {panicked2}; poisoned after second: {poisoned_after_second}"
    );

    let nomt = open(&path, None);
    assert_eq!(nomt.read(key(0x00)).unwrap(), Some(vec![1]));
    assert_eq!(nomt.read(key(0x80)).unwrap(), Some(vec![2]));
    assert_eq!(
        nomt.root(),
        root1,
        "first commit's values are durable, but its merkle pages were lost when the second commit replaced the WAL"
    );
    assert!(
        poisoned_after_first,
        "store not poisoned after a commit panicked past the meta write"
    );
}
