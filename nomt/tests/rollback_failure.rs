use nomt::{hasher::Blake3Hasher, KeyReadWrite, Nomt, Options, Root, SessionParams};
use nomt_test_utils::account_path;
use std::{
    ffi::CString,
    path::{Path, PathBuf},
    sync::mpsc,
    time::Duration,
};

fn open(path: &Path) -> Nomt<Blake3Hasher> {
    let mut o = Options::new();
    o.path(path);
    o.commit_concurrency(1);
    o.io_workers(1);
    o.bitbox_seed([0; 16]);
    o.hashtable_buckets(64_000);
    o.rollback(true);
    Nomt::open(o).unwrap()
}

fn commit(nomt: &Nomt<Blake3Hasher>, writes: impl IntoIterator<Item = (u64, u64)>) -> Root {
    let session = nomt.begin_session(SessionParams::default());
    let mut actuals = writes
        .into_iter()
        .map(|(id, v)| {
            (
                account_path(id),
                KeyReadWrite::Write(Some(v.to_le_bytes().to_vec())),
            )
        })
        .collect::<Vec<_>>();
    actuals.sort_by_key(|(k, _)| *k);
    let finished = session.finish(actuals).unwrap();
    finished.commit(nomt).unwrap();
    nomt.root()
}

fn fd_of(path: &Path) -> i32 {
    let target = std::fs::canonicalize(path).unwrap();
    for entry in std::fs::read_dir("/proc/self/fd").unwrap() {
        let entry = entry.unwrap();
        if std::fs::read_link(entry.path()).map_or(false, |l| l == target) {
            return entry.file_name().to_str().unwrap().parse().unwrap();
        }
    }
    panic!("no open fd for {}", path.display());
}

fn with_failing_reads<R>(fd: i32, f: impl FnOnce() -> R) -> R {
    let null = CString::new("/dev/null").unwrap();
    let saved = unsafe {
        let saved = libc::dup(fd);
        assert!(saved >= 0);
        let wronly = libc::open(null.as_ptr(), libc::O_WRONLY);
        assert!(wronly >= 0);
        assert_eq!(libc::dup2(wronly, fd), fd);
        libc::close(wronly);
        saved
    };
    let r = f();
    unsafe {
        assert_eq!(libc::dup2(saved, fd), fd);
        libc::close(saved);
    }
    r
}

#[test]
fn failed_rollback_leaves_the_log_intact() {
    let path = PathBuf::from("test/failed_rollback_leaves_the_log_intact");
    let _ = std::fs::remove_dir_all(&path);

    let nomt = open(&path);
    commit(&nomt, (0..2000).map(|i| (i, 1000)));
    commit(&nomt, [(7, 1)]);
    let root_3 = commit(&nomt, [(9, 2)]);
    drop(nomt);

    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let nomt = open(&path);
        let ht_fd = fd_of(&path.join("ht"));
        let failed = with_failing_reads(ht_fd, || {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| nomt.rollback(1)))
        });
        let failed = match failed {
            Ok(Ok(())) => "ok".to_string(),
            Ok(Err(e)) => format!("err: {e:#}"),
            Err(_) => "panic".to_string(),
        };
        let after_failure = (nomt.is_poisoned(), nomt.root());

        let commit_4 =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| commit(&nomt, [(11, 3)])))
                .is_ok();
        drop(nomt);

        let nomt = open(&path);
        let rollback = nomt.rollback(1).map_err(|e| format!("{e:#}"));
        let _ = tx.send((failed, after_failure, commit_4, rollback, nomt.root()));
    });
    let (failed, (poisoned, root_after_failure), commit_4, rollback, root_after_rollback) =
        rx.recv_timeout(Duration::from_secs(120)).expect("hung");

    eprintln!("rollback(1) with failing merkle reads: {failed}; poisoned: {poisoned}");
    assert!(
        failed.starts_with("err"),
        "fault injection did not produce an Err"
    );
    assert!(!poisoned);
    assert_eq!(root_after_failure, root_3);
    assert!(commit_4, "commit after the failed rollback failed");
    eprintln!("rollback(1) after reopen: {rollback:?}");
    assert_eq!(
        root_after_rollback, root_3,
        "rollback(1) after [failed rollback(1), commit, reopen] did not undo that commit"
    );
}
