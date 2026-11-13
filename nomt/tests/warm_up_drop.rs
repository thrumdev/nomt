//! Dropping a session with warm-up enabled must not panic its warm-up worker.
//!
//! The worker runs on a thread of its own and its panic is caught, so the test counts panics with a
//! hook instead of relying on one reaching it.

use std::{
    sync::atomic::{AtomicUsize, Ordering},
    time::{Duration, Instant},
};

use nomt::{hasher::Blake3Hasher, KeyReadWrite, Nomt, Options, SessionParams};
use nomt_test_utils::account_path;

static PANICS: AtomicUsize = AtomicUsize::new(0);

fn open(path: &std::path::Path) -> Nomt<Blake3Hasher> {
    let mut o = Options::new();
    o.path(path);
    o.bitbox_seed([0; 16]);
    o.hashtable_buckets(4096);
    o.io_workers(1);
    o.commit_concurrency(1);
    o.warm_up(true);
    Nomt::open(o).unwrap()
}

/// Waits for every warm-up thread to exit, so any panic of theirs has been counted.
fn wait_for_warm_up_threads() {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let running = std::fs::read_dir("/proc/self/task")
            .map(|tasks| {
                tasks.flatten().any(|task| {
                    std::fs::read_to_string(task.path().join("comm"))
                        .is_ok_and(|name| name.trim() == "nomt-warm-up")
                })
            })
            // Without procfs, give the threads time to exit instead.
            .unwrap_or_else(|_| {
                std::thread::sleep(Duration::from_millis(500));
                false
            });
        if !running {
            return;
        }
        assert!(Instant::now() < deadline, "warm-up threads didn't exit");
        std::thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn dropping_a_session_does_not_panic_the_warm_up() {
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        PANICS.fetch_add(1, Ordering::SeqCst);
        default_hook(info);
    }));

    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());

    // Give the warm-ups some pages to fetch.
    let session = db.begin_session(SessionParams::default());
    let mut writes = (0..1000)
        .map(|i| (account_path(i), KeyReadWrite::Write(Some(vec![1; 64]))))
        .collect::<Vec<_>>();
    writes.sort_by_key(|(key, _)| *key);
    for (key, _) in &writes {
        session.warm_up(*key);
    }
    session.finish(writes).unwrap().commit(&db).unwrap();

    // Dropped while idle.
    drop(db.begin_session(SessionParams::default()));

    // Dropped with warm-ups queued and in flight.
    for round in 0..10 {
        let session = db.begin_session(SessionParams::default());
        for i in 0..200 {
            session.warm_up(account_path(round * 100 + i));
        }
        drop(session);
    }

    // Dropped while another session is live.
    let outer = db.begin_session(SessionParams::default());
    let nested = db.begin_session(SessionParams::default());
    nested.warm_up(account_path(1));
    drop(nested);
    drop(outer);

    wait_for_warm_up_threads();
    assert_eq!(
        PANICS.load(Ordering::SeqCst),
        0,
        "a warm-up worker panicked"
    );

    // The database still works.
    let session = db.begin_session(SessionParams::default());
    let key = account_path(1);
    session.warm_up(key);
    session
        .finish(vec![(key, KeyReadWrite::Write(Some(vec![2])))])
        .unwrap()
        .commit(&db)
        .unwrap();
    assert_eq!(db.read(key).unwrap(), Some(vec![2]));
}
