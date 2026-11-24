mod common;

use common::Test;
use std::{sync::mpsc, time::Duration};

const ACCOUNTS: u64 = 2000;
const FSIZE_LIMIT: u64 = 40 << 20;

fn set_fsize_limit(limit: u64) {
    let lim = libc::rlimit {
        rlim_cur: limit,
        rlim_max: libc::RLIM_INFINITY,
    };
    unsafe {
        libc::signal(libc::SIGXFSZ, libc::SIG_IGN);
        assert_eq!(libc::setrlimit(libc::RLIMIT_FSIZE, &lim), 0);
    }
}

// A commit whose hash-table page writes fail must not report success.
//
// RLIMIT_FSIZE makes every write past 40 MiB fail with EFBIG. Only the 250 MiB hash-table file
// has pages there, so the WAL, meta and beatree writes of the second commit all succeed.
// This must stay the only test in this binary: the limit is process-wide.
#[test]
fn ht_write_error_is_reported() {
    let name = "ht_write_error";
    let (tx, rx) = mpsc::channel();
    // The database is used on one thread, which this thread watches for a hang.
    std::thread::spawn(move || {
        let mut t = Test::new_with_params(name, 1, 64_000, None, true);
        for i in 0..ACCOUNTS {
            common::set_balance(&mut t, i, 1000);
        }
        t.commit();

        for i in 0..ACCOUNTS {
            common::set_balance(&mut t, i, 2000);
        }
        set_fsize_limit(FSIZE_LIMIT);
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| t.commit().0));
        drop(t);
        let _ = tx.send(r.ok());
    });
    let committed = rx
        .recv_timeout(Duration::from_secs(120))
        .expect("commit with failing hash-table writes hung");
    set_fsize_limit(libc::RLIM_INFINITY);

    let Some(committed_root) = committed else {
        // The commit reported the failure. That is the correct outcome.
        return;
    };

    let mut reference = Test::new_with_params("ht_write_error_reference", 1, 64_000, None, true);
    for i in 0..ACCOUNTS {
        common::set_balance(&mut reference, i, 3000);
    }
    let (reference_root, _) = reference.commit();

    let mut t = Test::new_with_params(name, 1, 64_000, None, false);
    assert_eq!(
        t.root(),
        committed_root,
        "root after reopen differs from the root the commit reported"
    );
    for i in 0..ACCOUNTS {
        common::set_balance(&mut t, i, 3000);
    }
    let (root, _) = t.commit();
    assert_eq!(
        root, reference_root,
        "commit reported success, but the reopened database computes a wrong root"
    );
}
