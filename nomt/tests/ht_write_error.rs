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

// A commit whose hash-table page writes fail must report the failure, and the database must
// come back complete when it is reopened.
//
// RLIMIT_FSIZE makes every write past 40 MiB fail with EFBIG. Only the 250 MiB hash-table file
// has pages there, so the WAL, meta and beatree writes of the second commit all succeed. The
// commit is therefore durable: the meta names it, and the WAL holds the merkle pages that did
// not reach the hash table. Reopening must replay them.
//
// This must stay the only test in this binary: the limit is process-wide.
#[test]
fn ht_write_error_is_reported_and_recovered() {
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

    assert!(
        committed.is_none(),
        "commit reported success although hash-table writes failed"
    );

    // The same three commits on a database whose writes all succeed.
    let mut reference = Test::new_with_params("ht_write_error_reference", 1, 64_000, None, true);
    for i in 0..ACCOUNTS {
        common::set_balance(&mut reference, i, 1000);
    }
    reference.commit();
    for i in 0..ACCOUNTS {
        common::set_balance(&mut reference, i, 2000);
    }
    let (second_root, _) = reference.commit();
    for i in 0..ACCOUNTS {
        common::set_balance(&mut reference, i, 3000);
    }
    let (third_root, _) = reference.commit();

    let mut t = Test::new_with_params(name, 1, 64_000, None, false);
    assert_eq!(
        t.root(),
        second_root,
        "the failed commit wrote its meta, so it must be the state after reopening"
    );
    for i in 0..ACCOUNTS {
        assert_eq!(common::read_balance(&mut t, i), Some(2000));
    }

    // The root above is read from the root page alone. A commit that touches every account
    // walks every merkle page, and gets a wrong root if one of them was left behind.
    for i in 0..ACCOUNTS {
        common::set_balance(&mut t, i, 3000);
    }
    let (root, _) = t.commit();
    assert_eq!(
        root, third_root,
        "merkle pages of the failed commit were not recovered from the WAL"
    );
}
