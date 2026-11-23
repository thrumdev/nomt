mod common;

use common::Test;
use nomt::PanicOnSyncMode;

fn key(first: u8) -> [u8; 32] {
    let mut k = [0u8; 32];
    k[0] = first;
    k
}

#[test]
fn wal_recovery_test_post_meta_swap() {
    // Initialize the db with panic on sync equals true.
    let mut t = Test::new_with_params(
        "wal_add_remove_1000",
        1,                               // commit_concurrency,
        1000000,                         // hashtable_buckets,
        Some(PanicOnSyncMode::PostMeta), // panic_on_sync
        true,                            // clean
    );

    common::set_balance(&mut t, 0, 1000);
    common::set_balance(&mut t, 1, 2000);
    common::set_balance(&mut t, 2, 3000);

    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        t.commit();
    }));
    assert!(r.is_err());
    drop(t);

    // Re-open the db without cleaning the DB dir and without panic on sync.
    let mut t = Test::new_with_params(
        "wal_add_remove_1000",
        1,       // commit_concurrency,
        1000000, // hashtable_buckets,
        None,    // panic_on_sync
        false,   // clean
    );
    assert_eq!(common::read_balance(&mut t, 0), Some(1000));
    assert_eq!(common::read_balance(&mut t, 1), Some(2000));
    assert_eq!(common::read_balance(&mut t, 2), Some(3000));
}

#[test]
fn wal_recovery_test_pre_meta_swap() {
    // Initialize the db with panic on sync equals true.
    let mut t = Test::new_with_params(
        "wal_pre_meta_swap",
        1,                              // commit_concurrency,
        1000000,                        // hashtable_buckets,
        Some(PanicOnSyncMode::PostWal), // panic_on_sync
        true,                           // clean
    );

    for i in 0..1000 {
        common::set_balance(&mut t, i, 1000);
    }

    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        t.commit();
    }));
    assert!(r.is_err());
    drop(t);

    // Re-open the db without cleaning the DB dir and without panic on sync.
    let mut t = Test::new_with_params(
        "wal_pre_meta_swap",
        1,       // commit_concurrency,
        1000000, // hashtable_buckets,
        None,    // panic_on_sync
        false,   // clean
    );

    // DB should open cleanly and not have any incomplete changes; the WAL is too new and will be
    // discarded.
    for i in 0..1000 {
        assert_eq!(common::read_balance(&mut t, i), None);
    }
}

// The commit empties the left half of the root page and fills the right half. The emptied node is
// first recorded as "page cleared", which the write to the other half then erases.
#[test]
fn wal_recovery_restores_a_node_cleared_before_another_write() {
    let empty_left_fill_right = |t: &mut Test| {
        t.write(key(0x00), None);
        t.write(key(0x40), None);
        t.write(key(0x80), Some(vec![3]));
        t.write(key(0xC0), Some(vec![4]));
    };
    let seed = |name: &str| {
        let mut t = Test::new_with_params(name, 1, 64_000, None, true);
        t.write(key(0x00), Some(vec![1]));
        t.write(key(0x40), Some(vec![2]));
        t.commit();
    };

    let reference = "wal_cleared_node_reference";
    seed(reference);
    let mut t = Test::new_with_params(reference, 1, 64_000, None, false);
    empty_left_fill_right(&mut t);
    let (expected_root, _) = t.commit();
    drop(t);

    let name = "wal_cleared_node";
    seed(name);
    let mut t = Test::new_with_params(name, 1, 64_000, Some(PanicOnSyncMode::PostMeta), false);
    empty_left_fill_right(&mut t);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        t.commit();
    }));
    assert!(r.is_err());
    drop(t);

    let mut t = Test::new_with_params(name, 1, 64_000, None, false);
    assert_eq!(t.read(key(0x00)), None);
    assert_eq!(t.read(key(0x80)), Some(vec![3]));
    assert_eq!(t.root(), expected_root);
}
