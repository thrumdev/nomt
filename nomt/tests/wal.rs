mod common;

use common::Test;

#[test]
fn wal_recovery_test() {
    // Initialize the db with panic on sync equals true.
    let mut t = Test::new_with_params(
        "wal_add_remove_1000",
        /* panic_on_sync */ true,
        /* clean */ true,
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
        /* panic_on_sync */ false,
        /* clean */ false,
    );
    assert_eq!(common::read_balance(&mut t, 0), Some(1000));
    assert_eq!(common::read_balance(&mut t, 1), Some(2000));
    assert_eq!(common::read_balance(&mut t, 2), Some(3000));
}
