mod common;

use common::Test;

fn expected_root(items: Vec<([u8; 32], Vec<u8>)>) -> nomt_core::trie::Node {
    nomt_core::update::build_trie::<nomt::hasher::Blake3Hasher>(
        0,
        items
            .into_iter()
            .map(|(k, v)| (k, *blake3::hash(&v).as_bytes())),
        |_| {},
    )
}

#[test]
fn overlay_multiple_forks() {
    let mut test = Test::new("overlay_multiple_forks");

    let overlay_a = test.update().0;
    let overlay_b1 = {
        test.start_overlay_session([&overlay_a]);
        test.write([1; 32], Some(vec![1, 2, 3]));
        test.update().0
    };
    let overlay_b2 = {
        test.start_overlay_session([&overlay_a]);
        test.write([1; 32], Some(vec![4, 5, 6]));
        test.update().0
    };

    {
        test.start_overlay_session([&overlay_b1, &overlay_a]);
        assert_eq!(test.read([1; 32]), Some(vec![1, 2, 3]));
    }

    {
        test.start_overlay_session([&overlay_b2, &overlay_a]);
        assert_eq!(test.read([1; 32]), Some(vec![4, 5, 6]));
    }
}

#[test]
fn overlay_root_calculation() {
    let mut test = Test::new("overlay_root_calculation");
    test.write([1; 32], Some(vec![1, 2, 3]));
    let overlay_a = test.update().0;

    assert_eq!(
        overlay_a.root().into_inner(),
        expected_root(vec![([1; 32], vec![1, 2, 3])]),
    );

    test.start_overlay_session([&overlay_a]);
    test.write([2; 32], Some(vec![4, 5, 6]));
    let overlay_b = test.update().0;

    assert_eq!(
        overlay_b.root().into_inner(),
        expected_root(vec![([1; 32], vec![1, 2, 3]), ([2; 32], vec![4, 5, 6])]),
    );

    test.start_overlay_session([&overlay_b, &overlay_a]);
    test.write([1; 32], Some(vec![7, 8, 9]));
    test.write([3; 32], Some(vec![0, 1, 0]));
    let overlay_c = test.update().0;

    assert_eq!(
        overlay_c.root().into_inner(),
        expected_root(vec![
            ([1; 32], vec![7, 8, 9]),
            ([2; 32], vec![4, 5, 6]),
            ([3; 32], vec![0, 1, 0])
        ]),
    );
}

#[test]
#[should_panic]
fn overlays_must_be_committed_in_order() {
    let mut test = Test::new("overlays_committed_in_order");
    let overlay_a = test.update().0;
    test.start_overlay_session([&overlay_a]);
    let overlay_b = test.update().0;

    test.commit_overlay(overlay_b);
}

#[test]
#[should_panic]
fn overlay_competing_committed() {
    let mut test = Test::new("overlays_competing_committed");
    let overlay_a = test.update().0;
    test.start_overlay_session([&overlay_a]);
    let overlay_b1 = test.update().0;
    test.start_overlay_session([&overlay_a]);
    let overlay_b2 = test.update().0;

    test.commit_overlay(overlay_a);
    test.commit_overlay(overlay_b1);

    test.commit_overlay(overlay_b2);
}

#[test]
fn overlay_commit_in_order_works() {
    let mut test = Test::new("overlays_commit_in_order_works");
    let overlay_a = test.update().0;
    test.start_overlay_session([&overlay_a]);
    let overlay_b = test.update().0;

    test.commit_overlay(overlay_a);
    test.commit_overlay(overlay_b);
}

#[test]
fn overlay_changes_land_on_disk_when_committed() {
    {
        let mut test = Test::new("overlay_changes_land_on_disk");
        test.write([1; 32], Some(vec![1, 2, 3]));
        test.write([2; 32], Some(vec![4, 5, 6]));
        test.write([3; 32], Some(vec![7, 8, 9]));

        let overlay = test.update().0;
        test.commit_overlay(overlay);
    }

    let mut test = Test::new_with_params(
        "overlay_changes_land_on_disk",
        /* commit_concurrency */ 1,
        /* hashtable_buckets */ 1,
        /* panic_on_sync */ None,
        /* cleanup_dir */ false,
    );

    assert_eq!(test.read([1; 32]), Some(vec![1, 2, 3]));
    assert_eq!(test.read([2; 32]), Some(vec![4, 5, 6]));
    assert_eq!(test.read([3; 32]), Some(vec![7, 8, 9]));
}

#[test]
fn overlay_uncommitted_not_on_disk() {
    {
        let mut test = Test::new("overlay_uncommitted_not_on_disk");
        test.write([1; 32], Some(vec![1, 2, 3]));
        test.write([2; 32], Some(vec![4, 5, 6]));
        test.write([3; 32], Some(vec![7, 8, 9]));

        let _overlay = test.update().0;
    }

    let mut test = Test::new_with_params(
        "overlay_uncommitted_not_on_disk",
        /* commit_concurrency */ 1,
        /* hashtable_buckets */ 1,
        /* panic_on_sync */ None,
        /* cleanup_dir */ false,
    );

    assert_eq!(test.read([1; 32]), None);
    assert_eq!(test.read([2; 32]), None);
    assert_eq!(test.read([3; 32]), None);
}

#[test]
fn overlay_deletions() {
    let test_db = || -> Test {
        let mut test = Test::new("overlay_deletions");
        // subtree at 0000000_0/1
        test.write([0; 32], Some(vec![1, 1]));
        test.write([1; 32], Some(vec![2, 2]));

        // subtree at 001000_00/01/10
        test.write([32; 32], Some(vec![1, 1]));
        test.write([33; 32], Some(vec![2, 2]));
        test.write([34; 32], Some(vec![3, 3]));

        // subtree at 100000_00/01/10/11
        test.write([128; 32], Some(vec![4, 4]));
        test.write([129; 32], Some(vec![5, 5]));
        test.write([130; 32], Some(vec![6, 6]));
        test.write([131; 32], Some(vec![7, 7]));

        test.commit();
        test
    };

    // Delete the first item for each subtree
    let mut test = test_db();

    test.write([0; 32], None);
    test.write([32; 32], None);
    test.write([128; 32], None);
    let overlay_a = test.update().0;

    test.start_overlay_session([&overlay_a]);
    assert_eq!(test.read([0; 32]), None);
    assert_eq!(test.read([1; 32]), Some(vec![2, 2]));

    assert_eq!(test.read([32; 32]), None);
    assert_eq!(test.read([33; 32]), Some(vec![2, 2]));
    assert_eq!(test.read([34; 32]), Some(vec![3, 3]));

    assert_eq!(test.read([128; 32]), None);
    assert_eq!(test.read([129; 32]), Some(vec![5, 5]));
    assert_eq!(test.read([130; 32]), Some(vec![6, 6]));
    assert_eq!(test.read([131; 32]), Some(vec![7, 7]));

    let _overlay_b = test.update().0;

    // Delete the second item for each subtree
    let mut test = test_db();

    test.write([1; 32], None);
    test.write([33; 32], None);
    test.write([129; 32], None);
    let overlay_a = test.update().0;

    test.start_overlay_session([&overlay_a]);
    assert_eq!(test.read([0; 32]), Some(vec![1, 1]));
    assert_eq!(test.read([1; 32]), None);

    assert_eq!(test.read([32; 32]), Some(vec![1, 1]));
    assert_eq!(test.read([33; 32]), None);
    assert_eq!(test.read([34; 32]), Some(vec![3, 3]));

    assert_eq!(test.read([128; 32]), Some(vec![4, 4]));
    assert_eq!(test.read([129; 32]), None);
    assert_eq!(test.read([130; 32]), Some(vec![6, 6]));
    assert_eq!(test.read([131; 32]), Some(vec![7, 7]));

    let _overlay_b = test.update().0;

    // Sequence of deletes
    let mut test = test_db();

    test.write([32; 32], None);
    test.write([33; 32], None);
    test.write([128; 32], None);
    test.write([129; 32], None);
    test.write([131; 32], None);
    let overlay_a = test.update().0;

    test.start_overlay_session([&overlay_a]);
    assert_eq!(test.read([32; 32]), None);
    assert_eq!(test.read([33; 32]), None);
    assert_eq!(test.read([34; 32]), Some(vec![3, 3]));

    assert_eq!(test.read([128; 32]), None);
    assert_eq!(test.read([129; 32]), None);
    assert_eq!(test.read([130; 32]), Some(vec![6, 6]));
    assert_eq!(test.read([131; 32]), None);

    let _overlay_b = test.update().0;
}

#[test]
fn overlay_detect_alredy_committed_chain() {
    let mut test = Test::new("overlay_wrong_chains");

    test.write([0; 32], Some(vec![1]));
    let overlay_a = test.update().0;

    test.start_overlay_session([&overlay_a]);
    test.write([0; 32], Some(vec![2]));
    let overlay_b = test.update().0;

    test.start_overlay_session([&overlay_b, &overlay_a]);
    test.write([0; 32], Some(vec![3]));
    let overlay_c = test.update().0;

    let params = nomt::SessionParams::default()
        .witness_mode(nomt::WitnessMode::read_write())
        .overlay([&overlay_c, &overlay_b, &overlay_a])
        .unwrap();

    test.commit_overlay(overlay_a);
    test.commit_overlay(overlay_b);
    test.commit_overlay(overlay_c);

    assert!(test.try_begin_session(params).is_err());
}

#[test]
fn overlay_detect_parallel_non_committed_overlay_chain() {
    let mut test = Test::new("overlay_detect_parallel_non_committed_overlay_chain");

    test.write([0; 32], Some(vec![1]));
    let overlay_a = test.update().0;

    test.start_overlay_session([&overlay_a]);
    test.write([0; 32], Some(vec![2]));
    let overlay_b = test.update().0;

    test.start_overlay_session([&overlay_b, &overlay_a]);
    test.write([0; 32], Some(vec![3]));
    let overlay_c = test.update().0;

    test.write([0; 32], Some(vec![4]));
    let overlay_d = test.update().0;

    test.start_overlay_session([&overlay_d]);
    test.write([0; 32], Some(vec![5]));
    let overlay_e = test.update().0;

    test.start_overlay_session([&overlay_e, &overlay_d]);
    test.write([0; 32], Some(vec![6]));
    let overlay_f = test.update().0;

    test.start_overlay_session([&overlay_f, &overlay_e, &overlay_d]);
    test.write([0; 32], Some(vec![7]));
    let overlay_g = test.update().0;

    test.commit_overlay(overlay_d);
    test.commit_overlay(overlay_e);
    test.commit_overlay(overlay_f);

    let params = nomt::SessionParams::default()
        .witness_mode(nomt::WitnessMode::read_write())
        .overlay([&overlay_c, &overlay_b, &overlay_a])
        .unwrap();
    assert!(test.try_begin_session(params).is_err());

    let params = nomt::SessionParams::default()
        .witness_mode(nomt::WitnessMode::read_write())
        .overlay([&overlay_g])
        .unwrap();
    assert!(test.try_begin_session(params).is_ok());
}
