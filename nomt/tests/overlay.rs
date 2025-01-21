mod common;

use common::Test;

fn expected_root(items: Vec<([u8; 32], Vec<u8>)>) -> nomt_core::trie::Node {
    nomt_core::update::build_trie::<nomt::Blake3Hasher>(
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
        overlay_a.root(),
        expected_root(vec![([1; 32], vec![1, 2, 3])]),
    );

    test.start_overlay_session([&overlay_a]);
    test.write([2; 32], Some(vec![4, 5, 6]));
    let overlay_b = test.update().0;

    assert_eq!(
        overlay_b.root(),
        expected_root(vec![([1; 32], vec![1, 2, 3]), ([2; 32], vec![4, 5, 6])]),
    );

    test.start_overlay_session([&overlay_b, &overlay_a]);
    test.write([1; 32], Some(vec![7, 8, 9]));
    test.write([3; 32], Some(vec![0, 1, 0]));
    let overlay_c = test.update().0;

    assert_eq!(
        overlay_c.root(),
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
