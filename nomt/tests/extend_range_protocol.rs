mod common;
use common::Test;
use std::path::Path;

// nomt::beatree::branch::LEAF_NODE_BODY_SIZE is
// expected to be 4096 and thus the merge threshold is 2047.
//
// This parameter makes it possible to define the following vector of
// keys and values whose size, when inserted into the database, will result
// in the expected set of leaves. Each line adheres to the half full
// requirement, and the first element of the next row does not fit
// in the previous leaf, requiring a new one. The last row does not
// need to meet the half full requirement, as it may be the rightmost leaf.
#[rustfmt::skip]
const KEYS_AND_VALUE_SIZES: [(u8, usize); 16] =[
    // leaf 1
    (1, 1100), (2, 1000), (3, 1000),
    // leaf 2
    (4, 900), (5, 900), (7, 900), (8, 900),
    // leaf 3
    (10, 1200), (11, 1100), (13, 700),
    // leaf 4
    (15, 1300), (16, 1100), (17, 700),
    // leaf 5
    (18, 1100), (19, 1000), (20, 500),
];

// 2 update workers will be used and the first half of `to_delete` items
// which fall under the same set of leaves are assigned to the first worker
// and all the remaining keys to the next worker. This makes possible
// to expect the type of communication between the two workers
fn insert_delete_and_read(name: impl AsRef<Path>, to_delete: Vec<u8>) {
    let mut t = Test::new_with_params(name, 2, 64_000, None, true);

    // insert values
    for (k, value_size) in KEYS_AND_VALUE_SIZES.clone() {
        t.write(key(k), Some(vec![k; value_size]));
    }
    t.commit();

    // delete values
    for k in to_delete.clone() {
        t.write(key(k), None);
    }
    t.commit();

    // read values
    for (k, value_size) in KEYS_AND_VALUE_SIZES.clone() {
        if to_delete.contains(&k) {
            let res = t.read(key(k));
            assert_eq!(None, res);
        } else {
            let value = vec![k; value_size];
            let res = t.read(key(k));
            assert_eq!(Some(value), res);
        }
    }
}

fn key(id: u8) -> [u8; 32] {
    let mut key = [0; 32];
    key[0] = id;
    key
}

#[test]
fn extend_range_protocol_underfull_to_degenerate_split() {
    insert_delete_and_read("underfull_to_degenerate_split", vec![7, 8, 13])
}

#[test]
fn extend_range_protocol_final_unchanged_range() {
    insert_delete_and_read("final_unchanged_range", vec![7, 8, 10, 11, 13])
}

#[test]
fn extend_range_protocol_unchanged_range_to_changed() {
    insert_delete_and_read("unchanged_range_to_changed", vec![7, 8, 10, 11, 13, 20])
}

#[test]
fn extend_range_protocol_remove_cutoff() {
    insert_delete_and_read(
        "remove_cutoff",
        vec![7, 8, 10, 11, 13, 15, 16, 17, 18, 19, 20],
    );
}

// The second worker merges leaves 2 and 3 and hands the result to the first worker, which splits
// it at the old boundary between those leaves. When the first worker then extends its range
// again, it receives the second worker's note that the old leaf 3 is gone. That note must not
// replace the leaf the first worker has just placed under the same separator.
#[test]
fn extend_range_protocol_keeps_leaf_placed_under_deleted_separator() {
    #[rustfmt::skip]
    const KEYS_AND_VALUE_SIZES: [(u8, usize); 14] = [
        // leaf 1
        (1, 950), (2, 950), (3, 1200),
        // leaf 2
        (4, 1100), (5, 1100), (6, 440), (7, 440),
        // leaf 3
        (8, 1332), (9, 1200), (10, 430), (11, 32),
        // leaf 4
        (12, 1000), (13, 1000), (14, 1000),
    ];
    const TO_DELETE: [u8; 3] = [3, 4, 5];

    let mut t = Test::new_with_params(
        "extend_range_deleted_separator",
        2,      // commit_concurrency
        64_000, // hashtable_buckets
        None,   // panic_on_sync
        true,   // cleanup_dir
    );
    for (k, value_size) in KEYS_AND_VALUE_SIZES {
        t.write(key(k), Some(vec![k; value_size]));
    }
    t.commit();
    for k in TO_DELETE {
        t.write(key(k), None);
    }
    t.commit();

    for (k, value_size) in KEYS_AND_VALUE_SIZES {
        let expected = (!TO_DELETE.contains(&k)).then(|| vec![k; value_size]);
        assert!(t.read(key(k)) == expected, "key {k}");
    }
}
