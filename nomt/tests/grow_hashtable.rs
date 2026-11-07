use std::path::Path;

use nomt::{
    grow_hashtable, hasher::Blake3Hasher, trie::KeyPath, validate_hashtable, KeyReadWrite, Nomt,
    Options, Root, SessionParams, WitnessMode,
};

fn options(path: &Path, buckets: u32, rollback: bool) -> Options {
    let mut options = Options::new();
    options.path(path);
    options.bitbox_seed([0; 16]);
    options.hashtable_buckets(buckets);
    options.io_workers(1);
    options.rollback(rollback);
    options.preallocate_ht(false);
    options
}

fn key(i: u64) -> KeyPath {
    let mut input = [0u8; 32];
    input[24..].copy_from_slice(&i.to_be_bytes());
    *blake3::hash(&input).as_bytes()
}

fn value(i: u64) -> Vec<u8> {
    let mut value = Vec::with_capacity(16);
    value.extend_from_slice(&i.to_le_bytes());
    value.extend_from_slice(&(i * 7).to_le_bytes());
    value
}

fn commit_range(nomt: &Nomt<Blake3Hasher>, range: std::ops::Range<u64>) -> Root {
    let session =
        nomt.begin_session(SessionParams::default().witness_mode(WitnessMode::read_write()));
    let mut operations = range
        .map(|i| (key(i), KeyReadWrite::Write(Some(value(i)))))
        .collect::<Vec<_>>();
    operations.sort_by_key(|(key, _)| *key);
    for (key, _) in &operations {
        session.warm_up(*key);
    }

    let finished = session.finish(operations).unwrap();
    let root = finished.root();
    finished.commit(nomt).unwrap();
    root
}

#[test]
fn grow_hashtable_preserves_data_and_rollback() {
    let tempdir = tempfile::tempdir().unwrap();
    let path = tempdir.path();

    let nomt = Nomt::<Blake3Hasher>::open(options(path, 4096, true)).unwrap();
    let root_1 = commit_range(&nomt, 0..40);
    let root_2 = commit_range(&nomt, 40..80);
    assert_eq!(nomt.root(), root_2);
    drop(nomt);

    grow_hashtable(&options(path, 8192, true)).unwrap();
    let utilization = validate_hashtable(&options(path, 8192, true)).unwrap();
    assert_eq!(utilization.capacity, 8192);
    assert!(utilization.occupied > 0);

    let nomt = Nomt::<Blake3Hasher>::open(options(path, 4096, true)).unwrap();
    assert_eq!(nomt.root(), root_2);
    assert_eq!(nomt.hash_table_utilization().capacity, 8192);
    assert!(nomt.hash_table_utilization().occupied > 0);

    for i in 0..80 {
        assert_eq!(nomt.read(key(i)).unwrap(), Some(value(i)));
    }

    nomt.rollback(1).unwrap();
    assert_eq!(nomt.root(), root_1);
    for i in 0..40 {
        assert_eq!(nomt.read(key(i)).unwrap(), Some(value(i)));
    }
    for i in 40..80 {
        assert_eq!(nomt.read(key(i)).unwrap(), None);
    }

    let root_3 = commit_range(&nomt, 100..110);
    assert_eq!(nomt.root(), root_3);
    assert_eq!(nomt.hash_table_utilization().capacity, 8192);
}
