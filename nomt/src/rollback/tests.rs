use std::fs::OpenOptions;

use super::{BTreeMap, KeyPath, KeyReadWrite, LoadValue, Rollback};
use hex_literal::hex;

const MAX_ROLLBACK_LOG_LEN: u32 = 100;

/// A mock implementation of `LoadValue` for testing. Describes the "current" state of the
/// database.
#[derive(Clone)]
struct MockStore {
    values: BTreeMap<KeyPath, Option<Vec<u8>>>,
}

impl MockStore {
    fn new() -> Self {
        Self {
            values: BTreeMap::new(),
        }
    }

    fn insert(&mut self, key_path: KeyPath, value: Option<Vec<u8>>) {
        self.values.insert(key_path, value);
    }
}

impl LoadValue for MockStore {
    fn load_value(&self, key_path: KeyPath) -> anyhow::Result<Option<Vec<u8>>> {
        match self.values.get(&key_path) {
            Some(value) => return Ok(value.clone()),
            None => panic!("the caller requested a value that was not inserted by the test"),
        }
    }
}

#[test]
fn truncate_works() {
    let temp_dir = tempfile::tempdir().unwrap();
    let db_dir_path = temp_dir.path().join("db");
    std::fs::create_dir_all(&db_dir_path).unwrap();
    let db_dir_fd = OpenOptions::new()
        .read(true)
        .open(db_dir_path.clone())
        .unwrap();

    let mut store = MockStore::new();
    store.insert(
        hex!("0101010101010101010101010101010101010101010101010101010101010101"),
        Some(b"old_value1".to_vec()),
    );
    store.insert(
        hex!("0202020202020202020202020202020202020202020202020202020202020202"),
        Some(b"old_value2".to_vec()),
    );
    store.insert(
        hex!("0303030303030303030303030303030303030303030303030303030303030303"),
        Some(b"old_value3".to_vec()),
    );

    let rollback = Rollback::read(MAX_ROLLBACK_LOG_LEN, db_dir_path, db_dir_fd, 0, 0).unwrap();
    let builder = rollback.delta_builder();
    builder.tentative_preserve_prior(store.clone(), [1; 32]);
    builder.tentative_preserve_prior(store.clone(), [2; 32]);
    builder.tentative_preserve_prior(store.clone(), [3; 32]);
    rollback
        .commit(
            store.clone(),
            &[
                (
                    hex!("0101010101010101010101010101010101010101010101010101010101010101"),
                    KeyReadWrite::Write(Some(b"new_value1".to_vec())),
                ),
                (
                    hex!("0202020202020202020202020202020202020202020202020202020202020202"),
                    KeyReadWrite::Write(Some(b"new_value2".to_vec())),
                ),
            ],
            builder,
        )
        .unwrap();

    // We want to see the old values for all the keys that have been changed during the commit.
    let traceback = rollback.truncate(1).unwrap().unwrap();
    assert_eq!(traceback.len(), 2);
    assert_eq!(
        traceback
            .get(&hex!(
                "0101010101010101010101010101010101010101010101010101010101010101"
            ))
            .unwrap()
            .clone(),
        Some(b"old_value1".to_vec())
    );
    assert_eq!(
        traceback
            .get(&hex!(
                "0202020202020202020202020202020202020202020202020202020202020202"
            ))
            .unwrap()
            .clone(),
        Some(b"old_value2".to_vec())
    );
}

#[test]
fn without_tentative_preserve_prior() {
    // A test where we don't call tentative_preserve_prior and expect that the commit will
    // fetch the values itself.
    let temp_dir = tempfile::tempdir().unwrap();
    let db_dir_path = temp_dir.path().join("db");
    std::fs::create_dir_all(&db_dir_path).unwrap();
    let db_dir_fd = OpenOptions::new()
        .read(true)
        .open(db_dir_path.clone())
        .unwrap();

    let mut store = MockStore::new();
    store.insert(
        hex!("0101010101010101010101010101010101010101010101010101010101010101"),
        Some(b"old_value1".to_vec()),
    );
    store.insert(
        hex!("0202020202020202020202020202020202020202020202020202020202020202"),
        Some(b"old_value2".to_vec()),
    );
    store.insert(
        hex!("0303030303030303030303030303030303030303030303030303030303030303"),
        Some(b"old_value3".to_vec()),
    );

    let rollback = Rollback::read(MAX_ROLLBACK_LOG_LEN, db_dir_path, db_dir_fd, 0, 0).unwrap();
    let builder = rollback.delta_builder();
    rollback
        .commit(
            store.clone(),
            &[
                (
                    hex!("0101010101010101010101010101010101010101010101010101010101010101"),
                    KeyReadWrite::Write(Some(b"new_value1".to_vec())),
                ),
                (
                    hex!("0202020202020202020202020202020202020202020202020202020202020202"),
                    KeyReadWrite::Write(Some(b"new_value2".to_vec())),
                ),
            ],
            builder,
        )
        .unwrap();

    // We want to see the old values for all the keys that have been changed during the commit.
    let traceback = rollback.truncate(1).unwrap().unwrap();
    assert_eq!(traceback.len(), 2);
    assert_eq!(
        traceback
            .get(&hex!(
                "0101010101010101010101010101010101010101010101010101010101010101"
            ))
            .unwrap()
            .clone(),
        Some(b"old_value1".to_vec())
    );
    assert_eq!(
        traceback
            .get(&hex!(
                "0202020202020202020202020202020202020202020202020202020202020202"
            ))
            .unwrap()
            .clone(),
        Some(b"old_value2".to_vec())
    );
}
