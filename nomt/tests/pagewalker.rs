// Use: cargo test --test pagewalker

use nomt::{KeyReadWrite, Nomt, Options};
use std::{collections::BTreeMap, path::PathBuf};

const N: usize = 10;

#[test]
fn test_hashed_concurrent() {
    do_test(
        "hashed_concurrent",
        /* remove_prior_data */ false,
        /* bump_first_byte */ false,
        /* should_hash_key */ true,
        /* commit_concurrency */ 10,
    );
}

#[test]
fn test_hashed_nonconcurrent() {
    do_test(
        "hashed_nonconcurrent",
        /* remove_prior_data */ false,
        /* bump_first_byte */ false,
        /* should_hash_key */ true,
        /* commit_concurrency */ 1,
    );
}

#[test]
fn test_unhashed_concurrent() {
    do_test(
        "unhashed_concurrent",
        /* remove_prior_data */ false,
        /* bump_first_byte */ false,
        /* should_hash_key */ false,
        /* commit_concurrency */ 10,
    );
}

#[test]
fn test_unhashed_nonconcurrent() {
    do_test(
        "unhashed_nonconcurrent",
        /* remove_prior_data */ false,
        /* bump_first_byte */ false,
        /* should_hash_key */ false,
        /* commit_concurrency */ 1,
    );
}

#[test]
fn test_unhashed_nonconcurrent_but_remove_prior_data() {
    do_test(
        "unhashed_nonconcurrent_but_remove_prior_data",
        /* remove_prior_data */ true,
        /* bump_first_byte */ false,
        /* should_hash_key */ false,
        /* commit_concurrency */ 1,
    );
}

#[test]
fn test_bump_first_byte() {
    do_test(
        "bump_first_byte",
        /* remove_prior_data */ false,
        /* bump_first_byte */ true,
        /* should_hash_key */ false,
        /* commit_concurrency */ 1,
    );
}

#[test]
fn test_remove_prior_data() {
    do_test(
        "remove_prior_data",
        /* remove_prior_data */ true,
        /* bump_first_byte */ false,
        /* should_hash_key */ false,
        /* commit_concurrency */ 1,
    );
    // do_test(
    //     "remove_prior_data",
    //     /* remove_prior_data */ false,
    //     /* bump_first_byte */ false,
    //     /* should_hash_key */ false,
    //     /* commit_concurrency */ 1,
    // );
}

fn do_test(
    name: &str,
    remove_prior_data: bool,
    bump_first_byte: bool,
    should_hash_key: bool,
    commit_concurrency: usize,
) {
    let mut to_insert = Vec::new();
    for commit_ix in 0..N {
        let mut per_commit_insert = BTreeMap::new();

        // Add 3 new keys each iteration
        for j in 0..3 {
            let mut key = [0u8; 32];
            if bump_first_byte {
                key[0] = 128;
            }
            key[30] = commit_ix as u8;
            key[31] = j as u8;
            let key = if should_hash_key {
                *blake3::hash(&key).as_bytes()
            } else {
                key
            };
            per_commit_insert.insert(key, vec![]);
        }

        to_insert.push(per_commit_insert);
    }

    let mut o = Options::new();
    let tmp_path = {
        let mut p = PathBuf::from("test");
        p.push(format!("test_fail_{name}"));
        p
    };
    if remove_prior_data && tmp_path.exists() {
        std::fs::remove_dir_all(&tmp_path).unwrap();
    }
    o.path(tmp_path);
    o.commit_concurrency(commit_concurrency);
    o.panic_on_sync(false);
    o.bitbox_seed([0; 16]);
    let nomt = Nomt::open(o).unwrap();
    for commit_no in 0..N {
        println!("commit_no: {}", commit_no);
        println!("adding keys: {:x?}", to_insert[commit_no]);
        let session = nomt.begin_session();
        let mut operations = Vec::new();
        for (key, value) in to_insert[commit_no].iter() {
            operations.push((key.clone(), KeyReadWrite::Write(Some(value.clone()))));
        }
        nomt.commit(session, operations).unwrap();
    }
}
