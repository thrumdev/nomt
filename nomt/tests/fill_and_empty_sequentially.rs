mod common;
use common::Test;
use rand::{prelude::SliceRandom, Rng, SeedableRng};
use std::time::{SystemTime, UNIX_EPOCH};

fn fill_seq(commit_concurrency: usize) {
    //let seed = SystemTime::now()
    //.duration_since(UNIX_EPOCH)
    //.expect("no time?")
    //.as_nanos()
    //.to_le_bytes()[0..16]
    //.try_into()
    //.unwrap();
    //dbg!(&seed);
    //let seed = [204, 105, 172, 143, 116, 85, 12, 24, 0, 0, 0, 0, 0, 0, 0, 0];

    let db_size = 1 << 20;
    let commit_size = db_size / 8;

    //let mut items = std::collections::BTreeSet::new();
    //while items.len() < db_size as usize {
    //items.insert(rand_key(&mut rng));
    //}
    let items: Vec<_> = (0..db_size as u64)
        .into_iter()
        .map(|n| {
            let mut buf = [0; 32];
            //buf[0..8].copy_from_slice(&n.to_le_bytes());
            buf[0..8].copy_from_slice(&n.to_be_bytes());
            buf
        })
        .collect();

    let mut t = Test::new_with_params(
        format!("fill_seq{}", commit_concurrency), // name
        commit_concurrency,
        10_000_000, // hashtable_buckets
        false,      // panic_on_sync
        true,       //  cleanup_dir
    );

    // inserting all the values
    let mut to_check = vec![];
    for i in 0..db_size {
        let key = items[i];
        let value = vec![i as u8; 400];

        to_check.push((key, value.clone()));
        t.write(key, Some(value));

        if i != 0 && i % commit_size == 0 {
            println!("committing {}", i);
            t.commit();
            // check for presence
            for (key, value) in to_check.drain(..) {
                assert_eq!(t.read(key), Some(value));
            }
        }
    }
}

#[test]
fn fill_seq_1() {
    fill_seq(1);
}

#[test]
fn fill_seq_64() {
    fill_seq(64);
}
