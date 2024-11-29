mod common;
use common::Test;
use rand::{prelude::SliceRandom, Rng, SeedableRng};
use std::time::{SystemTime, UNIX_EPOCH};

fn fill_and_empty(commit_concurrency: usize) {
    //let seed = SystemTime::now()
    //.duration_since(UNIX_EPOCH)
    //.expect("no time?")
    //.as_nanos()
    //.to_le_bytes()[0..16]
    //.try_into()
    //.unwrap();
    //dbg!(&seed);

    // let seed use to create initial storage
    let seed = [249, 5, 3, 240, 129, 92, 12, 24, 0, 0, 0, 0, 0, 0, 0, 0];

    let mut rng = rand_pcg::Lcg64Xsh32::from_seed(seed);

    let db_size = 1 << 25;
    //let commit_size = db_size / 32;
    let commit_size = 2 * 1024 * 1024;

    let mut items = std::collections::BTreeSet::new();
    while items.len() < db_size as usize {
        items.insert(rand_key(&mut rng));
    }
    let items: Vec<_> = items.into_iter().collect();

    //let mut to_delete: Vec<usize> = (0..db_size as usize).collect();
    //to_delete.shuffle(&mut rng);

    let mut t = Test::new_with_params(
        format!("fill_and_empty_{}", commit_concurrency), // name
        commit_concurrency,
        50_000_000, // hashtable_buckets
        false,      // panic_on_sync
        true,       //  cleanup_dir
    );

    // inserting all the values
    //let mut to_check = vec![];
    let mut something_to_commit = false;
    for i in 0..db_size {
        let key = items[i];
        let value = vec![i as u8; 8];

        //to_check.push((key, value.clone()));
        something_to_commit = true;
        t.write(key, Some(value));

        if i != 0 && i % commit_size == 0 {
            println!("committing {}", i);
            something_to_commit = false;
            t.commit();
            // check for presence
            //for (key, value) in to_check.drain(..) {
            //assert_eq!(t.read(key), Some(value));
            //}
        }
    }

    if something_to_commit {
        println!("committing last");
        t.commit();
    }

    //// deleting all the values in different order
    //let mut to_check = vec![];
    //for i in 0..db_size {
    //let key = items[to_delete[i]];
    //
    //to_check.push(key);
    //t.write(key, None);
    //
    //if i != 0 && i % commit_size == 0 {
    //t.commit();
    //// check for absence
    //for key in to_check.drain(..) {
    //assert_eq!(t.read(key), None);
    //}
    //}
    //}
}

fn rand_key(rng: &mut impl Rng) -> [u8; 32] {
    let mut key = [0; 32];
    rng.fill(&mut key[..]);
    key
}

#[test]
fn fill_and_empty_1_commit_worker() {
    fill_and_empty(1);
}

#[test]
fn fill_and_empty_64_commit_worker() {
    fill_and_empty(64);
}
