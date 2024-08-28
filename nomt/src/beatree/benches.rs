#![cfg(feature = "benchmarks")]

use crate::beatree::{
    ops::{
        benches::*,
        update::{branch::benches::*, leaf::benches::*},
    },
    Key,
};

pub fn beatree_benchmark(c: &mut criterion::Criterion) {
    separate_benchmark(c);
    separator_len_benchmark(c);
    prefix_len_benchmark(c);
    search_branch_benchmark(c);
}

// returns two keys a and b where b > a and b shares the first n bits with a
pub fn get_keys(shared_bytes: usize) -> (Key, Key) {
    use rand::RngCore;

    let mut rand = rand::thread_rng();
    let mut a = [0; 32];
    rand.fill_bytes(&mut a[0..shared_bytes]);

    // b > a
    let mut b = a.clone();
    b[shared_bytes] = 1;

    (a, b)
}
