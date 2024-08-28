#![cfg(feature = "benchmarks")]

use super::ops::update::leaf::benches::separate_benchmark;

pub fn beatree_benchmark(c: &mut criterion::Criterion) {
    separate_benchmark(c);
}
