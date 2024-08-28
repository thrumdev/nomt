#![cfg(feature = "benchmarks")]

use super::ops::update::{
    branch::benches::separator_len_benchmark, leaf::benches::separate_benchmark,
};

pub fn beatree_benchmark(c: &mut criterion::Criterion) {
    separate_benchmark(c);
    separator_len_benchmark(c);
}
