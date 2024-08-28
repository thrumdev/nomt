#![cfg(feature = "benchmarks")]

use criterion::{criterion_group, criterion_main};
use nomt::beatree::benches::beatree_benchmark;

criterion_group!(benches, beatree_benchmark);
criterion_main!(benches);
