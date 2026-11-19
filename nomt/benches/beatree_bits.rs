use criterion::{criterion_group, criterion_main};
use nomt::beatree::benches::bit_packing_benchmark;

criterion_group!(benches, bit_packing_benchmark);
criterion_main!(benches);
