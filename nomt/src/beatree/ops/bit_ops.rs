use bitvec::{order::Msb0, slice::BitSlice, view::BitView};

use crate::beatree::Key;

// separate two keys a and b where b > a
pub fn separate(a: &Key, b: &Key) -> Key {
    // if b > a at some point b must have a 1 where a has a 0 and they are equal up to that point.
    let len = a
        .view_bits::<Msb0>()
        .iter()
        .zip(b.view_bits::<Msb0>().iter())
        .take_while(|(a, b)| a == b)
        .count()
        + 1;

    let mut separator = [0u8; 32];
    separator.view_bits_mut::<Msb0>()[..len].copy_from_bitslice(&b.view_bits::<Msb0>()[..len]);
    separator
}

pub fn prefix_len(key_a: &Key, key_b: &Key) -> usize {
    key_a
        .view_bits::<Msb0>()
        .iter()
        .zip(key_b.view_bits::<Msb0>().iter())
        .take_while(|(a, b)| a == b)
        .count()
}

pub fn separator_len(key: &Key) -> usize {
    if key == &[0u8; 32] {
        return 1;
    }
    let key = &key.view_bits::<Msb0>();
    key.len() - key.trailing_zeros()
}

pub fn reconstruct_key(prefix: Option<&BitSlice<u8, Msb0>>, separator: &BitSlice<u8, Msb0>) -> Key {
    let mut key = [0u8; 32];
    match prefix {
        Some(prefix) => {
            key.view_bits_mut::<Msb0>()[..prefix.len()].copy_from_bitslice(prefix);
            key.view_bits_mut::<Msb0>()[prefix.len()..][..separator.len()]
                .copy_from_bitslice(separator);
        }
        None => {
            key.view_bits_mut::<Msb0>()[..separator.len()].copy_from_bitslice(separator);
        }
    }
    key
}

#[cfg(feature = "benchmarks")]
pub mod benches {
    use crate::beatree::benches::get_key_pair;
    use bitvec::{prelude::Msb0, view::BitView};
    use criterion::{BenchmarkId, Criterion};
    use rand::RngCore;

    pub fn separate_benchmark(c: &mut Criterion) {
        let mut group = c.benchmark_group("separate");

        for shared_bytes in [0, 4, 8, 12, 16] {
            let (key1, key2) = get_key_pair(shared_bytes);
            group.bench_function(BenchmarkId::new("shared_bytes", shared_bytes), |b| {
                b.iter(|| super::separate(&key1, &key2));
            });
        }

        group.finish();
    }

    pub fn reconstruct_key_benchmark(c: &mut Criterion) {
        for i in 0..2 {
            let mut group = if i == 0 {
                c.benchmark_group("reconstruct_key_small_separtor")
            } else {
                c.benchmark_group("reconstruct_key_big_separtor")
            };

            for prefix_bits in [0, 1, 4, 7, 8, 9, 12, 15, 18] {
                let mut rand = rand::thread_rng();
                let mut key = [0; 32];
                rand.fill_bytes(&mut key);

                let prefix = &key.view_bits::<Msb0>()[0..prefix_bits];
                let mut separator = &key.view_bits::<Msb0>()[prefix_bits..];
                if i == 0 {
                    separator = &separator[..64];
                }

                group.bench_function(BenchmarkId::new("prefix_len_bits", prefix_bits), |b| {
                    b.iter(|| super::reconstruct_key(Some(prefix), separator))
                });
            }

            group.finish();
        }
    }

    pub fn separator_len_benchmark(c: &mut Criterion) {
        let mut group = c.benchmark_group("separator_len");

        // n_bytes represents the amount of bytes set to one
        // from the beginning of the key
        for n_bytes in [16, 20, 24, 28, 31].into_iter().rev() {
            let mut separator = [0; 32];
            for byte in separator.iter_mut().take(n_bytes) {
                *byte = 255;
            }

            group.bench_function(BenchmarkId::new("zero_bytes", 32 - n_bytes), |b| {
                b.iter(|| super::separator_len(&separator));
            });
        }

        group.finish();
    }

    pub fn prefix_len_benchmark(c: &mut Criterion) {
        let mut group = c.benchmark_group("prefix_len");

        for prefix_len_bytes in [0, 4, 8, 12, 16] {
            let (key1, key2) = get_key_pair(prefix_len_bytes);
            group.bench_function(BenchmarkId::new("shared_bytes", prefix_len_bytes), |b| {
                b.iter(|| super::prefix_len(&key1, &key2));
            });
        }

        group.finish();
    }
}
