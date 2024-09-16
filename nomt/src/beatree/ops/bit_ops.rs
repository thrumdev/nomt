use bitvec::{order::Msb0, view::BitView};

use crate::beatree::{
    branch::node::{RawPrefix, RawSeparator},
    Key,
};

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

// Reconstruct a key starting from a prefix and a misaligned separator.
// Note: bit offsets starts from 0 going to 7, and the most significant bit is the one with index 0
pub fn reconstruct_key(maybe_prefix: Option<RawPrefix>, separator: RawSeparator) -> Key {
    let mut key = [0u8; 32];

    let prefix_bit_len = maybe_prefix.as_ref().map(|p| p.1).unwrap_or(0);
    let prefix_byte_len = (prefix_bit_len + 7) / 8;
    let prefix_end_bit_offset = prefix_bit_len % 8;
    let (separator_bytes, separator_bit_init, separator_bit_len) = separator;

    // where the separator will start to be stored
    let mut key_offset = match prefix_byte_len {
        0 => 0,
        len if prefix_end_bit_offset == 0 => len,
        // overlap between the end of the prefix and the beginning of the separator
        len => len - 1,
    };

    enum Shift {
        Left(usize),                          // amount
        Right(usize, Option<u8>, Option<u8>), // amount, prev_remainder, curr_remainder
    }

    let mut shift = match prefix_end_bit_offset as isize - separator_bit_init as isize {
        0 => None,
        shift if shift < 0 => Some(Shift::Left(shift.abs() as usize)),
        shift => Some(Shift::Right(shift as usize, None, None)),
    };

    // chunk is an 8-byte slice of the separator which will be cast to a u64 to simplify shifting
    let n_chunks = separator_bytes.len() / 8;
    for chunk_index in 0..n_chunks {
        let chunk_start = chunk_index * 8;

        if let Some(Shift::Right(amount, _prev_remainder, curr_remainder)) = &mut shift {
            // store bits that will be covered by the right shift
            let mask = (1 << *amount) - 1;
            let bits = separator_bytes[chunk_start + 7] & mask;
            *curr_remainder = Some(bits << (8 - *amount));
        }

        let mut chunk = u64::from_be_bytes(
            separator_bytes[chunk_start..chunk_start + 8]
                .try_into()
                .unwrap(),
        );

        if chunk_index == 0 {
            // first chunk will probably have garbage at the beginning of the first byte
            let mask_shift = (7 - separator.1) as u32 + 1 + (8 * 7);
            let mask = 1u64
                .checked_shl(mask_shift)
                .map(|m| m - 1)
                .unwrap_or(u64::MAX);

            chunk &= mask;
        }

        if chunk_index == n_chunks - 1 {
            // last chunk will probably have garbage at end
            let unused_last_bits = separator_bit_init + separator_bit_len - ((n_chunks - 1) * 64);
            let mask = 1u64
                .checked_shl(64 - unused_last_bits as u32)
                .map(|m| !(m - 1))
                .unwrap_or(0);

            chunk &= mask;
        }

        match &mut shift {
            Some(Shift::Left(amount)) => chunk <<= *amount,
            Some(Shift::Right(amount, _, _)) => chunk >>= *amount,
            _ => (),
        };

        let mut chunk_shifted = chunk.to_be_bytes();

        // move bits remainder between chunk bounderies
        match &mut shift {
            Some(Shift::Left(amount)) if chunk_index < n_chunks - 1 => {
                let mask = !(1 << (8 - *amount) - 1);
                let remainder_bits = (separator_bytes[(chunk_index + 1) * 8] & mask) >> *amount;
                chunk_shifted[7] |= remainder_bits;
            }
            Some(Shift::Right(_, prev_remainder, curr_remainder)) => {
                if let Some(bits) = prev_remainder {
                    chunk_shifted[0] |= *bits;
                }
                *prev_remainder = *curr_remainder;
            }
            _ => (),
        };

        // store the shifted chunk into the key
        let n_byte = std::cmp::min(8, 32 - key_offset);
        key[key_offset..key_offset + n_byte].copy_from_slice(&chunk_shifted[..n_byte]);
        key_offset += n_byte;
    }

    if prefix_byte_len != 0 {
        // UNWRAP: prefix_byte_len can be different than 0 only if maybe_prefix is Some
        let prefix = maybe_prefix.unwrap();

        // copy the prefix into the key up to the penultimate byte
        key[0..prefix_byte_len - 1].copy_from_slice(&prefix.0[0..prefix_byte_len - 1]);

        // copy the last byte of the prefix without interfering with the separator
        let mask_shift = 8 - (prefix_bit_len % 8) as u32;
        let mask = 1u8.checked_shl(mask_shift).map(|m| !(m - 1)).unwrap_or(255);
        key[prefix_byte_len - 1] |= prefix.0[prefix_byte_len - 1] & mask;
    }

    key
}

#[cfg(feature = "benchmarks")]
pub mod benches {
    use crate::beatree::benches::get_key_pair;
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
                c.benchmark_group("reconstruct_key_small_separator")
            } else {
                c.benchmark_group("reconstruct_key_big_separator")
            };

            for prefix_bits in [0, 1, 4, 7, 8, 9, 12, 15, 18] {
                let mut rand = rand::thread_rng();
                // 50 to extract multiples of 8 bytes for the separator
                let mut key = [0; 50];
                rand.fill_bytes(&mut key);

                let prefix_bytes = (prefix_bits + 7) / 8;
                let separator_bit_init = prefix_bits % 8;
                let separator_bit_len = if i == 0 {
                    // ensure that the raw separator is made up of only 8 bytes
                    57
                } else {
                    256 - prefix_bits
                };

                let separator_byte_init = if prefix_bytes == 0 {
                    0
                } else if prefix_bits % 8 == 0 {
                    prefix_bytes
                } else {
                    prefix_bytes - 1
                };
                let separator_byte_len = ((separator_bit_len as usize + 7) / 8).next_multiple_of(8);

                group.bench_function(BenchmarkId::new("prefix_len_bits", prefix_bits), |b| {
                    b.iter_batched(
                        || {
                            (
                                (&key[0..prefix_bytes], prefix_bits),
                                (
                                    &key[separator_byte_init..][..separator_byte_len],
                                    separator_bit_init,
                                    separator_bit_len,
                                ),
                            )
                        },
                        |(prefix, separator)| super::reconstruct_key(Some(prefix), separator),
                        criterion::BatchSize::SmallInput,
                    )
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
