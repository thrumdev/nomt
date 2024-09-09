use bitvec::{order::Msb0, view::BitView};

use crate::beatree::{
    branch::node::{BranchNodePrefix, BranchNodeSeparator},
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
// Note: bit offsets starts from 0 going to 7, and inside the byte the msb is the one with index 0
pub fn reconstruct_key(
    maybe_prefix: Option<BranchNodePrefix>,
    separator: BranchNodeSeparator,
) -> Key {
    let mut key = [0u8; 32];

    let prefix_bit_len = maybe_prefix.as_ref().map(|p| p.bit_len).unwrap_or(0);
    let prefix_byte_len = (prefix_bit_len + 7) / 8;
    let prefix_end_bit_offset = prefix_bit_len % 8;

    enum Shift {
        Left(usize),                          // amount
        Right(usize, Option<u8>, Option<u8>), // amount, prev_remainder, curr_remainder
    }

    let mut shift = match prefix_end_bit_offset as isize - separator.bit_init as isize {
        0 => None,
        shift if shift < 0 => Some(Shift::Left(shift.abs() as usize)),
        shift => Some(Shift::Right(shift as usize, None, None)),
    };

    // where the separator will start to be stored
    let mut key_offset = match prefix_byte_len {
        0 => 0,
        len if prefix_end_bit_offset == 0 => len,
        // overlap between the end of the prefix and the beginning of the separator
        len => len - 1,
    };

    // SAFETY: The separator always contains a slice of a length that is a multiple of 8,
    // and the page from which the separator is being read is correctly allocated
    let separator_chunks: &[[u8; 8]] = unsafe {
        std::slice::from_raw_parts(
            separator.bytes.as_ptr() as *const [u8; 8],
            separator.bytes.len() / 8,
        )
    };

    for i in 0..separator_chunks.len() {
        if let Some(Shift::Right(amount, _prev_remainder, curr_remainder)) = &mut shift {
            // store bits that will be covered by the right shift
            let mask = (1 << *amount) - 1;
            let bits = separator_chunks[i][7] & mask;
            *curr_remainder = Some(bits << (8 - *amount));
        }

        let mut separator_chunk = u64::from_be_bytes(separator_chunks[i]);

        if i == 0 {
            // first chunk will probably have garbage at the beginning of the first byte
            let mask_shift = (7 - separator.bit_init) as u32 + 1 + (8 * 7);
            let mask = 1u64
                .checked_shl(mask_shift)
                .map(|m| m - 1)
                .unwrap_or(u64::MAX);

            separator_chunk &= mask;
        }

        if i == separator_chunks.len() - 1 {
            // last chunk will probably have garbage at end
            let n_chunks = (separator.bytes.len() / 8) - 1;
            let unused_last_bits = separator.bit_len + separator.bit_init - (n_chunks * 64);
            let mask = 1u64
                .checked_shl(64 - unused_last_bits as u32)
                .map(|m| !(m - 1))
                .unwrap_or(0);

            separator_chunk &= mask;
        }

        match &mut shift {
            Some(Shift::Left(amount)) => separator_chunk <<= *amount,
            Some(Shift::Right(amount, _, _)) => separator_chunk >>= *amount,
            _ => (),
        };

        let mut separator_chunk_shifted = separator_chunk.to_be_bytes();

        // move bits remainder between chunk bounderies
        match &mut shift {
            Some(Shift::Left(amount)) if i < separator_chunks.len() - 1 => {
                let mask = !(1 << (8 - *amount) - 1);
                let remainder_bits = (separator_chunks[i + 1][0] & mask) >> *amount;
                separator_chunk_shifted[7] |= remainder_bits;
            }
            Some(Shift::Right(_, prev_remainder, curr_remainder)) => {
                if let Some(bits) = prev_remainder {
                    separator_chunk_shifted[0] |= *bits;
                }
                *prev_remainder = *curr_remainder;
            }
            _ => (),
        };

        // store the shifted chunk into the key
        let n_byte = std::cmp::min(8, 32 - key_offset);
        key[key_offset..key_offset + n_byte].copy_from_slice(&separator_chunk_shifted[..n_byte]);
        key_offset += n_byte;
    }

    if prefix_byte_len != 0 {
        // UNWRAP: prefix_byte_len can be different than 0 only if maybe_prefix is Some
        let prefix = maybe_prefix.unwrap();

        // copy the prefix into the key up to the penultimate byte
        key[0..prefix_byte_len - 1].copy_from_slice(&prefix.bytes[0..prefix_byte_len - 1]);

        // copy the last byte of the prefix without interfering with the separator
        let mask_shift = 8 - (prefix_bit_len % 8) as u32;
        let mask = 1u8.checked_shl(mask_shift).map(|m| !(m - 1)).unwrap_or(255);
        key[prefix_byte_len - 1] |= prefix.bytes[prefix_byte_len - 1] & mask;
    }

    key
}

#[cfg(feature = "benchmarks")]
pub mod benches {
    use crate::beatree::{
        benches::get_key_pair,
        branch::node::{BranchNodePrefix, BranchNodeSeparator},
    };
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
                // 50 to extract multiples of 8 bytes for the separator
                let mut key = [0; 50];
                rand.fill_bytes(&mut key);

                let prefix_bytes = (prefix_bits + 7) / 8;
                let separator_bit_init = prefix_bits % 8;
                let separator_bit_len = if i == 0 {
                    57 // 57 to ensure the separator to use less the 8 bytes
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
                                BranchNodePrefix {
                                    bit_len: prefix_bits,
                                    bytes: &key[0..prefix_bytes],
                                },
                                BranchNodeSeparator {
                                    bit_init: separator_bit_init,
                                    bit_len: separator_bit_len,
                                    bytes: &key[separator_byte_init..][..separator_byte_len],
                                },
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
