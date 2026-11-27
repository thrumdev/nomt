//! Seeded branch workloads with uniform-key spacing and optional shared prefixes.
use std::hint::black_box;

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput};
use rand::{rngs::StdRng, Rng, RngExt, SeedableRng};

use crate::{
    beatree::{
        branch::{node::get_key, BranchNode, BranchNodeBuilder, BRANCH_NODE_BODY_SIZE},
        ops::{
            bit_ops::{prefix_len, separate, separator_len},
            search_branch,
        },
        Key,
    },
    io::PagePool,
};

const ITEMS_PER_LEAF: usize = 40;
const LEAVES_PER_BRANCH: usize = 384;
const WINDOWS: usize = 32;
const QUERIES: usize = 4096;

struct Window {
    separators: Vec<Key>,
    queries: Vec<Key>,
    prefix: usize,
    branch: BranchNode,
}

fn windows(log_items: u32, shared_bytes: usize, pool: &PagePool) -> Vec<Window> {
    let mut rng = StdRng::seed_from_u64(0xb17_b7ee_2026 + u64::from(log_items));
    (0..WINDOWS)
        .map(|_| {
            // Poisson approximation to uniform order-statistic spacings. Work is independent
            // of database size. Restrict the start to leave room before the u64 boundary.
            let mut position = rng.random_range(0..u64::MAX - (1u64 << 60));
            let mut next_key = || {
                let u = 1.0 - rng.random::<f64>(); // (0, 1]
                let gap = (-u.ln() * 2f64.powi(64 - log_items as i32)).round() as u64;
                position = position.checked_add(gap.max(1)).unwrap();
                let mut key = [0; 32];
                key[..8].copy_from_slice(&position.to_be_bytes());
                rng.fill_bytes(&mut key[8..]);
                key.copy_within(..32 - shared_bytes, shared_bytes);
                key[..shared_bytes].fill(0xa5);
                key
            };
            let mut previous = next_key();
            let mut separators = Vec::with_capacity(LEAVES_PER_BRANCH);
            let mut queries = Vec::with_capacity(LEAVES_PER_BRANCH);
            for _ in 0..LEAVES_PER_BRANCH {
                let first = next_key();
                separators.push(separate(&previous, &first));
                queries.push(first);
                previous = first;
                for _ in 1..ITEMS_PER_LEAF {
                    previous = next_key();
                }
            }
            let prefix = prefix_len(&separators[0], separators.last().unwrap());
            let lengths: Vec<_> = separators.iter().map(separator_len).collect();
            let packed_bits: usize = lengths.iter().map(|n| n.saturating_sub(prefix)).sum();
            assert!(
                crate::beatree::branch::body_size(prefix, packed_bits, LEAVES_PER_BRANCH)
                    <= BRANCH_NODE_BODY_SIZE
            );
            let mut builder = BranchNodeBuilder::new(
                BranchNode::new_in(pool),
                LEAVES_PER_BRANCH,
                LEAVES_PER_BRANCH,
                prefix,
            );
            for (i, (&key, &len)) in separators.iter().zip(&lengths).enumerate() {
                builder.push(key, len, i as u32);
            }
            let branch = builder.finish();
            // Validate generated fixtures, including non-byte-aligned reconstruction and routing.
            for i in 0..LEAVES_PER_BRANCH {
                assert_eq!(get_key(&branch, i), separators[i]);
                assert_eq!(search_branch(&branch, queries[i]).unwrap().0, i);
            }
            Window {
                separators,
                queries,
                prefix,
                branch,
            }
        })
        .collect()
}

pub fn bit_packing_benchmark(c: &mut Criterion) {
    use rand::seq::SliceRandom;

    let pool = PagePool::new();
    let mut group = c.benchmark_group("beatree_packed");
    for (log_items, shared_bytes) in [(20, 0), (25, 0), (30, 0), (30, 8), (30, 16)] {
        let scenario = format!("2^{log_items}_prefix{}", shared_bytes * 8);
        let fixtures = windows(log_items, shared_bytes, &pool);
        let mut rng = StdRng::seed_from_u64(0x5ea2c4 + u64::from(log_items));
        let queries: Vec<_> = (0..QUERIES)
            .map(|_| {
                (
                    rng.random_range(0..WINDOWS),
                    rng.random_range(0..LEAVES_PER_BRANCH),
                )
            })
            .collect();

        group.throughput(Throughput::Elements(LEAVES_PER_BRANCH as u64));
        group.bench_function(BenchmarkId::new("lengths", &scenario), |b| {
            let mut cursor = 0;
            b.iter(|| {
                let w = &fixtures[queries[cursor % QUERIES].0];
                cursor += 1;
                for key in &w.separators {
                    black_box(separator_len(black_box(key)));
                    black_box(prefix_len(black_box(&w.separators[0]), black_box(key)));
                }
            });
        });
        group.bench_function(BenchmarkId::new("build", &scenario), |b| {
            let mut cursor = 0;
            b.iter_batched(
                || {
                    let w = &fixtures[queries[cursor % QUERIES].0];
                    cursor += 1;
                    (w, BranchNode::new_in(&pool))
                },
                |(w, node)| {
                    let mut builder = BranchNodeBuilder::new(
                        node,
                        LEAVES_PER_BRANCH,
                        LEAVES_PER_BRANCH,
                        w.prefix,
                    );
                    for (i, key) in w.separators.iter().enumerate() {
                        builder.push(*black_box(key), separator_len(black_box(key)), i as u32);
                    }
                    black_box(builder.finish())
                },
                BatchSize::SmallInput,
            );
        });

        // Random full traversals keep the large working set from collapsing to a
        // small repeating subset. Both cases call the production lookup directly.
        group.throughput(Throughput::Elements(1));
        for (name, pages) in [("lookup_128KiB", WINDOWS), ("lookup_512MiB", 131072)] {
            let branches: Vec<_> = (0..pages)
                .map(|i| {
                    let mut node = BranchNode::new_in(&pool);
                    node.as_mut_slice()
                        .copy_from_slice(fixtures[i % WINDOWS].branch.as_slice());
                    node
                })
                .collect();
            let mut schedule: Vec<_> = (0..pages.max(QUERIES))
                .map(|i| (i % pages, rng.random_range(0..LEAVES_PER_BRANCH)))
                .collect();
            schedule.shuffle(&mut rng);
            group.bench_function(BenchmarkId::new(name, &scenario), |b| {
                let mut cursor = 0;
                b.iter(|| {
                    let (page, query) = schedule[cursor % schedule.len()];
                    cursor += 1;
                    black_box(search_branch(
                        black_box(&branches[page]),
                        black_box(fixtures[page % WINDOWS].queries[query]),
                    ))
                });
            });
        }
    }
    group.finish();
}
