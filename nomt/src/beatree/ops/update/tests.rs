use lazy_static::lazy_static;
use quickcheck::{Arbitrary, Gen, QuickCheck, TestResult};
use rand::{Rng, SeedableRng};
use std::collections::BTreeSet;
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

use crate::{
    beatree::{
        allocator::PageNumber,
        leaf::node::{body_size, LeafNode, MAX_LEAF_VALUE_SIZE},
        ops::update::{preload_leaves, ChangedNodeEntry, LEAF_MERGE_THRESHOLD},
    },
    io::PagePool,
    store::Store,
    KeyReadWrite, Nomt, Options,
};

const DB_INITIAL_CAPACITY: usize = 300_000;
const CHANGESET_AVG_SIZE: usize = 10_000;
const MAX_TESTS: u64 = 100;

const LEAF_STAGE_PATH: &'static str = "leaf_stage";
const LEAF_STAGE_EXTEND_RANGE_PATH: &'static str = "leaf_stage_extend_range";

// Required to increase reproducibility
lazy_static! {
    static ref SEED: [u8; 16] = {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("no time?")
            .as_nanos()
            .to_le_bytes()[0..16]
            .try_into()
            .unwrap()
    };
    static ref ITEMS: Vec<[u8; 32]> = {
        let mut rng = rand_pcg::Lcg64Xsh32::from_seed(*SEED);
        let mut items = BTreeSet::new();
        while items.len() < DB_INITIAL_CAPACITY {
            let mut key = [0; 32];
            rng.fill(&mut key);
            items.insert(key);
        }
        items.into_iter().collect()
    };
}

fn opts(name: impl AsRef<Path>, commit_concurrency: usize) -> Options {
    let mut opts = Options::new();
    let path = {
        let mut p = PathBuf::from("test");
        p.push(name);
        p
    };
    opts.path(path);
    opts.bitbox_seed([0; 16]);
    opts.commit_concurrency(commit_concurrency);
    opts.hashtable_buckets(DB_INITIAL_CAPACITY as u32 * 5);
    opts
}

// Initialize nomt utilizing the keys present in ITEMS and creating random value sizes.
//
// This initialization is required for three main reasons:
// + to make quickcheck tests faster, because otherwise, each iteration the db should have been filled up
// + to make reproducibility possible, knowing the seed with which the keys and value sizes are generated
// makes it possible to hardcode them once a bug is found
// + to let the quickcheck input shrinking work better, if each iteration works on the same underlying
// db then the shrinking could be more effective at finding the smallest input that makes the error occur
//
fn init_db(name: impl AsRef<Path>) {
    let mut rng = rand_pcg::Lcg64Xsh32::from_seed(*SEED);

    let opts = opts(name, 64);
    let _ = std::fs::remove_dir_all(&opts.path);

    println!(
        "Seed used to initialize {}: {:?}",
        opts.path.to_string_lossy(),
        *SEED
    );

    let nomt = Nomt::open(opts).unwrap();

    const MAX_INIT_PER_ITERATION: usize = 2 * 1024 * 1024;

    for items_chunk in ITEMS
        .iter()
        .cloned()
        .collect::<Vec<_>>()
        .chunks_mut(MAX_INIT_PER_ITERATION)
    {
        let actuals = items_chunk
            .into_iter()
            .map(|key| {
                (
                    *key,
                    KeyReadWrite::Write(Some(vec![170; rng.gen_range(500..MAX_LEAF_VALUE_SIZE)])),
                )
            })
            .collect();

        let session = nomt.begin_session();
        nomt.commit(session, actuals).unwrap();
    }
}

#[derive(Clone, Debug, Eq, PartialEq, PartialOrd, Ord)]
struct Key {
    inner: [u8; 32],
}

// required to let quickcheck generate arbitrary keys as arguments for the tests
impl Arbitrary for Key {
    fn arbitrary(g: &mut Gen) -> Key {
        let mut key = [0; 32];
        for k in key.iter_mut() {
            *k = u8::arbitrary(g);
        }
        Key { inner: key }
    }
}

// given a changeset execute the leaf stage on top of a pre-initialize nomt-db
fn exec_leaf_stage(
    name: impl AsRef<Path>,
    commit_concurrency: usize,
    changeset: BTreeMap<[u8; 32], Option<(Vec<u8>, bool)>>,
) -> (
    (Vec<([u8; 32], ChangedNodeEntry<LeafNode>)>, Vec<Vec<u8>>),
    BTreeSet<PageNumber>,
) {
    let opts = opts(name, commit_concurrency);
    let page_pool = PagePool::new();
    let store = Store::open(&opts, page_pool.clone()).unwrap();
    let beatree = store.get_beatree();
    let leaf_reader = &beatree.shared.read().leaf_store_rd;
    let bbn_index = &beatree.shared.read().bbn_index;
    let sync = beatree.sync.lock();

    let leaf_cache = preload_leaves(leaf_reader, bbn_index, changeset.keys().cloned()).unwrap();
    let leaf_page_numbers: BTreeSet<PageNumber> = leaf_cache.iter().map(|v| *v.pair().0).collect();

    let leaf_stage_output = super::leaf_stage::run(
        bbn_index,
        leaf_cache,
        leaf_reader,
        page_pool,
        changeset.into_iter().collect(),
        sync.tp.clone(),
        sync.commit_concurrency,
    );
    (leaf_stage_output, leaf_page_numbers)
}

fn is_valid_leaf_stage_output(
    output: (Vec<([u8; 32], ChangedNodeEntry<LeafNode>)>, Vec<Vec<u8>>),
    mut used_page_numbers: BTreeSet<PageNumber>,
) -> bool {
    let last_changed_leaf = output.0.len() - 1;
    for (i, (_key, changed_leaf)) in output.0.into_iter().enumerate() {
        if let Some(pn) = changed_leaf.deleted {
            used_page_numbers.remove(&pn);
        }

        if let Some(leaf_node) = changed_leaf.inserted {
            let n = leaf_node.n();
            let value_size_sum = (0..n).map(|j| leaf_node.value(j).0.len()).sum();

            // all new leaves must respect the half-full requirement except for the last one
            if i != last_changed_leaf && body_size(n, value_size_sum) < LEAF_MERGE_THRESHOLD {
                return false;
            }
        }
    }

    // all preload leaves must be deleted to be a valid output
    used_page_numbers.is_empty()
}

// given an u16 value rescale it proportionally to the new upper bound provided
fn rescale(init: u16, new_bound: usize) -> usize {
    ((init as f64 / u16::MAX as f64) * ((new_bound - 1) as f64)).round() as usize
}

// insertions is a map of arbitrary keys associated with value sizes, while deletions
// is a vector of numbers that will be used to index the vector of already present keys
fn leaf_stage_inner(insertions: BTreeMap<Key, u16>, deletions: Vec<u16>) -> TestResult {
    let deletions: BTreeMap<_, _> = deletions
        .into_iter()
        // rescale deletions to contain indexes over only alredy present items in the db
        .map(|d| rescale(d, ITEMS.len()))
        .map(|index| (ITEMS[index], None))
        .collect();

    let insertions: BTreeMap<_, _> = insertions
        .into_iter()
        // rescale raw_size to be between 0 and MAX_LEAF_VALUE_SIZE
        .map(|(k, raw_size)| (k, rescale(raw_size, MAX_LEAF_VALUE_SIZE)))
        .map(|(k, size)| (k.inner, Some((vec![170; size], false))))
        .collect();

    let mut changeset: BTreeMap<[u8; 32], Option<(Vec<u8>, bool)>> = insertions;
    changeset.extend(deletions);

    let (leaf_stage_output, leaf_page_numbers) = exec_leaf_stage(LEAF_STAGE_PATH, 10, changeset);

    TestResult::from_bool(is_valid_leaf_stage_output(
        leaf_stage_output,
        leaf_page_numbers,
    ))
}

// Insertions and Deletions are vectors of numbers used to perform operations around core points.
// This increases the likelihood of having those points at the boundaries between workers,
// thus augmenting the utilization of the range extension protocol
fn leaf_stage_stress_range_extension_protocol_inner(
    insertions: Vec<(u16, u16)>,
    deletions: Vec<u16>,
) -> TestResult {
    if insertions.len() < 64 || deletions.len() < 64 {
        return TestResult::discard();
    }

    // those are the points where the work will be focused,
    // the changeset will be uniformly distributed among these points
    let core_points: Vec<isize> = (1..64).map(|i| i * (ITEMS.len() / 64) as isize).collect();
    let core_points_distance = ITEMS.len() / 64;
    let delta = core_points_distance / 5;

    // there are 63 core_otpoints, let's split the deletions around those points
    let chunk_size = deletions.len() / 63;
    let deletions: BTreeMap<_, _> = deletions
        .chunks_exact(chunk_size)
        .enumerate()
        .take(63)
        .map(|(i, deletion_chunk)| {
            deletion_chunk
                .into_iter()
                .map(|d| {
                    // each deletion will be rescaled to be an offset relative to a core point
                    (rescale(*d, delta) as isize) - (delta / 2) as isize
                })
                .map(|index| (ITEMS[(core_points[i] + index) as usize], None))
                .collect::<Vec<(_, Option<(Vec<u8>, bool)>)>>()
        })
        .flatten()
        .collect();

    // insertions also are positioned around the core_points
    let chunk_size = insertions.len() / 63;
    let insertions: BTreeMap<_, _> = insertions
        .chunks_exact(chunk_size)
        .enumerate()
        .take(63)
        .map(|(i, deletion_chunk)| {
            deletion_chunk
                .into_iter()
                .map(|(d, size)| {
                    // each inserion will be rescaled to be an offset relative to a core point
                    (
                        (rescale(*d, delta) as isize) - (delta / 2) as isize,
                        rescale(*size, MAX_LEAF_VALUE_SIZE),
                    )
                })
                // the key is slightly altered to create a new key,
                // it is very unlikely that it already exists
                .map(|(index, size)| {
                    let mut k = ITEMS[(core_points[i] + index) as usize];
                    k[31] = k[31].wrapping_add(k[30]);
                    (k, Some((vec![170; size], false)))
                })
                .collect::<Vec<(_, Option<(Vec<u8>, bool)>)>>()
        })
        .flatten()
        .collect();

    let mut changeset: BTreeMap<[u8; 32], Option<(Vec<u8>, bool)>> = insertions;
    changeset.extend(deletions);

    let (leaf_stage_output, leaf_page_numbers) =
        exec_leaf_stage(LEAF_STAGE_EXTEND_RANGE_PATH, 64, changeset);

    TestResult::from_bool(is_valid_leaf_stage_output(
        leaf_stage_output,
        leaf_page_numbers,
    ))
}

#[test]
fn leaf_stage() {
    init_db(LEAF_STAGE_PATH);
    QuickCheck::new()
        .gen(Gen::new(CHANGESET_AVG_SIZE))
        .max_tests(MAX_TESTS)
        .quickcheck(leaf_stage_inner as fn(_, _) -> TestResult)
}

#[test]
fn leaf_stage_range_extension_protocol() {
    init_db(LEAF_STAGE_EXTEND_RANGE_PATH);
    QuickCheck::new()
        .gen(Gen::new(CHANGESET_AVG_SIZE))
        .max_tests(MAX_TESTS)
        .quickcheck(leaf_stage_stress_range_extension_protocol_inner as fn(_, _) -> TestResult)
}
