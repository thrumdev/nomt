use lazy_static::lazy_static;
use quickcheck::{Arbitrary, Gen, QuickCheck, TestResult};
use rand::{Rng, SeedableRng};
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::fs::File;

use crate::io::{start_test_io_pool, IoPool};
use crate::{
    beatree::{
        allocator::PageNumber,
        branch::BRANCH_NODE_SIZE,
        leaf::node::{body_size, LeafNode, MAX_LEAF_VALUE_SIZE},
        ops::update::{preload_leaves, ChangedNodeEntry, LEAF_MERGE_THRESHOLD},
        writeout, Tree,
    },
    io::PagePool,
};

const DB_INITIAL_CAPACITY: usize = 700;
const CHANGESET_AVG_SIZE: usize = 300;
const MAX_TESTS: u64 = 100;

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
    static ref PAGE_POOL: PagePool = PagePool::new();
    static ref IO_POOL: IoPool = start_test_io_pool(3, PAGE_POOL.clone());
    static ref TREE_DATA: TreeData = init_beatree();
}

struct TreeData {
    ln_fd: File,
    bbn_fd: File,
    ln_freelist_pn: u32,
    bbn_freelist_pn: u32,
    ln_bump: u32,
    bbn_bump: u32,
    init_items: BTreeMap<[u8; 32], Vec<u8>>,
}

// Initialize the beatree utilizing the keys present in ITEMS and creating random value sizes.
//
// This initialization is required for three main reasons:
// + to make quickcheck tests faster, because otherwise, each iteration the db should have been filled up
// + to make reproducibility possible, knowing the seed with which the keys and value sizes are generated
// makes it possible to hardcode them once a bug is found
// + to let the quickcheck input shrinking work better, if each iteration works on the same underlying
// db then the shrinking could be more effective at finding the smallest input that makes the error occur
fn init_beatree() -> TreeData {
    let mut rng = rand_pcg::Lcg64Xsh32::from_seed(*SEED);

    println!("Seed used to initialize db: {:?}", *SEED);

    let ln_fd = tempfile::tempfile().unwrap();
    let bbn_fd = tempfile::tempfile().unwrap();
    ln_fd.set_len(BRANCH_NODE_SIZE as u64).unwrap();
    bbn_fd.set_len(BRANCH_NODE_SIZE as u64).unwrap();

    let beatree = Tree::open(PAGE_POOL.clone(), &IO_POOL, 0, 0, 1, 1, &ln_fd, &bbn_fd, 1).unwrap();

    let mut tree_data = TreeData {
        ln_fd: ln_fd.try_clone().unwrap(),
        bbn_fd: bbn_fd.try_clone().unwrap(),
        ln_freelist_pn: 0,
        bbn_freelist_pn: 0,
        ln_bump: 1,
        bbn_bump: 1,
        init_items: BTreeMap::new(),
    };

    let actuals: Vec<_> = ITEMS
        .iter()
        .cloned()
        .into_iter()
        .map(|key| {
            (
                key,
                Some(vec![170; rng.gen_range(500..MAX_LEAF_VALUE_SIZE)]),
            )
        })
        .collect();
    tree_data
        .init_items
        .extend(actuals.iter().map(|(k, v)| (*k, v.clone().unwrap())));

    beatree.commit(actuals);
    let crate::beatree::WriteoutData {
        bbn,
        bbn_freelist_pages,
        bbn_extend_file_sz,
        ln,
        ln_freelist_pages,
        ln_extend_file_sz,
        ln_freelist_pn,
        ln_bump,
        bbn_freelist_pn,
        bbn_bump,
        bbn_index: _bbn_index,
    } = beatree.prepare_sync();

    writeout::write_bbn(
        IO_POOL.make_handle(),
        &bbn_fd,
        bbn,
        bbn_freelist_pages,
        bbn_extend_file_sz,
    )
    .unwrap();

    writeout::write_ln(
        IO_POOL.make_handle(),
        &ln_fd,
        ln,
        ln_freelist_pages,
        ln_extend_file_sz,
    )
    .unwrap();

    tree_data.ln_freelist_pn = ln_freelist_pn;
    tree_data.ln_bump = ln_bump;
    tree_data.bbn_freelist_pn = bbn_freelist_pn;
    tree_data.bbn_bump = bbn_bump;

    tree_data
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
    commit_concurrency: usize,
    changeset: BTreeMap<[u8; 32], Option<(Vec<u8>, bool)>>,
) -> (
    (Vec<([u8; 32], ChangedNodeEntry<LeafNode>)>, Vec<Vec<u8>>),
    BTreeSet<PageNumber>,
) {
    let beatree = crate::beatree::Tree::open(
        PAGE_POOL.clone(),
        &IO_POOL,
        TREE_DATA.ln_freelist_pn,
        TREE_DATA.bbn_freelist_pn,
        TREE_DATA.ln_bump,
        TREE_DATA.bbn_bump,
        &TREE_DATA.bbn_fd,
        &TREE_DATA.ln_fd,
        commit_concurrency,
    )
    .unwrap();

    let shared = beatree.shared.read();
    let sync = beatree.sync.lock();
    let leaf_reader = &shared.leaf_store_rd;
    let bbn_index = &shared.bbn_index;

    let leaf_cache = preload_leaves(leaf_reader, bbn_index, changeset.keys().cloned()).unwrap();
    let leaf_page_numbers: BTreeSet<PageNumber> = leaf_cache.iter().map(|v| *v.pair().0).collect();

    let leaf_stage_output = super::leaf_stage::run(
        bbn_index,
        leaf_cache,
        leaf_reader,
        PAGE_POOL.clone(),
        changeset.into_iter().collect(),
        sync.tp.clone(),
        commit_concurrency,
    );
    (leaf_stage_output, leaf_page_numbers)
}

fn is_valid_leaf_stage_output(
    output: (Vec<([u8; 32], ChangedNodeEntry<LeafNode>)>, Vec<Vec<u8>>),
    mut used_page_numbers: BTreeSet<PageNumber>,
    deletions: BTreeSet<[u8; 32]>,
    insertions: BTreeMap<[u8; 32], Vec<u8>>,
) -> bool {
    if output.0.is_empty() {
        return true;
    }

    let mut expected_values = TREE_DATA.init_items.clone();
    expected_values.extend(insertions.clone());
    expected_values.retain(|k, _| !deletions.contains(k));

    let mut found_underfull_leaf = false;
    for (_key, changed_leaf) in output.0.into_iter() {
        if let Some(pn) = changed_leaf.deleted {
            used_page_numbers.remove(&pn);
        }

        if let Some(leaf_node) = changed_leaf.inserted {
            let n = leaf_node.n();

            let mut value_size_sum = 0;

            // each leaf must contain all the keys that are expected in the range
            // between its first and last key
            let first = leaf_node.key(0);
            let last = leaf_node.key(n - 1);
            let mut expected = expected_values.range(first..=last);

            for i in 0..n {
                let key = leaf_node.key(i);
                if deletions.contains(&key) {
                    return false;
                }

                let value = leaf_node.value(i).0;
                value_size_sum += value.len();

                let Some((expected_key, expected_value)) = expected.next() else {
                    return false;
                };

                if key != *expected_key || value != expected_value {
                    return false;
                }
            }

            if expected.next().is_some() {
                return false;
            }

            // all new leaves must respect the half-full requirement except for the last one
            if body_size(n, value_size_sum) < LEAF_MERGE_THRESHOLD {
                if found_underfull_leaf == true {
                    return false;
                }
                found_underfull_leaf = true;
            }
        }
    }

    // all preload leaves must be deleted to be a valid output
    used_page_numbers.is_empty()
}

// given an u16 value rescale it proportionally to the new upper bound provided
fn rescale(init: u16, lower_bound: usize, upper_bound: usize) -> usize {
    ((init as f64 / u16::MAX as f64) * ((upper_bound - 1 - lower_bound) as f64)).round() as usize
        + lower_bound
}

// insertions is a map of arbitrary keys associated with value sizes, while deletions
// is a vector of numbers that will be used to index the vector of already present keys
fn leaf_stage_inner(insertions: BTreeMap<Key, u16>, deletions: Vec<u16>) -> TestResult {
    let deletions: BTreeMap<_, _> = deletions
        .into_iter()
        // rescale deletions to contain indexes over only alredy present items in the db
        .map(|d| rescale(d, 0, ITEMS.len()))
        .map(|index| (ITEMS[index], None))
        .collect();

    let insertions: BTreeMap<_, _> = insertions
        .into_iter()
        // rescale raw_size to be between 0 and MAX_LEAF_VALUE_SIZE
        .map(|(k, raw_size)| (k, rescale(raw_size, 1, MAX_LEAF_VALUE_SIZE)))
        .map(|(k, size)| (k.inner, Some((vec![170; size], false))))
        .collect();

    let mut changeset: BTreeMap<[u8; 32], Option<(Vec<u8>, bool)>> = insertions.clone();
    changeset.extend(deletions.clone());

    let (leaf_stage_output, leaf_page_numbers) = exec_leaf_stage(64, changeset);

    TestResult::from_bool(is_valid_leaf_stage_output(
        leaf_stage_output,
        leaf_page_numbers,
        deletions.into_iter().map(|(k, _)| k).collect(),
        insertions
            .into_iter()
            .map(|(k, v)| (k, v.unwrap().0))
            .collect(),
    ))
}

#[test]
fn leaf_stage() {
    QuickCheck::new()
        .gen(Gen::new(CHANGESET_AVG_SIZE))
        .max_tests(MAX_TESTS)
        .quickcheck(leaf_stage_inner as fn(_, _) -> TestResult)
}
