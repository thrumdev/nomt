mod common;

use common::account_path;
use nomt::{hasher::Blake3Hasher, KeyReadWrite, Nomt, Options, SessionParams, WitnessMode};
use std::path::{Path, PathBuf};

const ACCOUNTS: u64 = 20_000;
const OVERFLOW_ID: u64 = ACCOUNTS;
// Larger than the maximum value stored inline in a leaf, so it spans several overflow pages.
const OVERFLOW_VALUE_LEN: usize = 64 * 1024;

fn open(path: &Path) -> Nomt<Blake3Hasher> {
    let mut o = Options::new();
    o.path(path);
    o.metrics(true);
    o.bitbox_seed([0; 16]);
    o.hashtable_buckets(64_000);
    o.commit_concurrency(1);
    o.io_workers(1);
    // Small caches so that a cold reopen has to go to disk.
    o.page_cache_size(1);
    o.leaf_cache_size(1);
    Nomt::open(o).unwrap()
}

fn commit(nomt: &Nomt<Blake3Hasher>, mut actuals: Vec<(nomt::trie::KeyPath, KeyReadWrite)>) {
    actuals.sort_by_key(|(k, _)| *k);
    let session =
        nomt.begin_session(SessionParams::default().witness_mode(WitnessMode::disabled()));
    for (key, _) in &actuals {
        session.warm_up(*key);
    }
    session.finish(actuals).unwrap().commit(nomt).unwrap();
}

/// Disk reads of the commit path (merkle seek and beatree update) and of value reads must show up
/// in the metrics, not only page cache lookups.
#[test]
fn metrics_count_disk_reads() {
    let path = PathBuf::from("test/metrics_count_disk_reads");
    let _ = std::fs::remove_dir_all(&path);

    {
        let nomt = open(&path);
        let mut actuals: Vec<_> = (0..ACCOUNTS)
            .map(|id| {
                (
                    account_path(id),
                    KeyReadWrite::Write(Some(id.to_le_bytes().to_vec())),
                )
            })
            .collect();
        actuals.push((
            account_path(OVERFLOW_ID),
            KeyReadWrite::Write(Some(vec![7; OVERFLOW_VALUE_LEN])),
        ));
        commit(&nomt, actuals);
    }

    // Reopen with cold caches.
    let nomt = open(&path);
    let metrics = nomt.metrics();
    let ln_reads_at_open = metrics.get_ln_reads();

    // Update existing keys spread across the tree without reading their values: all disk reads
    // here come from the merkle seek (page loads, leaf fetches, elided page reconstruction) and
    // from the beatree update.
    let actuals: Vec<_> = (0..ACCOUNTS)
        .step_by(10)
        .map(|id| {
            (
                account_path(id),
                KeyReadWrite::Write(Some((id + 1).to_le_bytes().to_vec())),
            )
        })
        .collect();
    commit(&nomt, actuals);

    assert!(metrics.get_page_requests() > 0);
    assert!(metrics.get_page_cache_misses() > 0);
    assert!(metrics.get_ht_reads() > 0);
    assert!(metrics.get_page_fetch_time().is_some());

    let leaf_cache_requests = metrics.get_leaf_cache_requests();
    let leaf_cache_misses = metrics.get_leaf_cache_misses();
    assert!(leaf_cache_misses > 0);
    assert!(leaf_cache_requests >= leaf_cache_misses);
    // Every leaf cache miss reads the leaf from the `ln` file.
    assert!(metrics.get_ln_reads() > ln_reads_at_open);
    assert!(metrics.get_ln_reads() >= leaf_cache_misses);

    // A blocking value read of an overflow value reads all of its overflow pages.
    let ln_reads_before = metrics.get_ln_reads();
    let session = nomt.begin_session(SessionParams::default());
    let value = session.read(account_path(OVERFLOW_ID)).unwrap().unwrap();
    assert_eq!(value.len(), OVERFLOW_VALUE_LEN);
    drop(session);
    assert!(metrics.get_ln_reads() - ln_reads_before > (OVERFLOW_VALUE_LEN / 4096) as u64);
    assert!(metrics.get_value_fetch_time().is_some());
}
