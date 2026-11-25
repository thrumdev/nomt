use nomt::{hasher::Blake3Hasher, KeyReadWrite, Nomt, Options, SessionParams};
use nomt_test_utils::account_path;
use std::path::PathBuf;

fn open(path: &PathBuf, prepopulate: bool) -> Nomt<Blake3Hasher> {
    let mut o = Options::new();
    o.path(path);
    o.bitbox_seed([0; 16]);
    o.commit_concurrency(2);
    o.io_workers(1);
    o.leaf_cache_size(1);
    o.prepopulate_leaf_cache(prepopulate);
    Nomt::open(o).unwrap()
}

fn value(id: u64) -> Vec<u8> {
    vec![id as u8; 400]
}

#[test]
fn prepopulate_leaf_cache_reads_back_values() {
    let path = PathBuf::from("test/prepopulate_leaf_cache");
    let _ = std::fs::remove_dir_all(&path);
    // ~1.6 MiB of values: more leaves than a 1 MiB leaf cache holds.
    let n = 4000;

    {
        let nomt = open(&path, false);
        let session = nomt.begin_session(SessionParams::default());
        let mut actuals: Vec<_> = (0..n)
            .map(|id| (account_path(id), KeyReadWrite::Write(Some(value(id)))))
            .collect();
        actuals.sort_by_key(|(k, _)| *k);
        session.finish(actuals).unwrap().commit(&nomt).unwrap();
    }

    let nomt = open(&path, true);
    let session = nomt.begin_session(SessionParams::default());
    for id in 0..n {
        assert_eq!(session.read(account_path(id)).unwrap(), Some(value(id)));
    }
}
