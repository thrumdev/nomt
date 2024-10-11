/// Workload abstracts the type of work the DB will have to deal with.
///
/// Generally, the operations on a DB could be:
/// + read
/// + write
/// + delete
/// + update
///
/// Each workload will set up the DB differently and reads and writes arbitrarily,
/// whether the key is not present or already present.
use crate::{backend::Transaction, cli::WorkloadParams, custom_workload, transfer_workload};
use anyhow::Result;
use lru::LruCache;

/// An interface for generating new sets of actions.
pub trait Workload: Send {
    /// Run a step of the workload against the given database transaction.
    ///
    /// Workloads may be run repeatedly and should vary from run to run.
    fn run_step(&mut self, transaction: &mut dyn Transaction);

    /// Whether the workload is done.
    fn is_done(&self) -> bool;
}

pub fn parse(
    workload_params: &WorkloadParams,
    op_limit: u64,
) -> Result<(Box<dyn Workload>, Vec<Box<dyn Workload>>)> {
    let WorkloadParams {
        name,
        size: workload_size,
        initial_capacity: db_size,
        workload_concurrency: threads,
        fresh,
        cache_size,
        ..
    } = workload_params.clone();

    let db_size = db_size.map_or(0, |s| 1u64 << s);

    fn dyn_vec(
        cache_size: Option<u64>,
        threads: u32,
        v: Vec<impl Workload + 'static>,
    ) -> Vec<Box<dyn Workload>> {
        let make_workload = |w| match cache_size {
            None => Box::new(w) as Box<dyn Workload>,
            Some(c) => Box::new(LruCacheWorkload::new(w, c as usize / threads as usize))
                as Box<dyn Workload>,
        };

        v.into_iter().map(make_workload).collect()
    }

    Ok(match name.as_str() {
        "transfer" => (
            Box::new(transfer_workload::init(db_size)),
            dyn_vec(
                cache_size,
                threads,
                transfer_workload::build(
                    db_size,
                    workload_size,
                    fresh.unwrap_or(0),
                    op_limit,
                    threads as usize,
                ),
            ),
        ),
        "randw" => (
            Box::new(custom_workload::init(db_size)),
            dyn_vec(
                cache_size,
                threads,
                custom_workload::build(
                    0,
                    100,
                    workload_size,
                    fresh.unwrap_or(0),
                    db_size,
                    op_limit,
                    threads as usize,
                ),
            ),
        ),
        "randr" => (
            Box::new(custom_workload::init(db_size)),
            dyn_vec(
                cache_size,
                threads,
                custom_workload::build(
                    100,
                    0,
                    workload_size,
                    fresh.unwrap_or(0),
                    db_size,
                    op_limit,
                    threads as usize,
                ),
            ),
        ),
        "randrw" => (
            Box::new(custom_workload::init(db_size)),
            dyn_vec(
                cache_size,
                threads,
                custom_workload::build(
                    50,
                    50,
                    workload_size,
                    fresh.unwrap_or(0),
                    db_size,
                    op_limit,
                    threads as usize,
                ),
            ),
        ),
        name => anyhow::bail!("invalid workload name: {}", name),
    })
}

struct LruCacheWorkload<W> {
    cache: LruCache<Vec<u8>, Option<Vec<u8>>>,
    inner: W,
}

impl<W: Workload> LruCacheWorkload<W> {
    fn new(inner: W, cache_size: usize) -> Self {
        LruCacheWorkload {
            inner,
            cache: LruCache::new(cache_size.try_into().expect("non-zero cache size")),
        }
    }
}

impl<W: Workload> Workload for LruCacheWorkload<W> {
    fn run_step(&mut self, transaction: &mut dyn Transaction) {
        let mut tx = LruCacheTransaction {
            inner: transaction,
            cache: &mut self.cache,
        };
        self.inner.run_step(&mut tx);
    }

    fn is_done(&self) -> bool {
        self.inner.is_done()
    }
}

struct LruCacheTransaction<'a> {
    inner: &'a mut dyn Transaction,
    cache: &'a mut LruCache<Vec<u8>, Option<Vec<u8>>>,
}

impl<'a> Transaction for LruCacheTransaction<'a> {
    fn read(&mut self, key: &[u8]) -> Option<Vec<u8>> {
        match self.cache.get(key) {
            Some(v) => {
                let v = v.as_ref().map(|v| v.to_vec());
                self.inner.note_read(key, v.clone());
                v
            }
            None => self.inner.read(key),
        }
    }

    fn note_read(&mut self, key: &[u8], value: Option<Vec<u8>>) {
        self.inner.note_read(key, value);
    }

    fn write(&mut self, key: &[u8], value: Option<&[u8]>) {
        let _ = self
            .cache
            .push(key.to_vec(), value.as_ref().map(|v| v.to_vec()));
        self.inner.write(key, value);
    }
}
