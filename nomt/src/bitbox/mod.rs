use crossbeam::channel::{Receiver, Sender};
use nomt_core::page_id::PageId;
use parking_lot::RwLock;
use std::{
    collections::HashSet,
    path::PathBuf,
    sync::{Arc, Mutex},
};
use threadpool::ThreadPool;

use crate::{
    io::{CompleteIo, IoCommand, IoKind, Page, PAGE_SIZE},
    page_cache::PageDiff,
};

use self::{
    meta_map::MetaMap,
    store::{MetaPage, Store},
    wal::{Batch as WalBatch, ConsistencyError, Entry as WalEntry, WalWriter},
};

mod meta_map;
mod store;
mod wal;


const LOAD_PAGE_HANDLE_INDEX: usize = 0;
const LOAD_VALUE_HANDLE_INDEX: usize = 1;
const COMMIT_HANDLE_INDEX: usize = 2;
const N_WORKER: usize = 3;

pub struct DB {
    shared: Arc<Shared>,
}

pub struct Shared {
    store: Store,
    meta_page: RwLock<MetaPage>,
    // TODO: probably RwLock can be avoided being used only during commit
    wal: RwLock<WalWriter>,
    meta_map: RwLock<MetaMap>,
    // channel used to send requrest to the dispatcher
    get_page_tx: Mutex<Sender<(u64 /*bucket*/, Sender<Page>)>>,
    io_sender: Sender<IoCommand>,
    commit_page_receiver: Receiver<CompleteIo>,
    _dispatcher_tp: ThreadPool,
}

impl DB {
    pub fn open(num_rings: usize, path: PathBuf) -> anyhow::Result<Self> {
        let db_path = path.join("db");
        let wal_path = path.join("wal");

        if !db_path.is_file() {
            std::fs::create_dir_all(path).unwrap();
            store::create(db_path.clone(), 2_000_000).unwrap();
        }

        let (store, meta_page, meta_map) = match store::Store::open(db_path) {
            Ok(x) => x,
            Err(e) => {
                anyhow::bail!("encountered error in opening store: {e:?}");
            }
        };

        // Open the WAL, check its integrity and make sure the store is consistent with it
        let wal = wal::WalChecker::open_and_recover(wal_path.clone());
        let _pending_batch = match wal.check_consistency(meta_page.sequence_number()) {
            Ok(()) => {
                println!(
                    "Wal and Store are consistent, last sequence number: {}",
                    meta_page.sequence_number()
                );
                None
            }
            Err(ConsistencyError::LastBatchCrashed(crashed_batch)) => {
                println!(
                    "Wal and Store are not consistent, pending sequence number: {}",
                    meta_page.sequence_number() + 1
                );
                Some(crashed_batch)
            }
            Err(ConsistencyError::NotConsistent(wal_seqn)) => {
                // This is useful for testing. If the WAL sequence number is zero, it means the WAL is empty.
                // For example, it could have been deleted, and it's okay to continue working on the store
                // by appending new batches to the new WAL
                if wal_seqn == 0 {
                    None
                } else {
                    panic!(
                        "Store and Wal have two inconsistent serial numbers. wal: {}, store: {}",
                        wal_seqn,
                        meta_page.sequence_number()
                    );
                }
            }
        };

        // Create a WalWriter, able to append new batch and prune older ones
        let wal = match wal::WalWriter::open(wal_path) {
            Ok(x) => x,
            Err(e) => {
                anyhow::bail!("encountered error in opening wal: {e:?}")
            }
        };

        // Spawn io_workers
        let (io_sender, io_receivers) = crate::io::start_io_worker(N_WORKER, num_rings);

        let load_page_sender = io_sender.clone();
        let load_page_receiver = io_receivers[LOAD_PAGE_HANDLE_INDEX].clone();
        let _load_value_receiver = io_receivers[LOAD_VALUE_HANDLE_INDEX].clone();
        let commit_page_receiver = io_receivers[COMMIT_HANDLE_INDEX].clone();

        // Spawn PageRequest dispatcher
        let (get_page_tx, get_page_rx) = crossbeam::channel::unbounded();

        let dispatcher_tp = threadpool::Builder::new()
            .num_threads(N_WORKER)
            .thread_name("nomt-io-dispather".to_string())
            .build();

        dispatcher_tp.execute(page_dispatcher_task(
            store.clone(),
            load_page_sender,
            load_page_receiver,
            get_page_rx,
        ));

        Ok(Self {
            shared: Arc::new(Shared {
                store,
                commit_page_receiver,
                meta_page: RwLock::new(meta_page),
                wal: RwLock::new(wal),
                meta_map: RwLock::new(meta_map),
                get_page_tx: Mutex::new(get_page_tx),
                _dispatcher_tp: dispatcher_tp,
                io_sender,
            }),
        })
    }

    pub fn get(&self, page_id: &PageId) -> anyhow::Result<Option<Vec<u8>>> {
        self.inner_get(&page_id).map(|(_bucket, data)| data)
    }

    // returns bucket and data
    //
    // looking for a key only the key itself or an empty bucket could stop the search
    fn inner_get(&self, page_id: &PageId) -> anyhow::Result<(ProbeSequence, Option<Vec<u8>>)> {
        let mut probe_seq = ProbeSequence::new(page_id, &self.shared.meta_map.read());

        loop {
            match probe_seq.next(&self.shared.meta_map.read()) {
                ProbeResult::PossibleHit(bucket) => {
                    // if this could be a match we need to fetch the page and check its id

                    // send the read command
                    let (tx, rx) = crossbeam::channel::bounded::<Page>(1);
                    {
                        let get_page_tx = self.shared.get_page_tx.lock().unwrap();
                        get_page_tx.send((bucket, tx)).unwrap();
                    }

                    // wait for the dispacther
                    let Ok(page) = rx.recv() else {
                        panic!("something went wrong requesting pages");
                    };

                    if &page[PAGE_SIZE - 32..] == page_id.encode() {
                        break Ok((probe_seq, Some(page.to_vec())));
                    }
                }
                ProbeResult::Emtpy(_bucket) => break Ok((probe_seq, None)),
            }
        }
    }

    // TODO: update with async sync apporach
    pub fn commit(
        &self,
        new_pages: Vec<(PageId, Option<(Vec<u8>, PageDiff)>)>,
    ) -> anyhow::Result<()> {
        // Steps are:
        // 0. Increase sequence number
        // 1. compute the WalBatch
        // 2. append WalBatch to wal and fsync
        // 4. write new pages
        // 5. write meta map
        // 6. fsync
        // 7. write meta page and fsync
        // 8. prune the wal

        let mut changed_meta_pages = HashSet::new();
        let next_sequence_number = self.shared.meta_page.read().sequence_number() + 1;
        let mut wal_batch = WalBatch::new(next_sequence_number);
        let mut bucket_writes = Vec::new();

        for (page_id, page_info) in new_pages {
            // let's extract its bucket
            match page_info {
                Some((raw_page, page_diff)) => {
                    // the page_id must be written into the page itself
                    assert!(raw_page.len() == PAGE_SIZE);
                    let mut page = crate::bitbox::Page::zeroed();
                    page[..raw_page.len()].copy_from_slice(&raw_page);
                    page[PAGE_SIZE - 32..].copy_from_slice(&page_id.encode());

                    // This could be either an update or an insertion.
                    // Thus, we first need to check if the key is present.
                    // If so, it will be updated; otherwise, the page must be inserted
                    // and it will be inserted into the first tombstone encountered
                    // or in the first empty page found.

                    let Ok((probe_seq, maybe_old_page)) = self.inner_get(&page_id) else {
                        panic!("Impossible insert element into map")
                    };

                    let bucket = match maybe_old_page {
                        Some(_) => probe_seq.bucket(),
                        None => probe_seq.tombstone.unwrap_or_else(|| probe_seq.bucket()),
                    };

                    // update meta map with new info
                    self.shared
                        .meta_map
                        .write()
                        .set_full(bucket as usize, probe_seq.hash);
                    changed_meta_pages
                        .insert(self.shared.meta_map.read().page_index(bucket as usize));

                    // fill the WalBatch and the pages that need to be written to disk
                    wal_batch.append_entry(WalEntry::Update {
                        page_id: page_id.encode(),
                        changed: get_changed(&page, &page_diff),
                        page_diff: page_diff.get_raw(),
                        bucket_index: bucket as u64,
                    });

                    bucket_writes.push((bucket, page));
                }
                None => {
                    // the page must be deleted
                    let Ok((probe_seq, Some(_old_page))) = self.inner_get(&page_id) else {
                        panic!("Not existing pages is being eliminated");
                    };
                    let bucket = probe_seq.bucket() as usize;
                    self.shared.meta_map.write().set_tombstone(bucket);
                    changed_meta_pages.insert(self.shared.meta_map.read().page_index(bucket));

                    wal_batch.append_entry(WalEntry::Clear {
                        bucket_index: bucket as u64,
                    });
                }
            };
        }

        let prev_wal_size = self.shared.wal.read().file_size();
        wal_batch.data().len();
        self.shared.wal.write().apply_batch(&wal_batch).unwrap();

        let mut submitted: u32 = 0;
        let mut completed: u32 = 0;

        // Issue all bucket writes
        for (bucket, page) in bucket_writes {
            let command = IoCommand {
                kind: IoKind::Write(
                    self.shared.store.store_fd(),
                    self.shared.store.data_page_index(bucket),
                    Box::new(page),
                ),
                handle: COMMIT_HANDLE_INDEX,
                user_data: 0, // unimportant.
            };
            // TODO: handle error
            self.shared.io_sender.send(command).unwrap();
            submitted += 1;
        }

        // apply changed meta pages
        for changed_meta_page in changed_meta_pages {
            let mut buf = Box::new(Page::zeroed());
            buf[..].copy_from_slice(self.shared.meta_map.read().page_slice(changed_meta_page));
            let command = IoCommand {
                kind: IoKind::Write(
                    self.shared.store.store_fd(),
                    self.shared.store.meta_bytes_index(changed_meta_page as u64),
                    buf,
                ),
                handle: COMMIT_HANDLE_INDEX,
                user_data: 0, // unimportant
            };
            submitted += 1;
            // TODO: handle error
            self.shared.io_sender.send(command).unwrap();
        }

        // wait for all writes command to be finished
        while completed < submitted {
            let completion = self
                .shared
                .commit_page_receiver
                .recv()
                .expect("I/O worker dropped");
            assert!(completion.result.is_ok());
            completed += 1;
        }

        // sync all writes
        submit_and_wait_one(
            &self.shared.io_sender,
            &self.shared.commit_page_receiver,
            IoCommand {
                kind: IoKind::Fsync(self.shared.store.store_fd()),
                handle: COMMIT_HANDLE_INDEX,
                user_data: 0, // unimportant
            },
        );

        // update sequence number in metapage
        self.shared
            .meta_page
            .write()
            .set_sequence_number(next_sequence_number);

        submit_and_wait_one(
            &self.shared.io_sender,
            &self.shared.commit_page_receiver,
            IoCommand {
                kind: IoKind::Write(
                    self.shared.store.store_fd(),
                    0,
                    Box::new(self.shared.meta_page.read().to_page()),
                ),
                handle: COMMIT_HANDLE_INDEX,
                user_data: 0, // unimportant
            },
        );

        // sync meta page change.
        submit_and_wait_one(
            &self.shared.io_sender,
            &self.shared.commit_page_receiver,
            IoCommand {
                kind: IoKind::Fsync(self.shared.store.store_fd()),
                handle: COMMIT_HANDLE_INDEX,
                user_data: 0, // unimportant
            },
        );

        // clear the WAL.
        self.shared.wal.read().prune_front(prev_wal_size);

        Ok(())
    }
}

// call only when I/O queue is totally empty.
fn submit_and_wait_one(
    io_sender: &Sender<IoCommand>,
    io_receiver: &Receiver<CompleteIo>,
    command: IoCommand,
) {
    let _ = io_sender.send(command);
    let _ = io_receiver.recv();
}

fn get_changed(page: &Page, page_diff: &PageDiff) -> Vec<[u8; 32]> {
    page_diff
        .get_changed()
        .into_iter()
        .map(|changed| {
            let start = changed * 32;
            let end = start + 32;
            let mut node = [0; 32];
            node.copy_from_slice(&page[start..end]);
            node
        })
        .collect()
}

// A task that will receive all page requests and send back the result
// through the provided channel

// NOTE: It's not the best solution, as initial integration it avoids
// having to change the io_uring abstraction
fn page_dispatcher_task(
    store: Store,
    load_page_sender: Sender<IoCommand>,
    load_page_receiver: Receiver<CompleteIo>,
    get_page_rx: Receiver<(u64, Sender<Page>)>,
) -> impl Fn() {
    move || {
        let mut slab = slab::Slab::new();
        loop {
            crossbeam::select! {
                recv(get_page_rx) -> req => {
                    let Ok((bucket, tx)) = req else {
                        break;
                    };

                    let index = slab.insert(tx);
                    let command = IoCommand {
                        kind: IoKind::Read(store.store_fd(), store.data_page_index(bucket), Box::new(Page::zeroed())),
                        handle: LOAD_PAGE_HANDLE_INDEX,
                        user_data: index as u64,
                    };

                    // send the command
                    load_page_sender.send(command).unwrap();

                },
                recv(load_page_receiver) -> response => {
                    let Ok(CompleteIo { command, result }) = response else {
                        panic!("TODO")
                    };

                    assert!(result.is_ok());

                    let tx: Sender<Page> = slab.remove(command.user_data as usize);
                    tx.send(*command.kind.unwrap_buf()).unwrap();
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
struct ProbeSequence {
    hash: u64,
    bucket: u64,
    step: u64,
    tombstone: Option<u64>,
}

pub enum ProbeResult {
    PossibleHit(u64),
    Emtpy(u64),
}

impl ProbeSequence {
    pub fn new(page_id: &PageId, meta_map: &MetaMap) -> Self {
        let hash = {
            let mut buf = [0; 8];
            // TODO: the seed of the store should be used
            buf.copy_from_slice(&blake3::hash(&page_id.encode()).as_bytes()[..8]);
            u64::from_le_bytes(buf)
        };

        Self {
            hash,
            bucket: hash % meta_map.len() as u64,
            step: 0,
            tombstone: None,
        }
    }

    // probe until there is a possible hit or an empty bucket is found
    pub fn next(&mut self, meta_map: &MetaMap) -> ProbeResult {
        loop {
            // Triangular probing
            self.bucket += self.step;
            self.step += 1;
            self.bucket %= meta_map.len() as u64;

            // if metamap is empty, return early
            if meta_map.hint_empty(self.bucket as usize) {
                return ProbeResult::Emtpy(self.bucket);
            }

            if meta_map.hint_tombstone(self.bucket as usize) {
                self.tombstone = Some(self.bucket);
            }

            if meta_map.hint_not_match(self.bucket as usize, self.hash) {
                continue;
            }

            return ProbeResult::PossibleHit(self.bucket);
        }
    }

    pub fn bucket(&self) -> u64 {
        self.bucket
    }
}
