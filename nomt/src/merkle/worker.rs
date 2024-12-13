//! Worker logic.
//!
//! During the update phase, each worker gets a range of the keys
//! to work on, pre-fetches pages for all keys within that range, and performs
//! page updates.
//!
//! This module also exposes a warm-up worker, which can be used to pre-fetch pages before the
//! update command is issued.
//!
//! Updates are performed while the next fetch is pending, unless all fetches in
//! the range have completed.

use crossbeam::channel::{Receiver, Select, Sender, TryRecvError};

use nomt_core::{
    page_id::{PageId, ROOT_PAGE_ID},
    proof::PathProofTerminal,
    trie::{KeyPath, Node, ValueHash},
};

use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
};

use super::{
    page_walker::{NeedsPage, Output, PageSource, PageWalker},
    seek::{Completion, Seek, Seeker},
    KeyReadWrite, RootPagePending, UpdateCommand, UpdateShared, WarmUpCommand, WorkerOutput,
};

use crate::{
    io::PagePool,
    page_cache::{PageCache, ShardIndex},
    page_region::PageRegion,
    rw_pass_cell::{ReadPass, WritePass},
    store::Store,
    HashAlgorithm, PathProof, WitnessedPath,
};

pub(super) struct UpdateParams {
    pub page_cache: PageCache,
    pub page_pool: PagePool,
    pub store: Store,
    pub root: Node,
    pub warm_ups: Arc<HashMap<KeyPath, Seek>>,
    pub command: UpdateCommand,
    pub worker_id: usize,
}

pub(super) struct WarmUpParams {
    pub page_cache: PageCache,
    pub store: Store,
    pub root: Node,
}

pub(super) fn run_warm_up<H: HashAlgorithm>(
    params: WarmUpParams,
    warmup_rx: Receiver<WarmUpCommand>,
    finish_rx: Receiver<()>,
    output_tx: Sender<HashMap<KeyPath, Seek>>,
) {
    let read_pass = params.page_cache.new_read_pass();
    let page_loader = params.store.page_loader();
    let io_handle = params.store.io_pool().make_handle();
    let page_io_receiver = io_handle.receiver().clone();
    let seeker = Seeker::<H>::new(
        params.root,
        params.store.read_transaction(),
        params.page_cache.clone(),
        io_handle,
        page_loader,
        true,
    );

    let result = warm_up_phase(read_pass, page_io_receiver, seeker, warmup_rx, finish_rx);

    match result {
        Err(_) => return,
        Ok(res) => {
            let _ = output_tx.send(res);
        }
    }
}

pub(super) fn run_update<H: HashAlgorithm>(params: UpdateParams) -> anyhow::Result<WorkerOutput> {
    let UpdateParams {
        page_cache,
        page_pool,
        store,
        root,
        warm_ups,
        command,
        ..
    } = params;

    let seeker = Seeker::<H>::new(
        root,
        store.read_transaction(),
        page_cache.clone(),
        store.io_pool().make_handle(),
        store.page_loader(),
        command.shared.witness,
    );

    update::<H>(root, page_cache, page_pool, seeker, command, warm_ups)
}

fn warm_up_phase<H: HashAlgorithm>(
    read_pass: ReadPass<ShardIndex>,
    page_io_receiver: Receiver<crate::io::CompleteIo>,
    mut seeker: Seeker<H>,
    warmup_rx: Receiver<WarmUpCommand>,
    finish_rx: Receiver<()>,
) -> anyhow::Result<HashMap<KeyPath, Seek>> {
    let mut select_all = Select::new();
    let warmup_idx = select_all.recv(&warmup_rx);
    let finish_idx = select_all.recv(&finish_rx);
    let page_idx = select_all.recv(&page_io_receiver);

    let mut select_no_work = Select::new();
    let finish_no_work_idx = select_no_work.recv(&finish_rx);
    let page_no_work_idx = select_no_work.recv(&page_io_receiver);

    let mut warm_ups = HashMap::new();

    loop {
        if let Some(Completion::Seek(result)) = seeker.take_completion() {
            warm_ups.insert(result.key, result);
            continue;
        }

        seeker.submit_all(&read_pass)?;
        if !seeker.has_room() {
            // block on interrupt or next page ready.
            let index = select_no_work.ready();
            if index == finish_no_work_idx {
                match finish_rx.try_recv() {
                    Err(TryRecvError::Empty) => continue,
                    Err(e) => anyhow::bail!(e),
                    Ok(()) => break,
                }
            } else if index == page_no_work_idx {
                seeker.try_recv_page(&read_pass)?;
            } else {
                unreachable!()
            }
        } else {
            // has room. select on everything, pushing new work as available.
            let index = select_all.ready();
            if index == finish_idx {
                match finish_rx.try_recv() {
                    Err(TryRecvError::Empty) => continue,
                    Err(e) => anyhow::bail!(e),
                    Ok(()) => break,
                }
            } else if index == warmup_idx {
                let warm_up_command = match warmup_rx.try_recv() {
                    Ok(command) => command,
                    Err(TryRecvError::Empty) => continue,
                    Err(e) => anyhow::bail!(e),
                };

                seeker.push(warm_up_command.key_path);
            } else if index == page_idx {
                seeker.try_recv_page(&read_pass)?;
            } else {
                unreachable!()
            }
        }
    }

    while !seeker.is_empty() {
        if let Some(Completion::Seek(result)) = seeker.take_completion() {
            warm_ups.insert(result.key, result);
            continue;
        }
        seeker.submit_all(&read_pass)?;
        if seeker.has_live_requests() {
            seeker.recv_page(&read_pass)?;
        }
    }

    Ok(warm_ups)
}

fn update<H: HashAlgorithm>(
    root: Node,
    page_cache: PageCache,
    page_pool: PagePool,
    mut seeker: Seeker<H>,
    command: UpdateCommand,
    warm_ups: Arc<HashMap<KeyPath, Seek>>,
) -> anyhow::Result<WorkerOutput> {
    let UpdateCommand { shared, write_pass } = command;
    let write_pass = write_pass.into_inner();

    let mut output = WorkerOutput::new(shared.witness);

    let updater = RangeUpdater::<H>::new(root, shared.clone(), write_pass, &page_cache, &page_pool);

    // one lucky thread gets the master write pass.
    let mut write_pass = match updater.update(&mut seeker, &mut output, warm_ups)? {
        None => return Ok(output),
        Some(write_pass) => write_pass,
    };

    let pending_ops = shared.take_root_pending();
    let mut root_page_updater = PageWalker::<H>::new(
        root,
        PageSource::PageCache(page_cache.clone()),
        page_pool.clone(),
        None,
    );

    for (trie_pos, pending_op) in pending_ops {
        match pending_op {
            RootPagePending::Node(node) => loop {
                let page = match root_page_updater.advance_and_place_node(
                    &mut write_pass,
                    trie_pos.clone(),
                    node,
                ) {
                    Err(NeedsPage(page)) => page,
                    Ok(()) => break,
                };
                drive_page_fetch(&mut seeker, write_pass.downgrade(), page)?;
            },
            RootPagePending::SubTrie {
                range_start,
                range_end,
                prev_terminal,
            } => {
                let ops = subtrie_ops(&shared.read_write[range_start..range_end]);
                let ops = nomt_core::update::leaf_ops_spliced(prev_terminal, &ops);
                loop {
                    let page = match root_page_updater.advance_and_replace(
                        &mut write_pass,
                        trie_pos.clone(),
                        ops.clone(),
                    ) {
                        Err(NeedsPage(page)) => page,
                        Ok(()) => break,
                    };
                    drive_page_fetch(&mut seeker, write_pass.downgrade(), page)?;
                }
            }
        }
    }

    // PANIC: output is always root when no parent page is specified.
    loop {
        let page = match root_page_updater.conclude(&mut write_pass) {
            Ok(Output::Root(new_root, diffs)) => {
                output.page_diffs.extend(diffs);
                output.root = Some(new_root);
                break;
            }
            Ok(Output::ChildPageRoots(_, _)) => unreachable!(),
            Err((NeedsPage(page), page_walker)) => {
                root_page_updater = page_walker;
                page
            }
        };
        drive_page_fetch(&mut seeker, write_pass.downgrade(), page)?;
    }

    Ok(output)
}

fn drive_page_fetch<H: HashAlgorithm>(
    seeker: &mut Seeker<H>,
    read_pass: &ReadPass<ShardIndex>,
    page: PageId,
) -> anyhow::Result<()> {
    seeker.push_single_request(page);
    loop {
        match seeker.take_completion() {
            Some(Completion::SinglePage) => return Ok(()),
            Some(_) => continue,
            None => {
                seeker.submit_all(read_pass)?;
                if seeker.has_live_requests() {
                    seeker.recv_page(&read_pass)?;
                }
            }
        }
    }
}

// helper for iterating all paths in the range and performing
// updates.
//
// anything that touches the root page is deferred via `shared.pending`.
struct RangeUpdater<H> {
    shared: Arc<UpdateShared>,
    write_pass: WritePass<ShardIndex>,
    region: PageRegion,
    page_walker: PageWalker<H>,
    range_start: usize,
    range_end: usize,
    saved_advance: Option<SavedAdvance>,
}

impl<H: HashAlgorithm> RangeUpdater<H> {
    fn new(
        root: Node,
        shared: Arc<UpdateShared>,
        write_pass: WritePass<ShardIndex>,
        page_cache: &PageCache,
        page_pool: &PagePool,
    ) -> Self {
        let region = match write_pass.region() {
            ShardIndex::Root => PageRegion::universe(),
            ShardIndex::Shard(i) => page_cache.shard_region(*i),
        };
        let key_range_start = region.exclusive_min().min_key_path();
        let key_range_end = region.exclusive_max().max_key_path();

        let range_start = shared
            .read_write
            .binary_search_by_key(&key_range_start, |x| x.0)
            .unwrap_or_else(|i| i);

        let range_end = shared
            .read_write
            .binary_search_by_key(&key_range_end, |x| x.0)
            .unwrap_or_else(|i| i);

        let page_source = match write_pass.region() {
            ShardIndex::Root => PageSource::PageCache(page_cache.clone()),
            ShardIndex::Shard(i) => PageSource::PageCacheShard(page_cache.get_shard(*i)),
        };

        RangeUpdater {
            shared,
            write_pass,
            region,
            page_walker: PageWalker::<H>::new(
                root,
                page_source,
                page_pool.clone(),
                Some(ROOT_PAGE_ID),
            ),
            range_start,
            range_end,
            saved_advance: None,
        }
    }

    // returns the end index of the batch. returns the end index of the batches covered by
    // this terminal.
    fn handle_completion(
        &mut self,
        seeker: &mut Seeker<H>,
        output: &mut WorkerOutput,
        start_index: usize,
        seek_result: Seek,
    ) -> usize {
        assert!(self.saved_advance.is_none());

        // note that this is only true when the seek result is in the shared area - so multiple
        // workers will encounter it. we use this to defer to the first worker.
        let batch_starts_in_our_range = start_index != self.range_start
            || self.range_start == 0
            || !seek_result
                .position
                .subtrie_contains(&self.shared.read_write[start_index - 1].0);

        let (batch_size, has_writes) = {
            let mut batch_size = 0;
            let mut has_writes = false;

            // find batch size.
            for (k, v) in &self.shared.read_write[start_index..] {
                if !seek_result.position.subtrie_contains(&k) {
                    break;
                }
                batch_size += 1;
                has_writes |= v.is_write();
            }

            (batch_size, has_writes)
        };

        let next_index = start_index + batch_size;

        // witness / pushing pending responsibility falls on the worker whose range this falls
        // inside.
        if !batch_starts_in_our_range {
            return next_index;
        }

        let is_non_exclusive = seek_result
            .page_id
            .as_ref()
            .map_or(true, |p_id| !self.region.contains_exclusive(p_id));

        if is_non_exclusive {
            self.shared.push_pending_subtrie(
                seek_result.position.clone(),
                start_index,
                next_index,
                seek_result.terminal.clone(),
            );

            if let Some(ref mut witnessed_paths) = output.witnessed_paths {
                let path = WitnessedPath {
                    inner: PathProof {
                        // if the terminal lands in the non-exclusive area, then the path to it is
                        // guaranteed not to have been altered by anything we've done so far.
                        siblings: seek_result.siblings,
                        terminal: match seek_result.terminal.clone() {
                            Some(leaf_data) => PathProofTerminal::Leaf(leaf_data),
                            None => PathProofTerminal::Terminator(seek_result.position.clone()),
                        },
                    },
                    path: seek_result.position,
                };
                witnessed_paths.push((path, seek_result.terminal, batch_size));
            }

            return next_index;
        }

        // attempt to advance the trie walker. if it fails, pocket away for later.
        let ops = if has_writes {
            Some(subtrie_ops(
                &self.shared.read_write[start_index..next_index],
            ))
        } else {
            None
        };
        self.attempt_advance(seeker, output, seek_result, ops, batch_size);

        next_index
    }

    // attempt to advance the trie walker. if this fails, it submits a special page request to
    // the seeker and stores the `SavedAdvance`. if this succeeds, it updates the output.
    fn attempt_advance(
        &mut self,
        seeker: &mut Seeker<H>,
        output: &mut WorkerOutput,
        seek_result: Seek,
        ops: Option<Vec<(KeyPath, Option<ValueHash>)>>,
        batch_size: usize,
    ) {
        let res = match ops {
            None => self
                .page_walker
                .advance(&mut self.write_pass, seek_result.position.clone()),
            Some(ref ops) => {
                let ops = nomt_core::update::leaf_ops_spliced(seek_result.terminal.clone(), &ops);
                self.page_walker.advance_and_replace(
                    &mut self.write_pass,
                    seek_result.position.clone(),
                    ops,
                )
            }
        };

        if let Err(NeedsPage(page)) = res {
            seeker.push_single_request(page);
            self.saved_advance = Some(SavedAdvance {
                seek_result,
                ops,
                batch_size,
            });
            return;
        }

        if let Some(ref mut witnessed_paths) = output.witnessed_paths {
            let siblings = {
                // nodes may have been altered prior to seeking - the page walker tracks which ones.
                let mut siblings = seek_result.siblings;
                for (actual_sibling, depth) in self.page_walker.siblings() {
                    siblings[*depth - 1] = *actual_sibling;
                }
                siblings
            };

            let path = WitnessedPath {
                inner: PathProof {
                    siblings,
                    terminal: match seek_result.terminal.clone() {
                        Some(leaf_data) => PathProofTerminal::Leaf(leaf_data),
                        None => PathProofTerminal::Terminator(seek_result.position.clone()),
                    },
                },
                path: seek_result.position,
            };
            witnessed_paths.push((path, seek_result.terminal, batch_size));
        }
    }

    fn reattempt_advance(&mut self, seeker: &mut Seeker<H>, output: &mut WorkerOutput) {
        // UNWRAP: guaranteed by behavior of seeker / update / advance.
        let SavedAdvance {
            ops,
            seek_result,
            batch_size,
        } = self.saved_advance.take().unwrap();
        self.attempt_advance(seeker, output, seek_result, ops, batch_size);
    }

    fn update(
        mut self,
        seeker: &mut Seeker<H>,
        output: &mut WorkerOutput,
        warm_ups: Arc<HashMap<KeyPath, Seek>>,
    ) -> anyhow::Result<Option<WritePass<ShardIndex>>> {
        let mut start_index = self.range_start;
        let mut pushes = 0;
        let mut skips = 0;

        let mut warmed_up: VecDeque<Seek> = VecDeque::new();

        // 1. drive until work is done.
        while start_index < self.range_end || !seeker.is_empty() {
            let completion = if self.saved_advance.is_none()
                && warmed_up
                    .front()
                    .map_or(false, |res| match seeker.first_key() {
                        None => true,
                        Some(k) => &res.key < k,
                    }) {
                // take a "completion" from our warm-ups instead.
                // UNWRAP: checked front exists above.
                Some(Completion::Seek(warmed_up.pop_front().unwrap()))
            } else {
                seeker.take_completion()
            };

            // handle a single completion (only when blocked / at max capacity)
            match completion {
                None => {}
                Some(Completion::Seek(seek_result)) => {
                    // skip completions until we're past the end of the last batch.
                    if skips > 0 {
                        skips -= 1;
                    } else {
                        let end_index =
                            self.handle_completion(seeker, output, start_index, seek_result);

                        // account for stuff we pushed that was already covered by the terminal
                        // we just popped off.
                        let batch_size = end_index - start_index;
                        // note: pushes and batch size are both at least 1.
                        skips = std::cmp::min(pushes, batch_size) - 1;
                        pushes = pushes.saturating_sub(batch_size);

                        start_index = end_index;
                    }
                }
                Some(Completion::SinglePage) => {
                    self.reattempt_advance(seeker, output);
                }
            }

            seeker.submit_all(self.write_pass.downgrade())?;
            if !seeker.has_room() && seeker.has_live_requests() {
                // no way to push work until at least one page fetch has concluded.
                seeker.recv_page(self.write_pass.downgrade())?;
                continue;
            }

            // push work until out of work.
            while seeker.has_room() && start_index + pushes < self.range_end {
                let next_push = start_index + pushes;
                pushes += 1;

                if let Some(result) = warm_ups.get(&self.shared.read_write[next_push].0) {
                    warmed_up.push_back(result.clone());
                    if warmed_up.len() >= 512 {
                        break;
                    }
                } else {
                    seeker.push(self.shared.read_write[next_push].0);
                    seeker.submit_all(self.write_pass.downgrade())?;
                }
            }

            seeker.try_recv_page(self.write_pass.downgrade())?;
        }

        // 2. conclude, driving additional page fetches as necessary.
        loop {
            // PANIC: walker was configured with a parent page.
            let (new_nodes, diffs) = match self.page_walker.conclude(&mut self.write_pass) {
                Ok(Output::Root(_, _)) => unreachable!(),
                Ok(Output::ChildPageRoots(new_nodes, diffs)) => (new_nodes, diffs),
                Err((NeedsPage(page), page_walker)) => {
                    self.page_walker = page_walker;
                    drive_page_fetch(seeker, self.write_pass.downgrade(), page)?;
                    continue;
                }
            };

            assert!(!diffs.iter().any(|item| item.0 == ROOT_PAGE_ID));
            output.page_diffs = diffs;

            self.shared.push_pending_root_nodes(new_nodes);

            return Ok(self.write_pass.consume());
        }
    }
}

fn subtrie_ops(read_write: &[(KeyPath, KeyReadWrite)]) -> Vec<(KeyPath, Option<ValueHash>)> {
    read_write
        .iter()
        .filter_map(|(key, read_write)| match read_write {
            KeyReadWrite::Write(val) | KeyReadWrite::ReadThenWrite(val) => {
                Some((key.clone(), val.clone()))
            }
            KeyReadWrite::Read => None,
        })
        .collect::<Vec<_>>()
}

struct SavedAdvance {
    // none: no writes
    ops: Option<Vec<(KeyPath, Option<ValueHash>)>>,
    seek_result: Seek,
    batch_size: usize,
}
