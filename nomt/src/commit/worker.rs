//! Worker logic.
//!
//! Workers have two phases: the random warm-up phase and the sequential commit phase.
//!
//! During the random warm-up phase, workers pre-fetch pages necessary for accessing terminals
//! expected to be used during commit.
//!
//! During the sequential commit phase, each worker gets a range of the committed keys
//! to work on, continues pre-fetching pages for all keys within that range, and performs
//! page updates.
//!
//! Updates are performed while the next fetch is pending, unless all fetches in
//! the range have completed.

use crossbeam::channel::{Receiver, Sender, TryRecvError};

use nomt_core::{
    page_id::ROOT_PAGE_ID,
    proof::PathProofTerminal,
    trie::{KeyPath, Node, NodeHasher, ValueHash},
};

use std::sync::{Arc, Barrier};

use super::{
    CommitCommand, CommitShared, KeyReadWrite, RootPagePending, ToWorker, WarmUpCommand,
    WorkerOutput,
};

use crate::{
    new_seek::{Interrupt, Seek, Seeker as NewSeeker},
    page_cache::{PageCache, ShardIndex},
    page_region::PageRegion,
    page_walker::{Output, PageWalker},
    rw_pass_cell::{ReadPass, WritePass},
    store::Store,
    PathProof, WitnessedPath,
};

pub(super) struct Params {
    pub page_cache: PageCache,
    pub store: Store,
    pub root: Node,
    pub barrier: Arc<Barrier>,
}

pub(super) struct Comms {
    pub output_tx: Sender<WorkerOutput>,
    pub commit_rx: Receiver<ToWorker>,
    pub warmup_rx: Receiver<WarmUpCommand>,
}

pub(super) fn run<H: NodeHasher>(comms: Comms, params: Params) {
    let Params {
        page_cache,
        store,
        root,
        barrier,
    } = params;

    let read_pass = page_cache.new_read_pass();
    barrier.wait();

    let seeker = NewSeeker::new(root, page_cache.clone(), store.page_loader(), false);

    if let Err(_) = warm_up_phase(&comms, read_pass, seeker) {
        return;
    };

    // TODO: whether to record siblings should be a parameter on `CommitCommand`.
    let seeker = NewSeeker::new(root, page_cache.clone(), store.page_loader(), true);

    match comms.commit_rx.recv() {
        Err(_) => return, // early exit only.
        // UNWRAP: Commit always sent after Prepare.
        Ok(ToWorker::Prepare) => unreachable!(),
        Ok(ToWorker::Commit(command)) => {
            let output = match commit::<H>(root, page_cache, seeker, command) {
                Err(_) => return,
                Ok(o) => o,
            };
            let _ = comms.output_tx.send(output);
        }
    }
}

fn warm_up_phase(
    comms: &Comms,
    read_pass: ReadPass<ShardIndex>,
    mut seeker: NewSeeker,
) -> anyhow::Result<()> {
    let mut preparing = false;
    loop {
        let (block, push) = match seeker.advance(&read_pass)? {
            Interrupt::NoMoreWork => {
                if preparing {
                    return Ok(());
                } else {
                    (true, true)
                }
            }
            Interrupt::HasRoom => {
                if preparing {
                    (false, false)
                } else {
                    (false, true)
                }
            }
            Interrupt::Completion(_) | Interrupt::SpecialPageCompletion => (false, false),
        };

        if preparing || !push {
            continue;
        }

        let (commit_msg, warmup_msg) = if block {
            crossbeam_channel::select! {
                recv(comms.commit_rx) -> msg => (Some(msg?), None),
                recv(comms.warmup_rx) -> msg => (None, Some(msg?)),
            }
        } else {
            (
                match comms.commit_rx.try_recv() {
                    Ok(msg) => Some(msg),
                    Err(TryRecvError::Empty) => None,
                    Err(TryRecvError::Disconnected) => anyhow::bail!(TryRecvError::Disconnected),
                },
                match comms.warmup_rx.try_recv() {
                    Ok(msg) => Some(msg),
                    Err(TryRecvError::Empty) => None,
                    Err(TryRecvError::Disconnected) => anyhow::bail!(TryRecvError::Disconnected),
                },
            )
        };

        match commit_msg {
            None => {}
            Some(ToWorker::Prepare) => {
                preparing = true;
                continue;
            }
            // prepare is always sent before commit.
            Some(ToWorker::Commit(_)) => unreachable!(),
        }

        match warmup_msg {
            None => {}
            Some(warm_up) => seeker.push(warm_up.key_path),
        }
    }
}

fn commit<H: NodeHasher>(
    root: Node,
    page_cache: PageCache,
    mut seeker: NewSeeker,
    command: CommitCommand,
) -> anyhow::Result<WorkerOutput> {
    let CommitCommand { shared, write_pass } = command;
    let write_pass = write_pass.into_inner();

    let mut output = WorkerOutput::new(shared.witness);

    let committer = RangeCommitter::<H>::new(root, shared.clone(), write_pass, &page_cache);

    // one lucky thread gets the master write pass.
    let mut write_pass = match committer.commit(&mut seeker, &mut output)? {
        None => return Ok(output),
        Some(write_pass) => write_pass,
    };

    let pending_ops = shared.take_root_pending();
    let mut root_page_committer = PageWalker::<H>::new(root, page_cache.clone(), None);
    for (trie_pos, pending_op) in pending_ops {
        match pending_op {
            RootPagePending::Node(node) => {
                root_page_committer.advance_and_place_node(&mut write_pass, trie_pos, node);
            }
            RootPagePending::SubTrie {
                range_start,
                range_end,
                prev_terminal,
            } => {
                let ops = subtrie_ops(&shared.read_write[range_start..range_end]);
                let ops = nomt_core::update::leaf_ops_spliced(prev_terminal, &ops);
                root_page_committer.advance_and_replace(&mut write_pass, trie_pos, ops);
            }
        }
    }

    // PANIC: output is always root when no parent page is specified.
    match root_page_committer.conclude(&mut write_pass) {
        Output::Root(new_root, diffs) => {
            for (page_id, page_diff) in diffs {
                output.page_diffs.insert(page_id, page_diff);
            }

            output.root = Some(new_root);
        }
        Output::ChildPageRoots(_, _) => unreachable!(),
    }

    Ok(output)
}

// helper for iterating all paths in the range and performing
// updates.
//
// anything that touches the root page is deferred via `shared.pending`.
struct RangeCommitter<H> {
    root: Node,
    shared: Arc<CommitShared>,
    write_pass: WritePass<ShardIndex>,
    region: PageRegion,
    page_walker: PageWalker<H>,
    range_start: usize,
    range_end: usize,
    saved_advance: Option<SavedAdvance>,
}

impl<H: NodeHasher> RangeCommitter<H> {
    fn new(
        root: Node,
        shared: Arc<CommitShared>,
        write_pass: WritePass<ShardIndex>,
        page_cache: &PageCache,
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

        RangeCommitter {
            root,
            shared,
            write_pass,
            region,
            page_walker: PageWalker::<H>::new(root, page_cache.clone(), Some(ROOT_PAGE_ID)),
            range_start,
            range_end,
            saved_advance: None,
        }
    }

    // returns the end index of the batch. returns the end index of the batches covered by
    // this terminal.
    fn handle_completion(
        &mut self,
        seeker: &mut NewSeeker,
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
                            None => PathProofTerminal::Terminator(
                                seek_result.position.path().to_bitvec(),
                            ),
                        },
                    },
                    path: seek_result.position.path().to_bitvec(),
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
        seeker: &mut NewSeeker,
        output: &mut WorkerOutput,
        seek_result: Seek,
        ops: Option<Vec<(KeyPath, Option<ValueHash>)>>,
        batch_size: usize,
    ) {
        let res = match ops {
            None => self
                .page_walker
                .advance(&mut self.write_pass, seek_result.position.clone()),
            Some(ops) => {
                let ops = nomt_core::update::leaf_ops_spliced(seek_result.terminal.clone(), &ops);
                self.page_walker.advance_and_replace(
                    &mut self.write_pass,
                    seek_result.position.clone(),
                    ops,
                )
            }
        };

        // TODO: introduce an error in page walker and handle it by setting `SavedAdvance` to
        // some, scheduling the page load in the seeker, and returning.

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
                        None => {
                            PathProofTerminal::Terminator(seek_result.position.path().to_bitvec())
                        }
                    },
                },
                path: seek_result.position.path().to_bitvec(),
            };
            witnessed_paths.push((path, seek_result.terminal, batch_size));
        }
    }

    fn reattempt_advance(&mut self, seeker: &mut NewSeeker, output: &mut WorkerOutput) {
        // UNWRAP: guaranteed by behavior of seeker / commit / advance.
        let SavedAdvance {
            ops,
            seek_result,
            batch_size,
        } = self.saved_advance.take().unwrap();
        self.attempt_advance(seeker, output, seek_result, ops, batch_size);
    }

    fn commit(
        mut self,
        seeker: &mut NewSeeker,
        output: &mut WorkerOutput,
    ) -> anyhow::Result<Option<WritePass<ShardIndex>>> {
        let mut start_index = self.range_start;
        let mut pushes = 0;
        let mut skips = 0;

        let mut conclude_needs_extra = false;

        loop {
            if start_index >= self.range_end && !conclude_needs_extra {
                // PANIC: walker was configured with a parent page.
                let (new_nodes, diffs) = match self.page_walker.conclude(&mut self.write_pass) {
                    Output::Root(_, _) => unreachable!(),
                    Output::ChildPageRoots(new_nodes, diffs) => (new_nodes, diffs),
                    // TODO: handle need for extra page.
                };

                assert!(!diffs.contains_key(&ROOT_PAGE_ID));
                output.page_diffs = diffs;

                self.shared.push_pending_root_nodes(new_nodes);

                return Ok(self.write_pass.consume());
            }

            match seeker.advance(self.write_pass.downgrade())? {
                Interrupt::NoMoreWork | Interrupt::HasRoom => {
                    let next_push = start_index + pushes;
                    if next_push < self.range_end {
                        seeker.push(self.shared.read_write[next_push].0);
                        pushes += 1;
                    }
                }
                Interrupt::Completion(seek_result) => {
                    // skip completions until we're past the end of the last batch.
                    if skips > 0 {
                        skips -= 1;
                        continue;
                    }

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
                Interrupt::SpecialPageCompletion => {
                    if conclude_needs_extra {
                        conclude_needs_extra = false;
                    } else {
                        self.reattempt_advance(seeker, output);
                    }
                }
            }
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
