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

use crossbeam::channel::{Receiver, Select, Sender};

use nomt_core::{
    page_id::{PageId, ROOT_PAGE_ID},
    proof::PathProofTerminal,
    trie::{KeyPath, Node, NodeHasher, ValueHash},
};

use std::{
    sync::{Arc, Barrier},
    time::Duration,
};

use super::{
    CommitCommand, CommitShared, KeyReadWrite, RootPagePending, ToWorker, WarmUpCommand,
    WorkerOutput,
};

use crate::{
    page_cache::{PageCache, ShardIndex},
    page_region::PageRegion,
    page_walker::{NeedsPage, Output, PageWalker},
    rw_pass_cell::{ReadPass, WritePass},
    seek::{Interrupt, Seek, Seeker},
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

    let seeker = Seeker::new(root, page_cache.clone(), store.page_loader(), false);

    if let Err(_) = warm_up_phase(&comms, read_pass, seeker) {
        return;
    };

    match comms.commit_rx.recv() {
        Err(_) => return, // early exit only.
        // UNWRAP: Commit always sent after Prepare.
        Ok(ToWorker::Prepare) => unreachable!(),
        Ok(ToWorker::Commit(command)) => {
            let seeker = Seeker::new(
                root,
                page_cache.clone(),
                store.page_loader(),
                command.shared.witness,
            );

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
    mut seeker: Seeker,
) -> anyhow::Result<()> {
    const CHECK_PREPARE_DURATION: Duration = Duration::from_micros(250);

    let mut preparing = false;
    loop {
        match seeker.advance_deadline(&read_pass, CHECK_PREPARE_DURATION)? {
            Interrupt::HasRoom => {
                let is_empty = seeker.is_empty();
                if preparing && is_empty {
                    return Ok(());
                } else if preparing {
                    continue;
                }

                let mut select = Select::new();
                let commit_idx = select.recv(&comms.commit_rx);
                let _ = select.recv(&comms.warmup_rx);

                let selected = if is_empty {
                    select.select()
                } else {
                    match select.try_select() {
                        Err(_) => continue,
                        Ok(selected) => selected,
                    }
                };

                if selected.index() == commit_idx {
                    match selected.recv(&comms.commit_rx)? {
                        ToWorker::Prepare => {
                            preparing = true;
                            if is_empty {
                                return Ok(());
                            } else {
                                continue;
                            }
                        }
                        ToWorker::Commit(_) => unreachable!(),
                    }
                } else {
                    seeker.push(selected.recv(&comms.warmup_rx)?.key_path)
                }
            }
            Interrupt::Deadline if !preparing => {
                if let Ok(ToWorker::Prepare) = comms.commit_rx.try_recv() {
                    preparing = true;
                }
            }
            _ => continue,
        };
    }
}

fn commit<H: NodeHasher>(
    root: Node,
    page_cache: PageCache,
    mut seeker: Seeker,
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
            RootPagePending::Node(node) => loop {
                let page = match root_page_committer.advance_and_place_node(
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
                    let page = match root_page_committer.advance_and_replace(
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
        let page = match root_page_committer.conclude(&mut write_pass) {
            Ok(Output::Root(new_root, diffs)) => {
                for (page_id, page_diff) in diffs {
                    output.page_diffs.insert(page_id, page_diff);
                }

                output.root = Some(new_root);
                break;
            }
            Ok(Output::ChildPageRoots(_, _)) => unreachable!(),
            Err((NeedsPage(page), page_walker)) => {
                root_page_committer = page_walker;
                page
            }
        };
        drive_page_fetch(&mut seeker, write_pass.downgrade(), page)?;
    }

    Ok(output)
}

fn drive_page_fetch(
    seeker: &mut Seeker,
    read_pass: &ReadPass<ShardIndex>,
    page: PageId,
) -> anyhow::Result<()> {
    seeker.push_single_request(page);
    loop {
        match seeker.advance(read_pass) {
            Err(e) => return Err(e),
            Ok(Interrupt::SpecialPageCompletion) => return Ok(()),
            _ => continue,
        }
    }
}

// helper for iterating all paths in the range and performing
// updates.
//
// anything that touches the root page is deferred via `shared.pending`.
struct RangeCommitter<H> {
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
        seeker: &mut Seeker,
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
        seeker: &mut Seeker,
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

    fn reattempt_advance(&mut self, seeker: &mut Seeker, output: &mut WorkerOutput) {
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
        seeker: &mut Seeker,
        output: &mut WorkerOutput,
    ) -> anyhow::Result<Option<WritePass<ShardIndex>>> {
        let mut start_index = self.range_start;
        let mut pushes = 0;
        let mut skips = 0;

        // 1. drive until work is done.
        loop {
            match seeker.advance(self.write_pass.downgrade())? {
                Interrupt::HasRoom => {
                    let next_push = start_index + pushes;
                    if next_push < self.range_end {
                        pushes += 1;
                        seeker.push(self.shared.read_write[next_push].0);
                    } else if seeker.is_empty() {
                        break;
                    }
                }
                Interrupt::Deadline => {}
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
                    self.reattempt_advance(seeker, output);
                }
            }
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

            assert!(!diffs.contains_key(&ROOT_PAGE_ID));
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
