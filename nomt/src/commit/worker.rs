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

use crossbeam::channel::{Receiver, Sender};

use nomt_core::{
    page_id::ROOT_PAGE_ID,
    trie::{KeyPath, Node, NodeHasher, ValueHash},
};

use std::sync::{Arc, Barrier};

use super::{
    CommitCommand, CommitShared, KeyReadWrite, RootPagePending, ToWorker, WarmUpCommand,
    WorkerOutput,
};

use crate::{
    page_cache::PageCache,
    page_region::PageRegion,
    page_walker::{Output, PageWalker},
    rw_pass_cell::{ReadPass, WritePass},
    seek::{SeekOptions, Seeker},
    PathProof, WitnessedPath,
};

pub(super) struct Params {
    pub page_cache: PageCache,
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
        root,
        barrier,
    } = params;

    let mut read_pass = Some(page_cache.new_read_pass());
    barrier.wait();

    loop {
        crossbeam::select! {
            recv(comms.commit_rx) -> msg => match msg {
                Ok(ToWorker::Prepare) => {
                    let _ = read_pass.take();
                    break
                }
                // UNWRAP: `Commit` only sent after Prepare.
                Ok(ToWorker::Commit(_)) => unreachable!(),
                Err(_) => return,
            },
            recv(comms.warmup_rx) -> msg => match msg {
                Ok(command) => warm_up(
                    read_pass.as_ref().unwrap(),
                    root,
                    page_cache.clone(),
                    command,
                ),
                Err(_) => return,
            },
        }
    }

    match comms.commit_rx.recv() {
        Err(_) => return, // early exit only.
        // UNWRAP: Commit always sent after Prepare.
        Ok(ToWorker::Prepare) => unreachable!(),
        Ok(ToWorker::Commit(command)) => {
            let output = commit::<H>(root, page_cache, command);
            let _ = comms.output_tx.send(output);
        }
    }
}

fn warm_up(
    read_pass: &ReadPass<PageRegion>,
    root: Node,
    page_cache: PageCache,
    command: WarmUpCommand,
) {
    let WarmUpCommand { key_path, delete } = command;

    let seeker = Seeker::new(root, page_cache);
    let _seek_result = seeker.seek(
        key_path,
        SeekOptions {
            retrieve_sibling_leaf_children: delete,
            record_siblings: false,
        },
        read_pass,
    );
}

fn commit<H: NodeHasher>(
    root: Node,
    page_cache: PageCache,
    command: CommitCommand,
) -> WorkerOutput {
    let CommitCommand { shared, write_pass } = command;
    let write_pass = write_pass.into_inner();

    let mut output = WorkerOutput::default();

    let mut committer =
        RangeCommitter::<H>::new(root, shared.clone(), write_pass, page_cache.clone());

    while !committer.is_done() {
        committer.consume_terminal(&mut output);
    }

    // one lucky thread gets the master write pass.
    let mut write_pass = match committer.conclude(&mut output) {
        None => return output,
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

    output
}

// helper for iterating all paths in the range and performing
// updates.
//
// anything that touches the root page is deferred via `shared.pending`.
struct RangeCommitter<H> {
    root: Node,
    shared: Arc<CommitShared>,
    write_pass: WritePass<PageRegion>,
    page_walker: PageWalker<H>,
    page_cache: PageCache,
    range_start: usize,
    range_end: usize,
    cur_index: usize,
}

impl<H: NodeHasher> RangeCommitter<H> {
    fn new(
        root: Node,
        shared: Arc<CommitShared>,
        write_pass: WritePass<PageRegion>,
        page_cache: PageCache,
    ) -> Self {
        let key_range_start = write_pass.region().exclusive_min().min_key_path();
        let key_range_end = write_pass.region().exclusive_max().max_key_path();

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
            page_walker: PageWalker::<H>::new(root, page_cache.clone(), Some(ROOT_PAGE_ID)),
            page_cache,
            range_start,
            range_end,
            cur_index: range_start,
        }
    }

    fn is_done(&self) -> bool {
        self.cur_index >= self.range_end
    }

    fn consume_terminal(&mut self, output: &mut WorkerOutput) {
        assert!(!self.is_done());

        // TODO: it'd be slightly more efficient but more complex to interleave page fetches
        // and page updates, such that while we're waiting on a page fetch we process the previous
        // terminal, and so on.

        let seeker = Seeker::new(self.root, self.page_cache.clone());
        let start_index = self.cur_index;
        let seek_result = seeker.seek(
            self.shared.read_write[start_index].0,
            SeekOptions {
                // note: this is a very imperfect heuristic because multiple entries could
                // map to the same terminal node.
                retrieve_sibling_leaf_children: self.shared.read_write[start_index].1.is_delete(),
                record_siblings: true,
            },
            self.write_pass.downgrade(),
        );

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
        self.cur_index = next_index;

        // witness / pushing pending responsibility falls on the worker whose range this falls
        // inside.
        if !batch_starts_in_our_range {
            return;
        }

        let is_non_exclusive = seek_result.page_id.as_ref().map_or(true, |p_id| {
            !self.write_pass.region().contains_exclusive(p_id)
        });

        let siblings = if is_non_exclusive {
            self.shared.push_pending_subtrie(
                seek_result.position.clone(),
                start_index,
                next_index,
                seek_result.terminal.clone(),
            );

            // if the terminal lands in the non-exclusive area, then the path to it is guaranteed
            // not to have been altered by anything we've done so far.
            seek_result.siblings
        } else {
            if !has_writes {
                self.page_walker
                    .advance(&mut self.write_pass, seek_result.position.clone());
            } else {
                let ops = subtrie_ops(&self.shared.read_write[start_index..next_index]);
                let ops = nomt_core::update::leaf_ops_spliced(seek_result.terminal.clone(), &ops);
                self.page_walker.advance_and_replace(
                    &mut self.write_pass,
                    seek_result.position.clone(),
                    ops,
                );
            }

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
                terminal: seek_result.terminal.clone(),
            },
            path: seek_result.position.path().to_bitvec(),
        };
        output
            .witnessed_paths
            .push((path, seek_result.terminal, batch_size));
    }

    fn conclude(mut self, output: &mut WorkerOutput) -> Option<WritePass<PageRegion>> {
        // PANIC: walker was configured with a parent page.
        let (new_nodes, diffs) = match self.page_walker.conclude(&mut self.write_pass) {
            Output::Root(_, _) => unreachable!(),
            Output::ChildPageRoots(new_nodes, diffs) => (new_nodes, diffs),
        };

        assert!(!diffs.contains_key(&ROOT_PAGE_ID));
        output.page_diffs = diffs;

        self.shared.push_pending_root_nodes(new_nodes);

        self.write_pass.consume()
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
