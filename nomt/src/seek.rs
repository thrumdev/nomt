//! Multiplexer for page requests.

use crate::{
    page_cache::{Page, PageCache, ShardIndex},
    rw_pass_cell::ReadPass,
    store::{BucketIndex, PageLoad, PageLoadAdvance, PageLoadCompletion, PageLoader},
};

use nomt_core::{
    page::DEPTH,
    page_id::{PageId, ROOT_PAGE_ID},
    trie::{self, KeyPath, Node},
    trie_pos::{ChildNodeIndices, TriePosition},
};

use std::collections::{
    hash_map::{Entry, HashMap},
    VecDeque,
};

use bitvec::prelude::*;
use slab::Slab;

const MAX_REQUESTS: usize = 64;
const MAX_PARALLEL_LOADS: usize = 32;
const SINGLE_PAGE_REQUEST_INDEX: usize = usize::MAX;

struct SeekRequest {
    key: KeyPath,
    position: TriePosition,
    page_id: Option<PageId>,
    siblings: Vec<Node>,
    state: RequestState,
    continuations: usize,
}

impl SeekRequest {
    fn new(key: KeyPath, root: Node) -> SeekRequest {
        let state = if trie::is_terminator(&root) {
            RequestState::Completed(None)
        } else {
            RequestState::Seeking(root)
        };

        SeekRequest {
            key,
            position: TriePosition::new(),
            page_id: None,
            siblings: Vec::new(),
            state,
            continuations: 0,
        }
    }

    fn is_completed(&self) -> bool {
        match self.state {
            RequestState::Seeking(_) => false,
            RequestState::Completed(_) => true,
        }
    }

    fn next_page_id(&self) -> PageId {
        match self.page_id {
            None => ROOT_PAGE_ID,
            // UNWRAP: all page IDs for key paths are in scope.
            Some(ref page_id) => page_id
                .child_page_id(self.position.child_page_index())
                .unwrap(),
        }
    }

    fn continue_seek(
        &mut self,
        read_pass: &ReadPass<ShardIndex>,
        page_id: PageId,
        page: &Page,
        record_siblings: bool,
    ) {
        let RequestState::Seeking(mut cur_node) = self.state else {
            panic!("seek past end")
        };

        self.continuations += 1;

        assert!(self.continuations < 10);

        // terminator should have yielded completion.
        assert!(!trie::is_terminator(&cur_node));
        assert!(self.position.depth() as usize % DEPTH == 0);

        // don't set this when cur node is leaf, as we loaded this page just to get the leaf
        // children.
        if !trie::is_leaf(&cur_node) {
            self.page_id = Some(page_id);
        }

        // take enough bits to get us to the end of this page.
        let bits = self.key.view_bits::<Msb0>()[self.position.depth() as usize..]
            .iter()
            .by_vals()
            .take(DEPTH);

        for bit in bits {
            if trie::is_leaf(&cur_node) {
                let children = if self.position.depth() as usize % DEPTH > 0 {
                    self.position.child_node_indices()
                } else {
                    // note: the only time this is true in this loop is in the first iteration,
                    // because we only take `DEPTH` bits.
                    ChildNodeIndices::next_page()
                };

                self.state = RequestState::Completed(Some(trie::LeafData {
                    key_path: page.node(&read_pass, children.left()),
                    value_hash: page.node(&read_pass, children.right()),
                }));
                return;
            }

            self.position.down(bit);

            cur_node = page.node(&read_pass, self.position.node_index());
            if record_siblings {
                self.siblings
                    .push(page.node(&read_pass, self.position.sibling_index()));
            }

            if trie::is_terminator(&cur_node) {
                self.state = RequestState::Completed(None);
                return;
            }

            // leaf is handled in next iteration _or_ if this is the last iteration, in the next
            // `continue_seek`.
        }

        self.state = RequestState::Seeking(cur_node);
    }
}

enum RequestState {
    Seeking(Node),
    Completed(Option<trie::LeafData>),
}

enum SinglePageRequestState {
    Pending(PageId),
    Submitted,
    Completed,
}

/// The `Seeker` seeks for the terminal nodes of multiple keys simultaneously, multiplexing requests
/// over an I/O pool. Advance the seeker, provide new keys, and handle completions. A key path
/// request is considered completed when the terminal node has been found, and all pages along the
/// path to the terminal node are in the page cache. This includes the page that contains the leaf
/// children of the request.
///
/// While requests are normally completed in the order they're submitted,
/// it is possible to submit a special request for a single page, which will be prioritized over
/// other requests and whose completion will be handled first. This has a particular use case in
/// mind: the fetching of sibling leaf node child pages, as is an edge case during merkle tree
/// updates.
pub struct Seeker {
    root: Node,
    page_cache: PageCache,
    page_loader: PageLoader,
    processed: usize,
    requests: VecDeque<(bool, SeekRequest)>,
    page_loads: HashMap<PageId, Vec<usize>>,
    page_load_slab: Slab<PageLoad>,
    /// FIFO, pushed onto back, except when trying the front item and getting blocked.
    slab_retries: VecDeque<usize>,
    record_siblings: bool,
    single_page_request: Option<SinglePageRequestState>,
}

impl Seeker {
    /// Create a new `Seeker`.
    pub fn new(
        root: Node,
        page_cache: PageCache,
        page_loader: PageLoader,
        record_siblings: bool,
    ) -> Self {
        Seeker {
            root,
            page_cache,
            page_loader,
            processed: 0,
            requests: VecDeque::new(),
            page_loads: HashMap::new(),
            page_load_slab: Slab::with_capacity(MAX_PARALLEL_LOADS),
            slab_retries: VecDeque::new(),
            record_siblings,
            single_page_request: None,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    /// Advance the dispatcher until it's interrupted for some reason or an error occurs.
    ///
    /// When `Interrupt::HasRoom` is returned, call `push` if new work is available. Otherwise,
    /// don't block.
    /// When `Interrupt::NoMoreWork` is returned, call `push` if new work is available. Block if
    /// necessary.
    pub fn advance(&mut self, read_pass: &ReadPass<ShardIndex>) -> anyhow::Result<Interrupt> {
        loop {
            self.queue_page_requests(read_pass);
            let blocked = self.submit_slab_requests(read_pass)?;

            if let Some(SinglePageRequestState::Completed) = self.single_page_request {
                self.single_page_request = None;
                return Ok(Interrupt::SpecialPageCompletion);
            }

            if self
                .requests
                .front()
                .map_or(false, |(_, r)| r.is_completed())
            {
                // UNWRAP: just checked existence.
                let request = self.requests.pop_front().unwrap().1;
                // PANIC: checked above.
                let RequestState::Completed(terminal) = request.state else {
                    unreachable!()
                };

                self.processed += 1;

                return Ok(Interrupt::Completion(Seek {
                    key: request.key,
                    position: request.position,
                    page_id: request.page_id,
                    siblings: request.siblings,
                    terminal,
                }));
            }

            let completion = if blocked {
                Some(self.page_loader.complete()?)
            } else {
                self.page_loader.try_complete()?
            };

            if let Some(completion) = completion {
                self.handle_completion(read_pass, completion);
            } else if self.requests.is_empty() {
                return Ok(Interrupt::NoMoreWork);
            } else if !blocked && self.page_load_slab.len() < MAX_PARALLEL_LOADS && self.requests.len() < MAX_REQUESTS {
                // no completion, not blocked, slab not full, more requests might exist.
                return Ok(Interrupt::HasRoom);
            }
        }
    }

    /// Push a request for key path.
    pub fn push(&mut self, key: KeyPath) {
        self.requests
            .push_back((false, SeekRequest::new(key, self.root)));
    }

    /// Push a request for a single page to jump the line. Panics if another special page
    /// request has been pushed, but is not completed.
    pub fn push_single_request(&mut self, page_id: PageId) {
        assert!(self.single_page_request.is_none());
        self.single_page_request = Some(SinglePageRequestState::Pending(page_id));
    }

    fn submit_slab_requests(&mut self, read_pass: &ReadPass<ShardIndex>) -> anyhow::Result<bool> {
        while let Some(slab_index) = self.slab_retries.pop_front() {
            let page_load = &mut self.page_load_slab[slab_index];
            assert!(!page_load.needs_completion());
            match self.page_loader.try_advance(page_load, slab_index as u64)? {
                PageLoadAdvance::Blocked => {
                    self.slab_retries.push_front(slab_index);
                    return Ok(true);
                }
                PageLoadAdvance::Submitted => {}
                PageLoadAdvance::GuaranteedFresh => {
                    self.remove_and_continue_seeks(read_pass, slab_index, None)
                }
            }
        }

        Ok(false)
    }

    fn queue_page_requests(&mut self, read_pass: &ReadPass<ShardIndex>) {
        if self.page_load_slab.len() == MAX_PARALLEL_LOADS {
            return;
        }

        if let Some(SinglePageRequestState::Pending(ref page_id)) = self.single_page_request {
            let page_id = page_id.clone();
            self.page_loads
                .entry(page_id.clone())
                .or_default()
                .push(SINGLE_PAGE_REQUEST_INDEX);
            let page_load = self.page_loader.start_load(page_id);
            let slab_index = self.page_load_slab.insert(page_load);
            self.slab_retries.push_front(slab_index);
            self.single_page_request = Some(SinglePageRequestState::Submitted);
        }

        'a: for (i, (ref mut is_requesting, ref mut request)) in self
            .requests
            .iter_mut()
            .enumerate()
        {
            while !*is_requesting && !request.is_completed() {
                if self.page_load_slab.len() == MAX_PARALLEL_LOADS {
                    return;
                }

                let request_index = self.processed + i;

                let page_id: PageId = request.next_page_id();

                if let Some(page) = self.page_cache.get(page_id.clone()) {
                    request.continue_seek(read_pass, page_id, &page, self.record_siblings);
                    continue;
                }

                let vacant_entry = match self.page_loads.entry(page_id.clone()) {
                    Entry::Occupied(mut occupied) => {
                        occupied.get_mut().push(request_index);
                        *is_requesting = true;
                        continue 'a;
                    }
                    Entry::Vacant(vacant) => vacant,
                };

                let load = self.page_loader.start_load(page_id.clone());
                vacant_entry.insert(vec![request_index]);
                *is_requesting = true;
                let slab_index = self.page_load_slab.insert(load);
                self.slab_retries.push_back(slab_index);
            }
        }
    }

    fn handle_completion(
        &mut self,
        read_pass: &ReadPass<ShardIndex>,
        completion: PageLoadCompletion,
    ) {
        let slab_index = completion.user_data() as usize;

        // UNWRAP: requests are submitted with slab indices that are populated and never cleared
        // until this point is reached.
        let page_load = self.page_load_slab.get_mut(slab_index).unwrap();

        match completion.apply_to(page_load) {
            Some(p) => self.remove_and_continue_seeks(read_pass, slab_index, Some(p)),
            None => {
                self.slab_retries.push_back(slab_index);
                return;
            }
        }
    }

    fn remove_and_continue_seeks(
        &mut self,
        read_pass: &ReadPass<ShardIndex>,
        slab_index: usize,
        page_data: Option<(Vec<u8>, BucketIndex)>,
    ) {
        let page_load = self.page_load_slab.remove(slab_index);
        let page = self
            .page_cache
            .insert(page_load.page_id().clone(), page_data);

        for waiting_request in self
            .page_loads
            .remove(page_load.page_id())
            .into_iter()
            .flatten()
        {
            if waiting_request == SINGLE_PAGE_REQUEST_INDEX {
                self.single_page_request = Some(SinglePageRequestState::Completed);
                continue;
            }
            if waiting_request < self.processed {
                continue;
            }
            let idx = waiting_request - self.processed;
            let (ref mut is_requesting, ref mut request) = &mut self.requests[idx];
            assert!(!request.is_completed());

            *is_requesting = false;
            request.continue_seek(
                read_pass,
                page_load.page_id().clone(),
                &page,
                self.record_siblings,
            );
        }
    }
}

/// Interrupts during the advancement of the `Seeker`.
pub enum Interrupt {
    /// There is room to push a key, if any are available.
    HasRoom,
    /// There is no more work to perform at the moment.
    NoMoreWork,
    /// A request was completed.
    Completion(Seek),
    /// The request for a single page completed.
    SpecialPageCompletion,
}

/// The result of a seek.
pub struct Seek {
    /// The key being sought.
    pub key: KeyPath,
    /// The position in the trie where the terminal node was found.
    pub position: TriePosition,
    /// The page ID where the terminal node was found. `None` at root.
    pub page_id: Option<PageId>,
    /// The siblings along the path to the terminal, including the terminal's sibling.
    /// Empty if the seeker wasn't configured to record siblings.
    pub siblings: Vec<Node>,
    /// The terminal node encountered.
    pub terminal: Option<trie::LeafData>,
}
