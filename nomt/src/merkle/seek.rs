//! Multiplexer for page requests.

use crate::{
    beatree::{
        iterator::NeededLeavesIter, AsyncLeafLoad, BeatreeIterator, LeafNodeRef, PageNumber,
        ReadTransaction,
    },
    io::{CompleteIo, FatPage, IoHandle},
    page_cache::{Page, PageCache, ShardIndex},
    rw_pass_cell::ReadPass,
    store::{BucketIndex, PageLoad, PageLoader},
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

const MAX_INFLIGHT: usize = 1024;
const SINGLE_PAGE_REQUEST_INDEX: usize = usize::MAX;

struct SeekRequest {
    key: KeyPath,
    position: TriePosition,
    page_id: Option<PageId>,
    siblings: Vec<Node>,
    state: RequestState,
    ios: usize,
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
            ios: 0,
        }
    }

    fn note_io(&mut self) {
        self.ios += 1;
    }

    fn is_completed(&self) -> bool {
        match self.state {
            RequestState::Seeking(_) => false,
            RequestState::FetchingLeaf(_, _) => false,
            RequestState::Completed(_) => true,
        }
    }

    fn next_query(&mut self) -> Option<IoQuery> {
        match self.state {
            RequestState::Seeking(_) => Some(IoQuery::MerklePage(match self.page_id {
                None => ROOT_PAGE_ID,
                // UNWRAP: all page IDs for key paths are in scope.
                Some(ref page_id) => page_id
                    .child_page_id(self.position.child_page_index())
                    .unwrap(),
            })),
            RequestState::FetchingLeaf(_, ref mut needed) => {
                needed.next().map(|pn| IoQuery::LeafPage(pn))
            }
            RequestState::Completed(_) => None,
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

        if self.position.depth() == 256 {
            // If we have reached the last layer of the trie, we are sure that it needs to be a leaf,
            // and its children will be part of the same last page, which is only partially used
            assert!(trie::is_leaf(&cur_node));
            let children = self.position.child_node_indices();
            self.state = RequestState::Completed(Some(trie::LeafData {
                key_path: page.node(&read_pass, children.left()),
                value_hash: page.node(&read_pass, children.right()),
            }));
        } else {
            self.state = RequestState::Seeking(cur_node);
        }
    }
}

enum RequestState {
    Seeking(Node),
    FetchingLeaf(BeatreeIterator, NeededLeavesIter),
    Completed(Option<trie::LeafData>),
}

enum SinglePageRequestState {
    Pending(PageId),
    Submitted,
    Completed,
}

#[derive(Hash, PartialEq, Eq)]
enum IoQuery {
    MerklePage(PageId),
    LeafPage(PageNumber),
}

enum IoRequest {
    Merkle(PageLoad),
    Leaf(AsyncLeafLoad),
}

impl IoRequest {
    fn is_active(&self) -> bool {
        match self {
            IoRequest::Merkle(load) => load.needs_completion(),
            IoRequest::Leaf(_) => true,
        }
    }
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
    io_handle: IoHandle,
    page_loader: PageLoader,
    processed: usize,
    requests: VecDeque<SeekRequest>,
    io_waiters: HashMap<IoQuery, Vec<usize>>,
    io_slab: Slab<IoRequest>,
    /// FIFO, pushed onto back.
    idle_requests: VecDeque<usize>,
    /// FIFO, pushed onto back, except when trying the front item and getting blocked.
    idle_page_loads: VecDeque<usize>,
    record_siblings: bool,
    single_page_request: Option<SinglePageRequestState>,
}

impl Seeker {
    /// Create a new `Seeker`.
    pub fn new(
        root: Node,
        page_cache: PageCache,
        io_handle: IoHandle,
        page_loader: PageLoader,
        record_siblings: bool,
    ) -> Self {
        Seeker {
            root,
            page_cache,
            io_handle,
            page_loader,
            processed: 0,
            requests: VecDeque::new(),
            io_waiters: HashMap::new(),
            io_slab: Slab::new(),
            idle_requests: VecDeque::new(),
            idle_page_loads: VecDeque::new(),
            record_siblings,
            single_page_request: None,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.requests.is_empty() && self.single_page_request.is_none()
    }

    pub fn has_room(&self) -> bool {
        self.io_waiters.len() < MAX_INFLIGHT
    }

    pub fn first_key(&self) -> Option<&KeyPath> {
        self.requests.front().map(|request| &request.key)
    }

    pub fn has_live_requests(&self) -> bool {
        self.idle_page_loads.len() < self.io_slab.len()
    }

    /// Try to submit as many requests as possible. Returns `true` if blocked.
    pub fn submit_all(&mut self, read_pass: &ReadPass<ShardIndex>) -> anyhow::Result<bool> {
        if !self.has_room() {
            return Ok(true);
        }
        let blocked = self.submit_idle_page_loads(read_pass)?
            || self.submit_single_page_request(read_pass)?
            || self.submit_idle_key_path_requests(read_pass)?;

        Ok(blocked)
    }

    /// Take the result of a complete request.
    pub fn take_completion(&mut self) -> Option<Completion> {
        match self.single_page_request {
            Some(SinglePageRequestState::Completed) => {
                self.single_page_request = None;
                return Some(Completion::SinglePage);
            }
            Some(_) => return None,
            None => (),
        }

        if self.requests.front().map_or(false, |r| r.is_completed()) {
            // UNWRAP: just checked existence.
            let request = self.requests.pop_front().unwrap();
            // PANIC: checked above.
            let RequestState::Completed(terminal) = request.state else {
                unreachable!()
            };

            self.processed += 1;

            return Some(Completion::Seek(Seek {
                key: request.key,
                position: request.position,
                page_id: request.page_id,
                siblings: request.siblings,
                ios: request.ios,
                terminal,
            }));
        }

        None
    }

    /// Try to process the next I/O. Does not block the current thread.
    pub fn try_recv_page(&mut self, read_pass: &ReadPass<ShardIndex>) -> anyhow::Result<()> {
        if let Ok(io) = self.io_handle.try_recv() {
            self.handle_completion(read_pass, io)?;
        }

        Ok(())
    }

    /// Block on processing the next I/O. Blocks the current thread.
    pub fn recv_page(&mut self, read_pass: &ReadPass<ShardIndex>) -> anyhow::Result<()> {
        let io = self.io_handle.recv()?;
        self.handle_completion(read_pass, io)?;
        Ok(())
    }

    /// Push a request for key path.
    pub fn push(&mut self, key: KeyPath) {
        let request_index = self.processed + self.requests.len();
        self.requests.push_back(SeekRequest::new(key, self.root));
        self.idle_requests.push_back(request_index);
    }

    /// Push a request for a single page to jump the line. Panics if another special page
    /// request has been pushed, but is not completed.
    pub fn push_single_request(&mut self, page_id: PageId) {
        assert!(self.single_page_request.is_none());
        self.single_page_request = Some(SinglePageRequestState::Pending(page_id));
    }

    // resubmit all idle page loads until blocked or no more remain. returns true if blocked
    fn submit_idle_page_loads(&mut self, read_pass: &ReadPass<ShardIndex>) -> anyhow::Result<bool> {
        while let Some(slab_index) = self.idle_page_loads.pop_front() {
            let blocked = self.submit_page_load(read_pass, slab_index, true)?;
            if blocked {
                return Ok(blocked);
            }
        }

        Ok(false)
    }

    // submit the next page for each idle key path request until blocked or no more remain.
    // returns true if blocked.
    fn submit_idle_key_path_requests(
        &mut self,
        read_pass: &ReadPass<ShardIndex>,
    ) -> anyhow::Result<bool> {
        while let Some(request_index) = self.idle_requests.pop_front() {
            let blocked = self.submit_key_path_request(read_pass, request_index)?;
            if blocked {
                return Ok(blocked);
            }
        }

        Ok(false)
    }

    // submit a page load which is currently in the slab, but idle. returns true if blocked.
    fn submit_page_load(
        &mut self,
        read_pass: &ReadPass<ShardIndex>,
        slab_index: usize,
        front: bool,
    ) -> anyhow::Result<bool> {
        if !self.has_room() {
            if front {
                self.idle_page_loads.push_front(slab_index);
            } else {
                self.idle_page_loads.push_back(slab_index)
            }
            return Ok(true);
        }

        match self.io_slab[slab_index] {
            IoRequest::Merkle(ref mut page_load) => {
                if !self
                    .page_loader
                    .probe(page_load, &self.io_handle, slab_index as u64)?
                {
                    // guaranteed fresh page
                    self.handle_merkle_page_and_continue(read_pass, slab_index, None);
                }
            }
            IoRequest::Leaf(ref mut _leaf_load) => {
                todo!()
            }
        }

        Ok(false)
    }

    // submit a single page request, if any exists. returns true if blocked.
    fn submit_single_page_request(
        &mut self,
        read_pass: &ReadPass<ShardIndex>,
    ) -> anyhow::Result<bool> {
        if let Some(SinglePageRequestState::Pending(ref page_id)) = self.single_page_request {
            let page_id = page_id.clone();
            let waiters = self
                .io_waiters
                .entry(IoQuery::MerklePage(page_id.clone()))
                .or_default();

            waiters.push(SINGLE_PAGE_REQUEST_INDEX);

            self.single_page_request = Some(SinglePageRequestState::Submitted);

            if waiters.len() == 1 {
                let page_load = self.page_loader.start_load(page_id);
                let slab_index = self.io_slab.insert(IoRequest::Merkle(page_load));
                return self.submit_page_load(read_pass, slab_index, false);
            }
        }

        Ok(false)
    }

    // submit the next page for this key path request. return true if blocked.
    fn submit_key_path_request(
        &mut self,
        read_pass: &ReadPass<ShardIndex>,
        request_index: usize,
    ) -> anyhow::Result<bool> {
        let i = if request_index < self.processed {
            return Ok(false);
        } else {
            request_index - self.processed
        };

        let request = &mut self.requests[i];

        while let Some(query) = request.next_query() {
            match query {
                IoQuery::MerklePage(page_id) => {
                    if let Some(page) = self.page_cache.get(page_id.clone()) {
                        request.continue_seek(read_pass, page_id, &page, self.record_siblings);
                        continue;
                    }

                    let vacant_entry =
                        match self.io_waiters.entry(IoQuery::MerklePage(page_id.clone())) {
                            Entry::Occupied(mut occupied) => {
                                assert!(!occupied.get().contains(&request_index));
                                occupied.get_mut().push(request_index);
                                break;
                            }
                            Entry::Vacant(vacant) => vacant,
                        };

                    request.note_io();

                    let load = self.page_loader.start_load(page_id.clone());
                    vacant_entry.insert(vec![request_index]);
                    let slab_index = self.io_slab.insert(IoRequest::Merkle(load));
                    return self.submit_page_load(read_pass, slab_index, false);
                }
                IoQuery::LeafPage(_page_number) => {
                    todo!()
                }
            }
        }

        Ok(false)
    }

    fn handle_completion(
        &mut self,
        read_pass: &ReadPass<ShardIndex>,
        io: CompleteIo,
    ) -> anyhow::Result<()> {
        io.result?;
        let slab_index = io.command.user_data as usize;

        // UNWRAP: requests are submitted with slab indices that are populated and never cleared
        // until this point is reached.
        match self.io_slab.get_mut(slab_index).unwrap() {
            IoRequest::Merkle(page_load) => {
                // UNWRAP: page loader always submits a `Read` command that yields a fat page.
                let page = io.command.kind.unwrap_buf();
                match page_load.try_complete(page) {
                    Some(p) => self.handle_merkle_page_and_continue(read_pass, slab_index, Some(p)),
                    None => self.idle_page_loads.push_back(slab_index),
                };
            }
            IoRequest::Leaf(_) => {
                todo!()
            }
        }

        Ok(())
    }

    fn handle_merkle_page_and_continue(
        &mut self,
        read_pass: &ReadPass<ShardIndex>,
        slab_index: usize,
        page_data: Option<(FatPage, BucketIndex)>,
    ) {
        let IoRequest::Merkle(page_load) = self.io_slab.remove(slab_index) else {
            panic!()
        };

        let page = self
            .page_cache
            .insert(page_load.page_id().clone(), page_data);

        for waiting_request in self
            .io_waiters
            .remove(&IoQuery::MerklePage(page_load.page_id().clone()))
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
            let request = &mut self.requests[idx];
            assert!(!request.is_completed());

            request.continue_seek(
                read_pass,
                page_load.page_id().clone(),
                &page,
                self.record_siblings,
            );

            if !request.is_completed() {
                self.idle_requests.push_back(waiting_request);
            }
        }
    }
}

/// Complete requests.
pub enum Completion {
    /// A seek request was completed.
    Seek(Seek),
    /// The single page request was completed.
    SinglePage,
}

/// The result of a seek.
#[derive(Clone)]
pub struct Seek {
    /// The key being sought.
    #[allow(dead_code)]
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
    /// The number of I/Os loaded uniquely for this `Seek`.
    /// This does not include pages loaded from the cache, or pages which were already requested
    /// for another seek.
    #[allow(dead_code)]
    pub ios: usize,
}
