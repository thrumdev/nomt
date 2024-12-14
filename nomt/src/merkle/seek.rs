//! Multiplexer for page requests.

use crate::{
    beatree::{
        iterator::{IterOutput, NeededLeavesIter},
        AsyncLeafLoad, BeatreeIterator, LeafNodeRef, PageNumber, ReadTransaction as BeatreeReadTx,
    },
    io::{CompleteIo, FatPage, IoHandle},
    page_cache::{Page, PageCache, ShardIndex},
    rw_pass_cell::ReadPass,
    store::{BucketIndex, PageLoad, PageLoader},
    HashAlgorithm,
};

use nomt_core::{
    page::DEPTH,
    page_id::{PageId, ROOT_PAGE_ID},
    trie::{self, KeyPath, Node},
    trie_pos::TriePosition,
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
    fn new<H: HashAlgorithm>(
        read_transaction: &BeatreeReadTx,
        key: KeyPath,
        root: Node,
    ) -> SeekRequest {
        let state = if trie::is_terminator(&root) {
            RequestState::Completed(None)
        } else if trie::is_leaf(&root) {
            RequestState::begin_leaf_fetch(read_transaction, &TriePosition::new())
        } else {
            RequestState::Seeking
        };

        let mut request = SeekRequest {
            key,
            position: TriePosition::new(),
            page_id: None,
            siblings: Vec::new(),
            state,
            ios: 0,
        };

        if let RequestState::FetchingLeaf(_, _) = request.state {
            // we must advance the iterator until blocked.
            request.continue_leaf_fetch::<H>(None);
        }

        request
    }

    fn note_io(&mut self) {
        self.ios += 1;
    }

    fn is_completed(&self) -> bool {
        match self.state {
            RequestState::Seeking => false,
            RequestState::FetchingLeaf(_, _) => false,
            RequestState::Completed(_) => true,
        }
    }

    fn next_query(&mut self) -> Option<IoQuery> {
        match self.state {
            RequestState::Seeking => Some(IoQuery::MerklePage(match self.page_id {
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

    fn continue_seek<H: HashAlgorithm>(
        &mut self,
        read_pass: &ReadPass<ShardIndex>,
        read_transaction: &BeatreeReadTx,
        page_id: PageId,
        page: &Page,
        record_siblings: bool,
    ) {
        let RequestState::Seeking = self.state else {
            panic!("seek past end")
        };

        // terminator should have yielded completion.
        assert!(self.position.depth() as usize % DEPTH == 0);

        self.page_id = Some(page_id);

        // take enough bits to get us to the end of this page.
        let bits = self.key.view_bits::<Msb0>()[self.position.depth() as usize..]
            .iter()
            .by_vals()
            .take(DEPTH);

        let mut do_leaf_fetch = false;

        for bit in bits {
            self.position.down(bit);

            let cur_node = page.node(&read_pass, self.position.node_index());
            if record_siblings {
                self.siblings
                    .push(page.node(&read_pass, self.position.sibling_index()));
            }

            if trie::is_leaf(&cur_node) {
                self.state = RequestState::begin_leaf_fetch(read_transaction, &self.position);
                do_leaf_fetch = true;
                break;
            } else if trie::is_terminator(&cur_node) {
                self.state = RequestState::Completed(None);
                return;
            }
        }

        if do_leaf_fetch {
            self.continue_leaf_fetch::<H>(None);
        }
    }

    fn continue_leaf_fetch<H: HashAlgorithm>(&mut self, leaf: Option<LeafNodeRef>) {
        let RequestState::FetchingLeaf(ref mut iter, _) = self.state else {
            panic!("called continue_leaf_fetch without active iterator");
        };

        if let Some(leaf) = leaf {
            iter.provide_leaf(leaf);
        }

        let (key, value_hash) = match iter.next() {
            None => panic!("leaf must exist position={}", self.position.path()),
            Some(IterOutput::Blocked) => return,
            Some(IterOutput::Item(key, value)) => {
                (key, H::hash_value(&value)) // hash
            }
            Some(IterOutput::OverflowItem(key, value_hash, _)) => (key, value_hash),
        };

        self.state = RequestState::Completed(Some(trie::LeafData {
            key_path: key,
            value_hash,
        }));
    }
}

enum RequestState {
    Seeking,
    FetchingLeaf(BeatreeIterator, NeededLeavesIter),
    Completed(Option<trie::LeafData>),
}

impl RequestState {
    fn begin_leaf_fetch(read_transaction: &BeatreeReadTx, pos: &TriePosition) -> Self {
        let (start, end) = range_bounds(pos.raw_path(), pos.depth() as usize);
        let iter = read_transaction.iterator(start, end);
        let needed_leaves = iter.needed_leaves();
        RequestState::FetchingLeaf(iter, needed_leaves)
    }
}

fn range_bounds(raw_path: KeyPath, depth: usize) -> (KeyPath, Option<KeyPath>) {
    if depth == 0 {
        return (KeyPath::default(), None);
    }
    let start = raw_path;
    let mut end = start.clone();
    let mut depth = depth - 1;
    loop {
        let end_bits = end.view_bits_mut::<Msb0>();
        if !end_bits[depth] {
            end_bits.set(depth, true);
            break;
        } else {
            end_bits.set(depth, false);

            // this is only reached when the start position has the form `111111...`.
            if depth == 0 {
                return (start, None);
            } else {
                depth -= 1;
            }
        }
    }

    (start, Some(end))
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
pub struct Seeker<H: HashAlgorithm> {
    root: Node,
    beatree_read_transaction: BeatreeReadTx,
    page_cache: PageCache,
    io_handle: IoHandle,
    page_loader: PageLoader,
    processed: usize,
    requests: VecDeque<SeekRequest>,
    io_waiters: HashMap<IoQuery, Vec<usize>>,
    io_slab: Slab<IoRequest>,
    /// FIFO, pushed onto back.
    idle_requests: VecDeque<usize>,
    /// FIFO, pushed onto back.
    idle_page_loads: VecDeque<usize>,
    record_siblings: bool,
    single_page_request: Option<SinglePageRequestState>,
    _marker: std::marker::PhantomData<H>,
}

impl<H: HashAlgorithm> Seeker<H> {
    /// Create a new `Seeker`.
    pub fn new(
        root: Node,
        beatree_read_transaction: BeatreeReadTx,
        page_cache: PageCache,
        io_handle: IoHandle,
        page_loader: PageLoader,
        record_siblings: bool,
    ) -> Self {
        Seeker {
            root,
            beatree_read_transaction,
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
            _marker: std::marker::PhantomData,
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

    /// Try to submit as many requests as possible.
    pub fn submit_all(&mut self, read_pass: &ReadPass<ShardIndex>) -> anyhow::Result<()> {
        if !self.has_room() {
            return Ok(());
        }
        self.submit_idle_page_loads(read_pass)?;
        self.submit_single_page_request(read_pass)?;
        self.submit_idle_key_path_requests(read_pass)?;

        Ok(())
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
        self.requests.push_back(SeekRequest::new::<H>(
            &self.beatree_read_transaction,
            key,
            self.root,
        ));
        self.idle_requests.push_back(request_index);
    }

    /// Push a request for a single page to jump the line. Panics if another special page
    /// request has been pushed, but is not completed.
    pub fn push_single_request(&mut self, page_id: PageId) {
        assert!(self.single_page_request.is_none());
        self.single_page_request = Some(SinglePageRequestState::Pending(page_id));
    }

    // resubmit all idle page loads until no more remain.
    fn submit_idle_page_loads(&mut self, read_pass: &ReadPass<ShardIndex>) -> anyhow::Result<()> {
        while let Some(slab_index) = self.idle_page_loads.pop_front() {
            self.submit_idle_page_load(read_pass, slab_index)?;
        }

        Ok(())
    }

    // submit the next page for each idle key path request until backpressuring or no more progress
    // can be made.
    fn submit_idle_key_path_requests(
        &mut self,
        read_pass: &ReadPass<ShardIndex>,
    ) -> anyhow::Result<()> {
        while self.has_room() {
            match self.idle_requests.pop_front() {
                None => return Ok(()),
                Some(request_index) => {
                    self.submit_key_path_request(read_pass, request_index)?;
                }
            }
        }

        Ok(())
    }

    // submit a page load which is currently in the slab, but idle.
    fn submit_idle_page_load(
        &mut self,
        read_pass: &ReadPass<ShardIndex>,
        slab_index: usize,
    ) -> anyhow::Result<()> {
        if let IoRequest::Merkle(ref mut page_load) = self.io_slab[slab_index] {
            if !self
                .page_loader
                .probe(page_load, &self.io_handle, slab_index as u64)?
            {
                // guaranteed fresh page
                self.handle_merkle_page_and_continue(read_pass, slab_index, None);
            }
        }

        Ok(())
    }

    // submit a single page request, if any exists.
    fn submit_single_page_request(
        &mut self,
        read_pass: &ReadPass<ShardIndex>,
    ) -> anyhow::Result<()> {
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
                return self.submit_idle_page_load(read_pass, slab_index);
            }
        }

        Ok(())
    }

    // submit the next page for this key path request.
    fn submit_key_path_request(
        &mut self,
        read_pass: &ReadPass<ShardIndex>,
        request_index: usize,
    ) -> anyhow::Result<()> {
        let i = if request_index < self.processed {
            return Ok(());
        } else {
            request_index - self.processed
        };

        let request = &mut self.requests[i];

        while let Some(query) = request.next_query() {
            match query {
                IoQuery::MerklePage(page_id) => {
                    if let Some(page) = self.page_cache.get(page_id.clone()) {
                        request.continue_seek::<H>(
                            read_pass,
                            &self.beatree_read_transaction,
                            page_id,
                            &page,
                            self.record_siblings,
                        );
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
                    return self.submit_idle_page_load(read_pass, slab_index);
                }
                IoQuery::LeafPage(page_number) => {
                    let vacant_entry = match self.io_waiters.entry(IoQuery::LeafPage(page_number)) {
                        Entry::Occupied(mut occupied) => {
                            assert!(!occupied.get().contains(&request_index));
                            occupied.get_mut().push(request_index);
                            break;
                        }
                        Entry::Vacant(vacant) => vacant,
                    };

                    let slab_index = self.io_slab.vacant_key();
                    match self.beatree_read_transaction.load_leaf_async(
                        page_number,
                        &self.io_handle,
                        slab_index as u64,
                    ) {
                        Ok(leaf) => {
                            request.continue_leaf_fetch::<H>(Some(leaf));
                            continue;
                        }
                        Err(leaf_load) => {
                            request.note_io();

                            vacant_entry.insert(vec![request_index]);
                            assert_eq!(slab_index, self.io_slab.insert(IoRequest::Leaf(leaf_load)));
                            return Ok(());
                        }
                    }
                }
            }
        }

        Ok(())
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
            IoRequest::Merkle(merkle_load) => {
                // UNWRAP: page loader always submits a `Read` command that yields a fat page.
                let page = io.command.kind.unwrap_buf();
                match merkle_load.try_complete(page) {
                    Some(p) => self.handle_merkle_page_and_continue(read_pass, slab_index, Some(p)),
                    None => self.idle_page_loads.push_back(slab_index),
                };
            }
            IoRequest::Leaf(_) => {
                // UNWRAP: read transaction always submits a `Read` command that yields a fat page.
                self.handle_leaf_page_and_continue(slab_index, io.command.kind.unwrap_buf());
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

            request.continue_seek::<H>(
                read_pass,
                &self.beatree_read_transaction,
                page_load.page_id().clone(),
                &page,
                self.record_siblings,
            );

            if !request.is_completed() {
                self.idle_requests.push_back(waiting_request);
            }
        }
    }

    fn handle_leaf_page_and_continue(&mut self, slab_index: usize, page: FatPage) {
        let IoRequest::Leaf(leaf_load) = self.io_slab.remove(slab_index) else {
            panic!()
        };

        let page_number = leaf_load.page_number();
        let leaf = leaf_load.finish(page);
        for waiting_request in self
            .io_waiters
            .remove(&IoQuery::LeafPage(page_number))
            .into_iter()
            .flatten()
        {
            if waiting_request < self.processed {
                continue;
            }
            let idx = waiting_request - self.processed;
            let request = &mut self.requests[idx];
            assert!(!request.is_completed());

            request.continue_leaf_fetch::<H>(Some(leaf.clone()));
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

#[cfg(test)]
mod tests {
    use bitvec::prelude::*;
    use nomt_core::trie::KeyPath;

    use crate::merkle::seek::range_bounds;

    #[test]
    fn key_range() {
        fn make_path(path: BitVec<u8, Msb0>) -> KeyPath {
            let mut k = KeyPath::default();
            k.view_bits_mut::<Msb0>()[..path.len()].copy_from_bitslice(&path);
            k
        }

        let key_a = make_path(bitvec![u8, Msb0; 0, 0, 0, 0]);
        assert_eq!(range_bounds(key_a, 0).1, None);
        assert_eq!(
            range_bounds(key_a, 1).1,
            Some(make_path(bitvec![u8, Msb0; 1]))
        );
        assert_eq!(
            range_bounds(key_a, 2).1,
            Some(make_path(bitvec![u8, Msb0; 0, 1]))
        );
        assert_eq!(
            range_bounds(key_a, 3).1,
            Some(make_path(bitvec![u8, Msb0; 0, 0, 1]))
        );
        assert_eq!(
            range_bounds(key_a, 4).1,
            Some(make_path(bitvec![u8, Msb0; 0, 0, 0, 1]))
        );

        let key_b = make_path(bitvec![u8, Msb0; 1, 1, 1, 1]);
        assert_eq!(range_bounds(key_b, 0).1, None);
        assert_eq!(range_bounds(key_b, 1).1, None);
        assert_eq!(range_bounds(key_b, 2).1, None);
        assert_eq!(range_bounds(key_b, 3).1, None);
        assert_eq!(range_bounds(key_b, 4).1, None);

        let key_c = make_path(bitvec![u8, Msb0; 0, 1, 0]);
        assert_eq!(
            range_bounds(key_c, 3).1,
            Some(make_path(bitvec![u8, Msb0; 0, 1, 1]))
        );
        let key_d = make_path(bitvec![u8, Msb0; 0, 1, 0, 1]);
        assert_eq!(
            range_bounds(key_d, 4).1,
            Some(make_path(bitvec![u8, Msb0; 0, 1, 1, 0]))
        );
    }
}
