//! Left-to-right walking and updating the page tree.
//!
//! The core usage is to create a [`PageWalker`] and make repeated called to `advance`,
//! `advance_and_replace`, and `advance_and_place_node`, followed by a single call to `conclude`.
//!
//! The [`PageWalker`], upon concluding, causes the same effect to the trie as a series of
//! standalone put and delete operations, but with a minimal amount of hashing and revisiting of
//! nodes.
//!
//! ## Multi-Update
//!
//! `advance_and_replace` is based off of the observation that a set of put and delete operations
//! can be partitioned into groups based on which terminal node their keys currently look up to.
//! Each terminal node is then replaced with the sub-trie resulting from the set of given updates,
//! and the trie is compacted into its smallest possible form, and hashed.
//!
//! For example,
//!   - Replacing a single leaf node with another leaf node in the case of the previous leaf
//!     being deleted and a new one with the same key or at least key prefix being put.
//!   - Replacing a single leaf node with a terminator, in the case of deleting the leaf which was
//!     there prior.
//!   - Replacing a terminator with a leaf, in the case of a single put operation with that prefix
//!   - Replacing a leaf node or terminator with a larger sub-trie in the case of multiple puts for
//!     keys beginning with that prefix, possibly preserving the initial leaf.
//!
//! We refer to this as sub-trie replacement.
//!
//! Any newly created terminator nodes must be "compacted" upwards as long as their sibling is a
//! terminator or a leaf node to create the most tractable representation. We combine this operation
//! with hashing up towards the root, described in the following paragraph.
//!
//! Any changes in the trie must be reflected in the hashes of the nodes above them, all the way
//! up to the root. When we replace a terminal node with a new sub-trie, we apply the compaction
//! and hashing operations up to the point where no subsequently altered terminal will affect its
//! result. The last terminal finishes hashing to the root. We refer to this as partial compaction.
//!
//! ## Partial Update
//!
//! The PageWalker can also perform a partial update of the trie. By providing a parent page in
//! [`PageWalker::new`], you can restrict the operation only to trie positions which land in pages
//! below the parent. In this mode, the changes which _would_ have been made to the parent page
//! are recorded as part of the output. This is useful for splitting the work of updating pages
//! across multiple threads.

// TODO: comment all fucking UNWRAP

use bitvec::prelude::*;
use nomt_core::{
    hasher::NodeHasher,
    page::DEPTH,
    page_id::{ChildPageIndex, PageId, ROOT_PAGE_ID},
    trie::{self, KeyPath, Node, NodeKind, ValueHash, TERMINATOR},
    trie_pos::TriePosition,
    update::WriteNode,
};

use crate::{
    merkle::BucketInfo,
    page_cache::{Page, PageMut},
    page_diff::PageDiff,
};

/// Threshold representing the number of leaves required to be present in the two
/// subtrees contained in a page to be stored on disk.
/// If this threshold is not exceeded, the page will not be stored on disk
/// and will be constructed on the fly when needed.
const PAGE_ELISION_THRESHOLD: u64 = 10;

/// The output of the page walker.
pub enum Output {
    /// A new root node.
    ///
    /// This is always the output when no parent page is supplied to the walker.
    Root(Node, Vec<UpdatedPage>),
    /// Nodes to set in the bottom layer of the parent page, indexed by the position of the node
    /// to set.
    ///
    /// This is always the output when a parent page is supplied to the walker.
    ChildPageRoots(Vec<(TriePosition, Node)>, Vec<UpdatedPage>),
}

/// A page which was updated during the course of modifying the trie.
pub struct UpdatedPage {
    /// The ID of the page.
    pub page_id: PageId,
    /// An owned copy of the page, including the modified nodes.
    pub page: PageMut,
    /// A compact diff indicating all modified slots in the page.
    pub diff: PageDiff,
    /// The bucket info associated with the page.
    pub bucket_info: BucketInfo,
}

/// A page which currenlty stays in the stack of the page walker.
/// Contains not only data related to the page itself but also all
/// info related to respect the [`PAGE_ELISION_THRESHOLD`].
struct StackPage {
    /// Wether the page is fresh.
    fresh: bool,
    /// The page this stack item is workin on.
    updated_page: UpdatedPage,
    /// If Some contains a counter of all the leaves present in the two subtrees
    /// of the page. If the counter exceeds [`PAGE_ELISION_THRESHOLD`], we don't
    /// need to keep it updated, and it becomes None.
    leaves_counter: Option<u64>,
    /// Bitfield used to keep track of which childs pages have been elided.
    /// None if no child page has been elided.
    elided_childs: ElidedChilds,
}

impl StackPage {
    fn new(updated_page: UpdatedPage, fresh: bool) -> Self {
        Self {
            fresh,
            elided_childs: ElidedChilds::new(updated_page.page.elided_childs()),
            leaves_counter: Some(0),
            updated_page,
        }
    }
}

/// Bitfield used to note which child pages are elided, thus require
/// on the fly reconstruction.
pub struct ElidedChilds {
    elided: u64,
}

impl ElidedChilds {
    fn new(elided: u64) -> Self {
        Self { elided }
    }

    /// Note a child at `child_index` as elided.
    ///
    /// Panics if `child_index` is bigger than [`nomt_core::page_id::MAX_CHILD_INDEX`].
    fn elide(&mut self, child_page_index: ChildPageIndex) {
        self.elided |= 1 << (child_page_index.to_u8() as u64);
    }

    /// Note a child at `child_index` as not elided.
    ///
    /// Panics if `child_index` is bigger than [`nomt_core::page_id::MAX_CHILD_INDEX`].
    fn not_elide(&mut self, child_page_index: ChildPageIndex) {
        self.elided &= !(1 << (child_page_index.to_u8() as u64));
    }

    /// Checks if the child at `child_index` is elided.
    ///
    /// Panics if `child_index` is bigger than [`nomt_core::page_id::MAX_CHILD_INDEX`].
    fn is_elided(&self, child_page_index: ChildPageIndex) -> bool {
        (self.elided >> child_page_index.to_u8() as u64) & 1 == 1
    }
}

/// A set of pages that the page walker draws upon.
pub trait PageSet {
    /// Get a page from the set. `None` if it isn't exist.
    fn get(&self, page_id: &PageId) -> Option<(Page, BucketInfo)>;
    /// Create a new fresh page.
    fn fresh(&self, page_id: &PageId) -> (PageMut, BucketInfo);
}

/// Left-to-right updating walker over the page tree.
pub struct PageWalker<H> {
    // last position `advance` was invoked with.
    last_position: Option<TriePosition>,
    // actual position
    position: TriePosition,
    parent_page: Option<PageId>,
    child_page_roots: Vec<(TriePosition, Node)>,
    root: Node,
    updated_pages: Vec<UpdatedPage>,

    // the stack contains pages (ascending) which are descendants of the parent page, if any.
    stack: Vec<StackPage>,

    // the sibling stack contains the previous node values of siblings on the path to the current
    // position, annotated with their depths.
    sibling_stack: Vec<(Node, usize)>,
    prev_node: Option<Node>, // the node at `self.position` which was replaced in a previous call

    metrics: crate::metrics::Metrics,
    _marker: std::marker::PhantomData<H>,
}

impl<H: NodeHasher> PageWalker<H> {
    /// Create a new [`PageWalker`], with an optional parent page for constraining operations
    /// to a subsection of the page tree.
    pub fn new(root: Node, parent_page: Option<PageId>, metrics: crate::metrics::Metrics) -> Self {
        PageWalker {
            last_position: None,
            position: TriePosition::new(),
            parent_page,
            child_page_roots: Vec::new(),
            root,
            updated_pages: Vec::new(),
            stack: Vec::new(),
            sibling_stack: Vec::new(),
            prev_node: None,
            metrics,
            _marker: std::marker::PhantomData,
        }
    }

    /// Advance to a given trie position and replace the terminal node there with a trie
    /// based on the provided key-value pairs.
    ///
    /// The key-value pairs should be sorted and should all be suffixes of the given position.
    ///
    /// An empty vector deletes any existing terminal node.
    ///
    /// # Panics
    ///
    /// Panics if the current trie position is not a terminal node.
    ///
    /// Panics if this falls in a page which is not a descendant of the parent page, if any.
    /// Panics if this is not greater than the previous trie position.
    pub fn advance_and_replace(
        &mut self,
        page_set: &impl PageSet,
        new_pos: TriePosition,
        ops: impl IntoIterator<Item = (KeyPath, ValueHash)>,
    ) {
        if let Some(ref pos) = self.last_position {
            assert!(new_pos.path() > pos.path());
            self.compact_up(Some(new_pos.clone()));
        }
        self.last_position = Some(new_pos.clone());
        self.build_stack(page_set, new_pos);

        self.replace_terminal(page_set, ops);
    }

    /// Advance to a given trie position and place the given node at that position.
    ///
    /// It is the responsibility of the user to ensure that:
    ///   - if this is an internal node, the two child positions hashed together create this node.
    ///   - if this is a terminal node, then nothing exists in the two child positions.
    ///
    /// The expected usage of this function is to be called with the values of
    /// `Output::ChildPageRoots`.
    ///
    /// # Panics
    ///
    /// Panics if the current trie position is not a terminal node.
    ///
    /// Panics if this falls in a page which is not a descendant of the parent page, if any.
    /// Panics if this is not greater than the previous trie position.
    pub fn advance_and_place_node(
        &mut self,
        page_set: &impl PageSet,
        new_pos: TriePosition,
        node: Node,
    ) {
        if let Some(ref pos) = self.last_position {
            assert!(new_pos.path() > pos.path());
            self.compact_up(Some(new_pos.clone()));
        }
        self.last_position = Some(new_pos.clone());
        self.build_stack(page_set, new_pos);
        self.place_node(node);
    }

    /// Advance to a given trie position without updating.
    ///
    /// # Panics
    ///
    /// Panics if this falls in a page which is not a descendant of the parent page, if any.
    /// Panics if this is not greater than the previous trie position.
    pub fn advance(&mut self, new_pos: TriePosition) {
        if let Some(ref pos) = self.last_position {
            assert!(new_pos.path() > pos.path());
            self.compact_up(Some(new_pos.clone()));
        }

        let page_id = new_pos.page_id();
        self.assert_page_in_scope(page_id.as_ref());
        self.last_position = Some(new_pos);
    }

    fn place_node(&mut self, node: Node) {
        if self.position.is_root() {
            self.prev_node = Some(self.root);
            self.root = node;
        } else {
            self.prev_node = Some(self.node());
            self.set_node(node);
        }
    }

    fn replace_terminal(
        &mut self,
        page_set: &impl PageSet,
        ops: impl IntoIterator<Item = (KeyPath, ValueHash)>,
    ) {
        let node = if self.position.is_root() {
            self.root
        } else {
            self.node()
        };

        self.prev_node = Some(node);

        assert!(!trie::is_internal::<H>(&node));

        let start_position = self.position.clone();

        // replace sub-trie at the given position
        nomt_core::update::build_trie::<H>(self.position.depth() as usize, ops, |control| {
            let node = control.node();
            let up = control.up();
            let mut down = control.down();

            if let WriteNode::Internal {
                ref internal_data, ..
            } = control
            {
                // we assume pages are not necessarily zeroed. therefore, there might be
                // some garbage in the sibling slot we need to clear out.
                let zero_sibling = if self.position.peek_last_bit() {
                    trie::is_terminator::<H>(&internal_data.left)
                } else {
                    trie::is_terminator::<H>(&internal_data.right)
                };

                if zero_sibling {
                    self.set_sibling(trie::TERMINATOR);
                }
            };

            // avoid popping pages off the stack if we are jumping to a sibling.
            if up && !down.is_empty() {
                if down[0] == !self.position.peek_last_bit() {
                    // UNWRAP: checked above
                    self.position.sibling();
                    down = &down[1..];
                } else {
                    self.up();
                }
            } else if up {
                self.up()
            }

            let fresh = self.position.depth() > start_position.depth();

            if !fresh && !down.is_empty() {
                // first bit is only fresh if we are at the start position and the start is at the
                // end of its page (or at the root). after that, definitely is.
                self.down(
                    page_set,
                    &down[..1],
                    self.position.depth_in_page() == DEPTH || self.position.is_root(),
                );
                self.down(page_set, &down[1..], true);
            } else {
                self.down(page_set, &down, true);
            }

            if self.position.is_root() {
                self.root = node;
            } else {
                self.set_node(node);
            }
        });

        // build_trie should always return us to the original position.
        if !self.position.is_root() {
            assert_eq!(
                self.stack.last().unwrap().updated_page.page_id,
                self.position.page_id().unwrap()
            );
        } else {
            assert!(self.stack.is_empty());
        }
    }

    // move the current position up.
    fn up(&mut self) {
        if self.position.depth_in_page() == 1 {
            // UNWRAP: we never move up beyond the root / parent page.
            let stack_top = self.stack.pop().unwrap();
            handle_elision_threshold::<H>(
                stack_top,
                &mut self.stack,
                &mut self.updated_pages,
                &mut self.metrics,
            );
        }
        self.position.up(1);
    }

    // move the current position down, hinting whether the location is guaranteed to be fresh.
    fn down(&mut self, page_set: &impl PageSet, bit_path: &BitSlice<u8, Msb0>, fresh: bool) {
        for bit in bit_path.iter().by_vals() {
            if self.position.is_root() {
                let (page, bucket_info) = if fresh {
                    page_set.fresh(&ROOT_PAGE_ID)
                } else {
                    // UNWRAP: all pages on the path to the node should be in the cache.
                    page_set
                        .get(&ROOT_PAGE_ID)
                        .map(|(p, b)| (p.deep_copy(), b))
                        .unwrap()
                };

                self.stack.push(StackPage::new(
                    UpdatedPage {
                        page_id: ROOT_PAGE_ID,
                        page,
                        diff: PageDiff::default(),
                        bucket_info,
                    },
                    fresh,
                ));
            } else if self.position.depth_in_page() == DEPTH {
                // UNWRAP: the only legal positions are below the "parent" (root or parent_page)
                //         and stack always contains all pages to position.
                let parent_stack_page = &self.stack.last().unwrap();
                let child_page_index = self.position.child_page_index();

                // UNWRAP: we never overflow the page stack.
                let child_page_id = parent_stack_page
                    .updated_page
                    .page_id
                    .child_page_id(child_page_index.clone())
                    .unwrap();

                // Make sure the page was not elided, if so reconstruct it, otherwise fetch it.
                // TODO: use properly the bitfield to reconstruct the page.
                // TODO: probably it makes sense to fetch it once and carry it around, maybe in UpdatedPage,
                let _is_elided_child =
                    dbg!(parent_stack_page.elided_childs.is_elided(child_page_index));

                let updated_page = if fresh {
                    let (page, bucket_info) = page_set.fresh(&child_page_id);
                    UpdatedPage {
                        page_id: child_page_id,
                        page,
                        diff: PageDiff::default(),
                        bucket_info,
                    }
                } else {
                    // UNWRAP: all pages on the path to the node should be in the cache.
                    let (page, bucket_info) = page_set.get(&child_page_id).unwrap();
                    UpdatedPage {
                        page_id: child_page_id,
                        page: page.deep_copy(),
                        diff: PageDiff::default(),
                        bucket_info,
                    }
                };

                self.stack.push(StackPage::new(updated_page, fresh));
            }
            self.position.down(bit);
        }
    }

    /// Get the previous values of any siblings on the path to the current node, along with their depth.
    pub fn siblings(&self) -> &[(Node, usize)] {
        &self.sibling_stack
    }

    /// Conclude walking and updating and return an output - either a new root, or a list
    /// of node changes to apply to the parent page.
    pub fn conclude(mut self) -> Output {
        self.compact_up(None);
        if self.parent_page.is_none() {
            Output::Root(self.root, self.updated_pages)
        } else {
            Output::ChildPageRoots(self.child_page_roots, self.updated_pages)
        }
    }

    fn compact_up(&mut self, target_pos: Option<TriePosition>) {
        // This serves as a check to see if we have anything to compact.
        if self.stack.is_empty() {
            return;
        }

        let compact_layers = if let Some(target_pos) = target_pos {
            let current_depth = self.position.depth() as usize;
            let shared_depth = self.position.shared_depth(&target_pos);

            // prune all siblings after shared depth. this function will push one more pending
            // sibling at `shared_depth + 1`.
            let keep_sibling_depth = shared_depth;
            let keep_sibling_len = self
                .sibling_stack
                .iter()
                .take_while(|s| s.1 <= keep_sibling_depth)
                .count();
            self.sibling_stack.truncate(keep_sibling_len);

            // shared_depth is guaranteed less than current_depth because the full prefix isn't
            // shared.
            // we want to compact up (inclusive) to the depth `shared_depth + 1`
            let compact_layers = current_depth - (shared_depth + 1);

            if compact_layers == 0 {
                if let Some(prev_node) = self.prev_node.take() {
                    self.sibling_stack.push((prev_node, current_depth));
                }
            } else {
                self.prev_node = None;
            }

            compact_layers
        } else {
            self.sibling_stack.clear();
            self.position.depth() as usize
        };

        for i in 0..compact_layers {
            let next_node = self.compact_step();
            self.up();

            if self.stack.is_empty() {
                if self.parent_page.is_none() {
                    self.root = next_node;
                } else {
                    // though there are more layers to compact, we are all done. track the node
                    // to place into the parent page and stop compacting.
                    self.child_page_roots
                        .push((self.position.clone(), next_node));
                }

                break;
            } else {
                // save the final relevant sibling.
                if i == compact_layers - 1 {
                    self.sibling_stack
                        .push((self.node(), self.position.depth() as usize));
                }

                self.set_node(next_node);
            }
        }
    }

    fn compact_step(&mut self) -> Node {
        let node = self.node();
        let sibling = self.sibling_node();
        let bit = self.position.peek_last_bit();

        match (NodeKind::of::<H>(&node), NodeKind::of::<H>(&sibling)) {
            (NodeKind::Terminator, NodeKind::Terminator) => {
                // compact terminators.
                trie::TERMINATOR
            }
            (NodeKind::Leaf, NodeKind::Terminator) => {
                // compact: clear this node, move leaf up.
                self.set_node(trie::TERMINATOR);

                node
            }
            (NodeKind::Terminator, NodeKind::Leaf) => {
                // compact: clear sibling node, move leaf up.
                self.position.sibling();
                self.set_node(trie::TERMINATOR);

                sibling
            }
            _ => {
                // otherwise, internal
                let node_data = if bit {
                    trie::InternalData {
                        left: sibling,
                        right: node,
                    }
                } else {
                    trie::InternalData {
                        left: node,
                        right: sibling,
                    }
                };

                H::hash_internal(&node_data)
            }
        }
    }

    // read the node at the current position. panics if no current page.
    fn node(&self) -> Node {
        let node_index = self.position.node_index();
        let stack_top = self.stack.last().unwrap();
        stack_top.updated_page.page.node(node_index)
    }

    // read the sibling node at the current position. panics if no current page.
    fn sibling_node(&self) -> Node {
        let node_index = self.position.sibling_index();
        let stack_top = self.stack.last().unwrap();
        stack_top.updated_page.page.node(node_index)
    }

    // set a node in the current page at the given index. panics if no current page.
    fn set_node(&mut self, node: Node) {
        let node_index = self.position.node_index();
        let sibling_node = self.sibling_node();

        // UNWRAP: if a node is being set, then a page in the stack must be present
        let stack_top = self.stack.last_mut().unwrap();
        stack_top.updated_page.page.set_node(node_index, node);

        if self.position.is_first_layer_in_page()
            && node == TERMINATOR
            && sibling_node == TERMINATOR
        {
            stack_top.updated_page.diff.set_cleared();
        } else {
            stack_top.updated_page.diff.set_changed(node_index);
        }
    }

    // set the sibling node in the current page at the given index. panics if no current page.
    fn set_sibling(&mut self, node: Node) {
        let node_index = self.position.sibling_index();
        let stack_top = self.stack.last_mut().unwrap();
        stack_top.updated_page.page.set_node(node_index, node);

        stack_top.updated_page.diff.set_changed(node_index);
    }

    fn assert_page_in_scope(&self, page_id: Option<&PageId>) {
        match page_id {
            Some(page_id) => {
                if let Some(ref parent_page) = self.parent_page {
                    assert!(&page_id != &parent_page);
                    assert!(page_id.is_descendant_of(&parent_page));
                }
            }
            None => assert!(self.parent_page.is_none()),
        }
    }

    // Build the stack to target a particular position.
    //
    // Precondition: the stack is either empty or contains an ancestor of the page ID the position
    // lands in.
    fn build_stack(&mut self, page_set: &impl PageSet, position: TriePosition) {
        let new_page_id = position.page_id();
        self.assert_page_in_scope(new_page_id.as_ref());

        self.position = position;
        let Some(page_id) = new_page_id else {
            while let Some(stack_top) = self.stack.pop() {
                handle_elision_threshold::<H>(
                    stack_top,
                    &mut self.stack,
                    &mut self.updated_pages,
                    &mut self.metrics,
                );
            }
            return;
        };

        // push all pages from the given page down to (not including) the target onto the stack.
        // target is either:
        //   - last item in stack (guaranteed ancestor)
        //   - the over-arching parent page (if any)
        //   - or `None`, if we need to push the root page as well.
        let target = self
            .stack
            .last()
            .map(|item| item.updated_page.page_id.clone())
            .or(self.parent_page.as_ref().map(|p| p.clone()));

        let start_len = self.stack.len();
        let mut cur_ancestor = page_id;
        let mut push_count = 0;
        while Some(&cur_ancestor) != target.as_ref() {
            // UNWRAP: all pages on the path to the terminal are present in the page set.

            // TODO:  Not pushing elided pages makes this unwrap to possibly return None
            // because the page is *not* in memory.
            // There are two ways of solving this, not having the function build stack
            // and only lazily load pages while traversing the trie.
            // Doing it here anyway and just load the page if present, if not it will be
            // created on the fly later when it's time to traverse it.
            if let Some((page, bucket_info)) = page_set.get(&cur_ancestor) {
                self.stack.push(StackPage::new(
                    UpdatedPage {
                        page_id: cur_ancestor.clone(),
                        page: page.deep_copy(),
                        diff: PageDiff::default(),
                        bucket_info,
                    },
                    false, /* fresh */
                ));
                push_count += 1;
            }

            // stop pushing once we reach the root page.
            if cur_ancestor == ROOT_PAGE_ID {
                break;
            }
            cur_ancestor = cur_ancestor.parent_page_id();
        }

        // we pushed onto the stack in descending, so now reverse everything we just pushed to
        // make it ascending.
        self.stack[start_len..start_len + push_count].reverse();
    }
}

/// Count the number of leaves present *only* in the provided page,
/// without jumping into child pages.
fn count_leaves<H: NodeHasher>(page: &PageMut, print: bool) -> u64 {
    // A simplier linear scan cannot be done becase the page could contain some gargabe.
    let mut counter = 0;

    // TODO: this is a little trick to use TriePosition to go through the page.
    // I could use simply direclty the node_index.
    let mut pos = TriePosition::new();
    let initial_depth = pos.depth();
    pos.down(false);

    loop {
        if print {
            println!("pos: {}", pos.node_index());
        }
        let node = page.node(pos.node_index());
        // Continue to traverse the left child if the current node is internal,
        // stop if we reach the end of the page.
        if trie::is_internal::<H>(&node) && pos.depth_in_page() != DEPTH {
            pos.down(false);
            continue;
        }

        if trie::is_leaf::<H>(&node) {
            if print {
                println!("is leaf pos: {}", pos.node_index());
            }
            counter += 1;
        }

        // Going up until I reach a leaf sibling or the node I started from.
        while pos.depth() != initial_depth && pos.peek_last_bit() {
            pos.up(1);
        }

        if pos.depth() == initial_depth {
            break;
        }

        pos.sibling();
    }
    counter
}

fn handle_elision_threshold<H: NodeHasher>(
    mut stack_page: StackPage,
    stack: &mut Vec<StackPage>,
    updated_pages: &mut Vec<UpdatedPage>,
    metrics: &mut crate::metrics::Metrics,
) {
    stack_page
        .updated_page
        .page
        .set_elided_childs(stack_page.elided_childs.elided);

    if let Some(leaves_counter) = stack_page.leaves_counter {
        let n_leaves = count_leaves::<H>(&stack_page.updated_page.page, false);

        if stack.is_empty() {
            // If we pop the last page, even if it has less then the threshold,
            // it does not get elided. This will almost never happen but simplifies a lot the code.
            //println!("Pushind last page in stack");

            metrics.count(crate::metrics::Metric::TotalHashTablePages);
            updated_pages.push(stack_page.updated_page);

            return;
        } else if n_leaves + leaves_counter < PAGE_ELISION_THRESHOLD {
            //println!("page with less than PAGE_ELISION_THRESHOLD");
            metrics.count(crate::metrics::Metric::ElidedHashTablePages);
            let parent_stack_page = stack.last_mut().unwrap();
            if let Some(ref mut upper_leaves_in_childs) = parent_stack_page.leaves_counter {
                *upper_leaves_in_childs += n_leaves + leaves_counter;
            }

            // Elide current page from parent page.
            let page_id = &stack_page.updated_page.page_id;
            // This will never underflow because page_id.depth() would be 0
            // only if page_id is the root and it cannot happen because the stack
            // would have been empty if the last stack item pop was the root.
            let child_index = page_id.child_index_at_level(page_id.depth() - 1);
            parent_stack_page.elided_childs.elide(child_index);

            if !stack_page.fresh {
                // if the page was not fresh fresh to elided the page we need to set it as cleared.
                // TODO: uncomment this once elided page reconstruction is ready.
                //stack_page.updated_page.diff.set_cleared();
            }

            // if the page was fresh to elided the page it just doesn't get pushed to updated_pages

            // TODO: remove this push once elided page reconstruction is ready.
            metrics.count(crate::metrics::Metric::TotalHashTablePages);
            updated_pages.push(stack_page.updated_page);
            return;
        }
    }

    if !stack.is_empty() {
        // propagates the exceeded page elision threshold
        let parent_stack_page = stack.last_mut().unwrap();
        parent_stack_page.leaves_counter = None;

        // Toggle as not elide current page from parent page.
        let page_id = &stack_page.updated_page.page_id;
        // This will never underflow because page_id.depth() would be 0
        // only if page_id is the root and it cannot happen because the stack
        // would have been empty if the last stack item pop was the root.
        let child_index = page_id.child_index_at_level(page_id.depth() - 1);

        parent_stack_page.elided_childs.not_elide(child_index);
    }

    //println!("page with more than PAGE_ELISION_THRESHOLD");
    metrics.count(crate::metrics::Metric::TotalHashTablePages);
    updated_pages.push(stack_page.updated_page);
}

#[cfg(test)]
mod tests {
    use super::{
        trie, BucketInfo, Node, NodeHasher, Output, PageSet, PageWalker, TriePosition, UpdatedPage,
        ROOT_PAGE_ID,
    };
    use crate::{
        hasher::Blake3Hasher,
        io::PagePool,
        metrics::Metrics,
        page_cache::{Page, PageMut},
        page_diff::PageDiff,
    };
    use bitvec::prelude::*;
    use imbl::HashMap;
    use nomt_core::page_id::{ChildPageIndex, PageId, PageIdsIterator};

    macro_rules! trie_pos {
        ($($t:tt)+) => {
            TriePosition::from_bitslice(bits![u8, Msb0; $($t)+])
        }
    }

    macro_rules! key_path {
        ($($t:tt)+) => {{
            let mut path = [0u8; 32];
            let slice = bits![u8, Msb0; $($t)+];
            path.view_bits_mut::<Msb0>()[..slice.len()].copy_from_bitslice(&slice);
            path
        }}
    }

    fn val(i: u8) -> [u8; 32] {
        [i; 32]
    }

    struct MockPageSet {
        page_pool: PagePool,
        inner: HashMap<PageId, Page>,
    }

    impl Default for MockPageSet {
        fn default() -> Self {
            let page_pool = PagePool::new();
            let mut inner = HashMap::new();
            inner.insert(
                ROOT_PAGE_ID,
                PageMut::pristine_empty(&page_pool, &ROOT_PAGE_ID).freeze(),
            );
            MockPageSet { page_pool, inner }
        }
    }

    impl MockPageSet {
        fn apply(&mut self, updates: Vec<UpdatedPage>) {
            for page in updates {
                self.inner.insert(page.page_id, page.page.freeze());
            }
        }
    }

    impl PageSet for MockPageSet {
        fn fresh(&self, page_id: &PageId) -> (PageMut, BucketInfo) {
            let page = PageMut::pristine_empty(&self.page_pool, page_id);
            (page, BucketInfo::Fresh)
        }

        fn get(&self, page_id: &PageId) -> Option<(Page, BucketInfo)> {
            self.inner
                .get(page_id)
                .map(|p| (p.clone(), BucketInfo::Fresh))
        }
    }

    #[test]
    #[should_panic]
    fn advance_backwards_panics() {
        let root = trie::TERMINATOR;
        let mut walker = PageWalker::<Blake3Hasher>::new(root, None, Metrics::new(false));

        let trie_pos_a = trie_pos![1];
        let trie_pos_b = trie_pos![0];
        walker.advance(trie_pos_a);
        walker.advance(trie_pos_b);
    }

    #[test]
    #[should_panic]
    fn advance_same_panics() {
        let root = trie::TERMINATOR;
        let mut walker = PageWalker::<Blake3Hasher>::new(root, None, Metrics::new(false));
        let trie_pos_a = trie_pos![0];
        walker.advance(trie_pos_a.clone());
        walker.advance(trie_pos_a);
    }

    #[test]
    #[should_panic]
    fn advance_to_parent_page_panics() {
        let root = trie::TERMINATOR;
        let mut walker =
            PageWalker::<Blake3Hasher>::new(root, Some(ROOT_PAGE_ID), Metrics::new(false));
        let trie_pos_a = trie_pos![0, 0, 0, 0, 0, 0];
        walker.advance(trie_pos_a);
    }

    #[test]
    #[should_panic]
    fn advance_to_root_with_parent_page_panics() {
        let root = trie::TERMINATOR;
        let mut walker =
            PageWalker::<Blake3Hasher>::new(root, Some(ROOT_PAGE_ID), Metrics::new(false));
        walker.advance(TriePosition::new());
    }

    #[test]
    fn compacts_and_updates_root_single_page() {
        let root = trie::TERMINATOR;
        let page_set = MockPageSet::default();

        let mut walker = PageWalker::<Blake3Hasher>::new(root, None, Metrics::new(false));
        let trie_pos_a = trie_pos![0, 0];
        walker.advance_and_replace(
            &page_set,
            trie_pos_a,
            vec![
                (key_path![0, 0, 1, 0], val(1)),
                (key_path![0, 0, 1, 1], val(2)),
            ],
        );

        let trie_pos_b = trie_pos![0, 1];
        walker.advance(trie_pos_b);

        let trie_pos_c = trie_pos![1];
        walker.advance_and_replace(
            &page_set,
            trie_pos_c,
            vec![(key_path![1, 0], val(3)), (key_path![1, 1], val(4))],
        );

        match walker.conclude() {
            Output::Root(new_root, diffs) => {
                assert_eq!(
                    new_root,
                    nomt_core::update::build_trie::<Blake3Hasher>(
                        0,
                        vec![
                            (key_path![0, 0, 1, 0], val(1)),
                            (key_path![0, 0, 1, 1], val(2)),
                            (key_path![1, 0], val(3)),
                            (key_path![1, 1], val(4))
                        ],
                        |_| {}
                    )
                );
                assert_eq!(diffs.len(), 1);
                assert_eq!(&diffs[0].page_id, &ROOT_PAGE_ID);
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn compacts_and_updates_root_multiple_pages() {
        let root = trie::TERMINATOR;
        let mut page_set = MockPageSet::default();

        let mut walker = PageWalker::<Blake3Hasher>::new(root, None, Metrics::new(false));

        walker.advance_and_replace(
            &page_set,
            TriePosition::new(),
            vec![
                (key_path![0, 1, 0, 1, 1, 0], val(1)),
                (key_path![0, 1, 0, 1, 1, 1], val(2)),
            ],
        );

        match walker.conclude() {
            Output::Root(_, updates) => {
                page_set.apply(updates);
            }
            _ => unreachable!(),
        }

        let mut walker = PageWalker::<Blake3Hasher>::new(root, None, Metrics::new(false));

        walker.advance_and_replace(
            &page_set,
            trie_pos![0, 1, 0, 1, 1, 0],
            vec![
                (key_path![0, 1, 0, 1, 1, 0], val(1)),
                (key_path![0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0], val(3)),
                (key_path![0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1], val(4)),
            ],
        );

        match walker.conclude() {
            Output::Root(new_root, updates) => {
                assert_eq!(
                    new_root,
                    nomt_core::update::build_trie::<Blake3Hasher>(
                        0,
                        vec![
                            (key_path![0, 1, 0, 1, 1, 0], val(1)),
                            (key_path![0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0], val(3)),
                            (key_path![0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1], val(4)),
                            (key_path![0, 1, 0, 1, 1, 1], val(2)),
                        ],
                        |_| {}
                    )
                );
                assert_eq!(updates.len(), 3);
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn multiple_pages_compacts_up_to_root() {
        let root = trie::TERMINATOR;
        let mut page_set = MockPageSet::default();

        let leaf_a_key_path = key_path![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let leaf_b_pos = trie_pos![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1];
        let leaf_b_key_path = key_path![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1];
        let leaf_c_pos = trie_pos![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0];
        let leaf_c_key_path = key_path![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0];
        let leaf_d_pos = trie_pos![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1];
        let leaf_d_key_path = key_path![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1];
        let leaf_e_pos = trie_pos![0, 1, 0, 1, 0, 1, 0, 0, 0, 0];
        let leaf_e_key_path = key_path![0, 1, 0, 1, 0, 1, 0, 0, 0, 0];
        let leaf_f_pos = trie_pos![0, 1, 0, 1, 0, 1, 0, 0, 0, 1];
        let leaf_f_key_path = key_path![0, 1, 0, 1, 0, 1, 0, 0, 0, 1];

        let mut walker = PageWalker::<Blake3Hasher>::new(root, None, Metrics::new(false));
        walker.advance_and_replace(
            &page_set,
            TriePosition::new(),
            vec![
                (leaf_a_key_path, val(1)),
                (leaf_b_key_path, val(2)),
                (leaf_c_key_path, val(3)),
                (leaf_d_key_path, val(4)),
                (leaf_e_key_path, val(5)),
                (leaf_f_key_path, val(6)),
            ],
        );

        let new_root = match walker.conclude() {
            Output::Root(new_root, diffs) => {
                page_set.apply(diffs);
                new_root
            }
            _ => unreachable!(),
        };

        let mut walker = PageWalker::<Blake3Hasher>::new(new_root, None, Metrics::new(false));

        walker.advance_and_replace(&page_set, leaf_b_pos, vec![]);
        walker.advance_and_replace(&page_set, leaf_c_pos, vec![]);
        walker.advance_and_replace(&page_set, leaf_d_pos, vec![]);
        walker.advance_and_replace(&page_set, leaf_e_pos, vec![]);
        walker.advance_and_replace(&page_set, leaf_f_pos, vec![]);

        match walker.conclude() {
            Output::Root(new_root, diffs) => {
                assert_eq!(
                    new_root,
                    nomt_core::update::build_trie::<Blake3Hasher>(
                        0,
                        vec![(leaf_a_key_path, val(1)),],
                        |_| {}
                    )
                );
                assert_eq!(diffs.len(), 7);
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn sets_child_page_roots() {
        let root = trie::TERMINATOR;
        let mut page_set = MockPageSet::default();

        let mut walker =
            PageWalker::<Blake3Hasher>::new(root, Some(ROOT_PAGE_ID), Metrics::new(false));
        let trie_pos_a = trie_pos![0, 0, 0, 0, 0, 0, 0];
        let trie_pos_b = trie_pos![0, 0, 0, 0, 0, 0, 1];
        let trie_pos_c = trie_pos![0, 0, 0, 0, 0, 1, 0];
        let trie_pos_d = trie_pos![0, 0, 0, 0, 0, 1, 1];
        let page_id_a = trie_pos![0, 0, 0, 0, 0, 0, 0].page_id().unwrap();
        let page_id_b = trie_pos![0, 0, 0, 0, 0, 1, 0].page_id().unwrap();
        page_set.inner.insert(
            page_id_a.clone(),
            PageMut::pristine_empty(&page_set.page_pool, &page_id_a).freeze(),
        );
        page_set.inner.insert(
            page_id_b.clone(),
            PageMut::pristine_empty(&page_set.page_pool, &page_id_b).freeze(),
        );

        walker.advance_and_replace(
            &page_set,
            trie_pos_a,
            vec![(key_path![0, 0, 0, 0, 0, 0, 0], val(1))],
        );

        walker.advance_and_replace(
            &page_set,
            trie_pos_b,
            vec![(key_path![0, 0, 0, 0, 0, 0, 1], val(2))],
        );

        walker.advance_and_replace(
            &page_set,
            trie_pos_c,
            vec![(key_path![0, 0, 0, 0, 0, 1, 0], val(3))],
        );

        walker.advance_and_replace(
            &page_set,
            trie_pos_d,
            vec![(key_path![0, 0, 0, 0, 0, 1, 1], val(4))],
        );

        match walker.conclude() {
            Output::ChildPageRoots(page_roots, diffs) => {
                assert_eq!(page_roots.len(), 2);
                assert_eq!(diffs.len(), 2);
                let left_page_id = ROOT_PAGE_ID
                    .child_page_id(ChildPageIndex::new(0).unwrap())
                    .unwrap();
                let right_page_id = ROOT_PAGE_ID
                    .child_page_id(ChildPageIndex::new(1).unwrap())
                    .unwrap();

                let diffed_ids = diffs.iter().map(|p| p.page_id.clone()).collect::<Vec<_>>();
                assert!(diffed_ids.contains(&left_page_id));
                assert!(diffed_ids.contains(&right_page_id));
                assert_eq!(page_roots[0].0, trie_pos![0, 0, 0, 0, 0, 0]);
                assert_eq!(page_roots[1].0, trie_pos![0, 0, 0, 0, 0, 1]);

                assert_eq!(
                    page_roots[0].1,
                    nomt_core::update::build_trie::<Blake3Hasher>(
                        6,
                        vec![
                            (key_path![0, 0, 0, 0, 0, 0, 0], val(1)),
                            (key_path![0, 0, 0, 0, 0, 0, 1], val(2)),
                        ],
                        |_| {}
                    )
                );

                assert_eq!(
                    page_roots[1].1,
                    nomt_core::update::build_trie::<Blake3Hasher>(
                        6,
                        vec![
                            (key_path![0, 0, 0, 0, 0, 1, 0], val(3)),
                            (key_path![0, 0, 0, 0, 0, 1, 1], val(4)),
                        ],
                        |_| {}
                    )
                );
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn tracks_sibling_prev_values() {
        let root = trie::TERMINATOR;
        let mut page_set = MockPageSet::default();

        let path_1 = key_path![0, 0, 0, 0];
        let path_2 = key_path![1, 0, 0, 0];
        let path_3 = key_path![1, 1, 0, 0];
        let path_4 = key_path![1, 1, 1, 0];
        let path_5 = key_path![1, 1, 1, 1];

        // first build a trie with these 5 key-value pairs. it happens to have the property that
        // all the "left" nodes are leaves.
        let root = {
            let mut walker = PageWalker::<Blake3Hasher>::new(root, None, Metrics::new(false));
            walker.advance_and_replace(
                &page_set,
                TriePosition::new(),
                vec![
                    (path_1, val(1)),
                    (path_2, val(2)),
                    (path_3, val(3)),
                    (path_4, val(4)),
                    (path_5, val(5)),
                ],
            );

            match walker.conclude() {
                Output::Root(new_root, diffs) => {
                    page_set.apply(diffs);
                    new_root
                }
                _ => unreachable!(),
            }
        };

        let mut walker = PageWalker::<Blake3Hasher>::new(root, None, Metrics::new(false));

        let node_hash = |key_path, val| {
            Blake3Hasher::hash_leaf(&trie::LeafData {
                key_path,
                value_hash: val,
            })
        };

        let expected_siblings = vec![
            (node_hash(path_1, val(1)), 1),
            (node_hash(path_2, val(2)), 2),
            (node_hash(path_3, val(3)), 3),
            (node_hash(path_4, val(4)), 4),
        ];

        // replace those leaf nodes one at a time.
        // the sibling stack will be populated as we go.

        walker.advance_and_replace(
            &page_set,
            TriePosition::from_path_and_depth(path_1, 4),
            vec![(path_1, val(11))],
        );
        assert_eq!(walker.siblings(), &expected_siblings[..0]);

        walker.advance_and_replace(
            &page_set,
            TriePosition::from_path_and_depth(path_2, 4),
            vec![(path_2, val(12))],
        );
        assert_eq!(walker.siblings(), &expected_siblings[..1]);

        walker.advance_and_replace(
            &page_set,
            TriePosition::from_path_and_depth(path_3, 4),
            vec![(path_3, val(13))],
        );
        assert_eq!(walker.siblings(), &expected_siblings[..2]);

        walker.advance_and_replace(
            &page_set,
            TriePosition::from_path_and_depth(path_4, 4),
            vec![(path_4, val(14))],
        );
        assert_eq!(walker.siblings(), &expected_siblings[..3]);

        walker.advance_and_replace(
            &page_set,
            TriePosition::from_path_and_depth(path_5, 4),
            vec![(path_5, val(15))],
        );
        assert_eq!(walker.siblings(), &expected_siblings[..4]);
    }

    #[test]
    fn internal_node_zeroes_sibling() {
        let root = trie::TERMINATOR;
        let mut page_set = MockPageSet::default();

        // this is going to create new leaves, with internal nodes going up to the root.
        let leaf_1 = key_path![0, 0, 0, 0, 0, 0, 0, 0];
        let leaf_2 = key_path![0, 0, 0, 0, 0, 0, 0, 1];

        let terminator_1 = TriePosition::from_path_and_depth(key_path![1], 1);
        let terminator_2 = TriePosition::from_path_and_depth(key_path![0, 1], 2);
        let terminator_3 = TriePosition::from_path_and_depth(key_path![0, 0, 1], 3);
        let terminator_4 = TriePosition::from_path_and_depth(key_path![0, 0, 0, 1], 4);
        let terminator_5 = TriePosition::from_path_and_depth(key_path![0, 0, 0, 0, 1], 5);
        let terminator_6 = TriePosition::from_path_and_depth(key_path![0, 0, 0, 0, 0, 1], 6);
        let terminator_7 = TriePosition::from_path_and_depth(key_path![0, 0, 0, 0, 0, 0, 1], 7);

        let mut root_page = PageMut::pristine_empty(&page_set.page_pool, &ROOT_PAGE_ID);
        let mut page1 =
            PageMut::pristine_empty(&page_set.page_pool, &terminator_7.page_id().unwrap());

        // we place garbage in all the sibling positions for those internal  nodes.
        {
            let garbage: Node = val(69);

            root_page.set_node(terminator_1.node_index(), garbage);
            root_page.set_node(terminator_2.node_index(), garbage);
            root_page.set_node(terminator_3.node_index(), garbage);
            root_page.set_node(terminator_4.node_index(), garbage);
            root_page.set_node(terminator_5.node_index(), garbage);
            root_page.set_node(terminator_6.node_index(), garbage);
            page1.set_node(terminator_7.node_index(), garbage);
        }

        page_set.inner.insert(ROOT_PAGE_ID, root_page.freeze());
        page_set
            .inner
            .insert(terminator_7.page_id().unwrap(), page1.freeze());

        let mut walker = PageWalker::<Blake3Hasher>::new(root, None, Metrics::new(false));

        walker.advance_and_replace(
            &page_set,
            TriePosition::new(),
            vec![(leaf_1, val(1)), (leaf_2, val(2))],
        );

        match walker.conclude() {
            Output::Root(_, diffs) => {
                page_set.apply(diffs);
            }
            _ => panic!(),
        }

        let root_page = page_set.inner.get(&ROOT_PAGE_ID).unwrap();
        let page1 = page_set
            .inner
            .get(&terminator_7.page_id().unwrap())
            .unwrap();

        // building the internal nodes must zero the garbage slots, now, anything reachable from the
        // root is consistent.
        {
            assert_eq!(root_page.node(terminator_1.node_index()), trie::TERMINATOR);
            assert_eq!(root_page.node(terminator_2.node_index()), trie::TERMINATOR);
            assert_eq!(root_page.node(terminator_3.node_index()), trie::TERMINATOR);
            assert_eq!(root_page.node(terminator_4.node_index()), trie::TERMINATOR);
            assert_eq!(root_page.node(terminator_5.node_index()), trie::TERMINATOR);
            assert_eq!(root_page.node(terminator_6.node_index()), trie::TERMINATOR);
            assert_eq!(page1.node(terminator_7.node_index()), trie::TERMINATOR);
        }
    }

    #[test]
    fn clear_bit_set_on_erased_page() {
        let root = trie::TERMINATOR;
        let mut page_set = MockPageSet::default();

        let leaf_a_key_path = key_path![0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0];
        let leaf_b_key_path = key_path![0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1];
        let leaf_c_key_path = key_path![0, 0, 1, 0, 0, 0, 0];

        let leaf_b_pos = trie_pos![0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1];
        let leaf_c_pos = trie_pos![0, 0, 1, 0, 0, 0, 0];

        let mut page_id_iter = PageIdsIterator::new(leaf_a_key_path);
        let root_page = page_id_iter.next().unwrap();
        let page_id_1 = page_id_iter.next().unwrap();
        let page_id_2 = page_id_iter.next().unwrap();

        let root = {
            let mut walker = PageWalker::<Blake3Hasher>::new(root, None, Metrics::new(false));
            walker.advance_and_replace(
                &page_set,
                TriePosition::new(),
                vec![
                    (leaf_a_key_path, val(1)),
                    (leaf_b_key_path, val(2)),
                    (leaf_c_key_path, val(3)),
                ],
            );

            match walker.conclude() {
                Output::Root(new_root, diffs) => {
                    page_set.apply(diffs);
                    new_root
                }
                _ => unreachable!(),
            }
        };

        let mut walker = PageWalker::<Blake3Hasher>::new(root, None, Metrics::new(false));

        // Remove leaf B. This should clear page 2.
        walker.advance_and_replace(&page_set, leaf_b_pos, vec![]);

        let root = match walker.conclude() {
            Output::Root(new_root, updates) => {
                let diffs: HashMap<PageId, PageDiff> = updates
                    .iter()
                    .map(|p| (p.page_id.clone(), p.diff.clone()))
                    .collect();
                assert!(diffs.get(&page_id_2).unwrap().cleared());
                page_set.apply(updates);
                new_root
            }
            _ => unreachable!(),
        };

        let mut walker = PageWalker::<Blake3Hasher>::new(root, None, Metrics::new(false));

        // Now removing leaf C will clear page 1 and the root page.
        walker.advance_and_replace(&page_set, leaf_c_pos, vec![]);

        match walker.conclude() {
            Output::Root(_new_root, updates) => {
                let diffs: HashMap<PageId, PageDiff> = updates
                    .iter()
                    .map(|p| (p.page_id.clone(), p.diff.clone()))
                    .collect();
                assert!(diffs.get(&root_page).unwrap().cleared());
                assert!(diffs.get(&page_id_1).unwrap().cleared());
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn clear_bit_updated_correctly_within_same_page_walker_pass() {
        let root = trie::TERMINATOR;
        let mut page_set = MockPageSet::default();

        // Leaves a and b are siblings at positions 2 and 3 on page `page_id_1`.
        // Upon deletion, the page walker will compact up, clearing the page diff
        // at the top of the stack. The insertion of leaves c and d will populate
        // page_id_1 with internal nodes, expecting to erase the clear bit.
        // Finally, leaves will be placed on `page_id_2`.
        let leaf_a_key_path = key_path![0, 0, 1, 0, 1, 0, 0, 0];
        let leaf_b_key_path = key_path![0, 0, 1, 0, 1, 0, 0, 1];
        let a_pos = trie_pos![0, 0, 1, 0, 1, 0, 0, 0];
        let b_pos = trie_pos![0, 0, 1, 0, 1, 0, 0, 1];
        let leaf_c_key_path = key_path![0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0];
        let leaf_d_key_path = key_path![0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1];
        let cd_pos = trie_pos![0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1];

        let mut page_id_iter = PageIdsIterator::new(leaf_c_key_path);
        page_id_iter.next(); // root
        let page_id_1 = page_id_iter.next().unwrap();
        let page_id_2 = page_id_iter.next().unwrap();

        let root = {
            let mut walker = PageWalker::<Blake3Hasher>::new(root, None, Metrics::new(false));
            walker.advance_and_replace(
                &page_set,
                TriePosition::new(),
                vec![(leaf_a_key_path, val(1)), (leaf_b_key_path, val(2))],
            );

            let Output::Root(new_root, updates) = walker.conclude() else {
                panic!();
            };

            let diffs: HashMap<PageId, PageDiff> = updates
                .iter()
                .map(|p| (p.page_id.clone(), p.diff.clone()))
                .collect();
            let diff = diffs.get(&page_id_1).unwrap().clone();
            let mut expected_diff = PageDiff::default();
            expected_diff.set_changed(0);
            expected_diff.set_changed(1); // the sibling is zeroed
            expected_diff.set_changed(2);
            expected_diff.set_changed(3);
            assert_eq!(diff, expected_diff);

            page_set.apply(updates);
            new_root
        };

        let mut walker = PageWalker::<Blake3Hasher>::new(root, None, Metrics::new(false));

        walker.advance_and_replace(&page_set, a_pos, vec![]);
        walker.advance_and_replace(&page_set, b_pos, vec![]);
        // During this step, the clear bit is set during the first compaction
        // and later it is expected to be removed.
        walker.advance_and_replace(
            &page_set,
            cd_pos,
            vec![(leaf_c_key_path, val(3)), (leaf_d_key_path, val(4))],
        );

        let Output::Root(_new_root, updates) = walker.conclude() else {
            panic!();
        };
        // No page is expected to be cleared.
        let diffs: HashMap<PageId, PageDiff> = updates
            .iter()
            .map(|p| (p.page_id.clone(), p.diff.clone()))
            .collect();
        assert!(!diffs.get(&page_id_1).unwrap().cleared());
        assert!(!diffs.get(&page_id_2).unwrap().cleared());
    }

    // TODO: this will break when I will stop pushing pages which are elided
    #[test]
    fn count_leaves() {
        let root = trie::TERMINATOR;
        let page_set = MockPageSet::default();

        let mut walker = PageWalker::<Blake3Hasher>::new(root, None, Metrics::new(false));
        walker.advance_and_replace(
            &page_set,
            TriePosition::new(),
            vec![
                (key_path![0, 0, 1, 0, 0, 0, 0, 0, 0], val(1)),
                (key_path![0, 0, 1, 0, 0, 0, 0, 0, 1], val(2)),
                (key_path![0, 0, 1, 0, 0, 0, 1], val(3)),
                (key_path![0, 0, 1, 0, 1], val(4)),
                (key_path![1, 0, 1, 1, 0], val(5)),
                (key_path![1, 0, 1, 1, 1, 0, 1, 0], val(6)),
                (key_path![1, 0, 1, 1, 1, 0, 1, 1, 0], val(7)),
                (key_path![1, 0, 1, 1, 1, 0, 1, 1, 1], val(8)),
            ],
        );

        match walker.conclude() {
            Output::Root(_, mut diffs) => {
                assert_eq!(diffs.len(), 3);
                for updted_page in diffs.drain(..) {
                    let UpdatedPage { page_id, page, .. } = updted_page;
                    let n_leaves = super::count_leaves::<Blake3Hasher>(&page, false);
                    if page_id == ROOT_PAGE_ID {
                        assert_eq!(2, n_leaves);
                    } else {
                        assert_eq!(3, n_leaves);
                    }
                }
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn count_cumulative_leaves() {
        let root = trie::TERMINATOR;
        let mut page_set = MockPageSet::default();

        let mut walker = PageWalker::<Blake3Hasher>::new(root, None, Metrics::new(false));
        walker.advance_and_replace(
            &page_set,
            trie_pos![0, 0, 1, 0],
            vec![
                (key_path![0, 0, 1, 0, 0, 0, 0, 0, 0], val(1)),
                (key_path![0, 0, 1, 0, 0, 0, 0, 0, 1], val(2)),
                (key_path![0, 0, 1, 0, 0, 0, 1], val(3)),
                (key_path![0, 0, 1, 0, 1], val(4)),
            ],
        );
        let stack_top = walker.stack.last().unwrap();
        assert_eq!(stack_top.leaves_counter, Some(3));

        walker.advance_and_replace(
            &page_set,
            trie_pos![1, 0, 1, 1,],
            vec![
                (key_path![1, 0, 1, 1, 0], val(5)),
                (key_path![1, 0, 1, 1, 1, 0, 1, 0], val(6)),
                (key_path![1, 0, 1, 1, 1, 0, 1, 1, 0], val(7)),
                (key_path![1, 0, 1, 1, 1, 0, 1, 1, 1], val(8)),
            ],
        );
        let stack_top = walker.stack.last().unwrap();
        assert_eq!(stack_top.leaves_counter, Some(6));

        let Output::Root(new_root, updates) = walker.conclude() else {
            panic!();
        };
        assert_eq!(updates.len(), 3);
        page_set.apply(updates);
        let mut walker = PageWalker::<Blake3Hasher>::new(new_root, None, Metrics::new(false));

        walker.advance_and_replace(&page_set, trie_pos![0, 0, 1, 0, 0, 0, 0, 0, 0], vec![]);
        walker.advance(trie_pos![1]);
        let stack_top = walker.stack.last().unwrap();
        assert_eq!(stack_top.leaves_counter, Some(2));

        walker.advance_and_replace(&page_set, trie_pos![1, 0, 1, 1, 0], vec![]);
        walker.advance(trie_pos![1, 0, 1, 1, 1, 0, 1]);
        let stack_top = walker.stack.last().unwrap();
        assert_eq!(stack_top.leaves_counter, Some(2));

        walker.advance_and_replace(&page_set, trie_pos![1, 0, 1, 1, 1, 0, 1, 0], vec![]);
        walker.advance(trie_pos![1, 1]);
        let stack_top = walker.stack.last().unwrap();
        assert_eq!(stack_top.leaves_counter, Some(4));

        walker.compact_up(Some(TriePosition::new()));
        let stack_top = walker.stack.last().unwrap();
        let n_leaves = super::count_leaves::<Blake3Hasher>(&stack_top.updated_page.page, false);
        assert_eq!(5, n_leaves + stack_top.leaves_counter.unwrap());
    }

    #[test]
    fn elided_pages() {
        let root = trie::TERMINATOR;
        let mut page_set = MockPageSet::default();

        let mut walker = PageWalker::<Blake3Hasher>::new(root, None, Metrics::new(false));
        walker.advance_and_replace(
            &page_set,
            TriePosition::new(),
            vec![
                // child page index: 0
                (key_path![0, 0, 0, 0, 0, 0, 0], val(1)),
                (key_path![0, 0, 0, 0, 0, 0, 1], val(2)),
                // child page index: 8
                (key_path![0, 0, 1, 0, 0, 0, 0], val(3)),
                (key_path![0, 0, 1, 0, 0, 0, 1], val(4)),
                // child page index: 16
                (key_path![0, 1, 0, 0, 0, 0, 0], val(5)),
                (key_path![0, 1, 0, 0, 0, 0, 1], val(6)),
                // child page index: 40
                (key_path![1, 0, 1, 0, 0, 0, 0], val(7)),
                (key_path![1, 0, 1, 0, 0, 0, 1], val(8)),
                // child page index: 48
                (key_path![1, 1, 0, 0, 0, 0, 0], val(9)),
                (key_path![1, 1, 0, 0, 0, 0, 1], val(10)),
                // child page index: 63
                (key_path![1, 1, 1, 1, 1, 1, 0], val(11)),
                (key_path![1, 1, 1, 1, 1, 1, 1], val(12)),
            ],
        );

        let Output::Root(new_root, updates) = walker.conclude() else {
            panic!();
        };

        // TODO: this will succed once we stop pushing elided pages
        //assert_eq!(updates.len(), 1);

        let mut expected_elided: std::collections::BTreeSet<u8> =
            vec![0, 8, 16, 40, 48, 63].into_iter().collect();

        // TODO: here we should expect *only* the root when we will push only updated pages.
        for updated in &updates {
            if updated.page_id == ROOT_PAGE_ID {
                let elided_childs = super::ElidedChilds::new(updated.page.elided_childs());

                for i in 0..64 {
                    if expected_elided.contains(&i) {
                        assert!(elided_childs.is_elided(ChildPageIndex::new(i).unwrap()));
                    } else {
                        assert!(!elided_childs.is_elided(ChildPageIndex::new(i).unwrap()));
                    }
                }
            }
        }

        page_set.apply(updates);

        // Add new leaves to child page index 8 to "un-elide" it.
        let mut walker = PageWalker::<Blake3Hasher>::new(new_root, None, Metrics::new(false));
        walker.advance_and_replace(
            &page_set,
            trie_pos![0, 0, 1, 0, 0, 0, 0],
            vec![
                // child page index: 8
                (key_path![0, 0, 1, 0, 0, 0, 0, 0, 0, 1], val(14)),
                (key_path![0, 0, 1, 0, 0, 0, 0, 0, 1, 0], val(15)),
                (key_path![0, 0, 1, 0, 0, 0, 0, 0, 1, 1], val(16)),
                (key_path![0, 0, 1, 0, 0, 0, 0, 1, 0, 0], val(17)),
                (key_path![0, 0, 1, 0, 0, 0, 0, 1, 1, 0], val(18)),
                (key_path![0, 0, 1, 0, 0, 0, 0, 1, 1, 1], val(19)),
            ],
        );

        walker.advance_and_replace(
            &page_set,
            trie_pos![0, 0, 1, 0, 0, 0, 1],
            vec![
                // child page index: 8
                (key_path![0, 0, 1, 0, 0, 0, 1, 0, 0, 1], val(21)),
                (key_path![0, 0, 1, 0, 0, 0, 1, 0, 1, 0], val(22)),
                (key_path![0, 0, 1, 0, 0, 0, 1, 0, 1, 1], val(23)),
                (key_path![0, 0, 1, 0, 0, 0, 1, 1, 0, 0], val(24)),
                (key_path![0, 0, 1, 0, 0, 0, 1, 1, 1, 0], val(25)),
                (key_path![0, 0, 1, 0, 0, 0, 1, 1, 1, 1], val(26)),
            ],
        );

        let Output::Root(_new_root, updates) = walker.conclude() else {
            panic!();
        };

        // TODO: this will succed once we stop pushing elided pages
        //assert_eq!(updates.len(), 1);
        expected_elided.remove(&8);

        // TODO: here we should expect *only* the root when we will push only updated pages.
        for updated in updates {
            if updated.page_id == ROOT_PAGE_ID {
                let elided_childs = super::ElidedChilds::new(updated.page.elided_childs());

                for i in 0..64 {
                    if expected_elided.contains(&i) {
                        assert!(elided_childs.is_elided(ChildPageIndex::new(i).unwrap()));
                    } else {
                        assert!(!elided_childs.is_elided(ChildPageIndex::new(i).unwrap()));
                    }
                }
            }
        }
    }
}
