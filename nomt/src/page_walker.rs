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

use bitvec::prelude::*;
use nomt_core::{
    page::DEPTH,
    page_id::{PageId, ROOT_PAGE_ID},
    trie::{self, KeyPath, Node, NodeHasher, NodeHasherExt, NodeKind, ValueHash},
    trie_pos::TriePosition,
};
use std::collections::HashMap;

use crate::{
    page_cache::{Page, PageCache, PageDiff, ShardIndex},
    rw_pass_cell::{ReadPass, RegionContains, WritePass},
};

/// The output of the page walker.
pub enum Output {
    /// A new root node.
    ///
    /// This is always the output when no parent page is supplied to the walker.
    Root(Node, HashMap<PageId, PageDiff>),
    /// Nodes to set in the bottom layer of the parent page, indexed by the position of the node
    /// to set.
    ///
    /// This is always the output when a parent page is supplied to the walker.
    ChildPageRoots(Vec<(TriePosition, Node)>, HashMap<PageId, PageDiff>),
}

/// Left-to-right updating walker over the page tree.
pub struct PageWalker<H> {
    page_cache: PageCache,
    // last position `advance` was invoked with.
    last_position: Option<TriePosition>,
    // actual position
    position: TriePosition,
    parent_page: Option<(PageId, Page)>,
    child_page_roots: Vec<(TriePosition, Node)>,
    root: Node,
    diffs: HashMap<PageId, PageDiff>,

    // the stack contains pages (ascending) which are descendants of the parent page, if any.
    stack: Vec<(PageId, Page)>,

    // the sibling stack contains the previous node values of siblings on the path to the current
    // position, annotated with their depths.
    sibling_stack: Vec<(Node, usize)>,
    prev_node: Option<Node>, // the node at `self.position` which was replaced in a previous call

    _marker: std::marker::PhantomData<H>,
}

impl<H: NodeHasher> PageWalker<H> {
    /// Create a new [`PageWalker`], with an optional parent page for constraining operations
    /// to a subsection of the page tree.
    pub fn new(root: Node, page_cache: PageCache, parent_page: Option<PageId>) -> Self {
        let parent_page = parent_page.map(|id| (id.clone(), page_cache.retrieve_sync(id, false)));

        PageWalker {
            page_cache,
            last_position: None,
            position: TriePosition::new(),
            parent_page,
            child_page_roots: Vec::new(),
            root,
            diffs: HashMap::new(),
            stack: Vec::new(),
            sibling_stack: Vec::new(),
            prev_node: None,
            _marker: std::marker::PhantomData,
        }
    }

    /// Advance to a given trie position and replace the terminal node there with a trie
    /// based on the provided key-value pairs.
    ///
    /// The key-value pairs should be sorted and should all be suffixes of the given position.
    ///
    /// An empty vector deletes any existing terminal node.
    /// # Panics
    ///
    /// Panics if the current trie position is not a terminal node.
    ///
    /// Panics if this falls in a page which is not a descendant of the parent page, if any.
    /// Panics if this is not greater than the previous trie position.
    pub fn advance_and_replace(
        &mut self,
        write_pass: &mut WritePass<impl RegionContains<ShardIndex>>,
        new_pos: TriePosition,
        ops: impl IntoIterator<Item = (KeyPath, ValueHash)>,
    ) {
        if let Some(ref pos) = self.last_position {
            assert!(new_pos.path() > pos.path());
            self.compact_up(write_pass, Some(new_pos.clone()));
        }
        self.last_position = Some(new_pos.clone());
        self.build_stack(new_pos);

        self.replace_terminal(write_pass, ops);
    }

    /// Advance to a given trie position and place the given node at that position.
    ///
    /// It is the responsibility of the user to ensure that:
    ///   - if this is a leaf node, the leaf data have been written to the two child positions.
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
        write_pass: &mut WritePass<impl RegionContains<ShardIndex>>,
        new_pos: TriePosition,
        node: Node,
    ) {
        if let Some(ref pos) = self.last_position {
            assert!(new_pos.path() > pos.path());
            self.compact_up(write_pass, Some(new_pos.clone()));
        }
        self.last_position = Some(new_pos.clone());
        self.build_stack(new_pos);
        self.place_node(write_pass, node);
    }

    /// Advance to a given trie position without updating.
    ///
    /// # Panics
    ///
    /// Panics if this falls in a page which is not a descendant of the parent page, if any.
    /// Panics if this is not greater than the previous trie position.
    pub fn advance(
        &mut self,
        write_pass: &mut WritePass<impl RegionContains<ShardIndex>>,
        new_pos: TriePosition,
    ) {
        if let Some(ref pos) = self.last_position {
            assert!(new_pos.path() > pos.path());
            self.compact_up(write_pass, Some(new_pos.clone()));
        }

        let page_id = new_pos.page_id();
        self.assert_page_in_scope(page_id.as_ref());
        self.last_position = Some(new_pos);
    }

    fn place_node(
        &mut self,
        write_pass: &mut WritePass<impl RegionContains<ShardIndex>>,
        node: Node,
    ) {
        if self.position.is_root() {
            self.prev_node = Some(self.root);
            self.root = node;
        } else {
            self.prev_node = Some(self.node(write_pass.downgrade()));
            self.set_node(write_pass, node);
        }
    }

    fn replace_terminal(
        &mut self,
        write_pass: &mut WritePass<impl RegionContains<ShardIndex>>,
        ops: impl IntoIterator<Item = (KeyPath, ValueHash)>,
    ) {
        let node = if self.position.is_root() {
            self.root
        } else {
            self.node(write_pass.downgrade())
        };

        self.prev_node = Some(node);

        assert!(!trie::is_internal(&node));

        // clear leaf children before overwriting terminal.
        self.write_leaf_children(write_pass, None, false);

        // replace sub-trie at the given position
        nomt_core::update::build_trie::<H>(
            self.position.depth() as usize,
            ops,
            |mut control, node, leaf_data| {
                // avoid popping pages off the stack if we are jumping to a sibling.
                if control.up > 0 && !control.down.is_empty() {
                    for _ in 0..(control.up - 1) {
                        self.up();
                    }
                    if control.down[0] == !self.position.peek_last_bit() {
                        // UNWRAP: checked above
                        self.position.sibling();
                        control.down = &control.down[1..];
                    } else {
                        self.up();
                    }
                } else {
                    for _ in 0..control.up {
                        self.up();
                    }
                }

                self.down_fresh(control.down);

                if self.position.is_root() {
                    self.root = node;
                } else {
                    self.set_node(write_pass, node);
                }

                if let Some(leaf_data) = leaf_data {
                    self.write_leaf_children(write_pass, Some(leaf_data), true);
                }
            },
        );

        // build_trie should always return us to the original position.
        if !self.position.is_root() {
            assert_eq!(
                self.stack.last().unwrap().0,
                self.position.page_id().unwrap()
            );
        } else {
            assert!(self.stack.is_empty());
        }
    }

    // move the current position up.
    fn up(&mut self) {
        if self.position.depth_in_page() == 1 {
            assert!(self.stack.pop().is_some());
        }
        self.position.up(1);
    }

    // move the current position down into "fresh" territory: pages guaranteed not to exist yet.
    fn down_fresh(&mut self, bit_path: &BitSlice<u8, Msb0>) {
        for bit in bit_path.iter().by_vals() {
            if self.position.is_root() {
                self.stack.push((
                    ROOT_PAGE_ID,
                    self.page_cache.retrieve_sync(ROOT_PAGE_ID, true),
                ));
            } else if self.position.depth_in_page() == DEPTH {
                // UNWRAP: the only legal positions are below the "parent" (root or parent_page)
                //         and stack always contains all pages to position.
                let parent_page_id = &self.stack.last().unwrap().0;
                let child_page_index = self.position.child_page_index();

                // UNWRAP: we never overflow the page stack.
                let child_page_id = parent_page_id.child_page_id(child_page_index).unwrap();
                self.stack.push((
                    child_page_id.clone(),
                    self.page_cache.retrieve_sync(child_page_id, true),
                ));
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
    pub fn conclude(
        mut self,
        write_pass: &mut WritePass<impl RegionContains<ShardIndex>>,
    ) -> Output {
        self.compact_up(write_pass, None);
        if self.parent_page.is_none() {
            Output::Root(self.root, self.diffs)
        } else {
            Output::ChildPageRoots(self.child_page_roots, self.diffs)
        }
    }

    fn compact_up(
        &mut self,
        write_pass: &mut WritePass<impl RegionContains<ShardIndex>>,
        target_pos: Option<TriePosition>,
    ) {
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
            let (next_node, prev_leaf_data) = self.compact_step(write_pass);
            self.up();

            if let Some(prev_leaf_data) = prev_leaf_data {
                // writing leaf children is always guaranteed in-bounds.
                self.write_leaf_children(write_pass, Some(prev_leaf_data), false);
            }

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
                    self.sibling_stack.push((
                        self.node(write_pass.downgrade()),
                        self.position.depth() as usize,
                    ));
                }

                self.set_node(write_pass, next_node);
            }
        }
    }

    fn compact_step(
        &mut self,
        write_pass: &mut WritePass<impl RegionContains<ShardIndex>>,
    ) -> (Node, Option<trie::LeafData>) {
        let node = self.node(write_pass.downgrade());
        let sibling = self.sibling_node(write_pass.downgrade());
        let bit = self.position.peek_last_bit();

        match (NodeKind::of(&node), NodeKind::of(&sibling)) {
            (NodeKind::Terminator, NodeKind::Terminator) => {
                // compact terminators.
                (trie::TERMINATOR, None)
            }
            (NodeKind::Leaf, NodeKind::Terminator) => {
                // compact: clear this node, move leaf up.
                let leaf_data = self.read_leaf_children(write_pass.downgrade());
                self.write_leaf_children(write_pass, None, false);
                self.set_node(write_pass, trie::TERMINATOR);

                (node, Some(leaf_data))
            }
            (NodeKind::Terminator, NodeKind::Leaf) => {
                // compact: clear sibling node, move leaf up.
                self.position.sibling();
                let leaf_data = self.read_leaf_children(write_pass.downgrade());
                self.write_leaf_children(write_pass, None, false);
                self.set_node(write_pass, trie::TERMINATOR);

                (sibling, Some(leaf_data))
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

                (H::hash_internal(&node_data), None)
            }
        }
    }

    // read the node at the current position. panics if no current page.
    fn node(&self, read_pass: &ReadPass<impl RegionContains<ShardIndex>>) -> Node {
        let node_index = self.position.node_index();
        let page = self.stack.last().unwrap();
        page.1.node(read_pass, node_index)
    }

    // read the sibling node at the current position. panics if no current page.
    fn sibling_node(&self, read_pass: &ReadPass<impl RegionContains<ShardIndex>>) -> Node {
        let node_index = self.position.sibling_index();
        let page = self.stack.last().unwrap();
        page.1.node(read_pass, node_index)
    }

    // set a node in the current page at the given index. panics if no current page.
    fn set_node(
        &mut self,
        write_pass: &mut WritePass<impl RegionContains<ShardIndex>>,
        node: Node,
    ) {
        let node_index = self.position.node_index();
        let page = self.stack.last().unwrap();
        page.1.set_node(write_pass, node_index, node);

        self.diffs
            .entry(page.0.clone())
            .or_default()
            .set_changed(node_index);
    }

    // read the leaf children of a node in the current page at the given position.
    fn read_leaf_children(
        &self,
        read_pass: &ReadPass<impl RegionContains<ShardIndex>>,
    ) -> trie::LeafData {
        let page = self.stack.last().or(self.parent_page.as_ref());
        let (page, _, children) =
            crate::page_cache::locate_leaf_data(&self.position, page, |page_id| {
                self.page_cache.retrieve_sync(page_id, false)
            });

        trie::LeafData {
            key_path: page.node(&read_pass, children.left()),
            value_hash: page.node(&read_pass, children.right()),
        }
    }

    // write the leaf children of a node in the current page at the given index.
    fn write_leaf_children(
        &mut self,
        write_pass: &mut WritePass<impl RegionContains<ShardIndex>>,
        leaf_data: Option<trie::LeafData>,
        hint_fresh: bool,
    ) {
        let page = self.stack.last().or(self.parent_page.as_ref());
        let (page, page_id, children) =
            crate::page_cache::locate_leaf_data(&self.position, page, |page_id| {
                self.page_cache.retrieve_sync(page_id, hint_fresh)
            });

        match leaf_data {
            None => page.clear_leaf_data(write_pass, children),
            Some(leaf) => page.set_leaf_data(write_pass, children, leaf),
        }

        let diff = self.diffs.entry(page_id.clone()).or_default();
        diff.set_changed(children.left());
        diff.set_changed(children.right());
        diff.set_changed(crate::page_cache::LEAF_META_BITFIELD_SLOT);
    }

    fn assert_page_in_scope(&self, page_id: Option<&PageId>) {
        match page_id {
            Some(page_id) => {
                if let Some(ref parent_page) = self.parent_page {
                    assert!(page_id != &parent_page.0);
                    assert!(page_id.is_descendant_of(&parent_page.0));
                }
            }
            None => assert!(self.parent_page.is_none()),
        }
    }

    // Build the stack to target a particular position.
    //
    // Precondition: the stack is either empty or contains an ancestor of the page ID the position
    // lands in.
    fn build_stack(&mut self, position: TriePosition) {
        let new_page_id = position.page_id();
        self.assert_page_in_scope(new_page_id.as_ref());

        self.position = position;
        let Some(page_id) = new_page_id else {
            self.stack.clear();
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
            .or(self.parent_page.as_ref())
            .map(|(id, _)| id.clone());
        let start_len = self.stack.len();
        let mut cur_ancestor = page_id;
        let mut push_count = 0;
        while Some(&cur_ancestor) != target.as_ref() {
            let page = self.page_cache.retrieve_sync(cur_ancestor.clone(), false);
            self.stack.push((cur_ancestor.clone(), page));
            cur_ancestor = cur_ancestor.parent_page_id();
            push_count += 1;

            // stop pushing once we reach the root page.
            if cur_ancestor == ROOT_PAGE_ID {
                break;
            }
        }

        // we pushed onto the stack in descending, so now reverse everything we just pushed to
        // make it ascending.
        self.stack[start_len..start_len + push_count].reverse();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Blake3Hasher;
    use nomt_core::page_id::ChildPageIndex;

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

    #[test]
    #[should_panic]
    fn advance_backwards_panics() {
        let root = trie::TERMINATOR;
        let page_cache = PageCache::new_mocked(&crate::Options {
            path: "".into(),
            fetch_concurrency: 1,
            traversal_concurrency: 1,
        });

        let mut walker = PageWalker::<Blake3Hasher>::new(root, page_cache.clone(), None);
        let mut write_pass = page_cache.new_write_pass();
        let trie_pos_a = trie_pos![1];
        let trie_pos_b = trie_pos![0];
        walker.advance(&mut write_pass, trie_pos_a);
        walker.advance(&mut write_pass, trie_pos_b);
    }

    #[test]
    #[should_panic]
    fn advance_same_panics() {
        let root = trie::TERMINATOR;
        let page_cache = PageCache::new_mocked(&crate::Options {
            path: "".into(),
            fetch_concurrency: 1,
            traversal_concurrency: 1,
        });

        let mut walker = PageWalker::<Blake3Hasher>::new(root, page_cache.clone(), None);
        let mut write_pass = page_cache.new_write_pass();
        let trie_pos_a = trie_pos![0];
        walker.advance(&mut write_pass, trie_pos_a.clone());
        walker.advance(&mut write_pass, trie_pos_a);
    }

    #[test]
    #[should_panic]
    fn advance_to_parent_page_panics() {
        let root = trie::TERMINATOR;
        let page_cache = PageCache::new_mocked(&crate::Options {
            path: "".into(),
            fetch_concurrency: 1,
            traversal_concurrency: 1,
        });

        let mut walker =
            PageWalker::<Blake3Hasher>::new(root, page_cache.clone(), Some(ROOT_PAGE_ID));
        let mut write_pass = page_cache.new_write_pass();
        let trie_pos_a = trie_pos![0, 0, 0, 0, 0, 0];
        walker.advance(&mut write_pass, trie_pos_a);
    }

    #[test]
    #[should_panic]
    fn advance_to_root_with_parent_page_panics() {
        let root = trie::TERMINATOR;
        let page_cache = PageCache::new_mocked(&crate::Options {
            path: "".into(),
            fetch_concurrency: 1,
            traversal_concurrency: 1,
        });

        let mut walker =
            PageWalker::<Blake3Hasher>::new(root, page_cache.clone(), Some(ROOT_PAGE_ID));
        let mut write_pass = page_cache.new_write_pass();
        walker.advance(&mut write_pass, TriePosition::new());
    }

    #[test]
    fn compacts_and_updates_root() {
        let root = trie::TERMINATOR;
        let page_cache = PageCache::new_mocked(&crate::Options {
            path: "".into(),
            fetch_concurrency: 1,
            traversal_concurrency: 1,
        });

        let mut walker = PageWalker::<Blake3Hasher>::new(root, page_cache.clone(), None);
        let mut write_pass = page_cache.new_write_pass();
        let trie_pos_a = trie_pos![0, 0];
        walker.advance_and_replace(
            &mut write_pass,
            trie_pos_a,
            vec![
                (key_path![0, 0, 1, 0], val(1)),
                (key_path![0, 0, 1, 1], val(2)),
            ],
        );

        let trie_pos_b = trie_pos![0, 1];
        walker.advance(&mut write_pass, trie_pos_b);

        let trie_pos_c = trie_pos![1];
        walker.advance_and_replace(
            &mut write_pass,
            trie_pos_c,
            vec![(key_path![1, 0], val(3)), (key_path![1, 1], val(4))],
        );

        match walker.conclude(&mut write_pass) {
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
                        |_, _, _| {}
                    )
                );
                assert_eq!(diffs.len(), 1);
                assert!(diffs.contains_key(&ROOT_PAGE_ID));
            }
            Output::ChildPageRoots(_, _) => unreachable!(),
        }
    }

    #[test]
    fn sets_child_page_roots() {
        let root = trie::TERMINATOR;
        let page_cache = PageCache::new_mocked(&crate::Options {
            path: "".into(),
            fetch_concurrency: 1,
            traversal_concurrency: 1,
        });

        let mut walker =
            PageWalker::<Blake3Hasher>::new(root, page_cache.clone(), Some(ROOT_PAGE_ID));
        let mut write_pass = page_cache.new_write_pass();
        let trie_pos_a = trie_pos![0, 0, 0, 0, 0, 0, 0];

        walker.advance_and_replace(
            &mut write_pass,
            trie_pos_a.clone(),
            vec![(key_path![0, 0, 0, 0, 0, 0, 0], val(1))],
        );
        let trie_pos_b = trie_pos![0, 0, 0, 0, 0, 0, 1];
        walker.advance_and_replace(
            &mut write_pass,
            trie_pos_b.clone(),
            vec![(key_path![0, 0, 0, 0, 0, 0, 1], val(2))],
        );

        let trie_pos_c = trie_pos![0, 0, 0, 0, 0, 1, 0];
        walker.advance_and_replace(
            &mut write_pass,
            trie_pos_c.clone(),
            vec![(key_path![0, 0, 0, 0, 0, 1, 0], val(3))],
        );
        let trie_pos_d = trie_pos![0, 0, 0, 0, 0, 1, 1];
        walker.advance_and_replace(
            &mut write_pass,
            trie_pos_d.clone(),
            vec![(key_path![0, 0, 0, 0, 0, 1, 1], val(4))],
        );

        match walker.conclude(&mut write_pass) {
            Output::Root(_new_root, _diffs) => unreachable!(),
            Output::ChildPageRoots(page_roots, diffs) => {
                assert_eq!(page_roots.len(), 2);
                assert_eq!(diffs.len(), 2);
                let left_page_id = ROOT_PAGE_ID
                    .child_page_id(ChildPageIndex::new(0).unwrap())
                    .unwrap();
                let right_page_id = ROOT_PAGE_ID
                    .child_page_id(ChildPageIndex::new(1).unwrap())
                    .unwrap();

                assert!(diffs.contains_key(&left_page_id));
                assert!(diffs.contains_key(&right_page_id));
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
                        |_, _, _| {}
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
                        |_, _, _| {}
                    )
                );
            }
        }
    }

    #[test]
    fn tracks_sibling_prev_values() {
        let root = trie::TERMINATOR;
        let page_cache = PageCache::new_mocked(&crate::Options {
            path: "".into(),
            fetch_concurrency: 1,
            traversal_concurrency: 1,
        });
        let mut write_pass = page_cache.new_write_pass();

        let path_1 = key_path![0, 0, 0, 0];
        let path_2 = key_path![1, 0, 0, 0];
        let path_3 = key_path![1, 1, 0, 0];
        let path_4 = key_path![1, 1, 1, 0];
        let path_5 = key_path![1, 1, 1, 1];

        // first build a trie with these 5 key-value pairs. it happens to have the property that
        // all the "left" nodes are leaves.
        let root = {
            let mut walker = PageWalker::<Blake3Hasher>::new(root, page_cache.clone(), None);
            walker.advance_and_replace(
                &mut write_pass,
                TriePosition::new(),
                vec![
                    (path_1, val(1)),
                    (path_2, val(2)),
                    (path_3, val(3)),
                    (path_4, val(4)),
                    (path_5, val(5)),
                ],
            );

            match walker.conclude(&mut write_pass) {
                Output::Root(new_root, _) => new_root,
                _ => unreachable!(),
            }
        };

        let mut walker = PageWalker::<Blake3Hasher>::new(root, page_cache.clone(), None);

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
            &mut write_pass,
            TriePosition::from_path_and_depth(path_1, 4),
            vec![(path_1, val(11))],
        );
        assert_eq!(walker.siblings(), &expected_siblings[..0]);

        walker.advance_and_replace(
            &mut write_pass,
            TriePosition::from_path_and_depth(path_2, 4),
            vec![(path_2, val(12))],
        );
        assert_eq!(walker.siblings(), &expected_siblings[..1]);

        walker.advance_and_replace(
            &mut write_pass,
            TriePosition::from_path_and_depth(path_3, 4),
            vec![(path_3, val(13))],
        );
        assert_eq!(walker.siblings(), &expected_siblings[..2]);

        walker.advance_and_replace(
            &mut write_pass,
            TriePosition::from_path_and_depth(path_4, 4),
            vec![(path_4, val(14))],
        );
        assert_eq!(walker.siblings(), &expected_siblings[..3]);

        walker.advance_and_replace(
            &mut write_pass,
            TriePosition::from_path_and_depth(path_5, 4),
            vec![(path_5, val(15))],
        );
        assert_eq!(walker.siblings(), &expected_siblings[..4]);
    }
}
