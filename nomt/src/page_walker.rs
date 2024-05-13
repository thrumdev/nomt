//! Left-to-right walking and updating the page tree.
//!
//! The core usage is to create a [`PageWalker`] and alternate calls to `advance` and
//! `replace_terminal` repeatedly, followed by a single call to `conclude`.

use bitvec::prelude::*;
use nomt_core::{
    page::DEPTH,
    page_id::{PageId, ROOT_PAGE_ID},
    trie::{self, KeyPath, Node, NodeHasher, NodeHasherExt, NodeKind, ValueHash},
    trie_pos::TriePosition,
};
use std::collections::HashMap;

use crate::{
    page_cache::{Page, PageCache, PageDiff},
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
        write_pass: &mut WritePass<impl RegionContains<PageId>>,
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
        write_pass: &mut WritePass<impl RegionContains<PageId>>,
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
        write_pass: &mut WritePass<impl RegionContains<PageId>>,
        new_pos: TriePosition,
    ) {
        if let Some(ref pos) = self.last_position {
            assert!(new_pos.path() > pos.path());
            self.compact_up(write_pass, Some(new_pos.clone()));
        }
        self.last_position = Some(new_pos);
    }

    fn place_node(&mut self, write_pass: &mut WritePass<impl RegionContains<PageId>>, node: Node) {
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
        write_pass: &mut WritePass<impl RegionContains<PageId>>,
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
    pub fn conclude(mut self, write_pass: &mut WritePass<impl RegionContains<PageId>>) -> Output {
        self.compact_up(write_pass, None);
        if self.parent_page.is_none() {
            Output::Root(self.root, self.diffs)
        } else {
            Output::ChildPageRoots(self.child_page_roots, self.diffs)
        }
    }

    fn compact_up(
        &mut self,
        write_pass: &mut WritePass<impl RegionContains<PageId>>,
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
        write_pass: &mut WritePass<impl RegionContains<PageId>>,
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
    fn node(&self, read_pass: &ReadPass<impl RegionContains<PageId>>) -> Node {
        let node_index = self.position.node_index();
        let page = self.stack.last().unwrap();
        page.1.node(read_pass, node_index)
    }

    // read the sibling node at the current position. panics if no current page.
    fn sibling_node(&self, read_pass: &ReadPass<impl RegionContains<PageId>>) -> Node {
        let node_index = self.position.sibling_index();
        let page = self.stack.last().unwrap();
        page.1.node(read_pass, node_index)
    }

    // set a node in the current page at the given index. panics if no current page.
    fn set_node(&mut self, write_pass: &mut WritePass<impl RegionContains<PageId>>, node: Node) {
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
        read_pass: &ReadPass<impl RegionContains<PageId>>,
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
        write_pass: &mut WritePass<impl RegionContains<PageId>>,
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

    // Build the stack to target a particular position.
    //
    // Precondition: the stack is either empty or contains an ancestor of the page ID the position
    // lands in.
    fn build_stack(&mut self, position: TriePosition) {
        let new_page_id = position.page_id();
        self.position = position;
        if let Some(ref parent_page) = self.parent_page {
            assert!(new_page_id.is_some());
            assert!(new_page_id
                .as_ref()
                .unwrap()
                .is_descendant_of(&parent_page.0));
        }
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
