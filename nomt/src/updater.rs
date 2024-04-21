use itertools::Itertools;

use crate::{
    page_cache::{Page, PageAllocator, PageCache},
    rw_pass_cell::{ReadPass, WritePass},
};
use nomt_core::{
    page::DEPTH,
    page_id::{PageId, PageIdsIterator, ROOT_PAGE_ID},
    trie::{self, KeyPath, LeafData, Node, NodeHasher, NodeHasherExt, NodeKind, ValueHash},
    trie_pos::{ChildNodeIndices, TriePosition},
    update::{VisitControl, VisitedTerminal},
};

/// The page updater.
pub struct PageUpdater<H: NodeHasher> {
    pages: Vec<(PageId, Page)>,
    new_pages: Vec<(PageId, Page)>,
    root: Node,
    write_pass: WritePass,
    allocator: PageAllocator,

    pos: TriePosition,
    page_stack: Vec<usize>,
    // the page where the current terminal's leaf data is or would be stored.
    // None means "same page as terminal".
    leaf_data_page: Option<usize>,
    _marker: std::marker::PhantomData<H>,
}

impl<H: NodeHasher> PageUpdater<H> {
    /// Create a new [`PageUpdater`] given a set of pages in lexicographical order.
    ///
    /// The page-set must encompass all terminal nodes visited in subsequent calls to `visit`,
    /// along with the pages holding the child positions of any leaf nodes,
    /// as well as all pages directly below leaf siblings on the path to a leaf being deleted.
    pub fn new(
        pages: Vec<(PageId, Page)>,
        root: Node,
        write_pass: WritePass,
        allocator: PageAllocator,
    ) -> Self {
        let mut new_pages = Vec::new();
        if pages.is_empty() {
            let page = allocator.allocate();
            new_pages.push((ROOT_PAGE_ID, page));
        } else {
            assert_eq!(pages[0].0, ROOT_PAGE_ID);
        }

        PageUpdater {
            pages,
            root,
            write_pass,
            allocator,

            new_pages: Vec::new(),
            pos: TriePosition::new(),
            page_stack: Vec::new(),
            leaf_data_page: Some(0),
            _marker: std::marker::PhantomData,
        }
    }

    /// Visit a terminal node and replace the node with the sub-trie comprised of the
    /// given operations. An empty iterator deletes the node.
    ///
    /// This will panic if the updater does not hold the pages for the terminal node or the
    /// terminal node path is lexicographically not after the last call to `visit`.
    pub fn visit(
        &mut self,
        terminal: VisitedTerminal,
        ops: impl IntoIterator<Item = (KeyPath, ValueHash)>,
    ) {
        // 1. Prepare to replace the sub-trie at the given terminal.
        self.prepare(Some(terminal));

        // 2. Replace the sub-trie at the given terminal.
        nomt_core::update::build_trie::<H>(
            terminal.depth as usize,
            ops,
            |visit_control, node, leaf_data| {
                self.post_terminal_visit_control(visit_control, leaf_data.is_some());
                if let Some(leaf_data) = leaf_data {
                    self.place_leaf(node, leaf_data);
                } else {
                    self.place_non_leaf(node);
                }
            },
        );
    }

    /// Conclude the update. This returns the new root hash, along with an (ordered) iterator
    /// over all altered pages.
    pub fn conclude(mut self) -> (Node, impl IntoIterator<Item = (PageId, Page)>) {
        self.prepare(None);

        self.new_pages.sort_unstable_by(|a, b| {
            a.0.length_dependent_encoding()
                .cmp(b.0.length_dependent_encoding())
        });
        let pages = self.pages.into_iter().merge_by(self.new_pages, |x, y| {
            x.0.length_dependent_encoding() < y.0.length_dependent_encoding()
        });

        (self.root, pages)
    }

    fn prepare(&mut self, next: Option<VisitedTerminal>) {
        let Some(next) = next else {
            self.compact_up(self.pos.depth() as usize);
            return;
        };

        let new_pos = TriePosition::from_path_and_depth(next.path, next.depth);
        let shared_depth = self.pos.shared_bits(&new_pos);

        // shared_depth is guaranteed less than current_depth because the full key isn't
        // shared and is non-zero.
        // we want to compact up (inclusive) to the depth `shared_depth + 1`
        if !self.pos.is_root() {
            self.compact_up((self.pos.depth() as usize) - (shared_depth + 1))
        }

        let old_pos = std::mem::replace(&mut self.pos, new_pos);

        let n_pages = self.pos.page_depth();
        let n_shared_pages = (shared_depth + 5) / DEPTH;

        let mut next_search_start = self.page_stack.last().map_or(0, |x| *x);

        // keep any shared pages
        self.page_stack.truncate(n_shared_pages);

        if n_shared_pages == n_pages {
            return;
        }

        let mut new_active_pages = PageIdsIterator::new(next.path)
            .take(n_pages)
            .skip(n_shared_pages)
            .map(|id| {
                let pos = self.pages[next_search_start..]
                    .iter()
                    .position(|x| x.0 == id)
                    .expect("no page for key");
                let pos = pos + next_search_start;
                next_search_start = pos + 1;
                pos
            });

        self.page_stack.extend(new_active_pages);

        self.leaf_data_page = if self.pos.depth_in_page() == DEPTH {
            if next.leaf.is_some() {
                new_active_pages.next()
            } else {
                let page_idx = self.allocate_page();
                Some(page_idx)
            }
        } else {
            None
        };
    }

    fn page(&self, idx: usize) -> &Page {
        if idx >= self.pages.len() {
            &self.new_pages[idx - self.pages.len()].1
        } else {
            &self.pages[idx].1
        }
    }

    fn page_id(&self, idx: usize) -> &PageId {
        if idx >= self.pages.len() {
            &self.new_pages[idx - self.pages.len()].0
        } else {
            &self.pages[idx].0
        }
    }

    fn node(&self) -> Node {
        if let Some(last) = self.page_stack.last() {
            self.page(*last)
                .node(self.write_pass.downgrade(), self.pos.node_index())
        } else {
            self.root
        }
    }

    fn sibling(&self) -> Node {
        let last = self.page_stack.last().expect("requested root sibling");
        self.page(*last)
            .node(self.write_pass.downgrade(), self.pos.sibling_index())
    }

    fn up(&mut self) {
        assert!(!self.pos.is_root());

        if self.pos.depth_in_page() == 1 {
            self.leaf_data_page = self.page_stack.pop();
        } else {
            self.leaf_data_page = None;
        }

        self.pos.up(1);
    }

    fn allocate_page(&mut self) -> usize {
        let page_id = if self.pos.is_root() {
            ROOT_PAGE_ID
        } else {
            // UNWRAP: not at root
            let last_page_idx = self.page_stack.last().unwrap();
            let last_page_id = self.page_id(*last_page_idx);
            let child_index = self.pos.child_page_index();

            // UNWRAP: we never traverse beyond 256 levels.
            last_page_id.child_page_id(child_index).unwrap()
        };

        self.new_pages.push((page_id, self.allocator.allocate()));
        self.new_pages.len() - 1
    }

    // note: we assume that this never visits the same node twice and that visit control
    // proceeds lexicographically.
    fn post_terminal_visit_control(&mut self, mut control: VisitControl, leaf: bool) {
        if control.up > 0 {
            if !control.down.is_empty() {
                // avoid going all the way up if we can traverse to a sibling.
                // this preserves the page stack.
                self.pos.up(control.up - 1);
                let bit = self.pos.peek_last_bit();
                if control.down[0] != bit {
                    self.pos.sibling();
                }

                control.down = &control.down[1..];
            } else {
                self.pos.up(control.up);
            }
        }

        self.page_stack.truncate(self.pos.page_depth());
        self.leaf_data_page = None;

        // since we are in fresh territory, never revisit nodes, and avoid popping the page
        // stack above, all pages at this point are guaranteed to be new.
        for bit in control.down.iter().by_vals() {
            if self.pos.depth_in_page() == DEPTH {
                let page_idx = self.allocate_page();
                self.page_stack.push(page_idx);
            }

            self.pos.down(bit);
        }

        if leaf && self.pos.depth_in_page() == DEPTH {
            let page_idx = self.allocate_page();
            self.leaf_data_page = Some(page_idx);
        }
    }

    fn place_non_leaf(&mut self, node: Node) {
        assert!(!trie::is_leaf(&node));
        assert!(!trie::is_leaf(&self.node()));

        if let Some(page_idx) = self.page_stack.last() {
            let page = self.page(*page_idx);
            page.set_node(&mut self.write_pass, self.pos.node_index(), node);
        } else {
            self.root = node;
        }
    }

    fn place_leaf(&mut self, node: Node, leaf: trie::LeafData) {
        assert!(trie::is_leaf(&node));

        let (leaf_data_page_idx, positions) = if let Some(l) = self.leaf_data_page {
            (l, ChildNodeIndices::from_left(0))
        } else {
            // UNWRAP: leaf data page always some at root.
            let idx = *self.page_stack.last().unwrap();
            (idx, self.pos.child_node_indices())
        };

        self.page(leaf_data_page_idx)
            .set_leaf_data(&mut self.write_pass, positions, leaf);

        if let Some(page_idx) = self.page_stack.last() {
            let page = self.page(*page_idx);
            page.set_node(&mut self.write_pass, self.pos.node_index(), node);
        } else {
            self.root = node;
        }
    }

    fn read_and_clear_current_leaf(&mut self) -> (Node, LeafData) {
        let node = self.node();

        let (leaf_data_page_idx, positions) = if let Some(l) = self.leaf_data_page {
            (l, ChildNodeIndices::from_left(0))
        } else {
            // UNWRAP: leaf data page always some at root.
            let idx = *self.page_stack.last().unwrap();
            (idx, self.pos.child_node_indices())
        };

        let leaf_data = self
            .page(leaf_data_page_idx)
            .read_leaf_data(self.write_pass.downgrade(), positions);
        self.page(leaf_data_page_idx)
            .clear_leaf_data(&mut self.write_pass, positions);

        if self.pos.is_root() {
            self.root = trie::TERMINATOR;
        } else {
            let idx = *self.page_stack.last().unwrap();
            self.pages[idx].1.set_node(
                &mut self.write_pass,
                self.pos.node_index(),
                trie::TERMINATOR,
            );
        }

        (node, leaf_data)
    }

    fn read_and_clear_sibling_leaf(&mut self) -> (Node, LeafData) {
        let cur_page_idx = *self.page_stack.last().unwrap();

        let (sibling_data_page_idx, positions) = if self.pos.depth_in_page() == DEPTH {
            let sibling_data_page_idx = self.pos.sibling_child_page_index();

            // UNWRAP: page overflow by calling parent/child in sequence is impossible.
            let sibling_data_page_id = self.pages[cur_page_idx]
                .0
                .parent_page_id()
                .child_page_id(sibling_data_page_idx)
                .unwrap();

            let i = self
                .pages
                .binary_search_by(|x| {
                    sibling_data_page_id
                        .length_dependent_encoding()
                        .cmp(x.0.length_dependent_encoding())
                })
                .expect("missing page along deleted path");

            (i, ChildNodeIndices::from_left(0))
        } else {
            (cur_page_idx, self.pos.sibling_child_node_indices())
        };

        let sibling = self.sibling();
        self.page(cur_page_idx).set_node(
            &mut self.write_pass,
            self.pos.sibling_index(),
            trie::TERMINATOR,
        );

        let leaf_data = self.pages[sibling_data_page_idx]
            .1
            .read_leaf_data(self.write_pass.downgrade(), positions);
        self.pages[sibling_data_page_idx]
            .1
            .clear_leaf_data(&mut self.write_pass, positions);
        (sibling, leaf_data)
    }

    fn compact_up(&mut self, levels: usize) {
        if self.pos.depth() == 0 {
            return;
        }

        assert!(levels <= self.pos.depth() as usize);

        // leaf being relocated upwards.
        let mut saved_node = None;

        for _ in 0..levels {
            let node = self.node();
            let sibling = self.sibling();

            let internal_data = if self.pos.peek_last_bit() {
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

            let mut saved_node = None;

            match (NodeKind::of(&node), NodeKind::of(&sibling)) {
                (NodeKind::Terminator, NodeKind::Terminator) => {
                    self.up();
                    self.place_non_leaf(trie::TERMINATOR);
                }
                (NodeKind::Leaf, NodeKind::Terminator) => {
                    saved_node = Some(self.read_and_clear_current_leaf());
                    self.up();
                    self.place_non_leaf(trie::TERMINATOR);
                }
                (NodeKind::Terminator, NodeKind::Leaf) => {
                    if let Some((saved_leaf, saved_data)) = saved_node.take() {
                        self.place_leaf(saved_leaf, saved_data);

                        self.up();
                        self.place_non_leaf(H::hash_internal(&internal_data));
                    } else {
                        // if not, clear and save this leaf
                        saved_node = Some(self.read_and_clear_sibling_leaf());
                        self.up();
                        self.place_non_leaf(trie::TERMINATOR);
                    }
                }
                _ => {
                    self.up();
                    self.place_non_leaf(H::hash_internal(&internal_data));
                }
            }
        }

        if let Some((saved_leaf, saved_data)) = saved_node {
            // finished compacting - just write it back. will be fully relocated when
            // compacting is finished.
            self.place_leaf(saved_leaf, saved_data);
        }
    }
}
