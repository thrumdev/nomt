//! Seek for a path in the trie, pre-fetching pages as necessary.

use crate::{
    page_cache::{self, Page, PageCache},
    page_region::PageRegion,
    rw_pass_cell::ReadPass,
};
use bitvec::prelude::*;
use nomt_core::{
    page::DEPTH,
    page_id::{PageId, PageIdsIterator, ROOT_PAGE_ID},
    trie::{self, KeyPath, Node},
    trie_pos::TriePosition,
};

/// Modes for seeking to a key path.
#[derive(Debug, Clone, Copy)]
pub struct SeekOptions {
    /// Retrieve the pages with the child location of any sibling nodes which are also leaves.
    ///
    /// This should be used when preparing to delete a key, which can cause leaf nodes to be
    /// relocated.
    pub retrieve_sibling_leaf_children: bool,
    /// Record siblings.
    pub record_siblings: bool,
}

/// The results of a seek operation.
#[derive(Debug, Clone)]
pub struct Seek {
    /// The siblings encountered along the path, in ascending order by depth.
    ///
    /// This is empty if `SeekOptions.record_siblings` was false.
    pub siblings: Vec<Node>,
    /// The terminal node encountered.
    pub terminal: Option<trie::LeafData>,
    /// The trie position encountered.
    pub position: TriePosition,
    /// The page ID where the trie position is located. None if at root.
    pub page_id: Option<PageId>,
}

/// A [`Seeker`] can be used to seek for keys in the trie.
pub struct Seeker {
    cache: PageCache,
    root: Node,
}

impl Seeker {
    /// Create a new Seeker, given the cache, page read pass, and a root node.
    pub fn new(root: Node, cache: PageCache) -> Self {
        Seeker { cache, root }
    }

    fn read_leaf_children(
        &self,
        trie_pos: &TriePosition,
        current_page: Option<&(PageId, Page)>,
        read_pass: &ReadPass<PageRegion>,
    ) -> trie::LeafData {
        let (page, _, children) = page_cache::locate_leaf_data(trie_pos, current_page, |page_id| {
            self.cache.retrieve_sync(page_id, false)
        });
        trie::LeafData {
            key_path: page.node(&read_pass, children.left()),
            value_hash: page.node(&read_pass, children.right()),
        }
    }

    fn down(
        &self,
        bit: bool,
        pos: &mut TriePosition,
        cur_page: &mut Option<(PageId, Page)>,
        read_pass: &ReadPass<PageRegion>,
    ) -> (Node, Node) {
        if pos.depth() as usize % DEPTH == 0 {
            // attempt to load next page if we are at the end of our previous page or the root.
            // UNWRAP: page index is valid, nodes never fall beyond the 42nd page.
            let page_id = match cur_page {
                None => ROOT_PAGE_ID,
                Some((ref id, _)) => id
                    .child_page_id(pos.child_page_index())
                    .expect("Pages do not go deeper than the maximum layer, 42"),
            };

            *cur_page = Some((page_id.clone(), self.cache.retrieve_sync(page_id, false)));
        }
        pos.down(bit);

        // UNWRAP: safe, was just set if at root
        let page = &cur_page.as_ref().unwrap().1;

        (
            page.node(&read_pass, pos.node_index()),
            page.node(&read_pass, pos.sibling_index()),
        )
    }

    /// Seek to the given [`KeyPath`], loading the terminal node, all siblings on the path, and caching
    /// all pages.
    ///
    /// This returns a [`Seek`] object encapsulating the results of the seek.
    pub fn seek(
        &self,
        dest: KeyPath,
        options: SeekOptions,
        read_pass: &ReadPass<PageRegion>,
    ) -> Seek {
        /// The breadth of the prefetch request.
        ///
        /// The number of items we want to request in a single batch.
        const PREFETCH_N: usize = 7;

        let mut result = Seek {
            siblings: Vec::with_capacity(if options.record_siblings { 32 } else { 0 }),
            terminal: None,
            position: TriePosition::new(),
            page_id: None,
        };

        let mut trie_pos = TriePosition::new();
        let mut page: Option<(PageId, Page)> = None;

        if !trie::is_internal(&self.root) {
            // fast path: don't pre-fetch when trie is just a root.
            if trie::is_leaf(&self.root) {
                result.terminal = Some(self.read_leaf_children(&trie_pos, None, read_pass));
            };

            return result;
        }

        let mut ppf = PageIdsIterator::new(dest);

        let mut sibling = trie::TERMINATOR;
        let mut cur_node = self.root;

        let mut pending_prefetches = Vec::with_capacity(PREFETCH_N);
        for bit in dest.view_bits::<Msb0>().iter().by_vals() {
            if !trie::is_internal(&cur_node) {
                if trie::is_leaf(&cur_node) {
                    let leaf_data = self.read_leaf_children(&trie_pos, page.as_ref(), read_pass);
                    if trie_pos.depth() as usize % DEPTH == 0 {
                        // leaf data page was needed.
                        let _ = pending_prefetches.pop();
                    }
                    assert!(leaf_data
                        .key_path
                        .view_bits::<Msb0>()
                        .starts_with(&dest.view_bits::<Msb0>()[..trie_pos.depth() as usize]));

                    result.terminal = Some(leaf_data);
                };

                for page_id in pending_prefetches {
                    self.cache.cancel_prepopulate(page_id)
                }

                result.position = trie_pos;
                result.page_id = page.map(|(id, _)| id);
                return result;
            }
            if trie_pos.depth() as usize % DEPTH == 0 {
                if trie_pos.depth() as usize % PREFETCH_N == 0 {
                    for _ in 0..PREFETCH_N {
                        let page_id = match ppf.next() {
                            Some(page) => page,
                            None => break,
                        };
                        pending_prefetches.push(page_id.clone());
                        self.cache.prepopulate(page_id);
                    }
                    pending_prefetches.reverse();
                }

                // next step `down` after this if block relies on this page having been fetched.
                let _ = pending_prefetches.pop();

                if let (&Some((ref id, _)), true, true) = (
                    &page,
                    options.retrieve_sibling_leaf_children,
                    trie::is_leaf(&sibling),
                ) {
                    // sibling is a leaf and at the end of the (non-root) page.
                    // initiate a load of the sibling's page.
                    let child_page_id = id
                        .child_page_id(trie_pos.sibling_child_page_index())
                        .expect("Pages do not go deeper than the maximum layer, 42");
                    // async; just warm up.
                    let _ = self.cache.prepopulate(child_page_id);
                }
            }

            let (new_node, new_sibling) = self.down(bit, &mut trie_pos, &mut page, read_pass);
            cur_node = new_node;
            sibling = new_sibling;

            if options.record_siblings {
                result.siblings.push(new_sibling);
            }
        }

        panic!("no terminal along path {}", dest.view_bits::<Msb0>());
    }
}
