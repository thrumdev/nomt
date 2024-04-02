use std::{cell::RefCell, cmp};

use crate::page_cache::{Page, PageCache, PagePromise};
use bitvec::prelude::*;
use nomt_core::{
    page::DEPTH,
    page_id::{PageId, PageIdsIterator},
    trie::{self, KeyPath, Node},
};

/// The breadth of the prefetch request.
///
/// The number of items we want to request in a single batch.
const PREFETCH_N: usize = 7;

pub struct PageCacheCursor {
    root: Node,
    path: KeyPath,
    depth: u8,
    /// The cached page and it's ID.
    ///
    /// Justification for the interior mutability is to use &self in [`Self::node`].
    cached_page: RefCell<Option<(PageId, Page)>>,
}

impl PageCacheCursor {
    /// Creates the cursor pointing at the given root of the trie.
    pub fn at_root(root: Node) -> Self {
        Self {
            root,
            path: KeyPath::default(),
            depth: 0,
            cached_page: RefCell::new(None),
        }
    }

    /// Moves the cursor back to the root.
    pub fn reset(&mut self) {
        self.depth = 0;
        self.path = KeyPath::default();
    }

    /// Moves the cursor to the given [`KeyPath`].
    ///
    /// Moving the cursor using this function would be more efficient than using the navigation
    /// functions such as [`Self::down_left`] and [`Self::down_right`] due to leveraging warmup
    /// hints.
    ///
    /// After returning of this function, the cursor is positioned either at the given key path or
    /// at the closest key path that is on the way to the given key path.
    pub fn seek(&mut self, dest: KeyPath, page_cache: &PageCache) {
        self.reset();
        if !trie::is_internal(&self.root) {
            // The root either a leaf or the terminator. In either case, we can't navigate anywhere.
            return;
        }
        let mut ppf = PagePrefetcher::new(dest);
        let mut page = Page::Nil;
        for bit in dest.view_bits::<Msb0>().iter().by_vals() {
            let node_index = if self.depth as usize % DEPTH == 0 {
                page = match ppf.next(page_cache) {
                    Some(p) => p.wait(),
                    None => {
                        panic!("reached the end of the prefetcher without finding the terminal")
                    }
                };
                pick(bit, child_node_indices(BitSlice::empty()))
            } else {
                let path = last_page_path(&self.path, self.depth);
                pick(bit, child_node_indices(path))
            };
            let node = page.node(node_index);
            if trie::is_internal(&node) {
                self.path
                    .view_bits_mut::<Msb0>()
                    .set(self.depth as usize, bit);
                self.depth += 1;
            } else {
                break;
            }
        }

        fn pick(bit: bool, (lhs, rhs): (usize, usize)) -> usize {
            if !bit {
                lhs
            } else {
                rhs
            }
        }
    }

    /// Go up the trie by `d` levels.
    ///
    /// The cursor will not go beyond the root. If the cursor's location was sound before the call,
    /// it will be sound after the call.
    pub fn up(&mut self, d: u8) {
        let d = cmp::min(self.depth, d);
        self.depth -= d;
    }

    /// Traverse to the left child of this node.
    ///
    /// The assumption is that the cursor is pointing to the internal node and it's the
    /// responsibility of the caller to ensure that. If violated,the cursor's location will be
    /// invalid.
    pub fn down_left(&mut self) {
        self.path
            .view_bits_mut::<Msb0>()
            .set(self.depth as usize, false);
        self.depth += 1;
    }

    /// Traverse to the right child of this node.
    ///
    /// The assumption is that the cursor is pointing to the internal node and it's the
    /// responsibility of the caller to ensure that. If violated,the cursor's location will be
    /// invalid.
    pub fn down_right(&mut self) {
        self.path
            .view_bits_mut::<Msb0>()
            .set(self.depth as usize, true);
        self.depth += 1;
    }

    /// Returns the current location of the cursor, represented as a tuple of the key path and the
    /// depth.
    pub fn location(&self) -> (KeyPath, u8) {
        (self.path, self.depth)
    }

    /// Returns the node at the current location of the cursor.
    pub fn node(&self, page_cache: &PageCache) -> Node {
        if self.depth == 0 {
            self.root
        } else {
            // Calculate the page ID of the current page.
            let cur_page_id = PageIdsIterator::new(&self.path)
                .last()
                .expect("the cursor is not at the root");
            // Check if the page is already cached locally by checking it's page ID. If it's not,
            // retrieve the page from the cache and update the cache.
            let mut cached_page_guard = self.cached_page.borrow_mut();
            let page = match &*cached_page_guard {
                Some((cached_page_id, cached_page)) => {
                    if *cached_page_id == cur_page_id {
                        cached_page.clone()
                    } else {
                        let page = page_cache.retrieve(cur_page_id).wait();
                        *cached_page_guard = Some((cur_page_id, page.clone()));
                        page
                    }
                }
                _ => {
                    let page = page_cache.retrieve(cur_page_id).wait();
                    *cached_page_guard = Some((cur_page_id, page.clone()));
                    page
                }
            };
            // Now having the page, we can extract the node from it.
            let page_path = last_page_path(&self.path, self.depth);
            let node_index = node_index(page_path);
            page.node(node_index)
        }
    }
}

// extract the relevant portion of the key path to the last page. panics on empty path.
fn last_page_path(total_path: &KeyPath, total_depth: u8) -> &BitSlice<u8, Msb0> {
    let prev_page_end = ((total_depth as usize - 1) / DEPTH) * DEPTH;
    &total_path.view_bits::<Msb0>()[prev_page_end..total_depth as usize]
}

// Transform a bit-path to the indices of the two child positions in a page.
//
// The expected length of the page path is between 0 and `DEPTH - 1`, inclusive. All bits beyond
// `DEPTH - 1` are ignored.
fn child_node_indices(page_path: &BitSlice<u8, Msb0>) -> (usize, usize) {
    if page_path.is_empty() {
        return (0, 1);
    }
    let depth = core::cmp::min(DEPTH - 1, page_path.len());

    // parent is at (2^depth - 2) + as_uint(parent)
    // children are at (2^(depth+1) - 2) + as_uint(parent)*2 + (0 or 1)
    let base = (1 << depth + 1) - 2 + 2 * page_path[..depth].load_be::<usize>();
    (base, base + 1)
}

// Transform a bit-path to an index in a page.
//
// The expected length of the page path is between 1 and `DEPTH`, inclusive. A length of 0 returns
// 0 and all bits beyond `DEPTH` are ignored.
fn node_index(page_path: &BitSlice<u8, Msb0>) -> usize {
    let depth = core::cmp::min(DEPTH, page_path.len());

    if depth == 0 {
        0
    } else {
        // each node is stored at (2^depth - 2) + as_uint(path)
        (1 << depth) - 2 + page_path[..depth].load_be::<usize>()
    }
}

struct PagePrefetcher {
    /// The page IDs of the destination key path in reverse order.
    rev_page_ids: Vec<PageId>,
    /// The promises of the pages that are currently being fetched. The promises are laid out in
    /// such a way, that [`Vec::pop`] would return the next page to be processed.
    ///
    /// This vector is at most of size [`PREFETCH_N`].
    page_promises: Vec<PagePromise>,
}

impl PagePrefetcher {
    fn new(dest: KeyPath) -> Self {
        let rev_page_ids = {
            // Chop the destination key path into the corresponding page IDs and reverse them so
            // that we can `pop`.
            let mut x = PageIdsIterator::new(&dest).collect::<Vec<_>>();
            x.reverse();
            x
        };
        Self {
            rev_page_ids,
            page_promises: Vec::new(),
        }
    }

    fn next(&mut self, page_cache: &PageCache) -> Option<PagePromise> {
        if self.page_promises.is_empty() {
            // The prefetch hasn't started yet or we consumed all the promises. We need to initiate
            // a new prefetch.
            if self.rev_page_ids.is_empty() {
                // No pages left to prefetch.
                return None;
            }
            let n = cmp::min(PREFETCH_N, self.rev_page_ids.len());
            for _ in 0..n {
                // Unwrap: pop must succeed since the n is clamped to the length of the vector.
                let page_id = self.rev_page_ids.pop().unwrap();
                self.page_promises.push(page_cache.retrieve(page_id));
            }
            self.page_promises.reverse();
        }
        self.page_promises.pop()
    }
}
