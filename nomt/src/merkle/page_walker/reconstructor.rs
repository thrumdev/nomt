///! Special use of the page walker is done to reconstruct elided pages.
use bitvec::{order::Msb0, view::BitView};
use nomt_core::{
    page::DEPTH,
    page_id::PageId,
    trie::{KeyPath, ValueHash, TERMINATOR},
    trie_pos::TriePosition,
};

use crate::{
    merkle::{
        page_set::PageOrigin,
        page_walker::{PageSet, PageWalker},
    },
    page_cache::Page,
};

/// Reconstrct elided pages.
pub fn reconstruct_pages<H: nomt_core::hasher::NodeHasher>(
    page: &Page,
    page_id: PageId,
    position: TriePosition,
    page_set: &mut impl PageSet,
    ops: impl IntoIterator<Item = (KeyPath, ValueHash)>,
) -> impl Iterator<Item = (PageId, Page, u64)> {
    let subtree_root = page.node(position.node_index());

    let first_elided_page_id = page_id.child_page_id(position.child_page_index()).unwrap();
    let mut first_elided_page = page_set.fresh(&first_elided_page_id);
    first_elided_page.set_node(0, TERMINATOR);
    first_elided_page.set_node(1, TERMINATOR);
    page_set.insert(
        first_elided_page_id,
        first_elided_page.freeze(),
        PageOrigin::Reconstructed(0),
    );

    let mut page_walker = PageWalker::<H>::new_reconstructor(
        subtree_root,
        page_id.clone(),
        crate::metrics::Metrics::new(false),
    );

    let mut ops = ops.into_iter().peekable();

    let divisor_bit = (page_id.depth() + 1) * DEPTH;

    let left_subtree_ops: Vec<_> = std::iter::from_fn(|| {
        ops.next_if(|(key_path, _)| !key_path.view_bits::<Msb0>()[divisor_bit])
    })
    .collect();
    let mut left_subtree_position = position.clone();
    left_subtree_position.down(false);
    page_walker.advance_and_replace(page_set, left_subtree_position, left_subtree_ops);

    let right_subtree_ops = ops;
    let mut right_subtree_position = position.clone();
    right_subtree_position.down(true);
    page_walker.advance_and_replace(page_set, right_subtree_position, right_subtree_ops);

    let (root, reconstructed_pages) = page_walker.conclude_reconstructor();

    assert_eq!(root, subtree_root);

    // SAFETY: PageWlaker was initialized with the parent_page set to Some thus
    // no updated page is expected.
    reconstructed_pages.into_iter().map(|reconstructed_page| {
        (
            reconstructed_page.page_id,
            reconstructed_page.page.freeze(),
            reconstructed_page.leaves_counter,
        )
    })
}
