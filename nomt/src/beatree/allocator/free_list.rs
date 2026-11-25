use crate::{
    beatree::FREELIST_EMPTY,
    io::{self, page_pool::FatPage, PagePool, PAGE_SIZE},
};
use std::{collections::BTreeSet, fs::File};

use super::PageNumber;

const MAX_PNS_PER_PAGE: usize = (PAGE_SIZE - 6) / 4;

/// In-memory version of the free-list, which is stored on disk as a linked list of pages.
#[derive(Clone)]
pub struct FreeList {
    // head is last portion.
    portions: Vec<(PageNumber, Vec<PageNumber>)>,
    len: usize,
    // Databases written before `rewrite` may hold a fragmented list: the head page has exactly one
    // item and the second-to-last page one fewer than the maximum.
    fragmented: bool,
}

impl FreeList {
    fn new(portions: Vec<(PageNumber, Vec<PageNumber>)>) -> Self {
        let (len, fragmented) = len_and_fragmented(&portions);
        FreeList {
            portions,
            len,
            fragmented,
        }
    }

    pub fn read(
        page_pool: &PagePool,
        store_file: &File,
        free_list_head: Option<PageNumber>,
    ) -> anyhow::Result<FreeList> {
        let Some(mut free_list_pn) = free_list_head else {
            return Ok(FreeList::new(vec![]));
        };

        // restore free list from file
        let max_pn = (store_file.metadata()?.len() as usize / PAGE_SIZE) as u32;
        let mut free_list_portions = vec![];
        loop {
            if free_list_pn.is_nil() {
                break;
            }

            let page = io::read_page(page_pool, store_file, free_list_pn.0 as u64)?;

            let (prev, free_list) = decode_free_list_page(page, max_pn);
            free_list_portions.push((free_list_pn, free_list));
            free_list_pn = prev;
        }

        free_list_portions.reverse();

        Ok(FreeList::new(free_list_portions))
    }

    /// Get a set of all pages tracked in the free-list. This is all free pages plus all the
    /// free-list pages themselves.
    pub fn all_tracked_pages(&self) -> BTreeSet<PageNumber> {
        let pns = self
            .portions
            .iter()
            .map(|&(ref pn, ref pns)| std::iter::once(pn).chain(pns))
            .flatten()
            .copied();

        BTreeSet::from_iter(pns)
    }

    pub fn head_pn(&self) -> Option<PageNumber> {
        self.portions.last().map(|(head_pn, _)| head_pn).copied()
    }

    /// Get the length of the free-list.
    pub fn len(&self) -> usize {
        self.len
    }

    // All entries in the order `get_nth_pop` hands them out.
    fn pop_order(&self) -> Vec<PageNumber> {
        let mut order = Vec::with_capacity(self.len);
        for (_, pns) in self.portions.iter().rev() {
            order.extend(pns.iter().rev());
        }
        order
    }

    /// Rewrites the whole free-list in address order, starting after the last page used by this
    /// sync, so that the next sync's consecutive allocations land on nearby pages.
    ///
    /// `used` is the number of entries the sync allocated (its first `used` pops). The pages that
    /// held the previous free-list and the `freed` pages join the list; they stay untouched until
    /// the next sync. The new free-list pages are the next free entries in address order, or
    /// are bumped when there are not enough. Returns the pages to write.
    pub fn rewrite(
        &mut self,
        page_pool: &PagePool,
        used: usize,
        freed: Vec<PageNumber>,
        bump: &mut PageNumber,
    ) -> Vec<(PageNumber, FatPage)> {
        if used == 0 && freed.is_empty() {
            return vec![];
        }

        let order = self.pop_order();
        let remaining = &order[used..];

        let mut entries: Vec<PageNumber> = Vec::with_capacity(order.len() + freed.len());
        entries.extend(freed);
        entries.extend(self.portions.iter().map(|(pn, _)| *pn));

        // Take `taken` list pages from the remaining entries and bump the rest, such that the
        // list pages exactly hold the entries left over.
        let total = remaining.len() + entries.len();
        let pages_for = |taken: usize| (total - taken).div_ceil(MAX_PNS_PER_PAGE);
        let mut taken = 0;
        while taken < remaining.len() && pages_for(taken + 1) > taken {
            taken += 1;
        }
        let mut list_pages: Vec<PageNumber> = remaining[..taken].to_vec();
        for _ in taken..pages_for(taken) {
            list_pages.push(*bump);
            bump.0 += 1;
        }
        let cursor = remaining[..taken].last().or(order[..used].last()).copied();
        entries.extend_from_slice(&remaining[taken..]);

        entries.sort_unstable();
        if let Some(cursor) = cursor {
            let split = entries.partition_point(|pn| *pn <= cursor);
            entries.rotate_left(split);
        }

        // The last portion is the head and pops from its end, so lay the entries out reversed.
        entries.reverse();
        list_pages.sort_unstable();
        self.portions = list_pages
            .into_iter()
            .zip(entries.chunks(MAX_PNS_PER_PAGE))
            .map(|(pn, pns)| (pn, pns.to_vec()))
            .collect();
        self.len = entries.len();
        self.fragmented = false;

        let mut prev = FREELIST_EMPTY;
        self.portions
            .iter()
            .map(|(pn, pns)| {
                let page = encode_free_list_page(page_pool, prev, pns);
                prev = *pn;
                (*pn, page)
            })
            .collect()
    }

    /// Get the result of the nth pop on this free-list. Panics if n >= len. Runtime: O(1).
    pub fn get_nth_pop(&self, mut n: usize) -> PageNumber {
        let portions = &self.portions;
        let n_portions = portions.len();
        if self.fragmented {
            if n == 0 {
                portions[n_portions - 1].1[0]
            } else if n < MAX_PNS_PER_PAGE {
                portions[n_portions - 2].1[MAX_PNS_PER_PAGE - n - 1]
            } else {
                let portion_offset = 2 + (n / MAX_PNS_PER_PAGE);
                n = n % MAX_PNS_PER_PAGE;

                portions[n_portions - portion_offset].1[MAX_PNS_PER_PAGE - n - 1]
            }
        } else {
            let head_len = portions[n_portions - 1].1.len();
            if n < head_len {
                portions[n_portions - 1].1[head_len - n - 1]
            } else {
                n -= head_len;
                let portion_offset = 2 + (n / MAX_PNS_PER_PAGE);
                n = n % MAX_PNS_PER_PAGE;

                portions[n_portions - portion_offset].1[MAX_PNS_PER_PAGE - n - 1]
            }
        }
    }
}

fn len_and_fragmented(portions: &[(PageNumber, Vec<PageNumber>)]) -> (usize, bool) {
    match portions.last() {
        None => (0, false),
        Some((_, p)) if portions.len() > 1 && p.len() == 1 => {
            // account for fragmentation.
            let penultimate_len = portions[portions.len() - 2].1.len();
            let len = (portions.len() - 2) * MAX_PNS_PER_PAGE + penultimate_len + 1;

            (len, penultimate_len != MAX_PNS_PER_PAGE)
        }
        Some((_, p)) => ((portions.len() - 1) * MAX_PNS_PER_PAGE + p.len(), false),
    }
}

// returns the previous PageNumber and all the PageNumbers stored in the free list page
fn decode_free_list_page(page: FatPage, max_pn: u32) -> (PageNumber, Vec<PageNumber>) {
    let free_list_page_view = FreeListPageRef(&page[..]);

    let prev = free_list_page_view.prev_pn();
    let item_count = free_list_page_view.item_count();
    let mut free_list = vec![];
    for i in 0..item_count as usize {
        let pn = free_list_page_view.item(i);
        // Ensure each PageNumber falls within the file bounds.
        assert!(pn.0 < max_pn);
        free_list.push(pn);
    }

    (prev, free_list)
}

fn encode_free_list_page(page_pool: &PagePool, prev: PageNumber, pns: &[PageNumber]) -> FatPage {
    let mut page = page_pool.alloc_fat_page();

    {
        let mut e = FreeListPageMut(&mut page[..]);
        e.set_prev_pn(prev);
        e.set_item_count(pns.len() as u16);
        for (i, pn) in pns.into_iter().enumerate() {
            e.set_item(i, *pn);
        }
    }

    page
}

// A free page is laid out in the following form:
// + prev free page : u32
// + item_count : u16
// + free pages : [u32; item_count]
struct FreeListPageRef<'a>(&'a [u8]);
struct FreeListPageMut<'a>(&'a mut [u8]);

impl<'a> FreeListPageRef<'a> {
    fn prev_pn(&self) -> PageNumber {
        PageNumber(u32::from_le_bytes(self.0[0..4].try_into().unwrap()))
    }

    fn item_count(&self) -> u16 {
        u16::from_le_bytes(self.0[4..6].try_into().unwrap())
    }

    fn item(&self, i: usize) -> PageNumber {
        let start = 6 + i * 4;
        PageNumber(u32::from_le_bytes(
            self.0[start..start + 4].try_into().unwrap(),
        ))
    }
}

impl<'a> FreeListPageMut<'a> {
    fn set_prev_pn(&mut self, pn: PageNumber) {
        self.0[0..4].copy_from_slice(&pn.0.to_le_bytes());
    }

    fn set_item_count(&mut self, count: u16) {
        self.0[4..6].copy_from_slice(&count.to_le_bytes());
    }

    fn set_item(&mut self, i: usize, pn: PageNumber) {
        let start = 6 + i * 4;
        self.0[start..start + 4].copy_from_slice(&pn.0.to_le_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::{decode_free_list_page, FreeList, PageNumber, PagePool, MAX_PNS_PER_PAGE};
    use crate::beatree::FREELIST_EMPTY;
    use std::collections::BTreeSet;

    fn pns(range: std::ops::Range<u32>) -> Vec<PageNumber> {
        range.map(PageNumber).collect()
    }

    fn assert_nth_pop_matches_layout(free_list: &FreeList) {
        let predicted = (0..free_list.len())
            .map(|i| free_list.get_nth_pop(i))
            .collect::<Vec<_>>();
        assert_eq!(predicted, free_list.pop_order());
    }

    #[test]
    fn clean_nth_pop() {
        let free_list = FreeList::new(vec![
            (PageNumber(1), pns(1000..1000 + MAX_PNS_PER_PAGE as u32)),
            (PageNumber(2), pns(5000..5000 + MAX_PNS_PER_PAGE as u32)),
            (
                PageNumber(3),
                pns(10000..10000 + MAX_PNS_PER_PAGE as u32 / 2),
            ),
        ]);
        assert!(!free_list.fragmented);
        assert_nth_pop_matches_layout(&free_list);
    }

    #[test]
    fn fragmented_nth_pop() {
        let free_list = FreeList::new(vec![
            (PageNumber(1), pns(1000..1000 + MAX_PNS_PER_PAGE as u32)),
            (PageNumber(4), pns(5000..5000 + MAX_PNS_PER_PAGE as u32)),
            (
                PageNumber(5),
                pns(10000..10000 + MAX_PNS_PER_PAGE as u32 - 1),
            ),
            (PageNumber(6), vec![PageNumber(7)]),
        ]);
        assert!(free_list.fragmented);
        assert_eq!(free_list.len(), MAX_PNS_PER_PAGE * 3);
        assert_nth_pop_matches_layout(&free_list);
    }

    // Runs `rewrite` and checks the result against the free list it started from.
    fn check_rewrite(mut free_list: FreeList, used: usize, freed: Vec<PageNumber>) -> FreeList {
        let before = free_list.pop_order();
        let old_list_pages: Vec<PageNumber> = free_list.portions.iter().map(|p| p.0).collect();
        let bump_before = PageNumber(1_000_000);
        let mut bump = bump_before;

        let page_pool = PagePool::new();
        let written = free_list.rewrite(&page_pool, used, freed.clone(), &mut bump);

        // The written pages encode exactly the new portions, chained in order.
        assert_eq!(written.len(), free_list.portions.len());
        let mut prev = FREELIST_EMPTY;
        for ((pn, page), (portion_pn, portion)) in written.iter().zip(&free_list.portions) {
            assert_eq!(pn, portion_pn);
            let (decoded_prev, decoded) = decode_free_list_page(page.clone(), u32::MAX);
            assert_eq!(decoded_prev, prev);
            assert_eq!(&decoded, portion);
            prev = *pn;
        }

        // Every portion but the head is full, so the list is not fragmented.
        let (head, rest) = free_list.portions.split_last().unwrap();
        assert!(!head.1.is_empty());
        assert!(rest.iter().all(|p| p.1.len() == MAX_PNS_PER_PAGE));
        assert!(!free_list.fragmented);
        assert_nth_pop_matches_layout(&free_list);

        // No page is lost or duplicated.
        let list_pages: BTreeSet<PageNumber> = free_list.portions.iter().map(|p| p.0).collect();
        let entries: BTreeSet<PageNumber> = free_list.pop_order().into_iter().collect();
        assert_eq!(entries.len(), free_list.len());
        assert!(list_pages.is_disjoint(&entries));
        let bumped: BTreeSet<PageNumber> = (bump_before.0..bump.0).map(PageNumber).collect();
        let expected: BTreeSet<PageNumber> = before[used..]
            .iter()
            .chain(&freed)
            .chain(&old_list_pages)
            .copied()
            .chain(bumped.iter().copied())
            .collect();
        assert_eq!(&list_pages | &entries, expected);

        // The new list pages are written this sync, so they must not be anything the committed
        // state still references (old list pages, freed pages) or this sync's allocations.
        let fresh: BTreeSet<PageNumber> = before[used..].iter().copied().chain(bumped).collect();
        assert!(list_pages.is_subset(&fresh));

        // Pops sweep upwards through the file, wrapping around at most once.
        let order = free_list.pop_order();
        assert!(order.windows(2).filter(|w| w[0] > w[1]).count() <= 1);

        free_list
    }

    #[test]
    fn rewrite_nothing_to_do() {
        let mut free_list = FreeList::new(vec![(PageNumber(1), pns(2..10))]);
        let written = free_list.rewrite(&PagePool::new(), 0, vec![], &mut PageNumber(100));
        assert!(written.is_empty());
        assert_eq!(free_list.portions, vec![(PageNumber(1), pns(2..10))]);
    }

    #[test]
    fn rewrite_empty_list_bumps_list_page() {
        let free_list = check_rewrite(FreeList::new(vec![]), 0, pns(5..8));
        assert_eq!(
            free_list.portions,
            vec![(PageNumber(1_000_000), pns(5..8).into_iter().rev().collect())]
        );
    }

    #[test]
    fn rewrite_takes_list_pages_from_free_entries() {
        let free_list = FreeList::new(vec![(PageNumber(1), pns(2..10))]);
        let free_list = check_rewrite(free_list, 3, pns(20..22));
        // used: 9, 8, 7. The next entry in pop order, 6, holds the new list.
        assert_eq!(free_list.portions.len(), 1);
        assert_eq!(free_list.portions[0].0, PageNumber(6));
        // Pops resume after the cursor at 6 and wrap around.
        assert_eq!(
            free_list.pop_order(),
            [20, 21, 1, 2, 3, 4, 5].map(PageNumber).to_vec()
        );
    }

    #[test]
    fn rewrite_all_used() {
        let free_list = FreeList::new(vec![
            (PageNumber(1), pns(100..100 + MAX_PNS_PER_PAGE as u32)),
            (PageNumber(2), pns(5000..5010)),
        ]);
        let len = free_list.len();
        let free_list = check_rewrite(free_list, len, vec![]);
        // Only the two old list pages remain free; the new list page is bumped.
        assert_eq!(free_list.pop_order(), pns(1..3));
    }

    #[test]
    fn rewrite_multi_page_shuffled() {
        // A deterministic shuffle of three and a half pages of entries.
        let n = (MAX_PNS_PER_PAGE * 7 / 2) as u32;
        let mut all: Vec<PageNumber> = (0..n).map(|i| PageNumber(10 + (i * 7919) % n)).collect();
        let freed = all.split_off(all.len() - 500);
        let portions = all
            .chunks(MAX_PNS_PER_PAGE)
            .enumerate()
            .map(|(i, c)| (PageNumber(n + 100 + i as u32), c.to_vec()))
            .collect();
        let free_list = FreeList::new(portions);
        let len = free_list.len();
        let free_list = check_rewrite(free_list, len / 3, freed);
        let free_list = check_rewrite(free_list, 10, pns(n + 200..n + 300));
        check_rewrite(free_list, 0, pns(n + 300..n + 301));
    }

    #[test]
    fn rewrite_fragmented() {
        let free_list = FreeList::new(vec![
            (PageNumber(1), pns(100..100 + MAX_PNS_PER_PAGE as u32)),
            (PageNumber(4), pns(5000..5000 + MAX_PNS_PER_PAGE as u32 - 1)),
            (PageNumber(6), vec![PageNumber(7)]),
        ]);
        assert!(free_list.fragmented);
        check_rewrite(free_list, 5, pns(9000..9003));
    }
}
