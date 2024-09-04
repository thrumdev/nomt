use crate::{
    beatree::FREELIST_EMPTY,
    io::{self, Page, PAGE_SIZE},
};
use std::{collections::BTreeSet, fs::File};

use super::PageNumber;

const MAX_PNS_PER_PAGE: usize = (PAGE_SIZE - 6) / 4;

/// In-memory version of FreeList which provides a way to decode from and encode into pages
/// and provide two primitives, one for extracting free pages from the list and one to append
/// freed pages.
///
/// Pages that are freed due to the fetch of free pages are automatically added back during the encode phase,
/// which also covers the addition of new free pages.
#[derive(Clone)]
pub struct FreeList {
    // head is last portion.
    portions: Vec<(PageNumber, Vec<PageNumber>)>,
    // Becomes true if something is being popped, false otherwise
    pop: bool,
    released_portions: Vec<PageNumber>,
}

impl FreeList {
    pub fn read(store_file: &File, free_list_head: Option<PageNumber>) -> FreeList {
        let Some(mut free_list_pn) = free_list_head else {
            return FreeList {
                portions: vec![],
                pop: false,
                released_portions: vec![],
            };
        };

        // restore free list from file
        let mut free_list_portions = vec![];
        loop {
            if free_list_pn.is_nil() {
                break;
            }

            let page = io::read_page(store_file, free_list_pn.0 as u64).unwrap();

            let (prev, free_list) = decode_free_list_page(page);
            free_list_portions.push((free_list_pn, free_list));
            free_list_pn = prev;
        }

        free_list_portions.reverse();

        FreeList {
            pop: false,
            portions: free_list_portions,
            released_portions: vec![],
        }
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

    pub fn pop(&mut self) -> Option<PageNumber> {
        let pn;
        let head_empty = {
            let head = self.portions.last_mut()?;
            // UNWRAP: free-list portions are always non-empty.
            pn = head.1.pop().unwrap();
            self.pop = true;

            if head.1.is_empty() {
                Some(head.0)
            } else {
                None
            }
        };

        if let Some(prev_head_pn) = head_empty {
            let _ = self.portions.pop();
            self.released_portions.push(prev_head_pn);
        }

        Some(pn)
    }

    /// Apply the changes resulting from calls to `pop()` and pushing all the `to_push` page numbers
    /// to the free list. This returns a vector of pages to write, each tagged with the page number
    /// it should be written at.
    ///
    /// FreeList Pages are stored in the file using a Copy-On-Write (COW) approach.
    ///
    /// This function only applies `pop` invocations since the last call to this function,
    /// creating an encoded version of the updated free list state.
    ///
    /// The FreeList is conceptually a paginated stack. There are some additional constraints due
    /// to the copy-on-write mechanics: all touched pages must have their previous page
    /// number pushed at the end, and new page numbers are determined by popping from the list.
    /// Additionally, to maintain crash-consistency, we do not reuse any page which is being
    /// added afresh to the list within this function. Pops and pushes are therefore heavily
    /// intertwined. Pushes can cause pops. Pops reduce the length of the list, making pops
    /// possibly unnecessary. Pops can also cause further pops when they touch new pages.
    ///
    /// These restrictions add some code complexity, as well as the possibility of "fragmentation".
    /// This fragmentation is minor and takes a very specific form: the second page will have one
    /// fewer items than the maximum and the head page will have exactly one item. This occurs only
    /// when the last item pushed would be the first in a new page and the pop to allocate that
    /// new page dirties a previously untouched page. Needless to say, it is rare, something like
    /// 1-in-a-million (1022^2) and of near-zero consequence if forced somehow by an adversary.
    ///
    /// The next call to `commit` will eliminate any previous fragmentation, but may result in
    /// fragmentation itself.
    ///
    /// O(n) in the number of pops / pushes.
    pub fn commit(
        &mut self,
        mut to_push: Vec<PageNumber>,
        bump: &mut PageNumber,
    ) -> Vec<(PageNumber, Box<Page>)> {
        // No changes were made
        if !self.pop && to_push.is_empty() {
            return vec![];
        }

        // append the released free list pages
        to_push.extend(self.released_portions.drain(..));

        if self.pop && to_push.is_empty() {
            // note: empty vec when head is empty.
            return self.encode_head().into_iter().collect();
        }

        self.pop = false;

        let new_pages = self.preallocate(&mut to_push, bump);
        self.push_and_encode(&to_push, new_pages)
    }

    // determines the exact number of pops and bumps which are needed in order to fulfill the
    // request. also schedules pushing of all touched pages' previous page numbers.
    fn preallocate(
        &mut self,
        to_push: &mut Vec<PageNumber>,
        bump: &mut PageNumber,
    ) -> Vec<PageNumber> {
        let mut new_pages = Vec::new();

        // allocate a new page for rewriting the head (if any).
        let mut new_full_portion = true;
        let mut i;
        if let Some(pn) = self.pop() {
            // check if this freed a page and handle that
            match self.released_portions.pop() {
                Some(x) => {
                    if let Some((new_head_pn, new_head_pns)) = self
                        .portions
                        .last_mut()
                        .filter(|p| p.1.len() == MAX_PNS_PER_PAGE - 1)
                    {
                        // fix up fragmentation. only the second-to-last page can be fragmented,
                        // and it is. use this page to rewrite the second-to-last and position
                        // the cursor at the end of this page instead.
                        to_push.push(*new_head_pn);
                        to_push.push(x);
                        *new_head_pn = pn;
                        new_full_portion = false;
                        i = MAX_PNS_PER_PAGE - new_head_pns.len();
                    } else {
                        // previous page was full or doesn't exist.
                        to_push.push(x);
                        new_pages.push(pn);

                        // new page has already been allocated for this batch.
                        i = MAX_PNS_PER_PAGE;
                    }
                }
                None => {
                    new_full_portion = false;

                    // UNWRAP: protected since pop succeeded but head was not released.
                    let (head_pn, head_pns) = self.portions.last_mut().unwrap();
                    to_push.push(*head_pn);
                    *head_pn = pn;

                    // head is partially full. position cursor at the number of items needed to
                    // fill it.
                    i = MAX_PNS_PER_PAGE - head_pns.len();
                }
            }
        } else {
            // nothing popped, free list was totally empty.
            // defer to loop to create the page for the next section.
            i = 0;
        };

        // loop invariant: free list len + i is always divisible by MAX_PNS_PER_PAGE.
        // therefore, the loop condition asks "do we still have items beyond the last page?"
        // while accounting for pops and pushes within the loop.
        while i < to_push.len() {
            if let Some((ref mut head_pn, ref mut head_pns)) =
                self.portions.last_mut().filter(|_| new_full_portion)
            {
                // just popped into a new portion (which we will edit). prepare it for rewrite,
                // because we are about to pop from it and alter it.
                to_push.push(*head_pn);

                // UNWRAP: new_full_portion guarantees that the new portion is full.
                *head_pn = head_pns.pop().unwrap();

                new_full_portion = false;

                // pop reduces free list length, we adjust i to compensate.
                // however, we must unconditionally rewrite this page even if i equals the previous
                // value of to_push.len(). That is because if we didn't pop we wouldn't have enough
                // pages.
                // this is inescapable and is what causes fragmentation.
                i += 1;

                continue;
            }

            // loop invariant: from this point on, the head page (if any) always has a head_pn which
            // was itself drawn from the free-list.
            match self.pop() {
                Some(pn) => {
                    new_pages.push(pn);
                    // pop reduces free list length, we adjust i to compensate
                    i += 1;

                    if let Some(released) = self.released_portions.pop() {
                        new_full_portion = true;
                        // note: this is the PN we took out of the new portion; it's fresh.
                        //
                        // if i now equals to_push.len(), we have a forced fragmentation.
                        new_pages.push(released);
                    }
                }
                None => {
                    new_pages.push(*bump);
                    bump.0 += 1;
                }
            }

            // jump 'i' to the end of the next page.
            i += MAX_PNS_PER_PAGE;
        }

        new_pages
    }

    fn push_and_encode(
        &mut self,
        to_push: &[PageNumber],
        new_pages: Vec<PageNumber>,
    ) -> Vec<(PageNumber, Box<Page>)> {
        let mut encoded = Vec::new();
        let mut new_pages = new_pages.into_iter().peekable();
        for (i, pn) in to_push.iter().cloned().enumerate() {
            // the second condition is checking for the fragmentation described in the commit
            // doc comment. fragmentation can only occur in the second page and the head page
            // has a single item.
            let head_full = self
                .portions
                .last()
                .map_or(true, |h| h.1.len() == MAX_PNS_PER_PAGE);

            let fragmentation = self
                .portions
                .last()
                .map_or(false, |h| h.1.len() == MAX_PNS_PER_PAGE - 1)
                && new_pages.peek().is_some()
                && i + 1 == to_push.len();

            if head_full || fragmentation {
                encoded.extend(self.encode_head());
                // UNWRAP: we've always allocated enough PNs for all appended PNs.
                let new_head_pn = new_pages.next().unwrap();
                self.portions.push((new_head_pn, Vec::new()));
            }

            self.push(pn);
        }

        assert!(new_pages.next().is_none());

        encoded.extend(self.encode_head());
        encoded
    }

    fn push(&mut self, pn: PageNumber) {
        // UNWRAP: `push` is only called when head is not full.
        let head = self.portions.last_mut().unwrap();
        assert!(head.1.len() < MAX_PNS_PER_PAGE);
        head.1.push(pn);
    }

    fn encode_head(&self) -> Option<(PageNumber, Box<Page>)> {
        if let Some((head_pn, head_pns)) = self.portions.last() {
            let prev_pn = self
                .portions
                .len()
                .checked_sub(2)
                .map_or(FREELIST_EMPTY, |i| self.portions[i].0);

            Some((*head_pn, encode_free_list_page(prev_pn, &head_pns)))
        } else {
            None
        }
    }
}

// returns the previous PageNumber and all the PageNumbers stored in the free list page
//
// A free page is laid out in the following form:
// + prev free page : u32
// + item_count : u16
// + free pages : [u32; item_count]
fn decode_free_list_page(page: Box<Page>) -> (PageNumber, Vec<PageNumber>) {
    let prev = {
        let mut buf = [0u8; 4];
        buf.copy_from_slice(&page[0..4]);
        PageNumber(u32::from_le_bytes(buf))
    };

    let item_count = {
        let mut buf = [0u8; 2];
        buf.copy_from_slice(&page[4..6]);
        u16::from_le_bytes(buf)
    };

    let mut free_list = vec![];
    for i in 0..item_count as usize {
        let page_number = {
            let mut buf = [0u8; 4];

            let start = 6 + i * 4;
            buf.copy_from_slice(&page[start..start + 4]);

            u32::from_le_bytes(buf)
        };

        free_list.push(PageNumber(page_number));
    }

    (prev, free_list)
}

fn encode_free_list_page(prev: PageNumber, pns: &[PageNumber]) -> Box<Page> {
    let mut page = Page::zeroed();

    page[0..4].copy_from_slice(&prev.0.to_le_bytes());
    page[4..6].copy_from_slice(&(pns.len() as u16).to_le_bytes());

    for (i, pn) in pns.into_iter().enumerate() {
        let start = 6 + i * 4;
        page[start..start + 4].copy_from_slice(&pn.0.to_le_bytes());
    }

    Box::new(page)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pop_into_next_portion_one_page_needed() {
        let mut free_list = FreeList {
            portions: vec![(PageNumber(1), vec![PageNumber(2)])],
            pop: false,
            released_portions: Vec::new(),
        };

        // expected order of events:
        //   1. first (2) is popped for the new head.
        //   2. This exhausts the head. We use (2) for a new head and push (1) to the end.
        let result = free_list.commit(
            (3..).take(1).map(PageNumber).collect::<Vec<_>>(),
            &mut PageNumber(10000),
        );

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, PageNumber(2));

        assert_eq!(free_list.portions.len(), 1);
        assert_eq!(free_list.portions[0].0, PageNumber(2));
        assert_eq!(free_list.portions[0].1, vec![PageNumber(3), PageNumber(1)]);
    }

    #[test]
    fn pop_into_next_portion_fragmentation() {
        let mut free_list = FreeList {
            portions: vec![(PageNumber(1), vec![PageNumber(2), PageNumber(3)])],
            pop: false,
            released_portions: Vec::new(),
        };

        // expected order of events:
        //   1. first (3) is popped for the new head and (1) is pushed.
        //   2. We now need a new page to store all the items. (2) is popped.
        //   3. This empties the head portion, leaving (3) dangling. to_push == MAX
        //   4. We use (3) as a new page for writing the last item, fragmentation was forced.
        //      If we had pushed (3) to the end of the free-list instead, to_push would be MAX+1
        //      and we'd need to take a new page from bump or the previous head, dirtying it and
        //      allocating unnecessarily.
        let result = free_list.commit(
            (4..)
                .take(MAX_PNS_PER_PAGE - 1)
                .map(PageNumber)
                .collect::<Vec<_>>(),
            &mut PageNumber(10000),
        );

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].0, PageNumber(2));
        assert_eq!(result[1].0, PageNumber(3));

        assert_eq!(free_list.portions.len(), 2);
        assert_eq!(free_list.portions[0].0, PageNumber(2));
        assert_eq!(free_list.portions[1].0, PageNumber(3));
        assert_eq!(free_list.portions[0].1.len(), MAX_PNS_PER_PAGE - 1);
        assert_eq!(free_list.portions[1].1.len(), 1);
    }

    #[test]
    fn pop_into_next_portion_without_fragmentation() {
        let mut free_list = FreeList {
            portions: vec![(PageNumber(1), vec![PageNumber(2), PageNumber(3)])],
            pop: false,
            released_portions: Vec::new(),
        };

        // expected order of events:
        //   1. first (3) is popped for the new head and (1) is pushed.
        //   2. We now need a new page to store all the items. (2) is popped.
        //   3. This empties the head portion, leaving (3) dangling. to_push = MAX + 1
        //   4. We push (2) for the first new page and (3) for the next. (2) is completely full,
        //      and (3) has the last item.
        let result = free_list.commit(
            (4..)
                .take(MAX_PNS_PER_PAGE)
                .map(PageNumber)
                .collect::<Vec<_>>(),
            &mut PageNumber(10000),
        );

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].0, PageNumber(2));
        assert_eq!(result[1].0, PageNumber(3));

        assert_eq!(free_list.portions.len(), 2);
        assert_eq!(free_list.portions[0].0, PageNumber(2));
        assert_eq!(free_list.portions[1].0, PageNumber(3));
        assert_eq!(free_list.portions[0].1.len(), MAX_PNS_PER_PAGE);
        assert_eq!(free_list.portions[1].1.len(), 1);
    }

    #[test]
    fn fragmentation_handled() {
        let full_portion = (10000..)
            .take(MAX_PNS_PER_PAGE - 2)
            .chain(Some(2))
            .chain(Some(3))
            .map(PageNumber)
            .collect::<Vec<_>>();
        let mut free_list = FreeList {
            portions: vec![
                (PageNumber(1), full_portion),
                (PageNumber(4), vec![PageNumber(5)]),
            ],
            pop: false,
            released_portions: Vec::new(),
        };

        // expected order of events:
        //   1. first (5) is popped for the new head and (4) is pushed. to_push == MAX+1.
        //   2. This vacates the head page. (5) is used as a fresh page.
        //   2. We need a new page to store all the items. We need to pop into the previously
        //      full page.
        //   3. Its previous head (1) is pushed and (3) is taken to overwrite it. to_push == MAX+2
        //   4. (2) is popped to handle the new page.
        //   5. One of the to_push items is written into the full page. to_push == MAX+1
        //   6. We can't write MAX items into the next page and 1 into another, because we'd need
        //      to pop a new page for that but wouldn't be able to use it.
        //   7. The only solution is to pop the next page (69) and have the second-to-last page
        //      fragmented.

        let result = free_list.commit(
            (20000..)
                .take(MAX_PNS_PER_PAGE)
                .map(PageNumber)
                .collect::<Vec<_>>(),
            &mut PageNumber(100000),
        );

        assert_eq!(result.len(), 3);
        assert_eq!(result[0].0, PageNumber(3));
        assert_eq!(result[1].0, PageNumber(5));
        assert_eq!(result[2].0, PageNumber(2));

        assert_eq!(free_list.portions.len(), 3);

        assert_eq!(free_list.portions[0].0, PageNumber(3));
        assert_eq!(free_list.portions[1].0, PageNumber(5));
        assert_eq!(free_list.portions[2].0, PageNumber(2));

        assert_eq!(free_list.portions[0].1.len(), MAX_PNS_PER_PAGE);
        assert_eq!(free_list.portions[1].1.len(), MAX_PNS_PER_PAGE - 1);
        assert_eq!(free_list.portions[2].1.len(), 1);
    }

    #[test]
    fn clean_up_fragmentation() {
        let fragmented_portion = (10000..)
            .take(MAX_PNS_PER_PAGE - 3)
            .chain(Some(2))
            .chain(Some(3))
            .map(PageNumber)
            .collect::<Vec<_>>();

        let mut free_list = FreeList {
            portions: vec![
                (PageNumber(1), fragmented_portion),
                (PageNumber(4), vec![PageNumber(5)]),
            ],
            pop: false,
            released_portions: Vec::new(),
        };

        // expected order of operations:
        //   1. 5 is popped to rewrite the head. This vacates the head.
        //   2. Because the previous portion is fragmented, 5 is taken as its new head and its old
        //      head and the outdated (1) and (4) are pushed.
        //   3. A new page (3) is popped for the new items.
        //   4. The fragmented (5) is filled out and the rest of the items are pushed into (3)
        let result = free_list.commit(vec![PageNumber(6)], &mut PageNumber(10000));

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].0, PageNumber(5));
        assert_eq!(result[1].0, PageNumber(3));

        assert_eq!(free_list.portions.len(), 2);

        assert_eq!(free_list.portions[0].0, PageNumber(5));
        assert_eq!(free_list.portions[1].0, PageNumber(3));

        assert_eq!(free_list.portions[0].1.len(), MAX_PNS_PER_PAGE);
        assert_eq!(free_list.portions[1].1.len(), 1);
    }

    #[test]
    fn rewrite_head_after_pop_only() {
        let mut free_list = FreeList {
            portions: vec![(PageNumber(1), vec![PageNumber(2)])],
            pop: true,
            released_portions: Vec::new(),
        };

        let result = free_list.commit(Vec::new(), &mut PageNumber(10000));
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, PageNumber(1));
    }
}
