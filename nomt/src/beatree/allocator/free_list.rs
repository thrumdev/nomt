use crate::{
    beatree::FREELIST_EMPTY,
    io::{CompleteIo, IoCommand, IoKind},
    io::{Page, PAGE_SIZE},
};
use crossbeam_channel::{Receiver, Sender, TrySendError};
use std::{collections::BTreeSet, fs::File, os::fd::AsRawFd};

use super::PageNumber;

const MAX_PNS_PER_PAGE: usize = (PAGE_SIZE - 6) / 4;

/// In-memory version of FreeList which provides a way to decode from and encode into pages
/// and provide two primitives, one for extracting free pages from the list and one to append
/// freed pages.
///
/// Pages that are freed due to the fetch of free pages are automatically added back during the encode phase,
/// which also covers the addition of new free pages.
pub struct FreeList {
    // head is last portion.
    portions: Vec<(PageNumber, Vec<PageNumber>)>,
    // Becomes true if something is being popped, false otherwise
    pop: bool,
    released_portions: Vec<PageNumber>,
}

impl FreeList {
    pub fn read(
        store_file: &File,
        io_sender: &Sender<IoCommand>,
        io_handle_index: usize,
        io_receiver: &Receiver<CompleteIo>,
        free_list_head: Option<PageNumber>,
    ) -> FreeList {
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

            let page = Box::new(Page::zeroed());

            let mut command = Some(IoCommand {
                kind: IoKind::Read(store_file.as_raw_fd(), free_list_pn.0 as u64, page),
                handle: io_handle_index,
                user_data: 0,
            });

            while let Some(c) = command.take() {
                match io_sender.try_send(c) {
                    Ok(()) => break,
                    Err(TrySendError::Disconnected(_)) => panic!("I/O worker dropped"),
                    Err(TrySendError::Full(c)) => {
                        command = Some(c);
                    }
                }
            }

            let completion = io_receiver.recv().expect("I/O store worker dropped");
            assert!(completion.result.is_ok());
            let page = completion.command.kind.unwrap_buf();

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
        let pns = self.portions.iter()
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
        self.pop = false;

        // append the released free list pages
        to_push.extend(self.released_portions.drain(..));

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
            match self.released_portions.pop() {
                Some(x) => {
                    if let Some((new_head_pn, new_head_pns)) = self.portions.last_mut()
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
                },
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
            // nothing popped, defer to loop to create the page for the next section.
            i = 0;
        };

        // loop invariant: free list len + i is always divisible by MAX_PNS_PER_PAGE.
        // therefore, the loop condition asks "do we still have items beyond the last page?"
        // while accounting for pops and pushes within the loop.
        while i < to_push.len() {
            if let Some((ref mut head_pn, ref mut head_pns))
                = self.portions.last_mut().filter(|_| new_full_portion)
            {
                // just popped into a new portion (which we will edit). prepare it for rewrite
                to_push.push(*head_pn);

                // UNWRAP: new_full_portion guarantees that the new portion is full.
                *head_pn = head_pns.pop().unwrap();

                // pop reduces free list length, we adjust i to compensate.
                // however, we must unconditionally rewrite this page even if i equals the previous
                // value of to_push.len(). That is because if we didn't pop we wouldn't have enough
                // pages.
                // this is inescapable and is what causes fragmentation.
                i += 1;

                continue
            }

            match self.pop() {
                Some(pn) => {
                    new_pages.push(pn);
                    // pop reduces free list length, we adjust i to compensate
                    i += 1;

                    if let Some(released) = self.released_portions.pop() {
                        // note: this is the PN we took out of the new portion.
                        to_push.push(released);
                        new_full_portion = true;
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
            let new_head = self.portions.last().map_or(true, |h| h.1.len() % MAX_PNS_PER_PAGE == 0)
                || i + 2 == to_push.len() && new_pages.peek().is_some();

            if new_head {
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

    fn push(
        &mut self,
        pn: PageNumber,
    ) {
        // UNWRAP: `push` is only called when head is not full.
        let head = self.portions.last_mut().unwrap();
        assert!(head.1.len() < MAX_PNS_PER_PAGE);
        head.1.push(pn);
    }

    fn encode_head(&self) -> Option<(PageNumber, Box<Page>)> {
        if let Some((head_pn, head_pns)) = self.portions.last() {
            let prev_pn = self.portions.len().checked_sub(2)
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
