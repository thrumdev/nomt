use crate::beatree::leaf::{FreeListPage, PageNumber};
use crate::io::{CompleteIo, IoCommand, IoKind};
use crate::store::{Page, PAGE_SIZE};

use crossbeam_channel::{Receiver, Sender, TrySendError};
use std::fs::File;
use std::os::fd::AsRawFd;

const MAX_PNS_PER_FREE_PAGE: usize = (PAGE_SIZE - 6) / 4;

/// In-memory version of the FreeList which provides a way to decode from and encode into the LeafStore file
/// and provide two primitives, one for extracting free pages from the list and one to append
/// freed pages.
///
/// Pages that are freed due to the fetch of free pages are automatically added back during the encode phase,
/// which also covers the addition of new free pages.
pub struct FreeList {
    head: Option<(PageNumber, Vec<PageNumber>)>,
    portions: Vec<(PageNumber, Vec<PageNumber>)>,
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
                head: None,
                portions: vec![],
                released_portions: vec![],
            };
        };

        // restore free list form file
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
                    Err(TrySendError::Disconnected(_)) => panic!("I/O leaf store worker dropped"),
                    Err(TrySendError::Full(c)) => {
                        command = Some(c);
                    }
                }
            }

            let completion = io_receiver.recv().expect("I/O leaf store worker dropped");
            assert!(completion.result.is_ok());
            let page = completion.command.kind.unwrap_buf();

            let (prev, free_list) = decode_free_list_page(page);
            free_list_portions.push((free_list_pn, free_list));
            free_list_pn = prev;
        }

        FreeList {
            head: free_list_portions.pop(),
            portions: free_list_portions,
            released_portions: vec![],
        }
    }

    pub fn head_pn(&self) -> Option<PageNumber> {
        self.head.as_ref().map(|(head_pn, _)| head_pn).copied()
    }

    pub fn pop(&mut self) -> Option<PageNumber> {
        let Some((head_pn, head)) = &mut self.head else {
            // If there is no available head,
            // then it means that there are no more elements in the free list
            return None;
        };

        let leaf_pn = head.pop().unwrap();

        // replace head if we just emptied one
        if head.is_empty() {
            self.released_portions.push(*head_pn);
            self.head = self.portions.pop();
        }

        Some(leaf_pn)
    }

    fn push(
        &mut self,
        pn: PageNumber,
        next_head_pns: &mut impl Iterator<Item = PageNumber>,
        force_new_head: bool,
    ) -> Option<(PageNumber, FreeListPage)> {
        let mut encoded_head = None;

        // create new_head if required
        match &mut self.head {
            None => {
                self.head = Some((next_head_pns.next().unwrap(), vec![]));
            }
            Some((head_pn, head)) if force_new_head || head.len() == MAX_PNS_PER_FREE_PAGE => {
                let prev = self
                    .portions
                    .last()
                    .map(|(pn, _)| *pn)
                    .unwrap_or(PageNumber(0));

                encoded_head = Some((*head_pn, encode_free_list_page(prev, head)));

                self.portions.push((*head_pn, std::mem::take(head)));
                *head_pn = next_head_pns.next().unwrap();
            }
            _ => (),
        };

        // extract head safely
        let head = self.head.as_mut().map(|(_, h)| h).unwrap();
        head.push(pn);

        encoded_head
    }

    /// Pages numbers are updated during batches of changes by taking free pages from the free list
    /// or adding new ones to the list.
    ///
    /// FreeList Pages are stored in the LeafStore using a Copy-On-Write (COW) approach.
    ///
    /// This function considers every step taken since the last call to this function,
    /// creating an encoded version of the updated free list state. (ready to be saved into the LeafStore)
    ///
    /// Possible operations on a FreeList include removing a free page or adding a new one.
    /// These operations, while creating the encoded format, lead to multiple intermediate steps:
    ///    1. Removing pages consumes the list.
    ///    2. Adding pages appends them to the list's head.
    ///    3. Adding new pages require allocating a new page for the list's head.
    ///       This page should not overwrite pages in the leaf store and
    ///       should be taken from the head at the end of step one.
    ///
    /// Shifting page numbers to the old head involves moving all page numbers added at step two.
    /// Generally, this is not an issue, for example using FreeList pages that can store up to five numbers:
    ///
    /// The current head is composed of: [1, 7, 9, 2, 5].
    ///
    /// The performed operations include one pop and pushing five elements ([6, 10, 12, 15, 20]).
    ///
    /// As a result, 5 is popped out, leaving space for only 6 in the head.
    /// Consequently, a new head is created to store [10, 12, 15, 20].
    ///
    /// To accommodate the new head and overwrite the old one, two new page numbers are required.
    /// These are taken from the head at the end of stage 1, after removing all values.
    ///
    /// In this case, 2 and 9 will be used. With two additional free spaces in the previous head,
    /// values from the new head will shift to occupy all spots in the old head. The final result is:
    ///
    /// [(page_number: 2, free_list:[1, 7, 6, 10, 12]), (page_number: 5, free_list:[15, 20])]
    ///
    /// There is a special case where shifting values to the left can result in the new head becoming empty.
    /// For example, starting from the previous state, if we pop 5 elements and push only [10, 12, 15]:
    ///
    /// - Head after step 1: [1, 7, 9, 2]
    /// - We need to push 3 new elements, so 10 will be the last element in the head.
    /// - Elements 2 and 9 are removed from the old head and used as new page numbers for both the old and new heads.
    /// - The new head should store 12 and 15.
    /// - The removal of 2 and 9 causes a shift, resulting in all elements fitting into the old head: [1, 7, 10, 12, 15]
    ///   with a page number of 2 but unused page number 9.
    /// - To maintain the LIFO property and avoid wasting page number 9, a small fragmentation is created.
    ///   The old head will then contain only 4 elements, and the new head will contain one element, resulting in:
    ///
    /// [(Page number: 2, Free list: [1, 7, 10, 12]), (Page number: 9, Free list: [15])]
    ///
    /// It returns a Vec containing all the pages that need to be written into storage to reflect the current
    /// status of the free list along with their page numbers
    pub fn commit(
        &mut self,
        mut to_append: Vec<PageNumber>,
        bump: &mut PageNumber,
    ) -> Vec<(PageNumber, FreeListPage)> {
        // append the released free list pages
        to_append.extend(std::mem::take(&mut self.released_portions));

        let new_pns_len = to_append.len();

        // max number of pages required to be allocated
        let pages_to_allocate = (new_pns_len as f64 / MAX_PNS_PER_FREE_PAGE as f64).ceil() as usize;

        let additional = if self.head.is_none() { 0 } else { 1 };
        let mut free_list_pages_pns = (0..pages_to_allocate + additional)
            .map(|_| match self.pop() {
                Some(pn) => pn,
                None => {
                    let pn = *bump;
                    bump.0 += 1;
                    pn
                }
            })
            .collect::<Vec<_>>()
            .into_iter();

        let mut pns_iter = to_append.into_iter();
        let mut inner_frag = false;

        // if the head is not empty, then the previous pop
        // did not empty the free list, thus there will be some free space
        // left in the head.
        if let Some((head_pn, head)) = &mut self.head {
            // let's change the page_number of the current head to avoid overwriting it
            *head_pn = free_list_pages_pns.next().unwrap();

            let free_space = MAX_PNS_PER_FREE_PAGE - head.len();

            // if this holds then we will have to deal with a PageNumber fragmentation in the new
            // head
            if free_space + (pages_to_allocate - 1) * MAX_PNS_PER_FREE_PAGE == new_pns_len {
                head.extend(pns_iter.by_ref().take(free_space - 1));
                inner_frag = true;
            }
        }

        // encoding new free list pages
        let mut pages = vec![];

        for pn in pns_iter {
            let maybe_encoded_head = self.push(pn, &mut free_list_pages_pns, inner_frag);
            inner_frag = false;

            if let Some(encoded) = maybe_encoded_head {
                pages.push(encoded);
            }
        }

        pages
    }
}

// returns the previous PageNumber and all the PageNumbers stored in the free list page
//
// A free page is laid out in the following form:
// + prev free page : u32
// + item_count : u16
// + leaf page number : [u32; item_count]
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

fn encode_free_list_page(prev: PageNumber, pns: &Vec<PageNumber>) -> FreeListPage {
    let mut page = Page::zeroed();

    page[0..4].copy_from_slice(&prev.0.to_le_bytes());
    page[4..6].copy_from_slice(&(pns.len() as u16).to_le_bytes());

    for (i, pn) in pns.into_iter().enumerate() {
        let start = 6 + i * 4;
        page[start..start + 4].copy_from_slice(&pn.0.to_le_bytes());
    }

    FreeListPage {
        inner: Box::new(page),
    }
}
