/// Overflow pages are used to store values which exceed the maximum size.
///
/// Large values are chunked into pages in a deterministic way, optimized for parallel fetching.
///
/// The format of an overflow page is:
/// ```rust,ignore
/// n_pointers: u16
/// n_bytes: u16
/// pointers: [PageNumber; n_pointers]
/// bytes: [u8; n_bytes]
/// ```
use crate::{
    beatree::PageNumber,
    io::{Page, PAGE_SIZE},
};

use super::{node::MAX_OVERFLOW_CELL_NODE_POINTERS, store::LeafStoreWriter};

const BODY_SIZE: usize = PAGE_SIZE - 4;
const MAX_PNS: usize = BODY_SIZE / 4;

/// Encode a large value into freshly allocated overflow pages. Returns a vector of page pointers.
pub fn chunk(value: &[u8], leaf_writer: &mut LeafStoreWriter) -> Vec<PageNumber> {
    assert!(!value.is_empty());

    let total_pages = total_needed_pages(value.len());
    let cell_pages = std::cmp::min(total_pages, MAX_OVERFLOW_CELL_NODE_POINTERS);
    let cell = (0..cell_pages)
        .map(|_| leaf_writer.preallocate())
        .collect::<Vec<_>>();
    let other_pages = (0..total_pages)
        .skip(cell_pages)
        .map(|_| leaf_writer.preallocate())
        .collect::<Vec<_>>();

    let all_pages = cell.iter().cloned().chain(other_pages.iter().cloned());
    let mut to_write = other_pages.iter().cloned();

    let mut value = value;
    // loop over all page numbers.
    for pn in all_pages {
        assert!(!value.is_empty());

        // allocate a page.
        let mut page = Box::new(Page::zeroed());
        let mut pns_written = 0;

        // write as many page numbers as possible.
        while pns_written < MAX_PNS {
            let Some(pn) = to_write.next() else { break };
            let start = 4 + pns_written * 4;
            let end = start + 4;
            page[start..end].copy_from_slice(&pn.0.to_le_bytes());
            pns_written += 1;
        }

        // then write as many value bytes as possible.
        let bytes = std::cmp::min(BODY_SIZE - pns_written * 4, value.len());

        // write the header.
        page[0..2].copy_from_slice(&(pns_written as u16).to_le_bytes());
        page[2..4].copy_from_slice(&(bytes as u16).to_le_bytes());

        let start = 4 + pns_written * 4;
        let end = start + bytes;
        page[start..end].copy_from_slice(&value[..bytes]);
        value = &value[bytes..];

        // write the page.
        leaf_writer.write_preallocated(pn, page);
    }

    cell
}

/// Encode a list of page numbers into an overflow cell.
pub fn encode_cell(pages: &[PageNumber]) -> Vec<u8> {
    let mut v = vec![0u8; pages.len() * 4];
    for (pn, slice) in pages.iter().zip(v.chunks_mut(4)) {
        slice.copy_from_slice(&pn.0.to_le_bytes());
    }

    v
}

fn total_needed_pages(value_size: usize) -> usize {
    let mut encoded_size = value_size;
    let mut total_pages = needed_pages(encoded_size);

    // the encoded size is equal to the size of the value plus the number of node pointers that
    // will appear in pages.
    // TODO: there's probably a closed form for this.
    loop {
        // account for the fact that some of the pages are going to be in the cell and not
        // in pages, therefore they don't increase the payload size.
        let pages_in_pages = total_pages.saturating_sub(MAX_OVERFLOW_CELL_NODE_POINTERS);

        encoded_size += pages_in_pages * 4;
        let new_total = needed_pages(encoded_size);
        if new_total == total_pages {
            break;
        }
        total_pages = new_total;
    }

    total_pages
}

fn needed_pages(size: usize) -> usize {
    (size + BODY_SIZE - 1) / BODY_SIZE
}
