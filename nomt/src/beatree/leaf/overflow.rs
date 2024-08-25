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

use super::{
    node::MAX_OVERFLOW_CELL_NODE_POINTERS,
    store::{LeafStoreReader, LeafStoreWriter},
};

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

/// Decode an overflow cell, returning the size of the value plus the pages numbers within the cell.
pub fn decode_cell<'a>(raw: &'a [u8]) -> (usize, impl Iterator<Item = PageNumber> + 'a) {
    assert!(raw.len() >= 12);
    assert_eq!(raw.len() % 4, 0);

    let value_size = u64::from_le_bytes(raw[0..8].try_into().unwrap());

    let iter = raw[8..]
        .chunks(4)
        .map(|slice| PageNumber(u32::from_le_bytes(slice.try_into().unwrap())));

    (value_size as usize, iter)
}

/// Encode a list of page numbers into an overflow cell.
pub fn encode_cell(value_size: usize, pages: &[PageNumber]) -> Vec<u8> {
    let mut v = vec![0u8; 8 + pages.len() * 4];
    v[0..8].copy_from_slice(&(value_size as u64).to_le_bytes());
    for (pn, slice) in pages[8..].iter().zip(v.chunks_mut(4)) {
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

pub fn read(cell: &[u8], leaf_reader: &LeafStoreReader) -> Vec<u8> {
    let (value_size, cell_pages) = decode_cell(cell);
    let total_pages = total_needed_pages(value_size);

    let mut value = Vec::with_capacity(value_size);

    let mut page_numbers = Vec::with_capacity(total_pages);
    page_numbers.extend(cell_pages);

    for i in 0..total_pages {
        let page = leaf_reader.query(page_numbers[i]);
        let (page_pns, bytes) = read_page(&page);
        page_numbers.extend(page_pns);
        value.extend(bytes);
    }

    assert_eq!(page_numbers.len(), total_pages);
    assert_eq!(value.len(), value_size);

    value
}

pub fn delete(cell: &[u8], leaf_reader: &LeafStoreReader, leaf_writer: &mut LeafStoreWriter) {
    let (value_size, cell_pages) = decode_cell(cell);
    let total_pages = total_needed_pages(value_size);

    let mut page_numbers = Vec::with_capacity(total_pages);
    page_numbers.extend(cell_pages);

    for i in 0..total_pages {
        let page = leaf_reader.query(page_numbers[i]);
        let (page_pns, bytes) = read_page(&page);
        page_numbers.extend(page_pns);

        // stop at the first page containing value data. no more pages will have more
        // page numbers to release.
        if bytes.len() > 0 {
            break;
        }
    }

    assert_eq!(page_numbers.len(), total_pages);

    for pn in page_numbers {
        leaf_writer.release(pn);
    }
}

fn read_page<'a>(page: &'a Page) -> (impl Iterator<Item = PageNumber> + 'a, &'a [u8]) {
    let n_pages = u16::from_le_bytes(page[0..2].try_into().unwrap()) as usize;
    let n_bytes = u16::from_le_bytes(page[2..4].try_into().unwrap()) as usize;

    let iter = page[2..][..n_pages * 4]
        .chunks(4)
        .map(|slice| PageNumber(u32::from_le_bytes(slice.try_into().unwrap())));

    let bytes = &page[2 + n_pages * 4..][..n_bytes];
    (iter, bytes)
}
