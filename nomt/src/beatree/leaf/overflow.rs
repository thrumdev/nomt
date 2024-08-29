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
const HEADER_SIZE: usize = 4;

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
            let start = HEADER_SIZE + pns_written * 4;
            let end = start + 4;
            page[start..end].copy_from_slice(&pn.0.to_le_bytes());
            pns_written += 1;
        }

        // then write as many value bytes as possible.
        let bytes = std::cmp::min(BODY_SIZE - pns_written * 4, value.len());

        // write the header.
        page[0..2].copy_from_slice(&(pns_written as u16).to_le_bytes());
        page[2..4].copy_from_slice(&(bytes as u16).to_le_bytes());

        let start = HEADER_SIZE + pns_written * 4;
        let end = start + bytes;
        page[start..end].copy_from_slice(&value[..bytes]);
        value = &value[bytes..];

        // write the page.
        leaf_writer.write_preallocated(pn, page);
    }
    assert!(value.is_empty());

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
    for (pn, slice) in pages.iter().zip(v[8..].chunks_mut(4)) {
        slice.copy_from_slice(&pn.0.to_le_bytes());
    }

    v
}

fn total_needed_pages(value_size: usize) -> usize {
    // the encoded size is equal to the size of the value plus the number of node pointers that
    // will appear in pages.
    let needed_pages_raw_value = needed_pages(value_size);

    if needed_pages_raw_value <= MAX_OVERFLOW_CELL_NODE_POINTERS {
        // less than MAX_OVERFLOW_CELL_NODE_POINTERS,
        // space available in the leaf is enough
        return needed_pages_raw_value;
    }

    // not all needed_pages_raw_values are used to store values.
    // Therefore, there will be unused space in one of the pages
    // to store other page numbers
    let bytes_left = (needed_pages_raw_value * BODY_SIZE) - value_size;
    let available_page_numbers = bytes_left / 4;

    if needed_pages_raw_value <= MAX_OVERFLOW_CELL_NODE_POINTERS + available_page_numbers {
        // there are enough available bytes to store all the remaining page numbers
        return needed_pages_raw_value;
    }

    // additional pages are required
    //
    // given:
    // value_size = vs
    // MAX_OVERFLOW_CELL_NODE_POINTERS = M
    // needed_pages_raw_value = np
    // body_size = bs
    // required_additional_pages = rp
    //
    // total bytes available:
    // tot_bytes = (np + rp) * bs
    //
    // only a portion of the total available bytes will be used:
    // used_bytes = vs - ((np - M)) * 4 - (rp * 4)
    //
    // we want tot_bytes - used_bytes to be positive and as little as possible,
    // the only variable is rp, thus we can make everything dependent on rp:
    //
    // rp = ceil(n / (bs - 4))
    // where n = (vs + (np - M) * 4 - (np * bs))

    let n = value_size + (needed_pages_raw_value - MAX_OVERFLOW_CELL_NODE_POINTERS) * 4
        - (needed_pages_raw_value * BODY_SIZE);

    let required_additional_pages = (n + BODY_SIZE - 3) / (BODY_SIZE - 4);

    needed_pages_raw_value + required_additional_pages
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

    let iter = page[HEADER_SIZE..][..n_pages * 4]
        .chunks(4)
        .map(|slice| PageNumber(u32::from_le_bytes(slice.try_into().unwrap())));

    let bytes = &page[HEADER_SIZE + n_pages * 4..][..n_bytes];
    (iter, bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn total_needed_pages_all_in_cell() {
        for i in 1..=MAX_OVERFLOW_CELL_NODE_POINTERS {
            assert_eq!(total_needed_pages(BODY_SIZE * i), i);
        }
    }

    #[test]
    fn total_needed_pages_one_out_of_cell() {
        let size = BODY_SIZE * MAX_OVERFLOW_CELL_NODE_POINTERS + 1;
        assert_eq!(
            total_needed_pages(size),
            MAX_OVERFLOW_CELL_NODE_POINTERS + 1
        );
    }

    #[test]
    fn total_needed_pages_encoded_page_adds_more() {
        // last page is totally full
        let size = BODY_SIZE * (MAX_OVERFLOW_CELL_NODE_POINTERS + 1) - 4;
        assert_eq!(
            total_needed_pages(size),
            MAX_OVERFLOW_CELL_NODE_POINTERS + 1
        );

        // last page not full
        let size = BODY_SIZE * (MAX_OVERFLOW_CELL_NODE_POINTERS + 1) - 3;
        assert_eq!(
            total_needed_pages(size),
            MAX_OVERFLOW_CELL_NODE_POINTERS + 2
        );
    }

    #[test]
    fn total_needed_pages_additional_pages_adds_more() {
        let size = (BODY_SIZE * MAX_OVERFLOW_CELL_NODE_POINTERS) + (BODY_SIZE * MAX_PNS);
        assert_eq!(
            total_needed_pages(size),
            // one page is used to save MAX_PNS page numbers,
            // and another page is used to save the page number of the previous one
            MAX_OVERFLOW_CELL_NODE_POINTERS + MAX_PNS + 1 + 1
        );
    }

    #[test]
    fn total_needed_pages_really_big() {
        let size = 1 << 30;

        // this many pages for the value
        let pages0 = 262401;
        assert_eq!(needed_pages(size), pages0);

        let pages_in_pages0 = pages0 - MAX_OVERFLOW_CELL_NODE_POINTERS;

        // this many pages for the value plus those pages
        let size1 = size + pages_in_pages0 * 4;
        let pages1 = needed_pages(size1);

        let pages_in_pages1 = pages1 - MAX_OVERFLOW_CELL_NODE_POINTERS - pages_in_pages0;

        // this many pages for the value plus _those_ pages
        let size2 = size1 + pages_in_pages1 * 4;
        let pages2 = needed_pages(size2);

        assert_eq!(pages1, pages2);
        assert_eq!(pages1, total_needed_pages(size));
    }
}
