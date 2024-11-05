use super::{WalBlobBuilder, WalBlobReader, WalEntry};
use crate::{io::page_pool::PagePool, page_diff::PageDiff};
use std::{fs::OpenOptions, io::Write as _};

#[test]
fn test_write_read() {
    let tempdir = tempfile::tempdir().unwrap();
    let wal_filename = tempdir.path().join("wal");
    std::fs::create_dir_all(tempdir.path()).unwrap();
    let mut wal_fd = {
        let mut options = OpenOptions::new();
        options.read(true).write(true).create(true);
        options.open(&wal_filename).unwrap()
    };

    let mut builder = WalBlobBuilder::new().unwrap();
    builder.write_clear(0);
    builder.write_update(
        [0; 32],
        &PageDiff::from_bytes(hex_literal::hex!(
            "00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00"
        ))
        .unwrap(),
        vec![].into_iter(),
        0,
    );
    builder.write_clear(1);
    builder.write_update(
        [1; 32],
        &PageDiff::from_bytes(hex_literal::hex!(
            "01 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00"
        ))
        .unwrap(),
        vec![[1; 32]].into_iter(),
        1,
    );
    builder.write_update(
        [2; 32],
        &{
            let mut diff = PageDiff::default();
            for i in 0..126 {
                diff.set_changed(i);
            }
            diff
        },
        (0..126).map(|x| [x; 32]),
        2,
    );
    let (ptr, len) = builder.finalize();
    wal_fd
        .write_all(unsafe { std::slice::from_raw_parts(ptr, len) })
        .unwrap();
    wal_fd.sync_data().unwrap();

    let page_pool = PagePool::new();
    let mut reader = WalBlobReader::new(&page_pool, &wal_fd).unwrap();
    assert_eq!(
        reader.read_entry().unwrap(),
        Some(WalEntry::Clear { bucket: 0 })
    );
    assert_eq!(
        reader.read_entry().unwrap(),
        Some(WalEntry::Update {
            page_id: [0; 32],
            page_diff: PageDiff::default(),
            changed_nodes: vec![],
            bucket: 0,
        })
    );
    assert_eq!(
        reader.read_entry().unwrap(),
        Some(WalEntry::Clear { bucket: 1 })
    );
    assert_eq!(
        reader.read_entry().unwrap(),
        Some(WalEntry::Update {
            page_id: [1; 32],
            page_diff: {
                let mut diff = PageDiff::default();
                diff.set_changed(0);
                diff
            },
            changed_nodes: vec![[1; 32]],
            bucket: 1,
        })
    );
    assert_eq!(
        reader.read_entry().unwrap(),
        Some(WalEntry::Update {
            page_id: [2; 32],
            page_diff: {
                let mut diff = PageDiff::default();
                for i in 0..126 {
                    diff.set_changed(i);
                }
                diff
            },
            changed_nodes: (0..126).map(|x| [x; 32]).collect(),
            bucket: 2,
        })
    );
    assert_eq!(reader.read_entry().unwrap(), None);
}
