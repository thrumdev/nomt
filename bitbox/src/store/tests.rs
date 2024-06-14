use crate::store::*;

use std::io::{Read, Seek};

const EMPTY_PAGE: [u8; PAGE_SIZE] = [0; PAGE_SIZE];
const TEST_PAGE: [u8; PAGE_SIZE] = [1; PAGE_SIZE];

#[test]
fn single_write_single_page() {
    let pages_capacity = 10;
    let store_options = StoreOptions {
        file_path: PathBuf::from("single_write_single_page"),
        pages_capacity,
        io_uring_capacity: 12,
        reset: true,
    };
    let mut store = Store::new(store_options);

    let (write_handle, _) = store.write(&[(TEST_PAGE, 0)]).unwrap();

    write_handle.wait();

    // use normal syscall to make sure the page was written
    let _ = store
        .store_file
        .seek(std::io::SeekFrom::Start(0))
        .expect("Imp seeking store file");
    let mut written_page = [0u8; PAGE_SIZE];
    let _ = store
        .store_file
        .read(&mut written_page)
        .expect("Imp read from store file");
    // expected page must be written
    assert_eq!(written_page, TEST_PAGE);

    let mut empty_pages = vec![0u8; pages_capacity as usize - 1];
    let _ = store
        .store_file
        .read(&mut empty_pages)
        .expect("Imp read from store file");
    // nothing else is expected to be modified
    assert_eq!(empty_pages, vec![0u8; pages_capacity as usize - 1]);

    std::fs::remove_file("single_write_single_page").unwrap();
}

#[test]
fn single_write_multiple_pages() {
    let pages_capacity = 10;
    let store_options = StoreOptions {
        file_path: PathBuf::from("single_write_multiple_pages"),
        pages_capacity,
        io_uring_capacity: 12,
        reset: true,
    };
    let mut store = Store::new(store_options);

    let (write_handle, _) = store
        .write(&[
            ([1; PAGE_SIZE], 1),
            ([3; PAGE_SIZE], 3),
            ([5; PAGE_SIZE], 5),
            ([7; PAGE_SIZE], 7),
            ([9; PAGE_SIZE], 9),
        ])
        .unwrap();

    write_handle.wait();

    // use normal syscall to make sure the page was written
    let _ = store
        .store_file
        .seek(std::io::SeekFrom::Start(0))
        .expect("Imp seeking store file");
    for i in 0..pages_capacity / 2 {
        let mut page = [0u8; PAGE_SIZE];

        let _ = store
            .store_file
            .read(&mut page)
            .expect("Imp read from store file");
        // following page should not be modified
        assert_eq!(page, EMPTY_PAGE);

        let _ = store
            .store_file
            .read(&mut page)
            .expect("Imp read from store file");
        // expected page must be written
        assert_eq!(page, [(2 * i as u8) + 1; PAGE_SIZE]);
    }
    std::fs::remove_file("single_write_multiple_pages").unwrap();
}

#[test]
fn multiple_write_multiple_pages() {
    let pages_capacity = 12;
    let store_options = StoreOptions {
        file_path: PathBuf::from("multiple_write_multiple_pages"),
        pages_capacity,
        io_uring_capacity: 12,
        reset: true,
    };
    let mut store = Store::new(store_options);

    let (write_handle_1, _) = store
        .write(&[([1; PAGE_SIZE], 1), ([3; PAGE_SIZE], 3)])
        .unwrap();
    let (write_handle_2, _) = store
        .write(&[([5; PAGE_SIZE], 5), ([7; PAGE_SIZE], 7)])
        .unwrap();
    let (write_handle_3, _) = store
        .write(&[([9; PAGE_SIZE], 9), ([11; PAGE_SIZE], 11)])
        .unwrap();

    write_handle_1.wait();
    write_handle_2.wait();
    write_handle_3.wait();

    // use normal syscall to make sure the page was written
    let _ = store
        .store_file
        .seek(std::io::SeekFrom::Start(0))
        .expect("Imp seeking store file");
    for i in 0..pages_capacity / 2 {
        let mut page = [0u8; PAGE_SIZE];
        let _ = store
            .store_file
            .read(&mut page)
            .expect("Imp read from store file");
        // following page should not be modified
        assert_eq!(page, EMPTY_PAGE);

        let _ = store
            .store_file
            .read(&mut page)
            .expect("Imp read from store file");
        // expected page must be written
        assert_eq!(page, [(2 * i as u8) + 1; PAGE_SIZE]);
    }

    std::fs::remove_file("multiple_write_multiple_pages").unwrap();
}

#[test]
fn write_fullfill_submission_queue() {
    let pages_capacity = 1000;
    let store_options = StoreOptions {
        file_path: PathBuf::from("write_fullfill_submission_queue"),
        pages_capacity,
        io_uring_capacity: 1000,
        reset: true,
    };
    let mut store = Store::new(store_options);

    let pages: Vec<_> = (0..pages_capacity).map(|i| (TEST_PAGE, i)).collect();
    dbg!(pages.len());
    let (write_handle, _) = store.write(&pages[..]).unwrap();
    write_handle.wait();

    // use normal syscall to make sure the page was written
    let _ = store
        .store_file
        .seek(std::io::SeekFrom::Start(0))
        .expect("Imp seeking store file");
    for _ in 0..pages_capacity {
        let mut page = [0u8; PAGE_SIZE];
        let _ = store
            .store_file
            .read(&mut page)
            .expect("Imp read from store file");
        // expected page must be written
        assert_eq!(page, TEST_PAGE);
    }
    std::fs::remove_file("write_fullfill_submission_queue").unwrap();
}

#[test]
fn write_exceed_submission_queue() {
    let pages_capacity = 1000;
    let store_options = StoreOptions {
        file_path: PathBuf::from("write_exceed_submission_queue"),
        pages_capacity,
        io_uring_capacity: 256,
        reset: true,
    };
    let mut store = Store::new(store_options);

    let mut pages: Vec<_> = (0..pages_capacity).map(|i| (TEST_PAGE, i)).collect();
    let mut writes_handle = vec![];
    loop {
        match store.write(&pages[..]) {
            Some((write_handle, written)) if written == pages.len() as u64 => {
                writes_handle.push(write_handle);
                break;
            }
            Some((write_handle, written)) => {
                writes_handle.push(write_handle);
                pages.drain(0..written as usize);
            }
            None => (),
        }
    }

    for write_handle in writes_handle {
        write_handle.wait();
    }

    // use normal syscall to make sure the page was written
    let _ = store
        .store_file
        .seek(std::io::SeekFrom::Start(0))
        .expect("Imp seeking store file");
    for _ in 0..pages_capacity {
        let mut page = [0u8; PAGE_SIZE];
        let _ = store
            .store_file
            .read(&mut page)
            .expect("Imp read from store file");
        // expected page must be written
        assert_eq!(page, TEST_PAGE);
    }
    std::fs::remove_file("write_exceed_submission_queue").unwrap();
}

#[test]
fn single_read() {
    let pages_capacity = 10;
    let store_options = StoreOptions {
        file_path: PathBuf::from("single_read"),
        pages_capacity,
        io_uring_capacity: 12,
        reset: true,
    };
    let mut store = Store::new(store_options);

    let read_handle = store.read(6).unwrap();
    assert_eq!(read_handle.wait(), EMPTY_PAGE);

    let (write_handle, _) = store.write(&[([1; PAGE_SIZE], 6)]).unwrap();
    write_handle.wait();

    let read_handle = store.read(6).unwrap();
    assert_eq!(read_handle.wait(), [1; PAGE_SIZE]);
    std::fs::remove_file("single_read").unwrap();
}

#[test]
fn multiple_read() {
    let pages_capacity = 10;
    let store_options = StoreOptions {
        file_path: PathBuf::from("multiple_read"),
        pages_capacity,
        io_uring_capacity: 12,
        reset: true,
    };
    let mut store = Store::new(store_options);

    let indexes: Vec<u64> = (0..pages_capacity / 2).map(|i| (2 * i) + 1).collect();

    let pages: Vec<_> = indexes
        .iter()
        .map(|i| ([*i as u8; PAGE_SIZE], *i))
        .collect();
    let (write_handle, _) = store.write(&pages[..]).unwrap();
    write_handle.wait();

    let read_handles: Vec<_> = indexes.iter().map(|i| store.read(*i).unwrap()).collect();

    for (read_handle, i) in read_handles.into_iter().zip(indexes) {
        assert_eq!(read_handle.wait(), [i as u8; PAGE_SIZE]);
    }
    std::fs::remove_file("multiple_read").unwrap();
}
