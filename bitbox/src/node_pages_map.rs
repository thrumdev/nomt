use crate::store::{Store, StoreOptions};

pub const PAGE_SIZE: usize = 4096;

pub type Page = [u8; PAGE_SIZE];

struct NodePagesMap {
    store: Store,
}

impl NodePagesMap {
    pub fn new() -> Self {
        let store = Store::new(StoreOptions::default());

        NodePagesMap { store }
    }

    //TODO: everything related to insrting pages into a map and reading from it
}
