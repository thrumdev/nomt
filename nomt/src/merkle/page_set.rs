//! A set of pages that the page walker draws upon and which is filled by `Seek`ing.

use nomt_core::page_id::PageId;
use std::collections::HashMap;

use crate::{
    io::PagePool,
    page_cache::{Page, PageMut},
    store::{BucketInfo, SharedMaybeBucketIndex},
};

/// The mode to use when determining bucket indices for fresh pages.
#[derive(Clone, Copy)]
pub enum FreshPageBucketMode {
    WithDependents,
    WithoutDependents,
}

pub struct PageSet {
    map: HashMap<PageId, (Page, BucketInfo)>,
    page_pool: PagePool,
    fresh_page_bucket_mode: FreshPageBucketMode,
}

impl PageSet {
    pub fn new(page_pool: PagePool, mode: FreshPageBucketMode) -> Self {
        PageSet {
            map: HashMap::new(),
            page_pool,
            fresh_page_bucket_mode: mode,
        }
    }

    fn fresh_bucket_info(&self) -> BucketInfo {
        match self.fresh_page_bucket_mode {
            FreshPageBucketMode::WithDependents => {
                BucketInfo::FreshOrDependent(SharedMaybeBucketIndex::new(None))
            }
            FreshPageBucketMode::WithoutDependents => BucketInfo::FreshWithNoDependents,
        }
    }

    /// Insert a page with a known bucket index.
    pub fn insert(&mut self, page_id: PageId, page: Page, bucket_info: BucketInfo) {
        self.map.insert(page_id, (page, bucket_info));
    }
}

impl super::page_walker::PageSet for PageSet {
    fn fresh(&self, page_id: &PageId) -> (PageMut, BucketInfo) {
        let page = PageMut::pristine_empty(&self.page_pool, &page_id);
        let bucket_info = self.fresh_bucket_info();

        (page, bucket_info)
    }

    fn get(&self, page_id: &PageId) -> Option<(Page, BucketInfo)> {
        self.map
            .get(&page_id)
            .map(|(p, bucket_info)| (p.clone(), bucket_info.clone()))
            .map(|(p, b)| {
                if let (FreshPageBucketMode::WithDependents, &BucketInfo::FreshWithNoDependents) =
                    (self.fresh_page_bucket_mode, &b)
                {
                    // during warm-ups, we always run with `WithoutDependents`. This replaces the
                    // fresh variant if running with dependents during update.
                    (
                        p,
                        BucketInfo::FreshOrDependent(SharedMaybeBucketIndex::new(None)),
                    )
                } else {
                    (p, b)
                }
            })
    }
}
