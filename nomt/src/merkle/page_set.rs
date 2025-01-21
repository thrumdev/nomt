//! A set of pages that the page walker draws upon and which is filled by `Seek`ing.

use nomt_core::page_id::PageId;
use std::{collections::HashMap, sync::Arc};

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
    warm_up_map: Option<Arc<HashMap<PageId, (Page, BucketInfo)>>>,
    page_pool: PagePool,
    fresh_page_bucket_mode: FreshPageBucketMode,
}

impl PageSet {
    pub fn new(
        page_pool: PagePool,
        mode: FreshPageBucketMode,
        warmed_up: Option<FrozenSharedPageSet>,
    ) -> Self {
        PageSet {
            map: HashMap::new(),
            page_pool,
            warm_up_map: warmed_up.map(|x| x.0),
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

    /// Freeze this page-set and make a shareable version of it. This returns a frozen page set
    /// containing all insertions into this map.
    pub fn freeze(self) -> FrozenSharedPageSet {
        FrozenSharedPageSet(Arc::new(self.map))
    }

    fn get_warmed_up(&self, page_id: &PageId) -> Option<(Page, BucketInfo)> {
        self.warm_up_map
            .as_ref()
            .and_then(|m| m.get(page_id))
            .map(|(p, b)| (p.clone(), b.clone()))
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
            .or_else(|| self.get_warmed_up(page_id))
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

/// A frozen, shared page set. This is cheap to clone.
#[derive(Clone)]
pub struct FrozenSharedPageSet(Arc<HashMap<PageId, (Page, BucketInfo)>>);
