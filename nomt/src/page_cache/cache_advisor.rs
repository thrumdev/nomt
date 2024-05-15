//! A simple eviction manager with 256 shards
//! and two segments to provide for scan resistance.
//! Tells you when to evict items from a cache.
//!
//! features:
//!
//! * two-segment LRU, protects against cache pollution from single-hit items
//! * 256 shards accessed via non-blocking flatcombining
//! * local access buffer that must fill up before accessing shared state
//! * compresses the costs associated with each item to a `u8` using a compression
//!   technique that will converge to the overall true sum of costs over time, but
//!   allows for much less memory to be used for accounting.
//!
//! # Examples
//! ```
//! use cache_advisor::CacheAdvisor;
//!
//! // each shard stores 10 bytes, 10% of that is in the entry cache
//! let mut ca = CacheAdvisor::new(256 * 10, 10);
//!
//! // add item 0 into entry cache
//! let should_evict = ca.accessed_reuse_buffer(0, 1);
//! assert!(should_evict.is_empty());
//!
//! // promote item 0 into main cache
//! let should_evict = ca.accessed_reuse_buffer(0, 1);
//! assert!(should_evict.is_empty());
//!
//! // hit other items only once, like a big scan
//! for i in 1..5000 {
//!     let id = i * 256;
//!     let evicted = ca.accessed_reuse_buffer(id, 1);
//!
//!     // assert that 0 is never evicted while scanning
//!     assert!(!evicted.contains(&(0, 1)));
//! }
//!
//! let mut zero_evicted = false;
//!
//! // hit other items more than once, assert that zero does get
//! // evicted eventually.
//! for i in 1..5000 {
//!     let id = i * 256;
//!     zero_evicted |= ca.accessed_reuse_buffer(id, 1).contains(&(0, 1));
//!     zero_evicted |= ca.accessed_reuse_buffer(id, 1).contains(&(0, 1));
//!     zero_evicted |= ca.accessed_reuse_buffer(id, 1).contains(&(0, 1));
//! }
//!
//! assert!(zero_evicted);
//! ```
use std::{
    borrow::Borrow,
    cell::UnsafeCell,
    fmt,
    hash::{Hash, Hasher},
    ops::{Deref, DerefMut},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

use crossbeam_queue::SegQueue;
use nomt_core::page_id::PageId;

use super::dll::{DoublyLinkedList, Node};
//mod dll2;

const MAX_QUEUE_ITEMS: usize = 32;
// ensures that usize::MAX compresses to less than 128,
// since the max bit of a u8 size is used to represent
// the cache tier tag.
const RESIZE_CUTOFF: usize = 63;
const RESIZE_CUTOFF_U8: u8 = RESIZE_CUTOFF as u8;
const N_SHARDS: usize = 256;

// very very simple mutex that reduces instruction cache pollution
struct TryMutex<T> {
    inner: UnsafeCell<T>,
    mu: AtomicBool,
}

impl<T> TryMutex<T> {
    fn new(inner: T) -> TryMutex<T> {
        TryMutex {
            inner: inner.into(),
            mu: false.into(),
        }
    }

    #[inline]
    fn try_lock(&self) -> Option<TryMutexGuard<'_, T>> {
        if self.mu.swap(true, Ordering::Acquire) {
            // already locked
            None
        } else {
            Some(TryMutexGuard { tm: self })
        }
    }
}

struct TryMutexGuard<'a, T> {
    tm: &'a TryMutex<T>,
}

unsafe impl<T: Send> Send for TryMutex<T> {}

unsafe impl<T: Send> Sync for TryMutex<T> {}

impl<'a, T> Drop for TryMutexGuard<'a, T> {
    #[inline]
    fn drop(&mut self) {
        assert!(self.tm.mu.swap(false, Ordering::Release));
    }
}

impl<'a, T> Deref for TryMutexGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe { &*self.tm.inner.get() }
    }
}

impl<'a, T> DerefMut for TryMutexGuard<'a, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.tm.inner.get() }
    }
}

#[derive(Clone, Default)]
struct Resizer {
    actual: u128,
    decompressed: u128,
}

impl Resizer {
    /// Returns a compressed size which
    /// has been probabilistically chosen.
    fn compress(&mut self, raw_input: usize) -> u8 {
        if raw_input <= RESIZE_CUTOFF {
            return u8::try_from(raw_input).unwrap();
        }

        let upgraded_input = u128::try_from(raw_input).unwrap();
        let po2 = upgraded_input.next_power_of_two();
        let compressed = po2.trailing_zeros() as u8;
        let decompressed = decompress(compressed + RESIZE_CUTOFF_U8) as u128;
        self.actual += raw_input as u128;

        let ret = if self.decompressed + decompressed > self.actual {
            compressed - 1
        } else {
            compressed
        };

        self.decompressed += decompress(ret + RESIZE_CUTOFF_U8) as u128;

        let sz = ret + RESIZE_CUTOFF_U8;

        assert!(sz < 128);

        sz
    }
}

#[inline]
const fn decompress(input: u8) -> usize {
    // zero-out the access bit
    let masked = input & 127;
    match masked {
        0..=RESIZE_CUTOFF_U8 => masked as usize,
        _ => {
            if let Some(o) = 1_usize.checked_shl((masked - RESIZE_CUTOFF_U8) as u32) {
                o
            } else {
                usize::MAX
            }
        }
    }
}

struct Fnv(u64);

impl Default for Fnv {
    #[inline]
    fn default() -> Fnv {
        Fnv(0xcbf29ce484222325)
    }
}

impl std::hash::Hasher for Fnv {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let Fnv(mut hash) = *self;

        for byte in bytes.iter() {
            hash ^= *byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }

        *self = Fnv(hash);
    }
}

pub(crate) type FnvSet8<V> = std::collections::HashSet<V, std::hash::BuildHasherDefault<Fnv>>;

fn _sz_test() {
    let _: [u8; 33] = [0; std::mem::size_of::<CacheAccess>()];
    let _: [u8; 1] = [0; std::mem::align_of::<CacheAccess>()];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CacheAccess {
    promoted: bool,
    pid_bytes: [u8; 32],
}

impl CacheAccess {
    fn was_promoted(&self) -> bool {
        self.promoted
    }

    fn pid(&self) -> PageId {
        PageId::decode(self.pid_bytes).unwrap()
    }

    fn new(pid: PageId) -> CacheAccess {
        let pid_bytes = pid.encode();
        CacheAccess { promoted: false, pid_bytes }
    }
}

/// A simple eviction manager with 256 shards
/// and two segments to provide for scan resistance.
/// Tells you when to evict items from a cache.
///
/// features:
///
/// * two-segment LRU, protects against cache pollution from single-hit items
/// * 256 shards accessed via non-blocking flatcombining
/// * local access buffer that must fill up before accessing shared state
/// * compresses the costs associated with each item to a `u8` using a compression
///   technique that will converge to the overall true sum of costs over time, but
///   allows for much less memory to be used for accounting.
///
/// # Examples
/// ```
/// use cache_advisor::CacheAdvisor;
///
/// // each shard stores 10 bytes, 10% of that is in the entry cache
/// let mut ca = CacheAdvisor::new(256 * 10, 10);
///
/// // add item 0 into entry cache
/// let should_evict = ca.accessed(0, 1);
/// assert!(should_evict.is_empty());
///
/// // promote item 0 into main cache
/// let should_evict = ca.accessed(0, 1);
/// assert!(should_evict.is_empty());
///
/// // hit other items only once, like a big scan
/// for i in 1..5000 {
///     let id = i * 256;
///     let evicted = ca.accessed(id, 1);
///
///     // assert that 0 is never evicted while scanning
///     assert!(!evicted.contains(&(0, 1)));
/// }
///
/// let mut zero_evicted = false;
///
/// // hit other items more than once, assert that zero does get
/// // evicted eventually.
/// for i in 1..5000 {
///     let id = i * 256;
///     zero_evicted |= ca.accessed(id, 1).contains(&(0, 1));
///     zero_evicted |= ca.accessed(id, 1).contains(&(0, 1));
///     zero_evicted |= ca.accessed(id, 1).contains(&(0, 1));
/// }
///
/// assert!(zero_evicted);
/// ```
pub struct CacheAdvisor {
    shards: Arc<[TryMutex<Shard>]>,
    access_queues: Arc<[SegQueue<CacheAccess>]>,
    local_queue: Vec<PageId>,
    resizer: Resizer,
    access_buffer: Vec<PageId>,
}

impl Clone for CacheAdvisor {
    fn clone(&self) -> CacheAdvisor {
        CacheAdvisor {
            shards: self.shards.clone(),
            access_queues: self.access_queues.clone(),
            local_queue: vec![],
            resizer: self.resizer.clone(),
            access_buffer: vec![],
        }
    }
}

impl fmt::Debug for CacheAdvisor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CacheAdvisor").finish()
    }
}

impl Default for CacheAdvisor {
    /// Returns a `CacheAdvisor` with a default of 1 million capacity, and 20% entry cache
    fn default() -> CacheAdvisor {
        CacheAdvisor::new(1024 * 1024, 20)
    }
}

const fn _send_sync_ca() {
    const fn send_sync<T: Send + Sync>() {}
    send_sync::<CacheAdvisor>();
}

fn shard_index(id: &PageId) -> usize {
    let mut hasher = Fnv::default();
    id.hash(&mut hasher);
    (hasher.finish() % N_SHARDS as u64) as usize
}

impl CacheAdvisor {
    /// Instantiates a new `CacheAdvisor` eviction manager.
    ///
    /// `entry_percent` is how much of the cache should be
    /// devoted to the "entry" cache. When new items are added
    /// to the system, they are inserted into the entry cache
    /// first. If they are accessed at some point while still
    /// in the entry cache, they will be promoted to the main
    /// cache. This provides "scan resistance" where the cache
    /// will avoid being destroyed by things like a scan that
    /// could otherwise push all of the frequently-accessed
    /// items out. A value of `20` is a reasonable default,
    /// which will reserve 20% of the cache capacity for the
    /// entry cache, and 80% for the main cache. This value
    /// must be less than or equal to 100. If the main cache
    /// has never been filled to the point where items are
    /// evicted, items that are pushed out of the entry cache
    /// will flow into the main cache, so you don't need to
    /// worry about under-utilizing available memory. This
    /// only changes behavior once the cache is full to prevent
    /// scans from kicking other items out.
    pub fn new(capacity: usize, entry_percent: u8) -> Self {
        assert!(
            capacity >= N_SHARDS,
            "Please configure the cache \
             capacity to be at least 256"
        );
        let shard_capacity = capacity / N_SHARDS;

        let mut shards = Vec::with_capacity(N_SHARDS);
        for _ in 0..N_SHARDS {
            shards.push(TryMutex::new(Shard::new(shard_capacity, entry_percent)))
        }

        let mut access_queues = Vec::with_capacity(N_SHARDS);
        for _ in 0..N_SHARDS {
            access_queues.push(SegQueue::default());
        }

        Self {
            shards: shards.into(),
            access_queues: access_queues.into(),
            local_queue: Vec::with_capacity(MAX_QUEUE_ITEMS),
            resizer: Resizer::default(),
            access_buffer: vec![],
        }
    }

    /// Called when an item is accessed. Returns a Vec of items to be
    /// evicted. Avoids blocking under contention by using flat-combining
    /// on 256 LRU shards.
    pub fn accessed(&mut self, id: PageId, to_evict: &SegQueue<PageId>) {
        self.accessed_inner(id, to_evict);
    }

    fn accessed_inner(
        &mut self,
        id: PageId,
        ret: &SegQueue<PageId>,
    ) {
        self.local_queue.push(id);

        if self.local_queue.len() < MAX_QUEUE_ITEMS {
            return;
        }

        while let Some(id) = self.local_queue.pop() {
            self.submit_access(id, Some(ret));
        }
    }

    fn submit_access(
        &self,
        id: PageId,
        ret: Option<&SegQueue<PageId>>,
    ) {
        let shard_idx = shard_index(&id);
        let shard_mu = &self.shards[shard_idx];
        let access_queue = &self.access_queues[shard_idx];
        let cache_access = CacheAccess::new(id);

        // use flat-combining to avoid lock contention
        if let Some(mut shard) = shard_mu.try_lock() {
            // we take len here and bound pops to this number
            // because we don't want to keep going forever
            // if new items are flowing in - we need to get
            // back to our own work eventually.
            for _ in 0..access_queue.len() {
                if let Some(queued_cache_access) = access_queue.pop() {
                    shard.accessed(queued_cache_access, ret.as_deref());
                }
            }

            shard.accessed(cache_access, ret);
        } else {
            access_queue.push(cache_access);
        }
    }
}

impl Drop for CacheAdvisor {
    fn drop(&mut self) {
        let local_queue = std::mem::take(&mut self.local_queue);
        for id in local_queue {
            self.submit_access(id, None);
        }
    }
}

#[derive(Eq)]
struct Entry(*mut Node);

unsafe impl Send for Entry {}

impl Ord for Entry {
    fn cmp(&self, other: &Entry) -> std::cmp::Ordering {
        let left_pid: &[u8; 32] = self.borrow();
        let right_pid: &[u8; 32] = other.borrow();
        left_pid.cmp(&right_pid)
    }
}

impl PartialOrd<Entry> for Entry {
    fn partial_cmp(&self, other: &Entry) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Entry {
    fn eq(&self, other: &Entry) -> bool {
        unsafe { (*self.0).pid_bytes == (*other.0).pid_bytes }
    }
}

impl Borrow<[u8; 32]> for Entry {
    fn borrow(&self) -> &[u8; 32] {
        unsafe { &(*self.0).pid_bytes }
    }
}

// we only hash on pid, since we will change
// sz sometimes and we access the item by pid
impl Hash for Entry {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        unsafe { (*self.0).pid_bytes.hash(hasher) }
    }
}

struct Shard {
    entry_cache: DoublyLinkedList,
    main_cache: DoublyLinkedList,
    entries: FnvSet8<Entry>,
    entry_capacity: usize,
    entry_size: usize,
    main_capacity: usize,
    main_size: usize,
    ever_evicted_main: bool,
}

impl Shard {
    fn new(capacity: usize, entry_pct: u8) -> Self {
        assert!(
            entry_pct <= 100,
            "entry cache percent must be less than or equal to 100"
        );
        assert!(capacity > 0, "shard capacity must be non-zero");

        let entry_capacity = (capacity * entry_pct as usize) / 100;
        let main_capacity = capacity - entry_capacity;

        Self {
            entry_cache: DoublyLinkedList::default(),
            main_cache: DoublyLinkedList::default(),
            entries: FnvSet8::default(),
            entry_capacity,
            main_capacity,
            entry_size: 0,
            main_size: 0,
            ever_evicted_main: false,
        }
    }

    fn accessed(&mut self, cache_access: CacheAccess, ret: Option<&SegQueue<PageId>>) {
        if let Some(entry) = self.entries.get(&cache_access.pid_bytes) {
            let was_promoted = unsafe {
                let was_promoted = (*entry.0).was_promoted();

                // This is a bit hacky but it's done
                // this way because HashSet doesn't have
                // a get_mut method.
                //
                // This is safe to do because the hash
                // happens based on the PageId of the
                // CacheAccess, rather than the size
                // that we modify here.
                (*entry.0).inner.get_mut().promoted = true;

                was_promoted
            };

            if was_promoted {
                // item is already in main cache

                self.main_size -= 1;

                self.main_cache.unwire(entry.0);
                self.main_cache.install(entry.0);
            } else {
                // item is in entry cache

                self.entry_size -= 1;

                self.entry_cache.unwire(entry.0);
                self.main_cache.install(entry.0);
            }

            self.main_size += 1;
        } else if !self.ever_evicted_main {
            // We can put new writes into the
            // main cache directly until it fills
            // up, letting us get higher hit rates,
            // assuming the entry cache is smaller
            // than the main cache.
            let mut cache_access = cache_access;
            cache_access.promoted = true;
            let ptr = self.main_cache.push_head(cache_access);
            self.entries.insert(Entry(ptr));
            self.main_size += 1;
        } else {
            let ptr = self.entry_cache.push_head(cache_access);
            self.entries.insert(Entry(ptr));
            self.entry_size += 1;
        };

        if let Some(ret) = ret {
            self.evict(ret);
        }
    }

    fn evict(&mut self, ret: &SegQueue<PageId>) {
        while self.entry_size > self.entry_capacity && self.entry_cache.len() > 1 {
            let node: *mut Node = self.entry_cache.pop_tail().unwrap();

            let popped_entry: CacheAccess = unsafe { *(*node).inner.get() };
            let item = popped_entry.pid();

            self.entry_size -= 1;

            assert!(
                !popped_entry.was_promoted(),
                "somehow, promoted item was still in entry cache"
            );

            let pid_bytes = popped_entry.pid_bytes;
            assert!(self.entries.remove(&pid_bytes));

            ret.push(item);
            let node_box: Box<Node> = unsafe { Box::from_raw(node) };

            // NB: node is stored in our entries map
            // via a raw pointer, which points to
            // the same allocation used in the DLL.
            // We have to be careful to free node
            // only after removing it from both
            // the DLL and our entries map.
            drop(node_box);
        }

        while self.main_size > self.main_capacity && self.main_cache.len() > 1 {
            self.ever_evicted_main = true;

            let node: *mut Node = self.main_cache.pop_tail().unwrap();

            let popped_main: CacheAccess = unsafe { *(*node).inner.get() };
            let item = popped_main.pid();

            self.main_size -= 1;

            let pid_bytes = popped_main.pid_bytes;
            assert!(self.entries.remove(&pid_bytes));

            ret.push(item);

            let node_box: Box<Node> = unsafe { Box::from_raw(node) };

            // NB: node is stored in our entries map
            // via a raw pointer, which points to
            // the same allocation used in the DLL.
            // We have to be careful to free node
            // only after removing it from both
            // the DLL and our entries map.
            drop(node_box);
        }
    }
}

// #[test]
// fn lru_smoke_test() {
//     let mut lru = CacheAdvisor::new(256, 50);
//     let mut evicted = 0;
//     for i in 0..10_000 {
//         evicted += lru.accessed(mkid(i), 16).len();
//     }
//     assert!(evicted > 9700, "only evicted {} items", evicted);
// }

// #[test]
// fn probabilistic_sum() {
//     let mut resizer = Resizer::default();
//     let mut resized = 0;
//     let mut actual = 0;
//     for i in 0..1000 {
//         let compressed = resizer.compress(i);
//         let decompressed = decompress(compressed);
//         resized += decompressed;
//         actual += i;
//     }

//     let abs_delta = ((resized as f64 / actual as f64) - 1.).abs();

//     assert!(abs_delta < 0.005, "delta is actually {}", abs_delta);
// }

// #[test]
// fn probabilistic_ev() {
//     let mut resizer = Resizer::default();

//     fn assert_rt(i: usize, resizer: &mut Resizer) {
//         let mut resized = 0_u128;
//         let mut actual = 0_u128;
//         for _ in 1..10_000 {
//             let compressed = resizer.compress(i);
//             let decompressed = decompress(compressed);
//             resized += decompressed as u128;
//             actual += i as u128;
//         }

//         if i == 0 {
//             assert_eq!(actual, 0);
//             assert_eq!(resized, 0);
//         } else {
//             let abs_delta = ((resized as f64 / actual as f64) - 1.).abs();
//             assert!(
//                 abs_delta < 0.0001,
//                 "delta is actually {} for inputs of size {}. actual: {} round-trip: {}",
//                 abs_delta,
//                 i,
//                 actual,
//                 resized
//             );
//         }
//     }

//     for i in 0..1024 {
//         assert_rt(i, &mut resizer)
//     }

//     assert_rt(usize::MAX, &mut resizer)
// }

// #[test]
// fn probabilistic_n() {
//     const N: usize = 9;

//     let mut resizer = Resizer::default();
//     let mut resized = 0;
//     let mut actual = 0;

//     for _ in 0..1000 {
//         let compressed = resizer.compress(N);
//         let decompressed = decompress(compressed);
//         resized += decompressed;
//         actual += N;
//     }

//     let abs_delta = ((resized as f64 / actual as f64) - 1.).abs();

//     assert!(abs_delta < 0.005, "delta is actually {}", abs_delta);
// }

// #[test]
// fn scan_resistance() {
//     // each shard stores 10 bytes, 10% of that is in the entry cache
//     let mut ca = CacheAdvisor::new(256 * 10, 10);

//     // add 0 into entry cache
//     ca.accessed(mkid(0), 1);

//     // promote 0 into main cache
//     ca.accessed(mkid(0), 1);

//     // hit other items only once, like a big scan
//     for i in 1..5000 {
//         let id = i * 256;
//         let evicted = ca.accessed(mkid(id), 1);

//         // assert that 0 is never evicted while scanning
//         assert!(!evicted.contains(&(mkid(0), 1)));
//     }

//     let mut zero_evicted = false;

//     // hit other items more than once, assert that zero does get
//     // evicted eventually.
//     for i in 1..5000 {
//         let id = i * 256;
//         zero_evicted |= ca.accessed(mkid(id), 1).contains(&(mkid(0), 1));
//         zero_evicted |= ca.accessed(mkid(id), 1).contains(&(mkid(0), 1));
//         zero_evicted |= ca.accessed(mkid(id), 1).contains(&(mkid(0), 1));
//     }

//     assert!(zero_evicted);
// }

// #[cfg(test)]
// fn mkid(mut i: usize) -> PageId {
//     use nomt_core::page_id::{ChildPageIndex, MAX_CHILD_INDEX, ROOT_PAGE_ID};

//     let mut page_id = ROOT_PAGE_ID;
//     while i > 0 {
//         let child_index = ChildPageIndex::new((i % MAX_CHILD_INDEX as usize) as u8).unwrap();
//         page_id = page_id.child_page_id(child_index).unwrap();
//         i /= MAX_CHILD_INDEX as usize;
//     }
//     page_id
// }
