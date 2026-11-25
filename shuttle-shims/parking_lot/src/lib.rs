//! `parking_lot` 0.12 under shuttle. `Mutex` comes from `shuttle-parking_lot-impl`. `RwLock` and
//! `Condvar` are defined here: that crate's `RawRwLock` lacks `RawRwLockRecursive`, which NOMT's
//! session access lock needs, and it has no `Condvar`.

pub use shuttle_parking_lot_impl::*;

use shuttle::future::batch_semaphore::{Acquire, BatchSemaphore, Fairness};

type Guard<'a, T> = lock_api::MutexGuard<'a, RawMutex, T>;

/// Generation-counting condvar: a waiter takes the generation lock before releasing the user's
/// mutex, so a notify between the two can't be lost.
pub struct Condvar {
    generation: shuttle::sync::Mutex<u64>,
    cv: shuttle::sync::Condvar,
}

impl Condvar {
    pub fn new() -> Self {
        Condvar {
            generation: shuttle::sync::Mutex::new(0),
            cv: shuttle::sync::Condvar::new(),
        }
    }

    pub fn wait<T: ?Sized>(&self, guard: &mut Guard<'_, T>) {
        let mut generation = self.generation.lock().unwrap();
        let start = *generation;
        lock_api::MutexGuard::unlocked(guard, move || {
            while *generation == start {
                generation = self.cv.wait(generation).unwrap();
            }
        });
    }

    pub fn wait_while<T: ?Sized, F: FnMut(&mut T) -> bool>(
        &self,
        guard: &mut Guard<'_, T>,
        mut condition: F,
    ) {
        while condition(&mut *guard) {
            self.wait(guard);
        }
    }

    pub fn notify_one(&self) -> bool {
        self.notify_all();
        true
    }

    pub fn notify_all(&self) -> usize {
        *self.generation.lock().unwrap() += 1;
        self.cv.notify_all();
        0
    }
}

/// Fair reader-writer lock: a strictly fair semaphore where a reader takes one permit and a writer
/// all of them, so new readers queue behind a waiting writer, as in parking_lot.
pub struct RawRwLock {
    sem: BatchSemaphore,
}

const MAX_READERS: usize = usize::MAX >> 3;

fn block_on(acquire: Acquire<'_>) {
    shuttle::future::block_on(acquire).unwrap();
}

unsafe impl lock_api::RawRwLock for RawRwLock {
    #[allow(clippy::declare_interior_mutable_const)]
    const INIT: Self = RawRwLock {
        sem: BatchSemaphore::const_new(MAX_READERS, Fairness::StrictlyFair),
    };
    type GuardMarker = lock_api::GuardSend;

    fn lock_shared(&self) {
        block_on(self.sem.acquire(1));
    }
    fn try_lock_shared(&self) -> bool {
        self.sem.try_acquire(1).is_ok()
    }
    unsafe fn unlock_shared(&self) {
        self.sem.release(1);
    }
    fn lock_exclusive(&self) {
        block_on(self.sem.acquire(MAX_READERS));
    }
    fn try_lock_exclusive(&self) -> bool {
        self.sem.try_acquire(MAX_READERS).is_ok()
    }
    unsafe fn unlock_exclusive(&self) {
        self.sem.release(MAX_READERS);
    }
}

/// A recursive read overtakes queued writers: `upgrade` takes one more permit from the front of
/// the queue.
unsafe impl lock_api::RawRwLockRecursive for RawRwLock {
    fn lock_shared_recursive(&self) {
        block_on(self.sem.upgrade(1, 2));
    }
    fn try_lock_shared_recursive(&self) -> bool {
        self.sem.try_upgrade(1, 2).is_ok()
    }
}

pub type RwLock<T> = lock_api::RwLock<RawRwLock, T>;
pub type RwLockReadGuard<'a, T> = lock_api::RwLockReadGuard<'a, RawRwLock, T>;
pub type RwLockWriteGuard<'a, T> = lock_api::RwLockWriteGuard<'a, RawRwLock, T>;
