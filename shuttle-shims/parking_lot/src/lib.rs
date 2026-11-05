//! `parking_lot` 0.12 under shuttle: the locks come from `shuttle-parking_lot-impl`; `Condvar`,
//! which that crate lacks, is added here for `lock_api` mutex guards.

pub use shuttle_parking_lot_impl::*;

use std::time::Instant;

type Guard<'a, T> = lock_api::MutexGuard<'a, RawMutex, T>;

pub struct WaitTimeoutResult(bool);

impl WaitTimeoutResult {
    pub fn timed_out(&self) -> bool {
        self.0
    }
}

/// Generation-counting condvar: a waiter takes the generation lock before releasing the user's
/// mutex, so a notify between the two can't be lost.
pub struct Condvar {
    generation: shuttle::sync::Mutex<u64>,
    cv: shuttle::sync::Condvar,
}

impl Default for Condvar {
    fn default() -> Self {
        Self::new()
    }
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

    /// Shuttle has no clock: never times out, behaves like `wait`.
    pub fn wait_until<T: ?Sized>(&self, guard: &mut Guard<'_, T>, _: Instant) -> WaitTimeoutResult {
        self.wait(guard);
        WaitTimeoutResult(false)
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
