//! `thread_local` 1.1 under shuttle: one value per shuttle thread. The real crate keys values by
//! OS thread, and every shuttle thread runs on the same one.

use shuttle::current::TaskId;
use std::{collections::HashMap, sync::Mutex};

pub struct ThreadLocal<T: Send> {
    // A std mutex, not shuttle's: it's never held across a scheduling point, so it can't block,
    // and the lookup adds no interleavings that the real, lock-free crate doesn't have.
    values: Mutex<HashMap<TaskId, Box<T>>>,
}

impl<T: Send> ThreadLocal<T> {
    pub fn new() -> Self {
        ThreadLocal {
            values: Mutex::new(HashMap::new()),
        }
    }

    pub fn get_or<F: FnOnce() -> T>(&self, create: F) -> &T {
        let mut values = self.values.lock().unwrap();
        let value: *const T = &**values
            .entry(shuttle::current::me())
            .or_insert_with(|| Box::new(create()));
        // SAFETY: values are boxed and only dropped with `self`, so this outlives the guard.
        unsafe { &*value }
    }
}
