use std::sync::atomic::{AtomicBool, Ordering};

use tokio::sync::Notify;

/// A one-off alarm.
pub struct Alarm {
    notify: Notify,
    triggered: AtomicBool,
}

impl Alarm {
    pub fn new() -> Self {
        Self {
            notify: Notify::new(),
            triggered: AtomicBool::new(false),
        }
    }

    /// Trigger the alarm.
    ///
    /// Makes any tasks waiting on [`Self::triggered`] are unblocked.
    ///
    /// If the alarm has been triggered already does nothing.
    pub fn trigger(&self) {
        if self.triggered.swap(true, Ordering::Release) {
            return;
        }
        self.notify.notify_waiters();
    }

    /// Wait until the alarm is triggered.
    ///
    /// If the [`Self::trigger`] has been called already returns immediately.
    pub async fn triggered(&self) {
        if self.triggered.load(Ordering::Acquire) {
            return;
        }
        self.notify.notified().await;
    }
}
