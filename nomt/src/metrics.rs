use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

/// Metrics collector, if active, it provides Counters and Timers
#[derive(Clone)]
pub enum Metrics {
    Active(Arc<ActiveMetrics>),
    Inactive,
}

impl Metrics {
    /// Returns the Metrics object, active or not based on the specified input
    pub fn new(metrics: bool) -> Self {
        if metrics {
            Metrics::Active(Arc::new(ActiveMetrics {
                page_cache_misses: RelativeCounter::new(),
                page_fetch_time: Timer::new(),
                value_fetch_time: Timer::new(),
            }))
        } else {
            Metrics::Inactive
        }
    }

    /// Print collected metrics to stdout
    pub fn print(&self) {
        match self {
            Metrics::Active(metrics) => {
                println!("metrics");

                let tot_page_requests = metrics.page_cache_misses.tot();
                println!("  page requests         {}", tot_page_requests);

                if tot_page_requests != 0 {
                    let cache_misses = metrics.page_cache_misses.count();
                    let percentage_cache_misses =
                        (cache_misses as f64 / tot_page_requests as f64) * 100.0;

                    println!(
                        "  page cache misses     {}  [ {:.2}% ] ",
                        cache_misses, percentage_cache_misses
                    );
                }

                println!(
                    "  page fetch mean       {}",
                    pretty_display_ns(metrics.page_fetch_time.mean())
                );
                println!(
                    "  value fetch mean      {}",
                    pretty_display_ns(metrics.value_fetch_time.mean())
                );
            }
            Metrics::Inactive => {
                println!("Metrics collection was not activated")
            }
        }
    }
}

/// Active metrics that can be collected during execution.
pub struct ActiveMetrics {
    pub page_cache_misses: RelativeCounter,
    pub page_fetch_time: Timer,
    pub value_fetch_time: Timer,
}

fn pretty_display_ns(ns: u64) -> String {
    // preserve 3 sig figs at minimum.
    let (val, unit) = if ns > 100 * 1_000_000_000 {
        (ns / 1_000_000_000, "s")
    } else if ns > 100 * 1_000_000 {
        (ns / 1_000_000, "ms")
    } else if ns > 100 * 1_000 {
        (ns / 1_000, "us")
    } else {
        (ns, "ns")
    };

    format!("{val} {unit}")
}

/// Used in [`ActiveMetrics`] to record timings
pub struct Timer {
    number_of_records: AtomicU64,
    sum: AtomicU64,
}

impl Timer {
    fn new() -> Self {
        Timer {
            number_of_records: AtomicU64::new(0),
            sum: AtomicU64::new(0),
        }
    }

    fn mean(&self) -> u64 {
        let n = self.number_of_records.load(Ordering::Relaxed);
        let sum = self.sum.load(Ordering::Relaxed);
        sum / n
    }

    /// Returns a guard that, when dropped, will record in [`ActiveMetrics`]
    /// the time passed since creation
    pub fn record<'a>(&'a self) -> Option<impl Drop + 'a> {
        struct TimerGuard<'a> {
            start: std::time::Instant,
            n: &'a AtomicU64,
            sum: &'a AtomicU64,
        }

        impl Drop for TimerGuard<'_> {
            fn drop(&mut self) {
                let elapsed = self.start.elapsed().as_nanos() as u64;
                self.n.fetch_add(1, Ordering::Relaxed);
                self.sum.fetch_add(elapsed, Ordering::Relaxed);
            }
        }

        Some(TimerGuard {
            start: std::time::Instant::now(),
            n: &self.number_of_records,
            sum: &self.sum,
        })
    }
}

/// Relative Counter used in [`ActiveMetrics`]
pub struct RelativeCounter {
    tot: AtomicU64,
    counter: AtomicU64,
}

impl RelativeCounter {
    fn new() -> Self {
        Self {
            tot: AtomicU64::new(0),
            counter: AtomicU64::new(0),
        }
    }

    /// Increase the inner counter of Relative Counter
    pub fn increase_count(&self) {
        self.counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Increase the total counter of Relative Counter
    pub fn increase_tot(&self) {
        self.tot.fetch_add(1, Ordering::Relaxed);
    }

    fn tot(&self) -> u64 {
        self.tot.load(Ordering::Relaxed)
    }

    fn count(&self) -> u64 {
        self.counter.load(Ordering::Relaxed)
    }
}

/// Starts a time record given the [`Metrics`] object reference
/// and the [`Timer`] we want to start
#[macro_export]
macro_rules! record {
    ($metrics: expr, $name: ident) => {
        let _maybe_guard = match $metrics {
            Metrics::Active(ref metrics) => Some(metrics.$name.record()),
            Metrics::Inactive => None,
        };
    };
}

/// Increase the total counter of a [`RelativeTimer`] given the [`Metrics`] object
/// reference and the idetifier of the counter
#[macro_export]
macro_rules! increase_tot {
    ($metrics: expr, $name: ident) => {
        if let Metrics::Active(metrics) = &$metrics {
            metrics.$name.increase_tot()
        }
    };
}

/// Increase the counter of a [`RelativeTimer`] given the [`Metrics`] object
/// reference and the idetifier of the counter
#[macro_export]
macro_rules! increase_count {
    ($metrics: expr, $name: ident) => {
        if let Metrics::Active(metrics) = &$metrics {
            metrics.$name.increase_count()
        }
    };
}
