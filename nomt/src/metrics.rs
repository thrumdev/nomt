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
                page_requests_count: AtomicU64::new(0),
                page_cache_misses_count: AtomicU64::new(0),
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

                let tot_page_requests = metrics.page_requests_count.load(Ordering::Relaxed);
                println!("  page requests         {}", tot_page_requests);

                if tot_page_requests != 0 {
                    let cache_misses = metrics.page_cache_misses_count.load(Ordering::Relaxed);
                    let percentage_cache_misses =
                        (cache_misses as f64 / tot_page_requests as f64) * 100.0;

                    println!(
                        "  page cache misses     {} - {:.2}% of page requests",
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
    pub page_requests_count: AtomicU64,
    pub page_cache_misses_count: AtomicU64,
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
