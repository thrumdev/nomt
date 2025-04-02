use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

/// Metrics collector, if active, it provides Counters and Timers
#[derive(Clone)]
pub struct Metrics {
    metrics: Option<Arc<ActiveMetrics>>,
}

unsafe impl Send for Metrics {}

/// Metrics that can be collected during execution
#[derive(PartialEq, Eq, Hash)]
pub enum Metric {
    /// Counter of total page requests
    PageRequests,
    /// Counter of page requests cache misses over all page requests
    PageCacheMisses,
    /// Timer used to record average page fetch time
    PageFetchTime,
    /// Timer used to record average value fetch time during reads
    ValueFetchTime,
    /// Count every HashTable page created / modified. TODO bettercomment
    TotalHashTablePages,
    /// Counter of the number of hash table pages with fewer leaves
    /// than the page elision threshold. TODO bettercomment
    ElidedHashTablePages,
}

struct ActiveMetrics {
    page_requests: AtomicU64,
    page_cache_misses: AtomicU64,
    elided_hashtable_pages: AtomicU64,
    total_hashtable_pages: AtomicU64,
    page_fetch_time: Timer,
    value_fetch_time: Timer,
}

impl Metrics {
    /// Returns the Metrics object, active or not based on the specified input
    pub fn new(active: bool) -> Self {
        Self {
            metrics: if active {
                Some(Arc::new(ActiveMetrics {
                    page_requests: AtomicU64::new(0),
                    page_cache_misses: AtomicU64::new(0),
                    elided_hashtable_pages: AtomicU64::new(0),
                    total_hashtable_pages: AtomicU64::new(0),
                    page_fetch_time: Timer::new(),
                    value_fetch_time: Timer::new(),
                }))
            } else {
                None
            },
        }
    }

    /// Increase the Counter specified by the input
    ///
    /// panics if the specified [`Metric`] is not a Counter
    pub fn count(&self, metric: Metric) {
        if let Some(ref metrics) = self.metrics {
            let counter = match metric {
                Metric::PageRequests => &metrics.page_requests,
                Metric::PageCacheMisses => &metrics.page_cache_misses,
                Metric::ElidedHashTablePages => &metrics.elided_hashtable_pages,
                Metric::TotalHashTablePages => &metrics.total_hashtable_pages,
                _ => panic!("Specified metric is not a Counter"),
            };

            counter.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Returns a guard that, when dropped, will record the time passed since creation
    ///
    /// panics if the specified [`Metric`] is not a Timer
    pub fn record<'a>(&'a self, metric: Metric) -> Option<impl Drop + 'a> {
        self.metrics.as_ref().and_then(|metrics| {
            let timer = match metric {
                Metric::PageFetchTime => &metrics.page_fetch_time,
                Metric::ValueFetchTime => &metrics.value_fetch_time,
                _ => panic!("Specified metric is not a Timer"),
            };

            Some(timer.record())
        })
    }

    /// Print collected metrics to stdout
    pub fn print(&self) {
        if let Some(ref metrics) = self.metrics {
            println!("metrics");

            let elided_hashtable_pages = metrics.elided_hashtable_pages.load(Ordering::Relaxed);
            println!("  elided pages          {}", elided_hashtable_pages);

            let total_hashtable_pages = metrics.total_hashtable_pages.load(Ordering::Relaxed);
            println!("  total pages           {}", total_hashtable_pages);

            let tot_page_requests = metrics.page_requests.load(Ordering::Relaxed);
            println!("  page requests         {}", tot_page_requests);

            if tot_page_requests != 0 {
                let cache_misses = metrics.page_cache_misses.load(Ordering::Relaxed);
                let percentage_cache_misses =
                    (cache_misses as f64 / tot_page_requests as f64) * 100.0;

                println!(
                    "  page cache misses     {} - {:.2}% of page requests",
                    cache_misses, percentage_cache_misses
                );
            }

            if let Some(mean) = metrics.page_fetch_time.mean() {
                println!("  page fetch mean       {}", pretty_display_ns(mean));
            }

            if let Some(mean) = metrics.value_fetch_time.mean() {
                println!("  value fetch mean      {}", pretty_display_ns(mean));
            }
        } else {
            println!("Metrics collection was not activated")
        }
    }
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

struct Timer {
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

    fn mean(&self) -> Option<u64> {
        let n = self.number_of_records.load(Ordering::Relaxed);
        let sum = self.sum.load(Ordering::Relaxed);
        sum.checked_div(n)
    }

    fn record<'a>(&'a self) -> impl Drop + 'a {
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

        TimerGuard {
            start: std::time::Instant::now(),
            n: &self.number_of_records,
            sum: &self.sum,
        }
    }
}
