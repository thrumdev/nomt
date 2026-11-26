use std::{
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Duration,
};

const METRICS_NOT_ENABLED: &str = "Metrics are not enabled";

/// Metrics collector, if active, it provides Counters and Timers
#[derive(Clone)]
pub struct Metrics {
    metrics: Option<Arc<ActiveMetrics>>,
}

/// Metrics that can be collected during execution
#[derive(PartialEq, Eq, Hash)]
pub enum Metric {
    /// Counter of merkle page requests served by the page cache.
    ///
    /// Only requests not already answered by the session overlay reach the page cache. Pages below
    /// an elided child are never requested from the page cache: they are reconstructed from
    /// beatree leaves, which show up in [`Metric::LeafCacheRequests`] instead.
    PageRequests,
    /// Counter of page requests cache misses over all page requests.
    ///
    /// This is not a count of disk reads: a miss on a page whose load is already in flight joins
    /// that load, and a miss on a page absent from the hash-table needs no read at all. Disk reads
    /// are counted by [`Metric::HtReads`].
    PageCacheMisses,
    /// Timer used to record average page fetch time: from the first read of a merkle page load
    /// from the hash-table, after a page cache miss, until the page is found, including all
    /// probes. Loads of pages which turn out not to exist are not recorded, nor are the loads
    /// made when opening the database (the root page and page cache prepopulation).
    PageFetchTime,
    /// Timer used to record average value fetch time during reads
    ValueFetchTime,
    /// Counter of beatree leaf cache lookups. Leaves are looked up when reading values, when
    /// seeking leaf nodes or reconstructing elided merkle pages, during the beatree update of a
    /// commit and when loading prior values for rollback.
    LeafCacheRequests,
    /// Counter of beatree leaf cache lookups which missed. Every miss leads to a read of the `ln`
    /// file, counted by [`Metric::LnReads`].
    LeafCacheMisses,
    /// Counter of page reads issued to the hash-table (`ht`) file, which stores merkle pages.
    /// A single page load may take several reads when probing the hash-table. This includes the
    /// reads made when opening the database, notably page cache prepopulation.
    HtReads,
    /// Counter of page reads issued to the beatree leaf (`ln`) file, which stores leaves and
    /// overflow value pages.
    LnReads,
}

struct ActiveMetrics {
    page_requests: AtomicU64,
    page_cache_misses: AtomicU64,
    page_fetch_time: Timer,
    value_fetch_time: Timer,
    leaf_cache_requests: AtomicU64,
    leaf_cache_misses: AtomicU64,
    ht_reads: AtomicU64,
    ln_reads: AtomicU64,
}

impl Metrics {
    /// Returns the Metrics object, active or not based on the specified input
    pub fn new(active: bool) -> Self {
        Self {
            metrics: if active {
                Some(Arc::new(ActiveMetrics {
                    page_requests: AtomicU64::new(0),
                    page_cache_misses: AtomicU64::new(0),
                    page_fetch_time: Timer::new(),
                    value_fetch_time: Timer::new(),
                    leaf_cache_requests: AtomicU64::new(0),
                    leaf_cache_misses: AtomicU64::new(0),
                    ht_reads: AtomicU64::new(0),
                    ln_reads: AtomicU64::new(0),
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
                Metric::LeafCacheRequests => &metrics.leaf_cache_requests,
                Metric::LeafCacheMisses => &metrics.leaf_cache_misses,
                Metric::HtReads => &metrics.ht_reads,
                Metric::LnReads => &metrics.ln_reads,
                _ => panic!("Specified metric is not a Counter"),
            };

            counter.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Returns a guard that, when dropped, will record the time passed since creation
    ///
    /// panics if the specified [`Metric`] is not a Timer
    pub fn record<'a>(&'a self, metric: Metric) -> Option<impl Drop + 'a> {
        self.metrics
            .as_ref()
            .map(|metrics| metrics.timer(metric).record())
    }

    /// Record a duration measured by the caller, for operations which do not fit a scope guard,
    /// such as asynchronous loads.
    ///
    /// panics if the specified [`Metric`] is not a Timer
    pub fn record_duration(&self, metric: Metric, duration: Duration) {
        if let Some(ref metrics) = self.metrics {
            metrics.timer(metric).add(duration);
        }
    }

    /// Whether metrics are being collected.
    pub fn is_active(&self) -> bool {
        self.metrics.is_some()
    }

    /// Print collected metrics to stdout
    pub fn print(&self) {
        if let Some(ref metrics) = self.metrics {
            println!("metrics");

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

            let leaf_cache_requests = metrics.leaf_cache_requests.load(Ordering::Relaxed);
            println!("  leaf cache requests   {}", leaf_cache_requests);

            if leaf_cache_requests != 0 {
                let leaf_cache_misses = metrics.leaf_cache_misses.load(Ordering::Relaxed);
                let percentage_cache_misses =
                    (leaf_cache_misses as f64 / leaf_cache_requests as f64) * 100.0;

                println!(
                    "  leaf cache misses     {} - {:.2}% of leaf cache requests",
                    leaf_cache_misses, percentage_cache_misses
                );
            }

            println!(
                "  ht page reads         {}",
                metrics.ht_reads.load(Ordering::Relaxed)
            );
            println!(
                "  ln page reads         {}",
                metrics.ln_reads.load(Ordering::Relaxed)
            );

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

    /// Counter of total page requests.
    /// Panics if metrics are not enabled.
    pub fn get_page_requests(&self) -> u64 {
        self.metrics
            .as_ref()
            .map(|metrics| metrics.page_requests.load(Ordering::Relaxed))
            .expect(METRICS_NOT_ENABLED)
    }

    /// Counter of page requests cache misses over all page requests.
    /// Panics if metrics are not enabled.
    pub fn get_page_cache_misses(&self) -> u64 {
        self.metrics
            .as_ref()
            .map(|metrics| metrics.page_cache_misses.load(Ordering::Relaxed))
            .expect(METRICS_NOT_ENABLED)
    }

    /// Counter of beatree leaf cache lookups.
    /// Panics if metrics are not enabled.
    pub fn get_leaf_cache_requests(&self) -> u64 {
        self.load_counter(|m| &m.leaf_cache_requests)
    }

    /// Counter of beatree leaf cache lookups which missed and had to read the `ln` file.
    /// Panics if metrics are not enabled.
    pub fn get_leaf_cache_misses(&self) -> u64 {
        self.load_counter(|m| &m.leaf_cache_misses)
    }

    /// Counter of page reads issued to the hash-table (`ht`) file, which stores merkle pages.
    /// Panics if metrics are not enabled.
    pub fn get_ht_reads(&self) -> u64 {
        self.load_counter(|m| &m.ht_reads)
    }

    /// Counter of page reads issued to the beatree leaf (`ln`) file, which stores leaves and
    /// overflow value pages.
    /// Panics if metrics are not enabled.
    pub fn get_ln_reads(&self) -> u64 {
        self.load_counter(|m| &m.ln_reads)
    }

    fn load_counter(&self, counter: impl FnOnce(&ActiveMetrics) -> &AtomicU64) -> u64 {
        self.metrics
            .as_ref()
            .map(|metrics| counter(metrics).load(Ordering::Relaxed))
            .expect(METRICS_NOT_ENABLED)
    }

    /// Average page fetch time: the time to load a merkle page from the hash-table after a page
    /// cache miss, including all probes. See [`Metric::PageFetchTime`].
    /// Returns None if there were no requests.
    /// Panics if metrics are not enabled.
    pub fn get_page_fetch_time(&self) -> Option<u64> {
        self.metrics
            .as_ref()
            .map(|metrics| metrics.page_fetch_time.mean())
            .expect(METRICS_NOT_ENABLED)
    }

    /// Average value fetch time during reads.
    /// Returns None if there were no requests.
    /// Panics if metrics are not enabled.
    pub fn get_value_fetch_time(&self) -> Option<u64> {
        self.metrics
            .as_ref()
            .map(|metrics| metrics.value_fetch_time.mean())
            .expect(METRICS_NOT_ENABLED)
    }
}

impl ActiveMetrics {
    fn timer(&self, metric: Metric) -> &Timer {
        match metric {
            Metric::PageFetchTime => &self.page_fetch_time,
            Metric::ValueFetchTime => &self.value_fetch_time,
            _ => panic!("Specified metric is not a Timer"),
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

    fn add(&self, duration: Duration) {
        self.number_of_records.fetch_add(1, Ordering::Relaxed);
        self.sum
            .fetch_add(duration.as_nanos() as u64, Ordering::Relaxed);
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
