pub struct Timer {
    name: String,
    h: hdrhistogram::Histogram<u64>,
    ops: u64,
}

impl Timer {
    pub fn new(name: String) -> Self {
        Self {
            name,
            h: hdrhistogram::Histogram::<u64>::new(3).unwrap(),
            ops: 0,
        }
    }

    pub fn record<'a>(&'a mut self) -> impl Drop + 'a {
        struct RecordSpan<'a> {
            h: &'a mut hdrhistogram::Histogram<u64>,
            ops: &'a mut u64,
            start: std::time::Instant,
        }
        impl Drop for RecordSpan<'_> {
            fn drop(&mut self) {
                let elapsed = self.start.elapsed().as_nanos() as u64;
                self.h.record(elapsed).unwrap();
                *self.ops += 1;
            }
        }
        RecordSpan {
            h: &mut self.h,
            ops: &mut self.ops,
            start: std::time::Instant::now(),
        }
    }

    pub fn print(&mut self) {
        println!("{}", self.name);
        println!("  ops={}", self.ops);
        for q in [0.001, 0.01, 0.25, 0.50, 0.75, 0.95, 0.99, 0.999] {
            let lat = self.h.value_at_quantile(q);
            println!("  {}th: {}", q * 100.0, pretty_display_ns(lat));
        }
        println!("  mean={}", pretty_display_ns(self.h.mean() as u64));
        println!();
        self.ops = 0;
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
