pub struct Timer {
    name: String,
    h: hdrhistogram::Histogram<u64>,
    ops: u64,
}

impl Timer {
    pub fn new(name: String) -> Self {
        Self {
            name,
            h: hdrhistogram::Histogram::<u64>::new_with_bounds(1, 1000000000, 3).unwrap(),
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
            println!("  {}th: {} ns", q * 100.0, lat);
        }
        println!("   mean={} ns", self.h.mean());
        println!();
        self.ops = 0;
    }
}
