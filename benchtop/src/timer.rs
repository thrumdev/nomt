use std::{cell::RefCell, collections::HashMap, rc::Rc};

// At least three spans are expected to be measured
// + `workload`
// + `read`
// + `commit_and_prove`
pub struct Timer {
    name: String,
    spans: HashMap<&'static str, Rc<RefCell<hdrhistogram::Histogram<u64>>>>,
}

impl Timer {
    pub fn new(name: String) -> Self {
        Self {
            name,
            spans: HashMap::new(),
        }
    }

    pub fn record_span(&mut self, span_name: &'static str) -> impl Drop {
        struct RecordSpan {
            h: Rc<RefCell<hdrhistogram::Histogram<u64>>>,
            start: std::time::Instant,
        }
        impl Drop for RecordSpan {
            fn drop(&mut self) {
                let elapsed = self.start.elapsed().as_nanos() as u64;
                self.h.borrow_mut().record(elapsed).unwrap();
            }
        }

        let h = self.spans.entry(span_name).or_insert_with(|| Rc::new(RefCell::new(
            hdrhistogram::Histogram::<u64>::new(3).unwrap(),
        )));

        RecordSpan {
            h: h.clone(),
            start: std::time::Instant::now(),
        }
    }

    pub fn get_last_workload_duration(&self) -> anyhow::Result<u64> {
        let h = self
            .spans
            .get("workload")
            .ok_or(anyhow::anyhow!("`workload` span not recordered"))?;

        Ok(h.borrow()
            .iter_recorded()
            .last()
            .ok_or(anyhow::anyhow!("No recorded value for `workload` span"))?
            .value_iterated_to())
    }

    pub fn get_mean_workload_duration(&self) -> anyhow::Result<u64> {
        Ok(self
            .spans
            .get("workload")
            .ok_or(anyhow::anyhow!("`workload` span not recordered"))?
            .borrow()
            .mean() as u64)
    }

    pub fn print(&mut self) {
        println!("{}", self.name);

        let expected_spans = ["workload", "read", "commit_and_prove"];

        // print expectd spans in order
        for span_name in expected_spans {
            let h = self.spans.get(span_name);
            match h {
                Some(h) => println!(
                    "  mean {}: {}",
                    span_name,
                    pretty_display_ns(h.borrow().mean() as u64)
                ),
                None => println!("{} not measured", span_name),
            };
        }

        // print all other measured spans
        for (span_name, h) in &self.spans {
            if expected_spans.contains(span_name) {
                continue;
            }

            println!(
                "  mean {}: {}",
                span_name,
                pretty_display_ns(h.borrow().mean() as u64)
            )
        }
    }
}

pub fn pretty_display_ns(ns: u64) -> String {
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
