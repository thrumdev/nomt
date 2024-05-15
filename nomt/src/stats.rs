use std::{
    cell::RefCell,
    collections::HashMap,
    sync::{
        mpsc::{self, Receiver, Sender},
        Arc,
    },
};

static mut STATISTICS_SENDER: Option<Sender<Statistic>> = None;

pub enum Statistic {
    Time(&'static str, u64),
    Counter(&'static str),
    Print,
}

impl Statistic {
    pub fn send(self) {
        // TODO: maybe be silent?
        let Some(sender) = (unsafe { crate::stats::STATISTICS_SENDER.clone() }) else {
            panic!("Not initialized Statistics Channel")
        };
        sender.send(self).expect("Impossible Send Statistic");
    }
}

pub fn init() {
    let (tx, rx) = mpsc::channel();

    unsafe {
        STATISTICS_SENDER = Some(tx);
    }

    std::thread::spawn(move || {
        let mut timings = HashMap::<&'static str, hdrhistogram::Histogram<u64>>::new();
        let mut counters = HashMap::<&'static str, Vec<u64>>::new();

        loop {
            match rx.recv() {
                Ok(Statistic::Time(name, elapsed)) => {
                    timings
                        .entry(name)
                        .or_insert(hdrhistogram::Histogram::<u64>::new(3).expect("TODO"))
                        .record(elapsed);
                }
                Ok(Statistic::Counter(name)) => {
                    *counters.entry(name).or_insert(vec![0]).last_mut().unwrap() += 1;
                }
                Ok(Statistic::Print) => {
                    println!("Recored statistics:");
                    for (name, h) in timings.iter() {
                        println!("  {name}: {}", pretty_display_ns(h.mean() as u64));
                    }
                    for (name, cs) in counters.iter() {
                        let c = cs.iter().sum::<u64>() / cs.len() as u64;
                        println!("  {name}: {c} times",);
                    }
                    break;
                }
                // TODO: quit silently or not?
                Err(_) => break,
            };
        }
    });
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

pub struct Timer(&'static str, std::time::Instant);

impl Timer {
    pub fn new(name: &'static str, now: std::time::Instant) -> Self {
        Timer(name, now)
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        let elapsed = self.1.elapsed().as_nanos() as u64;
        crate::stats::Statistic::Time(self.0, elapsed).send()
    }
}

#[macro_export]
macro_rules! record {
    ($name: literal) => {
        crate::stats::Timer::new($name, std::time::Instant::now())
    };
}

#[macro_export]
macro_rules! counter {
    ($name: literal) => {
        crate::stats::Statistic::Counter($name).send()
    };
}

#[macro_export]
macro_rules! print_metrics {
    () => {
        crate::stats::Statistic::Print.send()
    };
}
