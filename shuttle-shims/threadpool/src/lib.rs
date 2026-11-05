//! `threadpool` 1.8 under shuttle. Workers are shuttle threads that exit once every `ThreadPool`
//! handle is dropped and the queue is drained, so an execution can end cleanly.

use shuttle::sync::{Condvar, Mutex};
use std::{collections::VecDeque, sync::Arc};

type Job = Box<dyn FnOnce() + Send + 'static>;

struct State {
    queue: VecDeque<Job>,
    active: usize,
    handles: usize,
}

struct Inner {
    state: Mutex<State>,
    work: Condvar,
    idle: Condvar,
    max: usize,
}

pub struct ThreadPool(Arc<Inner>);

impl ThreadPool {
    pub fn new(num_threads: usize) -> Self {
        Builder::new().num_threads(num_threads).build()
    }

    pub fn with_name(name: String, num_threads: usize) -> Self {
        Builder::new()
            .num_threads(num_threads)
            .thread_name(name)
            .build()
    }

    pub fn execute<F: FnOnce() + Send + 'static>(&self, job: F) {
        self.0.state.lock().unwrap().queue.push_back(Box::new(job));
        self.0.work.notify_one();
    }

    pub fn join(&self) {
        let mut s = self.0.state.lock().unwrap();
        while !s.queue.is_empty() || s.active > 0 {
            s = self.0.idle.wait(s).unwrap();
        }
    }

    pub fn max_count(&self) -> usize {
        self.0.max
    }

    pub fn queued_count(&self) -> usize {
        self.0.state.lock().unwrap().queue.len()
    }

    pub fn active_count(&self) -> usize {
        self.0.state.lock().unwrap().active
    }
}

impl Clone for ThreadPool {
    fn clone(&self) -> Self {
        self.0.state.lock().unwrap().handles += 1;
        ThreadPool(self.0.clone())
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        self.0.state.lock().unwrap().handles -= 1;
        self.0.work.notify_all();
    }
}

fn worker(inner: Arc<Inner>) {
    loop {
        let job = {
            let mut s = inner.state.lock().unwrap();
            loop {
                if let Some(job) = s.queue.pop_front() {
                    s.active += 1;
                    break job;
                }
                if s.handles == 0 {
                    return;
                }
                s = inner.work.wait(s).unwrap();
            }
        };
        job();
        inner.state.lock().unwrap().active -= 1;
        inner.idle.notify_all();
    }
}

#[derive(Default)]
pub struct Builder {
    num_threads: Option<usize>,
    name: Option<String>,
}

impl Builder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn num_threads(mut self, n: usize) -> Self {
        assert!(n > 0);
        self.num_threads = Some(n);
        self
    }

    pub fn thread_name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }

    pub fn thread_stack_size(self, _: usize) -> Self {
        self
    }

    pub fn build(self) -> ThreadPool {
        let max = self.num_threads.unwrap_or(1);
        let inner = Arc::new(Inner {
            state: Mutex::new(State {
                queue: VecDeque::new(),
                active: 0,
                handles: 1,
            }),
            work: Condvar::new(),
            idle: Condvar::new(),
            max,
        });
        for _ in 0..max {
            let mut b = shuttle::thread::Builder::new();
            if let Some(name) = &self.name {
                b = b.name(name.clone());
            }
            let inner = inner.clone();
            b.spawn(move || worker(inner)).unwrap();
        }
        ThreadPool(inner)
    }
}
