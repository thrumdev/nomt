//! Shuttle-backed subset of `crossbeam-channel` 0.5: `bounded`/`unbounded` MPMC channels,
//! `Select::{recv, ready}` and a `select!` limited to `recv(..) -> .. => ..` arms. Zero-capacity
//! (rendezvous) channels can be created but not sent on while a receiver is alive: NOMT only uses
//! `bounded(0)` for a placeholder sender whose receiver is dropped at once.

use shuttle::sync::{Condvar, Mutex};
use std::{collections::VecDeque, fmt, sync::Arc};

struct Waker {
    fired: Mutex<bool>,
    cv: Condvar,
}

impl Waker {
    fn fire(&self) {
        *self.fired.lock().unwrap() = true;
        self.cv.notify_all();
    }
}

struct State<T> {
    queue: VecDeque<T>,
    senders: usize,
    receivers: usize,
    selectors: Vec<Arc<Waker>>,
}

struct Chan<T> {
    cap: Option<usize>,
    state: Mutex<State<T>>,
    cv: Condvar,
}

impl<T> Chan<T> {
    /// Wakes blocked receivers and selectors: a message was pushed or the last sender left.
    fn notify_ready(&self, state: &State<T>) {
        self.cv.notify_all();
        for w in &state.selectors {
            w.fire();
        }
    }
}

trait Selectable {
    fn is_ready(&self) -> bool;
    fn register(&self, waker: &Arc<Waker>);
    fn unregister(&self, waker: &Arc<Waker>);
}

impl<T> Selectable for Chan<T> {
    fn is_ready(&self) -> bool {
        let s = self.state.lock().unwrap();
        !s.queue.is_empty() || s.senders == 0
    }
    fn register(&self, waker: &Arc<Waker>) {
        self.state.lock().unwrap().selectors.push(waker.clone());
    }
    fn unregister(&self, waker: &Arc<Waker>) {
        self.state
            .lock()
            .unwrap()
            .selectors
            .retain(|w| !Arc::ptr_eq(w, waker));
    }
}

pub struct Sender<T>(Arc<Chan<T>>);
pub struct Receiver<T>(Arc<Chan<T>>);

fn channel<T>(cap: Option<usize>) -> (Sender<T>, Receiver<T>) {
    let chan = Arc::new(Chan {
        cap,
        state: Mutex::new(State {
            queue: VecDeque::new(),
            senders: 1,
            receivers: 1,
            selectors: Vec::new(),
        }),
        cv: Condvar::new(),
    });
    (Sender(chan.clone()), Receiver(chan))
}

pub fn bounded<T>(cap: usize) -> (Sender<T>, Receiver<T>) {
    channel(Some(cap))
}

pub fn unbounded<T>() -> (Sender<T>, Receiver<T>) {
    channel(None)
}

impl<T> Sender<T> {
    pub fn send(&self, msg: T) -> Result<(), SendError<T>> {
        let chan = &*self.0;
        let mut s = chan.state.lock().unwrap();
        loop {
            if s.receivers == 0 {
                return Err(SendError(msg));
            }
            assert_ne!(chan.cap, Some(0), "rendezvous channels are not supported");
            match chan.cap {
                Some(cap) if s.queue.len() >= cap => s = chan.cv.wait(s).unwrap(),
                _ => break,
            }
        }
        s.queue.push_back(msg);
        chan.notify_ready(&s);
        Ok(())
    }
}

impl<T> Clone for Sender<T> {
    fn clone(&self) -> Self {
        self.0.state.lock().unwrap().senders += 1;
        Sender(self.0.clone())
    }
}

impl<T> Drop for Sender<T> {
    fn drop(&mut self) {
        let mut s = self.0.state.lock().unwrap();
        s.senders -= 1;
        if s.senders == 0 {
            self.0.notify_ready(&s);
        }
    }
}

impl<T> Receiver<T> {
    fn pop(&self, s: &mut State<T>) -> Option<T> {
        let msg = s.queue.pop_front()?;
        self.0.cv.notify_all();
        Some(msg)
    }

    pub fn recv(&self) -> Result<T, RecvError> {
        let mut s = self.0.state.lock().unwrap();
        loop {
            if let Some(msg) = self.pop(&mut s) {
                return Ok(msg);
            }
            if s.senders == 0 {
                return Err(RecvError);
            }
            s = self.0.cv.wait(s).unwrap();
        }
    }

    pub fn try_recv(&self) -> Result<T, TryRecvError> {
        let mut s = self.0.state.lock().unwrap();
        match self.pop(&mut s) {
            Some(msg) => Ok(msg),
            None if s.senders == 0 => Err(TryRecvError::Disconnected),
            None => Err(TryRecvError::Empty),
        }
    }
}

impl<T> Clone for Receiver<T> {
    fn clone(&self) -> Self {
        self.0.state.lock().unwrap().receivers += 1;
        Receiver(self.0.clone())
    }
}

impl<T> Drop for Receiver<T> {
    fn drop(&mut self) {
        let mut s = self.0.state.lock().unwrap();
        s.receivers -= 1;
        if s.receivers == 0 {
            self.0.cv.notify_all();
        }
    }
}

pub struct IntoIter<T>(Receiver<T>);

impl<T> Iterator for IntoIter<T> {
    type Item = T;
    fn next(&mut self) -> Option<T> {
        self.0.recv().ok()
    }
}

impl<T> IntoIterator for Receiver<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;
    fn into_iter(self) -> IntoIter<T> {
        IntoIter(self)
    }
}

/// Waits on several receivers. `ready` returns the index of one that has a message or is
/// disconnected; when several are, shuttle's random source picks, so the choice is explored.
pub struct Select<'a> {
    chans: Vec<&'a dyn Selectable>,
}

impl<'a> Select<'a> {
    pub fn new() -> Self {
        Select { chans: Vec::new() }
    }

    pub fn recv<T>(&mut self, r: &'a Receiver<T>) -> usize {
        self.chans.push(&*r.0);
        self.chans.len() - 1
    }

    pub fn ready(&mut self) -> usize {
        let waker = Arc::new(Waker {
            fired: Mutex::new(false),
            cv: Condvar::new(),
        });
        for c in &self.chans {
            c.register(&waker);
        }
        let index = loop {
            let ready: Vec<usize> = (0..self.chans.len())
                .filter(|&i| self.chans[i].is_ready())
                .collect();
            if !ready.is_empty() {
                use shuttle::rand::Rng;
                break ready[shuttle::rand::thread_rng().gen_range(0..ready.len())];
            }
            let mut fired = waker.fired.lock().unwrap();
            while !*fired {
                fired = waker.cv.wait(fired).unwrap();
            }
            *fired = false;
        };
        for c in &self.chans {
            c.unregister(&waker);
        }
        index
    }
}

/// Only `recv(rx) -> pat => body` arms. Assumes the selected receiver isn't drained by another
/// consumer between `ready` and `recv`, which holds for NOMT's single-consumer uses.
#[macro_export]
macro_rules! select {
    (@arms $index:ident, $k:expr;) => { unreachable!() };
    (@arms $index:ident, $k:expr; recv($rx:expr) -> $res:pat => $body:expr $(, recv($rxs:expr) -> $ress:pat => $bodies:expr)*) => {
        if $index == $k {
            let $res = $rx.recv();
            $body
        } else {
            $crate::select!(@arms $index, $k + 1; $(recv($rxs) -> $ress => $bodies),*)
        }
    };
    ($(recv($rx:expr) -> $res:pat => $body:expr),+ $(,)?) => {{
        let mut __sel = $crate::Select::new();
        $( __sel.recv(&$rx); )+
        let __index = __sel.ready();
        $crate::select!(@arms __index, 0usize; $(recv($rx) -> $res => $body),+)
    }};
}

#[derive(PartialEq, Eq, Clone, Copy)]
pub struct SendError<T>(pub T);

impl<T> fmt::Debug for SendError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad("SendError { .. }")
    }
}

impl<T> fmt::Display for SendError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad("sending on a disconnected channel")
    }
}

impl<T> std::error::Error for SendError<T> {}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub struct RecvError;

impl fmt::Display for RecvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad("receiving on an empty and disconnected channel")
    }
}

impl std::error::Error for RecvError {}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub enum TryRecvError {
    Empty,
    Disconnected,
}

impl fmt::Display for TryRecvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TryRecvError::Empty => f.pad("receiving on an empty channel"),
            TryRecvError::Disconnected => f.pad("receiving on an empty and disconnected channel"),
        }
    }
}

impl std::error::Error for TryRecvError {}
