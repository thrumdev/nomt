//! Thread primitives that differ under shuttle (`--cfg shuttle`), whose threads all run on one OS
//! thread: NOMT's threads must be shuttle threads, and its thread-locals per shuttle thread. Library
//! dependencies are swapped by `shuttle-shims/` instead; std can't be patched, hence this module.

#[cfg(shuttle)]
pub(crate) use shuttle::{thread::Builder as ThreadBuilder, thread_local};
#[cfg(not(shuttle))]
pub(crate) use std::{thread::Builder as ThreadBuilder, thread_local};
