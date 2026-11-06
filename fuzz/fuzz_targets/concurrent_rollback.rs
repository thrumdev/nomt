//! Rollbacks issued from any thread while other threads hold sessions, finish and commit them,
//! with a small rollback log so old deltas get pruned.

#![no_main]

mod concurrent;

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| concurrent::run(data, concurrent::Mode::Ordered));
