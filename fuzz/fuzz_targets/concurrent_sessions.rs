//! Sessions, nested sessions, reads, proofs and commits from several threads at once, with
//! commits racing each other for the write lock.

#![no_main]

mod concurrent;

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| concurrent::run(data, concurrent::Mode::Racing));
