//! Core operations and types within the Nearly Optimal Merkle Trie.
//!
//! This crate defines the schema and basic operations over the merkle trie in a backend-agnostic
//! manner.
//!
//! The core types and proof verification routines of this crate do not require the
//! standard library, but do require Rust's alloc crate.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod proof;
pub mod trie;
