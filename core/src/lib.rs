//! Core operations and types within the Nearly Optimal Merkle Trie.
//!
//! This crate defines the schema and basic operations over the merkle trie in a backend-agnostic
//! manner.
//!
//! The core types and proof verification routines of this crate do not require the
//! standard library. Generating proofs does require the standard library.

#![cfg_attr(not(feature = "std"), no_std)]

pub mod trie;
