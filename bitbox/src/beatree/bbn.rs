//! BBN IO is write-only during normal operation.
//!
//! An essential component of the BBN IO is the freelist. It tracks the page numbers ready to be
//! overwritten.
//!
//! The updates to BBN storage applied in batches. A batch contains a list of nodes that should be
//! written. Those could be updated nodes (bbn_seqn of which is already present in the file), new
//! nodes (nodes with unique bbn_seqn) and tombstone nodes (a node that marks end-of-life of the
//! given bbn_seqn).
//!
//! The general algorithm is as follows:
//!
//! 1. allocate PNs from the freelist for all the nodes written.
//! 2. if we are N PNs short then increase the file space by calling ftruncate(M), where M is the
//! next multiply of the bulk allocation size.
//! 3. issue IO write requests for each node-PN pair to the BBN storage fd.
//! 4. fsync on the BBN storage fd.
//! 5. add all the deallocated BBNs into the freelist. The tombstones are considered immediately
//! deallocated.

/// The page number of a bottom-level branch node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BbnPn(u32);

pub struct BbnDumper {
}
