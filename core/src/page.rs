//! TODO: add docs

/// Depth of the rootless sub-binary tree stored in a page
pub const DEPTH: usize = 6;

// Total number of nodes stored in one Page. It depends on the `DEPTH`
// of the rootless sub-binary tree stored in a page
pub const NODES_PER_PAGE: usize = (1 << DEPTH) - 2;
