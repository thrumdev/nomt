use anyhow::{bail, ensure, Result};
use bitvec::prelude::*;
use std::{
    cmp::Ordering,
    collections::BTreeMap,
    fs::File,
    io::{ErrorKind, Read, Seek},
};

use crate::beatree::{
    allocator::PageNumber,
    bbn,
    branch::{self as branch_node, BranchNode, BRANCH_NODE_SIZE, BRANCH_NODE_BODY_SIZE},
    index::Index, 
    leaf::{
        node::{LeafNode, LeafBuilder, LEAF_NODE_BODY_SIZE},
        store::{LeafStoreReader, LeafStoreWriter},
    }, Key,
};

use super::{BRANCH_MERGE_THRESHOLD, BRANCH_BULK_SPLIT_TARGET, BRANCH_BULK_SPLIT_THRESHOLD, BranchId};

pub struct BaseBranch {
    pub node: BranchNode,
    pub id: BranchId,
    pub iter_pos: usize,
}

pub struct ActiveBranch {
    // the 'base' node we are working from. does not exist if DB is empty.
    base: Option<BaseBranch>,
    // the cutoff key, which determines if an operation is in-scope.
    // does not exist for the last branch in the database.
    cutoff: Option<Key>,
    // a separator key which is to be possibly deleted and will be at the point that another
    // separator greater than it is ingested.
    possibly_deleted: Option<Key>,
    ops: Vec<(Key, Option<PageNumber>)>,
}

impl ActiveBranch {
    pub fn new(base: Option<BaseBranch>, cutoff: Option<Key>) -> Self {
        ActiveBranch {
            base,
            cutoff,
            possibly_deleted: None,
            ops: Vec::new(),
        }
    }

    pub fn ingest(&mut self, key: Key, pn: PageNumber) {
        todo!()
    } 

    pub fn complete(&mut self) {
        todo!()
    }

    pub fn is_in_scope(&self, key: &Key) -> bool {
        self.cutoff.map_or(true, |k| *key < k)
    }

    pub fn possibly_delete(&mut self, key: Key) {
        self.possibly_deleted = Some(key);
    }
}

struct BranchGauge {
    first_separator: Option<Key>,
    prefix_len: usize,
    // total separator len, not counting prefix.
    separator_len: usize,
    n: usize,
}

impl BranchGauge {
    fn new() -> Self {
        BranchGauge {
            first_separator: None,
            prefix_len: 0,
            separator_len: 0,
            n: 0,
        }
    }

    fn ingest(&mut self, key: Key) {
        let Some(ref first) = self.first_separator else {
            self.first_separator = Some(key);
            self.separator_len = separator_len(&key);
            self.prefix_len = self.separator_len;

            self.n = 1;
            return
        };

        self.separator_len = std::cmp::max(separator_len(&key), self.separator_len);
        self.prefix_len = prefix_len(first, &key);
        self.n += 1;
    }

    fn body_size(&self) -> usize {
        branch_node::body_size(self.prefix_len, self.separator_len - self.prefix_len, self.n)
    }
}

fn prefix_len(key_a: &Key, key_b: &Key) -> usize {
    key_a.view_bits::<Lsb0>()
        .iter()
        .zip(key_b.view_bits::<Lsb0>().iter())
        .take_while(|(a, b)| a == b)
        .count()
}

fn separator_len(key: &Key) -> usize {
    std::cmp::min(1, key.view_bits::<Lsb0>().last_one().unwrap_or(1))
}
