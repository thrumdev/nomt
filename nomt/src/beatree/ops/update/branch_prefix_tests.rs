use super::branch_updater::{
    tests::make_branch, BaseBranch, BranchGauge, BranchUpdater, DigestResult, HandleNewBranch,
};
use crate::beatree::{
    allocator::PageNumber,
    branch::{node::get_key, BranchNode, BRANCH_NODE_BODY_SIZE},
    ops::bit_ops::separator_len,
    Key,
};
use crate::io::PagePool;

#[derive(Default)]
struct Collect(Vec<(Key, BranchNode)>);

impl HandleNewBranch for Collect {
    fn handle_new_branch(
        &mut self,
        separator: Key,
        node: BranchNode,
        _cutoff: Option<Key>,
    ) -> std::io::Result<()> {
        self.0.push((separator, node));
        Ok(())
    }
}

fn gauge_body_size(keys: impl Iterator<Item = Key>) -> usize {
    let mut gauge = BranchGauge::default();
    for k in keys {
        gauge.ingest_key(k, separator_len(&k));
    }
    gauge.body_size()
}

fn set_bit(key: &mut Key, bit: usize) {
    key[bit / 8] |= 0x80 >> (bit % 8);
}

fn rebuild_with_tail(base: Vec<(Key, usize)>, tail: (Key, u32)) -> BranchNode {
    let pool = PagePool::new();
    let mut updater = BranchUpdater::new(pool, Some(BaseBranch::new(make_branch(base))), None);
    updater.ingest(tail.0, Some(PageNumber(tail.1)));
    let mut out = Collect::default();
    let DigestResult::Finished = updater.digest(&mut out).unwrap() else {
        panic!("unexpected NeedsMerge")
    };
    assert_eq!(out.0.len(), 1);
    out.0.pop().unwrap().1
}

fn assert_node_matches(node: &BranchNode, expected: &[(Key, u32)]) {
    assert_eq!(node.n() as usize, expected.len());
    for (i, (key, pn)) in expected.iter().enumerate() {
        assert_eq!(
            node.node_pointer(i),
            *pn,
            "node_pointer({i}) changed (n = {})",
            node.n()
        );
        assert_eq!(get_key(node, i), *key, "separator {i} changed");
    }
}

// The all-zero first separator is stored with zero bits once the node has a prefix. Rebuilding
// the node with a shorter prefix must not make it longer than the gauge counted.
#[test]
fn kept_first_separator_fits_the_gauge_when_the_prefix_shrinks() {
    let mut base = vec![([0u8; 32], 4096usize)];
    for i in 0..398usize {
        let mut k = [0u8; 32];
        k[1] = 0x10 | (i >> 8) as u8;
        k[2] = i as u8;
        let len = 43 + (i < 343) as usize;
        set_bit(&mut k, len - 1);
        assert_eq!(separator_len(&k), len);
        base.push((k, 5000 + i));
    }
    let mut tail = [0u8; 32];
    tail[1] = 0x20;

    let mut expected: Vec<(Key, u32)> = base.iter().map(|(k, pn)| (*k, *pn as u32)).collect();
    expected.push((tail, 9999));

    assert_eq!(
        gauge_body_size(expected.iter().map(|(k, _)| *k)),
        BRANCH_NODE_BODY_SIZE
    );

    let node = rebuild_with_tail(base, (tail, 9999));
    assert_eq!(node.prefix_len(), 10);
    assert_node_matches(&node, &expected);
}

#[test]
fn kept_short_first_separator_does_not_spill_into_node_pointers() {
    let mut k0 = [0u8; 32];
    k0[0] = 0x80;
    let mut tail = [0u8; 32];
    tail[0] = 0xC0;
    let mid = |i: usize| {
        let mut k = k0;
        k[16..18].copy_from_slice(&(i as u16).to_be_bytes());
        k
    };

    let mut found = None;
    'search: for m in 100..250usize {
        for ext in (143..256).rev() {
            let mut keys = vec![k0];
            keys.extend((1..=m).map(mid));
            if ext > 143 {
                set_bit(&mut keys[m], ext);
            }
            keys.push(tail);
            if gauge_body_size(keys.iter().copied()) == BRANCH_NODE_BODY_SIZE {
                found = Some(keys);
                break 'search;
            }
        }
    }
    let keys = found.expect("no exactly-full configuration found");

    let expected: Vec<(Key, u32)> = keys
        .iter()
        .enumerate()
        .map(|(i, k)| (*k, 70_000 + i as u32))
        .collect();
    let base = expected[..expected.len() - 1]
        .iter()
        .map(|(k, pn)| (*k, *pn as usize))
        .collect();

    let node = rebuild_with_tail(base, *expected.last().unwrap());
    assert_eq!(node.prefix_len(), 1);
    assert_node_matches(&node, &expected);
}

#[test]
fn carried_prefix_is_copied_in_full_when_the_prefix_shrinks_by_many_bits() {
    let mut expected = Vec::new();
    for i in 0..8u8 {
        let mut k = [0u8; 32];
        k[0] = 0x07;
        k[1..8].fill(0xff);
        k[8] = 0xf1 + 2 * i;
        expected.push((k, 100 + i as u32));
    }
    let base = expected.iter().map(|(k, pn)| (*k, *pn as usize)).collect();
    let mut tail = [0u8; 32];
    tail[0] = 0x08;
    expected.push((tail, 99));

    let node = rebuild_with_tail(base, (tail, 99));
    assert_eq!(node.prefix_len(), 4);
    assert_node_matches(&node, &expected);
}

#[test]
fn long_separator_keeps_its_last_bits_when_the_prefix_grows() {
    let mut k0 = [0u8; 32];
    k0[0] = 0x40;
    let mut k1 = [0u8; 32];
    k1[..9].copy_from_slice(&[0xbd, 0x9b, 0xa9, 0x28, 0xdd, 0x85, 0x46, 0xd8, 0xb0]);
    let mut k2 = [0u8; 32];
    k2[..16].copy_from_slice(&[
        0xbd, 0x9b, 0xa9, 0x28, 0xdd, 0x85, 0x46, 0xd8, 0xc4, 0x00, 0x78, 0x61, 0xb5, 0x4b, 0xb3,
        0x20,
    ]);

    let pool = PagePool::new();
    let base = make_branch(vec![(k0, 10), (k1, 11), (k2, 12)]);
    let mut updater = BranchUpdater::new(pool, Some(BaseBranch::new(base)), None);
    updater.ingest(k0, None);
    let mut out = Collect::default();
    let DigestResult::Finished = updater.digest(&mut out).unwrap() else {
        panic!("unexpected NeedsMerge")
    };
    assert_eq!(out.0.len(), 1);
    let node = &out.0[0].1;
    assert_eq!(node.prefix_len(), 65);
    assert_node_matches(node, &[(k1, 11), (k2, 12)]);
}

// Rebuild nodes whose keys share a long prefix, inserting or deleting one key that does not share
// it, so that the prefix shrinks or grows by many bits.
#[test]
fn rebuilt_nodes_keep_every_separator_and_pointer() {
    let mut state = 0x9E3779B97F4A7C15u64;
    let mut next = move |bound: usize| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state % bound as u64) as usize
    };

    for round in 0..4000 {
        let prefix_bits = 1 + next(200);
        let mut prefix = [0u8; 32];
        for bit in 0..prefix_bits {
            if next(2) == 1 {
                set_bit(&mut prefix, bit);
            }
        }

        let mut keys = std::collections::BTreeSet::new();
        if next(2) == 0 {
            keys.insert(prefix);
        }
        let n = 2 + next(30);
        while keys.len() < n {
            let mut key = prefix;
            let suffix_bits = 1 + next(40);
            for bit in 0..suffix_bits {
                if next(2) == 1 {
                    set_bit(&mut key, prefix_bits + bit);
                }
            }
            set_bit(&mut key, prefix_bits + suffix_bits - 1);
            keys.insert(key);
        }

        let mut outsider = prefix;
        let flipped = next(prefix_bits);
        outsider[flipped / 8] ^= 0x80 >> (flipped % 8);
        for bit in flipped + 1..256 {
            outsider[bit / 8] &= !(0x80 >> (bit % 8));
        }
        if next(2) == 1 {
            set_bit(&mut outsider, flipped + 1 + next(255 - flipped));
        }

        let insert = next(2) == 0;
        let mut all: Vec<Key> = keys.iter().copied().collect();
        all.push(outsider);
        all.sort();
        all.dedup();
        let with_pn = |keys: &[Key]| -> Vec<(Key, u32)> {
            keys.iter()
                .map(|k| (*k, 1000 + all.binary_search(k).unwrap() as u32))
                .collect()
        };
        let without: Vec<Key> = all.iter().copied().filter(|k| *k != outsider).collect();
        let (base, expected) = if insert {
            (with_pn(&without), with_pn(&all))
        } else {
            (with_pn(&all), with_pn(&without))
        };

        let pool = PagePool::new();
        let base = make_branch(base.iter().map(|(k, pn)| (*k, *pn as usize)).collect());
        let mut updater = BranchUpdater::new(pool, Some(BaseBranch::new(base)), None);
        let outsider_pn = expected
            .iter()
            .find(|(k, _)| *k == outsider)
            .map(|(_, pn)| *pn);
        updater.ingest(outsider, outsider_pn.map(PageNumber));
        let mut out = Collect::default();
        let DigestResult::Finished = updater.digest(&mut out).unwrap() else {
            panic!("unexpected NeedsMerge")
        };

        out.0.sort_by_key(|(separator, _)| *separator);
        let rebuilt: Vec<(Key, u32)> = out
            .0
            .iter()
            .flat_map(|(_, node)| {
                (0..node.n() as usize).map(move |i| (get_key(node, i), node.node_pointer(i)))
            })
            .collect();
        assert!(
            rebuilt == expected,
            "round {round}: prefix {prefix_bits} bits, insert {insert}"
        );
    }
}
