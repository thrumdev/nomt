use nomt_core::trie::KeyPath;
use quickcheck::{empty_shrinker, single_shrinker, Arbitrary, Gen};
use rand::{Rng as _, SeedableRng as _};
use rand_pcg::Lcg64Xsh32;
use std::collections::BTreeSet;

const EDGE_PREFIX_LENS: &[usize] = &[0, 1, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255];

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct TestKeyPath(pub KeyPath);

impl TestKeyPath {
    pub fn raw(inner: KeyPath) -> Self {
        Self(inner)
    }

    pub fn account_like(id: u64) -> Self {
        Self(seeded_key(id))
    }

    pub fn diverging_at(bit_depth: usize, right: bool) -> Self {
        assert!(bit_depth < 256);

        let mut key = KeyPath::default();
        set_bit(&mut key, bit_depth, right);
        Self(key)
    }

    pub fn with_prefix_bits(bits: impl IntoIterator<Item = bool>, seed: u64) -> Self {
        let mut key = seeded_key(seed);

        for (i, bit) in bits.into_iter().enumerate() {
            assert!(i < 256);
            set_bit(&mut key, i, bit);
        }

        Self(key)
    }

    pub fn into_inner(self) -> KeyPath {
        self.0
    }
}

impl Arbitrary for TestKeyPath {
    fn arbitrary(g: &mut Gen) -> Self {
        match u8::arbitrary(g) % 100 {
            0..=29 => Self(raw_key(g)),
            30..=54 => Self::account_like(u64::arbitrary(g)),
            55..=79 => {
                let prefix_len = arbitrary_prefix_len(g, true);
                let prefix_bits = random_bits(g, prefix_len);
                Self::with_prefix_bits(prefix_bits, u64::arbitrary(g))
            }
            _ => arbitrary_edge_key(g),
        }
    }

    fn shrink(&self) -> Box<dyn Iterator<Item = Self>> {
        if self.0 == [0; 32] {
            return empty_shrinker();
        }

        let current = *self;
        let mut candidates = vec![Self::raw([0; 32]), Self::account_like(0)];

        if let Some(bit) = first_set_bit(&self.0) {
            candidates.push(Self::diverging_at(bit, true));
        }

        candidates.sort_unstable();
        candidates.dedup();

        Box::new(
            candidates
                .into_iter()
                .filter(move |candidate| *candidate != current),
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DivergingPair {
    pub left: KeyPath,
    pub right: KeyPath,
    pub first_diff_bit: usize,
}

impl DivergingPair {
    pub fn from_prefix_bits(
        bits: impl IntoIterator<Item = bool>,
        left_seed: u64,
        right_seed: u64,
    ) -> Self {
        let prefix_bits = collect_bits(bits);
        assert!(prefix_bits.len() < 256);

        let first_diff_bit = prefix_bits.len();
        let mut left =
            TestKeyPath::with_prefix_bits(prefix_bits.iter().copied(), left_seed).into_inner();
        let mut right =
            TestKeyPath::with_prefix_bits(prefix_bits.iter().copied(), right_seed).into_inner();

        set_bit(&mut left, first_diff_bit, false);
        set_bit(&mut right, first_diff_bit, true);

        Self {
            left,
            right,
            first_diff_bit,
        }
    }
}

impl Arbitrary for DivergingPair {
    fn arbitrary(g: &mut Gen) -> Self {
        let first_diff_bit = arbitrary_prefix_len(g, false);
        let prefix_bits = random_bits(g, first_diff_bit);
        Self::from_prefix_bits(prefix_bits, u64::arbitrary(g), u64::arbitrary(g))
    }

    fn shrink(&self) -> Box<dyn Iterator<Item = Self>> {
        single_shrinker(Self {
            left: TestKeyPath::diverging_at(self.first_diff_bit, false).into_inner(),
            right: TestKeyPath::diverging_at(self.first_diff_bit, true).into_inner(),
            first_diff_bit: self.first_diff_bit,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SharedPrefixCluster {
    pub prefix_len_bits: usize,
    pub members: Vec<KeyPath>,
}

impl SharedPrefixCluster {
    /// `count` must satisfy `2 <= count <= max_cluster_members(prefix_bits.len())`,
    /// otherwise the post-dedup length assertion will fire.
    pub fn from_prefix_bits(bits: impl IntoIterator<Item = bool>, count: usize, seed: u64) -> Self {
        let prefix_bits = collect_bits(bits);
        let prefix_len_bits = prefix_bits.len();
        assert!(prefix_len_bits < 256);

        let max_members = max_cluster_members(prefix_len_bits);
        assert!(count >= 2);
        assert!(count <= max_members);

        let extra_width = cluster_extra_width(prefix_len_bits);
        let mut members = Vec::with_capacity(count);

        for index in 0..count {
            let mut key = TestKeyPath::with_prefix_bits(
                prefix_bits.iter().copied(),
                seed.wrapping_add(index as u64),
            )
            .into_inner();

            set_bit(&mut key, prefix_len_bits, index % 2 == 1);

            let suffix_id = index / 2;
            for offset in 0..extra_width {
                let bit = ((suffix_id >> (extra_width - 1 - offset)) & 1) == 1;
                set_bit(&mut key, prefix_len_bits + 1 + offset, bit);
            }

            members.push(key);
        }

        members.sort_unstable();
        members.dedup();
        assert_eq!(members.len(), count);

        Self {
            prefix_len_bits,
            members,
        }
    }
}

impl Arbitrary for SharedPrefixCluster {
    fn arbitrary(g: &mut Gen) -> Self {
        let prefix_len_bits = arbitrary_prefix_len(g, false);
        let prefix_bits = random_bits(g, prefix_len_bits);
        let max_members = max_cluster_members(prefix_len_bits);
        let count = 2 + (usize::arbitrary(g) % (max_members - 1));

        Self::from_prefix_bits(prefix_bits, count, u64::arbitrary(g))
    }

    fn shrink(&self) -> Box<dyn Iterator<Item = Self>> {
        let prefix_bits = prefix_bits(&self.members[0], self.prefix_len_bits);
        let count = if self.members.len() > 2 {
            2
        } else {
            self.members.len()
        };
        let candidate = Self::from_prefix_bits(prefix_bits, count, 0);

        if &candidate == self {
            empty_shrinker()
        } else {
            single_shrinker(candidate)
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UniqueSortedKeyPaths(pub Vec<KeyPath>);

impl UniqueSortedKeyPaths {
    pub fn new(paths: impl IntoIterator<Item = KeyPath>) -> Self {
        let mut unique = paths.into_iter().collect::<Vec<_>>();
        unique.sort_unstable();
        unique.dedup();
        Self(unique)
    }

    pub fn as_slice(&self) -> &[KeyPath] {
        &self.0
    }

    pub fn into_inner(self) -> Vec<KeyPath> {
        self.0
    }
}

impl Arbitrary for UniqueSortedKeyPaths {
    fn arbitrary(g: &mut Gen) -> Self {
        let target_len = usize::arbitrary(g) % 9;
        let mut paths = BTreeSet::new();
        let mut attempts = 0;

        while paths.len() < target_len && attempts < (target_len.saturating_mul(8) + 8) {
            paths.insert(TestKeyPath::arbitrary(g).into_inner());
            attempts += 1;
        }

        let mut fill_seed = 0u64;
        while paths.len() < target_len {
            paths.insert(TestKeyPath::account_like(fill_seed).into_inner());
            fill_seed += 1;
        }

        Self(paths.into_iter().collect())
    }

    fn shrink(&self) -> Box<dyn Iterator<Item = Self>> {
        if self.0.is_empty() {
            return empty_shrinker();
        }

        let next_len = self.0.len() / 2;
        let candidate = Self(self.0[..next_len].to_vec());
        single_shrinker(candidate)
    }
}

pub fn account_path(id: u64) -> KeyPath {
    TestKeyPath::account_like(id).into_inner()
}

pub fn key_diverging_at(bit_depth: usize, right: bool) -> KeyPath {
    TestKeyPath::diverging_at(bit_depth, right).into_inner()
}

pub fn key_with_prefix(bits: impl IntoIterator<Item = bool>) -> KeyPath {
    let mut key = KeyPath::default();

    for (index, bit) in bits.into_iter().enumerate() {
        assert!(index < 256);
        set_bit(&mut key, index, bit);
    }

    key
}

pub fn key_with_prefix_seed(bits: impl IntoIterator<Item = bool>, seed: u64) -> KeyPath {
    TestKeyPath::with_prefix_bits(bits, seed).into_inner()
}

fn collect_bits(bits: impl IntoIterator<Item = bool>) -> Vec<bool> {
    let collected = bits.into_iter().collect::<Vec<_>>();
    assert!(collected.len() <= 256);
    collected
}

fn arbitrary_prefix_len(g: &mut Gen, allow_full: bool) -> usize {
    if bool::arbitrary(g) {
        EDGE_PREFIX_LENS[usize::arbitrary(g) % EDGE_PREFIX_LENS.len()]
    } else if allow_full {
        usize::arbitrary(g) % 257
    } else {
        usize::arbitrary(g) % 256
    }
}

fn arbitrary_edge_key(g: &mut Gen) -> TestKeyPath {
    match u8::arbitrary(g) % 7 {
        0 => TestKeyPath::raw([0; 32]),
        1 => TestKeyPath::raw([0xFF; 32]),
        2 => TestKeyPath::raw([0xAA; 32]),
        3 => TestKeyPath::raw([0x55; 32]),
        4 => TestKeyPath::diverging_at(
            EDGE_PREFIX_LENS[usize::arbitrary(g) % EDGE_PREFIX_LENS.len()],
            true,
        ),
        5 => {
            let mut key = [0; 32];
            let byte = usize::arbitrary(g) % key.len();
            key[byte] = 0xFF;
            TestKeyPath::raw(key)
        }
        _ => TestKeyPath::account_like(0),
    }
}

fn cluster_extra_width(prefix_len_bits: usize) -> usize {
    (255 - prefix_len_bits).min(3)
}

fn first_set_bit(key: &KeyPath) -> Option<usize> {
    (0..256).find(|&index| bit(key, index))
}

fn prefix_bits(key: &KeyPath, prefix_len_bits: usize) -> Vec<bool> {
    (0..prefix_len_bits).map(|index| bit(key, index)).collect()
}

fn random_bits(g: &mut Gen, len: usize) -> Vec<bool> {
    (0..len).map(|_| bool::arbitrary(g)).collect()
}

fn raw_key(g: &mut Gen) -> KeyPath {
    let mut key = [0; 32];
    for byte in &mut key {
        *byte = u8::arbitrary(g);
    }
    key
}

fn seeded_key(seed: u64) -> KeyPath {
    let mut rng_seed = [0; 16];
    rng_seed[..8].copy_from_slice(&seed.to_le_bytes());

    let mut rng = Lcg64Xsh32::from_seed(rng_seed);
    let mut key = [0; 32];

    for chunk in key.chunks_exact_mut(4) {
        chunk.copy_from_slice(&rng.next_u32().to_le_bytes());
    }

    key
}

fn set_bit(key: &mut KeyPath, index: usize, value: bool) {
    let byte = index / 8;
    let shift = 7 - (index % 8);
    let mask = 1 << shift;

    if value {
        key[byte] |= mask;
    } else {
        key[byte] &= !mask;
    }
}

fn bit(key: &KeyPath, index: usize) -> bool {
    let byte = index / 8;
    let shift = 7 - (index % 8);
    ((key[byte] >> shift) & 1) == 1
}

#[cfg(test)]
fn first_difference(left: &KeyPath, right: &KeyPath) -> Option<usize> {
    (0..256).find(|&index| bit(left, index) != bit(right, index))
}

#[cfg(test)]
fn has_shared_prefix(a: &KeyPath, b: &KeyPath, prefix_len_bits: usize) -> bool {
    (0..prefix_len_bits).all(|index| bit(a, index) == bit(b, index))
}

#[cfg(test)]
fn has_exact_cluster_prefix(cluster: &SharedPrefixCluster) -> bool {
    if cluster.members.len() < 2 {
        return false;
    }

    cluster
        .members
        .iter()
        .all(|member| has_shared_prefix(&cluster.members[0], member, cluster.prefix_len_bits))
        && cluster.members.iter().enumerate().any(|(i, left)| {
            cluster.members[i + 1..]
                .iter()
                .any(|right| first_difference(left, right) == Some(cluster.prefix_len_bits))
        })
}

#[cfg(test)]
fn is_sorted_unique(paths: &[KeyPath]) -> bool {
    paths.windows(2).all(|window| window[0] < window[1])
}

fn max_cluster_members(prefix_len_bits: usize) -> usize {
    2usize << cluster_extra_width(prefix_len_bits)
}

#[cfg(test)]
mod tests {
    use super::{
        account_path, bit, first_difference, has_exact_cluster_prefix, is_sorted_unique,
        key_diverging_at, key_with_prefix, key_with_prefix_seed, DivergingPair,
        SharedPrefixCluster, TestKeyPath, UniqueSortedKeyPaths,
    };
    use quickcheck::{Arbitrary, QuickCheck};

    #[test]
    fn diverging_at_differs_at_requested_bit() {
        for bit_depth in [0, 1, 7, 8, 15, 16, 127, 255] {
            let left = key_diverging_at(bit_depth, false);
            let right = key_diverging_at(bit_depth, true);

            assert_eq!(first_difference(&left, &right), Some(bit_depth));
        }
    }

    #[test]
    fn with_prefix_bits_preserves_prefix() {
        let prefix = [true, false, true, true, false, false, true, false, true];
        let key = key_with_prefix_seed(prefix, 17);

        for (index, expected) in prefix.into_iter().enumerate() {
            assert_eq!(bit(&key, index), expected);
        }

        assert_eq!(key, key_with_prefix_seed(prefix, 17));
    }

    #[test]
    fn key_with_prefix_zero_fills_suffix() {
        let key = key_with_prefix([true, false, true, false]);

        assert!(bit(&key, 0));
        assert!(!bit(&key, 1));
        assert!(bit(&key, 2));
        assert!(!bit(&key, 3));
        assert!(!bit(&key, 4));
        assert_eq!(key[1..], [0; 31]);
    }

    #[test]
    fn account_like_is_stable_and_unique_for_nearby_seeds() {
        let mut paths = Vec::new();

        for seed in 0..64 {
            let path = account_path(seed);
            assert_eq!(path, account_path(seed));
            paths.push(path);
        }

        paths.sort_unstable();
        paths.dedup();
        assert_eq!(paths.len(), 64);
    }

    #[test]
    fn unique_sorted_keypaths_sorts_and_deduplicates() {
        let paths = UniqueSortedKeyPaths::new([
            TestKeyPath::account_like(7).into_inner(),
            TestKeyPath::account_like(3).into_inner(),
            TestKeyPath::account_like(7).into_inner(),
        ]);

        assert!(is_sorted_unique(paths.as_slice()));
        assert_eq!(paths.as_slice().len(), 2);
    }

    #[test]
    fn shared_prefix_cluster_shares_prefix_and_diverges() {
        let cluster = SharedPrefixCluster::from_prefix_bits([true, false, true, false], 4, 11);

        assert_eq!(cluster.prefix_len_bits, 4);
        assert!(has_exact_cluster_prefix(&cluster));
    }

    #[test]
    fn quickcheck_diverging_pair_invariants() {
        fn property(pair: DivergingPair) -> bool {
            if first_difference(&pair.left, &pair.right) != Some(pair.first_diff_bit) {
                return false;
            }

            pair.shrink().take(8).all(|shrunk| {
                first_difference(&shrunk.left, &shrunk.right) == Some(shrunk.first_diff_bit)
            })
        }

        QuickCheck::new()
            .tests(64)
            .quickcheck(property as fn(DivergingPair) -> bool);
    }

    #[test]
    fn quickcheck_shared_prefix_cluster_invariants() {
        fn property(cluster: SharedPrefixCluster) -> bool {
            if !has_exact_cluster_prefix(&cluster) {
                return false;
            }

            cluster
                .shrink()
                .take(8)
                .all(|shrunk| has_exact_cluster_prefix(&shrunk))
        }

        QuickCheck::new()
            .tests(64)
            .quickcheck(property as fn(SharedPrefixCluster) -> bool);
    }

    #[test]
    fn quickcheck_unique_sorted_keypaths_invariants() {
        fn property(paths: UniqueSortedKeyPaths) -> bool {
            if !is_sorted_unique(paths.as_slice()) {
                return false;
            }

            paths
                .shrink()
                .take(8)
                .all(|shrunk| is_sorted_unique(shrunk.as_slice()))
        }

        QuickCheck::new()
            .tests(64)
            .quickcheck(property as fn(UniqueSortedKeyPaths) -> bool);
    }
}
