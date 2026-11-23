mod common;

use common::Test;
use nomt_core::trie::KeyPath;
use std::collections::BTreeMap;

/// Keys sharing their first 16 bits, so a few hundred of them give pages two levels below the root
/// with 20 to 255 leaves: elided, but kept in the page cache's elided-page map.
fn clustered_key(rng: &mut u64) -> KeyPath {
    let mut key = [0u8; 32];
    key[0] = 0x5a;
    key[1] = 0xc3;
    for byte in key[2..].iter_mut() {
        *rng ^= *rng << 13;
        *rng ^= *rng >> 7;
        *rng ^= *rng << 17;
        *byte = *rng as u8;
    }
    key
}

/// The root of a fresh database holding `state`, written in one commit.
fn reference_root(name: &str, state: &BTreeMap<KeyPath, Vec<u8>>) -> nomt::Root {
    let mut t = Test::new_with_params(name, 2, 64_000, None, true);
    for (key, value) in state {
        t.write(*key, Some(value.clone()));
    }
    t.commit();
    t.root()
}

/// Many commits in one process update elided pages that are cached in memory between commits.
/// Each must match the committed state, or seeking through them yields a wrong root.
#[test]
fn elided_page_cache_matches_committed_state() {
    let mut rng = 0x9e37_79b9_7f4a_7c15u64;
    let mut keys = Vec::new();
    let mut state = BTreeMap::new();
    let mut t = Test::new_with_params("elided_cache", 2, 64_000, None, true);

    for round in 0..40u32 {
        let mut written = BTreeMap::new();
        for _ in 0..15 {
            keys.push(clustered_key(&mut rng));
        }
        for i in 0..60 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let key = if i < 15 {
                keys[keys.len() - 15 + i]
            } else {
                keys[(rng % keys.len() as u64) as usize]
            };
            written.insert(key, round.to_le_bytes().repeat(1 + i % 5));
        }
        for (key, value) in &written {
            t.write(*key, Some(value.clone()));
        }
        state.extend(written);
        t.commit();

        if round % 10 == 9 {
            assert_eq!(
                t.root(),
                reference_root("elided_cache_reference", &state),
                "root after round {round}"
            );
        }
    }
}
