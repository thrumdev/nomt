mod common;

use bitvec::prelude::*;
use common::{fresh_test_name, ProofCase, Test};
use nomt::hasher::Blake3Hasher;
use nomt_core::hasher::ValueHasher;
use nomt_core::trie::{KeyPath, LeafData, Node};
use quickcheck::QuickCheck;
use std::collections::{BTreeMap, BTreeSet};

#[test]
fn prove_in_session() {
    let mut accounts = 0;
    let mut t = Test::new("prove_in_session");

    let _ = t.read_id(0);
    for _ in 0..100 {
        common::set_balance(&mut t, accounts, 1000);
        accounts += 1;
    }
    let root = t.commit().0.into_inner();

    for i in 0..100 {
        let k = common::account_path(i);

        let proof = t.prove_id(i);
        let expected_leaf = LeafData {
            key_path: k,
            value_hash: proof.terminal.as_leaf_option().unwrap().value_hash,
        };
        assert!(proof
            .verify::<nomt::hasher::Blake3Hasher>(k.view_bits::<Msb0>(), root)
            .expect("verification failed")
            .confirm_value(&expected_leaf)
            .unwrap());
    }

    for i in 100..150 {
        let k = common::account_path(i);
        let proof = t.prove_id(i);
        assert!(proof
            .verify::<nomt::hasher::Blake3Hasher>(k.view_bits::<Msb0>(), root)
            .expect("verification failed")
            .confirm_nonexistence(&k)
            .unwrap());
    }
}

#[test]
fn prove_in_session_against_overlay() {
    let mut accounts = 0;
    let mut t = Test::new("prove_in_session_against_overlay");

    let _ = t.read_id(0);
    for _ in 0..100 {
        common::set_balance(&mut t, accounts, 1000);
        accounts += 1;
    }
    let (overlay_a, _) = t.update();
    let root = overlay_a.root().into_inner();
    t.start_overlay_session(&[overlay_a]);

    for i in 0..100 {
        let k = common::account_path(i);

        let proof = t.prove_id(i);
        let expected_leaf = LeafData {
            key_path: k,
            value_hash: proof.terminal.as_leaf_option().unwrap().value_hash,
        };
        assert!(proof
            .verify::<nomt::hasher::Blake3Hasher>(k.view_bits::<Msb0>(), root)
            .expect("verification failed")
            .confirm_value(&expected_leaf)
            .unwrap());
    }

    for i in 100..150 {
        let k = common::account_path(i);
        let proof = t.prove_id(i);
        assert!(proof
            .verify::<nomt::hasher::Blake3Hasher>(k.view_bits::<Msb0>(), root)
            .expect("verification failed")
            .confirm_nonexistence(&k)
            .unwrap());
    }
}

#[test]
fn prove_in_session_no_cache() {
    let mut accounts = 0;

    {
        let mut t = Test::new("prove_in_session_no_cache");

        let _ = t.read_id(0);

        // Write 5000 accounts to ensure a few I/Os will be needed to read the proofs.
        for _ in 0..5000 {
            common::set_balance(&mut t, accounts, 1000);
            accounts += 1;
        }
        t.commit().0.into_inner();
    }

    // Reopen the DB to clear the cache.
    let t = Test::new_with_params(
        "prove_in_session_no_cache",
        1,      // commit concurrency
        10_000, // hashtable buckets
        None,   // panic on sync
        false,  // cleanup dir
    );

    let root = t.root().into_inner();

    for i in 0..100 {
        let k = common::account_path(i);

        let proof = t.prove_id(i);
        let expected_leaf = LeafData {
            key_path: k,
            value_hash: proof.terminal.as_leaf_option().unwrap().value_hash,
        };
        assert!(proof
            .verify::<nomt::hasher::Blake3Hasher>(k.view_bits::<Msb0>(), root)
            .expect("verification failed")
            .confirm_value(&expected_leaf)
            .unwrap());
    }

    for i in 10000..10050 {
        let k = common::account_path(i);
        let proof = t.prove_id(i);
        assert!(proof
            .verify::<nomt::hasher::Blake3Hasher>(k.view_bits::<Msb0>(), root)
            .expect("verification failed")
            .confirm_nonexistence(&k)
            .unwrap());
    }
}

#[test]
fn property_generated_proofs_verify() {
    fn property(case: ProofCase) -> bool {
        let name = fresh_test_name("prove_prop");
        let sample_keys = case
            .present_samples
            .iter()
            .chain(case.missing_samples.iter())
            .copied()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();

        let base_state = case
            .state
            .iter()
            .map(|(key, value)| (*key, value.clone()))
            .collect::<BTreeMap<_, _>>();

        let base_root = {
            let mut t = Test::new(&name);
            for (key, value) in &case.state {
                t.write(*key, Some(value.clone()));
            }
            let root = t.commit().0.into_inner();

            assert_proofs_match(&t, root, &base_state, &sample_keys);

            let mut overlay_state = base_state.clone();
            if let Some(update_key) = overlay_state.keys().next().copied() {
                let updated_value = vec![0xA5, overlay_state.len() as u8];
                t.write(update_key, Some(updated_value.clone()));
                overlay_state.insert(update_key, updated_value);
            } else {
                let inserted_key = case.missing_samples[0];
                let inserted_value = vec![0x5A];
                t.write(inserted_key, Some(inserted_value.clone()));
                overlay_state.insert(inserted_key, inserted_value);
            }

            if overlay_state.len() > 1 {
                let delete_key = overlay_state.keys().last().copied().unwrap();
                t.write(delete_key, None);
                overlay_state.remove(&delete_key);
            }

            let (overlay, _) = t.update();
            let overlay_root = overlay.root().into_inner();
            t.start_overlay_session([&overlay]);
            assert_proofs_match(&t, overlay_root, &overlay_state, &sample_keys);

            root
        };

        let reopened = Test::new_with_params(&name, 1, 10_000, None, false);
        assert_proofs_match(&reopened, base_root, &base_state, &sample_keys);
        true
    }

    QuickCheck::new()
        .tests(12)
        .quickcheck(property as fn(ProofCase) -> bool);
}

fn assert_proofs_match(
    test: &Test,
    root: Node,
    expected: &BTreeMap<KeyPath, Vec<u8>>,
    sample_keys: &[KeyPath],
) {
    for key in sample_keys {
        let proof = test.prove(*key);
        let verified = proof
            .verify::<nomt::hasher::Blake3Hasher>(key.view_bits::<Msb0>(), root)
            .expect("verification failed");

        match expected.get(key) {
            Some(value) => {
                let expected_leaf = LeafData {
                    key_path: *key,
                    value_hash: Blake3Hasher::hash_value(value),
                };
                assert_eq!(
                    proof.terminal.as_leaf_option().unwrap().value_hash,
                    expected_leaf.value_hash,
                );
                assert!(verified.confirm_value(&expected_leaf).unwrap());
                assert!(proof.terminal.as_leaf_option().is_some());
            }
            None => assert!(verified.confirm_nonexistence(key).unwrap()),
        }
    }
}
