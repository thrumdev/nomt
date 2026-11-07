mod common;

use common::{
    account_path, apply_accesses, fresh_test_name, key_diverging_at, SessionAccessCase, Test,
};
use nomt::{hasher::Blake3Hasher, trie::NodeKind, KeyReadWrite, Value};
use nomt_core::hasher::ValueHasher;
use nomt_core::proof::{verify_multi_proof, verify_multi_proof_update, MultiProof};
use nomt_core::trie::{KeyPath, Node, ValueHash};
use quickcheck::QuickCheck;
use std::path::Path;

#[test]
fn root_on_empty_db() {
    let t = Test::new("compute_root_empty");
    let root = t.root();
    assert_eq!(
        NodeKind::of::<Blake3Hasher>(&root.into_inner()),
        NodeKind::Terminator
    );
}

#[test]
fn root_on_leaf() {
    {
        let mut t = Test::new("compute_root_leaf");
        t.write([1; 32], Some(vec![1, 2, 3]));
        t.commit();
    }

    let t = Test::new_with_params("compute_root_leaf", 1, 1, None, false);
    let root = t.root();
    assert_eq!(
        NodeKind::of::<Blake3Hasher>(&root.into_inner()),
        NodeKind::Leaf
    );
}

#[test]
fn root_on_internal() {
    {
        let mut t = Test::new("compute_root_internal");
        t.write([0; 32], Some(vec![1, 2, 3]));
        t.write([1; 32], Some(vec![1, 2, 3]));
        t.commit();
    }

    let t = Test::new_with_params("compute_root_internal", 1, 1, None, false);
    let root = t.root();
    assert_eq!(
        NodeKind::of::<Blake3Hasher>(&root.into_inner()),
        NodeKind::Internal
    );
}

#[test]
fn simple_roots_match() {
    let accesses = vec![
        ([0; 32], KeyReadWrite::Write(Some(vec![1, 2, 3]))),
        ([1; 32], KeyReadWrite::Write(Some(vec![1, 2, 3]))),
    ];

    test_root_match_with_inputs("simple_roots_match", Vec::new(), &accesses);
}

fn test_root_match_with_inputs(
    name: impl AsRef<Path>,
    prev_data: Vec<(KeyPath, Option<Value>)>,
    accesses: &[(KeyPath, KeyReadWrite)],
) {
    let mut t = Test::new(name);
    {
        for (key_path, write) in prev_data {
            t.write(key_path, write);
        }
        t.commit();
    }
    let prev_root = t.root();
    for (key_path, operation) in accesses {
        match operation {
            KeyReadWrite::Read(y) => {
                let x = t.read(*key_path);
                assert_eq!(&x, y);
            }
            KeyReadWrite::Write(v) => {
                t.write(*key_path, v.clone());
            }
            KeyReadWrite::ReadThenWrite(r, v) => {
                let x = t.read(*key_path);
                assert_eq!(&x, r);
                t.write(*key_path, v.clone());
            }
        }
    }

    let (prover_root, witness) = t.commit();
    let nomt::Witness {
        path_proofs,
        operations: nomt::WitnessedOperations { .. },
    } = witness;
    let mut inner = path_proofs.into_iter().map(|p| p.inner).collect::<Vec<_>>();
    inner.sort_by(|a, b| a.terminal.path().cmp(b.terminal.path()));
    let multi_proof = MultiProof::from_path_proofs(inner);

    let verifier_root = run_verifier(multi_proof, accesses, prev_root.into_inner());

    assert_eq!(
        prover_root.into_inner(),
        verifier_root,
        "Verifier and Prover have mismatched next state roots"
    );
}

fn run_verifier(
    multi_proof: MultiProof,
    accesses: &[(KeyPath, KeyReadWrite)],
    prev_root: Node,
) -> Node {
    let verified = verify_multi_proof::<Blake3Hasher>(&multi_proof, prev_root)
        .expect("multiproof must verify against the prior root");

    let mut updates: Vec<(KeyPath, Option<ValueHash>)> = accesses
        .iter()
        .filter_map(|(k, op)| match op {
            KeyReadWrite::Write(v) | KeyReadWrite::ReadThenWrite(_, v) => {
                Some((*k, v.as_deref().map(Blake3Hasher::hash_value)))
            }
            KeyReadWrite::Read(_) => None,
        })
        .collect();
    updates.sort_by(|a, b| a.0.cmp(&b.0));

    verify_multi_proof_update::<Blake3Hasher>(&verified, updates)
        .expect("verify_multi_proof_update must succeed")
}

fn run_depth_sweep_case(name: &str, depth: usize) {
    let k0 = key_diverging_at(depth, false);
    let k1 = key_diverging_at(depth, true);
    let (k_lo, k_hi) = if k0 < k1 { (k0, k1) } else { (k1, k0) };
    let accesses = vec![
        (k_lo, KeyReadWrite::Write(Some(vec![0xAA]))),
        (k_hi, KeyReadWrite::Write(Some(vec![0xBB]))),
    ];
    test_root_match_with_inputs(name, Vec::new(), &accesses);
}

#[test]
fn key_diverging_at_differs_by_one_bit() {
    for depth in [0usize, 1, 7, 8, 15, 16, 127, 255] {
        let k0 = key_diverging_at(depth, false);
        let k1 = key_diverging_at(depth, true);
        let xor: Vec<u8> = k0.iter().zip(k1.iter()).map(|(a, b)| a ^ b).collect();
        let set_bits: u32 = xor.iter().map(|b| b.count_ones()).sum();
        assert_eq!(set_bits, 1, "depth {depth}: keys should differ in one bit");
        let diverging_byte = depth / 8;
        let diverging_mask = 1u8 << (7 - (depth % 8));
        assert_eq!(
            xor[diverging_byte], diverging_mask,
            "depth {depth}: divergence should be at MSB-position"
        );
    }
}

macro_rules! depth_sweep_test {
    ($name:ident, $depth:expr) => {
        #[test]
        fn $name() {
            run_depth_sweep_case(stringify!($name), $depth);
        }
    };
}

depth_sweep_test!(depth_sweep_d0, 0);
depth_sweep_test!(depth_sweep_d1, 1);
depth_sweep_test!(depth_sweep_d7, 7);
depth_sweep_test!(depth_sweep_d8, 8);
depth_sweep_test!(depth_sweep_d15, 15);
depth_sweep_test!(depth_sweep_d16, 16);
depth_sweep_test!(depth_sweep_d127, 127);
depth_sweep_test!(depth_sweep_d255, 255);

#[test]
fn overwrite_and_empty_value() {
    let k_a = key_diverging_at(16, false);
    let k_b = key_diverging_at(16, true);
    let prev = vec![(k_a, Some(vec![1, 2, 3])), (k_b, Some(vec![9, 9, 9]))];
    let accesses = vec![
        (k_a, KeyReadWrite::Write(Some(vec![1, 2, 3]))),
        (k_b, KeyReadWrite::Write(Some(vec![]))),
    ];
    test_root_match_with_inputs("overwrite_and_empty_value", prev, &accesses);
}

#[test]
fn delete_collapses_internal_to_leaf() {
    let k_a = key_diverging_at(16, false);
    let k_b = key_diverging_at(16, true);
    let prev = vec![(k_a, Some(vec![0xAA])), (k_b, Some(vec![0xBB]))];
    let accesses = vec![(k_b, KeyReadWrite::Write(None))];
    test_root_match_with_inputs("delete_collapses_internal_to_leaf", prev, &accesses);
}

#[test]
fn delete_last_key_to_terminator() {
    let k = key_diverging_at(32, true);
    let prev = vec![(k, Some(vec![0xAA]))];
    let accesses = vec![(k, KeyReadWrite::Write(None))];
    test_root_match_with_inputs("delete_last_key_to_terminator", prev, &accesses);
}

#[test]
fn delete_nonexistent_key() {
    let k_a = key_diverging_at(8, false);
    let k_b = key_diverging_at(8, true);
    let prev = vec![(k_a, Some(vec![0xAA]))];
    let accesses = vec![(k_b, KeyReadWrite::Write(None))];
    test_root_match_with_inputs("delete_nonexistent_key", prev, &accesses);
}

#[test]
fn read_then_write_mixed() {
    let k0 = key_diverging_at(0, true);
    let k1 = key_diverging_at(8, true);
    let k2 = key_diverging_at(64, true);
    let k_missing = key_diverging_at(128, true);
    let v0 = vec![0xA0];
    let v1 = vec![0xA1];
    let v2 = vec![0xA2];
    let prev = vec![
        (k0, Some(v0.clone())),
        (k1, Some(v1.clone())),
        (k2, Some(v2.clone())),
    ];
    let accesses = vec![
        (k0, KeyReadWrite::Read(Some(v0))),
        (k1, KeyReadWrite::ReadThenWrite(Some(v1), Some(vec![0xB1]))),
        (k_missing, KeyReadWrite::Read(None)),
    ];
    test_root_match_with_inputs("read_then_write_mixed", prev, &accesses);
}

#[test]
fn cluster_asymmetric_tree() {
    let mk = |first_byte: u8| {
        let mut k = KeyPath::default();
        k[0] = first_byte;
        k
    };
    let accesses = vec![
        (mk(0x10), KeyReadWrite::Write(Some(vec![1]))),
        (mk(0x20), KeyReadWrite::Write(Some(vec![2]))),
        (mk(0x40), KeyReadWrite::Write(Some(vec![3]))),
        (mk(0xA0), KeyReadWrite::Write(Some(vec![4]))),
    ];
    test_root_match_with_inputs("cluster_asymmetric_tree", Vec::new(), &accesses);
}

#[test]
fn insert_empty_value_new_key() {
    let k_other = key_diverging_at(8, false);
    let k_new = key_diverging_at(8, true);
    let prev = vec![(k_other, Some(vec![0xCC]))];
    let accesses = vec![(k_new, KeyReadWrite::Write(Some(vec![])))];
    test_root_match_with_inputs("insert_empty_value_new_key", prev, &accesses);
}

#[test]
fn read_then_delete() {
    let k_a = key_diverging_at(16, false);
    let k_b = key_diverging_at(16, true);
    let v_b = vec![0xBB];
    let prev = vec![(k_a, Some(vec![0xAA])), (k_b, Some(v_b.clone()))];
    let accesses = vec![(k_b, KeyReadWrite::ReadThenWrite(Some(v_b), None))];
    test_root_match_with_inputs("read_then_delete", prev, &accesses);
}

#[test]
fn delete_all_keys_to_terminator() {
    let k0 = key_diverging_at(0, true);
    let k1 = key_diverging_at(8, true);
    let k2 = key_diverging_at(64, true);
    let prev = vec![
        (k0, Some(vec![0xA0])),
        (k1, Some(vec![0xA1])),
        (k2, Some(vec![0xA2])),
    ];
    let accesses = vec![
        (k0, KeyReadWrite::Write(None)),
        (k1, KeyReadWrite::Write(None)),
        (k2, KeyReadWrite::Write(None)),
    ];
    test_root_match_with_inputs("delete_all_keys_to_terminator", prev, &accesses);
}

#[test]
fn insert_splits_existing_leaf() {
    let k_left = key_diverging_at(0, false);
    let k_right = key_diverging_at(0, true);
    let k_new = key_diverging_at(7, true);
    let prev = vec![(k_left, Some(vec![0x11])), (k_right, Some(vec![0x22]))];
    let accesses = vec![(k_new, KeyReadWrite::Write(Some(vec![0xEE])))];
    test_root_match_with_inputs("insert_splits_existing_leaf", prev, &accesses);
}

#[test]
fn read_then_write_missing_key() {
    let k_other = key_diverging_at(8, false);
    let k_missing = key_diverging_at(8, true);
    let prev = vec![(k_other, Some(vec![0xCC]))];
    let accesses = vec![(
        k_missing,
        KeyReadWrite::ReadThenWrite(None, Some(vec![0xEE])),
    )];
    test_root_match_with_inputs("read_then_write_missing_key", prev, &accesses);
}

#[test]
fn many_keys_round_trip() {
    let prev: Vec<_> = (0..32)
        .map(|i| (account_path(i), Some(vec![i as u8, 0x01])))
        .collect();
    let mut accesses: Vec<(KeyPath, KeyReadWrite)> = Vec::new();
    for i in 32..48 {
        accesses.push((
            account_path(i),
            KeyReadWrite::Write(Some(vec![i as u8, 0x02])),
        ));
    }
    for i in 0..8 {
        accesses.push((
            account_path(i),
            KeyReadWrite::Write(Some(vec![i as u8, 0x03])),
        ));
    }
    for i in 8..16 {
        accesses.push((account_path(i), KeyReadWrite::Write(None)));
    }
    test_root_match_with_inputs("many_keys_round_trip", prev, &accesses);
}

#[test]
fn many_keys_inserts_only() {
    let prev: Vec<_> = (0..32)
        .map(|i| (account_path(i), Some(vec![i as u8, 0x01])))
        .collect();
    let accesses: Vec<(KeyPath, KeyReadWrite)> = (32..48)
        .map(|i| {
            (
                account_path(i),
                KeyReadWrite::Write(Some(vec![i as u8, 0x02])),
            )
        })
        .collect();
    test_root_match_with_inputs("many_keys_inserts_only", prev, &accesses);
}

#[test]
fn many_keys_overwrites_only() {
    let prev: Vec<_> = (0..32)
        .map(|i| (account_path(i), Some(vec![i as u8, 0x01])))
        .collect();
    let accesses: Vec<(KeyPath, KeyReadWrite)> = (0..8)
        .map(|i| {
            (
                account_path(i),
                KeyReadWrite::Write(Some(vec![i as u8, 0x03])),
            )
        })
        .collect();
    test_root_match_with_inputs("many_keys_overwrites_only", prev, &accesses);
}

#[test]
fn many_keys_deletes_only() {
    let prev: Vec<_> = (0..32)
        .map(|i| (account_path(i), Some(vec![i as u8, 0x01])))
        .collect();
    let accesses: Vec<(KeyPath, KeyReadWrite)> = (8..16)
        .map(|i| (account_path(i), KeyReadWrite::Write(None)))
        .collect();
    test_root_match_with_inputs("many_keys_deletes_only", prev, &accesses);
}

#[test]
fn property_generated_roots_match() {
    fn property(case: SessionAccessCase) -> bool {
        let prev_data = case.prev_data_with_options();
        test_root_match_with_inputs(
            fresh_test_name("compute_root_prop"),
            prev_data,
            &case.accesses,
        );
        true
    }

    QuickCheck::new()
        .tests(24)
        .quickcheck(property as fn(SessionAccessCase) -> bool);
}

#[test]
fn property_post_commit_state_matches_reference() {
    fn property(case: SessionAccessCase) -> bool {
        let mut t = Test::new(fresh_test_name("state_oracle_prop"));

        for (key, value) in &case.prev_data {
            t.write(*key, Some(value.clone()));
        }
        t.commit();

        apply_accesses(&mut t, &case.accesses);
        let (root, _) = t.commit();

        let expected_state = case.expected_final_state();

        for (key, expected_value) in &expected_state {
            assert_eq!(
                t.read(*key).as_ref(),
                Some(expected_value),
                "post-commit read mismatch for present key"
            );
        }

        for (key, _) in &case.accesses {
            if !expected_state.contains_key(key) {
                assert_eq!(
                    t.read(*key),
                    None,
                    "expected key to be absent after committed deletes"
                );
            }
        }

        let mut ops = expected_state
            .iter()
            .map(|(k, v)| (*k, Blake3Hasher::hash_value(v)))
            .collect::<Vec<_>>();
        ops.sort_by_key(|(k, _)| *k);
        let reference_root = nomt_core::update::build_trie::<Blake3Hasher>(0, ops, |_| {});

        assert_eq!(
            root.into_inner(),
            reference_root,
            "Nomt root disagrees with build_trie oracle"
        );

        true
    }

    QuickCheck::new()
        .tests(16)
        .quickcheck(property as fn(SessionAccessCase) -> bool);
}

#[test]
fn property_root_invariant_under_reopen() {
    fn property(case: SessionAccessCase) -> bool {
        let name = fresh_test_name("reopen_root_prop");

        let pre_close_root = {
            let mut t = Test::new(&name);
            for (key, value) in &case.prev_data {
                t.write(*key, Some(value.clone()));
            }
            t.commit();
            apply_accesses(&mut t, &case.accesses);
            t.commit().0
        };

        let reopened = Test::new_with_params(&name, 1, 64_000, None, false);
        assert_eq!(
            reopened.root().into_inner(),
            pre_close_root.into_inner(),
            "root drifted across close/reopen"
        );

        true
    }

    QuickCheck::new()
        .tests(16)
        .quickcheck(property as fn(SessionAccessCase) -> bool);
}
