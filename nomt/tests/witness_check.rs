mod common;

use common::{apply_accesses, fresh_test_name, SessionAccessCase, Test};
use nomt::{hasher::Blake3Hasher, proof, trie::LeafData};
use quickcheck::QuickCheck;

fn sort_updates_by_path(updates: &mut [proof::PathUpdate]) {
    // Witness path proofs may be emitted in worker order rather than trie path order.
    updates.sort_by(|a, b| {
        a.inner
            .path()
            .partial_cmp(b.inner.path())
            .expect("verified witness paths should be comparable")
    });
}

#[test]
fn produced_witness_validity() {
    let mut accounts = 0;
    let mut t = Test::new("witness_validity");

    let (prev_root, _) = {
        for _ in 0..10 {
            common::set_balance(&mut t, accounts, 1000);
            accounts += 1;
        }
        t.commit()
    };

    let (new_root, witness) = {
        // read all existing accounts.
        for i in 0..accounts {
            t.read_id(i);
        }

        // read some nonexistent accounts.
        for i in 100..105 {
            t.read_id(i);
        }

        // kill half the existing ones.
        for i in 0..5 {
            common::kill(&mut t, i);
        }

        // and add 5 more.
        for _ in 0..5 {
            common::set_balance(&mut t, accounts, 1000);
            accounts += 1;
        }
        t.commit()
    };

    assert_eq!(witness.operations.reads.len(), 15); // 10 existing + 5 nonexisting
    assert_eq!(witness.operations.writes.len(), 10); // 5 deletes + 5 inserts

    let mut updates = Vec::new();
    for (i, witnessed_path) in witness.path_proofs.iter().enumerate() {
        let verified = witnessed_path
            .inner
            .verify::<Blake3Hasher>(&witnessed_path.path.path(), prev_root.into_inner())
            .unwrap();
        for read in witness
            .operations
            .reads
            .iter()
            .skip_while(|r| r.path_index != i)
            .take_while(|r| r.path_index == i)
        {
            match read.value {
                None => assert!(verified.confirm_nonexistence(&read.key).unwrap()),
                Some(ref v) => {
                    let leaf = LeafData {
                        key_path: read.key,
                        value_hash: *v,
                    };
                    assert!(verified.confirm_value(&leaf).unwrap());
                }
            }
        }

        let mut write_ops = Vec::new();
        for write in witness
            .operations
            .writes
            .iter()
            .skip_while(|r| r.path_index != i)
            .take_while(|r| r.path_index == i)
        {
            write_ops.push((write.key, write.value.clone()));
        }

        if !write_ops.is_empty() {
            updates.push(proof::PathUpdate {
                inner: verified,
                ops: write_ops,
            });
        }
    }
    sort_updates_by_path(&mut updates);

    assert_eq!(
        proof::verify_update::<Blake3Hasher>(prev_root.into_inner(), &updates).unwrap(),
        new_root.into_inner(),
    );
}

#[test]
fn empty_witness() {
    let mut accounts = 0;
    let mut t = Test::new("empty_witness");

    let (prev_root, _) = {
        for _ in 0..10 {
            common::set_balance(&mut t, accounts, 1000);
            accounts += 1;
        }
        t.commit()
    };

    // Create a commit with no operations performed
    let (new_root, witness) = t.commit();

    // The roots should be identical since no changes were made
    assert_eq!(prev_root, new_root);

    // The witness should be empty
    assert_eq!(witness.operations.reads.len(), 0);
    assert_eq!(witness.operations.writes.len(), 0);
    assert_eq!(witness.path_proofs.len(), 0);

    // Verify that an empty update produces the same root
    let updates: Vec<proof::PathUpdate> = Vec::new();
    assert_eq!(
        proof::verify_update::<Blake3Hasher>(prev_root.into_inner(), &updates).unwrap(),
        new_root.into_inner(),
    );
}

#[test]
fn test_verify_update_with_identical_paths() {
    use nomt::{
        hasher::Blake3Hasher,
        proof::{verify_update, PathUpdate},
        trie::ValueHash,
    };

    let account0 = 0;

    // Create a simple trie, create an update witness.
    let mut t = Test::new("identical_paths_test");
    common::set_balance(&mut t, account0, 1000);
    let (root, _) = t.commit();
    t.read_id(account0);
    let (_, witness) = t.commit();

    // Using that witness extract and verify the proof.
    let witnessed_path = &witness.path_proofs[0];
    let verified_proof = witnessed_path
        .inner
        .verify::<Blake3Hasher>(&witnessed_path.path.path(), root.into_inner())
        .unwrap();

    // Create two identical PathUpdate objects
    let mut updates = Vec::new();

    // First update
    let value1 = ValueHash::default();
    let ops1 = vec![([0; 32], Some(value1))];
    updates.push(PathUpdate {
        inner: verified_proof.clone(),
        ops: ops1,
    });

    // Second update with identical path
    let value2 = ValueHash::default();
    let ops2 = vec![([1; 32], Some(value2))];
    updates.push(PathUpdate {
        inner: verified_proof, // Using the same verified proof
        ops: ops2,
    });

    // Try to verify the update. We expect an error due to identical paths, because that violates
    // the requirement of ascending keys.
    verify_update::<Blake3Hasher>(root.into_inner(), &updates).unwrap_err();
}

#[test]
fn property_generated_witnesses_verify() {
    fn property(case: SessionAccessCase) -> bool {
        let mut t = Test::new(fresh_test_name("witness_prop"));

        for (key, value) in &case.prev_data {
            t.write(*key, Some(value.clone()));
        }
        let (prev_root, _) = t.commit();

        apply_accesses(&mut t, &case.accesses);

        let (new_root, witness) = t.commit();

        let mut updates = Vec::new();
        for (i, witnessed_path) in witness.path_proofs.iter().enumerate() {
            let verified = witnessed_path
                .inner
                .verify::<Blake3Hasher>(&witnessed_path.path.path(), prev_root.into_inner())
                .unwrap();

            for read in witness
                .operations
                .reads
                .iter()
                .skip_while(|r| r.path_index != i)
                .take_while(|r| r.path_index == i)
            {
                match read.value {
                    None => assert!(verified.confirm_nonexistence(&read.key).unwrap()),
                    Some(ref value_hash) => {
                        let leaf = LeafData {
                            key_path: read.key,
                            value_hash: *value_hash,
                        };
                        assert!(verified.confirm_value(&leaf).unwrap());
                    }
                }
            }

            let mut write_ops = Vec::new();
            for write in witness
                .operations
                .writes
                .iter()
                .skip_while(|w| w.path_index != i)
                .take_while(|w| w.path_index == i)
            {
                write_ops.push((write.key, write.value.clone()));
            }

            if !write_ops.is_empty() {
                updates.push(proof::PathUpdate {
                    inner: verified,
                    ops: write_ops,
                });
            }
        }
        sort_updates_by_path(&mut updates);

        assert_eq!(
            proof::verify_update::<Blake3Hasher>(prev_root.into_inner(), &updates).unwrap(),
            new_root.into_inner(),
        );
        true
    }

    QuickCheck::new()
        .tests(16)
        .quickcheck(property as fn(SessionAccessCase) -> bool);
}

#[test]
fn witness_validity_with_concurrent_workers() {
    // Commit workers may finish in any order; every witnessed operation must still refer to the
    // path proof that covers its key.
    let mut t = Test::new_with_params("witness_concurrent_workers", 4, 64_000, None, true);
    for i in 0..1000 {
        common::set_balance(&mut t, i, 1000);
    }
    let (mut prev_root, _) = t.commit();

    for round in 0..20 {
        for i in 0..500 {
            t.read_id(i);
        }
        for i in 500..1000 {
            common::set_balance(&mut t, i, round);
        }
        let (new_root, witness) = t.commit();

        let verified = witness
            .path_proofs
            .iter()
            .map(|p| {
                p.inner
                    .verify::<Blake3Hasher>(&p.path.path(), prev_root.into_inner())
                    .unwrap()
            })
            .collect::<Vec<_>>();

        for read in &witness.operations.reads {
            let leaf = LeafData {
                key_path: read.key,
                value_hash: read.value.unwrap(),
            };
            assert!(verified[read.path_index].confirm_value(&leaf).unwrap());
        }

        let mut ops_by_path = std::collections::BTreeMap::<usize, Vec<_>>::new();
        for write in &witness.operations.writes {
            ops_by_path
                .entry(write.path_index)
                .or_default()
                .push((write.key, write.value));
        }
        let mut updates = ops_by_path
            .into_iter()
            .map(|(i, ops)| proof::PathUpdate {
                inner: verified[i].clone(),
                ops,
            })
            .collect::<Vec<_>>();
        sort_updates_by_path(&mut updates);
        assert_eq!(
            proof::verify_update::<Blake3Hasher>(prev_root.into_inner(), &updates).unwrap(),
            new_root.into_inner(),
        );

        prev_root = new_root;
    }
}
