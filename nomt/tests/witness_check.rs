mod common;

use common::Test;
use nomt::{hasher::Blake3Hasher, proof, trie::LeafData};

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

    assert_eq!(
        proof::verify_update::<Blake3Hasher>(prev_root.into_inner(), &updates).unwrap(),
        new_root.into_inner(),
    );
}
