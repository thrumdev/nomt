use anyhow::Result;
use nomt_core::{
    proof,
    trie::{self, LeafData, Node, NodeHasher, NodeKind, TERMINATOR},
};

// Hash nodes using blake3. The Hasher below will be utilized in the witness
// verification stage and must conform to the hash used in the commitment
// phase and subsequently during witness creation
pub struct Blake3Hasher;

impl NodeHasher for Blake3Hasher {
    fn hash_leaf(data: &trie::LeafData) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&data.key_path);
        hasher.update(&data.value_hash);
        let mut hash: [u8; 32] = hasher.finalize().into();

        // Label with MSB
        hash[0] |= 0b10000000;
        hash
    }

    fn hash_internal(data: &trie::InternalData) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&data.left);
        hasher.update(&data.right);
        let mut hash: [u8; 32] = hasher.finalize().into();

        // Label with MSB
        hash[0] &= 0b01111111;
        hash
    }

    fn node_kind(node: &Node) -> NodeKind {
        if node[0] >> 7 == 1 {
            NodeKind::Leaf
        } else if node == &TERMINATOR {
            NodeKind::Terminator
        } else {
            NodeKind::Internal
        }
    }
}

fn main() -> Result<()> {
    // The witness produced in the example `commit_batch` will be used
    let (prev_root, new_root, witness) = commit_batch::NomtDB::commit_batch().unwrap();

    let mut updates = Vec::new();

    // A witness is composed of multiple WitnessedPath objects,
    // which stores all the necessary information to verify the operations
    // performed on the same path
    for (i, witnessed_path) in witness.path_proofs.iter().enumerate() {
        // Constructing the verified operations
        let verified = witnessed_path
            .inner
            .verify::<Blake3Hasher>(&witnessed_path.path.path(), prev_root.into_inner())
            .unwrap();

        // Among all read operations performed the ones that interact
        // with the current verified path are selected
        //
        // Each witnessed operation contains an index to the path it needs to be verified against
        //
        // This information could already be known if we committed the batch initially,
        // and thus, the witnessed field could be discarded entirely.
        for read in witness
            .operations
            .reads
            .iter()
            .skip_while(|r| r.path_index != i)
            .take_while(|r| r.path_index == i)
        {
            match read.value {
                // Check for non-existence if the return value was None
                None => assert!(verified.confirm_nonexistence(&read.key).unwrap()),
                // Verify the correctness of the returned value when it is Some(_)
                Some(ref v) => {
                    let leaf = LeafData {
                        key_path: read.key,
                        value_hash: *blake3::hash(v).as_bytes(),
                    };
                    assert!(verified.confirm_value(&leaf).unwrap());
                }
            }
        }

        // The correctness of write operations cannot be easily verified like reads.
        // Write operations need to be collected.
        // All writes that have worked on shared prefixes,
        // such as the witnessed_path, need to be bundled together.
        // Later, it needs to be verified that all these writes bring
        // the new trie to the expected state
        let mut write_ops = Vec::new();
        for write in witness
            .operations
            .writes
            .iter()
            .skip_while(|r| r.path_index != i)
            .take_while(|r| r.path_index == i)
        {
            write_ops.push((
                write.key,
                write.value.as_ref().map(|v| *blake3::hash(v).as_bytes()),
            ));
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

    Ok(())
}
