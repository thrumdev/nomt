use crate::{backend::Action, workload::Workload};
use anyhow::Result;
use rand::{Rng, RngCore as _, SeedableRng as _};
use ruint::Uint;
use std::time::{SystemTime, UNIX_EPOCH};

// The custom workload will follow these rules:
// 1. Reads and writes are randomly and uniformly distributed across the key space,
//    or sequentially starting from a random address (depending on the `seq` value).
// 2. Deletes and updates will be performed on already present keys.
// 3. additional_initial_capacity represents the amount of items already present
//     in the DB in addition to the ones required to perform deletes and updates.
// 4. Size represents the total number of operations, where reads, writes, etc
//     are numbers that need to sum to 100 and represent a percentage of the total size.
pub fn new_custom_workload(
    reads: u8,
    writes: u8,
    deletes: u8,
    updates: u8,
    seq: bool,
    size: u64,
    additional_initial_capacity: u64,
) -> Result<Workload> {
    if reads + writes + deletes + updates != 100 {
        anyhow::bail!("Operations (reads, writes, deletes, updates) must sum to 100");
    }

    let from_percentage = |p: u8| (size as f64 * p as f64 / 100.0) as u64;
    let n_reads = from_percentage(reads);
    let n_writes = from_percentage(writes);
    let n_deletes = from_percentage(deletes);
    let n_updates = from_percentage(updates);

    let n_required_key = n_deletes + n_updates;
    let initial_writes = (0..n_required_key + additional_initial_capacity)
        .map(|i| Action::Write {
            key: i.to_be_bytes().to_vec(),
            value: Some(vec![64u8; 32]),
        })
        .collect();

    // new keys will be random or sequential from a random one
    let rand_or_seq =
        |prev: &mut Option<Uint<256, 4>>, rng: &mut rand_pcg::Lcg64Xsh32| -> Vec<u8> {
            if seq {
                let new_key = match prev {
                    Some(prev_key) => *prev_key + Uint::<256, 4>::from(1),
                    None => Uint::<256, 4>::from_be_bytes(rand_key(rng)),
                };
                *prev = Some(new_key);
                new_key.to_be_bytes::<32>().to_vec()
            } else {
                rand_key(rng).to_vec()
            }
        };

    // create two rng, one for read and one for write
    let read_seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)?
        .as_nanos()
        .to_le_bytes()[0..16]
        .try_into()?;
    let mut rng_read = rand_pcg::Lcg64Xsh32::from_seed(read_seed);
    let mut write_seed = [0; 16];
    rng_read.fill_bytes(&mut write_seed);
    let mut rng_write = rand_pcg::Lcg64Xsh32::from_seed(write_seed);

    let mut read_init_key = None;
    let read_actions = (0..n_reads).map(|_| Action::Read {
        key: rand_or_seq(&mut read_init_key, &mut rng_read),
    });

    let mut write_init_key = None;
    let write_actions = (n_reads..n_reads + n_writes).map(|_| Action::Write {
        key: rand_or_seq(&mut write_init_key, &mut rng_write),
        value: Some(vec![8; 16]),
    });

    let delete_actions = (0..n_deletes).map(|i| Action::Write {
        key: i.to_be_bytes().to_vec(),
        value: None,
    });
    let update_actions = (n_deletes..n_deletes + n_updates).map(|i| Action::Write {
        key: i.to_be_bytes().to_vec(),
        value: Some(vec![2; 16]),
    });

    let custom_actions = read_actions
        .chain(write_actions)
        .chain(delete_actions)
        .chain(update_actions)
        .collect();

    Ok(Workload {
        init_actions: initial_writes,
        run_actions: custom_actions,
    })
}

fn rand_key(rng: &mut impl Rng) -> [u8; 32] {
    // keys must be uniformly distributed
    let mut key = [0; 32];
    key[0..4].copy_from_slice(&rng.next_u32().to_le_bytes());
    key[4..8].copy_from_slice(&rng.next_u32().to_le_bytes());
    key[8..12].copy_from_slice(&rng.next_u32().to_le_bytes());
    key[12..16].copy_from_slice(&rng.next_u32().to_le_bytes());
    key
}
