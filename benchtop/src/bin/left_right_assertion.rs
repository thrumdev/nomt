use nomt::{Nomt, Options, SessionParams, WitnessMode, KeyReadWrite};
use nomt::hasher::Blake3Hasher;
use sha2::{Sha256, Digest};
use std::fs;
use nomt::trie::KeyPath;

const DB_PATH: &str = "./nomt_overlay_db";
const NUM_KEYS: usize = 1000;

fn to_key(data: &[u8]) -> KeyPath {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().into()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = fs::remove_dir_all(DB_PATH);

    let mut o = Options::new();
    o.path(DB_PATH);
    o.hashtable_buckets(64000);
    let db = Nomt::<Blake3Hasher>::open(o)?;

    let session1 = db.begin_session(SessionParams::default().witness_mode(WitnessMode::disabled()));
    let mut changes1 = Vec::with_capacity(NUM_KEYS);
    for i in 0..NUM_KEYS {
        let key_bytes = i.to_be_bytes();
        let key = to_key(&key_bytes);
        changes1.push((key, KeyReadWrite::Write(Some(vec![1]))));
    }
    changes1.sort_by_key(|(k, _)| *k);
    session1.finish(changes1)?.commit(&db)?;

    let session1 = db.begin_session(SessionParams::default().witness_mode(WitnessMode::disabled()));
    let mut changes1 = Vec::with_capacity(NUM_KEYS);
    for i in 0..NUM_KEYS {
        let key_bytes = i.to_be_bytes();
        let key = to_key(&key_bytes);
        let op = match i % 2 == 0 { // Use `round % 2 == 0 || i % 2 == 0` for another error
            true => KeyReadWrite::Write(Some(vec![1])),
            false => KeyReadWrite::Write(None),
        };
        changes1.push((key, op));
    }
    changes1.sort_by_key(|(k, _)| *k);
    let o1 = session1.finish(changes1)?.into_overlay();

    let session2 = db.begin_session(SessionParams::default().witness_mode(WitnessMode::disabled()).overlay(vec![&o1]).unwrap());
    let mut changes2 = Vec::with_capacity(NUM_KEYS);
    for i in 0..NUM_KEYS {
        let key_bytes = i.to_be_bytes();
        let key = to_key(&key_bytes);
        changes2.push((key, KeyReadWrite::Write(Some(vec![1]))));
    }
    changes2.sort_by_key(|(k, _)| *k);
    let _ = session2.finish(changes2)?.into_overlay();
    Ok(())
}
