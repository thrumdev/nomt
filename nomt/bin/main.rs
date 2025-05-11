use std::path::Path;
use rand::Rng;
use nomt::{KeyReadWrite, Nomt, Options, SessionParams, WitnessMode};
use nomt::hasher::Blake3Hasher;
use nomt::trie::KeyPath;

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "nomt_db".to_string());
    println!("Using path: {:?}", path);
    let path = Path::new(&path);
    if path.exists() {
        eprintln!("Path already exists, please remove it or use a different path");
        std::process::exit(1);
    }
    let db = init_nomt_db(&path);

    let mut rng = rand::thread_rng();

    let key_value_pairs: Vec<(KeyPath, KeyReadWrite)>=
        (0..10000).map(|_| {
            let mut key = [0_u8; 32];
            // put random in key
            rng.fill(&mut key);
            let value = vec![0; 48];
            (key, KeyReadWrite::Write(Some(value)))
        }).collect();
    write_to_nomt(&db, key_value_pairs);

    for _ in 0..100 {
        let key_value_pairs: Vec<(KeyPath, KeyReadWrite)>=
            (0..1000).map(|_| {
                let mut key = [0_u8; 32];
                // put random in key
                let random_u64 = rng.gen::<u64>();
                key[32-8..].copy_from_slice(&random_u64.to_le_bytes());
                let value = vec![0; 48];
                (key, KeyReadWrite::Write(Some(value)))
            }).collect();
        write_to_nomt(&db, key_value_pairs);
    }
}

fn write_to_nomt(db: &Nomt<Blake3Hasher>, mut key_value_pairs: Vec<(KeyPath, KeyReadWrite)>) {
    key_value_pairs.sort_by_key(|(key, _)| *key);
    let session = db.begin_session(SessionParams::default().witness_mode(WitnessMode::disabled()));
    println!("Writing {} items", key_value_pairs.len());
    let a = session.finish(key_value_pairs).unwrap();
    a.commit(db).unwrap();
}

fn init_nomt_db(path: &Path) -> Nomt<Blake3Hasher> {
    let mut o = Options::new();
    o.path(path);
    o.hashtable_buckets(327680);
    Nomt::open(o).unwrap()
}
