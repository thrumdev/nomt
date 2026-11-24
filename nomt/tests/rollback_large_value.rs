use nomt::{hasher::Blake3Hasher, KeyReadWrite, Nomt, Options, SessionParams};
use std::path::Path;

fn open(path: &Path) -> Nomt<Blake3Hasher> {
    let mut o = Options::new();
    o.path(path);
    o.commit_concurrency(1);
    o.io_workers(1);
    o.bitbox_seed([0; 16]);
    o.hashtable_buckets(4096);
    o.rollback(true);
    o.warm_up(false);
    Nomt::open(o).unwrap()
}

fn write(nomt: &Nomt<Blake3Hasher>, key: [u8; 32], value: Option<Vec<u8>>) {
    nomt.begin_session(SessionParams::default())
        .finish(vec![(key, KeyReadWrite::Write(value))])
        .unwrap()
        .commit(nomt)
        .unwrap();
}

const KEY: [u8; 32] = [0x42; 32];
const CHILD: &str = "LARGE_PRIOR_CHILD";

#[test]
fn prior_of_large_overflow_value_is_fetched() {
    use std::os::unix::process::ExitStatusExt as _;

    let path = std::path::PathBuf::from("test/prior_of_large_overflow_value");
    if std::env::var_os(CHILD).is_some() {
        let nomt = open(&path);
        let session = nomt.begin_session(SessionParams::default());
        session.preserve_prior_value(KEY);
        let finished = session.finish(vec![(KEY, KeyReadWrite::Write(None))]);
        eprintln!("child: finish returned is_ok={}", finished.is_ok());
        finished.unwrap().commit(&nomt).unwrap();
        nomt.rollback(1).unwrap();
        assert_eq!(nomt.read(KEY).unwrap(), Some(vec![7u8; 100_000]));
        return;
    }

    let _ = std::fs::remove_dir_all(&path);
    let nomt = open(&path);
    write(&nomt, KEY, Some(vec![7u8; 100_000]));
    write(&nomt, [0x01; 32], Some(vec![1]));
    write(&nomt, [0xf1; 32], Some(vec![2]));
    drop(nomt);

    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "prior_of_large_overflow_value_is_fetched",
            "--nocapture",
        ])
        .env(CHILD, "1")
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    for line in stderr.lines().filter(|l| {
        l.contains("panicked")
            || l.contains("index out of")
            || l.contains("abort")
            || l.contains("child:")
    }) {
        eprintln!("child stderr: {line}");
    }
    assert!(
        out.status.success(),
        "deleting a 100 kB value with a cold leaf cache killed the process: signal {:?}, code {:?}",
        out.status.signal(),
        out.status.code()
    );
}
