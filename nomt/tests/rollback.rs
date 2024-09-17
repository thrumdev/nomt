use hex_literal::hex;
use nomt::{KeyReadWrite, Nomt, Options};
use std::{path::PathBuf, rc::Rc};

#[test]
fn test_rollback_disabled() {
    let path = {
        let mut p = PathBuf::from("test");
        p.push("rollback_disabled");
        p
    };
    let _ = std::fs::remove_dir_all(&path);
    let mut o = Options::new();
    o.path(path);
    o.commit_concurrency(1);
    o.panic_on_sync(false);
    o.bitbox_seed([0; 16]);
    o.rollback(false);
    let nomt = Nomt::open(o).unwrap();

    let session = nomt.begin_session();
    nomt.commit(
        session,
        vec![(
            hex!("0000000000000000000000000000000000000000000000000000000000000001"),
            KeyReadWrite::Write(Some(Rc::new(vec![1]))),
        )],
    )
    .unwrap();

    let result = nomt.rollback_n(1);
    // we expect this to fail, because rollback is disabled
    assert!(result.is_err());
}

#[test]
fn test_rollback_to_initial() {
    let path = {
        let mut p = PathBuf::from("test");
        p.push("rollback_to_initial");
        p
    };
    let _ = std::fs::remove_dir_all(&path);
    let mut o = Options::new();
    o.path(path);
    o.commit_concurrency(1);
    o.panic_on_sync(false);
    o.bitbox_seed([0; 16]);
    o.rollback(true);
    let nomt = Nomt::open(o).unwrap();

    let session = nomt.begin_session();
    nomt.commit(
        session,
        vec![(
            hex!("0000000000000000000000000000000000000000000000000000000000000001"),
            KeyReadWrite::Write(Some(Rc::new(vec![1]))),
        )],
    )
    .unwrap();
    assert_eq!(
        nomt.root(),
        hex!("c6e25744545ddabdaf0a95201f8285e670ee9b3e0c1ced4a3006baafd1ac2fdf")
    );

    let result = nomt.rollback_n(1);
    assert!(result.is_ok());
    assert_eq!(
        nomt.root(),
        hex!("0000000000000000000000000000000000000000000000000000000000000000")
    );
}

// TODO: Implement tests for the following scenarios:
// 1. N separate rollbacks of single commit each should be equivalent to one rollback of N commits.
// 2. N separate rollbacks of single commit each should be equivalent to:
//    - One rollback of 1 commit, followed by one rollback of N-1 commits
//    - One rollback of 2 commits, followed by one rollback of N-2 commits
//    - ...
//    - One rollback of N-1 commits, followed by one rollback of 1 commit
// This ensures that the rollback operation is consistent regardless of how it's split up.
