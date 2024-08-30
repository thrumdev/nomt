use std::{path::PathBuf, rc::Rc};

use bitvec::{order::Msb0, view::BitView};
use nomt::{KeyReadWrite, Nomt, Options};

mod common;

#[test]
fn branch_node_builder_test() {
    let path = {
        let mut p = PathBuf::from("test");
        p.push("branch_node");
        p
    };
    let _ = std::fs::remove_dir_all(&path);

    let mut opts = Options::new();
    opts.path(path);
    opts.commit_concurrency(1);
    opts.panic_on_sync(false);
    opts.bitbox_seed([0; 16]);

    let nomt = Nomt::open(opts).unwrap();
    let session = nomt.begin_session();

    let mut key1 = [255; 32];
    key1[0] = 126;
    let mut key2 = [0; 32];
    key2[0] = 128;
    key2[31] = 254;

    let mut key3 = [0; 32];
    key3[0] = 128;
    key3[31] = 255;
    let mut key4 = [0; 32];
    key4[0] = 129;
    key4[31] = 255;

    let actuals = vec![
        (key1, KeyReadWrite::Write(Some(Rc::new(vec![1; 1024])))),
        (key2, KeyReadWrite::Write(Some(Rc::new(vec![1; 1024])))),
        (key3, KeyReadWrite::Write(Some(Rc::new(vec![1; 1024])))),
        (key4, KeyReadWrite::Write(Some(Rc::new(vec![1; 1024])))),
    ];

    let _ = nomt.commit_and_prove(session, actuals);
}
