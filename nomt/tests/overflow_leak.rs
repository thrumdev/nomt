mod common;

use common::Test;

// Replacing a large value must free the overflow pages of the value it replaces.
//
// The leaf file grows in steps of 8192 pages (32 MiB). One live 256 KiB value fits in the first
// step however often it is rewritten; 300 leaked copies need 19200 pages.
#[test]
fn replaced_overflow_value_is_freed() {
    let name = "overflow_leak";
    let mut t = Test::new(name);
    for i in 0..300u32 {
        t.write_id(0, Some(vec![i as u8; 4096 * 64]));
        t.commit();
    }
    let ln_len = std::fs::metadata(format!("test/{name}/ln")).unwrap().len();
    assert!(
        ln_len <= 8192 * 4096,
        "leaf file is {ln_len} bytes for a single live 256 KiB value"
    );
}
