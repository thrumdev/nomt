mod common;
use nomt::KeyReadWrite;
use common::Test;

#[test]
fn apply_actuals() {
    // 1. Change the path to where the actuals are stored.
    let actuals_folder_path = "/home/gab/work/nomt/nomt/actuals";
    let actual_path =  format!("{}/actual", actuals_folder_path);

    // 2. Match with the used options. Cache sizes should not matter.
    let commit_concurrency = 1;
    let n_buckets = 50_000_000;

    let mut t = Test::new_with_params("apply_actuals", commit_concurrency, n_buckets , None, true);

    for i in 0.. {
        let actual_path_name = format!("{}{}", actual_path, i);

        if !std::fs::exists(&actual_path_name).unwrap() {
            break;
        }

        let raw_actuals = std::fs::read_to_string(&actual_path_name).unwrap();
        let actuals: Vec<([u8; 32], KeyReadWrite)> = serde_json::from_str(&raw_actuals).unwrap();

        t.commit_actual(actuals);
    }
}
