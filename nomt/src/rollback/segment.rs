use std::{fs::File, path::Path};

pub fn format_segment_file_name(segment_id: u32) -> String {
    // The format string specifies a 10-digit number, so we pad with leading zeros from
    // the left. This assumes that segment_id is a 32-bit integer, which is confirmed by
    // the assert below. If you came here because it failed due to changing it to u64,
    // you will need to update the format string as well.
    assert_eq!(segment_id.to_le_bytes().len(), 4);
    format!("rollback.{segment_id:0>10}.log")
}

pub fn is_valid_segment_file(path: &Path) -> bool {
    if let Some(file_name) = path.file_name() {
        if let Some(file_name_str) = file_name.to_str() {
            // we assume that the file name is always a valid ASCII and thus utf-8 string.
            return file_name_str.starts_with("rollback.");
        }
    }
    false
}

pub fn mk_rollback_segment(db_dir_path: &Path, new_segment_id: u32) -> anyhow::Result<File> {
    let file_name = format_segment_file_name(new_segment_id);
    let file_path = db_dir_path.join(file_name);
    let new_segment_file = File::options().create(true).append(true).open(file_path)?;
    Ok(new_segment_file)
}
