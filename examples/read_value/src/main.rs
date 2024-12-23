use anyhow::Result;
use nomt::{Blake3Hasher, KeyReadWrite, Nomt, Options};
use sha2::Digest;

const NOMT_DB_FOLDER: &str = "nomt_db";

fn main() -> Result<()> {
    // Define the options used to open NOMT
    let mut opts = Options::new();
    opts.path(NOMT_DB_FOLDER);
    opts.commit_concurrency(1);

    // Open nomt database. This will create the folder if it does not exist
    let nomt = Nomt::<Blake3Hasher>::open(opts)?;

    // Instantiate a new Session object to handle read and write operations
    // and generate a Witness later on
    let session = nomt.begin_session();

    // Reading a key from the database
    let key_path = sha2::Sha256::digest(b"key").into();
    let value = session.read(key_path)?;

    // Even though this key is only being read, we ask NOMT to warm up the on-disk data because
    // we will prove the read.
    session.warm_up(key_path);

    let _witness =
        nomt.update_commit_and_prove(session, vec![(key_path, KeyReadWrite::Read(value))])?;

    Ok(())
}
