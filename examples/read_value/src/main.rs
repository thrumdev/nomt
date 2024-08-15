use anyhow::Result;
use nomt::{Nomt, Options};
use sha2::Digest;

const NOMT_DB_FOLDER: &str = "nomt_db";

fn main() -> Result<()> {
    // Define the options used to open NOMT
    let mut opts = Options::new();
    opts.path(NOMT_DB_FOLDER);
    opts.commit_concurrency(1);

    // Open nomt database, it will create the folder if it does not exist
    let nomt = Nomt::open(opts)?;

    // Instantiate a new Session object to handle read and write operations
    // and generate a Witness later on
    let mut session = nomt.begin_session();

    // Reading a key from the database
    let key_path = sha2::Sha256::digest(b"key").into();
    let _value = session.tentative_read_slot(key_path)?;

    Ok(())
}
