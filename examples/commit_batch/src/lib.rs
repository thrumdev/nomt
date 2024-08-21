use anyhow::Result;
use nomt::{KeyReadWrite, Node, Nomt, Options, Witness, WitnessedOperations};
use sha2::Digest;

const NOMT_DB_FOLDER: &str = "nomt_db";

pub struct NomtDB;

impl NomtDB {
    pub fn commit_batch() -> Result<(Node, Node, Witness, WitnessedOperations)> {
        // Define the options used to open NOMT
        let mut opts = Options::new();
        opts.path(NOMT_DB_FOLDER);
        opts.commit_concurrency(1);

        // Open NOMT database, it will create the folder if it does not exist
        let nomt = Nomt::open(opts)?;

        // Create a new Session object
        //
        // During a session, the backend is responsible for returning read keys
        // and receiving hints about future writes
        //
        // Writes do not occur immediately, instead,
        // they are cached and applied all at once later on
        let mut session = nomt.begin_session();

        // Here we will move the data saved under b"key1" to b"key2" and deletes it
        //
        // NOMT expects keys to be uniformly distributed across the key space
        let key_path_1 = sha2::Sha256::digest(b"key1").into();
        let key_path_2 = sha2::Sha256::digest(b"key2").into();

        // First, read what is under key_path_1
        //
        // `tentative_read_slot` will immediately return the value present in the database
        let value = session.tentative_read_slot(key_path_1)?;

        // Second, write the value to key_path_2 and delete key_path_1
        //
        // As we can observe, the value is not being written at the moment.
        // NOMT is just advertised here to inform that those keys
        // will be written during the commit and prove stage
        session.tentative_write_slot(key_path_1);
        session.tentative_write_slot(key_path_2);

        // Retrieve the previous value of the root before committing changes
        let prev_root = nomt.root();

        // To commit the batch to the backend we need to collect every
        // performed actions into a vector where items are ordered by the key_path
        let mut actual_access: Vec<_> = vec![
            (key_path_1, KeyReadWrite::ReadThenWrite(value.clone(), None)),
            (key_path_2, KeyReadWrite::Write(value)),
        ];
        actual_access.sort_by_key(|(k, _)| *k);

        // The final step in handling a session involves committing all changes
        // to update the trie structure and obtaining the new root of the trie,
        // along with a witness and the witnessed operations.
        let (root, witness, witnessed) = nomt.commit_and_prove(session, actual_access)?;

        Ok((prev_root, root, witness, witnessed))
    }
}
