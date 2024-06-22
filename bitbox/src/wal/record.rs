use crate::wal::entry::Entry;
use crate::wal::WAL_CHECKSUM_MAGIC;
use crate::wal::WAL_RECORD_SIZE;

use anyhow::bail;

const LAST_IN_SEQUENCE: u8 = 0xFF;
const NOT_LAST_IN_SEQUENCE: u8 = 0x00;
const DATA_START: usize = 15;

// Item stored in WAL, mult
#[derive(Clone, PartialEq, Debug)]
pub struct Record {
    sequence_number: u64,
    last: bool,
    data: Vec<Entry>,
}

impl Record {
    // panics if the size of the record is bigger then WAL_RECORD_SIZE
    pub fn new(sequence_number: u64, last: bool, data: Vec<Entry>) -> Self {
        let record = Self {
            sequence_number,
            last,
            data,
        };

        //TODO check size
        record
    }

    pub fn sequence_number(&self) -> u64 {
        self.sequence_number
    }

    pub fn last(&self) -> bool {
        self.last
    }

    pub fn data(self) -> Vec<Entry> {
        self.data
    }

    pub fn size_without_data() -> usize {
        8 + 4 + 2 + 1
    }

    pub fn from_bytes(mut raw_record: [u8; WAL_RECORD_SIZE]) -> anyhow::Result<Self> {
        let sequence_number = {
            let mut buf = [0u8; 8];
            buf.copy_from_slice(&raw_record[..8]);
            u64::from_le_bytes(buf)
        };

        let checksum_expected = {
            let mut buf = [0u8; 4];
            buf.copy_from_slice(&raw_record[8..12]);
            u32::from_le_bytes(buf)
        };

        let entry_count = {
            let mut buf = [0u8; 2];
            buf.copy_from_slice(&raw_record[12..14]);
            u16::from_le_bytes(buf)
        };

        let last_indicator = raw_record[14];
        let last = if last_indicator == LAST_IN_SEQUENCE {
            true
        } else if last_indicator == NOT_LAST_IN_SEQUENCE {
            false
        } else {
            bail!("invalid last-indicator byte");
        };

        let checksum = checksum(last_indicator, &raw_record[DATA_START..]);

        if checksum_expected != checksum {
            bail!("Wrong checksum")
        }

        let mut offset = DATA_START;
        let data = (0..entry_count)
            .map(|_| {
                let (entry, consumed) = Entry::from_bytes(&raw_record[offset..])?;
                offset += consumed;
                Ok(entry)
            })
            .collect::<anyhow::Result<Vec<Entry>>>()?;

        Ok(Self {
            sequence_number,
            last,
            data,
        })
    }

    pub fn to_bytes(self) -> [u8; WAL_RECORD_SIZE] {
        let mut raw_record = [0u8; WAL_RECORD_SIZE];
        raw_record[0..8].copy_from_slice(&self.sequence_number.to_le_bytes());
        // skip checksum, fill later

        let entry_count: u16 = self
            .data
            .len()
            .try_into()
            .expect("Entries must fit in WAL_RECORD_SIZE");

        let last_indicator = if self.last {
            LAST_IN_SEQUENCE
        } else {
            NOT_LAST_IN_SEQUENCE
        };

        raw_record[14] = last_indicator;

        let mut offset = DATA_START;
        for entry in self.data {
            let entry_len = entry.len();
            raw_record[offset..offset + entry_len].copy_from_slice(&entry.to_bytes());
            offset += entry_len;
        }

        let checksum = checksum(last_indicator, &raw_record[DATA_START..]);

        raw_record[8..12].copy_from_slice(&checksum.to_le_bytes());
        raw_record[12..14].copy_from_slice(&entry_count.to_le_bytes());
        raw_record
    }
}

fn checksum(last: u8, data: &[u8]) -> u32 {
    let mut buf = [0u8; 4];
    let mut hasher = blake3::Hasher::new();
    hasher.update(&WAL_CHECKSUM_MAGIC.to_le_bytes());
    hasher.update(&[last]);
    hasher.update(data);

    let checksum_hash = hasher.finalize();

    buf[..].copy_from_slice(&checksum_hash.as_bytes()[..4]);
    u32::from_le_bytes(buf)
}
