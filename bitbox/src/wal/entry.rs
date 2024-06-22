use crate::sim::{PageDiff, PageId};
use anyhow::bail;
use bitvec::{prelude::Msb0, view::AsBits};

#[derive(Clone, PartialEq, Debug)]
pub enum Entry {
    // kind = 0
    Clear {
        bucket_index: u64,
    },
    // kind = 1
    Update {
        page_id: PageId,
        page_diff: PageDiff,
        changed: Vec<[u8; 32]>,
        bucket_index: u64,
    },
}

impl Entry {
    fn id(&self) -> u8 {
        match self {
            Entry::Update { .. } => 1,
            Entry::Clear { .. } => 0,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Entry::Update { changed, .. } => 1 + 16 + 16 + changed.len() * 32 + 8,
            Entry::Clear { .. } => 1 + 8,
        }
    }

    // Try parse an entry from the given bytes, returning the entry
    // and the number of consumed bytes
    pub fn from_bytes(raw_entry: &[u8]) -> anyhow::Result<(Self, usize)> {
        // the minumum size is 9, in the case it is and Entry::Clear
        if raw_entry.len() < 9 {
            bail!("Error in Entry format")
        }

        // parse entry kind
        let kind = {
            let mut buf = [0; 1];
            buf.copy_from_slice(&raw_entry[0..1]);
            u8::from_le_bytes(buf)
        };

        if kind == 0 {
            // Parse Entry::Clear
            let bucket_index = {
                let mut buf = [0; 8];
                buf.copy_from_slice(&raw_entry[1..9]);
                u64::from_le_bytes(buf)
            };

            return Ok((Entry::Clear { bucket_index }, 9));
        } else if kind != 1 {
            bail!("Invalid Entry Kind")
        }

        // Parse Entry::Update
        let (raw_page_id, raw_page_diff) = {
            if raw_entry.len() < 33 {
                bail!("Error in Entry format")
            }
            (&raw_entry[1..17], &raw_entry[17..33])
        };

        let page_id = {
            let mut buf = [0; 16];
            buf.copy_from_slice(raw_page_id);
            buf
        };

        let page_diff = {
            let mut buf = [0; 16];
            buf.copy_from_slice(raw_page_diff);
            buf
        };

        let n_changed = page_diff.as_bits::<Msb0>().count_ones();

        let expected_consume = 33 + n_changed * 32 + 8;
        if raw_entry.len() < expected_consume {
            bail!("Not enough space for changed and bucket_index in Entry")
        }

        let mut changed_ptr = 33;
        let changed: Vec<_> = (0..n_changed)
            .map(|_| {
                let mut buf = [0; 32];

                buf.copy_from_slice(&raw_entry[changed_ptr..changed_ptr + 32]);
                changed_ptr += 32;

                buf
            })
            .collect();

        let end_changed = 33 + n_changed * 32;
        // TODO: remove
        assert_eq!(changed_ptr, end_changed);

        let bucket_index = {
            let mut buf = [0; 8];
            buf.copy_from_slice(&raw_entry[end_changed..end_changed + 8]);
            u64::from_le_bytes(buf)
        };

        Ok((
            Entry::Update {
                page_id,
                page_diff,
                changed,
                bucket_index,
            },
            expected_consume,
        ))
    }

    pub fn to_bytes(self) -> Vec<u8> {
        let mut raw_entry = vec![];

        match self {
            Entry::Clear { bucket_index } => {
                raw_entry.push(0u8);
                raw_entry.extend(bucket_index.to_le_bytes());
            }
            Entry::Update {
                page_id,
                page_diff,
                changed,
                bucket_index,
            } => {
                raw_entry.push(1u8);
                raw_entry.extend(page_id);
                raw_entry.extend(page_diff);
                raw_entry.extend(changed.into_iter().flatten());
                raw_entry.extend(bucket_index.to_le_bytes());
            }
        };
        raw_entry
    }
}

#[cfg(test)]
mod tests {

    use super::Entry;

    #[test]
    fn entry_code_and_decode() {
        let entry = Entry::Clear { bucket_index: 15 };
        let raw_entry = entry.clone().to_bytes();
        let (decoded_entry, consumed) = Entry::from_bytes(&raw_entry).unwrap();
        assert_eq!(entry, decoded_entry);
        assert_eq!(consumed, entry.len());

        let mut page_diff = [0; 16];
        page_diff[0] = 0b00001001;
        let entry = Entry::Update {
            page_id: [1; 16],
            page_diff,
            changed: vec![[7; 32], [9; 32]],
            bucket_index: 1107,
        };
        let raw_entry = entry.clone().to_bytes();
        let (decoded_entry, consumed) = Entry::from_bytes(&raw_entry).unwrap();
        assert_eq!(entry, decoded_entry);
        assert_eq!(consumed, entry.len());
    }
}
