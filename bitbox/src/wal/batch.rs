use crate::wal::{entry::Entry, record::Record, WAL_RECORD_SIZE};

// It can be constructed by multiple records
#[derive(Clone, PartialEq, Debug)]
pub struct Batch {
    sequence_number: u64,
    data: Vec<Entry>,
}

impl Batch {
    pub fn new(sequence_number: u64) -> Self {
        Batch {
            sequence_number,
            data: vec![],
        }
    }

    pub fn sequence_number(&self) -> u64 {
        self.sequence_number
    }

    pub fn data(self) -> Vec<Entry> {
        self.data
    }

    pub fn append_entry(&mut self, new_entry: Entry) {
        self.data.push(new_entry)
    }

    // Create a Batch given a set of record, it expect all of them
    // to have the same sequence_number and it keeps the order of entries
    //
    // panics if records is empty or one of the records has a different sequence_number
    pub fn from_records(records: Vec<Record>) -> Self {
        let sequence_number = records[0].sequence_number();

        let data = records
            .into_iter()
            .flat_map(|record| {
                if record.sequence_number() != sequence_number {
                    panic!("Record with different sequence_number were provided ")
                }
                record.data()
            })
            .collect();

        Self {
            sequence_number,
            data,
        }
    }

    pub fn to_records(&self) -> Vec<Record> {
        let mut records = vec![];
        let mut curr_record_data = vec![];
        let mut curr_record_size = Record::size_without_data();

        for entry in self.data.iter() {
            if curr_record_size + entry.len() > WAL_RECORD_SIZE {
                records.push((curr_record_data, false));
                curr_record_data = vec![];
                curr_record_size = Record::size_without_data();
            }
            curr_record_size += entry.len();
            curr_record_data.push(entry.clone());
        }

        if !curr_record_data.is_empty() {
            records.push((curr_record_data, true));
        } else if let Some((_data, ref mut last)) = records.last_mut() {
            *last = true;
        }

        records
            .into_iter()
            .map(|(record_data, last)| Record::new(self.sequence_number, last, record_data))
            .collect()
    }
}
