#![no_main]

use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use nomt::{
    grow_hashtable, hasher::Blake3Hasher, trie::KeyPath, validate_hashtable, KeyReadWrite, Nomt,
    Options, SessionParams, Value,
};

fuzz_target!(|run: Run| {
    let tempdir = tempfile::tempdir().unwrap();
    let path = tempdir.path().join("db");
    let mut buckets = run.initial_buckets;
    let mut db = Some(Nomt::<Blake3Hasher>::open(options(&path, buckets)).unwrap());
    let mut model = BTreeMap::<KeyPath, Value>::new();
    let mut touched = BTreeSet::<KeyPath>::new();
    let mut snapshots = vec![model.clone()];

    for op in run.ops {
        match op {
            Op::Commit(changes) => {
                let Some(nomt) = db.as_ref() else {
                    unreachable!("database should be open before commit")
                };
                if commit(nomt, changes, &mut model, &mut touched) {
                    snapshots.push(model.clone());
                }
                assert_model(nomt, &model, &touched);
            }
            Op::Grow(extra) => {
                let Some(nomt) = db.take() else {
                    unreachable!("database should be open before grow")
                };
                let root = nomt.root();
                drop(nomt);

                let requested = buckets.saturating_add(1).saturating_add(extra as u32 * 512);
                grow_hashtable(&options(&path, requested)).unwrap();
                buckets = requested;

                let utilization = validate_hashtable(&options(&path, buckets)).unwrap();
                assert_eq!(utilization.capacity, buckets as usize);

                let nomt = Nomt::<Blake3Hasher>::open(options(&path, buckets)).unwrap();
                assert_eq!(nomt.root(), root);
                assert_model(&nomt, &model, &touched);
                db = Some(nomt);
            }
            Op::Rollback(raw_n) => {
                if snapshots.len() <= 1 {
                    continue;
                }

                let n = raw_n as usize % (snapshots.len() - 1) + 1;
                let target = snapshots.len() - 1 - n;
                let Some(nomt) = db.as_ref() else {
                    unreachable!("database should be open before rollback")
                };
                nomt.rollback(n).unwrap();
                model = snapshots[target].clone();
                snapshots.push(model.clone());
                assert_model(nomt, &model, &touched);
            }
            Op::Reopen => {
                drop(db.take());
                let nomt = Nomt::<Blake3Hasher>::open(options(&path, buckets)).unwrap();
                assert_model(&nomt, &model, &touched);
                db = Some(nomt);
            }
            Op::Validate => {
                let Some(nomt) = db.take() else {
                    unreachable!("database should be open before validate")
                };
                let root = nomt.root();
                drop(nomt);

                validate_hashtable(&options(&path, buckets)).unwrap();

                let nomt = Nomt::<Blake3Hasher>::open(options(&path, buckets)).unwrap();
                assert_eq!(nomt.root(), root);
                assert_model(&nomt, &model, &touched);
                db = Some(nomt);
            }
        }
    }
});

fn options(path: &Path, buckets: u32) -> Options {
    let mut options = Options::new();
    options.path(path);
    options.bitbox_seed([0; 16]);
    options.hashtable_buckets(buckets);
    options.io_workers(1);
    options.rollback(true);
    options.max_rollback_log_len(128);
    options.preallocate_ht(false);
    options
}

fn commit(
    nomt: &Nomt<Blake3Hasher>,
    changes: Vec<Change>,
    model: &mut BTreeMap<KeyPath, Value>,
    touched: &mut BTreeSet<KeyPath>,
) -> bool {
    let mut dedup = BTreeMap::<KeyPath, Option<Value>>::new();
    for change in changes {
        dedup.insert(change.key, change.value);
    }
    if dedup.is_empty() {
        return false;
    }

    let session = nomt.begin_session(SessionParams::default());
    let operations = dedup
        .iter()
        .map(|(key, value)| (*key, KeyReadWrite::Write(value.clone())))
        .collect::<Vec<_>>();
    for (key, _) in &operations {
        session.warm_up(*key);
    }
    session.finish(operations).unwrap().commit(nomt).unwrap();

    for (key, value) in dedup {
        touched.insert(key);
        if let Some(value) = value {
            model.insert(key, value);
        } else {
            model.remove(&key);
        }
    }

    true
}

fn assert_model(
    nomt: &Nomt<Blake3Hasher>,
    model: &BTreeMap<KeyPath, Value>,
    touched: &BTreeSet<KeyPath>,
) {
    for key in touched {
        assert_eq!(nomt.read(*key).unwrap().as_ref(), model.get(key));
    }
}

#[derive(Debug)]
struct Run {
    initial_buckets: u32,
    ops: Vec<Op>,
}

#[derive(Debug)]
enum Op {
    Commit(Vec<Change>),
    Grow(u8),
    Rollback(u8),
    Reopen,
    Validate,
}

#[derive(Debug)]
struct Change {
    key: KeyPath,
    value: Option<Value>,
}

impl<'a> Arbitrary<'a> for Run {
    fn arbitrary(input: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        let initial_buckets = *input.choose(&[4096u32, 8192])?;
        let op_count = input.int_in_range(0..=16)?;
        let mut ops = Vec::with_capacity(op_count);
        for _ in 0..op_count {
            ops.push(Op::arbitrary(input)?);
        }

        Ok(Self {
            initial_buckets,
            ops,
        })
    }
}

impl<'a> Arbitrary<'a> for Op {
    fn arbitrary(input: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        Ok(match input.int_in_range(0..=9)? {
            0..=4 => {
                let len = input.int_in_range(0..=8)?;
                let mut changes = Vec::with_capacity(len);
                for _ in 0..len {
                    changes.push(Change::arbitrary(input)?);
                }
                Self::Commit(changes)
            }
            5..=6 => Self::Grow(input.arbitrary()?),
            7 => Self::Rollback(input.arbitrary()?),
            8 => Self::Reopen,
            9 => Self::Validate,
            _ => unreachable!(),
        })
    }
}

impl<'a> Arbitrary<'a> for Change {
    fn arbitrary(input: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        let mut key = [0; 32];
        input.fill_buffer(&mut key)?;

        let value = if input.ratio(1, 4)? {
            None
        } else {
            Some(arbitrary_value(input)?)
        };

        Ok(Self { key, value })
    }
}

fn arbitrary_value(input: &mut arbitrary::Unstructured<'_>) -> arbitrary::Result<Value> {
    let len = match input.int_in_range(0..=7)? {
        0 => 0,
        1 => 1,
        2 => input.int_in_range(2..=32)?,
        3 => input.int_in_range(33..=256)?,
        4 => input.int_in_range(257..=1333)?,
        5 => input.int_in_range(1334..=2048)?,
        _ => input.int_in_range(2049..=4096)?,
    };
    let mut value = vec![0; len];
    input.fill_buffer(&mut value)?;
    Ok(value)
}
