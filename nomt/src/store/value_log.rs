//! The value log holds the beatree changeset of the latest sync.
//!
//! A sync makes the log durable before the manifest, so the manifest can be written before the
//! beatree pages are. Until the beatree pages are durable, the manifest marks the log as pending
//! and recovery applies it to the previous beatree state.

use crate::beatree::{Key, ValueChange};
use std::{fs::File, os::unix::fs::FileExt as _};

const DELETE: u8 = 0;
const INSERT: u8 = 1;
const INSERT_OVERFLOW: u8 = 2;

/// Encodes the changeset of sync `sync_seqn`.
pub fn encode(sync_seqn: u32, changeset: &[(Key, ValueChange)]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(16 + changeset.len() * 64);
    buf.extend_from_slice(&sync_seqn.to_le_bytes());
    buf.extend_from_slice(&(changeset.len() as u64).to_le_bytes());
    for (key, change) in changeset {
        buf.extend_from_slice(key);
        match change {
            ValueChange::Delete => buf.push(DELETE),
            ValueChange::Insert(value) => {
                buf.push(INSERT);
                buf.extend_from_slice(&(value.len() as u32).to_le_bytes());
                buf.extend_from_slice(value);
            }
            ValueChange::InsertOverflow(value, hash) => {
                buf.push(INSERT_OVERFLOW);
                buf.extend_from_slice(&(value.len() as u32).to_le_bytes());
                buf.extend_from_slice(value);
                buf.extend_from_slice(hash);
            }
        }
    }
    buf
}

/// Replaces the log contents and makes them durable.
pub fn write(fd: &File, buf: &[u8]) -> std::io::Result<()> {
    fd.write_all_at(buf, 0)?;
    fd.set_len(buf.len() as u64)?;
    fd.sync_data()
}

/// Reads the changeset of sync `sync_seqn`.
pub fn read(fd: &File, sync_seqn: u32) -> anyhow::Result<Vec<(Key, ValueChange)>> {
    let len = fd.metadata()?.len() as usize;
    let mut buf = vec![0; len];
    fd.read_exact_at(&mut buf, 0)?;

    let mut pos = 0;
    let mut take = |n: usize| -> anyhow::Result<&[u8]> {
        let bytes = buf
            .get(pos..pos + n)
            .ok_or_else(|| anyhow::anyhow!("value log truncated"))?;
        pos += n;
        Ok(bytes)
    };
    let u32_at = |b: &[u8]| u32::from_le_bytes(b.try_into().unwrap());

    let log_seqn = u32_at(take(4)?);
    if log_seqn != sync_seqn {
        anyhow::bail!("value log is for sync {log_seqn}, expected {sync_seqn}");
    }
    let count = u64::from_le_bytes(take(8)?.try_into().unwrap()) as usize;
    let mut changeset = Vec::with_capacity(count);
    for _ in 0..count {
        let key: Key = take(32)?.try_into().unwrap();
        let change = match take(1)?[0] {
            DELETE => ValueChange::Delete,
            INSERT => {
                let len = u32_at(take(4)?) as usize;
                ValueChange::Insert(take(len)?.to_vec())
            }
            INSERT_OVERFLOW => {
                let len = u32_at(take(4)?) as usize;
                let value = take(len)?.to_vec();
                ValueChange::InsertOverflow(value, take(32)?.try_into().unwrap())
            }
            tag => anyhow::bail!("invalid value log entry tag {tag}"),
        };
        changeset.push((key, change));
    }
    Ok(changeset)
}
