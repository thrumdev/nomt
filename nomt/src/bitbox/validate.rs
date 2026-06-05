//! Validation utilities for the on-disk Bitbox hash table.

use std::{collections::HashSet, fs::OpenOptions, path::Path};

use anyhow::Context as _;

use crate::{
    io::{self, PagePool, PAGE_SIZE},
    store::meta::Meta,
};

use super::{
    finish_pending_rehash, hash_raw_page_id, ht_file, meta_map::MetaMap, recover,
    HashTableUtilization,
};

const MAX_VALIDATION_PROBES: usize = 100_000;

/// Validate the on-disk hash table layout.
pub(crate) fn validate_hashtable(
    path: &Path,
    page_pool: &PagePool,
) -> anyhow::Result<HashTableUtilization> {
    finish_pending_rehash(path, page_pool)?;

    let meta_path = path.join("meta");
    let meta_fd = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&meta_path)
        .with_context(|| format!("failed to open {}", meta_path.display()))?;
    let meta = Meta::read(page_pool, &meta_fd)
        .with_context(|| format!("failed to read {}", meta_path.display()))?;
    meta.validate()?;

    let ht_path = path.join("ht");
    let ht_fd = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&ht_path)
        .with_context(|| format!("failed to open {}", ht_path.display()))?;
    let wal_path = path.join("wal");
    let wal_fd = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&wal_path)
        .with_context(|| format!("failed to open {}", wal_path.display()))?;

    let (offsets, mut meta_map) = ht_file::open(meta.bitbox_num_pages, page_pool, &ht_fd)?;
    if meta_map.len() == 0 {
        anyhow::bail!("hash table has zero buckets");
    }

    if wal_fd.metadata()?.len() > 0 {
        recover(
            meta.sync_seqn,
            &ht_fd,
            &wal_fd,
            page_pool,
            &offsets,
            &mut meta_map,
            meta.bitbox_seed,
        )?;
    }

    if !meta_map.padding_is_empty() {
        anyhow::bail!("hash table meta-map has non-empty padding past logical buckets");
    }

    let mut labels = HashSet::new();
    for bucket in 0..meta_map.len() {
        if !meta_map.is_full(bucket) {
            continue;
        }

        let pn = offsets.data_page_index(bucket as u64);
        let page = io::read_page(page_pool, &ht_fd, pn)
            .with_context(|| format!("failed to read hash table bucket {}", bucket))?;
        let page_label: [u8; 32] = page[PAGE_SIZE - 32..].try_into().unwrap();
        if !labels.insert(page_label) {
            anyhow::bail!(
                "duplicate page label {} in hash table bucket {}",
                hex_label(page_label),
                bucket
            );
        }

        let hash = hash_raw_page_id(page_label, &meta.bitbox_seed);
        if meta_map.hint_not_match(bucket, hash) {
            anyhow::bail!(
                "hash table bucket {} has page label {} but mismatched meta hint",
                bucket,
                hex_label(page_label)
            );
        }

        ensure_bucket_reachable(&meta_map, page_label, bucket, &meta.bitbox_seed)?;
    }

    Ok(HashTableUtilization {
        capacity: meta_map.len(),
        occupied: labels.len(),
    })
}

fn ensure_bucket_reachable(
    meta_map: &MetaMap,
    page_label: [u8; 32],
    expected_bucket: usize,
    seed: &[u8; 16],
) -> anyhow::Result<()> {
    let hash = hash_raw_page_id(page_label, seed);
    let mut bucket = hash % meta_map.len() as u64;
    let mut step = 0u64;
    let max_probes = meta_map.len().saturating_mul(2).max(MAX_VALIDATION_PROBES);

    for _ in 0..max_probes {
        bucket += step;
        step += 1;
        bucket %= meta_map.len() as u64;

        let bucket = bucket as usize;
        if bucket == expected_bucket {
            return Ok(());
        }

        if meta_map.hint_empty(bucket) {
            anyhow::bail!(
                "hash table bucket {} for page label {} is unreachable; probe hit empty bucket {} first",
                expected_bucket,
                hex_label(page_label),
                bucket,
            );
        }
    }

    anyhow::bail!(
        "hash table bucket {} for page label {} was not reached within {} probes",
        expected_bucket,
        hex_label(page_label),
        max_probes,
    );
}

fn hex_label(label: [u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(64);
    for byte in label {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}
