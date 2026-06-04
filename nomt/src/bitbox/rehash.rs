use std::{
    fs::{File, OpenOptions},
    io::{Read as _, Write as _},
    os::unix::fs::FileExt as _,
    path::Path,
};

use anyhow::Context as _;

use crate::{
    io::{self, PagePool, PAGE_SIZE},
    store::meta::Meta,
};

use super::{allocate_bucket_raw, ht_file, meta_map::MetaMap, recover};

const TMP_HT_FILE: &str = "ht.rehashing";
const MARKER_FILE: &str = "ht.rehashing-marker";
const MARKER_MAGIC: [u8; 8] = *b"NOMTRH1\0";
const MARKER_LEN: usize = 16;

struct RehashMarker {
    old_num_pages: u32,
    new_num_pages: u32,
}

impl RehashMarker {
    fn encode(&self) -> [u8; MARKER_LEN] {
        let mut out = [0u8; MARKER_LEN];
        out[..8].copy_from_slice(&MARKER_MAGIC);
        out[8..12].copy_from_slice(&self.old_num_pages.to_le_bytes());
        out[12..16].copy_from_slice(&self.new_num_pages.to_le_bytes());
        out
    }

    fn decode(bytes: &[u8]) -> anyhow::Result<Self> {
        if bytes.len() != MARKER_LEN {
            anyhow::bail!("invalid rehash marker length: {}", bytes.len());
        }
        if bytes[..8] != MARKER_MAGIC {
            anyhow::bail!("invalid rehash marker magic");
        }

        let old_num_pages = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        let new_num_pages = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        if new_num_pages <= old_num_pages {
            anyhow::bail!(
                "invalid rehash marker bucket counts: old={}, new={}",
                old_num_pages,
                new_num_pages
            );
        }

        Ok(Self {
            old_num_pages,
            new_num_pages,
        })
    }
}

pub(crate) fn finish_pending_rehash(path: &Path, page_pool: &PagePool) -> anyhow::Result<()> {
    let Some(marker) = read_marker(path)? else {
        return Ok(());
    };

    let meta_path = path.join("meta");
    let meta_fd = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&meta_path)
        .with_context(|| format!("failed to open {}", meta_path.display()))?;
    let mut meta = Meta::read(page_pool, &meta_fd)
        .with_context(|| format!("failed to read {}", meta_path.display()))?;
    meta.validate()?;

    let ht_path = path.join("ht");
    let ht_len = ht_path
        .metadata()
        .with_context(|| format!("failed to stat {}", ht_path.display()))?
        .len();
    let old_len = ht_file::expected_file_len(marker.old_num_pages);
    let new_len = ht_file::expected_file_len(marker.new_num_pages);

    match meta.bitbox_num_pages {
        n if n == marker.new_num_pages => {
            if ht_len != new_len {
                anyhow::bail!(
                    "pending rehash marker says meta is updated, but ht length is {}; expected {}",
                    ht_len,
                    new_len
                );
            }
        }
        n if n == marker.old_num_pages && ht_len == new_len => {
            meta.bitbox_num_pages = marker.new_num_pages;
            Meta::write(page_pool, &meta_fd, &meta)
                .context("failed to finish pending rehash metadata update")?;
        }
        n if n == marker.old_num_pages && ht_len == old_len => {
            // The marker was persisted before the replacement rename completed. Keep the old table
            // and discard the fully rebuilt temporary table, if it exists.
        }
        n => {
            anyhow::bail!(
                "pending rehash marker is inconsistent with meta/ht: meta buckets={}, old={}, new={}, ht_len={}",
                n,
                marker.old_num_pages,
                marker.new_num_pages,
                ht_len
            );
        }
    }

    remove_file_if_exists(&path.join(TMP_HT_FILE))?;
    remove_file_if_exists(&path.join(MARKER_FILE))?;
    sync_dir(path)?;
    Ok(())
}

pub(crate) fn grow_hashtable(
    path: &Path,
    page_pool: &PagePool,
    new_num_pages: u32,
    preallocate: bool,
) -> anyhow::Result<()> {
    if new_num_pages == 0 {
        anyhow::bail!("hashtable bucket count must be greater than zero");
    }

    finish_pending_rehash(path, page_pool)?;

    let meta_path = path.join("meta");
    let meta_fd = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&meta_path)
        .with_context(|| format!("failed to open {}", meta_path.display()))?;
    let mut meta = Meta::read(page_pool, &meta_fd)
        .with_context(|| format!("failed to read {}", meta_path.display()))?;
    meta.validate()?;

    if new_num_pages < meta.bitbox_num_pages {
        anyhow::bail!(
            "cannot shrink hashtable from {} to {} buckets",
            meta.bitbox_num_pages,
            new_num_pages
        );
    }
    if new_num_pages == meta.bitbox_num_pages {
        return Ok(());
    }

    remove_file_if_exists(&path.join(TMP_HT_FILE))?;
    rehash_to_tmp(path, page_pool, &meta, new_num_pages, preallocate)?;

    write_marker(
        path,
        RehashMarker {
            old_num_pages: meta.bitbox_num_pages,
            new_num_pages,
        },
    )?;

    std::fs::rename(path.join(TMP_HT_FILE), path.join("ht"))
        .context("failed to replace hashtable file")?;
    sync_dir(path)?;

    meta.bitbox_num_pages = new_num_pages;
    Meta::write(page_pool, &meta_fd, &meta).context("failed to update NOMT metadata")?;

    remove_file_if_exists(&path.join(MARKER_FILE))?;
    sync_dir(path)?;
    Ok(())
}

fn rehash_to_tmp(
    path: &Path,
    page_pool: &PagePool,
    meta: &Meta,
    new_num_pages: u32,
    preallocate: bool,
) -> anyhow::Result<()> {
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

    let (old_offsets, mut old_meta_map) = ht_file::open(meta.bitbox_num_pages, page_pool, &ht_fd)?;
    if wal_fd.metadata()?.len() > 0 {
        recover(
            meta.sync_seqn,
            &ht_fd,
            &wal_fd,
            page_pool,
            &old_offsets,
            &mut old_meta_map,
            meta.bitbox_seed,
        )?;
    }

    let new_meta_bytes =
        vec![0u8; ht_file::num_meta_byte_pages(new_num_pages) as usize * PAGE_SIZE];
    let mut new_meta_map = MetaMap::from_bytes(new_meta_bytes, new_num_pages as usize);
    if old_meta_map.full_count() > new_meta_map.len() {
        anyhow::bail!(
            "new hashtable has {} buckets but old table has {} occupied buckets",
            new_meta_map.len(),
            old_meta_map.full_count()
        );
    }

    let tmp_path = path.join(TMP_HT_FILE);
    let tmp_fd = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(&tmp_path)
        .with_context(|| format!("failed to create {}", tmp_path.display()))?;
    ht_file::resize_and_prealloc(
        &tmp_fd,
        ht_file::expected_file_len(new_num_pages),
        preallocate,
    )
    .with_context(|| format!("failed to resize {}", tmp_path.display()))?;

    let new_offsets = ht_file::HTOffsets::new(new_num_pages);
    for old_bucket in 0..old_meta_map.len() {
        if !old_meta_map.is_full(old_bucket) {
            continue;
        }

        let old_pn = old_offsets.data_page_index(old_bucket as u64);
        let page = io::read_page(page_pool, &ht_fd, old_pn)
            .with_context(|| format!("failed to read old hashtable bucket {}", old_bucket))?;
        let page_id_bytes: [u8; 32] = page[PAGE_SIZE - 32..].try_into().unwrap();
        let Some(new_bucket) =
            allocate_bucket_raw(page_id_bytes, &mut new_meta_map, &meta.bitbox_seed)
        else {
            anyhow::bail!(
                "failed to allocate bucket while rehashing old bucket {}",
                old_bucket
            );
        };
        let new_pn = new_offsets.data_page_index(new_bucket.0);
        tmp_fd
            .write_all_at(&page, new_pn * PAGE_SIZE as u64)
            .with_context(|| format!("failed to write new hashtable bucket {}", new_bucket.0))?;
    }

    for meta_page_ix in 0..ht_file::num_meta_byte_pages(new_num_pages) as usize {
        tmp_fd
            .write_all_at(
                new_meta_map.page_slice(meta_page_ix),
                meta_page_ix as u64 * PAGE_SIZE as u64,
            )
            .with_context(|| format!("failed to write new meta-map page {}", meta_page_ix))?;
    }

    tmp_fd
        .sync_all()
        .with_context(|| format!("failed to sync {}", tmp_path.display()))?;
    sync_dir(path)?;
    Ok(())
}

fn read_marker(path: &Path) -> anyhow::Result<Option<RehashMarker>> {
    let marker_path = path.join(MARKER_FILE);
    let mut fd = match File::open(&marker_path) {
        Ok(fd) => fd,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => {
            return Err(e).with_context(|| format!("failed to open {}", marker_path.display()))
        }
    };

    let mut bytes = Vec::new();
    fd.read_to_end(&mut bytes)
        .with_context(|| format!("failed to read {}", marker_path.display()))?;
    Ok(Some(RehashMarker::decode(&bytes)?))
}

fn write_marker(path: &Path, marker: RehashMarker) -> anyhow::Result<()> {
    let marker_path = path.join(MARKER_FILE);
    let mut fd = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&marker_path)
        .with_context(|| format!("failed to create {}", marker_path.display()))?;
    fd.write_all(&marker.encode())
        .with_context(|| format!("failed to write {}", marker_path.display()))?;
    fd.sync_all()
        .with_context(|| format!("failed to sync {}", marker_path.display()))?;
    sync_dir(path)?;
    Ok(())
}

fn remove_file_if_exists(path: &Path) -> std::io::Result<()> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e),
    }
}

fn sync_dir(path: &Path) -> std::io::Result<()> {
    File::open(path)?.sync_all()
}
