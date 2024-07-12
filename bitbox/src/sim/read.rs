use bitvec::prelude::*;
use crossbeam_channel::{Receiver, Sender, TrySendError};
use rand::Rng;

use std::collections::VecDeque;
use std::sync::{Arc, Barrier, RwLock};

use crate::io::{CompleteIo, IoCommand, IoKind};
use crate::meta_map::MetaMap;
use crate::store::{Page, Store};

use super::{
    slot_range, BucketIndex, ChangedPage, Map, PageDiff, PageId, Params, ProbeSequence,
    SLOTS_PER_PAGE,
};

struct WorkItem {
    pages: Vec<Fetch>,
    requested: usize,
    received: usize,
}

struct Fetch {
    page_id: PageId,
    probe: Option<ProbeSequence>,
    buf: Option<Box<Page>>,
}

fn make_workload(worker_index: usize, params: &Params) -> Vec<WorkItem> {
    // create workload - (preload_count + 1) * workload_size random "page IDs"
    (0..params.workload_size)
        .map(|_| {
            let mut rng = rand::thread_rng();

            // sample from typical range if not cold, out of range if cold
            let range = if rng.gen::<f32>() < params.cold_rate {
                params.num_pages..usize::max_value()
            } else {
                0..params.num_pages
            };

            // turn each sample into a random hash, set a unique byte per worker to discriminate.
            let pages = (0..params.preload_count + 1)
                .map(move |_| rand::thread_rng().gen_range(range.clone()))
                .map(|sample| {
                    // partition the page set across workers so it's the same regardless
                    // of num workers with no overlaps.
                    let sample = sample * params.num_workers + worker_index;
                    blake3::hash(sample.to_le_bytes().as_slice())
                })
                .map(|hash| {
                    let mut page_id = [0; 16];
                    page_id.copy_from_slice(&hash.as_bytes()[..16]);
                    page_id
                })
                .map(|page_id| Fetch {
                    page_id,
                    probe: None,
                    buf: Some(Box::new(Page::zeroed())),
                })
                .collect::<Vec<_>>();

            WorkItem {
                pages,
                requested: 0,
                received: 0,
            }
        })
        .collect::<Vec<_>>()
}

pub(crate) fn run_worker(
    worker_index: usize,
    params: Params,
    map: Map,
    store: Arc<Store>,
    meta_map: Arc<RwLock<MetaMap>>,
    changed_page_tx: Sender<ChangedPage>,
    start_work_rx: Receiver<Arc<Barrier>>,
) {
    let mut workload = make_workload(worker_index, &params);
    loop {
        let barrier = match start_work_rx.recv() {
            Err(_) => break,
            Ok(b) => b,
        };

        // read
        read_phase(
            worker_index,
            &params,
            &map,
            &store,
            &meta_map,
            workload,
            &changed_page_tx,
        );

        let _ = barrier.wait();

        // make next workload while write phase is ongoing.
        workload = make_workload(worker_index, &params);
    }
}

fn read_phase(
    io_handle_index: usize,
    params: &Params,
    map: &Map,
    store: &Store,
    meta_map: &Arc<RwLock<MetaMap>>,
    mut workload: Vec<WorkItem>,
    changed_page_tx: &Sender<ChangedPage>,
) {
    let pack_user_data =
        |item_index: usize, page_index: usize| ((item_index as u64) << 32) + page_index as u64;

    let unpack_user_data = |user_data: u64| {
        (
            (user_data >> 32) as usize,
            (user_data & 0x00000000FFFFFFFF) as usize,
        )
    };

    let start = std::time::Instant::now();

    let meta_map = meta_map.read().unwrap();
    let mut rng = rand::thread_rng();

    // for extra fetches + misprobes
    let mut extra_fetches: VecDeque<(BucketIndex, Box<Page>, u64)> = VecDeque::new();
    let mut preload_item = 0;
    let mut processed = 0;
    let mut seek_extra = 0;
    let mut io_completions = 0;
    'a: while processed < workload.len() {
        let process_seek_extra = seek_extra < workload.len() && {
            let seek_front = &workload[seek_extra];
            seek_front.received == seek_front.requested
                && seek_front.requested == params.preload_count
        };

        if process_seek_extra {
            let seek_front = &mut workload[seek_extra];
            if rng.gen::<f32>() < params.load_extra_rate {
                let extra_page_index = params.preload_count;
                let fetch = &mut seek_front.pages[extra_page_index];
                let probe = map.begin_probe(&fetch.page_id, &meta_map);
                seek_front.requested += 1;

                match map.search(&meta_map, probe) {
                    None => {
                        seek_front.received += 1;
                        fetch.probe = None;
                    }
                    Some(probe) => {
                        let buf = fetch.buf.take().unwrap();
                        fetch.probe = Some(probe);
                        let user_data = pack_user_data(seek_extra, extra_page_index);
                        extra_fetches.push_front((probe.bucket, buf, user_data));
                    }
                }
            }

            seek_extra += 1;
        }

        let process_front = {
            let front = &workload[processed];
            seek_extra > processed && front.received == front.requested
        };

        if process_front {
            let front = &mut workload[processed];
            processed += 1;

            for i in 0..front.received {
                let fetch = &mut front.pages[i];
                let mut buf = fetch.buf.take().unwrap();
                let (diff, new_entries) = update_page(&mut buf, fetch.page_id, params);
                let _ = changed_page_tx.send(ChangedPage {
                    page_id: fetch.page_id,
                    bucket: fetch.probe.map(|p| p.bucket),
                    buf,
                    diff,
                    new_entries,
                });
            }
        }

        if let Ok(complete_io) = map.io_receiver.try_recv() {
            let CompleteIo { command, result } = complete_io;
            assert!(result.is_ok());

            let (item_index, page_index) = unpack_user_data(command.user_data);
            let cur_complete = &mut workload[item_index];
            let fetch = &mut cur_complete.pages[page_index];

            io_completions += 1;

            let mut buf = command.kind.unwrap_buf();
            if !page_id_matches(&*buf, &fetch.page_id) {
                // misprobe.

                // unwrap: probe is always set.
                let probe = fetch.probe.unwrap();
                match map.search(&meta_map, probe) {
                    None => {
                        *buf = Page::zeroed();
                        cur_complete.received += 1;
                        fetch.buf = Some(buf);
                        fetch.probe = None;
                    }
                    Some(probe) => {
                        fetch.probe = Some(probe);
                        extra_fetches.push_back((probe.bucket, buf, command.user_data));
                    }
                }
            } else {
                fetch.buf = Some(buf);
                cur_complete.received += 1;
            }
        }

        while let Some((bucket, buf, user_data)) = extra_fetches.pop_front() {
            let command = IoCommand {
                kind: IoKind::Read(store.store_fd(), store.data_page_index(bucket), buf),
                handle: io_handle_index,
                user_data,
            };
            match map.io_sender.try_send(command) {
                Ok(()) => {}
                Err(TrySendError::Disconnected(_)) => panic!(),
                Err(TrySendError::Full(command)) => {
                    let buf = command.kind.unwrap_buf();
                    extra_fetches.push_front((bucket, buf, user_data));
                    continue 'a;
                }
            }
        }

        while preload_item < workload.len() {
            let cur_preload = &mut workload[preload_item];
            if cur_preload.requested >= params.preload_count {
                preload_item += 1;
                continue;
            } else {
                let fetch = &mut cur_preload.pages[cur_preload.requested];
                let probe = map.begin_probe(&fetch.page_id, &meta_map);
                match map.search(&meta_map, probe) {
                    None => {
                        cur_preload.requested += 1;
                        cur_preload.received += 1;
                    }
                    Some(probe) => {
                        let buf = fetch.buf.take().unwrap();
                        let user_data = pack_user_data(preload_item, cur_preload.requested);
                        let command = IoCommand {
                            kind: IoKind::Read(
                                store.store_fd(),
                                store.data_page_index(probe.bucket),
                                buf,
                            ),
                            handle: io_handle_index,
                            user_data,
                        };
                        fetch.probe = Some(probe);

                        cur_preload.requested += 1;
                        match map.io_sender.try_send(command) {
                            Ok(()) => {}
                            Err(TrySendError::Disconnected(_)) => panic!(),
                            Err(TrySendError::Full(command)) => {
                                let buf = command.kind.unwrap_buf();
                                extra_fetches.push_back((probe.bucket, buf, user_data));
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    println!(
        "finished read phase (w#{io_handle_index}) in {}ms, {}ios, {} IOPS",
        start.elapsed().as_millis(),
        io_completions,
        1000.0 * io_completions as f64 / (start.elapsed().as_millis() as f64)
    );
}

fn update_page(page: &mut Page, page_id: PageId, params: &Params) -> (PageDiff, Vec<[u8; 32]>) {
    let mut diff = [0u8; 16];
    page[..page_id.len()].copy_from_slice(page_id.as_slice());

    let mut rng = rand::thread_rng();
    let mut updates = Vec::new();

    for i in 0..SLOTS_PER_PAGE {
        if rng.gen::<f32>() > params.page_item_update_rate {
            continue;
        }
        diff.view_bits_mut::<Msb0>().set(i, true);
        rng.fill(&mut page[slot_range(i)]);
        let mut buf = [0; 32];
        buf.copy_from_slice(&page[slot_range(i)]);
        updates.push(buf);
    }

    (diff, updates)
}

fn page_id_matches(page: &Page, expected_page_id: &PageId) -> bool {
    &page[..expected_page_id.len()] == expected_page_id.as_slice()
}
