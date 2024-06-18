use bitvec::prelude::*;
use crossbeam_channel::{Receiver, Sender, TrySendError};
use rand::Rng;

use std::collections::VecDeque;
use std::sync::{atomic::Ordering, Arc, Barrier, RwLock};

use crate::meta_map::MetaMap;
use crate::store::{
    io::{CompleteIo, IoCommand, IoKind, PageIndex},
    Page, PAGE_SIZE,
};

use super::{
    slot_range, BucketIndex, ChangedPage, Map, PageDiff, PageId, Params, ProbeSequence,
    SLOTS_PER_PAGE,
};

pub(crate) fn run_worker(
    worker_index: usize,
    params: Params,
    map: Map,
    meta_map: Arc<RwLock<MetaMap>>,
    changed_page_tx: Sender<ChangedPage>,
    start_work_rx: Receiver<Arc<Barrier>>,
) {
    loop {
        let barrier = match start_work_rx.recv() {
            Err(_) => break,
            Ok(b) => b,
        };

        // create workload - (preload_count + 1) * workload_size random "page IDs"
        let workload_pages = (0..params.workload_size)
            .flat_map(|_| {
                let mut rng = rand::thread_rng();

                // sample from typical range if not cold, out of range if cold
                let range = if rng.gen::<f32>() < params.cold_rate {
                    params.num_pages..usize::max_value()
                } else {
                    0..params.num_pages
                };

                // turn each sample into a random hash, set a unique byte per worker to discriminate.
                (0..params.preload_count + 1)
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
            })
            .collect::<Vec<_>>();

        // read
        read_phase(
            worker_index,
            &params,
            &map,
            &meta_map,
            workload_pages,
            &changed_page_tx,
        );

        let _ = barrier.wait();
    }
}

enum PageState {
    Unneeded,
    Pending {
        probe: ProbeSequence,
    },
    Submitted {
        probe: ProbeSequence,
    },
    Received {
        location: Option<BucketIndex>, // `None` if fresh
        page: Box<Page>,
    },
}

impl PageState {
    fn is_unneeded(&self) -> bool {
        match self {
            PageState::Unneeded => true,
            _ => false,
        }
    }
}

struct ReadJob {
    pages: Vec<(PageId, PageState)>,
    job: usize,
}

impl ReadJob {
    fn all_ready(&self) -> bool {
        self.pages.iter().all(|(_, state)| match state {
            PageState::Unneeded | PageState::Received { .. } => true,
            _ => false,
        })
    }

    fn waiting_on_io(&self) -> bool {
        self.pages.iter().any(|(_, state)| match state {
            PageState::Submitted { .. } => true,
            _ => false,
        })
    }

    fn next_pending(&mut self) -> Option<(usize, ProbeSequence)> {
        self.pages
            .iter()
            .enumerate()
            .filter_map(|(i, (_, state))| match state {
                PageState::Pending { probe } => Some((i, *probe)),
                _ => None,
            })
            .next()
    }

    fn state_mut(&mut self, index: usize) -> &mut PageState {
        &mut self.pages[index].1
    }

    fn page_id(&self, index: usize) -> PageId {
        self.pages[index].0
    }
}

fn read_phase(
    io_handle_index: usize,
    params: &Params,
    map: &Map,
    meta_map: &Arc<RwLock<MetaMap>>,
    workload: Vec<PageId>,
    changed_page_tx: &Sender<ChangedPage>,
) {
    const MAX_IN_FLIGHT: usize = 16;

    let meta_map = meta_map.read().unwrap();
    let mut rng = rand::thread_rng();
    let mut misprobes = 0;
    let mut page_queries = params.workload_size * params.preload_count;

    // contains jobs we are actively waiting on I/O for.
    let mut in_flight: VecDeque<ReadJob> = VecDeque::with_capacity(MAX_IN_FLIGHT);

    // we process our workload sequentially but look forward and attempt to keep the I/O saturated.
    // this index tracks the index of the job at the front of the in_flight queue, or a new unique
    // index if the queue is empty.
    let mut job_head = 0;
    loop {
        // handle complete I/O
        for complete_io in map.io_receiver.try_iter() {
            if handle_complete(complete_io, &mut in_flight, map, &*meta_map) {
                misprobes += 1;
            }
        }

        // submit requests from in-flight batches.
        let can_add_inflight = submit_pending(&mut in_flight, map, io_handle_index);

        // if submit queue appears to have space, add to our in-flight set.
        if can_add_inflight && in_flight.len() < MAX_IN_FLIGHT {
            let all_done = add_inflight(
                &mut in_flight,
                job_head,
                &params,
                map,
                &*meta_map,
                &workload,
            );
            if all_done {
                println!("finished querying {page_queries} pages with {misprobes} misprobes");
                break;
            }
        }

        // process ready batches from the front.
        while in_flight.front().map_or(false, |state| state.all_ready()) {
            let mut item = in_flight.pop_front().unwrap();

            // randomly force another round-trip sometimes.
            if item.pages.last().unwrap().1.is_unneeded()
                && rng.gen::<f32>() < params.load_extra_rate
            {
                page_queries += 1;
                let page_id = item.page_id(params.preload_count);

                // probe and set as pending.
                let probe = map.begin_probe(&page_id, &meta_map);
                *item.state_mut(params.preload_count) = match map.search(&meta_map, probe) {
                    None => PageState::Received {
                        location: None,
                        page: Box::new(Page::zeroed()),
                    },
                    Some(probe) => PageState::Pending { probe },
                };
                in_flight.push_front(item);
                break;
            }

            job_head += 1;

            // update fetched pages randomly, record changes.
            for (page_id, page_state) in item.pages {
                if let PageState::Unneeded = page_state {
                    continue;
                }
                let PageState::Received { location, mut page } = page_state else {
                    panic!("all pages must be received at this point")
                };
                let diff = update_page(&mut page, page_id, &params, location.is_none());
                let _ = changed_page_tx.send(ChangedPage {
                    page_id,
                    bucket: location,
                    buf: page,
                    diff,
                });
            }
        }
    }
}

fn handle_complete(
    complete_io: CompleteIo,
    in_flight: &mut VecDeque<ReadJob>,
    map: &Map,
    meta_map: &MetaMap,
) -> bool {
    let unpack_user_data = |user_data: u64| {
        (
            (user_data >> 32) as usize,
            (user_data & 0x00000000FFFFFFFF) as usize,
        )
    };

    complete_io.result.expect("I/O failed");
    let command = complete_io.command;

    let (job_idx, index_in_job) = unpack_user_data(command.user_data);
    let front_job = in_flight[0].job;
    let job = &mut in_flight[job_idx - front_job];
    let expected_id = job.page_id(index_in_job);

    let PageState::Submitted { probe } = job.state_mut(index_in_job) else {
        panic!("received I/O for unsubmitted request");
    };

    // check that page idx matches the fetched page.
    let page = command.kind.unwrap_buf();
    let mut misprobe = false;
    *job.state_mut(index_in_job) = if page_id_matches(&*page, &expected_id) {
        PageState::Received {
            location: Some(probe.bucket),
            page,
        }
    } else {
        // probe failure. continue searching.
        misprobe = true;
        match map.search(&meta_map, *probe) {
            None => PageState::Received {
                location: None,
                page: Box::new(Page::zeroed()),
            },
            Some(probe) => PageState::Pending { probe },
        }
    };

    misprobe
}

// returns true if it's likely we can push another job to `in_flight`.
fn submit_pending(in_flight: &mut VecDeque<ReadJob>, map: &Map, io_handle_index: usize) -> bool {
    let pack_user_data = |job: usize, index: usize| ((job as u64) << 32) + index as u64;

    let job_head = match in_flight.front() {
        None => return true,
        Some(j) => j.job,
    };

    let mut can_submit = true;

    'a: for batch in in_flight.iter_mut() {
        while let Some((i, probe)) = batch.next_pending() {
            let command = IoCommand {
                kind: IoKind::Read(PageIndex::Data(probe.bucket), Box::new(Page::zeroed())),
                handle: io_handle_index,
                user_data: pack_user_data(batch.job, i),
            };

            match map.io_sender.try_send(command) {
                Ok(()) => {
                    *batch.state_mut(i) = PageState::Submitted { probe };
                }
                Err(TrySendError::Full(_)) => {
                    can_submit = false;
                    break 'a;
                }
                Err(TrySendError::Disconnected(_)) => {
                    panic!("I/O worker dropped");
                }
            }
        }
    }

    can_submit
}

// returns true if all work is done.
fn add_inflight(
    in_flight: &mut VecDeque<ReadJob>,
    job_head: usize,
    params: &Params,
    map: &Map,
    meta_map: &MetaMap,
    workload: &[PageId],
) -> bool {
    let start = job_head * (params.preload_count + 1);
    let end = start + params.preload_count + 1;

    if end > workload.len() {
        return in_flight.is_empty();
    }

    let job_idx = job_head + in_flight.len();
    in_flight.push_back(ReadJob {
        pages: workload[start..end]
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, page_id)| {
                if i == params.preload_count {
                    return (page_id, PageState::Unneeded);
                }

                let probe = map.begin_probe(&page_id, &meta_map);
                let state = match map.search(&meta_map, probe) {
                    None => PageState::Received {
                        location: None,
                        page: Box::new(Page::zeroed()),
                    },
                    Some(probe) => PageState::Pending { probe },
                };

                (page_id, state)
            })
            .collect(),
        job: job_idx,
    });

    false
}

fn update_page(page: &mut Page, page_id: PageId, params: &Params, fresh: bool) -> PageDiff {
    let mut diff = [0u8; 16];
    if fresh {
        page[..page_id.len()].copy_from_slice(page_id.as_slice());
    }

    let mut rng = rand::thread_rng();

    for i in 0..SLOTS_PER_PAGE {
        if rng.gen::<f32>() > params.page_item_update_rate {
            continue;
        }
        diff.view_bits_mut::<Msb0>().set(i, true);
        rng.fill(&mut page[slot_range(i)]);
    }

    diff
}

fn page_id_matches(page: &Page, expected_page_id: &PageId) -> bool {
    &page[..expected_page_id.len()] == expected_page_id.as_slice()
}
