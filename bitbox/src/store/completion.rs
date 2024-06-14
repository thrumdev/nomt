use std::{ops::Range, sync::Arc, time::Duration};

use crossbeam_channel::{after, Receiver, Sender};
use io_uring::IoUring;

use crate::node_pages_map::Page;

pub(crate) fn start_completion_handler(
    ring: Arc<IoUring>,
    _ring_capacity: u32,
    rx_wait_completion: Receiver<(Range<u64>, Option<Vec<Box<Page>>>, Sender<()>)>,
    rx_kill_completion_handler: Receiver<()>,
) {
    std::thread::spawn(move || {
        let mut complete = RangeVector::new();
        // batched of operations that waits for completition
        let mut pendings = Vec::<(Range<u64>, Option<Vec<Box<Page>>>, Sender<()>)>::new();

        // TODO: Waking up the completion thread every X microseconds may not be the best solution.
        // I see two possible alternative implementations:
        // 1. Use a timeout of 1/3 of the total time required to perform an RRT for all entries in the ring
        // 2. Do not use a timeout and rely solely on WriteHandle to wake up the completion thread,
        // utilizing `feature_nodrop` for the kernel to handle completion_queue overflow
        // assert!(ring.params().is_feature_nodrop());
        let timeout_duration = Duration::from_micros(100);

        loop {
            crossbeam_channel::select! {
                recv(rx_wait_completion) -> res => {
                    if let Ok((range, maybe_writes_data, tx)) = res {
                        pendings.push((range, maybe_writes_data, tx));
                    }
                },
                recv(rx_kill_completion_handler) -> _ => {
                    break;
                },
                recv(after(timeout_duration)) -> _ => (),
            }

            // SAFETY: This thread holds an Arc over IoUring and is the only one
            // that access the completion queue. Therefore, it is safe to access
            // it, ensuring that it will live as long as the IoUring object
            let mut completition_queue = unsafe { ring.completion_shared() };
            completition_queue.sync();

            for cqe in completition_queue {
                // TODO: Verify the result and consider notifying about any errors if necessary
                complete.insert(cqe.user_data());
            }

            pendings.retain(|(pending_range, _writes_data, pending_sx)| {
                if complete.contains(pending_range) {
                    // notify and remove from pendings
                    pending_sx
                        .send(())
                        .expect("Impossible notify finish writes");
                    return false;
                }
                true
            });
        }
    });
}

struct RangeVector {
    ranges: Vec<Range<u64>>,
}

enum RangeOperation {
    Insert(usize, Range<u64>),
    Update(usize, Range<u64>),
    Join(usize, Range<u64>),
    Append,
}

impl RangeVector {
    fn new() -> Self {
        Self { ranges: Vec::new() }
    }

    fn insert(&mut self, new_item: u64) {
        // index will point to the first range that is completely bigger than new_item
        let Some(index) = self.ranges.iter().position(|range| new_item <= range.start) else {
            // new_item needs to be append to the end, it could be part of the last range or a new range
            match self.ranges.last_mut() {
                Some(last_range) if last_range.end == new_item => {
                    last_range.end = new_item + 1;
                }
                _ => self.ranges.push(new_item..new_item + 1),
            }
            return;
        };

        // index == 0 is a special case where ranges cannot be indexed at index - 1
        if index == 0 {
            if self.ranges[0].start - 1 == new_item {
                self.ranges[0].start = new_item;
            } else {
                self.ranges.insert(0, new_item..new_item + 1);
            }
            return;
        }

        // return if the element is part of the previous range
        if self.ranges[index - 1].contains(&new_item) {
            return;
        }

        let follow_prev = self.ranges[index - 1].end == new_item;
        let anticipate_next = self.ranges[index].start - 1 == new_item;

        if follow_prev && anticipate_next {
            self.ranges[index - 1] = self.ranges[index - 1].start..self.ranges[index].end;
            self.ranges.remove(index);
        } else if follow_prev {
            self.ranges[index - 1].end = new_item + 1;
        } else if anticipate_next {
            self.ranges[index].start = new_item;
        } else {
            self.ranges.insert(index, new_item..new_item + 1);
        }
    }

    fn contains(&self, range_to_check: &Range<u64>) -> bool {
        if let Some(possible_range) = self
            .ranges
            .iter()
            .find(|range| range.start <= range_to_check.start)
        {
            if possible_range.end >= range_to_check.end {
                return true;
            }
        }
        false
    }
}

// TODO: test RangeVector
