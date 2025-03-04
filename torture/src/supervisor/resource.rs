use std::{
    io::Read,
    os::unix::fs::MetadataExt,
    path::{Path, PathBuf},
};

use rand::{Rng, SeedableRng};

/// The maximum percentage of total disk space that torture will occupy.
const MAX_DISK_OCCUPANCY_RATIO: f64 = 0.6;

/// 250MiB is the minimum amount of disk space that can be assigned to a workload.
///
/// With a space utilization of 80% given to the hash table,
/// it is equivalent to 50_000 buckets.
const MIN_ASSIGNED_DISK_SPACE: u64 = 250 * (1 << 20);

/// The maximum percentage of total memory that torture will occupy.
const MAX_MEMORY_OCCUPANCY_RATIO: f64 = 0.6;

/// 100MiB is the minimum amount of memory that can be assigned to a workload.
const MIN_ASSIGNED_MEMORY: u64 = 250 * (1 << 20);

/// ResourceAllocator is used to split resources randomly across multiple workloads.
///
/// Resources are Memory and Disk space.
pub struct ResourceAllocator {
    rng: rand_pcg::Pcg64,
    // (workload_id, assigned_disk, assigned_mem)
    assigned: Vec<(u64, u64, u64)>,
    initially_avail_disk_space: u64,
    total_disk_space: u64,
    total_assigned_disk_space: u64,
}

impl ResourceAllocator {
    /// Creates a `ResourceAllocator` given the available disk space at the path of the workdir,
    /// where all workload data will be saved.
    pub fn new(workdir_path: PathBuf, seed: u64) -> Self {
        let (initially_avail_disk_space, total_disk_space) = disk_info(&workdir_path);
        Self {
            rng: rand_pcg::Pcg64::seed_from_u64(seed),
            assigned: vec![],
            initially_avail_disk_space,
            total_disk_space,
            total_assigned_disk_space: 0,
        }
    }

    /// It assigns portions of disk space and memory randomly to the specified `workload_id`.
    pub fn alloc(&mut self, workload_id: u64) -> Result<(), ResourceExhaustation> {
        // Assign disk space
        let avail_disk = self.initially_avail_disk_space - self.total_assigned_disk_space;
        let disk_avail_ratio = avail_disk as f64 / self.total_disk_space as f64;
        if disk_avail_ratio < (1. - MAX_DISK_OCCUPANCY_RATIO)
            || avail_disk <= MIN_ASSIGNED_DISK_SPACE
        {
            return Err(ResourceExhaustation::Disk);
        }

        let assigned_disk = self.rng.gen_range(MIN_ASSIGNED_DISK_SPACE..avail_disk);

        // Assign memory
        let (avail_mem, total_mem) = mem_info();
        let avil_mem_ratio = avail_mem as f64 / total_mem as f64;
        if avil_mem_ratio < (1. - MAX_MEMORY_OCCUPANCY_RATIO) || avail_mem <= MIN_ASSIGNED_MEMORY {
            return Err(ResourceExhaustation::Memory);
        }

        let assigned_disk_ratio = assigned_disk as f64 / avail_disk as f64;
        let assigned_mem = (avail_mem as f64 * assigned_disk_ratio) as u64;

        self.total_assigned_disk_space += assigned_disk;
        self.assigned
            .push((workload_id, assigned_disk, assigned_mem));

        Ok(())
    }

    pub fn alloc_memory(&mut self, workload_id: u64) -> Result<(), ResourceExhaustation> {
        let (avail_mem, total_mem) = mem_info();
        let avil_mem_ratio = avail_mem as f64 / total_mem as f64;
        if avil_mem_ratio < (1. - MAX_MEMORY_OCCUPANCY_RATIO) || avail_mem <= MIN_ASSIGNED_MEMORY {
            return Err(ResourceExhaustation::Memory);
        }

        let assigned_mem = self.rng.gen_range(MIN_ASSIGNED_MEMORY..avail_mem);

        self.assigned.push((workload_id, 0, assigned_mem));
        Ok(())
    }

    /// Remove the `workload_id` from tracked ones.
    ///
    /// Panics if the specified workload_id is not present in the tracked ones.
    pub fn free(&mut self, workload_id: u64) {
        let idx = self
            .assigned
            .iter()
            .position(|(id, ..)| *id == workload_id)
            .unwrap();
        let (_, assigned_disk, _) = self.assigned.remove(idx);
        self.total_assigned_disk_space -= assigned_disk;
    }

    /// Ensures that the workload_dir does not occupy more disk space than the assigned limit
    /// and the same applies to the memory utilization of the specified process_id.
    ///
    /// Panics if:
    /// 1. workload_id is not present in the tracked ones.
    /// 2. workload_dir_path does not exist.
    /// 3. process_id does not refer to an ongoing process.
    pub fn is_exceeding_resources(
        &self,
        workload_id: u64,
        workload_dir_path: &Path,
        process_id: u32,
    ) -> bool {
        let (_, assigned_disk, assigned_mem) = self
            .assigned
            .iter()
            .find(|(id, ..)| *id == workload_id)
            .unwrap();

        fn dir_size(path: &Path) -> u64 {
            std::fs::read_dir(path)
                .unwrap()
                .into_iter()
                .filter_map(|entry| entry.ok())
                .map(|entry| {
                    let entry_metadata = entry.metadata().unwrap();
                    if entry_metadata.is_dir() {
                        dir_size(&entry.path())
                    } else {
                        entry_metadata.size()
                    }
                })
                .sum()
        }

        let used_disk = dir_size(workload_dir_path);

        if used_disk >= *assigned_disk {
            return true;
        }

        let used_mem = process_memory_occupied(process_id);
        if used_mem >= *assigned_mem {
            return true;
        }

        false
    }

    /// Fetch the amount of assigned disk space and memory to the specified `workload_id`.
    ///
    /// Panics if the specified workload_id is not present in the tracked ones.
    pub fn assigned_resources(&mut self, workload_id: u64) -> (u64, u64) {
        self.assigned
            .iter()
            .find(|(id, ..)| *id == workload_id)
            .map(|(_id, assigned_disk, assigned_mem)| (*assigned_disk, *assigned_mem))
            .unwrap()
    }
}

/// During resource allocation the `ResourceAllocator` could return the following errors.
#[derive(Debug)]
pub enum ResourceExhaustation {
    /// `MAX_DISK_OCCUPANCY_RATIO` was reached or
    /// not enough disk space left to instantiate a new workload.
    Disk,
    /// `MAX_MEMORY_OCCUPANCY_RATIO` was reached or
    /// not enough memory left to instantiate a new workload.
    Memory,
}

/// Return the amount of physical memory utilized by the specified pid.
///
/// Panics if pid does not refer to an ongoing process.
pub fn process_memory_occupied(pid: u32) -> u64 {
    let file_path = format!("/proc/{}/statm", pid);
    let mut statm_file = std::fs::File::open(&file_path).expect(
        format!(
            "Getting process memory information failed, {} does not exist",
            file_path
        )
        .as_str(),
    );
    let format_err = format!("{} not formatted as expected", file_path);
    let mut statm = String::new();
    statm_file.read_to_string(&mut statm).expect(&format_err);
    statm
        .split(' ')
        .skip(1)
        .next()
        .map(|value| value.parse::<u64>().expect(&format_err))
        .expect(&format_err)
}

/// Returns the number of avaiable and total memory in bytes.
///
/// Panics if /proc/meminfo does not exist or is not formatted as usual.
fn mem_info() -> (u64, u64) {
    let mut meminfo_file =
        std::fs::File::open("/proc/meminfo").expect("Getting memory information failed");
    let mut meminfo = String::new();
    let format_err = "/proc/meminfo not formatted as expected";
    meminfo_file.read_to_string(&mut meminfo).expect(format_err);
    let find = |name| {
        meminfo
            .split('\n')
            .find(|entry| entry.starts_with(name))
            .and_then(|entry| entry.split(':').nth(1))
            .and_then(|value| value.trim_start().split(' ').nth(0))
            .and_then(|value| value.parse::<u64>().ok())
    };
    // /proc/meminfo shows values in KiB.
    let avail_mem = find("MemAvailable").expect(format_err) * 1024;
    let total_mem = find("MemTotal").expect(format_err) * 1024;
    (avail_mem, total_mem)
}

/// Returns the number of available and total bytes present in the disk where `path`
/// is located.
///
/// Panics if `libc::statvfs` fails.
fn disk_info(path: &Path) -> (u64, u64) {
    let path_cstr = std::ffi::CString::new(path.display().to_string()).unwrap();
    let mut statvfs: libc::statvfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::statvfs(path_cstr.as_ptr(), &mut statvfs as *mut libc::statvfs) } < 0 {
        panic!("Getting disk information failed");
    }

    let avail_disk = statvfs.f_bsize * statvfs.f_bavail;
    let total_disk = statvfs.f_bsize * statvfs.f_blocks;
    (avail_disk, total_disk)
}
