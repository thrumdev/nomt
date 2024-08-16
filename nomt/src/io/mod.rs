#[cfg(not(target_os = "linux"))]
std::compile_error!("NOMT only supports Linux");

#[cfg(target_os = "linux")]
mod linux;

#[cfg(target_os = "linux")]
pub use linux::{start_io_worker, Page, HandleIndex, IoKind, IoCommand, CompleteIo, PAGE_SIZE};
