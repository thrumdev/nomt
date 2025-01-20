// A wrapper needed to make it compile on non-Linux platforms.

cfg_if::cfg_if! {
    if #[cfg(target_os = "linux")] {
        pub use trickfs::TrickHandle;

        pub fn is_supported() -> bool {
            true
        }
    } else {
        pub struct TrickHandle;

        impl TrickHandle {
            pub fn set_trigger_enospc(&self, enabled: bool) {
                let _ = enabled;
                unimplemented!("TrickHandle::set_trigger_enospc");
            }

            pub fn unmount_and_join(self) {
            }
        }

        pub fn is_supported() -> bool {
            false
        }
    }
}
