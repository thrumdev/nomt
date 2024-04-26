//! A data structure similar to a `RwLock`, but with a twist: it can only be accessed when the
//! correct pass is shown. This is useful when you want to obtain a read or write lock on some
//! subset of data, that is not necessarily owned by a single object.
//!
//! The usage pattern is as follows:
//!
//! 1. Create a [`RwPassDomain`] with [`RwPassDomain::new()`].
//! 2. Protect the data you want to access with a [`RwPassCell`] by calling
//!    [`RwPassDomain::protect()`].
//! 3. Obtain a read or write pass by calling [`RwPassDomain::new_read_pass()`] or
//!    [`RwPassDomain::new_write_pass()`].
//! 4. Use the pass to access the data within any of [`RwPassCell`]-s created within the domain
//!    using the [`RwPassCell::read()`] or [`RwPassCell::write()`] methods.
//!
//! # Example
//!
//! ```ignore
//! # use nomt::rw_pass_cell::{RwPassDomain, RwPassCell, ReadPass, WritePass};
//!
//! let domain = RwPassDomain::new();
//! let cell: RwPassCell<usize> = domain.protect(42);
//! let read_pass = domain.new_read_pass();
//! assert_eq!(cell.read(&read_pass).get(), 42);
//! ```

use std::{
    cell::UnsafeCell,
    marker::PhantomData,
    ops::{Deref, DerefMut},
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, Weak,
    },
};

use parking_lot::{RawRwLock, RwLock};

type RwLockReadGuard = parking_lot::lock_api::ArcRwLockReadGuard<RawRwLock, ()>;
type RwLockWriteGuard = parking_lot::lock_api::ArcRwLockWriteGuard<RawRwLock, ()>;
type Shared = RwLock<()>;

/// A domain that oversees [`RwPassCell`]s and provides read and write passes to access them.
#[derive(Clone)]
pub struct RwPassDomain {
    shared: Arc<Shared>,
}

impl RwPassDomain {
    /// Create a new [`RwPassDomain`]
    pub fn new() -> Self {
        Self {
            shared: Arc::new(RwLock::new(())),
        }
    }

    /// Protects the given inner value with a [`RwPassCell`].
    pub fn protect<T>(&self, inner: T) -> RwPassCell<T> {
        RwPassCell::new(Arc::downgrade(&self.shared), inner, ())
    }

    /// Protects the given inner value, along with an immutable identifier inside a [`RwPassCell`].
    ///
    /// This enables you to make use of [`RegionedWritePass`] to mutably access multiple cells
    /// simultaneously.
    #[allow(unused)] // TODO - parallel update will use this.
    pub fn protect_with_id<T, Id>(&self, inner: T, id: Id) -> RwPassCell<T, Id> {
        RwPassCell::new(Arc::downgrade(&self.shared), inner, id)
    }

    /// Creates a new read pass.
    ///
    /// The pass can be used to access the data within any [`RwPassCell`]s created within this
    /// domain.
    ///
    /// If there are any write passes active, this method will block until they are dropped.
    pub fn new_read_pass(&self) -> ReadPass {
        let guard = self.shared.read_arc();
        ReadPass {
            domain: self.shared.clone(),
            _guard: RwGuard::Read(guard),
        }
    }

    /// Creates a new write pass.
    ///
    /// The pass can be used to access the data within any [`RwPassCell`]s created within this
    /// domain.
    ///
    /// If there are any read or write passes active, this method will block until they are dropped.
    pub fn new_write_pass(&self) -> WritePass {
        let guard = self.shared.write_arc();
        WritePass {
            read_pass: ReadPass {
                domain: self.shared.clone(),
                _guard: RwGuard::Write(guard),
            },
        }
    }

    /// Create a new regioned write pass.
    ///
    /// The pass can be used to access the data within any [`RwPassCell`]s created within this
    /// domain.
    ///
    /// Furthermore, the pass can be split recursively into sub-regions, such that each pass
    /// can read data tagged with an ID that belongs to its respective sub-region.
    ///
    /// If there are any read or write passes active, this method will block until they are dropped.
    #[allow(unused)] // TODO - parallel update will use this.
    pub fn new_regioned_write_pass<R: Region>(&self) -> RegionedWritePass<R> {
        let guard = self.shared.write_arc();

        RegionedWritePass {
            parent: None,
            region: R::universe(),
            read_pass: Arc::new(ReadPass {
                domain: self.shared.clone(),
                _guard: RwGuard::Write(guard),
            }),
        }
    }
}

enum RwGuard {
    Read(#[allow(unused)] RwLockReadGuard),
    Write(#[allow(unused)] RwLockWriteGuard),
}

/// A token that allows read access to the data within one domain.
pub struct ReadPass {
    domain: Arc<Shared>,
    _guard: RwGuard,
}

/// A token that allows read-write access to the data within one domain.
pub struct WritePass {
    read_pass: ReadPass,
}

impl WritePass {
    /// Downgrades the write pass to a read pass.
    pub fn downgrade(&mut self) -> &ReadPass {
        // `mut` because otherwise, it will be possible to create an alias.
        &self.read_pass
    }
}

/// A cell corresponding with a [`RwPassDomain`]. This may be read and written only with a pass from
/// the domain.
///
/// The cell may also hold a unique ID which can be used with the [`RegionedWritePass`] to further
/// shard access.
pub struct RwPassCell<T, Id = ()> {
    // A weak reference to the creator of the cell.
    //
    // It's weak because we don't actually care about it's internal data, we just need to make
    // sure that the creator is alive so that the address is valid. Otherwise, the address could
    // be reused, and the provenance check would be invalid.
    provenance: Weak<Shared>,
    inner: UnsafeCell<T>,
    id: Id,
}

impl<T, Id> RwPassCell<T, Id> {
    fn new(provenance: Weak<Shared>, inner: T, id: Id) -> Self {
        Self {
            provenance,
            inner: UnsafeCell::new(inner),
            id,
        }
    }

    /// Returns a handle to read the value. Requires showing a read pass.
    ///
    /// Panics if the provided pass belongs to a different [`RwPassDomain`] than the cell.
    pub fn read<'a, 'pass>(&'a self, read_pass: &'pass ReadPass) -> ReadGuard<'a, 'pass, T> {
        self.check_domain(&read_pass.domain);
        ReadGuard {
            inner: &self.inner,
            _read_pass: PhantomData,
        }
    }

    /// Returns a handle to write the value. Requires showing a write pass.
    ///
    /// Panics if the provided pass belongs to a different [`RwPassDomain`] than the cell.
    pub fn write<'a, 'pass>(
        &'a self,
        write_pass: &'pass mut WritePass,
    ) -> WriteGuard<'a, 'pass, T> {
        self.check_domain(&write_pass.read_pass.domain);
        WriteGuard {
            inner: &self.inner,
            _write_pass: PhantomData,
        }
    }

    /// Returns a handle to write the value. Requires showing a regioned write pass.
    ///
    /// Panics if the provided pass belongs to a different [`RwPassDomain`] than the cell or
    /// the value's ID is not contained within the region.
    #[allow(unused)] // TODO: parallel update will use this
    pub fn write_regioned<'a, 'pass, R: Region<Id = Id> + Clone>(
        &'a self,
        regioned_write_pass: &'pass mut RegionedWritePass<R>,
    ) -> WriteGuard<'a, 'pass, T> {
        self.check_domain(&regioned_write_pass.read_pass.domain);
        assert!(regioned_write_pass.region().contains(&self.id));
        WriteGuard {
            inner: &self.inner,
            _write_pass: PhantomData,
        }
    }

    /// Prevents any funny business by ensuring the domain of the pass and the domain of the cell
    /// match.
    fn check_domain(&self, domain: &Arc<Shared>) {
        let domain_ptr = Arc::as_ptr(domain);
        let provenance_ptr = Weak::as_ptr(&self.provenance);
        assert!(std::ptr::eq(domain_ptr, provenance_ptr), "Domain mismatch");
    }
}

// SAFETY: The RwPassCell is Send and Sync if the inner value is Send and Sync. This is because the
//         undelying value is protected by the RwPassDomain, which ensures that the value is only
//         accessed when the correct pass is shown.
// SAFETY: The RwPasscell is Send if the inner value is Send and the Id is Sync. The underlying
//         value is protected by the RwPassDomain, but mutable exclusive access of the value
//         may occur on multiple threads. The Id is read-only.
unsafe impl<T: Send, Id: Sync> Send for RwPassCell<T, Id> {}
// SAFETY: The RwPasscell is Sync if both the inner value the Id are Sync. The underlying
//         They may both be read simultaneously from multiple threads if the appropriate passes
//         are shown.
unsafe impl<T: Sync, Id: Sync> Sync for RwPassCell<T, Id> {}

/// A read guard for the value of an [`RwPassCell`]. This may exist concurrently with other
/// readers.
// SAFETY: this cannot be `Clone`.
pub struct ReadGuard<'a, 'pass, T> {
    inner: &'a UnsafeCell<T>,
    _read_pass: PhantomData<&'pass ReadPass>,
}

impl<'a, 'pass, T> ReadGuard<'a, 'pass, T>
where
    'pass: 'a,
{
    /// Get a reference to the underlying value.
    pub fn get(&self) -> &'pass T {
        // SAFETY: The existence of the guard ensures that there are only shared references
        //         to the inner value. The returned reference cannot outlive the guard.
        unsafe { &*self.inner.get() }
    }
}

impl<'a, 'pass, T> Deref for ReadGuard<'a, 'pass, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.get()
    }
}

/// A read guard for the value of an [`RwPassCell`]. This may exist concurrently with other
/// readers.
pub struct WriteGuard<'a, 'pass: 'a, T> {
    inner: &'a UnsafeCell<T>,
    _write_pass: PhantomData<&'pass mut WritePass>,
}

impl<'a, 'pass, T> WriteGuard<'a, 'pass, T>
where
    'pass: 'a,
{
    /// Get a reference to the underlying value.
    pub fn get(&self) -> &T {
        // SAFETY: The existence of the guard ensures that there is only one mutable reference
        //         to the inner value. The returned reference cannot outlive the guard.
        unsafe { &*self.inner.get() }
    }

    /// Get a mutable reference to the underlying value.
    pub fn get_mut(&mut self) -> &mut T {
        // SAFETY: The existence of the mutable guard reference ensures that there is only one
        //         mutable reference to the inner value. The returned reference cannot outlive the
        //         guard.
        unsafe { &mut *self.inner.get() }
    }
}

impl<'a, 'pass, T> Deref for WriteGuard<'a, 'pass, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.get()
    }
}

impl<'a, 'pass, T> DerefMut for WriteGuard<'a, 'pass, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.get_mut()
    }
}

/// `Region`s indicate a range of data identifiers. This trait is unsafe because the memory safety
/// of [`RegionedWritePass`] depends on the results of its functions.
///
/// Safely implementing this trait means correctly implementing the functions and choosing an ID
/// type which cannot break the invariants of the trait.
pub unsafe trait Region {
    /// The ID type this region spans.
    type Id;

    /// Create a region which encompasses the entire universe of the type.
    ///
    /// # Safety
    ///
    /// This region must encompass all other regions, intersect all other regions, and contain
    /// all values.
    fn universe() -> Self;

    /// Whether the region contains a value.
    ///
    /// # Safety
    ///
    /// If this function is ever invoked with an ID, it must return the same result whenever it is
    /// called in the future.
    fn contains(&self, value: &Self::Id) -> bool;

    /// Whether the region completely encompasses another region.
    ///
    /// # Safety
    ///
    /// If this returns true, then it must be the case that this region contains every ID the other
    /// region does.
    fn encompasses(&self, other: &Self) -> bool;

    /// Whether the region has no overlaps with another region.
    ///
    /// # Safety
    ///
    /// If this returns true, then it must be the case that this region does not contains any ID
    /// the other region does, and vice versa.
    fn excludes(&self, other: &Self) -> bool;
}

struct ParentRegionedWritePass<R> {
    parent: Option<Arc<ParentRegionedWritePass<R>>>,
    remaining_children: AtomicUsize,
    region: R,
}

/// A wrapper around a [`RegionedWritePass`] which can be sent between threads.
///
/// The reason for this type is that `RegionedWritePass` is not safe to send between threads.
#[allow(unused)] // TODO - parallel update will use this.
pub struct RegionedWritePassEnvelope<R> {
    inner: RegionedWritePass<R>,
    sync_flag: Arc<AtomicBool>,
}

#[allow(unused)] // TODO - parallel update will use this.
impl<R> RegionedWritePassEnvelope<R> {
    /// Open the envelope, yielding a write pass to be used on another thread.
    pub fn into_inner(self) -> RegionedWritePass<R> {
        // SAFETY: acquire writes from the sending thread.
        //         while this is technically a spin loop, it should be rare for it to hang for long,
        //         since this flag is set before the envelope is even created. so spinning
        //         at all requires some substantial reordering + unfortunate context switching away
        //         from the sending thread.
        while !self.sync_flag.load(Ordering::Acquire) {
            std::hint::spin_loop();
        }
        self.inner
    }
}

/// SAFETY: this is safe to send because we deliberately read propagated writes from other threads
///         and all regioned write pass drops and envelope creations propagate those writes.
unsafe impl<R: Send> Send for RegionedWritePassEnvelope<R> {}

/// A [`RegionedWritePass`] allows for a write pass to be sharded across a dataset based on
/// by making use of non-overlapping regions.
///
/// Use [`RwPassDomain::new_regioned_write_pass`] to create this, and then `split` and `consume`
/// as needed.
#[allow(unused)] // TODO - parallel update will use this.
pub struct RegionedWritePass<R> {
    parent: Option<Arc<ParentRegionedWritePass<R>>>,
    region: R,
    read_pass: Arc<ReadPass>,
}

#[allow(unused)] // TODO - parallel update will use this.
impl<R: Region + Clone> RegionedWritePass<R> {
    /// Get the region associated with this write pass.
    pub fn region(&self) -> &R {
        &self.region
    }

    /// Wrap this in an envelope to be safely sent across threads.
    ///
    /// The [`RegionedWritePassEnvelope`] ensures that any writes to memory will be propagated
    /// to the remote thread before allowing the recipient to read or write.
    pub fn into_envelope(self) -> RegionedWritePassEnvelope<R> {
        // SAFETY: We release all writes from this thread before sending to another.
        //
        //         Arc here is probably overkill and boxing would be fine, but according to the
        //         rust docs: "A Rust atomic type that is exclusively owned or behind a mutable
        //         reference does not correspond to an “atomic object” in C++".
        //         (ref: https://doc.rust-lang.org/stable/std/sync/atomic/index.html)
        //
        //         Atomic variables that aren't at a stable reference and furthermore which aren't
        //         behind a _shared pointer_ are not required to emit atomic instructions. So we
        //         have to allocate and choose err on the side of caution.
        let sync_flag = Arc::new(AtomicBool::new(false));
        sync_flag.store(true, Ordering::Release);

        RegionedWritePassEnvelope {
            inner: self,
            sync_flag,
        }
    }

    /// Split this write pass into two parts encompassing non-overlapping sub-regions.
    ///
    /// The result will be a `(left, right)` tuple corresponding to the input argument regions.
    ///
    /// # Panics
    ///
    /// Panics if the regions overlap with each other or are not encompassed by the region of this
    /// pass.
    pub fn split(mut self, left: R, right: R) -> (Self, Self) {
        assert!(self.region.encompasses(&left));
        assert!(self.region.encompasses(&right));
        assert!(left.excludes(&right), "left and right regions overlap");

        let new_parent = Arc::new(ParentRegionedWritePass {
            parent: self.parent.take(),
            region: self.region.clone(),
            remaining_children: AtomicUsize::new(2),
        });

        let left_pass = RegionedWritePass {
            parent: Some(new_parent.clone()),
            region: left,
            read_pass: self.read_pass.clone(),
        };

        let right_pass = RegionedWritePass {
            parent: Some(new_parent),
            region: right,
            read_pass: self.read_pass.clone(),
        };

        (left_pass, right_pass)
    }

    /// Consume this regioned write-pass, possibly yielding the parent region back.
    ///
    /// If all other split descendents of the parent write pass have been dropped or consumed,
    /// this will return a region equivalent to the parent's.
    pub fn consume(self) -> Option<Self> {
        let Some(ref parent) = self.parent else {
            return None;
        };

        // SAFETY: release our writes to other threads if _not_ the last sibling.
        //         acquire writes from other threads if the last sibling.
        //         the value of '1' must have been written by another thread with `Release`.
        if parent.remaining_children.fetch_sub(1, Ordering::AcqRel) == 1 {
            Some(RegionedWritePass {
                parent: parent.parent.clone(),
                region: parent.region.clone(),
                read_pass: self.read_pass.clone(),
            })
        } else {
            None
        }
    }
}

impl<R> Drop for RegionedWritePass<R> {
    fn drop(&mut self) {
        if let Some(ref parent) = self.parent {
            // SAFETY: release our writes to other threads.
            parent.remaining_children.fetch_sub(1, Ordering::Release);
        }
    }
}
