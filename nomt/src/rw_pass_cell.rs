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
    ops::{Deref, DerefMut},
    sync::{Arc, Weak},
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
        RwPassCell::new(Arc::downgrade(&self.shared), inner)
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
}

enum RwGuard {
    Read(RwLockReadGuard),
    Write(RwLockWriteGuard),
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
pub struct RwPassCell<T> {
    // A weak reference to the creator of the cell.
    //
    // It's weak because we don't actually care about it's internal data, we just need to make
    // sure that the creator is alive so that the address is valid. Otherwise, the address could
    // be reused, and the provenance check would be invalid.
    provenance: Weak<Shared>,
    inner: UnsafeCell<T>,
}

impl<T> RwPassCell<T> {
    fn new(provenance: Weak<Shared>, inner: T) -> Self {
        Self {
            provenance,
            inner: UnsafeCell::new(inner),
        }
    }

    /// Returns a handle to read the value. Requires showing a read pass.
    ///
    /// Panics if the provided pass belongs to a different [`RwPassDomain`] than the cell.
    pub fn read<'a, 'pass>(&'a self, read_pass: &'pass ReadPass) -> ReadGuard<'a, 'pass, T> {
        self.check_domain(&read_pass.domain);
        ReadGuard {
            inner: self,
            _read_pass: read_pass,
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
            inner: self,
            _write_pass: write_pass,
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
unsafe impl<T: Send> Send for RwPassCell<T> {}
unsafe impl<T: Sync> Sync for RwPassCell<T> {}

/// A read guard for the value of an [`RwPassCell`]. This may exist concurrently with other
/// readers.
// SAFETY: this cannot be `Clone`.
pub struct ReadGuard<'a, 'pass, T> {
    inner: &'a RwPassCell<T>,
    _read_pass: &'pass ReadPass,
}

impl<'a, 'pass, T> ReadGuard<'a, 'pass, T>
where
    'pass: 'a,
{
    /// Get a reference to the underlying value.
    pub fn get(&self) -> &'pass T {
        // SAFETY: The existence of the guard ensures that there are only shared references
        //         to the inner value. The returned reference cannot outlive the guard.
        unsafe { &*self.inner.inner.get() }
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
    inner: &'a RwPassCell<T>,
    _write_pass: &'pass mut WritePass,
}

impl<'a, 'pass, T> WriteGuard<'a, 'pass, T>
where
    'pass: 'a,
{
    /// Get a reference to the underlying value.
    pub fn get(&self) -> &T {
        // SAFETY: The existence of the guard ensures that there is only one mutable reference
        //         to the inner value. The returned reference cannot outlive the guard.
        unsafe { &*self.inner.inner.get() }
    }

    /// Get a mutable reference to the underlying value.
    pub fn get_mut(&mut self) -> &mut T {
        // SAFETY: The existence of the mutable guard reference ensures that there is only one
        //         mutable reference to the inner value. The returned reference cannot outlive the
        //         guard.
        unsafe { &mut *self.inner.inner.get() }
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
