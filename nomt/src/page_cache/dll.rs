#![allow(unsafe_code)]

use std::{cell::UnsafeCell, ptr};

use super::cache_advisor::CacheAccess;

/// A simple doubly linked list for use in the `Lru`
#[derive(Debug)]
pub(crate) struct Node {
    pub inner: UnsafeCell<CacheAccess>,
    next: *mut Node,
    prev: *mut Node,
}

impl std::ops::Deref for Node {
    type Target = CacheAccess;

    fn deref(&self) -> &CacheAccess {
        unsafe { &(*self.inner.get()) }
    }
}

impl Node {
    fn unwire(&mut self) {
        unsafe {
            if !self.prev.is_null() {
                (*self.prev).next = self.next;
            }

            if !self.next.is_null() {
                (*self.next).prev = self.prev;
            }
        }

        self.next = ptr::null_mut();
        self.prev = ptr::null_mut();
    }
}

/// A simple non-cyclical doubly linked
/// list where items can be efficiently
/// removed from the middle, for the purposes
/// of backing an LRU cache.
pub struct DoublyLinkedList {
    head: *mut Node,
    tail: *mut Node,
    len: usize,
}

unsafe impl Send for DoublyLinkedList {}

impl Drop for DoublyLinkedList {
    fn drop(&mut self) {
        let mut cursor = self.head;
        while !cursor.is_null() {
            unsafe {
                let node = Box::from_raw(cursor);

                // don't need to check for cycles
                // because this Dll is non-cyclical
                cursor = node.prev;

                // this happens without the manual drop,
                // but we keep it for explicitness
                drop(node);
            }
        }
    }
}

impl Default for DoublyLinkedList {
    fn default() -> Self {
        Self {
            head: ptr::null_mut(),
            tail: ptr::null_mut(),
            len: 0,
        }
    }
}

impl DoublyLinkedList {
    pub(crate) const fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn push_head(&mut self, item: CacheAccess) -> *mut Node {
        self.len += 1;

        let node = Node {
            inner: UnsafeCell::new(item),
            next: ptr::null_mut(),
            prev: self.head,
        };

        let ptr = Box::into_raw(Box::new(node));

        self.push_head_ptr(ptr);

        ptr
    }

    fn push_head_ptr(&mut self, ptr: *mut Node) {
        if !self.head.is_null() {
            unsafe {
                (*self.head).next = ptr;
                (*ptr).prev = self.head;
            }
        }

        if self.tail.is_null() {
            self.tail = ptr;
        }

        self.head = ptr;
    }

    pub(crate) fn unwire(&mut self, ptr: *mut Node) {
        unsafe {
            if self.tail == ptr {
                self.tail = (*ptr).next;
            }

            if self.head == ptr {
                self.head = (*ptr).prev;
            }

            (*ptr).unwire();
        }

        self.len -= 1;
    }

    pub(crate) fn install(&mut self, ptr: *mut Node) {
        self.len += 1;
        self.push_head_ptr(ptr);
    }

    // NB: returns the Box<Node> instead of just the Option<CacheAccess>
    // because the LRU is a map to the Node as well, and if the LRU
    // accessed the map via PID, it would cause a use after free if
    // we had already freed the Node in this function.
    pub(crate) fn pop_tail(&mut self) -> Option<*mut Node> {
        if self.tail.is_null() {
            return None;
        }

        self.len -= 1;
        let tail_ptr = self.tail;
        if self.head == self.tail {
            self.head = ptr::null_mut();
        }

        unsafe {
            self.tail = (*tail_ptr).next;

            (*tail_ptr).unwire();
        }

        Some(tail_ptr)
    }
}
