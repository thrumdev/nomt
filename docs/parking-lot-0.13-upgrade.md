# Upgrading parking_lot to 0.13

parking_lot is pinned to 0.12 (`Cargo.toml`, `benchtop/Cargo.lock`) because 0.13 removed the per-call recursive read methods (`RwLock::read_recursive`, `RwLock::read_arc_recursive`) that `Nomt::read` and `SessionAccessGuard::acquire` in `nomt/src/lib.rs` rely on. 0.13 replaces them with `RecursiveRwLock`, which makes every reader skip a queued writer, not just the calls that need to.

## Why the per-call distinction matters

`access_lock` is writer-preferring: once a commit or rollback queues for the write side, new readers wait. A thread that already holds a live session must not wait, because the queued commit waits for that session (deadlock). Every other thread must wait, so a stream of overlapping sessions can't hold a commit off indefinitely. The tests `session_opens_while_commit_waits_for_another_session` and `session_on_another_thread_waits_for_queued_commit` in `nomt/src/lib.rs` (they spin on `access_lock.try_read()` until it fails) depend on this: with `RecursiveRwLock`, `try_read` keeps succeeding while a commit is queued and they loop forever.

## Options

1. Share the thread's existing guard instead of taking a second read lock (recommended). Keep plain `RwLock`. Store a `Weak<ArcRwLockReadGuard<..>>` in the thread-local instead of the session counter. In `acquire`, if the `Weak` upgrades, clone that `Arc`; otherwise take `read_arc` and store a new `Weak`. Each session holds an `Arc` to the guard, so the read lock stays held until the last session sharing it drops, and no thread ever reads past a queued writer. `Nomt::read` can do the same. Not yet checked: that `Arc<ArcRwLockReadGuard<..>>` is `Send` with the `send_guard` feature (it needs the guard to be `Sync` as well as `Send`).
2. Switch `access_lock` to `RecursiveRwLock`. Smallest diff, but commits and rollbacks can starve under continuous overlapping sessions, and the two queued-commit tests have to be rewritten or dropped.
3. Stay on 0.12. No code change. 0.12 is still maintained for now.

## Known gap in the current scheme (applies to options 1 and 3)

Both key the "this thread already holds a session" check on a thread-local. `Session` is `Send`, so a session can be created on thread A and used or held on thread B (including a tokio task holding a session across an `.await` and resuming on another worker). If B then opens another session while a commit is queued, B is not recognised as a holder, waits behind the commit, and the commit waits for B's session: deadlock. Same if A's session is moved away and A keeps the recursive pass (harmless: only extra reader bias).
