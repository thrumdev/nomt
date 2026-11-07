pub type TaskResult<R> = std::thread::Result<R>;

/// Spawn the given task within the given ThreadPool.
/// Use the provided Sender to send the result of the task execution.
///
/// The result will contain the effective result or the payload
/// of the panic that occurred.
pub fn spawn_task<F, R>(
    thread_pool: &threadpool::ThreadPool,
    task: F,
    tx: crossbeam_channel::Sender<TaskResult<R>>,
) where
    R: Send + 'static,
    F: FnOnce() -> R + Send + 'static,
{
    thread_pool.execute(move || {
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| task()));
        let _ = tx.send(res);
    });
}

/// Like [`spawn_task`], but on a new thread of its own. For tasks that live as long as something
/// the caller holds (e.g. a session), which would otherwise starve a fixed-size pool.
pub fn spawn_thread_task<F, R>(name: &str, task: F, tx: crossbeam_channel::Sender<TaskResult<R>>)
where
    R: Send + 'static,
    F: FnOnce() -> R + Send + 'static,
{
    #[cfg(shuttle)]
    use shuttle::thread::Builder;
    #[cfg(not(shuttle))]
    use std::thread::Builder;
    Builder::new()
        .name(name.to_string())
        .spawn(move || {
            let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| task()));
            let _ = tx.send(res);
        })
        .expect("failed to spawn thread");
}

/// Blocks waiting for completion of the task spawned with [`spawn_task`] or [`spawn_thread_task`].
/// It requires the receiver associated to the sender used to spawn the task.
///
/// Panics if the sender is dropped.
pub fn join_task<R>(receiver: &crossbeam_channel::Receiver<TaskResult<R>>) -> R
where
    R: Send + 'static,
{
    // UNWRAP: The sender is not expected to be dropped by the spawned task.
    let res = receiver.recv().unwrap();
    match res {
        Ok(res) => res,
        Err(err_payload) => std::panic::resume_unwind(err_payload),
    }
}
