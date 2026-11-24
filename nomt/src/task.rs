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
    std::thread::Builder::new()
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
    unwrap_task_result(receiver.recv().unwrap())
}

/// Propagate a completed task's panic after all required sibling tasks are joined.
pub(crate) fn unwrap_task_result<R>(result: TaskResult<R>) -> R {
    match result {
        Ok(res) => res,
        Err(err_payload) => std::panic::resume_unwind(err_payload),
    }
}

/// Join one fallible task, draining the remaining sibling results before
/// returning an error or resuming a panic. Successful results remain streaming.
pub(crate) fn join_task_result<R, E>(
    receiver: &crossbeam_channel::Receiver<TaskResult<Result<R, E>>>,
    remaining: usize,
) -> Result<R, E> {
    let result = receiver.recv().unwrap();
    if !matches!(&result, Ok(Ok(_))) {
        for _ in 0..remaining {
            // All siblings must stop producing I/O before the failure escapes.
            let _ = receiver.recv().unwrap();
        }
    }
    match result {
        Ok(result) => result,
        Err(payload) => std::panic::resume_unwind(payload),
    }
}
