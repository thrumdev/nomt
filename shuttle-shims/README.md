# shuttle-shims

Shuttle-backed stand-ins for `crossbeam-channel`, `parking_lot` and `threadpool`. `run.sh` injects them into a NOMT build with `cargo --config 'patch.crates-io.<crate>.path=...'`, so NOMT runs under the [shuttle](https://github.com/awslabs/shuttle) scheduler without changing its dependency list. Only the API surface NOMT uses is implemented. The directory is excluded from the workspace.

Run the scenarios in `nomt/tests/concurrency_scenarios.rs` under shuttle:

```sh
shuttle-shims/run.sh                                  # all scenarios, PCT, 100 iterations each
SHUTTLE_ITERS=5000 shuttle-shims/run.sh nested_session_vs_commit
SHUTTLE_SCHED=random shuttle-shims/run.sh
SHUTTLE_CAPTURE_BACKTRACE=1 shuttle-shims/run.sh <name>   # per-task backtraces on deadlock
```

The build uses `--cfg shuttle --cfg nomt_sync_io`: `nomt_sync_io` swaps io_uring for the blocking pread/pwrite backend, whose completions shuttle can schedule. `Cargo.lock` is restored after the run, since the patches rewrite it.

The same scenarios run natively (`cargo test -p nomt --release --test concurrency_scenarios -- --ignored`) and under [FFS](https://github.com/aoli-al/ffs):

```sh
CS_NO_DEADLINE=1 ffs explore --iterations 600 --policy pct -- \
  <test-binary> --ignored --exact <name> --test-threads=1
```

FFS failures on NOMT have not re-executed reliably from its saved schedules; confirm them under shuttle.
