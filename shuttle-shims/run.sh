#!/usr/bin/env bash
# Build and run NOMT's concurrency scenarios under shuttle.
#   shuttle-shims/run.sh [cargo test args...]
# e.g. SHUTTLE_ITERS=500 shuttle-shims/run.sh commit_vs_readers -- --nocapture
# Cargo.lock is restored afterwards: the patches rewrite it.
set -euo pipefail
SHIMS=$(cd "$(dirname "$0")" && pwd)
ROOT=$(dirname "$SHIMS")
cp "$ROOT/Cargo.lock" "$SHIMS/.Cargo.lock.bak"
trap 'mv "$SHIMS/.Cargo.lock.bak" "$ROOT/Cargo.lock"' EXIT
cd "$ROOT"
RUSTFLAGS="--cfg shuttle" \
CARGO_TARGET_DIR="$ROOT/target/shuttle" \
cargo \
  --config "patch.crates-io.crossbeam-channel.path=\"$SHIMS/crossbeam-channel\"" \
  --config "patch.crates-io.parking_lot.path=\"$SHIMS/parking_lot\"" \
  --config "patch.crates-io.threadpool.path=\"$SHIMS/threadpool\"" \
  --config "patch.crates-io.thread_local.path=\"$SHIMS/thread_local\"" \
  test -p nomt --release --test concurrency_scenarios "$@"
