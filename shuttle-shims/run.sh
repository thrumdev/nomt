#!/usr/bin/env bash
# Build and run NOMT's concurrency scenarios under shuttle.
#   shuttle-shims/run.sh [cargo test args...]   (NOMT=/path/to/nomt to override the checkout)
# e.g. SHUTTLE_ITERS=500 ./run.sh commit_vs_readers -- --nocapture
# Cargo.lock is restored afterwards: the patches rewrite it.
set -euo pipefail
SHIMS=$(cd "$(dirname "$0")" && pwd)
NOMT=${NOMT:-$SHIMS/..}
cp "$NOMT/Cargo.lock" "$SHIMS/.Cargo.lock.bak"
trap 'cp "$SHIMS/.Cargo.lock.bak" "$NOMT/Cargo.lock"' EXIT
cd "$NOMT"
RUSTFLAGS="--cfg shuttle --cfg nomt_sync_io" \
CARGO_TARGET_DIR="$NOMT/target/shuttle" \
cargo \
  --config "patch.crates-io.crossbeam-channel.path=\"$SHIMS/crossbeam-channel\"" \
  --config "patch.crates-io.parking_lot.path=\"$SHIMS/parking_lot\"" \
  --config "patch.crates-io.threadpool.path=\"$SHIMS/threadpool\"" \
  test -p nomt --release --test concurrency_scenarios "$@"
