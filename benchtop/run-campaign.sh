#!/usr/bin/env bash
# Fixed, reproducible transfer-throughput benchmark. Run from the repository root.
set -euo pipefail
export RUSTFLAGS=''
export CARGO_INCREMENTAL=0
export NOMT_BENCH_SEED=20261008
campaign_dir=/mnt/rollup/nomt-test/nomt-throughput
seed_dir="$campaign_dir/seed-v1"
export NOMT_DB_FOLDER="$campaign_dir/db"
benchmark=benchtop/target/release/benchtop
cargo build --release --locked --manifest-path benchtop/Cargo.toml
if [[ "${1:-}" == --init ]]; then
    if [[ -e "$seed_dir" ]]; then
        echo 'Refusing to replace an existing seed database.' >&2
        exit 1
    fi
    mkdir -p "$campaign_dir"
    NOMT_DB_FOLDER="$seed_dir" taskset -c 2-7,10-15 "$benchmark" init \
        -b nomt -w transfer -c 22 --buckets 4000000 --commit-concurrency 4 --io-workers 3
    touch "$seed_dir/.campaign-ready"
    exit 0
fi
if [[ ! -f "$seed_dir/.campaign-ready" ]]; then
    echo 'Run bash benchtop/run-campaign.sh --init once before benchmarking.' >&2
    exit 1
fi
# The only deleted path is this campaign's disposable working database.
rm -rf -- "$campaign_dir/db"
cp -a --reflink=auto --sparse=always "$seed_dir" "$campaign_dir/db"
sync -f "$campaign_dir/db"
taskset -c 2-7,10-15 "$benchmark" run \
    -b nomt -w transfer -c 22 -s 12000 \
    --workload-fresh 0 --distribution uniform \
    --workload-concurrency 6 --commit-concurrency 4 --io-workers 3 \
    --buckets 4000000 --overlay-window-length 0 \
    --page-cache-size 256 --leaf-cache-size 256 --page-cache-upper-levels 3 \
    --warm-up-ops 60000 --time-limit 2s
