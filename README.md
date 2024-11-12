## NOMT: Nearly Optimal Merkle Trie

An implementation of a novel binary Merkle Trie and DB, written in Rust.

NOMT is an embedded key-value store that maintains a Merklized representation of key-value pairs with a simple key-value API, powering high throughput authenticated commits with billions of key-value pairs on relatively inexpensive hardware. It is largely designed for use in a blockchain setting as a drop-in replacement for RocksDB, MDBX, LevelDB, or ParityDB.

NOMT is optimized for fast random lookups of values, fast merkle tree updates, and fast writeout. It supports the generation of Merkle multiproofs for large batches of changes.

NOMT is designed to take advantage of hardware improvements in Solid State Drives (SSDs) using NVMe and Linux's io-uring API for asynchronous I/O. NOMT adequately supports generic Unix as well as macOS for daily development and testing, but primarily targets Linux for performance. The impressive trend in performance and capacity in modern SSDs enables us to build a DB that scales along with the hardware.

NOMT exposes a many-readers-one-writer API organized around batch transactions referred to as `Session`s. Predictable performance in a metered execution environment is a key goal of NOMT, and therefore only one `Session` may be live at a time.

## Architecture

Internally, NOMT consists of two parallel stores, Beatree and Bitbox. Beatree stores raw key-value pairs and is based around a B-Tree variant optimized for stable, fast random access patterns and high-entropy keys. Bitbox stores a custom sparse binary merkle tree in an on-disk hashtable in a format amenable to SSDs.

For more information on NOMT, the thesis behind it, and performance targets, see [this Novemeber 2024 presentation](https://x.com/TheKusamarian/status/1855477208762261910) by @rphmeier or [view the slides here](https://hackmd.io/@Xo-wxO7bQkKidH1LrqACsw/rkG0lmjWyg#/).

We have built a benchmarking tool, `benchtop`, which is located in the `benchtop` directory as a separate subcrate.

## Contributing

See [CONTRIBUTING.md](docs/CONTRIBUTING.md)

## Acknowledgements

The development of this project is supported financially by [Sovereign Labs](https://www.sovereign.xyz/), creators of the [Sovereign SDK](https://github.com/Sovereign-Labs/sovereign-sdk/). The idea for this project originated in [this post by Preston Evans](https://sovereign.mirror.xyz/jfx_cJ_15saejG9ZuQWjnGnG-NfahbazQH98i1J3NN8).
