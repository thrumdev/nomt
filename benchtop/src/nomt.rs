use crate::{backend::Transaction, timer::Timer, workload::Workload};
use fxhash::FxHashMap;
use nomt::{
    hasher::{NodeHasher, ValueHasher, Blake3Hasher}, trie::{KeyPath, NodeKind, Node, LeafData, InternalData}, KeyReadWrite, Nomt, Options, Overlay, Session,
    SessionParams, WitnessMode,
};
use p3_field::{FieldAlgebra, PrimeField32};
use p3_koala_bear::KoalaBear;
use p3_symmetric::PseudoCompressionFunction;
use rand::SeedableRng;
use sha2::Digest;
use std::{
    collections::{hash_map::Entry, VecDeque},
    sync::Mutex,
};

const NOMT_DB_FOLDER: &str = "nomt_db";

pub struct NomtDB {
    nomt: Nomt<Poseidon2KoalaBear>,
    overlay_window_capacity: usize,
    overlay_window: Mutex<VecDeque<Overlay>>,
}

impl NomtDB {
    pub fn open(
        reset: bool,
        commit_concurrency: usize,
        io_workers: usize,
        hashtable_buckets: Option<u32>,
        page_cache_size: Option<usize>,
        leaf_cache_size: Option<usize>,
        page_cache_upper_levels: usize,
        prepopulate_page_cache: bool,
        overlay_window_capacity: usize,
    ) -> Self {
        let nomt_db_folder =
            std::env::var("NOMT_DB_FOLDER").unwrap_or_else(|_| NOMT_DB_FOLDER.to_string());

        if reset {
            // Delete previously existing db
            let _ = std::fs::remove_dir_all(&nomt_db_folder);
        }

        let mut opts = Options::new();
        opts.path(nomt_db_folder);
        opts.commit_concurrency(commit_concurrency);
        opts.io_workers(io_workers);
        opts.metrics(true);
        if let Some(size) = page_cache_size {
            opts.page_cache_size(size);
        }
        if let Some(size) = leaf_cache_size {
            opts.leaf_cache_size(size);
        }
        if let Some(buckets) = hashtable_buckets {
            opts.hashtable_buckets(buckets);
        }
        opts.page_cache_upper_levels(page_cache_upper_levels);
        opts.prepopulate_page_cache(prepopulate_page_cache);

        let nomt = Nomt::open(opts).unwrap();
        Self {
            nomt,
            overlay_window_capacity,
            overlay_window: Mutex::new(VecDeque::new()),
        }
    }

    fn commit_overlay(
        &self,
        overlay_window: &mut VecDeque<Overlay>,
        mut timer: Option<&mut Timer>,
    ) {
        if self.overlay_window_capacity == 0 {
            return;
        }

        if overlay_window.len() == self.overlay_window_capacity {
            let _ = timer.as_mut().map(|t| t.record_span("commit_overlay"));
            let overlay = overlay_window.pop_back().unwrap();
            overlay.commit(&self.nomt).unwrap();
        }
    }

    pub fn execute(&self, mut timer: Option<&mut Timer>, workload: &mut dyn Workload) {
        let mut overlay_window = self.overlay_window.lock().unwrap();
        if overlay_window.len() < self.overlay_window_capacity {
            timer = None;
        }
        let _timer_guard_total = timer.as_mut().map(|t| t.record_span("workload"));

        self.commit_overlay(&mut overlay_window, timer.as_mut().map(|t| &mut **t));

        let session_params = SessionParams::default().witness_mode(WitnessMode::read_write());

        let session_params = if self.overlay_window_capacity == 0 {
            session_params
        } else {
            session_params.overlay(overlay_window.iter()).unwrap()
        };
        let session = self.nomt.begin_session(session_params);

        let mut transaction = Tx {
            session: &session,
            access: FxHashMap::default(),
            timer,
        };

        workload.run_step(&mut transaction);

        let Tx {
            access, mut timer, ..
        } = transaction;

        let _timer_guard_commit = timer.as_mut().map(|t| t.record_span("commit_and_prove"));
        let mut actual_access: Vec<_> = access.into_iter().collect();
        actual_access.sort_by_key(|(k, _)| *k);

        let finished = session.finish(actual_access).unwrap();
        if self.overlay_window_capacity == 0 {
            finished.commit(&self.nomt).unwrap();
        } else {
            let new_overlay = finished.into_overlay();
            overlay_window.push_front(new_overlay);
        }
    }

    // note: this is only intended to be used with workloads which are disjoint, i.e. no workload
    // writes a key which another workload reads. re-implementing BlockSTM or other OCC methods are
    // beyond the scope of benchtop.
    pub fn parallel_execute(
        &self,
        mut timer: Option<&mut Timer>,
        thread_pool: &rayon::ThreadPool,
        workloads: &mut [Box<dyn Workload>],
    ) {
        let mut overlay_window = self.overlay_window.lock().unwrap();
        if overlay_window.len() < self.overlay_window_capacity {
            timer = None;
        }

        let _timer_guard_total = timer.as_mut().map(|t| t.record_span("workload"));

        self.commit_overlay(&mut overlay_window, timer.as_mut().map(|t| &mut **t));

        let session_params = SessionParams::default().witness_mode(WitnessMode::read_write());

        let session_params = if self.overlay_window_capacity == 0 {
            session_params
        } else {
            session_params.overlay(overlay_window.iter()).unwrap()
        };
        let session = self.nomt.begin_session(session_params);
        let mut results: Vec<Option<_>> = (0..workloads.len()).map(|_| None).collect();

        let use_timer = timer.is_some();
        thread_pool.in_place_scope(|scope| {
            for (workload, result) in workloads.into_iter().zip(results.iter_mut()) {
                let session = &session;
                scope.spawn(move |_| {
                    let mut workload_timer = if use_timer {
                        Some(Timer::new(String::new()))
                    } else {
                        None
                    };
                    let mut transaction = Tx {
                        session,
                        access: FxHashMap::default(),
                        timer: workload_timer.as_mut(),
                    };
                    workload.run_step(&mut transaction);
                    *result = Some((transaction.access, workload_timer.map(|t| t.freeze())));
                })
            }
        });

        // absorb instrumented times from workload timers.
        for (_, ref mut workload_timer) in results.iter_mut().flatten() {
            if let (Some(ref mut t), Some(wt)) = (timer.as_mut(), workload_timer.take()) {
                t.add(wt);
            }
        }

        let _timer_guard_commit = timer.as_mut().map(|t| t.record_span("commit_and_prove"));
        let mut actual_access: Vec<_> = results
            .into_iter()
            .flatten()
            .map(|(access, _)| access)
            .flatten()
            .collect();
        actual_access.sort_by_key(|(k, _)| *k);

        let finished = session.finish(actual_access).unwrap();
        if self.overlay_window_capacity == 0 {
            finished.commit(&self.nomt).unwrap();
        } else {
            let new_overlay = finished.into_overlay();
            overlay_window.push_front(new_overlay);
        }
    }

    pub fn print_metrics(&self) {
        self.nomt.metrics().print();
        let ht_stats = self.nomt.hash_table_utilization();
        println!(
            "  buckets {}/{} ({})",
            ht_stats.occupied,
            ht_stats.capacity,
            ht_stats.occupancy_rate()
        );
    }
}

struct Tx<'a> {
    timer: Option<&'a mut Timer>,
    session: &'a Session<Poseidon2KoalaBear>,
    access: FxHashMap<KeyPath, KeyReadWrite>,
}

impl<'a> Transaction for Tx<'a> {
    fn read(&mut self, key: &[u8]) -> Option<Vec<u8>> {
        let key_path = koalabear_friendly_keypath(sha2::Sha256::digest(key).into());
        let _timer_guard_read = self.timer.as_mut().map(|t| t.record_span("read"));

        match self.access.entry(key_path) {
            Entry::Occupied(o) => o.get().last_value().map(|v| v.to_vec()),
            Entry::Vacant(v) => {
                let value = self.session.read(key_path).unwrap();
                self.session.warm_up(key_path);

                v.insert(KeyReadWrite::Read(value.clone()));
                value.map(|v| v.to_vec())
            }
        }
    }

    fn note_read(&mut self, key: &[u8], value: Option<Vec<u8>>) {
        let key_path = koalabear_friendly_keypath(sha2::Sha256::digest(key).into());

        match self.access.entry(key_path) {
            Entry::Occupied(mut o) => {
                o.get_mut().read(value);
            }
            Entry::Vacant(v) => {
                self.session.warm_up(key_path);
                v.insert(KeyReadWrite::Read(value));
            }
        }
    }

    fn write(&mut self, key: &[u8], value: Option<&[u8]>) {
        let key_path = koalabear_friendly_keypath(sha2::Sha256::digest(key).into());
        let value = value.map(|v| v.to_vec());

        match self.access.entry(key_path) {
            Entry::Occupied(mut o) => {
                o.get_mut().write(value);
            }
            Entry::Vacant(v) => {
                v.insert(KeyReadWrite::Write(value));
            }
        }

        self.session.warm_up(key_path);
        self.session.preserve_prior_value(key_path);
    }
}

/// The KoalaBear prime: 2^31 - 2^24 + 1 (source: plonky3)
const KOALABEAR_PRIME: u32 = KoalaBear::ORDER_U32;

// we truncate key-paths to 240 bits and treat a key-path as 8 packed 30-bit strings. which always
// fit into 8 koalabear field elements without losing precision.
fn koalabear_friendly_keypath(mut key_path: KeyPath) -> KeyPath {
    key_path[30] = 0;
    key_path[31] = 0;
    key_path
}

// given a 32-byte array, transform it such that every 32-bit word within it (little-endian) is
// taken modulo the koalabear prime.
fn koalabear_friendly_hash(mut arr: [u8; 32]) -> [u8; 32] {
    for i in 0..8 {
        let start = i * 4;
        let end = start + 4;
        let word = u32::from_le_bytes(arr[start..end].try_into().unwrap()) % KOALABEAR_PRIME;
        arr[start..end].copy_from_slice(&word.to_le_bytes());
    }

    arr
}

fn pack_koalabear_hash(hash: [KoalaBear; 8]) -> [u8; 32] {
    let mut bytes = [0u8; 32];
    for (i, elem) in hash.into_iter().enumerate() {
        let start = i * 4;
        let end = start + 4;
        let word = elem.as_canonical_u32();
        bytes[start..end].copy_from_slice(&u32::to_le_bytes(word));
    }
    bytes
}

// parse a byte-string as 8 koalabear field items. assume that all 4-bit words (little-endian) are
// within range.
fn unpack_koalabear_hash(bytes: &[u8]) -> [KoalaBear; 8] {
    let mut field_elems = [KoalaBear::new(0); 8];
    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
        field_elems[i] = unpack_le_koalabear(chunk.try_into().unwrap());
    }

    field_elems
}

// unpack a little-endian KoalaBear field element from 4 bytes.
fn unpack_le_koalabear(bytes: [u8; 4]) -> KoalaBear {
    KoalaBear::from_canonical_u32(u32::from_le_bytes(bytes))
}

// parse a keypath (32 bytes with only 30 occupied) into 8 u32s containing the 30-bit (big-endian) expansion of each.
fn unpack_koalabear_keypath(bytes: &[u8]) -> [KoalaBear; 8] {
    assert_eq!(bytes.len(), 32);

    let mut field_elems = [KoalaBear::new(0); 8];

    let mut word = 0u64;
    let mut leftover_bits = 0;

    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
        word <<= 32;
        word += u32::from_be_bytes(chunk.try_into().unwrap()) as u64;

        leftover_bits += 2;

        // the high 30 bits of 'word' are the field element.
        field_elems[i] = KoalaBear::from_canonical_u32((word >> leftover_bits) as u32);
        // mask out so only the leftover bits remain
        word &= (1 << leftover_bits) - 1
    }

    field_elems
}

// The seed for a ChaCha20 CSPRNG used to generate round constants for the Poseidon2 hash function.
// Note that these constants need to be kept the same each time.
//
// The official Poseidon2 generator script, linked in the paper, is here:
// https://extgit.isec.tugraz.at/krypto/hadeshash/-/blob/master/code/generate_params_poseidon.sage
//
// Note that it randomly generates these constants (with a no-good awful CSPRNG that infinite loops
// on my machine). But my main takeaway here is that it's fine to generate these randomly and they
// don't need to be specially chosen.
//
// Plonky3 gives you the option to generate them randomly from any RNG, so we use ChaCha20 because
// it's pretty good.
//
// I got these from /dev/random.
const POSEIDON2_CONSTANT_SEED: [u8; 32] = [
    181, 70, 200, 56, 24, 112, 39, 225, 208, 76, 36, 0, 67, 63, 174, 118, 109, 44, 40, 227, 250,
    226, 108, 70, 237, 113, 109, 73, 54, 51, 74, 169,
];

lazy_static::lazy_static! {
    // 16-field-element permutation wrapped in a truncation.
    // this takes only the first 8 elements of the poseidon2 hash.
    static ref POSEIDON_HASHER: p3_symmetric::TruncatedPermutation<p3_koala_bear::Poseidon2KoalaBear<16>, 2, 8, 16> = {
        let mut rng = rand_chacha::ChaCha20Rng::from_seed(POSEIDON2_CONSTANT_SEED);
        let poseidon_perm = p3_koala_bear::Poseidon2KoalaBear::new_from_rng_128(&mut rng);

        // the Poseidon2 paper (section 3.1) indicates that compression by truncation is valid
        p3_symmetric::TruncatedPermutation::new(poseidon_perm)
    };
}

// based on https://github.com/Plonky3/Plonky3/blob/48f785cc32fbc9e5134082a01a1a73cadbe9ace7/examples/src/types.rs#L43
struct Poseidon2KoalaBear;

impl ValueHasher for Poseidon2KoalaBear {
    fn hash_value(data: &[u8]) -> [u8; 32] {
        // hash values with blake3 and then take each 32-bit word modulo the koalabear prime.
        koalabear_friendly_hash(<Blake3Hasher as ValueHasher>::hash_value(data))
    }
}

// we hash the nodes and then label them as leaf/internal based on whether they are quadratic
// residues. see comment on LegendreSymbol.
impl NodeHasher for Poseidon2KoalaBear {
    fn hash_leaf(leaf: &LeafData) -> [u8; 32] {
        let keypath_elems = unpack_koalabear_keypath(&leaf.key_path);
        let value_elems = unpack_koalabear_hash(&leaf.value_hash);
        let mut hash = POSEIDON_HASHER.compress([keypath_elems, value_elems]);

        // label leaf nodes by ensuring the first element of the hash is a quadratic residue
        hash[0] = make_residue(hash[0]);
        pack_koalabear_hash(hash)
    }

    fn hash_internal(internal: &InternalData) -> [u8; 32] {
        let left_elems = unpack_koalabear_hash(&internal.left);
        let right_elems = unpack_koalabear_hash(&internal.right);
        let mut hash = POSEIDON_HASHER.compress([left_elems, right_elems]);

        // label internal nodes by ensuring the first element of the hash is a quadratic nonresidue.
        hash[0] = make_nonresidue(hash[0]);
        pack_koalabear_hash(hash)
    }

    fn node_kind(node: &Node) -> NodeKind {
        let elem = unpack_le_koalabear(node[0..4].try_into().unwrap());
        match legendre_symbol(elem) {
            LegendreSymbol::Zero => NodeKind::Terminator,
            LegendreSymbol::Residue => NodeKind::Leaf,
            LegendreSymbol::NonResidue => NodeKind::Internal,
        }
    }
}

// note: "new" != "from_canonical_u32" - it puts them into montgomery form (x << 32 mod p).
//
// however, the residue property is preserved because the bitshift for montgomery form is 32 and
// 32 happens to be a residue in KoalaBear. if it were not a residue, we would just flip these.
const KNOWN_KOALABEAR_RESIDUE: KoalaBear = KoalaBear::new(1);
const KNOWN_KOALABEAR_NONRESIDUE: KoalaBear = KoalaBear::new(3);

// The Legendre Symbol of a field element.
//
// Quadratic residues are analogous to even/odd numbers in a prime field.
// other than 0, exactly half the elements of the field will be residues and half are non-residues
//
// under multiplication, they behave just like evens/odds as well.
// residue * residue = residue,
// non-residue * non-residue = residue,
// non-residue * residue = non-residue
enum LegendreSymbol {
    // The field element is zero.
    Zero,
    // The field element is a quadratic residue (it is the square of at least one other element)
    Residue,
    // The field element is not a quadratic residue.
    NonResidue,
}

fn legendre_symbol(x: KoalaBear) -> LegendreSymbol {
    let residue = x.exp_u64((KOALABEAR_PRIME as u64 - 1) / 2);

    if residue == KoalaBear::ZERO {
        assert_eq!(x, KoalaBear::ZERO);
        LegendreSymbol::Zero
    } else if residue == KoalaBear::ONE {
        LegendreSymbol::Residue
    } else {
        LegendreSymbol::NonResidue
    }
}

// Given a field element, force it to be a quadratic residue.
fn make_residue(x: KoalaBear) -> KoalaBear {
    match legendre_symbol(x) {
        LegendreSymbol::Zero => KNOWN_KOALABEAR_RESIDUE,
        LegendreSymbol::Residue => x,
        LegendreSymbol::NonResidue => x * KNOWN_KOALABEAR_NONRESIDUE,
    }
}

// Given a field element, force it to be a quadratic non-residue.
fn make_nonresidue(x: KoalaBear) -> KoalaBear {
    match legendre_symbol(x) {
        LegendreSymbol::Zero => KNOWN_KOALABEAR_NONRESIDUE,
        LegendreSymbol::Residue => x * KNOWN_KOALABEAR_NONRESIDUE,
        LegendreSymbol::NonResidue => x,
    }
}
