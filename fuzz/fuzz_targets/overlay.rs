#![no_main]

mod reference;

use std::collections::BTreeMap;

use arbitrary::{Result, Unstructured};
use libfuzzer_sys::fuzz_target;
use nomt::{
    hasher::{Blake3Hasher, ValueHasher},
    trie::{KeyPath, Node},
    InvalidAncestors, KeyReadWrite, Nomt, Options, Overlay, SessionParams, Value,
};

type Model = BTreeMap<KeyPath, Value>;

/// An overlay created during the run, and what it should contain.
struct Slot {
    /// `None` once the overlay is committed or dropped.
    overlay: Option<Overlay>,
    parent: Option<usize>,
    committed: bool,
    prev_root: Node,
    /// The full state as of this overlay.
    model: Model,
}

struct State {
    db: Nomt<Blake3Hasher>,
    committed: Model,
    /// The overlay committed most recently, unless a session was committed directly after it.
    last_committed_overlay: Option<usize>,
    slots: Vec<Slot>,
    keys: Vec<KeyPath>,
}

fuzz_target!(|data: &[u8]| {
    let tempdir = tempfile::tempdir().unwrap();
    let mut u = Unstructured::new(data);
    let _ = run(&mut u, &tempdir.path().join("db"));
});

fn run(u: &mut Unstructured, path: &std::path::Path) -> Result<()> {
    let mut options = Options::new();
    options.path(path);
    options.commit_concurrency(u.int_in_range(1..=4)?);
    options.preallocate_ht(false);
    let mut state = State {
        db: Nomt::open(options).unwrap(),
        committed: Model::new(),
        last_committed_overlay: None,
        slots: Vec::new(),
        keys: Vec::new(),
    };

    for _ in 0..u.int_in_range(0..=32)? {
        match u.int_in_range(0..=5)? {
            0..=2 => session(u, &mut state)?,
            3 | 4 => commit_overlay(u, &mut state)?,
            _ => {
                if let Some(i) = choose_live(u, &state)? {
                    state.slots[i].overlay = None;
                }
            }
        }

        assert_eq!(state.db.root().into_inner(), root_of(&state.committed));
        for key in &state.keys {
            assert_eq!(
                state.db.read(*key).unwrap().as_ref(),
                state.committed.get(key)
            );
        }
    }
    Ok(())
}

fn choose_live(u: &mut Unstructured, state: &State) -> Result<Option<usize>> {
    let live = (0..state.slots.len())
        .filter(|&i| state.slots[i].overlay.is_some())
        .collect::<Vec<_>>();
    if live.is_empty() {
        return Ok(None);
    }
    Ok(Some(live[u.choose_index(live.len())?]))
}

/// Run a session on top of the DB or a live overlay, then turn it into an overlay, commit it
/// directly, or drop it.
fn session(u: &mut Unstructured, state: &mut State) -> Result<()> {
    let base = if u.ratio(3, 4)? {
        choose_live(u, state)?
    } else {
        None
    };

    // The uncommitted ancestors, nearest first, as far as they are still alive.
    let mut chain = Vec::new();
    let mut expected_err = None;
    let mut next = base;
    while let Some(i) = next {
        let slot = &state.slots[i];
        if slot.committed {
            break;
        }
        if slot.overlay.is_none() {
            expected_err = Some(InvalidAncestors::Incomplete);
            break;
        }
        chain.push(i);
        next = slot.parent;
    }

    let mut ancestors = chain
        .iter()
        .map(|&i| state.slots[i].overlay.as_ref().unwrap())
        .collect::<Vec<_>>();
    match u.int_in_range(0..=7)? {
        0 if chain.len() >= 2 => {
            ancestors.pop();
            expected_err = Some(InvalidAncestors::Incomplete);
        }
        1 if chain.len() >= 2 => {
            let pos = u.int_in_range(1..=chain.len() - 1)?;
            let others = (0..state.slots.len())
                .filter(|&i| i != chain[pos] && state.slots[i].overlay.is_some())
                .collect::<Vec<_>>();
            if !others.is_empty() {
                let other = others[u.choose_index(others.len())?];
                ancestors[pos] = state.slots[other].overlay.as_ref().unwrap();
                expected_err = Some(InvalidAncestors::NotAncestor);
            }
        }
        _ => {}
    }

    let params = match (SessionParams::default().overlay(ancestors), expected_err) {
        (Err(err), Some(expected)) => {
            assert_eq!(err, expected);
            return Ok(());
        }
        (Ok(params), None) => params,
        (result, expected) => panic!("got {:?}, expected {:?}", result.err(), expected),
    };

    let db_root = state.db.root().into_inner();
    let base_model = base.map_or(&state.committed, |i| &state.slots[i].model);
    let base_root = base.map_or(db_root, |i| root_of(&state.slots[i].model));
    // Sessions see the overlays on top of the DB, so the DB must still be at the state the chain
    // was built on. nomt does not check this and panics while seeking if it is not.
    if chain
        .last()
        .is_some_and(|&i| state.slots[i].prev_root != db_root)
    {
        return Ok(());
    }

    let session = state.db.begin_session(params);
    assert_eq!(session.prev_root().into_inner(), base_root);

    let mut accesses = BTreeMap::<KeyPath, KeyReadWrite>::new();
    for _ in 0..u.int_in_range(0..=16)? {
        let key = if !state.keys.is_empty() && u.ratio(3, 4)? {
            state.keys[u.choose_index(state.keys.len())?]
        } else {
            let key = reference::new_key(u, &state.keys)?;
            state.keys.push(key);
            key
        };
        let access = accesses
            .entry(key)
            .or_insert(KeyReadWrite::Read(base_model.get(&key).cloned()));
        if u.arbitrary()? {
            access.write(new_value(u)?);
            session.warm_up(key);
        } else {
            assert_eq!(session.read(key).unwrap().as_ref(), base_model.get(&key));
        }
    }
    let mut model = base_model.clone();
    for (key, access) in &accesses {
        if let KeyReadWrite::Write(value) | KeyReadWrite::ReadThenWrite(_, value) = access {
            match value {
                Some(value) => model.insert(*key, value.clone()),
                None => model.remove(key),
            };
        }
    }

    let finished = session.finish(accesses.into_iter().collect()).unwrap();
    assert_eq!(finished.prev_root().into_inner(), base_root);
    assert_eq!(finished.root().into_inner(), root_of(&model));

    match u.int_in_range(0..=2)? {
        0 => {
            let overlay = finished.into_overlay();
            assert_eq!(overlay.root().into_inner(), root_of(&model));
            state.slots.push(Slot {
                overlay: Some(overlay),
                parent: base,
                committed: false,
                prev_root: base_root,
                model,
            });
        }
        1 => {
            let result = finished.commit(&state.db);
            // Equal roots mean the DB holds the same state the session was based on.
            if db_root == base_root {
                result.unwrap();
                state.committed = model;
                state.last_committed_overlay = None;
            } else {
                assert!(result.is_err());
            }
        }
        _ => {}
    }
    Ok(())
}

fn commit_overlay(u: &mut Unstructured, state: &mut State) -> Result<()> {
    let Some(i) = choose_live(u, state)? else {
        return Ok(());
    };
    let slot = &mut state.slots[i];
    let parent_ok = slot
        .parent
        .map_or(true, |p| state.last_committed_overlay == Some(p));
    let expect_ok = parent_ok && slot.prev_root == state.db.root().into_inner();

    // A commit attempt consumes the overlay, whether or not it succeeds.
    let overlay = slot.overlay.take().unwrap();
    let result = if u.arbitrary()? {
        overlay.commit(&state.db)
    } else {
        overlay
            .try_commit_nonblocking(&state.db)
            .map(|rest| assert!(rest.is_none(), "nothing else holds the DB"))
    };
    if expect_ok {
        result.unwrap();
        slot.committed = true;
        state.committed = slot.model.clone();
        state.last_committed_overlay = Some(i);
    } else {
        assert!(result.is_err());
    }
    Ok(())
}

fn new_value(u: &mut Unstructured) -> Result<Option<Value>> {
    if u.ratio(1, 4)? {
        return Ok(None);
    }
    // Mostly inline values, sometimes overflow values (MAX_LEAF_VALUE_SIZE = 1332).
    let len = if u.ratio(1, 8)? {
        u.int_in_range(1333..=4096)?
    } else {
        u.int_in_range(0..=64)?
    };
    let mut value = vec![0; len];
    u.fill_buffer(&mut value)?;
    Ok(Some(value))
}

fn root_of(model: &Model) -> Node {
    let leaves = model
        .iter()
        .map(|(k, v)| (*k, Blake3Hasher::hash_value(v)))
        .collect::<Vec<_>>();
    reference::node(&leaves, 0)
}
