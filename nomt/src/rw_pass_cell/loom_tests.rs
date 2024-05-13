#[test]
fn test_propagate_writes() {
    use crate::rw_pass_cell::*;

    loom::model(|| {
        let domain = RwPassDomain::new();
        let cell: Arc<RwPassCell<u8>> = Arc::new(domain.protect_with_id(0u8, ()));
        let (tx, rx) = loom::sync::mpsc::channel();

        let sender = loom::thread::spawn({
            let cell = Arc::clone(&cell);
            move || {
                let read_pass = domain.new_read_pass();
                assert_eq!(cell.read(&read_pass).with(|v| v.clone()), 0);
                drop(read_pass);

                let mut write_pass = domain.new_write_pass();
                let write_guard = cell.write(&mut write_pass);
                write_guard.with_mut(|v| *v = 10);

                let write_pass_envelope = write_pass.into_envelope();
                tx.send(write_pass_envelope).unwrap();
            }
        });

        let receiver = loom::thread::spawn({
            let cell = Arc::clone(&cell);
            move || {
                let write_pass_envelope = rx.recv().unwrap();
                let mut write_pass = write_pass_envelope.into_inner();
                let read_pass = write_pass.downgrade();
                assert_eq!(cell.read(&read_pass).with(|v| v.clone()), 10);
            }
        });

        sender.join().unwrap();
        receiver.join().unwrap();
    });
}

#[test]
fn test_consume_in_two_threads() {
    use crate::{page_region::PageRegion, rw_pass_cell::*};
    use nomt_core::page_id::{ChildPageIndex, ROOT_PAGE_ID};

    loom::model(|| {
        let domain = RwPassDomain::new();
        let _cell: RwPassCell<()> = domain.protect_with_id((), ());

        let write_pass = domain.new_write_pass();

        let region_a = PageRegion::from_page_id_descendants(
            ROOT_PAGE_ID.clone(),
            ChildPageIndex::new(0).unwrap(),
            ChildPageIndex::new(12).unwrap(),
        );

        let region_b = PageRegion::from_page_id_descendants(
            ROOT_PAGE_ID.clone(),
            ChildPageIndex::new(13).unwrap(),
            ChildPageIndex::new(63).unwrap(),
        );

        let write_pass = write_pass.with_region::<PageRegion>(PageRegion::universe());
        let mut write_passes = write_pass.split_n(vec![region_a, region_b]);
        let write_pass_b = write_passes.pop().unwrap();
        let write_pass_a = write_passes.pop().unwrap();

        let write_pass_a = write_pass_a.into_envelope();
        let write_pass_b = write_pass_b.into_envelope();

        let (tx, rx) = loom::sync::mpsc::channel::<Option<WritePass<PageRegion>>>();

        let _ = loom::thread::spawn({
            let tx = tx.clone();
            move || {
                let write_pass_a = write_pass_a.into_inner();
                let parent = write_pass_a.consume();
                tx.send(parent).unwrap();
            }
        });

        let _ = loom::thread::spawn({
            let tx = tx.clone();
            move || {
                let write_pass_b = write_pass_b.into_inner();
                let parent = write_pass_b.consume();
                tx.send(parent).unwrap();
            }
        });

        match (rx.recv().unwrap(), rx.recv().unwrap()) {
            (Some(wp), None) | (None, Some(wp)) => assert_eq!(*wp.region(), PageRegion::universe()),
            _ => panic!("Last WritePass to be consumed must return the parent"),
        }
    });
}
