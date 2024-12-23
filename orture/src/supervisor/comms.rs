use std::{
    collections::{BTreeMap, HashMap},
    future::Future,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

use tokio::{
    io::{AsyncWrite, AsyncWriteExt, BufWriter},
    net::{unix::OwnedReadHalf, UnixStream},
    sync::{oneshot, Mutex},
};
use tokio_util::codec::{FramedRead, FramedWrite, LengthDelimitedCodec};

use crate::message::{Envelope, ToAgent, ToSupervisor};

/// A means to communicate with an agent.
///
/// This is a request-response client. Cheaply cloneable.
#[derive(Clone)]
pub struct RequestResponse {
    shared: Arc<Shared>,
}

struct Shared {
    /// The next request number to be used.
    reqno: AtomicU64,
    wr_stream: Mutex<Box<dyn AsyncWrite>>,
    pending: Mutex<HashMap<u64, oneshot::Sender<()>>>,
}

impl RequestResponse {
    /// Sends a given request to the agent.
    ///
    /// Returns a response.
    pub async fn send_request(&self, message: ToAgent) -> ToSupervisor {
        let reqno = self.shared.reqno.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();

        let mut pending = self.shared.pending.lock().await;
        pending.insert(reqno, tx);
        drop(pending);

        let buf = bincode::serialize(&Envelope { reqno, message }).unwrap();
        let wr_stream = &mut *self.shared.wr_stream.lock().await;
        wr_stream.write_all(&buf).await.unwrap();
        drop(wr_stream);

        rx.await.unwrap();

        todo!()
    }
}

async fn handle_inbound(shared: Arc<Shared>, rd_stream: OwnedReadHalf) -> Result<()> {
    loop {
        let buf = bincode::deserialize_from(&rd_stream)?;
        let Envelope { reqno, message } = buf;
        let mut pending = shared.pending.lock().await;
        if let Some(tx) = pending.remove(&reqno) {
            // We don't care if the receiver is gone.
            let _ = tx.send(());
        } else {
            // TODO: this should signal an issue.
        }
    }
    Ok(())
}

/// Runs the communication handling logic.
///
/// The returned future should be polled.
pub fn run(stream: UnixStream) -> (RequestResponse, impl Future<Output = ()>) {
    let (rd, wr) = stream.into_split();

    let buf_writer = BufWriter::new(wr);
    let framed_writer = FramedWrite::new(buf_writer, LengthDelimitedCodec::new());

    let shared = Arc::new(Shared {
        reqno: AtomicU64::new(0),
        wr_stream: Mutex::new(),
        pending: Mutex::new(HashMap::new()),
    });
    let rr = RequestResponse {
        shared: shared.clone(),
    };

    // let framed = FramedRead::new(LengthDelimitedCodec::new());

    todo!()
}
