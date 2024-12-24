use std::{
    collections::HashMap,
    future::Future,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

use anyhow::bail;
use futures::SinkExt;
use tokio::{
    io::{BufReader, BufWriter},
    net::{
        unix::{OwnedReadHalf, OwnedWriteHalf},
        UnixStream,
    },
    sync::{oneshot, Mutex},
};
use tokio_stream::StreamExt;
use tokio_util::codec::{FramedRead, FramedWrite, LengthDelimitedCodec};

use crate::message::{Envelope, ToAgent, ToSupervisor};

type WrStream = FramedWrite<BufWriter<OwnedWriteHalf>, LengthDelimitedCodec>;
type RdStream = FramedRead<BufReader<OwnedReadHalf>, LengthDelimitedCodec>;

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
    wr_stream: Mutex<WrStream>,
    pending: Mutex<HashMap<u64, oneshot::Sender<ToSupervisor>>>,
}

impl RequestResponse {
    /// Sends a given request to the agent.
    ///
    /// Returns a response.
    pub async fn send_request(&self, message: ToAgent) -> anyhow::Result<ToSupervisor> {
        // TODO: return an error.
        let reqno = self.shared.reqno.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();

        let mut pending = self.shared.pending.lock().await;
        pending.insert(reqno, tx);
        drop(pending);

        let buf = bincode::serialize(&Envelope { reqno, message }).unwrap();
        let mut wr_stream = self.shared.wr_stream.lock().await;
        wr_stream.send(buf.into()).await.unwrap();
        drop(wr_stream);

        rx.await.unwrap();

        todo!()
    }
}

/// A task that handles inbound messages and dispatches them to the corresponding request listener.
///
/// Returns an error if the stream is closed, the request number is unknown, or the message is
/// malformed.
async fn handle_inbound(shared: Arc<Shared>, mut rd_stream: RdStream) -> anyhow::Result<()> {
    loop {
        let buf = match rd_stream.next().await {
            Some(result) => match result {
                Ok(buf) => buf,
                Err(err) => return Err(err.into()),
            },
            None => bail!("agent unixstream read half finished"),
        };
        let envelope = bincode::deserialize::<Envelope<ToSupervisor>>(&buf)?;
        let Envelope { reqno, message } = envelope;
        let mut pending = shared.pending.lock().await;
        if let Some(tx) = pending.remove(&reqno) {
            // We don't care if the receiver is gone.
            let _ = tx.send(message);
        } else {
            bail!("unknown reqno")
        }
    }
}

/// Runs the communication handling logic.
///
/// The returned future should be polled. If the future resolves to an error, the connection should
/// be closed.
pub fn run(stream: UnixStream) -> (RequestResponse, impl Future<Output = anyhow::Result<()>>) {
    let (rd, wr) = stream.into_split();

    let framed_writer = FramedWrite::new(BufWriter::new(wr), LengthDelimitedCodec::new());
    let framed_reader = FramedRead::new(BufReader::new(rd), LengthDelimitedCodec::new());

    let shared = Arc::new(Shared {
        reqno: AtomicU64::new(0),
        wr_stream: Mutex::new(framed_writer),
        pending: Mutex::new(HashMap::new()),
    });
    let rr = RequestResponse {
        shared: shared.clone(),
    };

    let inbound = handle_inbound(shared, framed_reader);

    (rr, inbound)
}
