use crate::message;
use tokio::sync::mpsc;

/// A notification sent by the agent.
pub struct Notif {}

/// A communication channel between the supervisor and the agent.
pub struct Comms {
    rx: mpsc::Receiver<()>,
}

impl Comms {
    pub fn new(_stream: tokio::net::UnixStream) -> Self {
        todo!()
    }
}

pub struct RequestResponse {}

impl RequestResponse {
    /// Sends a given request to the agent.
    ///
    /// Returns a response.
    pub async fn send_request(&self, _to_agent: message::ToAgent) -> message::ToSupervisor {
        todo!()
    }
}

pub async fn start_comms(_stream: tokio::net::UnixStream) -> RequestResponse {
    todo!()
}
