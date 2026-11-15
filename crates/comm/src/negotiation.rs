use bincode::{Decode, Encode};
use serde::{Deserialize, Serialize};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpStream,
};

use crate::CommError;

/// Messages exchanged during the initial handshake to negotiate the transport.
#[derive(Serialize, Deserialize, Debug, Encode, Decode)]
pub enum NegotiationMessage {
    /// Server's offer to upgrade to a local IPC transport.
    /// Contains the unique name/path for the IPC socket.
    UpgradeToIpc { name: String },
    /// Server's decision to continue using the current TCP transport.
    ContinueOnTcp,
    /// Client's response accepting the IPC upgrade.
    AcceptIpcUpgrade,
    /// Client's response rejecting the IPC upgrade.
    RejectIpcUpgrade,
}

/// Sends a negotiation message over a TCP stream.
/// Messages are prefixed with their length to ensure proper framing.
pub(crate) async fn send_message(
    stream: &mut TcpStream,
    msg: &NegotiationMessage,
) -> Result<(), CommError> {
    let encoded = bincode::encode_to_vec(msg, bincode::config::standard()).unwrap();
    stream
        .write_u64(encoded.len() as u64)
        .await
        .map_err(|e| CommError::Io { source: e })?;
    stream
        .write_all(&encoded)
        .await
        .map_err(|e| CommError::Io { source: e })
}

/// Reads a negotiation message from a TCP stream.
pub(crate) async fn read_message(stream: &mut TcpStream) -> Result<NegotiationMessage, CommError> {
    let len = stream
        .read_u64()
        .await
        .map_err(|e| CommError::Io { source: e })?;
    let mut buf = vec![0; len as usize];
    stream
        .read_exact(&mut buf)
        .await
        .map_err(|e| CommError::Io { source: e })?;
    let (msg, _) = bincode::decode_from_slice(&buf, bincode::config::standard()).unwrap();
    Ok(msg)
}
