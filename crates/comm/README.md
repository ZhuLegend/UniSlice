# Communication

Async TCP/local IPC transport, framing, serialization and connection reuse for internal RPC. The scheduler owns the HTTP interface.

| Component | Role |
|---|---|
| [`IoStream`](src/transport.rs) | TCP transport and local IPC negotiation |
| [`ConnectionService`](src/service/connection.rs) | Length-prefixed byte frames |
| [`Layer` / `Service`](src/service/mod.rs) | Composable connection behavior, including chunking |
| [`ConnectionBuilder` / `Connection`](src/connection.rs) | Byte-level connection construction |
| [`TypedConnection`](src/connection.rs) | Typed bincode encoding and decoding |
| [`Listener` / `ChannelCreator`](src/listener.rs) | Accept connections and dispatch requests |
| [`RpcPool`](src/rpc_pool.rs) | Reuse idle connections after complete responses |

Each in-flight RPC owns a connection. The pool retains at most 32 idle connections; that is not a limit on concurrent RPCs. Failed, cancelled or timed-out connections are discarded. Requests are not replayed automatically: a timed-out control operation may already have taken effect remotely. Implementing serde alone does not satisfy every typed bincode connection bound. See the [connection tests](src/test).

The protocol assumes trusted peers. Frame limits do not constitute a complete bound on all handshake/chunk reassembly input. Follow the [network boundary](../../README.md#hardware-and-network-scope).

```sh
cargo test --locked -p comm rpc_pool --lib
```
