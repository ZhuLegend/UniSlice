# Utilities

Shared task lifecycle, path validation and simple error types.

[`sync::Task`](src/sync/mod.rs) combines `CancellationToken` and `TaskTracker`. Clones share cancellation/registration state; `sub_task()` creates a child cancellation token. Cancellation stops polling the submitted future. Dropping a separately spawned `JoinHandle` does not stop its task, and already running `spawn_blocking()` work cannot be forcibly stopped by the token. Shutdown must wait for registered tasks.

[`fs`](src/lib.rs) provides directory/path validation, with CLI parsers under the `clap` feature. `ErrorMessage` wraps simple error messages.

```sh
cargo test --locked -p utils --lib
```
