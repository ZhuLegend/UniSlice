# Contributing

Describe the observed behavior, expected behavior and the configuration needed to reproduce a change. Keep scheduling-policy changes separate from backend simulation and workload tuning. Never claim paper reproduction from a compile check alone.

## CPU-only checks

Run from the repository root with a current stable Rust toolchain:

```bash
cargo test --locked -p latency_profile
cargo test --locked -p worker --features mock --lib
cargo test --locked -p compute_node --features mock,vgpu,bin_help --lib
cargo test --locked -p compute_node --features mock,vgpu,bin_help --test simulation
cargo test --locked -p scheduler --features scheduler_daemon,task_sender,mock,huawei,nvidia --lib --bin task_sender
python3 -m unittest discover -s scripts/simulation
python3 scripts/simulation/run.py --seconds 12 --warmup-seconds 2 \
  --create-delay-ms 20 --destroy-delay-ms 20 --schedule-interval 10
```

Add `--offline` to Cargo commands when dependencies are already cached. Integration tests bind localhost ports and spawn mock worker processes. The slow wall-clock test is opt-in:

```bash
cargo test --locked -p compute_node --features mock,vgpu,bin_help \
  --test simulation wall_clock_replacement_is_sixty_seconds -- --ignored --nocapture
```

This test explicitly uses 30 seconds for create and destroy and takes roughly 120 seconds: initial create, one replacement, final cleanup. Normal simulation defaults are 15 seconds for create and 15 seconds for destroy; keep shorter development settings separate from experiment settings.

## Formatting and release checks

Use nightly rustfmt: `.rustfmt.toml` configures unstable import grouping rules, so stable rustfmt does not produce the configured output. Format a selected Rust file with `rustfmt +nightly --edition 2024 path/to/file.rs`; keep formatting separate from behavior changes. For an explicit repository-wide formatting change:

```bash
cargo +nightly fmt --all
cargo +nightly fmt --all -- --check
```

Run the default workspace checks and dependency audit before a release:

```bash
cargo check --workspace --locked
cargo test --workspace --locked
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked
cargo audit
```

Do not ignore an advisory solely because normal runs pass. Check the affected version, enabled features and actual call path; update vulnerable dependencies or document why the advisory does not apply. Missing runtime paths or local overrides must not substitute for a clean checkout check. Include tests for resource exhaustion, cancellation or lifecycle changes when relevant.

## Data and results

Preserve the original measured CSV. Add profile provenance separately. Do not commit model weights, generated logs, private paper notes or local configuration secrets. Store configurations, code revision and CSV checksum with experimental results; report failed and unfinished work.

The original README declares MIT OR Apache-2.0. Contributions are distributed under the same terms unless explicitly stated otherwise.
