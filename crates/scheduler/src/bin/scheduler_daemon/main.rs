//! Nvidia MIG 调度器守护进程主入口点

use clap::Parser;
use scheduler::daemon::runner;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = runner::Args::parse();
    runner::run(args).await
}
