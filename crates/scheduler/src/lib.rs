//! The Scheduler of HSGS

#![warn(missing_docs)]

// TODO: for now, just ignore dead code warnings
#[allow(dead_code)]
pub mod components;

// TODO: for now, just ignore dead code warnings
#[allow(dead_code)]
pub mod schedule_system;

#[cfg(test)]
mod test;

#[cfg(feature = "scheduler_daemon")]
pub mod daemon;

/// Scheduler handle
pub trait Scheduler<Context: SystemContext> {
    /// Error when scheduling
    type Error;

    /// Observe the system context and make the new state the context should be
    fn schedule(
        &self,
        context: impl AsRef<Context>,
    ) -> impl Future<Output = Result<<Context as SystemContext>::State, Self::Error>>;
}

/// A [SystemContext] has necessary information for scheduling,
/// including configurable system state (e.g., resource partitions, requests task batch size, etc.)
/// and system status information like GPU usage, task wait list length.
pub trait SystemContext {
    /// configurable state for system context
    type State;

    /// get the reference of the system context
    fn state(&self) -> &Self::State;
}

/// A [SchedulableSystem] can provide a kind of [SystemContext],
/// and support some Command which can be executed by the system
#[trait_variant::make(Send)]
pub trait SchedulableSystem {
    /// The Command that can be executed by the system
    type Command;

    /// System Context
    type Context: SystemContext;

    /// Command execute error
    type Error: Send + Sync + 'static;

    /// Query system context
    async fn query_context(&self) -> Self::Context;

    /// Diff the current state with the target state, generate commands which can achieve this change
    fn generate_command(
        current: &Self::Context,
        target: <Self::Context as SystemContext>::State,
    ) -> impl IntoIterator<Item = Self::Command, IntoIter: Send>;

    /// Execute commands
    async fn execute_commands<I: IntoIterator<Item = Self::Command> + Send>(
        &self,
        commands: I,
    ) -> Result<(), Self::Error>
    where
        <I as IntoIterator>::IntoIter: Send;
}
