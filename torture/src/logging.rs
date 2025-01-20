use std::io::{self, IsTerminal as _};
use std::path::Path;

use tracing::level_filters::LevelFilter;
use tracing::{span, Level};
use tracing_subscriber::{fmt, EnvFilter};
use tracing_subscriber::{prelude::*, registry::Registry, Layer};

const ENV_NAME_COMMON: &str = "TORTURE_ALL_LOG";
const ENV_NAME_AGENT: &str = "TORTURE_AGENT_LOG";
const ENV_NAME_SUPERVISOR: &str = "TORTURE_SUPERVISOR_LOG";

enum Kind {
    Agent,
    Supervisor,
}

fn istty() -> bool {
    io::stdout().is_terminal() && io::stderr().is_terminal()
}

/// Creates env filter for the agent or supervisor (depending on the `agent_not_supervisor`
/// argument).
///
/// This function tries to read the most specific environment variable first, then falls back to
/// the common one ([`ENV_NAME_COMMON`]).
fn env_filter(kind: Kind) -> EnvFilter {
    let specific_env_name = match kind {
        Kind::Agent => ENV_NAME_AGENT,
        Kind::Supervisor => ENV_NAME_SUPERVISOR,
    };

    return try_parse_env(specific_env_name).unwrap_or_else(|| {
        try_parse_env(ENV_NAME_COMMON).unwrap_or_else(|| {
            EnvFilter::builder()
                .with_default_directive(LevelFilter::INFO.into())
                .parse("")
                .unwrap()
        })
    });

    fn try_parse_env(var_name: &str) -> Option<EnvFilter> {
        match std::env::var(var_name) {
            Ok(env) => Some(
                EnvFilter::builder()
                    .with_default_directive(LevelFilter::INFO.into())
                    .parse(env)
                    .unwrap(),
            ),
            Err(std::env::VarError::NotPresent) => {
                return None;
            }
            Err(std::env::VarError::NotUnicode(_)) => {
                panic!("Environment variable {} is not unicode", var_name);
            }
        }
    }
}

pub fn init_supervisor() {
    let format = fmt::format()
        .with_level(true)
        .with_target(false)
        .with_thread_ids(false)
        .with_thread_names(false)
        .compact()
        .with_timer(fmt::time::SystemTime::default());
    let subscriber = fmt::Subscriber::builder()
        .with_env_filter(env_filter(Kind::Supervisor))
        .with_writer(io::stdout)
        .with_ansi(istty())
        .event_format(format)
        .finish();

    tracing::subscriber::set_global_default(subscriber)
        .expect("Failed to set supervisor subscriber");
}

pub fn init_agent(agent_id: &str, workdir: &Path) {
    // Console layer with ANSI colors
    let console_layer = fmt::layer()
        .with_writer(io::stdout)
        .event_format(
            fmt::format()
                .with_level(true)
                .with_target(false)
                .with_thread_ids(false)
                .with_thread_names(false)
                .with_ansi(istty())
                .compact()
                .with_timer(fmt::time::SystemTime::default()),
        )
        .with_filter(env_filter(Kind::Agent));

    // File layer with ANSI disabled
    // TODO: this has an issue currently. While the ANSI is false the colors are not disabled
    // everywhere.
    let file = std::fs::File::options()
        .create(true)
        .append(true)
        .open(workdir.join("agent.log"))
        .unwrap();

    // TODO: this has an issue currently. While the ANSI is false the colors are not disabled
    // everywhere.
    let file_layer = fmt::layer()
        .with_writer(file)
        .event_format(
            fmt::format()
                .with_level(true)
                .with_target(false)
                .with_thread_ids(false)
                .with_thread_names(false)
                .with_ansi(false)
                .compact()
                .with_timer(fmt::time::SystemTime::default()),
        )
        .with_filter(env_filter(Kind::Agent));

    // Combine both layers in a single Subscriber
    let subscriber = Registry::default().with(console_layer).with(file_layer);

    // Set the global subscriber
    tracing::subscriber::set_global_default(subscriber).expect("Failed to set agent subscriber");

    let pid = std::process::id();
    let span = span!(Level::INFO, "agent", agent_id, pid);
    let _enter = span.enter();
    // We intentionally `forget` the guard so the span remains open
    // for the lifetime of the entire agent process if desired.
    std::mem::forget(_enter);
}
