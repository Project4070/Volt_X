//! Volt X server entry point.
//!
//! Starts the Axum HTTP server on port 8080 and spawns the
//! background sleep consolidation scheduler.
//!
//! ## CLI Usage
//!
//! ```text
//! volt-server                    Start the server (default)
//! volt-server serve              Start the server
//! volt-server modules list       List installed modules
//! volt-server modules install X  Show instructions to enable module X
//! volt-server modules uninstall X Show instructions to disable module X
//! ```

use std::sync::Arc;

use volt_learn::rlvf::RlvfConfig;
use volt_learn::sleep::{SleepConfig, SleepScheduler};
use volt_server::registry::ModuleRegistry;
use volt_server::state::AppState;

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();

    match args.get(1).map(|s| s.as_str()) {
        Some("modules") => handle_modules(&args[2..]),
        Some("serve") => start_server().await,
        Some(other) => {
            eprintln!("Unknown command: {other}");
            eprintln!();
            print_usage();
            std::process::exit(1);
        }
        None => start_server().await,
    }
}

/// Print CLI usage information.
fn print_usage() {
    eprintln!("Volt X Server");
    eprintln!();
    eprintln!("Usage:");
    eprintln!("  volt-server                       Start the server (default)");
    eprintln!("  volt-server serve                  Start the server");
    eprintln!("  volt-server modules list           List installed modules");
    eprintln!("  volt-server modules install <name> Show install instructions");
    eprintln!("  volt-server modules uninstall <name> Show uninstall instructions");
}

/// Handle `volt-server modules ...` subcommands.
fn handle_modules(args: &[String]) {
    match args.first().map(|s| s.as_str()) {
        Some("list") => {
            let registry = ModuleRegistry::discover();
            let modules = registry.list_modules();
            println!("Installed modules ({}):", modules.len());
            println!();
            for m in modules {
                println!(
                    "  {} v{} [{}]",
                    m.id, m.version, m.module_type
                );
                println!("    {}", m.description);
            }
        }
        Some("install") => {
            if let Some(name) = args.get(1) {
                println!("To install module '{name}':");
                println!();
                println!("  1. Add the module crate as a workspace dependency in Cargo.toml:");
                println!(
                    "     volt-strand-{name} = {{ path = \"crates/volt-strand-{name}\" }}"
                );
                println!();
                println!("  2. Add a feature flag to crates/volt-hard/Cargo.toml:");
                println!("     [features]");
                println!("     {name} = []");
                println!();
                println!("  3. Add the feature propagation to crates/volt-server/Cargo.toml:");
                println!("     {name} = [\"volt-hard/{name}\"]");
                println!();
                println!("  4. Rebuild with the feature enabled:");
                println!("     cargo build --release --features {name}");
                println!();
                println!("  5. Restart the server.");
            } else {
                eprintln!("Usage: volt-server modules install <module-name>");
                std::process::exit(1);
            }
        }
        Some("uninstall") => {
            if let Some(name) = args.get(1) {
                println!("To uninstall module '{name}':");
                println!();
                println!("  1. Remove '{name}' from the default features in crates/volt-server/Cargo.toml");
                println!();
                println!("  2. Rebuild without the feature:");
                println!("     cargo build --release");
                println!();
                println!("  3. Restart the server.");
                println!();
                println!("  (Optional) Remove the crate from the workspace.");
            } else {
                eprintln!("Usage: volt-server modules uninstall <module-name>");
                std::process::exit(1);
            }
        }
        _ => {
            eprintln!("Usage: volt-server modules [list|install|uninstall]");
            std::process::exit(1);
        }
    }
}

/// Start the HTTP server with sleep scheduler.
async fn start_server() {
    tracing_subscriber::fmt::init();

    tracing::info!("Starting Volt X server on 0.0.0.0:8080");

    // Create shared state so we can pass references to the sleep scheduler.
    let state: Arc<AppState> = AppState::new();

    tracing::info!(
        "Module registry: {} modules discovered",
        state.registry.module_count()
    );

    // Spawn the background sleep consolidation scheduler.
    // It shares the VFN, memory store, and event logger with the server.
    let sleep_config = SleepConfig {
        rlvf_config: Some(RlvfConfig::default()),
        ..SleepConfig::default()
    };
    let sleep_handle = SleepScheduler::spawn_background(
        sleep_config,
        state.memory.inner_arc(),
        Arc::clone(&state.vfn),
        Arc::clone(&state.event_logger),
    )
    .expect("failed to spawn sleep scheduler");

    tracing::info!("Sleep consolidation scheduler started (idle timeout: 10 min)");

    let app = volt_server::build_app_with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080")
        .await
        .expect("failed to bind to port 8080");

    tracing::info!("Volt X server listening on 0.0.0.0:8080");

    axum::serve(listener, app).await.expect("server error");

    // Graceful shutdown: stop the sleep scheduler.
    sleep_handle.stop();
    let _ = sleep_handle.join();
}
