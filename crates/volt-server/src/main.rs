//! Volt X server entry point.
//!
//! Starts the Axum HTTP server on port 8080.

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    tracing::info!("Starting Volt X server on 0.0.0.0:8080");

    let app = volt_server::build_app();

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080")
        .await
        .expect("failed to bind to port 8080");

    tracing::info!("Volt X server listening on 0.0.0.0:8080");

    axum::serve(listener, app)
        .await
        .expect("server error");
}
