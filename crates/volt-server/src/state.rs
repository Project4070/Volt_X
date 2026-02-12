//! Shared application state for the Axum server.

use std::sync::{Arc, RwLock};
use volt_db::{ConcurrentVoltStore, VoltStore};
use volt_learn::EventLogger;
use volt_translate::StubTranslator;

/// Thread-safe event logger shared across handlers.
pub type ConcurrentEventLogger = Arc<RwLock<EventLogger>>;

/// Shared application state, passed to all route handlers via Axum `State`.
///
/// The [`StubTranslator`] uses internal `RwLock` for thread safety.
/// The [`ConcurrentVoltStore`] uses `Arc<RwLock<VoltStore>>` for
/// multi-reader / single-writer access to the memory system.
/// The [`ConcurrentEventLogger`] accumulates learning events from
/// every inference run.
///
/// # Example
///
/// ```
/// use volt_server::state::AppState;
///
/// let state = AppState::new();
/// ```
pub struct AppState {
    /// The translator for encode/decode operations.
    pub translator: StubTranslator,
    /// The three-tier memory store (T0 + T1 + HNSW + Ghost Bleed).
    pub memory: ConcurrentVoltStore,
    /// The learning event logger (Milestone 5.1).
    pub event_logger: ConcurrentEventLogger,
}

impl AppState {
    /// Create new application state with a fresh [`StubTranslator`],
    /// an in-memory [`VoltStore`], and an empty [`EventLogger`].
    ///
    /// Returns an `Arc<Self>` ready for sharing across Axum handlers.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_server::state::AppState;
    ///
    /// let state = AppState::new();
    /// ```
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            translator: StubTranslator::new(),
            memory: ConcurrentVoltStore::new(VoltStore::new()),
            event_logger: Arc::new(RwLock::new(EventLogger::new())),
        })
    }
}
