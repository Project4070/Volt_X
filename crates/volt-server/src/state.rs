//! Shared application state for the Axum server.

use std::sync::Arc;
use volt_db::{ConcurrentVoltStore, VoltStore};
use volt_translate::StubTranslator;

/// Shared application state, passed to all route handlers via Axum `State`.
///
/// The [`StubTranslator`] uses internal `RwLock` for thread safety.
/// The [`ConcurrentVoltStore`] uses `Arc<RwLock<VoltStore>>` for
/// multi-reader / single-writer access to the memory system.
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
}

impl AppState {
    /// Create new application state with a fresh [`StubTranslator`]
    /// and an in-memory [`VoltStore`].
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
        })
    }
}
