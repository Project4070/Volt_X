//! Shared application state for the Axum server.

use std::sync::Arc;
use volt_translate::StubTranslator;

/// Shared application state, passed to all route handlers via Axum `State`.
///
/// The [`StubTranslator`] uses internal `RwLock` for thread safety,
/// so no additional synchronization is needed here.
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
}

impl AppState {
    /// Create new application state with a fresh [`StubTranslator`].
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
        })
    }
}
