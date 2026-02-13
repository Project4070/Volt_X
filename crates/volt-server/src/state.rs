//! Shared application state for the Axum server.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::SystemTime;
use volt_core::VoltError;
use volt_db::{ConcurrentVoltStore, VoltStore};
use volt_learn::EventLogger;
use volt_soft::vfn::Vfn;
use volt_translate::StubTranslator;

use crate::models::ConversationMeta;
use crate::registry::ModuleRegistry;

/// Thread-safe event logger shared across handlers.
pub type ConcurrentEventLogger = Arc<RwLock<EventLogger>>;

/// Thread-safe VFN shared between inference and the sleep scheduler.
pub type SharedVfn = Arc<RwLock<Vfn>>;

/// Shared application state, passed to all route handlers via Axum `State`.
///
/// The [`StubTranslator`] uses internal `RwLock` for thread safety.
/// The [`ConcurrentVoltStore`] uses `Arc<RwLock<VoltStore>>` for
/// multi-reader / single-writer access to the memory system.
/// The [`ConcurrentEventLogger`] accumulates learning events from
/// every inference run.
/// The [`SharedVfn`] is read during inference and written during
/// sleep consolidation (Forward-Forward + RLVF training).
/// The `conversations` map tracks conversation metadata (created_at,
/// last_message_at, message_count) for all active conversations.
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
    /// The shared VFN used by both inference (read) and learning (write).
    pub vfn: SharedVfn,
    /// Registry of all installed modules (Milestone 6.1).
    pub registry: ModuleRegistry,
    /// Conversation metadata indexed by conversation ID.
    pub conversations: Arc<RwLock<HashMap<u64, ConversationMeta>>>,
}

impl AppState {
    /// Create new application state with a fresh [`StubTranslator`],
    /// an in-memory [`VoltStore`], an empty [`EventLogger`], and a
    /// randomly-initialized [`Vfn`].
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
            vfn: Arc::new(RwLock::new(Vfn::new_random(42))),
            registry: ModuleRegistry::discover(),
            conversations: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Get an existing conversation or create a new one.
    ///
    /// If `id` is `Some`, returns that ID (creates metadata if needed).
    /// If `id` is `None`, generates a new conversation ID and creates metadata.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::StoreError`] if strand creation fails.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_server::state::AppState;
    ///
    /// let state = AppState::new();
    /// let conv_id = state.get_or_create_conversation(None).unwrap();
    /// assert!(conv_id > 0);
    /// ```
    pub fn get_or_create_conversation(&self, id: Option<u64>) -> Result<u64, VoltError> {
        match id {
            Some(existing_id) => {
                // Check if conversation metadata exists, create if not
                let mut convs = self.conversations.write().map_err(|e| VoltError::Internal {
                    message: format!("conversations lock poisoned: {e}"),
                })?;

                convs.entry(existing_id).or_insert_with(|| {
                    // Create metadata for existing strand
                    let now = SystemTime::now()
                        .duration_since(SystemTime::UNIX_EPOCH)
                        .map(|d| d.as_micros() as u64)
                        .unwrap_or(0);

                    ConversationMeta {
                        id: existing_id,
                        created_at: now,
                        last_message_at: now,
                        message_count: 0,
                    }
                });

                Ok(existing_id)
            }
            None => {
                // Generate new conversation ID (use timestamp-based ID for uniqueness)
                let now = SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .map(|d| d.as_micros() as u64)
                    .unwrap_or(0);

                let new_id = now; // Simple: use timestamp as ID

                // Create metadata
                let mut convs = self.conversations.write().map_err(|e| VoltError::Internal {
                    message: format!("conversations lock poisoned: {e}"),
                })?;

                convs.insert(
                    new_id,
                    ConversationMeta {
                        id: new_id,
                        created_at: now,
                        last_message_at: now,
                        message_count: 0,
                    },
                );

                // Create strand in VoltDB
                self.memory.write().map_err(|e| VoltError::Internal {
                    message: format!("memory lock poisoned: {e}"),
                })?.create_strand(new_id)?;

                Ok(new_id)
            }
        }
    }

    /// Update conversation metadata after a message is processed.
    ///
    /// Increments message_count and updates last_message_at timestamp.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_server::state::AppState;
    ///
    /// let state = AppState::new();
    /// let conv_id = state.get_or_create_conversation(None).unwrap();
    /// state.update_conversation_metadata(conv_id);
    /// ```
    pub fn update_conversation_metadata(&self, id: u64) {
        if let Ok(mut convs) = self.conversations.write()
            && let Some(meta) = convs.get_mut(&id)
        {
            let now = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .map(|d| d.as_micros() as u64)
                .unwrap_or(0);

            meta.last_message_at = now;
            meta.message_count += 1;
        }
    }
}
