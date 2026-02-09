//! # volt-translate
//!
//! Input/output translators for Volt X.
//!
//! Translators convert between external modalities and TensorFrames:
//! - **Forward Translator**: NL text → TensorFrame (encode)
//! - **Reverse Translator**: TensorFrame → NL text (decode)
//! - **Community Modules**: Pluggable translators via `impl Translator` trait
//!
//! ## Supported Modalities (planned)
//!
//! - Text (BPE → Frame slots)
//! - Vision (ViT → Frame slots)
//! - Audio (Whisper → Frame)
//! - Structured data (JSON/CSV → Frame)
//!
//! ## Architecture Rules
//!
//! - Translators implement the `Translator` trait.
//! - Community modules are hot-pluggable.
//! - Depends on `volt-core`, `volt-bus`, `volt-db`.

pub use volt_core;

// MILESTONE: 1.1 — Text Translator (forward + reverse)
// TODO: Define Translator trait { fn encode(&self, raw: &[u8]) -> TensorFrame }
// TODO: Implement basic text-to-frame translator
// TODO: Implement frame-to-text reverse translator
// TODO: Implement slot-parallel decoding
