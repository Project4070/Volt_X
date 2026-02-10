//! LLM-backed forward translator (Milestone 2.2).
//!
//! Uses a frozen Qwen3-0.6B backbone with a trained Frame Projection Head
//! to map natural language into TensorFrames with semantic role labels and
//! codebook-quantized slot vectors.
//!
//! Feature-gated behind `llm`:
//! ```bash
//! cargo test -p volt-translate --features llm
//! ```

pub mod backbone;
pub mod projection;
pub mod roles;
pub mod translator;

pub use backbone::LlmBackbone;
pub use projection::{aggregate_to_slots, FrameProjectionHead, ProjectionConfig};
pub use roles::PropBankRole;
pub use translator::{LlmTranslator, LlmTranslatorConfig};
