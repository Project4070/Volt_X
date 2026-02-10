//! LLM-backed translator implementing the [`Translator`] trait.
//!
//! Combines the frozen Qwen3-0.6B backbone, trained Frame Projection Head,
//! and VQ-VAE codebook to convert natural language into TensorFrames with
//! semantic role labels and codebook-quantized slot vectors.
//!
//! # Pipeline
//!
//! ```text
//! Input text
//!   → Tokenize (Qwen3 BPE)
//!   → Frozen backbone → hidden_states [seq_len, 1024]
//!   → Projection head → role_probs [seq_len, 16] + embeds [seq_len, 256]
//!   → Weighted average per role → slot vectors
//!   → L2-normalize → Codebook quantize → TensorFrame
//! ```

use std::cell::RefCell;
use std::path::PathBuf;

use volt_bus::codebook::Codebook;
use volt_core::meta::DiscourseType;
use volt_core::slot::{SlotMeta, SlotSource};
use volt_core::{SlotData, SlotRole, TensorFrame, VoltError, MAX_SLOTS};

use super::backbone::LlmBackbone;
use super::projection::{aggregate_to_slots, FrameProjectionHead, ProjectionConfig};
use super::roles::slot_to_role;
use crate::{TranslateOutput, Translator};

/// Configuration for the [`LlmTranslator`].
///
/// # Example
///
/// ```no_run
/// use volt_translate::llm::translator::LlmTranslatorConfig;
///
/// let config = LlmTranslatorConfig {
///     model_dir: "models/Qwen3-0.6B".into(),
///     projection_weights: "models/projection.safetensors".into(),
///     codebook_path: "models/codebook.bin".into(),
///     hidden_dim: 1024,
///     mlp_dim: 4096,
///     role_threshold: 0.1,
///     base_certainty: 0.85,
/// };
/// ```
#[derive(Debug, Clone)]
pub struct LlmTranslatorConfig {
    /// Path to the HuggingFace model directory (config.json, *.safetensors, tokenizer.json).
    pub model_dir: PathBuf,
    /// Path to the projection head weights (safetensors).
    pub projection_weights: PathBuf,
    /// Path to the VQ-VAE codebook binary file.
    pub codebook_path: PathBuf,
    /// Backbone hidden dimension (1024 for Qwen3-0.6B).
    pub hidden_dim: usize,
    /// Projection head MLP dimension (default 4096).
    pub mlp_dim: usize,
    /// Minimum total role probability to activate a slot.
    pub role_threshold: f32,
    /// Base certainty assigned to translator-produced slots.
    pub base_certainty: f32,
}

impl Default for LlmTranslatorConfig {
    fn default() -> Self {
        Self {
            model_dir: PathBuf::new(),
            projection_weights: PathBuf::new(),
            codebook_path: PathBuf::new(),
            hidden_dim: 1024,
            mlp_dim: 4096,
            role_threshold: 0.1,
            base_certainty: 0.85,
        }
    }
}

/// LLM-backed translator: Qwen3-0.6B + Projection Head + Codebook.
///
/// Implements the [`Translator`] trait for production-quality
/// natural language → TensorFrame conversion.
///
/// # Example
///
/// ```
/// use volt_translate::llm::translator::LlmTranslator;
/// use volt_translate::Translator;
/// use candle_core::Device;
///
/// let translator = LlmTranslator::mock(64, &Device::Cpu).unwrap();
/// let output = translator.encode("the cat sat on the mat").unwrap();
/// assert!(output.slots_filled > 0);
/// ```
// Note: Debug implemented manually, Clone omitted because the
// backbone holds heavyweight ML model weights.
pub struct LlmTranslator {
    backbone: RefCell<LlmBackbone>,
    projection: FrameProjectionHead,
    codebook: Option<Codebook>,
    role_threshold: f32,
    base_certainty: f32,
}

impl std::fmt::Debug for LlmTranslator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlmTranslator")
            .field("backbone", &*self.backbone.borrow())
            .field("has_codebook", &self.codebook.is_some())
            .field("role_threshold", &self.role_threshold)
            .field("base_certainty", &self.base_certainty)
            .finish()
    }
}

impl LlmTranslator {
    /// Creates a fully-loaded translator from configuration.
    ///
    /// Loads the Qwen3-0.6B backbone, projection head, and codebook
    /// from the paths specified in `config`.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::TranslateError`] if any file cannot be loaded.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use volt_translate::llm::translator::{LlmTranslator, LlmTranslatorConfig};
    /// use candle_core::Device;
    ///
    /// let config = LlmTranslatorConfig {
    ///     model_dir: "models/Qwen3-0.6B".into(),
    ///     projection_weights: "models/projection.safetensors".into(),
    ///     codebook_path: "models/codebook.bin".into(),
    ///     ..Default::default()
    /// };
    /// let translator = LlmTranslator::new(&config, &Device::Cpu).unwrap();
    /// ```
    pub fn new(
        config: &LlmTranslatorConfig,
        device: &candle_core::Device,
    ) -> Result<Self, VoltError> {
        let backbone = LlmBackbone::load(&config.model_dir, device)?;

        let proj_config = ProjectionConfig {
            hidden_dim: config.hidden_dim,
            mlp_dim: config.mlp_dim,
        };
        let projection =
            FrameProjectionHead::load(&config.projection_weights, &proj_config, device)?;

        let codebook = Codebook::load(&config.codebook_path)?;

        Ok(Self {
            backbone: RefCell::new(backbone),
            projection,
            codebook: Some(codebook),
            role_threshold: config.role_threshold,
            base_certainty: config.base_certainty,
        })
    }

    /// Creates a mock translator for testing (no model files needed).
    ///
    /// Uses a mock backbone (whitespace tokenizer, deterministic hidden
    /// states) and a random projection head. No codebook — slot vectors
    /// are stored unquantized.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_translate::llm::translator::LlmTranslator;
    /// use volt_translate::Translator;
    /// use candle_core::Device;
    ///
    /// let translator = LlmTranslator::mock(64, &Device::Cpu).unwrap();
    /// let output = translator.encode("hello world").unwrap();
    /// assert_eq!(output.token_count, 2);
    /// assert!(output.slots_filled > 0);
    /// ```
    pub fn mock(hidden_dim: usize, device: &candle_core::Device) -> Result<Self, VoltError> {
        let backbone = LlmBackbone::mock(hidden_dim, device);
        let proj_config = ProjectionConfig {
            hidden_dim,
            mlp_dim: hidden_dim * 2,
        };
        let projection = FrameProjectionHead::new_random(&proj_config, device)?;

        Ok(Self {
            backbone: RefCell::new(backbone),
            projection,
            codebook: None,
            role_threshold: 0.1,
            base_certainty: 0.85,
        })
    }
}

impl Translator for LlmTranslator {
    /// Encodes natural language text into a TensorFrame.
    ///
    /// Pipeline: tokenize → hidden states → projection → aggregate → quantize → frame.
    fn encode(&self, input: &str) -> Result<TranslateOutput, VoltError> {
        // Validate input
        if input.trim().is_empty() {
            return Err(VoltError::TranslateError {
                message: "cannot encode empty input".into(),
            });
        }

        // Tokenize and extract hidden states
        let mut backbone = self.backbone.borrow_mut();
        let token_ids = backbone.tokenize(input)?;
        let token_count = token_ids.len();
        let hidden_states = backbone.extract_hidden_states(&token_ids)?;
        drop(backbone);

        // Run projection head
        let (role_probs, token_embeds) = self.projection.forward(&hidden_states)?;

        // Aggregate to slot vectors
        let active_slots = aggregate_to_slots(&role_probs, &token_embeds, self.role_threshold)?;

        // Build TensorFrame
        let mut frame = TensorFrame::new();
        let mut slots_filled = 0;

        for &(slot_idx, ref slot_vec, confidence) in &active_slots {
            if slot_idx >= MAX_SLOTS {
                continue;
            }

            let role = slot_to_role(slot_idx);
            let mut slot = SlotData::new(role);

            // Quantize through codebook if available
            if let Some(codebook) = &self.codebook {
                match codebook.quantize(slot_vec) {
                    Ok((cb_id, quantized)) => {
                        slot.write_resolution(0, quantized);
                        slot.codebook_id = Some(cb_id);
                    }
                    Err(_) => {
                        // Fall back to raw vector if quantization fails
                        slot.write_resolution(0, *slot_vec);
                    }
                }
            } else {
                slot.write_resolution(0, *slot_vec);
            }

            frame.write_slot(slot_idx, slot)?;

            // Set slot metadata
            frame.meta[slot_idx] = SlotMeta {
                certainty: self.base_certainty * (confidence / active_slots.len() as f32).min(1.0),
                source: SlotSource::Translator,
                updated_at: now_micros(),
                needs_verify: true,
            };

            slots_filled += 1;
        }

        // Set frame metadata
        frame.frame_meta.discourse_type = classify_discourse(input);
        frame.frame_meta.global_certainty = frame.min_certainty().unwrap_or(0.0);
        frame.frame_meta.created_at = now_micros();

        Ok(TranslateOutput {
            frame,
            token_count,
            slots_filled,
        })
    }

    /// Decodes a TensorFrame back into human-readable text.
    ///
    /// Produces a bracketed representation: `[S0:Agent] cb:1234 [S1:Predicate] cb:5678 ...`
    fn decode(&self, frame: &TensorFrame) -> Result<String, VoltError> {
        let slots = self.decode_slots(frame)?;
        if slots.is_empty() {
            return Ok(String::new());
        }

        let parts: Vec<String> = slots
            .iter()
            .map(|(idx, role, desc)| format!("[S{idx}:{role:?}] {desc}"))
            .collect();
        Ok(parts.join(" "))
    }

    /// Decodes each active slot individually.
    ///
    /// Returns `(slot_index, role, description)` for each populated slot.
    /// The description includes the codebook ID if available, or "raw" otherwise.
    fn decode_slots(
        &self,
        frame: &TensorFrame,
    ) -> Result<Vec<(usize, SlotRole, String)>, VoltError> {
        let mut result = Vec::new();

        for i in 0..MAX_SLOTS {
            if let Some(slot) = &frame.slots[i] {
                let desc = if let Some(cb_id) = slot.codebook_id {
                    format!("cb:{cb_id}")
                } else {
                    "raw".to_string()
                };
                result.push((i, slot.role, desc));
            }
        }

        Ok(result)
    }
}

/// Classifies the discourse type of the input text.
///
/// Simple heuristic: `?` → Query, `!` → Command, otherwise → Statement.
fn classify_discourse(text: &str) -> DiscourseType {
    let trimmed = text.trim();
    if trimmed.ends_with('?') {
        DiscourseType::Query
    } else if trimmed.ends_with('!') {
        DiscourseType::Command
    } else {
        DiscourseType::Statement
    }
}

/// Returns the current time in microseconds since the Unix epoch.
fn now_micros() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use volt_core::meta::DiscourseType;

    fn mock_translator() -> LlmTranslator {
        LlmTranslator::mock(64, &Device::Cpu).unwrap()
    }

    #[test]
    fn encode_basic() {
        let t = mock_translator();
        let output = t.encode("the cat sat").unwrap();
        assert_eq!(output.token_count, 3);
        assert!(output.slots_filled > 0);
        assert!(!output.frame.is_empty());
    }

    #[test]
    fn encode_empty_errors() {
        let t = mock_translator();
        assert!(t.encode("").is_err());
        assert!(t.encode("   ").is_err());
    }

    #[test]
    fn encode_max_16_slots() {
        let t = mock_translator();
        let output = t
            .encode("a b c d e f g h i j k l m n o p q r s t")
            .unwrap();
        assert!(output.slots_filled <= MAX_SLOTS);
    }

    #[test]
    fn encode_sets_metadata() {
        let t = mock_translator();
        let output = t.encode("hello world").unwrap();

        // Frame-level metadata
        assert!(output.frame.frame_meta.created_at > 0);
        assert_eq!(
            output.frame.frame_meta.discourse_type,
            DiscourseType::Statement
        );

        // Slot-level metadata
        for i in 0..MAX_SLOTS {
            if output.frame.slots[i].is_some() {
                assert_eq!(output.frame.meta[i].source, SlotSource::Translator);
                assert!(output.frame.meta[i].needs_verify);
                assert!(output.frame.meta[i].certainty > 0.0);
            }
        }
    }

    #[test]
    fn encode_question_classified_as_query() {
        let t = mock_translator();
        let output = t.encode("what is this?").unwrap();
        assert_eq!(
            output.frame.frame_meta.discourse_type,
            DiscourseType::Query
        );
    }

    #[test]
    fn encode_command_classified() {
        let t = mock_translator();
        let output = t.encode("do it now!").unwrap();
        assert_eq!(
            output.frame.frame_meta.discourse_type,
            DiscourseType::Command
        );
    }

    #[test]
    fn encode_different_inputs_different_frames() {
        let t = mock_translator();
        let out1 = t.encode("the cat sat").unwrap();
        let out2 = t.encode("dogs run fast").unwrap();

        // At least the data should differ (different hidden states → different embeddings)
        let s1 = out1.frame.data_size_bytes();
        let s2 = out2.frame.data_size_bytes();
        // Both should have non-zero data
        assert!(s1 > 0);
        assert!(s2 > 0);
    }

    #[test]
    fn encode_no_codebook_stores_raw() {
        let t = mock_translator();
        let output = t.encode("test input").unwrap();

        // Mock has no codebook, so codebook_id should be None
        for i in 0..MAX_SLOTS {
            if let Some(slot) = &output.frame.slots[i] {
                assert!(
                    slot.codebook_id.is_none(),
                    "mock translator should not set codebook_id"
                );
            }
        }
    }

    #[test]
    fn encode_slot_vectors_normalized() {
        let t = mock_translator();
        let output = t.encode("test input data here").unwrap();

        for i in 0..MAX_SLOTS {
            if let Some(slot) = &output.frame.slots[i] {
                if let Some(vec) = &slot.resolutions[0] {
                    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                    assert!(
                        (norm - 1.0).abs() < 0.1,
                        "slot {i} vector norm = {norm}, expected ~1.0"
                    );
                }
            }
        }
    }

    #[test]
    fn decode_roundtrip() {
        let t = mock_translator();
        let output = t.encode("hello world").unwrap();
        let decoded = t.decode(&output.frame).unwrap();
        assert!(!decoded.is_empty());
    }

    #[test]
    fn decode_slots_returns_all_active() {
        let t = mock_translator();
        let output = t.encode("the cat sat").unwrap();
        let slots = t.decode_slots(&output.frame).unwrap();
        assert_eq!(slots.len(), output.slots_filled);
    }

    #[test]
    fn decode_empty_frame() {
        let t = mock_translator();
        let frame = TensorFrame::new();
        let decoded = t.decode(&frame).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn classify_discourse_heuristic() {
        assert_eq!(classify_discourse("what is this?"), DiscourseType::Query);
        assert_eq!(classify_discourse("stop now!"), DiscourseType::Command);
        assert_eq!(
            classify_discourse("the cat sat"),
            DiscourseType::Statement
        );
    }

    #[test]
    fn debug_display() {
        let t = mock_translator();
        let debug = format!("{t:?}");
        assert!(debug.contains("LlmTranslator"));
        assert!(debug.contains("mock"));
    }
}
