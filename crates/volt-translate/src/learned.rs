//! Learned translator wrapping the CNN code encoder and decoder.
//!
//! Implements the [`Translator`] trait using trained weights from
//! Phase 1. Requires the `code-training` feature flag.
//!
//! # Example
//!
//! ```ignore
//! use volt_translate::learned::LearnedTranslator;
//! use std::path::Path;
//!
//! let translator = LearnedTranslator::load(
//!     Path::new("checkpoints/code_tokenizer.json"),
//!     Path::new("checkpoints/code_encoder.safetensors"),
//!     Path::new("checkpoints/code_decoder.safetensors"),
//! ).unwrap();
//!
//! let output = translator.encode("def add(a, b): return a + b").unwrap();
//! let decoded = translator.decode(&output.frame).unwrap();
//! ```

use std::path::Path;

use candle_core::{Device, Tensor};
use tokenizers::Tokenizer;
use volt_core::{SlotRole, TensorFrame, VoltError, MAX_SLOTS, SLOT_DIM};

use crate::code_decoder::{CodeDecoder, CodeDecoderConfig};
use crate::code_encoder::{CodeEncoder, CodeEncoderConfig};
use crate::{TranslateOutput, Translator};

/// A learned code translator using CNN encoder + non-autoregressive decoder.
///
/// Wraps a BPE tokenizer, [`CodeEncoder`], and [`CodeDecoder`] to
/// implement the [`Translator`] trait.
pub struct LearnedTranslator {
    tokenizer: Tokenizer,
    encoder: CodeEncoder,
    decoder: CodeDecoder,
}

impl std::fmt::Debug for LearnedTranslator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "LearnedTranslator(encoder={:?}, decoder={:?})",
            self.encoder, self.decoder
        )
    }
}

impl LearnedTranslator {
    /// Load a trained translator from disk.
    ///
    /// # Arguments
    ///
    /// * `tokenizer_path` — Path to the BPE tokenizer JSON file.
    /// * `encoder_path` — Path to encoder safetensors weights.
    /// * `decoder_path` — Path to decoder safetensors weights.
    ///
    /// # Errors
    ///
    /// Returns error if any file cannot be loaded.
    pub fn load(
        tokenizer_path: &Path,
        encoder_path: &Path,
        decoder_path: &Path,
    ) -> Result<Self, VoltError> {
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| VoltError::TranslateError {
            message: format!(
                "failed to load tokenizer from {}: {e}",
                tokenizer_path.display()
            ),
        })?;

        let device = Device::Cpu;
        let enc_config = CodeEncoderConfig::default();
        let dec_config = CodeDecoderConfig::default();

        let encoder = CodeEncoder::load(&enc_config, encoder_path, &device)?;
        let decoder = CodeDecoder::load(&dec_config, decoder_path, &device)?;

        Ok(Self {
            tokenizer,
            encoder,
            decoder,
        })
    }

    /// Tokenize text into BPE token IDs.
    fn tokenize(&self, text: &str) -> Result<Vec<u32>, VoltError> {
        let encoding = self.tokenizer.encode(text, false).map_err(|e| {
            VoltError::TranslateError {
                message: format!("tokenization failed: {e}"),
            }
        })?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Create a tensor from token IDs (single sample, batch dim = 1).
    fn ids_to_tensor(&self, ids: &[u32]) -> Result<Tensor, VoltError> {
        let max_len = self.encoder.config().max_seq_len;
        let ids: Vec<u32> = ids.iter().take(max_len).copied().collect();
        let len = ids.len();
        Tensor::from_vec(ids, (1, len), &Device::Cpu).map_err(|e| VoltError::Internal {
            message: format!("tensor creation failed: {e}"),
        })
    }
}

impl Translator for LearnedTranslator {
    fn encode(&self, input: &str) -> Result<TranslateOutput, VoltError> {
        if input.is_empty() {
            return Err(VoltError::TranslateError {
                message: "input is empty".to_string(),
            });
        }

        let ids = self.tokenize(input)?;
        let token_count = ids.len();
        let tensor = self.ids_to_tensor(&ids)?;
        let frame = self.encoder.encode_to_frame(&tensor)?;
        let slots_filled = frame.active_slot_count();

        Ok(TranslateOutput {
            frame,
            token_count,
            slots_filled,
        })
    }

    fn decode(&self, frame: &TensorFrame) -> Result<String, VoltError> {
        // Collect all slot vectors into a 3D tensor [1, 16, 256]
        let mut data = vec![0.0f32; MAX_SLOTS * SLOT_DIM];
        for i in 0..MAX_SLOTS {
            if let Some(slot) = &frame.slots[i]
                && let Some(vec) = &slot.resolutions[0]
            {
                data[i * SLOT_DIM..(i + 1) * SLOT_DIM].copy_from_slice(vec);
            }
        }

        let context = Tensor::from_vec(data, (1, MAX_SLOTS, SLOT_DIM), &Device::Cpu)
            .map_err(|e| VoltError::Internal {
                message: format!("tensor creation failed: {e}"),
            })?;

        // Generate tokens (decoder cross-attends to 16 slot vectors)
        let token_ids = self.decoder.generate(&context, 128)?;

        // Decode back to text
        let text = self.tokenizer.decode(&token_ids, true).map_err(|e| {
            VoltError::Internal {
                message: format!("detokenization failed: {e}"),
            }
        })?;

        Ok(text)
    }

    fn decode_slots(
        &self,
        frame: &TensorFrame,
    ) -> Result<Vec<(usize, SlotRole, String)>, VoltError> {
        let mut result = Vec::new();

        for i in 0..MAX_SLOTS {
            if let Some(slot) = &frame.slots[i]
                && slot.resolutions[0].is_some()
            {
                let role = slot.role;
                let certainty = frame.meta[i].certainty;
                let desc = format!("[{role:?} γ={certainty:.2}]");
                result.push((i, role, desc));
            }
        }

        Ok(result)
    }
}
