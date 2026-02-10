//! Frozen LLM backbone for hidden state extraction.
//!
//! Loads Qwen3-0.6B (via the candle-transformers Qwen2 module) and
//! extracts the last hidden state before the LM head, producing
//! `[seq_len, 1024]` tensors for the projection head.
//!
//! # Mock Mode
//!
//! For testing without model files, [`LlmBackbone::mock`] creates a
//! lightweight backbone that returns deterministic hidden states based
//! on token positions (no model weights needed).

use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen2 as qwen2;
use volt_core::VoltError;

use super::projection::candle_to_volt;

/// Internal representation: real Qwen2 model or lightweight mock.
enum BackboneInner {
    /// Real Qwen3-0.6B loaded from safetensors.
    Real {
        model: Box<qwen2::Model>,
        tokenizer: Box<tokenizers::Tokenizer>,
        config: qwen2::Config,
        device: Device,
    },
    /// Mock backbone for testing (no model files needed).
    Mock { hidden_dim: usize, device: Device },
}

/// Frozen LLM backbone for extracting hidden states from text.
///
/// Wraps a Qwen3-0.6B model (loaded via candle-transformers' Qwen2
/// module) and its tokenizer. The backbone is frozen — no gradient
/// computation, no weight updates.
///
/// # Example
///
/// ```
/// use volt_translate::llm::backbone::LlmBackbone;
/// use candle_core::Device;
///
/// let mut backbone = LlmBackbone::mock(64, &Device::Cpu);
/// let tokens = backbone.tokenize("hello world").unwrap();
/// let hidden = backbone.extract_hidden_states(&tokens).unwrap();
/// assert_eq!(hidden.dims(), &[2, 64]);
/// ```
// Note: Debug implemented manually, Clone omitted because candle's
// Model and tokenizers::Tokenizer are heavyweight ML objects.
pub struct LlmBackbone {
    inner: BackboneInner,
}

impl std::fmt::Debug for LlmBackbone {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.inner {
            BackboneInner::Real { config, .. } => f
                .debug_struct("LlmBackbone")
                .field("mode", &"real")
                .field("hidden_dim", &config.hidden_size)
                .finish(),
            BackboneInner::Mock { hidden_dim, .. } => f
                .debug_struct("LlmBackbone")
                .field("mode", &"mock")
                .field("hidden_dim", hidden_dim)
                .finish(),
        }
    }
}

impl LlmBackbone {
    /// Loads a Qwen3-0.6B backbone from a HuggingFace model directory.
    ///
    /// The directory must contain:
    /// - `config.json` — Qwen2/Qwen3 model configuration
    /// - `*.safetensors` — model weights (single or sharded)
    /// - `tokenizer.json` — BPE tokenizer
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::TranslateError`] if any file is missing,
    /// cannot be parsed, or weight shapes are inconsistent.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use volt_translate::llm::backbone::LlmBackbone;
    /// use candle_core::Device;
    /// use std::path::Path;
    ///
    /// let mut backbone = LlmBackbone::load(
    ///     Path::new("models/Qwen3-0.6B"),
    ///     &Device::Cpu,
    /// ).unwrap();
    /// ```
    pub fn load(model_dir: &Path, device: &Device) -> Result<Self, VoltError> {
        // Load config.json
        let config_path = model_dir.join("config.json");
        let config_str =
            std::fs::read_to_string(&config_path).map_err(|e| VoltError::TranslateError {
                message: format!(
                    "failed to read model config from {}: {e}",
                    config_path.display()
                ),
            })?;
        let config: qwen2::Config =
            serde_json::from_str(&config_str).map_err(|e| VoltError::TranslateError {
                message: format!("failed to parse model config: {e}"),
            })?;

        // Find and load safetensors weights
        let safetensors_files = find_safetensors_files(model_dir)?;
        // SAFETY: The safetensors files are only read during initialization
        // and not modified while the memory map is alive.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&safetensors_files, DType::F32, device)
                .map_err(|e| VoltError::TranslateError {
                    message: format!("failed to load model weights: {e}"),
                })?
        };
        let model = Box::new(qwen2::Model::new(&config, vb).map_err(candle_to_volt)?);

        // Load tokenizer.json
        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer =
            Box::new(tokenizers::Tokenizer::from_file(&tokenizer_path).map_err(|e| {
                VoltError::TranslateError {
                    message: format!(
                        "failed to load tokenizer from {}: {e}",
                        tokenizer_path.display()
                    ),
                }
            })?);

        Ok(Self {
            inner: BackboneInner::Real {
                model,
                tokenizer,
                config,
                device: device.clone(),
            },
        })
    }

    /// Creates a mock backbone for testing without model files.
    ///
    /// The mock backbone:
    /// - Tokenizes by splitting on whitespace (sequential 1-based IDs)
    /// - Returns deterministic hidden states based on token positions
    ///
    /// # Example
    ///
    /// ```
    /// use volt_translate::llm::backbone::LlmBackbone;
    /// use candle_core::Device;
    ///
    /// let mut backbone = LlmBackbone::mock(64, &Device::Cpu);
    /// let tokens = backbone.tokenize("hello world").unwrap();
    /// assert_eq!(tokens.len(), 2);
    /// let hidden = backbone.extract_hidden_states(&tokens).unwrap();
    /// assert_eq!(hidden.dims(), &[2, 64]);
    /// ```
    pub fn mock(hidden_dim: usize, device: &Device) -> Self {
        Self {
            inner: BackboneInner::Mock {
                hidden_dim,
                device: device.clone(),
            },
        }
    }

    /// Tokenizes input text into token IDs.
    ///
    /// In real mode, uses the Qwen3 BPE tokenizer (151K vocab).
    /// In mock mode, splits on whitespace with sequential IDs.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::TranslateError`] if tokenization fails
    /// or the input is empty/whitespace-only.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_translate::llm::backbone::LlmBackbone;
    /// use candle_core::Device;
    ///
    /// let backbone = LlmBackbone::mock(64, &Device::Cpu);
    /// let tokens = backbone.tokenize("the cat sat").unwrap();
    /// assert_eq!(tokens.len(), 3);
    /// assert!(backbone.tokenize("").is_err());
    /// ```
    pub fn tokenize(&self, text: &str) -> Result<Vec<u32>, VoltError> {
        if text.trim().is_empty() {
            return Err(VoltError::TranslateError {
                message: "cannot tokenize empty input".into(),
            });
        }

        match &self.inner {
            BackboneInner::Real { tokenizer, .. } => {
                let encoding =
                    tokenizer
                        .encode(text, true)
                        .map_err(|e| VoltError::TranslateError {
                            message: format!("tokenization failed: {e}"),
                        })?;
                Ok(encoding.get_ids().to_vec())
            }
            BackboneInner::Mock { .. } => Ok(text
                .split_whitespace()
                .enumerate()
                .map(|(i, _)| i as u32 + 1)
                .collect()),
        }
    }

    /// Extracts the last hidden state from the frozen backbone.
    ///
    /// Returns a tensor of shape `[seq_len, hidden_dim]` suitable
    /// for the [`FrameProjectionHead`](super::projection::FrameProjectionHead).
    ///
    /// In mock mode, returns deterministic values derived from token
    /// positions (sin-based hash for reproducibility).
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::TranslateError`] on computation error.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_translate::llm::backbone::LlmBackbone;
    /// use candle_core::Device;
    ///
    /// let mut backbone = LlmBackbone::mock(64, &Device::Cpu);
    /// let tokens = vec![1, 2, 3];
    /// let hidden = backbone.extract_hidden_states(&tokens).unwrap();
    /// assert_eq!(hidden.dims(), &[3, 64]);
    /// ```
    pub fn extract_hidden_states(&mut self, token_ids: &[u32]) -> Result<Tensor, VoltError> {
        match &mut self.inner {
            BackboneInner::Real { model, device, .. } => {
                model.clear_kv_cache();
                let input = Tensor::new(token_ids, &*device)
                    .map_err(candle_to_volt)?
                    .unsqueeze(0)
                    .map_err(candle_to_volt)?;
                let hidden = model.forward(&input, 0, None).map_err(candle_to_volt)?;
                // [1, seq_len, hidden_size] → [seq_len, hidden_size]
                hidden.squeeze(0).map_err(candle_to_volt)
            }
            BackboneInner::Mock { hidden_dim, device } => {
                let seq_len = token_ids.len();
                let hd = *hidden_dim;
                let mut data = vec![0.0_f32; seq_len * hd];
                for (t, &tid) in token_ids.iter().enumerate() {
                    for d in 0..hd {
                        // Deterministic pseudo-random based on token ID and dimension
                        data[t * hd + d] =
                            ((tid as f32 + 1.0) * (d as f32 + 1.0)).sin() * 0.1;
                    }
                }
                Tensor::from_vec(data, (seq_len, hd), &*device).map_err(candle_to_volt)
            }
        }
    }

    /// Returns the hidden dimension of the backbone.
    ///
    /// For Qwen3-0.6B this is 1024. For mock backbones, returns the
    /// configured dimension.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_translate::llm::backbone::LlmBackbone;
    /// use candle_core::Device;
    ///
    /// let backbone = LlmBackbone::mock(128, &Device::Cpu);
    /// assert_eq!(backbone.hidden_dim(), 128);
    /// ```
    pub fn hidden_dim(&self) -> usize {
        match &self.inner {
            BackboneInner::Real { config, .. } => config.hidden_size,
            BackboneInner::Mock { hidden_dim, .. } => *hidden_dim,
        }
    }
}

/// Finds all safetensors files in a model directory.
///
/// Checks for a single `model.safetensors` first, then falls back
/// to collecting all `*.safetensors` files (sorted for sharded models).
fn find_safetensors_files(dir: &Path) -> Result<Vec<PathBuf>, VoltError> {
    let single = dir.join("model.safetensors");
    if single.exists() {
        return Ok(vec![single]);
    }

    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| VoltError::TranslateError {
            message: format!("failed to read model directory {}: {e}", dir.display()),
        })?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("safetensors"))
        .collect();

    files.sort();

    if files.is_empty() {
        return Err(VoltError::TranslateError {
            message: format!("no safetensors files found in {}", dir.display()),
        });
    }

    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_tokenize() {
        let backbone = LlmBackbone::mock(64, &Device::Cpu);
        let tokens = backbone.tokenize("the cat sat").unwrap();
        assert_eq!(tokens, vec![1, 2, 3]);
    }

    #[test]
    fn mock_empty_input_errors() {
        let backbone = LlmBackbone::mock(64, &Device::Cpu);
        assert!(backbone.tokenize("").is_err());
        assert!(backbone.tokenize("   ").is_err());
    }

    #[test]
    fn mock_hidden_states_shape() {
        let mut backbone = LlmBackbone::mock(64, &Device::Cpu);
        let tokens = backbone.tokenize("hello world").unwrap();
        let hidden = backbone.extract_hidden_states(&tokens).unwrap();
        assert_eq!(hidden.dims(), &[2, 64]);
    }

    #[test]
    fn mock_hidden_dim() {
        let backbone = LlmBackbone::mock(128, &Device::Cpu);
        assert_eq!(backbone.hidden_dim(), 128);
    }

    #[test]
    fn mock_deterministic() {
        let mut backbone = LlmBackbone::mock(64, &Device::Cpu);
        let tokens = backbone.tokenize("test input").unwrap();
        let h1 = backbone.extract_hidden_states(&tokens).unwrap();
        let h2 = backbone.extract_hidden_states(&tokens).unwrap();

        let diff = (&h1 - &h2)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_vec0::<f32>()
            .unwrap();
        assert!(diff < 1e-6, "mock backbone should be deterministic");
    }

    #[test]
    fn debug_display() {
        let backbone = LlmBackbone::mock(64, &Device::Cpu);
        let debug = format!("{backbone:?}");
        assert!(debug.contains("mock"));
        assert!(debug.contains("64"));
    }
}
