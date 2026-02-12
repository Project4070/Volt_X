//! The [`ActionCore`] trait — pluggable output modules for Volt X.
//!
//! Action cores convert verified TensorFrames into human-consumable output
//! in various modalities (text, audio, image, etc.). They are the output
//! counterpart to [`Translator`](crate::Translator) (input).
//!
//! ## Building an ActionCore Module
//!
//! 1. Implement [`ActionCore`] for your struct.
//! 2. Return the supported modalities from [`supported_modalities()`](ActionCore::supported_modalities).
//! 3. Implement [`execute()`](ActionCore::execute) to decode a frame into your output format.
//! 4. Optionally implement [`info()`](ActionCore::info) to provide module metadata.
//! 5. Register with the server's module registry.

use volt_core::{ModuleInfo, TensorFrame, VoltError};

/// The modality of output produced by an [`ActionCore`].
///
/// # Example
///
/// ```
/// use volt_translate::action_core::OutputModality;
///
/// let m = OutputModality::Text;
/// assert_eq!(m, OutputModality::Text);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OutputModality {
    /// Plain UTF-8 text output.
    Text,
    /// Audio output (WAV/PCM bytes).
    Audio,
    /// Image output (PNG/JPEG bytes).
    Image,
    /// Structured data (JSON-compatible bytes).
    StructuredData,
    /// Custom modality with a string identifier.
    Custom(String),
}

impl core::fmt::Display for OutputModality {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Text => write!(f, "Text"),
            Self::Audio => write!(f, "Audio"),
            Self::Image => write!(f, "Image"),
            Self::StructuredData => write!(f, "StructuredData"),
            Self::Custom(name) => write!(f, "Custom({name})"),
        }
    }
}

/// The output of an [`ActionCore::execute`] call.
///
/// # Example
///
/// ```
/// use volt_translate::action_core::{ActionOutput, OutputModality};
///
/// let output = ActionOutput {
///     modality: OutputModality::Text,
///     data: b"Hello, world!".to_vec(),
///     description: "Decoded text from frame".to_string(),
/// };
/// assert_eq!(output.modality, OutputModality::Text);
/// assert_eq!(std::str::from_utf8(&output.data).unwrap(), "Hello, world!");
/// ```
#[derive(Debug, Clone)]
pub struct ActionOutput {
    /// The modality of this output.
    pub modality: OutputModality,
    /// The raw output bytes (UTF-8 for text, binary for audio/image).
    pub data: Vec<u8>,
    /// Human-readable description of what was produced.
    pub description: String,
}

/// Trait for output modules that convert TensorFrames into
/// human-consumable output.
///
/// Action cores are the output counterpart to Translators (input).
/// Each action core supports one or more output modalities and
/// decodes a verified TensorFrame into that modality.
///
/// # Example
///
/// ```
/// use volt_translate::action_core::{ActionCore, TextAction};
///
/// let action = TextAction::new();
/// assert_eq!(action.name(), "text_action");
/// assert_eq!(action.supported_modalities().len(), 1);
/// ```
pub trait ActionCore: Send + Sync {
    /// Human-readable name for this action core (e.g., `"text_action"`).
    fn name(&self) -> &str;

    /// Execute the action core: decode a verified TensorFrame into output.
    ///
    /// # Errors
    ///
    /// Returns `Err(VoltError)` if decoding fails.
    fn execute(&self, frame: &TensorFrame) -> Result<ActionOutput, VoltError>;

    /// List the output modalities this core supports.
    fn supported_modalities(&self) -> Vec<OutputModality>;

    /// Optional metadata about this module.
    ///
    /// Returns `None` by default. Community modules should override
    /// this to provide introspectable metadata for the module registry.
    fn info(&self) -> Option<ModuleInfo> {
        None
    }
}

/// Default text-output action core.
///
/// Decodes a TensorFrame into plain UTF-8 text by iterating over
/// active slots and concatenating their role-based decoded words.
/// This is the built-in action core used by the server's `/api/think`
/// endpoint.
///
/// # Example
///
/// ```
/// use volt_translate::action_core::{ActionCore, TextAction, OutputModality};
/// use volt_core::{TensorFrame, SlotData, SlotRole, SLOT_DIM};
///
/// std::thread::Builder::new().stack_size(4 * 1024 * 1024).spawn(|| {
///     let action = TextAction::new();
///     let frame = TensorFrame::new();
///     let output = action.execute(&frame).unwrap();
///     assert_eq!(output.modality, OutputModality::Text);
/// }).unwrap().join().unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct TextAction;

impl TextAction {
    /// Create a new `TextAction` instance.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_translate::action_core::{ActionCore, TextAction};
    ///
    /// let action = TextAction::new();
    /// assert_eq!(action.name(), "text_action");
    /// ```
    pub fn new() -> Self {
        Self
    }
}

impl Default for TextAction {
    fn default() -> Self {
        Self::new()
    }
}

impl ActionCore for TextAction {
    fn name(&self) -> &str {
        "text_action"
    }

    fn execute(&self, frame: &TensorFrame) -> Result<ActionOutput, VoltError> {
        // Collect active slots and their roles into a simple text output.
        let mut parts = Vec::new();
        for (i, slot_opt) in frame.slots.iter().enumerate() {
            if let Some(slot) = slot_opt {
                let role = &slot.role;
                let gamma = frame.meta[i].certainty;
                parts.push(format!("[{role:?} S{i} γ={gamma:.2}]"));

                // If the slot has R0 data, show the first few non-zero dims
                if let Some(r0) = &slot.resolutions[0] {
                    let significant: Vec<String> = r0
                        .iter()
                        .take(8)
                        .enumerate()
                        .filter(|(_, v)| v.abs() > 1e-6)
                        .map(|(j, v)| format!("d{j}={v:.3}"))
                        .collect();
                    if !significant.is_empty() {
                        parts.push(significant.join(", "));
                    }
                }
            }
        }

        let text = if parts.is_empty() {
            "[empty frame]".to_string()
        } else {
            parts.join(" ")
        };

        Ok(ActionOutput {
            modality: OutputModality::Text,
            data: text.into_bytes(),
            description: format!("Text decode of {} active slot(s)", frame.active_slot_count()),
        })
    }

    fn supported_modalities(&self) -> Vec<OutputModality> {
        vec![OutputModality::Text]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use volt_core::{SlotData, SlotRole, SLOT_DIM};

    const TEST_STACK: usize = 4 * 1024 * 1024;

    #[test]
    fn text_action_name() {
        let action = TextAction::new();
        assert_eq!(action.name(), "text_action");
    }

    #[test]
    fn text_action_modalities() {
        let action = TextAction::new();
        let mods = action.supported_modalities();
        assert_eq!(mods.len(), 1);
        assert_eq!(mods[0], OutputModality::Text);
    }

    #[test]
    fn text_action_info_is_none() {
        let action = TextAction::new();
        assert!(action.info().is_none());
    }

    #[test]
    fn text_action_empty_frame() {
        std::thread::Builder::new()
            .stack_size(TEST_STACK)
            .spawn(|| {
                let action = TextAction::new();
                let frame = TensorFrame::new();
                let output = action.execute(&frame).unwrap();
                assert_eq!(output.modality, OutputModality::Text);
                let text = String::from_utf8(output.data).unwrap();
                assert_eq!(text, "[empty frame]");
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn text_action_with_slots() {
        std::thread::Builder::new()
            .stack_size(TEST_STACK)
            .spawn(|| {
                let action = TextAction::new();
                let mut frame = TensorFrame::new();
                let mut slot = SlotData::new(SlotRole::Agent);
                let mut data = [0.0_f32; SLOT_DIM];
                data[0] = 1.0;
                slot.write_resolution(0, data);
                frame.write_slot(0, slot).unwrap();
                frame.meta[0].certainty = 0.8;

                let output = action.execute(&frame).unwrap();
                let text = String::from_utf8(output.data).unwrap();
                assert!(text.contains("Agent"));
                assert!(text.contains("0.80"));
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn output_modality_display() {
        assert_eq!(OutputModality::Text.to_string(), "Text");
        assert_eq!(OutputModality::Audio.to_string(), "Audio");
        assert_eq!(
            OutputModality::Custom("Braille".into()).to_string(),
            "Custom(Braille)"
        );
    }

    #[test]
    fn text_action_default() {
        let action = TextAction::default();
        assert_eq!(action.name(), "text_action");
    }
}
