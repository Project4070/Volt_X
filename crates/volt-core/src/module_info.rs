//! Module metadata types for the Volt X ecosystem.
//!
//! Every pluggable module (Translator, HardStrand, ActionCore) can describe
//! itself via a [`ModuleInfo`] struct. This metadata enables runtime
//! introspection, module registry listings, and CLI tooling.
//!
//! These types live in `volt-core` so all crates can use them without
//! circular dependencies.

/// The category of module (which trait it implements).
///
/// # Example
///
/// ```
/// use volt_core::module_info::ModuleType;
///
/// let ty = ModuleType::HardStrand;
/// assert_ne!(ty, ModuleType::Translator);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModuleType {
    /// Implements the `Translator` trait (input encoding / decoding).
    Translator,
    /// Implements the `HardStrand` trait (CPU-side deterministic processing).
    HardStrand,
    /// Implements the `ActionCore` trait (output modality decoding).
    ActionCore,
}

impl core::fmt::Display for ModuleType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Translator => write!(f, "Translator"),
            Self::HardStrand => write!(f, "HardStrand"),
            Self::ActionCore => write!(f, "ActionCore"),
        }
    }
}

/// Metadata describing a Volt module.
///
/// Every `Translator`, `HardStrand`, and `ActionCore` implementation
/// can optionally return a `ModuleInfo` via the `info()` default method
/// on its trait.
///
/// # Example
///
/// ```
/// use volt_core::module_info::{ModuleInfo, ModuleType};
///
/// let info = ModuleInfo {
///     id: "volt-strand-weather".to_string(),
///     display_name: "Weather Strand".to_string(),
///     version: "0.1.0".to_string(),
///     author: "Volt X Team".to_string(),
///     description: "Provides mock weather data for demonstration.".to_string(),
///     module_type: ModuleType::HardStrand,
/// };
/// assert_eq!(info.module_type, ModuleType::HardStrand);
/// ```
#[derive(Debug, Clone)]
pub struct ModuleInfo {
    /// Unique identifier (typically the crate name), e.g. `"volt-strand-weather"`.
    pub id: String,
    /// Human-readable display name, e.g. `"Weather Strand"`.
    pub display_name: String,
    /// Semantic version string, e.g. `"0.1.0"`.
    pub version: String,
    /// Module author(s).
    pub author: String,
    /// Short description of what this module does.
    pub description: String,
    /// Which trait this module implements.
    pub module_type: ModuleType,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn module_info_construction() {
        let info = ModuleInfo {
            id: "test-module".to_string(),
            display_name: "Test".to_string(),
            version: "0.1.0".to_string(),
            author: "Author".to_string(),
            description: "A test module.".to_string(),
            module_type: ModuleType::HardStrand,
        };
        assert_eq!(info.id, "test-module");
        assert_eq!(info.module_type, ModuleType::HardStrand);
    }

    #[test]
    fn module_type_display() {
        assert_eq!(ModuleType::Translator.to_string(), "Translator");
        assert_eq!(ModuleType::HardStrand.to_string(), "HardStrand");
        assert_eq!(ModuleType::ActionCore.to_string(), "ActionCore");
    }

    #[test]
    fn module_type_equality() {
        assert_eq!(ModuleType::HardStrand, ModuleType::HardStrand);
        assert_ne!(ModuleType::Translator, ModuleType::ActionCore);
    }

    #[test]
    fn module_info_clone() {
        let info = ModuleInfo {
            id: "clone-test".to_string(),
            display_name: "Clone".to_string(),
            version: "1.0.0".to_string(),
            author: "Author".to_string(),
            description: "Test cloning.".to_string(),
            module_type: ModuleType::ActionCore,
        };
        let cloned = info.clone();
        assert_eq!(cloned.id, info.id);
        assert_eq!(cloned.module_type, info.module_type);
    }
}
