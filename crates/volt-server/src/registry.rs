//! Module registry — discovers and manages installed Volt modules.
//!
//! At startup, the registry scans for modules that are compiled in via
//! feature flags and records their metadata. The registry is read-only
//! after construction (modules are compile-time, not runtime plugins).
//!
//! ## Discovery Mechanism
//!
//! Modules are included via Cargo feature flags. At build time, `#[cfg]`
//! blocks conditionally compile module code. The registry mirrors this
//! by recording which modules are present.
//!
//! Built-in modules (MathEngine, HDCAlgebra, StubTranslator, TextAction)
//! are always registered.

use volt_core::module_info::{ModuleInfo, ModuleType};

/// Registry of all installed Volt modules.
///
/// Constructed once at startup via [`discover()`](ModuleRegistry::discover),
/// then shared read-only across the application.
///
/// # Example
///
/// ```
/// use volt_server::registry::ModuleRegistry;
///
/// let registry = ModuleRegistry::discover();
/// assert!(registry.module_count() >= 4); // built-in modules
/// ```
#[derive(Debug, Clone)]
pub struct ModuleRegistry {
    modules: Vec<ModuleInfo>,
}

impl ModuleRegistry {
    /// Discover all available modules at compile time.
    ///
    /// Registers built-in modules and any feature-gated community modules.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_server::registry::ModuleRegistry;
    ///
    /// let registry = ModuleRegistry::discover();
    /// assert!(registry.is_installed("math_engine"));
    /// assert!(registry.is_installed("hdc_algebra"));
    /// assert!(registry.is_installed("stub_translator"));
    /// assert!(registry.is_installed("text_action"));
    /// ```
    pub fn discover() -> Self {
        #[allow(unused_mut)]
        let mut modules = vec![
            // Built-in Hard Strands (always present)
            ModuleInfo {
                id: "math_engine".to_string(),
                display_name: "Math Engine".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
                author: "Volt X Team".to_string(),
                description: "Exact arithmetic: add, sub, mul, div, pow, sqrt, abs, neg."
                    .to_string(),
                module_type: ModuleType::HardStrand,
            },
            ModuleInfo {
                id: "hdc_algebra".to_string(),
                display_name: "HDC Algebra".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
                author: "Volt X Team".to_string(),
                description: "Compositional reasoning via HDC bind/unbind/superpose/permute."
                    .to_string(),
                module_type: ModuleType::HardStrand,
            },
            // Built-in Translator (always present)
            ModuleInfo {
                id: "stub_translator".to_string(),
                display_name: "Stub Translator".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
                author: "Volt X Team".to_string(),
                description: "Heuristic word-to-slot mapping with deterministic hash vectors."
                    .to_string(),
                module_type: ModuleType::Translator,
            },
            // Built-in ActionCore (always present)
            ModuleInfo {
                id: "text_action".to_string(),
                display_name: "Text Action".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
                author: "Volt X Team".to_string(),
                description: "Default text output: decode TensorFrame slots to UTF-8.".to_string(),
                module_type: ModuleType::ActionCore,
            },
        ];

        // CodeRunner (behind sandbox feature)
        #[cfg(feature = "sandbox")]
        modules.push(ModuleInfo {
            id: "code_runner".to_string(),
            display_name: "Code Runner".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            author: "Volt X Team".to_string(),
            description: "WASM-sandboxed code execution with fuel limits.".to_string(),
            module_type: ModuleType::HardStrand,
        });

        // WeatherStrand (behind weather feature)
        #[cfg(feature = "weather")]
        modules.push(ModuleInfo {
            id: "volt-strand-weather".to_string(),
            display_name: "Weather Strand".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            author: "Volt X Team".to_string(),
            description: "Example community module providing mock weather data.".to_string(),
            module_type: ModuleType::HardStrand,
        });

        // LLM Translator (behind volt-translate/llm feature)
        #[cfg(feature = "llm")]
        modules.push(ModuleInfo {
            id: "llm_translator".to_string(),
            display_name: "LLM Translator".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            author: "Volt X Team".to_string(),
            description: "Qwen3-0.6B backbone with projection head + VQ-VAE codebook."
                .to_string(),
            module_type: ModuleType::Translator,
        });

        Self { modules }
    }

    /// Total number of registered modules.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_server::registry::ModuleRegistry;
    ///
    /// let registry = ModuleRegistry::discover();
    /// assert!(registry.module_count() >= 4);
    /// ```
    pub fn module_count(&self) -> usize {
        self.modules.len()
    }

    /// Check if a module with the given ID is installed.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_server::registry::ModuleRegistry;
    ///
    /// let registry = ModuleRegistry::discover();
    /// assert!(registry.is_installed("math_engine"));
    /// assert!(!registry.is_installed("nonexistent_module"));
    /// ```
    pub fn is_installed(&self, module_id: &str) -> bool {
        self.modules.iter().any(|m| m.id == module_id)
    }

    /// List all registered modules.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_server::registry::ModuleRegistry;
    ///
    /// let registry = ModuleRegistry::discover();
    /// let modules = registry.list_modules();
    /// assert!(modules.iter().any(|m| m.id == "math_engine"));
    /// ```
    pub fn list_modules(&self) -> &[ModuleInfo] {
        &self.modules
    }

    /// List modules of a specific type.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_server::registry::ModuleRegistry;
    /// use volt_core::module_info::ModuleType;
    ///
    /// let registry = ModuleRegistry::discover();
    /// let strands: Vec<_> = registry.list_by_type(ModuleType::HardStrand);
    /// assert!(strands.len() >= 2); // MathEngine + HDCAlgebra
    /// ```
    pub fn list_by_type(&self, module_type: ModuleType) -> Vec<&ModuleInfo> {
        self.modules
            .iter()
            .filter(|m| m.module_type == module_type)
            .collect()
    }
}

impl Default for ModuleRegistry {
    fn default() -> Self {
        Self::discover()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discover_finds_built_in_modules() {
        let registry = ModuleRegistry::discover();
        assert!(registry.is_installed("math_engine"));
        assert!(registry.is_installed("hdc_algebra"));
        assert!(registry.is_installed("stub_translator"));
        assert!(registry.is_installed("text_action"));
    }

    #[test]
    fn discover_at_least_four_modules() {
        let registry = ModuleRegistry::discover();
        assert!(
            registry.module_count() >= 4,
            "expected >= 4 modules, got {}",
            registry.module_count()
        );
    }

    #[test]
    fn nonexistent_module_not_installed() {
        let registry = ModuleRegistry::discover();
        assert!(!registry.is_installed("nonexistent_module_xyz"));
    }

    #[test]
    fn list_by_type_hard_strand() {
        let registry = ModuleRegistry::discover();
        let strands = registry.list_by_type(ModuleType::HardStrand);
        assert!(strands.len() >= 2);
        assert!(strands.iter().any(|m| m.id == "math_engine"));
    }

    #[test]
    fn list_by_type_translator() {
        let registry = ModuleRegistry::discover();
        let translators = registry.list_by_type(ModuleType::Translator);
        assert!(translators.iter().any(|m| m.id == "stub_translator"));
    }

    #[test]
    fn list_by_type_action_core() {
        let registry = ModuleRegistry::discover();
        let actions = registry.list_by_type(ModuleType::ActionCore);
        assert!(actions.iter().any(|m| m.id == "text_action"));
    }
}
