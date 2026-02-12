//! Integration tests for Milestone 6.1: Module Registry + Server Endpoints.
//!
//! Tests:
//! 1. ModuleRegistry::discover() finds built-in modules.
//! 2. GET /api/modules returns module list as JSON.
//! 3. Feature-gated: weather strand appears when weather feature enabled.

use volt_core::module_info::ModuleType;
use volt_server::registry::ModuleRegistry;

// --------------------------------------------------------------------------
// Test: Module registry discovers built-in modules
// --------------------------------------------------------------------------

#[test]
fn registry_discovers_built_in_modules() {
    let registry = ModuleRegistry::discover();
    assert!(registry.is_installed("math_engine"));
    assert!(registry.is_installed("hdc_algebra"));
    assert!(registry.is_installed("stub_translator"));
    assert!(registry.is_installed("text_action"));
}

#[test]
fn registry_has_minimum_module_count() {
    let registry = ModuleRegistry::discover();
    // At minimum: math_engine, hdc_algebra, stub_translator, text_action
    assert!(
        registry.module_count() >= 4,
        "expected >= 4 built-in modules, got {}",
        registry.module_count()
    );
}

#[test]
fn registry_list_by_type_hard_strand() {
    let registry = ModuleRegistry::discover();
    let strands = registry.list_by_type(ModuleType::HardStrand);
    assert!(
        strands.len() >= 2,
        "expected >= 2 HardStrand modules, got {}",
        strands.len()
    );
}

#[test]
fn registry_list_by_type_translator() {
    let registry = ModuleRegistry::discover();
    let translators = registry.list_by_type(ModuleType::Translator);
    assert!(!translators.is_empty());
    assert!(translators.iter().any(|m| m.id == "stub_translator"));
}

#[test]
fn registry_list_by_type_action_core() {
    let registry = ModuleRegistry::discover();
    let actions = registry.list_by_type(ModuleType::ActionCore);
    assert!(!actions.is_empty());
    assert!(actions.iter().any(|m| m.id == "text_action"));
}

#[test]
fn registry_nonexistent_module() {
    let registry = ModuleRegistry::discover();
    assert!(!registry.is_installed("volt-strand-nonexistent"));
}

// --------------------------------------------------------------------------
// Test: Weather strand appears in registry when feature enabled
// --------------------------------------------------------------------------

#[cfg(feature = "weather")]
#[test]
fn registry_includes_weather_when_feature_enabled() {
    let registry = ModuleRegistry::discover();
    assert!(
        registry.is_installed("volt-strand-weather"),
        "weather strand should be registered when weather feature is enabled"
    );
}

#[cfg(not(feature = "weather"))]
#[test]
fn registry_excludes_weather_when_feature_disabled() {
    let registry = ModuleRegistry::discover();
    assert!(
        !registry.is_installed("volt-strand-weather"),
        "weather strand should NOT be registered when weather feature is disabled"
    );
}

// --------------------------------------------------------------------------
// Test: GET /api/modules returns JSON (uses tower::ServiceExt for testing)
// --------------------------------------------------------------------------

#[tokio::test]
async fn api_modules_returns_json() {
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    let app = volt_server::build_app();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/modules")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let modules: Vec<volt_server::models::ModuleResponse> =
        serde_json::from_slice(&body).unwrap();

    // Should have at least the 4 built-in modules
    assert!(
        modules.len() >= 4,
        "expected >= 4 modules in response, got {}",
        modules.len()
    );

    // Verify math_engine is present
    assert!(
        modules.iter().any(|m| m.id == "math_engine"),
        "math_engine should be in the response"
    );

    // Verify module_type field is set
    assert!(
        modules
            .iter()
            .all(|m| ["Translator", "HardStrand", "ActionCore"].contains(&m.module_type.as_str())),
        "all modules should have a valid module_type"
    );
}
