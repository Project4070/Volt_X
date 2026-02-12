//! Integration tests for Milestone 6.1: Trait Specification + Module Hot-Plug.
//!
//! Tests:
//! 1. Weather strand activates on matching capability vector.
//! 2. Weather strand returns correct deterministic mock data.
//! 3. Weather strand info() returns valid ModuleInfo.
//! 4. Router with panicking strand catches panic, returns Ok.
//! 5. Router unregister() removes strand, queries fall through.
//! 6. Router strand_names() returns expected list.
//! 7. HardStrand info() default is None (backward compat).
//! 8. Trait documentation: all three traits are importable and usable.

use volt_core::{SlotData, SlotRole, TensorFrame, VoltError, SLOT_DIM};
use volt_hard::math_engine::MathEngine;
use volt_hard::router::IntentRouter;
use volt_hard::strand::{HardStrand, StrandResult};

const TEST_STACK: usize = 4 * 1024 * 1024;

/// Helper: build a frame with a capability vector in the Predicate slot
/// and operation data in the Instrument slot.
fn build_weather_frame(city_hash: f32) -> TensorFrame {
    // Import at runtime since the weather feature may or may not be enabled.
    // We create the vector manually using the same seed as WeatherStrand.
    let cap = weather_capability_vector();

    let mut frame = TensorFrame::new();

    // Predicate slot tagged with weather capability
    let mut pred = SlotData::new(SlotRole::Predicate);
    pred.write_resolution(0, cap);
    frame.write_slot(1, pred).unwrap();
    frame.meta[1].certainty = 0.8;

    // Instrument slot with weather op code
    let mut inst = SlotData::new(SlotRole::Instrument);
    let mut data = [0.0_f32; SLOT_DIM];
    data[0] = 20.0; // OP_WEATHER
    data[1] = city_hash;
    inst.write_resolution(0, data);
    frame.write_slot(6, inst).unwrap();
    frame.meta[6].certainty = 0.9;

    frame
}

/// Reproduce the weather capability vector using the same seed.
fn weather_capability_vector() -> [f32; SLOT_DIM] {
    const WEATHER_SEED: u64 = 0x5745_4154_4845_5231;
    let mut v = [0.0_f32; SLOT_DIM];
    for (i, val) in v.iter_mut().enumerate() {
        let mut h = WEATHER_SEED.wrapping_mul(0xd2b7_4407_b1ce_6e93);
        h = h.wrapping_add(i as u64);
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
        h ^= h >> 33;
        h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
        h ^= h >> 33;
        *val = ((h as f64 / u64::MAX as f64) * 2.0 - 1.0) as f32;
    }
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

// --------------------------------------------------------------------------
// Test: Weather strand activates and returns correct data
// --------------------------------------------------------------------------

#[cfg(feature = "weather")]
#[test]
fn weather_strand_activates_on_capability_match() {
    std::thread::Builder::new()
        .stack_size(TEST_STACK)
        .spawn(|| {
            use volt_hard::weather_strand::WeatherStrand;

            let mut router = IntentRouter::new();
            router.register(Box::new(MathEngine::new()));
            router.register(Box::new(WeatherStrand::new()));

            let frame = build_weather_frame(42.0);
            let result = router.route(&frame).unwrap();

            // Should activate the weather strand (not math)
            let activated = result.decisions.iter().find(|d| d.activated);
            assert!(
                activated.is_some(),
                "weather frame should activate a strand"
            );
            assert_eq!(activated.unwrap().strand_name, "weather");

            // Check result slot: city_hash=42 → 42%5=2 → profile (35, 30, 5)
            let r = result.frame.read_slot(8).unwrap();
            let r0 = r.resolutions[0].unwrap();
            assert!(
                (r0[0] - 35.0).abs() < 0.01,
                "temp should be 35°C, got {}",
                r0[0]
            );
            assert!(
                (r0[1] - 30.0).abs() < 0.01,
                "humidity should be 30%, got {}",
                r0[1]
            );
            assert!(
                (r0[2] - 5.0).abs() < 0.01,
                "wind should be 5 km/h, got {}",
                r0[2]
            );
            assert!(
                (r0[3] - 1.0).abs() < 0.01,
                "valid flag should be 1.0, got {}",
                r0[3]
            );
        })
        .unwrap()
        .join()
        .unwrap();
}

#[cfg(feature = "weather")]
#[test]
fn weather_strand_info_returns_module_info() {
    use volt_core::module_info::ModuleType;
    use volt_hard::weather_strand::WeatherStrand;

    let strand = WeatherStrand::new();
    let info = strand.info().expect("weather strand should return ModuleInfo");
    assert_eq!(info.id, "volt-strand-weather");
    assert_eq!(info.module_type, ModuleType::HardStrand);
    assert!(!info.description.is_empty());
}

// --------------------------------------------------------------------------
// Test: Router without weather strand falls back (weather queries pass through)
// --------------------------------------------------------------------------

#[test]
fn router_without_weather_strand_passes_through() {
    std::thread::Builder::new()
        .stack_size(TEST_STACK)
        .spawn(|| {
            let mut router = IntentRouter::new();
            router.register(Box::new(MathEngine::new()));
            // No weather strand registered

            let frame = build_weather_frame(42.0);
            let result = router.route(&frame).unwrap();

            // The weather frame should NOT activate math engine
            let activated = result.decisions.iter().any(|d| d.activated);
            assert!(
                !activated,
                "weather frame should not activate math engine"
            );
        })
        .unwrap()
        .join()
        .unwrap();
}

// --------------------------------------------------------------------------
// Test: Panicking strand is caught, does not crash
// --------------------------------------------------------------------------

/// A test strand that always panics when processing.
struct PanickingStrand {
    capability: [f32; SLOT_DIM],
}

impl PanickingStrand {
    fn new() -> Self {
        // Use a vector that will match any frame with active slots
        let mut cap = [0.0_f32; SLOT_DIM];
        for (i, v) in cap.iter_mut().enumerate() {
            *v = if i % 2 == 0 { 0.1 } else { -0.1 };
        }
        let norm: f32 = cap.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in &mut cap {
            *x /= norm;
        }
        Self { capability: cap }
    }
}

impl HardStrand for PanickingStrand {
    fn name(&self) -> &str {
        "panicking_strand"
    }

    fn capability_vector(&self) -> &[f32; SLOT_DIM] {
        &self.capability
    }

    fn threshold(&self) -> f32 {
        0.0 // always activate if any slot matches at all
    }

    fn process(&self, _frame: &TensorFrame) -> Result<StrandResult, VoltError> {
        panic!("intentional test panic from panicking_strand");
    }
}

#[test]
fn router_catches_panicking_strand() {
    std::thread::Builder::new()
        .stack_size(TEST_STACK)
        .spawn(|| {
            let mut router = IntentRouter::new();
            router.register(Box::new(PanickingStrand::new()));

            let mut frame = TensorFrame::new();
            let mut slot = SlotData::new(SlotRole::Agent);
            let cap = PanickingStrand::new().capability;
            slot.write_resolution(0, cap);
            frame.write_slot(0, slot).unwrap();
            frame.meta[0].certainty = 0.8;

            // This should NOT panic — the router catches the strand's panic
            let result = router.route(&frame);
            assert!(
                result.is_ok(),
                "router should catch panicking strand, got: {:?}",
                result.err()
            );

            let result = result.unwrap();
            // The panicking strand should appear as non-activated
            let decision = result.decisions.iter().find(|d| d.strand_name == "panicking_strand");
            assert!(decision.is_some(), "panicking strand should appear in decisions");
            assert!(
                !decision.unwrap().activated,
                "panicking strand should not be marked as activated"
            );
        })
        .unwrap()
        .join()
        .unwrap();
}

// --------------------------------------------------------------------------
// Test: Router unregister and strand_names
// --------------------------------------------------------------------------

#[test]
fn router_unregister_removes_strand() {
    let mut router = IntentRouter::new();
    router.register(Box::new(MathEngine::new()));
    assert_eq!(router.strand_count(), 1);
    assert!(router.strand_names().contains(&"math_engine"));

    assert!(router.unregister("math_engine"));
    assert_eq!(router.strand_count(), 0);
    assert!(router.strand_names().is_empty());
}

#[test]
fn router_unregister_nonexistent_returns_false() {
    let mut router = IntentRouter::new();
    router.register(Box::new(MathEngine::new()));
    assert!(!router.unregister("nonexistent_strand"));
    assert_eq!(router.strand_count(), 1);
}

#[test]
fn router_strand_names_lists_all() {
    let mut router = IntentRouter::new();
    router.register(Box::new(MathEngine::new()));
    let names = router.strand_names();
    assert_eq!(names, vec!["math_engine"]);
}

#[cfg(feature = "weather")]
#[test]
fn router_strand_names_includes_weather() {
    use volt_hard::weather_strand::WeatherStrand;

    let mut router = IntentRouter::new();
    router.register(Box::new(MathEngine::new()));
    router.register(Box::new(WeatherStrand::new()));
    let names = router.strand_names();
    assert!(names.contains(&"math_engine"));
    assert!(names.contains(&"weather"));
}

// --------------------------------------------------------------------------
// Test: HardStrand info() default returns None (backward compat)
// --------------------------------------------------------------------------

#[test]
fn hard_strand_info_default_is_none() {
    let engine = MathEngine::new();
    assert!(
        engine.info().is_none(),
        "built-in MathEngine should return None for info()"
    );
}

// --------------------------------------------------------------------------
// Test: Unregister weather strand makes queries fall through
// --------------------------------------------------------------------------

#[cfg(feature = "weather")]
#[test]
fn unregister_weather_strand_falls_back() {
    std::thread::Builder::new()
        .stack_size(TEST_STACK)
        .spawn(|| {
            use volt_hard::weather_strand::WeatherStrand;

            let mut router = IntentRouter::new();
            router.register(Box::new(MathEngine::new()));
            router.register(Box::new(WeatherStrand::new()));

            // Verify weather strand activates
            let frame = build_weather_frame(42.0);
            let result = router.route(&frame).unwrap();
            assert!(
                result.decisions.iter().any(|d| d.activated && d.strand_name == "weather"),
                "weather strand should activate before unregister"
            );

            // Unregister weather strand
            assert!(router.unregister("weather"));

            // Now the same frame should NOT activate any strand
            let result = router.route(&frame).unwrap();
            let activated = result.decisions.iter().any(|d| d.activated);
            assert!(
                !activated,
                "after unregistering weather, weather queries should fall through"
            );
        })
        .unwrap()
        .join()
        .unwrap();
}
