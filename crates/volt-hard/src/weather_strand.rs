//! WeatherStrand — example community module demonstrating [`HardStrand`].
//!
//! This is a reference implementation for module developers. It shows how
//! to build a complete Hard Strand with:
//! - A deterministic capability vector for routing.
//! - A slot convention for input/output.
//! - Module metadata via [`info()`](HardStrand::info).
//!
//! The weather data is **deterministic mock data** — no real API calls.
//! This keeps tests network-free while demonstrating the full lifecycle.
//!
//! ## Slot Convention
//!
//! | Slot | Resolution | Dim | Meaning |
//! |------|-----------|-----|---------|
//! | S6 (Instrument) | R0 | dim\[0\] = 20.0 | WEATHER operation code |
//! | S6 (Instrument) | R0 | dim\[1\] | City hash (f32) |
//! | S8 (Result) | R0 | dim\[0\] | Temperature (°C) |
//! | S8 (Result) | R0 | dim\[1\] | Humidity (%) |
//! | S8 (Result) | R0 | dim\[2\] | Wind speed (km/h) |
//! | S8 (Result) | R0 | dim\[3\] | 1.0 = valid result |
//!
//! ## Feature Gate
//!
//! This module is behind `#[cfg(feature = "weather")]`. Enable it with:
//! ```toml
//! [features]
//! weather = ["volt-hard/weather"]
//! ```

use volt_core::{
    module_info::{ModuleInfo, ModuleType},
    slot::{SlotMeta, SlotSource},
    SlotData, SlotRole, TensorFrame, VoltError, MAX_SLOTS, SLOT_DIM,
};

use crate::strand::{HardStrand, StrandResult};

/// Operation code for weather queries (stored in S6 R0 dim[0]).
const OP_WEATHER: f32 = 20.0;

/// Slot index for operation input (Instrument = S6).
const INSTRUMENT_SLOT: usize = 6;
/// Slot index for result output (Result = S8).
const RESULT_SLOT: usize = 8;

/// Five deterministic weather profiles: (temperature °C, humidity %, wind km/h).
const WEATHER_PROFILES: [(f32, f32, f32); 5] = [
    (22.0, 65.0, 12.0), // Profile 0: warm, moderate humidity, light breeze
    (-5.0, 80.0, 25.0), // Profile 1: cold, high humidity, windy
    (35.0, 30.0, 5.0),  // Profile 2: hot, dry, calm
    (15.0, 70.0, 18.0), // Profile 3: mild, humid, breezy
    (8.0, 55.0, 8.0),   // Profile 4: cool, moderate, light wind
];

/// Example Hard Strand that provides mock weather data.
///
/// Demonstrates the full module lifecycle:
/// 1. Capability vector → Intent Router routes weather queries here.
/// 2. Slot convention → Input via S6 (Instrument), output via S8 (Result).
/// 3. Module metadata → `info()` returns [`ModuleInfo`].
///
/// # Example
///
/// ```
/// use volt_hard::weather_strand::WeatherStrand;
/// use volt_hard::strand::HardStrand;
/// use volt_core::{TensorFrame, SlotData, SlotRole, SLOT_DIM};
///
/// std::thread::Builder::new().stack_size(4 * 1024 * 1024).spawn(|| {
///     let strand = WeatherStrand::new();
///     assert_eq!(strand.name(), "weather");
///
///     let mut frame = TensorFrame::new();
///     // Tag predicate with weather capability for routing
///     let mut pred = SlotData::new(SlotRole::Predicate);
///     pred.write_resolution(0, *strand.capability_vector());
///     frame.write_slot(1, pred).unwrap();
///     frame.meta[1].certainty = 0.8;
///
///     // Encode weather query in instrument slot
///     let mut inst = SlotData::new(SlotRole::Instrument);
///     let mut data = [0.0_f32; SLOT_DIM];
///     data[0] = 20.0; // OP_WEATHER
///     data[1] = 42.0; // city hash
///     inst.write_resolution(0, data);
///     frame.write_slot(6, inst).unwrap();
///     frame.meta[6].certainty = 0.9;
///
///     let result = strand.process(&frame).unwrap();
///     assert!(result.activated);
///     let r = result.frame.read_slot(8).unwrap();
///     let r0 = r.resolutions[0].unwrap();
///     assert!((r0[3] - 1.0).abs() < 0.01, "valid flag should be 1.0");
/// }).unwrap().join().unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct WeatherStrand {
    /// Pre-computed capability vector for routing.
    capability: [f32; SLOT_DIM],
}

impl WeatherStrand {
    /// Create a new `WeatherStrand` with a deterministic capability vector.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_hard::weather_strand::WeatherStrand;
    /// use volt_hard::strand::HardStrand;
    ///
    /// let strand = WeatherStrand::new();
    /// assert_eq!(strand.name(), "weather");
    /// assert!(strand.threshold() > 0.0);
    /// ```
    pub fn new() -> Self {
        Self {
            capability: Self::build_capability_vector(),
        }
    }

    /// Build the deterministic capability vector for weather queries.
    ///
    /// Uses the same splitmix64 hash pattern as [`MathEngine`](crate::math_engine::MathEngine)
    /// with a different seed to produce a unique, normalized 256-dim vector.
    fn build_capability_vector() -> [f32; SLOT_DIM] {
        const WEATHER_SEED: u64 = 0x5745_4154_4845_5231; // "WEATHER1"
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
        // L2 normalize
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for x in &mut v {
                *x /= norm;
            }
        }
        v
    }

    /// Look up a deterministic weather profile from a city hash.
    ///
    /// Maps the hash to one of 5 fixed weather profiles via modulo.
    /// Returns `(temperature, humidity, wind_speed)`.
    fn lookup_weather(city_hash: f32) -> (f32, f32, f32) {
        let index = (city_hash.abs() as u32) % WEATHER_PROFILES.len() as u32;
        WEATHER_PROFILES[index as usize]
    }
}

impl Default for WeatherStrand {
    fn default() -> Self {
        Self::new()
    }
}

impl HardStrand for WeatherStrand {
    fn name(&self) -> &str {
        "weather"
    }

    fn capability_vector(&self) -> &[f32; SLOT_DIM] {
        &self.capability
    }

    fn threshold(&self) -> f32 {
        0.3
    }

    fn process(&self, frame: &TensorFrame) -> Result<StrandResult, VoltError> {
        // Check for weather operation in Instrument slot (S6)
        let inst_data = frame
            .slots[INSTRUMENT_SLOT]
            .as_ref()
            .and_then(|slot| slot.resolutions[0].as_ref());

        let Some(r0) = inst_data else {
            return Ok(StrandResult {
                frame: frame.clone(),
                activated: false,
                description: "weather: no instrument data".to_string(),
            });
        };

        let op_code = r0[0];
        if (op_code - OP_WEATHER).abs() >= 0.5 {
            return Ok(StrandResult {
                frame: frame.clone(),
                activated: false,
                description: format!("weather: unrecognized op code {op_code}"),
            });
        }

        // Read city hash from dim[1]
        let city_hash = r0[1];
        let (temp, humidity, wind) = Self::lookup_weather(city_hash);

        // Build result frame
        let mut result = frame.clone();

        // Write weather data to Result slot (S8)
        let mut result_data = [0.0_f32; SLOT_DIM];
        result_data[0] = temp;
        result_data[1] = humidity;
        result_data[2] = wind;
        result_data[3] = 1.0; // valid flag

        let mut result_slot = SlotData::new(SlotRole::Result);
        result_slot.write_resolution(0, result_data);
        result.write_slot(RESULT_SLOT, result_slot)?;

        // Set metadata: gamma = 0.9 (mock data, not exact like math)
        result.meta[RESULT_SLOT] = SlotMeta {
            certainty: 0.9,
            source: SlotSource::HardCore,
            updated_at: 0,
            needs_verify: false,
        };

        // Update global certainty (min-rule over active slots)
        let min_gamma = (0..MAX_SLOTS)
            .filter(|&i| result.slots[i].is_some())
            .map(|i| result.meta[i].certainty)
            .fold(f32::INFINITY, f32::min);
        if min_gamma.is_finite() {
            result.frame_meta.global_certainty = min_gamma;
        }

        let description = format!(
            "weather: city_hash={city_hash:.0} → temp={temp}°C, humidity={humidity}%, wind={wind}km/h"
        );

        Ok(StrandResult {
            frame: result,
            activated: true,
            description,
        })
    }

    fn info(&self) -> Option<ModuleInfo> {
        Some(ModuleInfo {
            id: "volt-strand-weather".to_string(),
            display_name: "Weather Strand".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            author: "Volt X Team".to_string(),
            description: "Example community module providing mock weather data.".to_string(),
            module_type: ModuleType::HardStrand,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use volt_core::{SlotData, SlotRole};

    const TEST_STACK: usize = 4 * 1024 * 1024;

    #[test]
    fn weather_strand_name_and_threshold() {
        let strand = WeatherStrand::new();
        assert_eq!(strand.name(), "weather");
        assert!((strand.threshold() - 0.3).abs() < 0.01);
    }

    #[test]
    fn weather_capability_vector_is_normalized() {
        let strand = WeatherStrand::new();
        let cap = strand.capability_vector();
        let norm: f32 = cap.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "capability vector should be L2-normalized, got norm={norm}"
        );
    }

    #[test]
    fn weather_info_returns_module_info() {
        let strand = WeatherStrand::new();
        let info = strand.info().expect("weather strand should return ModuleInfo");
        assert_eq!(info.id, "volt-strand-weather");
        assert_eq!(info.module_type, ModuleType::HardStrand);
    }

    #[test]
    fn weather_process_valid_query() {
        std::thread::Builder::new()
            .stack_size(TEST_STACK)
            .spawn(|| {
                let strand = WeatherStrand::new();
                let mut frame = TensorFrame::new();

                // Instrument slot with weather op code
                let mut inst = SlotData::new(SlotRole::Instrument);
                let mut data = [0.0_f32; SLOT_DIM];
                data[0] = OP_WEATHER; // weather op
                data[1] = 42.0; // city hash → 42 % 5 = 2 → profile 2 (hot, dry, calm)
                inst.write_resolution(0, data);
                frame.write_slot(INSTRUMENT_SLOT, inst).unwrap();
                frame.meta[INSTRUMENT_SLOT].certainty = 0.9;

                let result = strand.process(&frame).unwrap();
                assert!(result.activated);

                let r = result.frame.read_slot(RESULT_SLOT).unwrap();
                let r0 = r.resolutions[0].unwrap();
                assert!((r0[0] - 35.0).abs() < 0.01, "temp should be 35.0°C");
                assert!((r0[1] - 30.0).abs() < 0.01, "humidity should be 30%");
                assert!((r0[2] - 5.0).abs() < 0.01, "wind should be 5 km/h");
                assert!((r0[3] - 1.0).abs() < 0.01, "valid flag should be 1.0");
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn weather_process_no_instrument_data() {
        std::thread::Builder::new()
            .stack_size(TEST_STACK)
            .spawn(|| {
                let strand = WeatherStrand::new();
                let frame = TensorFrame::new();
                let result = strand.process(&frame).unwrap();
                assert!(!result.activated);
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn weather_process_wrong_op_code() {
        std::thread::Builder::new()
            .stack_size(TEST_STACK)
            .spawn(|| {
                let strand = WeatherStrand::new();
                let mut frame = TensorFrame::new();

                let mut inst = SlotData::new(SlotRole::Instrument);
                let mut data = [0.0_f32; SLOT_DIM];
                data[0] = 3.0; // MUL op, not weather
                data[1] = 42.0;
                inst.write_resolution(0, data);
                frame.write_slot(INSTRUMENT_SLOT, inst).unwrap();
                frame.meta[INSTRUMENT_SLOT].certainty = 0.9;

                let result = strand.process(&frame).unwrap();
                assert!(!result.activated);
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn weather_all_five_profiles_reachable() {
        for city in 0..5u32 {
            let (t, h, w) = WeatherStrand::lookup_weather(city as f32);
            assert_eq!((t, h, w), WEATHER_PROFILES[city as usize]);
        }
    }

    #[test]
    fn weather_deterministic_results() {
        // Same city hash always produces same weather
        let (t1, h1, w1) = WeatherStrand::lookup_weather(42.0);
        let (t2, h2, w2) = WeatherStrand::lookup_weather(42.0);
        assert_eq!(t1, t2);
        assert_eq!(h1, h2);
        assert_eq!(w1, w2);
    }
}
