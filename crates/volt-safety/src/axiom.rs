//! Axiomatic Guard — five hardcoded safety invariants (K1-K5).
//!
//! Each axiom is a constant 256-dimensional vector living in the same HDC
//! space as TensorFrame slot embeddings. The safety layer computes cosine
//! similarity between frame slot vectors and each axiom vector. High
//! similarity indicates the frame's content is close to a prohibited region.
//!
//! ## Axioms
//!
//! | ID | Name       | Protects Against                         |
//! |----|------------|------------------------------------------|
//! | K1 | Harm       | Direct physical harm instructions        |
//! | K2 | Deception  | Impersonation or false identity claims   |
//! | K3 | Privacy    | Personal data extraction or exposure     |
//! | K4 | Autonomy   | Suppression of user agency               |
//! | K5 | Integrity  | Corruption of system state or memory     |
//!
//! ## Design
//!
//! Axioms are **code, not weights**. They cannot be modified at runtime,
//! cannot be overridden by any neural component, and are deterministic.
//! See ADR-007: Safety as Code, Not Weights.
//!
//! # Example
//!
//! ```
//! use volt_safety::axiom::{Axiom, Severity, default_axioms};
//! use volt_bus::similarity;
//!
//! let axioms = default_axioms();
//! assert_eq!(axioms.len(), 5);
//! assert_eq!(axioms[0].name, "K1_harm");
//!
//! // Each axiom vector is unit-normalized
//! let norm: f32 = axioms[0].vector.iter().map(|x| x * x).sum::<f32>().sqrt();
//! assert!((norm - 1.0).abs() < 1e-4);
//! ```

use volt_core::SLOT_DIM;

/// Severity level for axiom violations.
///
/// Determines the system response when a frame's content is similar
/// to an axiom's prohibited region.
///
/// # Example
///
/// ```
/// use volt_safety::axiom::Severity;
///
/// let s = Severity::Halt;
/// assert_ne!(s, Severity::Warning);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    /// Log the event but allow processing to continue.
    Warning,
    /// Immediately halt processing and trigger Omega Veto.
    Halt,
}

/// A single safety axiom — an immutable invariant in HDC space.
///
/// The axiom's `vector` defines a region of the embedding space that
/// represents prohibited content. Frames with slot embeddings whose
/// cosine similarity to this vector exceeds `threshold` are flagged.
///
/// # Example
///
/// ```
/// use volt_safety::axiom::{Axiom, Severity};
/// use volt_core::SLOT_DIM;
///
/// let axiom = Axiom {
///     name: "K1_harm",
///     vector: [0.0; SLOT_DIM],
///     threshold: 0.7,
///     severity: Severity::Halt,
/// };
/// assert_eq!(axiom.name, "K1_harm");
/// ```
#[derive(Debug, Clone)]
pub struct Axiom {
    /// Human-readable name (e.g., "K1_harm").
    pub name: &'static str,

    /// The 256-dimensional axiom vector in HDC space.
    pub vector: [f32; SLOT_DIM],

    /// Cosine similarity threshold above which a violation is flagged.
    pub threshold: f32,

    /// How the system responds to a violation of this axiom.
    pub severity: Severity,
}

/// Build a deterministic 256-dim unit vector from a seed.
///
/// Uses the same splitmix64-style hash as the Hard Strand capability
/// vectors, ensuring all axiom vectors live in the same HDC space.
fn build_axiom_vector(seed: u64) -> [f32; SLOT_DIM] {
    let mut v = [0.0_f32; SLOT_DIM];
    for (i, val) in v.iter_mut().enumerate() {
        let mut h = seed.wrapping_mul(0xd2b7_4407_b1ce_6e93);
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

// Deterministic seeds for each axiom (ASCII-encoded mnemonics).
const K1_SEED: u64 = 0x4b31_4841_524d_5f5f; // "K1HARM__"
const K2_SEED: u64 = 0x4b32_4445_4345_5054; // "K2DECEPT"
const K3_SEED: u64 = 0x4b33_5052_4956_4143; // "K3PRIVAC"
const K4_SEED: u64 = 0x4b34_4155_544f_4e4f; // "K4AUTONO"
const K5_SEED: u64 = 0x4b35_494e_5445_4752; // "K5INTEGR"

/// Returns the five default safety axioms (K1-K5).
///
/// These axioms are immutable constants. They define the prohibited
/// regions of embedding space that no frame output may enter.
///
/// # Example
///
/// ```
/// use volt_safety::axiom::default_axioms;
///
/// let axioms = default_axioms();
/// assert_eq!(axioms.len(), 5);
///
/// // All axiom vectors are unit-normalized
/// for axiom in &axioms {
///     let norm: f32 = axiom.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
///     assert!((norm - 1.0).abs() < 1e-4, "{} not normalized", axiom.name);
/// }
/// ```
pub fn default_axioms() -> Vec<Axiom> {
    vec![
        Axiom {
            name: "K1_harm",
            vector: build_axiom_vector(K1_SEED),
            threshold: 0.7,
            severity: Severity::Halt,
        },
        Axiom {
            name: "K2_deception",
            vector: build_axiom_vector(K2_SEED),
            threshold: 0.7,
            severity: Severity::Halt,
        },
        Axiom {
            name: "K3_privacy",
            vector: build_axiom_vector(K3_SEED),
            threshold: 0.7,
            severity: Severity::Halt,
        },
        Axiom {
            name: "K4_autonomy",
            vector: build_axiom_vector(K4_SEED),
            threshold: 0.65,
            severity: Severity::Warning,
        },
        Axiom {
            name: "K5_integrity",
            vector: build_axiom_vector(K5_SEED),
            threshold: 0.7,
            severity: Severity::Halt,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use volt_bus::similarity;

    #[test]
    fn default_axioms_returns_five() {
        let axioms = default_axioms();
        assert_eq!(axioms.len(), 5);
    }

    #[test]
    fn axiom_names_correct() {
        let axioms = default_axioms();
        assert_eq!(axioms[0].name, "K1_harm");
        assert_eq!(axioms[1].name, "K2_deception");
        assert_eq!(axioms[2].name, "K3_privacy");
        assert_eq!(axioms[3].name, "K4_autonomy");
        assert_eq!(axioms[4].name, "K5_integrity");
    }

    #[test]
    fn axiom_vectors_are_unit_normalized() {
        let axioms = default_axioms();
        for axiom in &axioms {
            let norm: f32 = axiom.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-4,
                "{} norm = {}, expected ~1.0",
                axiom.name,
                norm
            );
        }
    }

    #[test]
    fn axiom_vectors_are_mutually_dissimilar() {
        let axioms = default_axioms();
        for i in 0..axioms.len() {
            for j in (i + 1)..axioms.len() {
                let sim = similarity(&axioms[i].vector, &axioms[j].vector);
                assert!(
                    sim.abs() < 0.3,
                    "{} vs {} similarity = {:.4}, expected < 0.3",
                    axioms[i].name,
                    axioms[j].name,
                    sim
                );
            }
        }
    }

    #[test]
    fn axiom_vectors_are_deterministic() {
        let a1 = default_axioms();
        let a2 = default_axioms();
        for (ax1, ax2) in a1.iter().zip(a2.iter()) {
            let sim = similarity(&ax1.vector, &ax2.vector);
            assert!(
                (sim - 1.0).abs() < 1e-6,
                "{} not deterministic: sim = {:.6}",
                ax1.name,
                sim
            );
        }
    }

    #[test]
    fn axiom_thresholds_in_valid_range() {
        let axioms = default_axioms();
        for axiom in &axioms {
            assert!(
                axiom.threshold > 0.0 && axiom.threshold <= 1.0,
                "{} threshold {} out of range",
                axiom.name,
                axiom.threshold
            );
        }
    }

    #[test]
    fn k1_k3_are_halt_severity() {
        let axioms = default_axioms();
        assert_eq!(axioms[0].severity, Severity::Halt); // K1 harm
        assert_eq!(axioms[1].severity, Severity::Halt); // K2 deception
        assert_eq!(axioms[2].severity, Severity::Halt); // K3 privacy
    }

    #[test]
    fn k4_is_warning_severity() {
        let axioms = default_axioms();
        assert_eq!(axioms[3].severity, Severity::Warning); // K4 autonomy
    }

    #[test]
    fn random_vector_does_not_trigger_axioms() {
        let axioms = default_axioms();
        // A generic random vector should not be near any axiom
        let random = build_axiom_vector(0xDEAD_BEEF_CAFE_BABE);
        for axiom in &axioms {
            let sim = similarity(&random, &axiom.vector);
            assert!(
                sim < axiom.threshold,
                "random vector triggered {}: sim={:.4} >= threshold={:.2}",
                axiom.name,
                sim,
                axiom.threshold
            );
        }
    }
}
