//! Core Hyperdimensional Computing (HDC) algebra operations.
//!
//! This module implements the five fundamental operations for HDC:
//! - **similarity**: Cosine similarity between vectors
//! - **permute**: Cyclic shift for role rotation
//! - **superpose**: Additive superposition with normalization
//! - **bind**: FFT-based circular convolution for role-filler binding
//! - **unbind**: Approximate inverse of bind via circular correlation

use volt_core::{VoltError, SLOT_DIM};

/// Cosine similarity between two 256-dimensional vectors.
///
/// Returns a value in [-1, 1]:
/// - 1.0 = identical direction (maximally similar)
/// - 0.0 = orthogonal (unrelated)
/// - -1.0 = opposite direction (maximally dissimilar)
///
/// Returns 0.0 if either vector is zero (undefined similarity).
///
/// # Example
///
/// ```
/// use volt_bus::similarity;
/// use volt_core::SLOT_DIM;
///
/// let a = [1.0; SLOT_DIM];
/// let b = [1.0; SLOT_DIM];
/// let sim = similarity(&a, &b);
/// assert!((sim - 1.0).abs() < 1e-6); // Identical vectors
/// ```
pub fn similarity(a: &[f32; SLOT_DIM], b: &[f32; SLOT_DIM]) -> f32 {
    // Compute dot product
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

    // Compute norms
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    // Handle zero vectors
    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

/// Permute (cyclically shift) a vector by k positions.
///
/// Positive k shifts right, negative k shifts left.
/// Used in HDC for role rotation and structural manipulation.
///
/// # Example
///
/// ```
/// use volt_bus::{permute, similarity};
/// use volt_core::SLOT_DIM;
///
/// let a = [1.0; SLOT_DIM];
/// let shifted = permute(&a, 5);
/// let back = permute(&shifted, -5);
///
/// // Roundtrip should recover original
/// assert!(similarity(&a, &back) > 0.99);
/// ```
pub fn permute(a: &[f32; SLOT_DIM], k: isize) -> [f32; SLOT_DIM] {
    let mut result = [0.0; SLOT_DIM];

    // Normalize k to [0, SLOT_DIM)
    let k_normalized = k.rem_euclid(SLOT_DIM as isize) as usize;

    for (i, &value) in a.iter().enumerate() {
        let new_index = (i + k_normalized) % SLOT_DIM;
        result[new_index] = value;
    }

    result
}

/// Superposition: element-wise sum of multiple vectors, then normalize to unit length.
///
/// In HDC, superposition creates a representation where all constituents
/// are detectable via similarity (unlike bind, which hides constituents).
///
/// # Errors
///
/// Returns [`VoltError::BusError`] if:
/// - Input slice is empty
/// - All input vectors sum to zero or near-zero
///
/// # Example
///
/// ```
/// use volt_bus::{superpose, similarity};
/// use volt_core::SLOT_DIM;
///
/// let a = [1.0; SLOT_DIM];
/// let b = [0.5; SLOT_DIM];
/// let composite = superpose(&[&a, &b]).unwrap();
///
/// // Both constituents should be detectable
/// assert!(similarity(&composite, &a) > 0.0);
/// assert!(similarity(&composite, &b) > 0.0);
/// ```
pub fn superpose(vectors: &[&[f32; SLOT_DIM]]) -> Result<[f32; SLOT_DIM], VoltError> {
    if vectors.is_empty() {
        return Err(VoltError::BusError {
            message: "superpose requires at least one vector".to_string(),
        });
    }

    // Element-wise sum
    let mut result = [0.0; SLOT_DIM];
    for vec in vectors {
        for i in 0..SLOT_DIM {
            result[i] += vec[i];
        }
    }

    // Check for zero result
    let norm_sq: f32 = result.iter().map(|x| x * x).sum();
    if norm_sq < 1e-10 {
        return Err(VoltError::BusError {
            message: "superposition resulted in zero or near-zero vector".to_string(),
        });
    }

    // L2 normalize to unit length
    let norm = norm_sq.sqrt();
    for x in &mut result {
        *x /= norm;
    }

    Ok(result)
}

/// Bind two vectors via circular convolution (HDC role-filler binding).
///
/// In HDC, bind(ROLE, FILLER) creates a compound representation where
/// neither ROLE nor FILLER is directly accessible via similarity.
/// The bound result is dissimilar to both inputs (binding property).
///
/// # Errors
///
/// Returns [`VoltError::BusError`] if:
/// - Either input vector is zero or near-zero
/// - Either input contains NaN or Inf
///
/// # Performance
///
/// Target: < 10µs for 256-dimensional vectors (measured on commodity hardware)
///
/// # Example
///
/// ```
/// use volt_bus::bind;
/// use volt_core::SLOT_DIM;
///
/// // Create normalized vectors
/// let mut role = [0.0; SLOT_DIM];
/// role[0] = 1.0; // Unit vector
/// let mut filler = [0.0; SLOT_DIM];
/// filler[1] = 1.0; // Different unit vector
///
/// // Bind creates a compound representation
/// let bound = bind(&role, &filler).unwrap();
///
/// // Bound vector exists and is non-zero
/// let norm: f32 = bound.iter().map(|x| x * x).sum::<f32>().sqrt();
/// assert!(norm > 0.1);
/// ```
pub fn bind(a: &[f32; SLOT_DIM], b: &[f32; SLOT_DIM]) -> Result<[f32; SLOT_DIM], VoltError> {
    // Validate inputs
    validate_vector(a, "bind")?;
    validate_vector(b, "bind")?;

    let result = crate::fft::circular_convolution(a, b);
    Ok(result)
}

/// Unbind a vector using circular correlation (approximate inverse of bind).
///
/// In HDC, unbind(bind(ROLE, FILLER), ROLE) ≈ FILLER.
/// This allows content retrieval from bound representations.
///
/// # Accuracy
///
/// Cosine similarity between unbind(bind(a, b), a) and b should be > 0.85
/// (as per Milestone 1.2 requirement).
///
/// # Errors
///
/// Returns [`VoltError::BusError`] if:
/// - Either input vector is zero or near-zero
/// - Either input contains NaN or Inf
///
/// # Example
///
/// ```
/// use volt_bus::{bind, unbind, similarity};
/// use volt_core::SLOT_DIM;
///
/// let role = [1.0; SLOT_DIM];
/// let filler = [0.5; SLOT_DIM];
/// let bound = bind(&role, &filler).unwrap();
/// let recovered = unbind(&bound, &role).unwrap();
///
/// // Should recover filler with high similarity
/// assert!(similarity(&recovered, &filler) > 0.85);
/// ```
pub fn unbind(c: &[f32; SLOT_DIM], a: &[f32; SLOT_DIM]) -> Result<[f32; SLOT_DIM], VoltError> {
    // Validate inputs
    validate_vector(c, "unbind")?;
    validate_vector(a, "unbind")?;

    let result = crate::fft::circular_correlation(c, a);
    Ok(result)
}

/// Validate vector for algebra operations.
///
/// Checks for:
/// - Zero or near-zero vectors (L2 norm < 1e-10)
/// - NaN or Inf values
fn validate_vector(v: &[f32; SLOT_DIM], op_name: &str) -> Result<(), VoltError> {
    // Check for NaN/Inf
    if v.iter().any(|x| !x.is_finite()) {
        return Err(VoltError::BusError {
            message: format!("{}: input contains NaN or Inf", op_name),
        });
    }

    // Check for zero vector
    let norm_sq: f32 = v.iter().map(|x| x * x).sum();
    if norm_sq < 1e-10 {
        return Err(VoltError::BusError {
            message: format!("{}: input is zero or near-zero vector", op_name),
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a normalized test vector from seed (deterministic pseudo-random using hash)
    fn test_vector(seed: u64) -> [f32; SLOT_DIM] {
        let mut v = [0.0; SLOT_DIM];
        // Use hash-like mixing for true pseudo-randomness
        for i in 0..SLOT_DIM {
            let mut h = seed.wrapping_mul(0xd2b74407b1ce6e93);
            h = h.wrapping_add(i as u64);
            h ^= h >> 33;
            h = h.wrapping_mul(0xff51afd7ed558ccd);
            h ^= h >> 33;
            h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
            h ^= h >> 33;
            v[i] = ((h as f64 / u64::MAX as f64) * 2.0 - 1.0) as f32;
        }
        // Normalize to unit length
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in &mut v {
            *x /= norm;
        }
        v
    }

    #[test]
    fn similarity_identical_vectors() {
        let a = test_vector(42);
        let sim = similarity(&a, &a);
        assert!((sim - 1.0).abs() < 1e-6, "Identical vectors should have similarity 1.0");
    }

    #[test]
    fn similarity_orthogonal_vectors() {
        let mut a = [0.0; SLOT_DIM];
        a[0] = 1.0;
        let mut b = [0.0; SLOT_DIM];
        b[1] = 1.0;

        let sim = similarity(&a, &b);
        assert!(sim.abs() < 1e-6, "Orthogonal vectors should have similarity ≈ 0");
    }

    #[test]
    fn similarity_zero_vector_returns_zero() {
        let a = test_vector(42);
        let zero = [0.0; SLOT_DIM];

        assert_eq!(similarity(&a, &zero), 0.0);
        assert_eq!(similarity(&zero, &a), 0.0);
    }

    #[test]
    fn permute_roundtrip() {
        let a = test_vector(42);
        let shifted = permute(&a, 7);
        let back = permute(&shifted, -7);

        let sim = similarity(&a, &back);
        assert!((sim - 1.0).abs() < 1e-6, "Permute roundtrip should recover original");
    }

    #[test]
    fn permute_different_shifts_orthogonal() {
        let a = test_vector(42);
        let shift1 = permute(&a, 1);
        let shift2 = permute(&a, 2);

        let sim = similarity(&shift1, &shift2);
        assert!(sim.abs() < 0.2, "Different shifts should be nearly orthogonal, got {}", sim);
    }

    #[test]
    fn permute_preserves_norm() {
        let a = test_vector(42);
        let shifted = permute(&a, 5);

        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_shifted: f32 = shifted.iter().map(|x| x * x).sum::<f32>().sqrt();

        assert!((norm_a - norm_shifted).abs() < 1e-6, "Permute should preserve norm");
    }

    #[test]
    fn superpose_detects_constituents() {
        let a = test_vector(42);
        let b = test_vector(99);

        let composite = superpose(&[&a, &b]).unwrap();

        let sim_a = similarity(&composite, &a);
        let sim_b = similarity(&composite, &b);

        assert!(sim_a > 0.0, "Constituent a should be detectable");
        assert!(sim_b > 0.0, "Constituent b should be detectable");
    }

    #[test]
    fn superpose_output_normalized() {
        let a = test_vector(42);
        let b = test_vector(99);

        let composite = superpose(&[&a, &b]).unwrap();

        let norm: f32 = composite.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "Superpose output should be unit norm");
    }

    #[test]
    fn superpose_empty_input_errors() {
        let result = superpose(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn superpose_zero_result_errors() {
        let a = test_vector(42);
        let mut neg_a = a;
        for x in &mut neg_a {
            *x = -*x;
        }

        let result = superpose(&[&a, &neg_a]);
        assert!(result.is_err(), "Superpose of opposite vectors should error");
    }

    #[test]
    fn bind_unbind_recovers_original() {
        let a = test_vector(42);
        let b = test_vector(99);

        let bound = bind(&a, &b).unwrap();
        let recovered = unbind(&bound, &a).unwrap();

        let sim = similarity(&recovered, &b);
        assert!(
            sim > 0.85,
            "Milestone requirement: unbind(bind(a, b), a) ~ b with similarity > 0.85, got {}",
            sim
        );
    }

    #[test]
    fn bind_produces_dissimilar_result() {
        let a = test_vector(42);
        let b = test_vector(99);

        let bound = bind(&a, &b).unwrap();

        let sim_a = similarity(&bound, &a);
        let sim_b = similarity(&bound, &b);

        assert!(sim_a.abs() < 0.1, "Bound should be dissimilar to input a, got {}", sim_a);
        assert!(sim_b.abs() < 0.1, "Bound should be dissimilar to input b, got {}", sim_b);
    }

    #[test]
    fn bind_commutative() {
        let a = test_vector(42);
        let b = test_vector(99);

        let ab = bind(&a, &b).unwrap();
        let ba = bind(&b, &a).unwrap();

        let sim = similarity(&ab, &ba);
        assert!(sim > 0.99, "Bind should be commutative, similarity = {}", sim);
    }

    #[test]
    fn bind_zero_vector_errors() {
        let a = test_vector(42);
        let zero = [0.0; SLOT_DIM];

        assert!(bind(&zero, &a).is_err());
        assert!(bind(&a, &zero).is_err());
    }

    #[test]
    fn unbind_zero_vector_errors() {
        let a = test_vector(42);
        let zero = [0.0; SLOT_DIM];

        assert!(unbind(&zero, &a).is_err());
        assert!(unbind(&a, &zero).is_err());
    }

    #[test]
    fn bind_nan_input_errors() {
        let a = test_vector(42);
        let mut nan_vec = a;
        nan_vec[0] = f32::NAN;

        assert!(bind(&nan_vec, &a).is_err());
        assert!(bind(&a, &nan_vec).is_err());
    }

    #[test]
    fn bind_inf_input_errors() {
        let a = test_vector(42);
        let mut inf_vec = a;
        inf_vec[0] = f32::INFINITY;

        assert!(bind(&inf_vec, &a).is_err());
        assert!(bind(&a, &inf_vec).is_err());
    }
}
