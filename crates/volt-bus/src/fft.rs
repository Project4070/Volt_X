//! FFT infrastructure for hyperdimensional computing operations.
//!
//! This module provides FFT-based circular convolution and correlation
//! operations used by bind and unbind operations. Uses a thread-local
//! FFT planner for performance (avoids repeated planner allocation).

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use std::cell::RefCell;
use volt_core::SLOT_DIM;

thread_local! {
    /// Thread-local FFT planner for 256-point transforms.
    /// Reusing the planner avoids ~100µs allocation overhead per operation.
    static FFT_PLANNER: RefCell<FftPlanner<f32>> = RefCell::new(FftPlanner::new());
}

/// Convert real f32 array to complex array (imaginary parts = 0).
fn real_to_complex(input: &[f32; SLOT_DIM]) -> Vec<Complex<f32>> {
    input.iter().map(|&x| Complex::new(x, 0.0)).collect()
}

/// Extract real parts from complex array, discarding imaginary parts.
fn complex_to_real(input: &[Complex<f32>]) -> [f32; SLOT_DIM] {
    let mut result = [0.0; SLOT_DIM];
    for (i, c) in input.iter().enumerate().take(SLOT_DIM) {
        result[i] = c.re;
    }
    result
}

/// Compute circular convolution via FFT: a ⊗ b
///
/// Algorithm:
/// 1. Convert a, b to complex (imaginary = 0)
/// 2. Forward FFT on both
/// 3. Element-wise multiply in frequency domain
/// 4. Inverse FFT
/// 5. Normalize by SLOT_DIM
/// 6. Extract real parts
///
/// # Performance
/// ~15-20µs for 256-dimensional vectors (2 forward + 1 inverse FFT)
pub(crate) fn circular_convolution(a: &[f32; SLOT_DIM], b: &[f32; SLOT_DIM]) -> [f32; SLOT_DIM] {
    FFT_PLANNER.with(|planner| {
        let mut planner = planner.borrow_mut();
        let fft = planner.plan_fft_forward(SLOT_DIM);
        let ifft = planner.plan_fft_inverse(SLOT_DIM);

        // Convert to complex
        let mut freq_a = real_to_complex(a);
        let mut freq_b = real_to_complex(b);

        // Forward FFT
        fft.process(&mut freq_a);
        fft.process(&mut freq_b);

        // Element-wise multiplication in frequency domain
        let mut freq_result: Vec<Complex<f32>> = freq_a
            .iter()
            .zip(freq_b.iter())
            .map(|(x, y)| x * y)
            .collect();

        // Inverse FFT
        ifft.process(&mut freq_result);

        // Normalize by SLOT_DIM (FFT convention)
        let scale = 1.0 / SLOT_DIM as f32;
        for c in &mut freq_result {
            *c *= scale;
        }

        complex_to_real(&freq_result)
    })
}

/// Compute circular correlation via FFT: a ⊙ b
///
/// Used for unbind operation. Implements approximate inverse of circular convolution.
///
/// Algorithm:
/// 1. Convert a, b to complex
/// 2. Forward FFT on both
/// 3. Element-wise multiply: FFT(a) * conj(FFT(b))
/// 4. Inverse FFT
/// 5. Normalize by SLOT_DIM
/// 6. Extract real parts
///
/// # HDC Property
/// circular_correlation(circular_convolution(a, b), a) ≈ b
pub(crate) fn circular_correlation(a: &[f32; SLOT_DIM], b: &[f32; SLOT_DIM]) -> [f32; SLOT_DIM] {
    FFT_PLANNER.with(|planner| {
        let mut planner = planner.borrow_mut();
        let fft = planner.plan_fft_forward(SLOT_DIM);
        let ifft = planner.plan_fft_inverse(SLOT_DIM);

        // Convert to complex
        let mut freq_a = real_to_complex(a);
        let mut freq_b = real_to_complex(b);

        // Forward FFT
        fft.process(&mut freq_a);
        fft.process(&mut freq_b);

        // Element-wise division in frequency domain (true inverse)
        // unbind(c, a) = IFFT(FFT(c) / FFT(a))
        let mut freq_result: Vec<Complex<f32>> = freq_a
            .iter()
            .zip(freq_b.iter())
            .map(|(x, y)| {
                // Division: x / y = x * conj(y) / |y|^2
                let y_norm_sq = y.re * y.re + y.im * y.im;
                if y_norm_sq < 1e-10 {
                    // Avoid division by zero
                    Complex::new(0.0, 0.0)
                } else {
                    (x * y.conj()) / y_norm_sq
                }
            })
            .collect();

        // Inverse FFT
        ifft.process(&mut freq_result);

        // Normalize by SLOT_DIM
        let scale = 1.0 / SLOT_DIM as f32;
        for c in &mut freq_result {
            *c *= scale;
        }

        complex_to_real(&freq_result)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a simple test vector (unit vector along first axis)
    fn unit_vector() -> [f32; SLOT_DIM] {
        let mut v = [0.0; SLOT_DIM];
        v[0] = 1.0;
        v
    }

    /// Create a normalized random-like vector from seed (deterministic pseudo-random using hash)
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
    fn fft_roundtrip_preserves_vector() {
        let a = test_vector(42);

        FFT_PLANNER.with(|planner| {
            let mut planner = planner.borrow_mut();
            let fft = planner.plan_fft_forward(SLOT_DIM);
            let ifft = planner.plan_fft_inverse(SLOT_DIM);

            let mut freq = real_to_complex(&a);
            fft.process(&mut freq);
            ifft.process(&mut freq);

            // Normalize
            let scale = 1.0 / SLOT_DIM as f32;
            for c in &mut freq {
                *c *= scale;
            }

            let recovered = complex_to_real(&freq);

            // Check recovery is close to original
            for i in 0..SLOT_DIM {
                assert!((a[i] - recovered[i]).abs() < 1e-5);
            }
        });
    }

    #[test]
    fn circular_convolution_commutative() {
        let a = test_vector(42);
        let b = test_vector(99);

        let ab = circular_convolution(&a, &b);
        let ba = circular_convolution(&b, &a);

        // Circular convolution should be commutative
        for i in 0..SLOT_DIM {
            assert!((ab[i] - ba[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn circular_correlation_approximate_inverse() {
        let a = test_vector(42);
        let b = test_vector(99);

        let conv = circular_convolution(&a, &b);
        let recovered = circular_correlation(&conv, &a);

        // Calculate cosine similarity between recovered and original b
        let dot: f32 = recovered.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_recovered: f32 = recovered.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        let similarity = dot / (norm_recovered * norm_b);

        // Should recover with high similarity (> 0.85 as per milestone requirement)
        assert!(
            similarity > 0.85,
            "Correlation should approximate inverse, similarity = {}",
            similarity
        );
    }

    #[test]
    fn unit_vector_convolution() {
        let unit = unit_vector();
        let b = test_vector(42);

        let result = circular_convolution(&unit, &b);

        // Convolving with unit vector along first axis should produce
        // a shifted version (circular convolution property)
        // Result should not be all zeros
        let sum: f32 = result.iter().map(|x| x.abs()).sum();
        assert!(sum > 0.1, "Result should not be near-zero");
    }
}
