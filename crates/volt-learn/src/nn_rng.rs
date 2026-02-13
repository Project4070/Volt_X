//! Simple deterministic PRNG for training data generation.
//!
//! Provides a lightweight splitmix64-based RNG for deterministic
//! encoding in the code frame pair pipeline. Does not depend on
//! external crate internals.

/// Deterministic PRNG based on splitmix64.
///
/// # Example
///
/// ```
/// use volt_learn::nn_rng::SimpleRng;
///
/// let mut rng = SimpleRng::new(42);
/// let val = rng.next_f32();
/// assert!((0.0..1.0).contains(&val));
/// ```
#[derive(Clone)]
pub struct SimpleRng(u64);

impl SimpleRng {
    /// Creates a new PRNG with the given seed.
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// Returns the next pseudo-random u64.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    /// Returns a uniform f32 in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / ((1u64 << 24) as f32)
    }

    /// Returns a uniform f32 in [lo, hi).
    pub fn next_f32_range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_f32()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic() {
        let mut r1 = SimpleRng::new(42);
        let mut r2 = SimpleRng::new(42);
        for _ in 0..100 {
            assert_eq!(r1.next_u64(), r2.next_u64());
        }
    }

    #[test]
    fn f32_in_range() {
        let mut rng = SimpleRng::new(42);
        for _ in 0..1000 {
            let v = rng.next_f32();
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn f32_range_in_bounds() {
        let mut rng = SimpleRng::new(42);
        for _ in 0..1000 {
            let v = rng.next_f32_range(-0.5, 0.5);
            assert!((-0.5..0.5).contains(&v));
        }
    }
}
