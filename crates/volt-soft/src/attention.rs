//! Cross-slot attention mechanism for the RAR Attend phase.
//!
//! Computes scaled dot-product attention across TensorFrame slots:
//! - Q, K, V projections per active slot
//! - Softmax attention weights: `softmax(Q·Kᵀ / √d)`
//! - Weighted value aggregation: `msg_i = Σⱼ αᵢⱼ · Vⱼ`
//!
//! Attention is O(S² × D) where S=16 slots and D=256 dims,
//! far cheaper than token-level O(n²) in transformers.

use crate::nn::{Linear, Rng};
use volt_core::{VoltError, MAX_SLOTS, SLOT_DIM};

/// Cross-slot attention module for the RAR Attend phase.
///
/// Each active slot produces a query, key, and value vector via
/// learned linear projections. Attention weights determine how
/// much each slot influences every other slot.
///
/// # Example
///
/// ```
/// use volt_soft::attention::SlotAttention;
/// use volt_core::{MAX_SLOTS, SLOT_DIM};
///
/// let attn = SlotAttention::new_random(42);
/// let mut states = [const { None }; MAX_SLOTS];
/// states[0] = Some([0.1_f32; SLOT_DIM]);
/// states[1] = Some([0.2_f32; SLOT_DIM]);
///
/// let messages = attn.forward(&states).unwrap();
/// // Active slots receive messages; inactive slots get zeros
/// assert!(messages[0].iter().any(|&x| x != 0.0));
/// assert!(messages[2].iter().all(|&x| x == 0.0));
/// ```
#[derive(Clone)]
pub struct SlotAttention {
    wq: Linear,
    wk: Linear,
    wv: Linear,
    scale: f32,
}

impl std::fmt::Debug for SlotAttention {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SlotAttention(dim={}, scale={:.4})", SLOT_DIM, self.scale)
    }
}

impl SlotAttention {
    /// Creates a new attention module with Xavier/Glorot random initialization.
    ///
    /// The seed determines the projection weights. Same seed produces
    /// identical weights (deterministic).
    ///
    /// # Example
    ///
    /// ```
    /// use volt_soft::attention::SlotAttention;
    ///
    /// let attn = SlotAttention::new_random(42);
    /// ```
    pub fn new_random(seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        Self {
            wq: Linear::new_xavier(&mut rng, SLOT_DIM, SLOT_DIM),
            wk: Linear::new_xavier(&mut rng, SLOT_DIM, SLOT_DIM),
            wv: Linear::new_xavier(&mut rng, SLOT_DIM, SLOT_DIM),
            scale: 1.0 / (SLOT_DIM as f32).sqrt(),
        }
    }

    /// Computes cross-slot attention messages.
    ///
    /// For each active slot, computes scaled dot-product attention over
    /// all active slots and returns the aggregated value vectors as messages.
    /// Inactive slots (None) receive zero messages.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::Internal`] if any computation produces NaN or Inf.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_soft::attention::SlotAttention;
    /// use volt_core::{MAX_SLOTS, SLOT_DIM};
    ///
    /// let attn = SlotAttention::new_random(42);
    /// let states = [const { None }; MAX_SLOTS];
    /// let messages = attn.forward(&states).unwrap();
    /// // All messages are zero for empty states
    /// for msg in &messages {
    ///     assert!(msg.iter().all(|&x| x == 0.0));
    /// }
    /// ```
    pub fn forward(
        &self,
        states: &[Option<[f32; SLOT_DIM]>; MAX_SLOTS],
    ) -> Result<[[f32; SLOT_DIM]; MAX_SLOTS], VoltError> {
        let mut messages = [[0.0f32; SLOT_DIM]; MAX_SLOTS];

        // Collect active slot indices and their states
        let active: Vec<(usize, &[f32; SLOT_DIM])> = states
            .iter()
            .enumerate()
            .filter_map(|(i, s)| s.as_ref().map(|v| (i, v)))
            .collect();

        if active.is_empty() {
            return Ok(messages);
        }

        // Compute Q, K, V projections for all active slots
        let qs: Vec<Vec<f32>> = active.iter().map(|(_, s)| self.wq.forward(*s)).collect();
        let ks: Vec<Vec<f32>> = active.iter().map(|(_, s)| self.wk.forward(*s)).collect();
        let vs: Vec<Vec<f32>> = active.iter().map(|(_, s)| self.wv.forward(*s)).collect();

        // For each query slot, compute attention weights and aggregate values
        for (qi, &(slot_i, _)) in active.iter().enumerate() {
            // Compute attention scores: Q_i · K_j / sqrt(d)
            let mut scores = vec![0.0f32; active.len()];
            for (kj, _) in active.iter().enumerate() {
                let dot: f32 = qs[qi].iter().zip(ks[kj].iter()).map(|(a, b)| a * b).sum();
                scores[kj] = dot * self.scale;
            }

            // Softmax with numerical stability (subtract max)
            let max_score = scores
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let mut exp_sum = 0.0f32;
            for s in &mut scores {
                *s = (*s - max_score).exp();
                exp_sum += *s;
            }
            if exp_sum < 1e-10 {
                // All scores are -inf or zero; skip this slot
                continue;
            }
            for s in &mut scores {
                *s /= exp_sum;
            }

            // Weighted sum of values
            for (vj, &weight) in scores.iter().enumerate() {
                for d in 0..SLOT_DIM {
                    messages[slot_i][d] += weight * vs[vj][d];
                }
            }
        }

        // Validate output
        for (i, msg) in messages.iter().enumerate() {
            if msg.iter().any(|x| !x.is_finite()) {
                return Err(VoltError::Internal {
                    message: format!(
                        "SlotAttention forward: message for slot {} contains NaN or Inf",
                        i
                    ),
                });
            }
        }

        Ok(messages)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_vector(dim_index: usize) -> [f32; SLOT_DIM] {
        let mut v = [0.0f32; SLOT_DIM];
        v[dim_index] = 1.0;
        v
    }

    fn random_vector(seed: u64) -> [f32; SLOT_DIM] {
        let mut rng = Rng::new(seed);
        let mut v = [0.0f32; SLOT_DIM];
        for x in &mut v {
            *x = rng.next_f32_range(-1.0, 1.0);
        }
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for x in &mut v {
                *x /= norm;
            }
        }
        v
    }

    #[test]
    fn empty_states_produce_zero_messages() {
        let attn = SlotAttention::new_random(42);
        let states = [const { None }; MAX_SLOTS];
        let messages = attn.forward(&states).unwrap();
        for msg in &messages {
            for &x in msg {
                assert_eq!(x, 0.0);
            }
        }
    }

    #[test]
    fn single_slot_self_attention() {
        let attn = SlotAttention::new_random(42);
        let mut states = [const { None }; MAX_SLOTS];
        states[0] = Some(unit_vector(0));
        let messages = attn.forward(&states).unwrap();
        // Slot 0 should have a non-zero message (self-attention)
        assert!(
            messages[0].iter().any(|&x| x != 0.0),
            "single-slot self-attention should produce non-zero message"
        );
        // All other slots should have zero messages
        for msg in &messages[1..] {
            assert!(msg.iter().all(|&x| x == 0.0));
        }
    }

    #[test]
    fn messages_are_finite() {
        let attn = SlotAttention::new_random(42);
        let mut states = [const { None }; MAX_SLOTS];
        for i in 0..8 {
            states[i] = Some(random_vector(i as u64 + 100));
        }
        let messages = attn.forward(&states).unwrap();
        for msg in &messages {
            assert!(msg.iter().all(|x| x.is_finite()));
        }
    }

    #[test]
    fn deterministic_output() {
        let attn1 = SlotAttention::new_random(42);
        let attn2 = SlotAttention::new_random(42);
        let mut states = [const { None }; MAX_SLOTS];
        states[0] = Some(unit_vector(0));
        states[3] = Some(unit_vector(3));
        let msg1 = attn1.forward(&states).unwrap();
        let msg2 = attn2.forward(&states).unwrap();
        assert_eq!(msg1, msg2);
    }

    #[test]
    fn more_slots_produce_different_messages() {
        let attn = SlotAttention::new_random(42);

        // One slot
        let mut states1 = [const { None }; MAX_SLOTS];
        states1[0] = Some(random_vector(100));
        let msg1 = attn.forward(&states1).unwrap();

        // Same slot plus another
        let mut states2 = [const { None }; MAX_SLOTS];
        states2[0] = Some(random_vector(100));
        states2[1] = Some(random_vector(200));
        let msg2 = attn.forward(&states2).unwrap();

        // Messages for slot 0 should differ (new slot influences attention)
        assert_ne!(msg1[0], msg2[0]);
    }

    #[test]
    fn debug_format_readable() {
        let attn = SlotAttention::new_random(42);
        let debug = format!("{:?}", attn);
        assert!(debug.contains("SlotAttention"));
        assert!(debug.contains("dim=256"));
    }
}
