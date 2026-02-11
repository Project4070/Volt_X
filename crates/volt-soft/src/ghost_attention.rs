//! Ghost frame cross-attention for the RAR Attend phase.
//!
//! Ghost gists (R₀ vectors from the Bleed Buffer) participate as
//! additional Key/Value sources in cross-slot attention. The ghost
//! attention operates with a **separate softmax** and is blended with
//! the standard slot attention via an alpha weight:
//!
//! ```text
//! final_msg_i = (1 - α) × slot_msg_i + α × ghost_msg_i
//! ```
//!
//! This ensures ghost frames provide subtle memory influence without
//! destabilizing the primary slot-to-slot attention dynamics.

use crate::attention::SlotAttention;
use volt_core::{VoltError, MAX_SLOTS, SLOT_DIM};

/// Configuration for ghost frame attention blending.
///
/// # Example
///
/// ```
/// use volt_soft::ghost_attention::GhostAttentionConfig;
///
/// let config = GhostAttentionConfig::default();
/// assert!((config.alpha - 0.1).abs() < 1e-6);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct GhostAttentionConfig {
    /// Weight for ghost attention messages (0.0 = no ghost influence, 1.0 = ghost only).
    ///
    /// Default: 0.1 (subtle memory influence).
    pub alpha: f32,
}

impl Default for GhostAttentionConfig {
    fn default() -> Self {
        Self { alpha: 0.1 }
    }
}

/// Computes cross-slot attention with ghost frame participation.
///
/// Ghost gists are additional Key/Value sources. Each active slot's
/// query attends over ghost gists in a separate softmax, producing
/// ghost messages. These are blended with the normal slot attention
/// messages via the alpha weight.
///
/// # Arguments
///
/// * `attention` — the existing [`SlotAttention`] module (provides Q/K/V projections)
/// * `states` — current slot states at the target resolution
/// * `ghost_gists` — R₀ gist vectors from the ghost bleed buffer
/// * `config` — ghost attention configuration (alpha weight)
///
/// # Returns
///
/// `[[f32; SLOT_DIM]; MAX_SLOTS]` blended messages, same shape as
/// [`SlotAttention::forward`].
///
/// # Behavior
///
/// - Empty `ghost_gists` or `alpha == 0.0` → returns `attention.forward(states)` unchanged.
/// - `alpha == 1.0` → returns only ghost attention messages (slot messages discarded).
///
/// # Errors
///
/// Returns [`VoltError::Internal`] if any output message contains NaN or Inf.
///
/// # Example
///
/// ```
/// use volt_soft::attention::SlotAttention;
/// use volt_soft::ghost_attention::{forward_with_ghosts, GhostAttentionConfig};
/// use volt_core::{MAX_SLOTS, SLOT_DIM};
///
/// let attn = SlotAttention::new_random(42);
/// let mut states = [const { None }; MAX_SLOTS];
/// states[0] = Some([0.1_f32; SLOT_DIM]);
///
/// // No ghosts → same as forward()
/// let msgs = forward_with_ghosts(&attn, &states, &[], &GhostAttentionConfig::default()).unwrap();
/// let normal = attn.forward(&states).unwrap();
/// assert_eq!(msgs, normal);
/// ```
pub fn forward_with_ghosts(
    attention: &SlotAttention,
    states: &[Option<[f32; SLOT_DIM]>; MAX_SLOTS],
    ghost_gists: &[[f32; SLOT_DIM]],
    config: &GhostAttentionConfig,
) -> Result<[[f32; SLOT_DIM]; MAX_SLOTS], VoltError> {
    // Step 1: Compute normal slot-to-slot attention
    let slot_messages = attention.forward(states)?;

    // Early exit if no ghosts or alpha is effectively zero
    if ghost_gists.is_empty() || config.alpha <= 0.0 {
        return Ok(slot_messages);
    }

    // Step 2: Compute ghost attention
    let (wq, wk, wv) = attention.projections();
    let scale = 1.0 / (SLOT_DIM as f32).sqrt();

    // Collect active slots
    let active: Vec<(usize, &[f32; SLOT_DIM])> = states
        .iter()
        .enumerate()
        .filter_map(|(i, s)| s.as_ref().map(|v| (i, v)))
        .collect();

    if active.is_empty() {
        return Ok(slot_messages);
    }

    // Project slot queries
    let qs: Vec<Vec<f32>> = active.iter().map(|(_, s)| wq.forward(*s)).collect();

    // Project ghost gists as keys and values
    let ghost_ks: Vec<Vec<f32>> = ghost_gists.iter().map(|g| wk.forward(g)).collect();
    let ghost_vs: Vec<Vec<f32>> = ghost_gists.iter().map(|g| wv.forward(g)).collect();

    // Compute ghost attention for each active slot
    let mut ghost_messages = [[0.0f32; SLOT_DIM]; MAX_SLOTS];

    for (qi, &(slot_i, _)) in active.iter().enumerate() {
        // Compute Q·K scores for this slot against all ghosts
        let mut scores = vec![0.0f32; ghost_gists.len()];
        for (gj, gk) in ghost_ks.iter().enumerate() {
            let dot: f32 = qs[qi].iter().zip(gk.iter()).map(|(a, b)| a * b).sum();
            scores[gj] = dot * scale;
        }

        // Softmax with numerical stability
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
            continue; // All scores underflowed; skip this slot
        }
        for s in &mut scores {
            *s /= exp_sum;
        }

        // Weighted sum of ghost values
        for (gj, gv) in ghost_vs.iter().enumerate() {
            let weight = scores[gj];
            for (d, gv_d) in gv.iter().enumerate() {
                ghost_messages[slot_i][d] += weight * gv_d;
            }
        }
    }

    // Step 3: Blend slot and ghost messages
    let alpha = config.alpha.clamp(0.0, 1.0);
    let one_minus_alpha = 1.0 - alpha;
    let mut blended = [[0.0f32; SLOT_DIM]; MAX_SLOTS];
    for i in 0..MAX_SLOTS {
        for d in 0..SLOT_DIM {
            blended[i][d] = one_minus_alpha * slot_messages[i][d] + alpha * ghost_messages[i][d];
        }
    }

    // Validate output
    for (i, msg) in blended.iter().enumerate() {
        if msg.iter().any(|x| !x.is_finite()) {
            return Err(VoltError::Internal {
                message: format!(
                    "forward_with_ghosts: message for slot {} contains NaN or Inf",
                    i
                ),
            });
        }
    }

    Ok(blended)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_states(n: usize, value: f32) -> [Option<[f32; SLOT_DIM]>; MAX_SLOTS] {
        let mut states = [const { None }; MAX_SLOTS];
        for i in 0..n.min(MAX_SLOTS) {
            // Slightly different values per slot to break symmetry
            let mut v = [value; SLOT_DIM];
            v[i] = value + 0.01 * (i as f32);
            states[i] = Some(v);
        }
        states
    }

    fn make_ghost_gist(dim: usize) -> [f32; SLOT_DIM] {
        let mut v = [0.0f32; SLOT_DIM];
        v[dim % SLOT_DIM] = 1.0;
        v
    }

    #[test]
    fn empty_ghosts_equals_forward() {
        let attn = SlotAttention::new_random(42);
        let states = make_states(3, 0.1);
        let config = GhostAttentionConfig::default();

        let normal = attn.forward(&states).unwrap();
        let with_ghosts = forward_with_ghosts(&attn, &states, &[], &config).unwrap();

        assert_eq!(normal, with_ghosts);
    }

    #[test]
    fn alpha_zero_equals_forward() {
        let attn = SlotAttention::new_random(42);
        let states = make_states(2, 0.1);
        let config = GhostAttentionConfig { alpha: 0.0 };
        let ghosts = [make_ghost_gist(0), make_ghost_gist(50)];

        let normal = attn.forward(&states).unwrap();
        let with_ghosts = forward_with_ghosts(&attn, &states, &ghosts, &config).unwrap();

        assert_eq!(normal, with_ghosts);
    }

    #[test]
    fn ghosts_produce_different_messages() {
        let attn = SlotAttention::new_random(42);
        let states = make_states(2, 0.1);
        let config = GhostAttentionConfig { alpha: 0.5 };
        let ghosts = [make_ghost_gist(0), make_ghost_gist(128)];

        let normal = attn.forward(&states).unwrap();
        let with_ghosts = forward_with_ghosts(&attn, &states, &ghosts, &config).unwrap();

        // Messages should differ due to ghost influence
        assert_ne!(normal, with_ghosts);
    }

    #[test]
    fn output_is_deterministic() {
        let attn = SlotAttention::new_random(42);
        let states = make_states(2, 0.1);
        let config = GhostAttentionConfig { alpha: 0.3 };
        let ghosts = [make_ghost_gist(10)];

        let result1 = forward_with_ghosts(&attn, &states, &ghosts, &config).unwrap();
        let result2 = forward_with_ghosts(&attn, &states, &ghosts, &config).unwrap();

        assert_eq!(result1, result2);
    }

    #[test]
    fn output_is_finite() {
        let attn = SlotAttention::new_random(42);
        let states = make_states(4, 0.1);
        let config = GhostAttentionConfig { alpha: 0.5 };

        // Create several ghost gists
        let ghosts: Vec<[f32; SLOT_DIM]> = (0..10).map(|i| make_ghost_gist(i * 25)).collect();

        let result = forward_with_ghosts(&attn, &states, &ghosts, &config).unwrap();
        for msg in &result {
            assert!(msg.iter().all(|x| x.is_finite()));
        }
    }

    #[test]
    fn empty_states_returns_zeros() {
        let attn = SlotAttention::new_random(42);
        let states = [const { None }; MAX_SLOTS];
        let config = GhostAttentionConfig { alpha: 0.5 };
        let ghosts = [make_ghost_gist(0)];

        let result = forward_with_ghosts(&attn, &states, &ghosts, &config).unwrap();
        for msg in &result {
            assert!(msg.iter().all(|&x| x == 0.0));
        }
    }

    #[test]
    fn alpha_one_gives_only_ghost_messages() {
        let attn = SlotAttention::new_random(42);
        let states = make_states(2, 0.1);
        let config = GhostAttentionConfig { alpha: 1.0 };
        let ghosts = [make_ghost_gist(0), make_ghost_gist(128)];

        let result = forward_with_ghosts(&attn, &states, &ghosts, &config).unwrap();

        // At alpha=1.0, slot messages have weight 0 and ghost messages have weight 1.
        // Active slots should have non-zero messages (from ghosts only).
        assert!(result[0].iter().any(|&x| x != 0.0));
    }
}
