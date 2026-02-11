//! Garbage Collection pipeline with retention scoring and 4-tier decay.
//!
//! Frames decay through levels: Full → Compressed → Gist → Tombstoned.
//! The retention score determines when a frame should be demoted.
//!
//! ## Retention Score
//!
//! ```text
//! score = 0.40 * exp(-age_days / 30.0)   // recency decay
//!       + 0.35 * gamma                     // certainty
//!       + 0.15 * ln(1 + refs)             // reference count
//!       + 0.10 * pinned_bonus             // 1.0 if pinned, else 0.0
//! ```
//!
//! ## Decay Thresholds
//!
//! - Full → Compressed: score < 0.7
//! - Compressed → Gist: score < 0.4
//! - Gist → Tombstoned: score < 0.1

use std::collections::{HashMap, HashSet};

use crate::compressed::DecayLevel;

/// Microseconds per day.
const MICROS_PER_DAY: f64 = 86_400_000_000.0;

/// GC configuration.
///
/// # Example
///
/// ```
/// use volt_db::gc::GcConfig;
///
/// let config = GcConfig::default();
/// assert_eq!(config.tau_days, 30.0);
/// ```
#[derive(Debug, Clone)]
pub struct GcConfig {
    /// Weight for age decay factor. Default: 0.40.
    pub w_age: f64,
    /// Weight for certainty (gamma). Default: 0.35.
    pub w_gamma: f64,
    /// Weight for reference count. Default: 0.15.
    pub w_refs: f64,
    /// Weight for pinned bonus. Default: 0.10.
    pub w_pinned: f64,
    /// Time constant for age decay in days. Default: 30.0.
    pub tau_days: f64,
    /// Threshold: Full → Compressed. Default: 0.7.
    pub threshold_full_to_compressed: f64,
    /// Threshold: Compressed → Gist. Default: 0.4.
    pub threshold_compressed_to_gist: f64,
    /// Threshold: Gist → Tombstoned. Default: 0.1.
    pub threshold_gist_to_tombstone: f64,
}

impl Default for GcConfig {
    fn default() -> Self {
        Self {
            w_age: 0.40,
            w_gamma: 0.35,
            w_refs: 0.15,
            w_pinned: 0.10,
            tau_days: 30.0,
            threshold_full_to_compressed: 0.7,
            threshold_compressed_to_gist: 0.4,
            threshold_gist_to_tombstone: 0.1,
        }
    }
}

/// Metadata about a frame needed for GC scoring.
///
/// # Example
///
/// ```
/// use volt_db::gc::FrameGcMeta;
/// use volt_db::compressed::DecayLevel;
///
/// let meta = FrameGcMeta {
///     frame_id: 1,
///     strand_id: 0,
///     created_at: 0,
///     global_certainty: 0.5,
///     current_level: DecayLevel::Full,
///     reference_count: 0,
///     is_pinned: false,
///     is_wisdom: false,
/// };
/// assert_eq!(meta.frame_id, 1);
/// ```
#[derive(Debug, Clone)]
pub struct FrameGcMeta {
    /// Frame identifier.
    pub frame_id: u64,
    /// Strand identifier.
    pub strand_id: u64,
    /// Creation timestamp (microseconds since epoch).
    pub created_at: u64,
    /// Global certainty score (gamma, 0.0 - 1.0).
    pub global_certainty: f32,
    /// Current decay level.
    pub current_level: DecayLevel,
    /// How many other frames reference this one.
    pub reference_count: u32,
    /// Whether the user has pinned this frame.
    pub is_pinned: bool,
    /// Whether this is a consolidated wisdom frame.
    pub is_wisdom: bool,
}

/// Result of a GC evaluation pass.
///
/// # Example
///
/// ```
/// use volt_db::gc::GcResult;
///
/// let result = GcResult::default();
/// assert_eq!(result.frames_compressed, 0);
/// ```
#[derive(Debug, Clone, Default)]
pub struct GcResult {
    /// Number of frames demoted Full → Compressed.
    pub frames_compressed: usize,
    /// Number of frames demoted Compressed → Gist.
    pub frames_gisted: usize,
    /// Number of frames demoted Gist → Tombstoned.
    pub frames_tombstoned: usize,
    /// Number of frames preserved at their current level.
    pub frames_preserved: usize,
}

/// The GC engine evaluates retention scores and decides decay levels.
///
/// # Example
///
/// ```
/// use volt_db::gc::{GcEngine, FrameGcMeta};
/// use volt_db::compressed::DecayLevel;
///
/// let engine = GcEngine::with_defaults();
///
/// let meta = FrameGcMeta {
///     frame_id: 1,
///     strand_id: 0,
///     created_at: 0,       // very old
///     global_certainty: 0.0, // low gamma
///     current_level: DecayLevel::Full,
///     reference_count: 0,
///     is_pinned: false,
///     is_wisdom: false,
/// };
///
/// // Very old, zero gamma → should be demoted
/// let now = 100 * 86_400_000_000u64; // 100 days later
/// let score = engine.retention_score(&meta, now);
/// assert!(score < 0.7);
/// ```
#[derive(Debug, Clone)]
pub struct GcEngine {
    config: GcConfig,
    /// Set of pinned frame IDs.
    pinned: HashSet<u64>,
    /// Reference counts per frame.
    ref_counts: HashMap<u64, u32>,
}

impl GcEngine {
    /// Creates a new GC engine with the given configuration.
    pub fn new(config: GcConfig) -> Self {
        Self {
            config,
            pinned: HashSet::new(),
            ref_counts: HashMap::new(),
        }
    }

    /// Creates a new GC engine with default configuration.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::gc::GcEngine;
    ///
    /// let engine = GcEngine::with_defaults();
    /// ```
    pub fn with_defaults() -> Self {
        Self::new(GcConfig::default())
    }

    /// Computes the retention score for a frame.
    ///
    /// Higher scores mean the frame should be kept longer.
    /// Score range: approximately 0.0 to 1.0.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::gc::{GcEngine, FrameGcMeta};
    /// use volt_db::compressed::DecayLevel;
    ///
    /// let engine = GcEngine::with_defaults();
    ///
    /// // Recent, high-gamma frame → high score
    /// let meta = FrameGcMeta {
    ///     frame_id: 1, strand_id: 0,
    ///     created_at: 1000,
    ///     global_certainty: 0.9,
    ///     current_level: DecayLevel::Full,
    ///     reference_count: 5,
    ///     is_pinned: false, is_wisdom: false,
    /// };
    /// let score = engine.retention_score(&meta, 1000);
    /// assert!(score > 0.7);
    /// ```
    pub fn retention_score(&self, meta: &FrameGcMeta, now: u64) -> f64 {
        // Immortal frames always get max score
        if meta.is_pinned || self.pinned.contains(&meta.frame_id) {
            return 1.0;
        }
        if meta.global_certainty >= 1.0 {
            return 1.0;
        }
        if meta.is_wisdom {
            return 1.0;
        }

        let age_micros = now.saturating_sub(meta.created_at) as f64;
        let age_days = age_micros / MICROS_PER_DAY;

        let ref_count = self
            .ref_counts
            .get(&meta.frame_id)
            .copied()
            .unwrap_or(meta.reference_count);

        let pinned_bonus = if self.pinned.contains(&meta.frame_id) {
            1.0
        } else {
            0.0
        };

        let score = self.config.w_age * (-age_days / self.config.tau_days).exp()
            + self.config.w_gamma * (meta.global_certainty as f64)
            + self.config.w_refs * (1.0 + ref_count as f64).ln()
            + self.config.w_pinned * pinned_bonus;

        score.clamp(0.0, 1.0)
    }

    /// Determines the target decay level for a given retention score.
    ///
    /// The target level is never higher (more retained) than the current level.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::gc::GcEngine;
    /// use volt_db::compressed::DecayLevel;
    ///
    /// let engine = GcEngine::with_defaults();
    ///
    /// // Score below Full→Compressed threshold
    /// let target = engine.target_level(0.5, DecayLevel::Full);
    /// assert_eq!(target, DecayLevel::Compressed);
    ///
    /// // Already compressed, score below Compressed→Gist threshold
    /// let target = engine.target_level(0.3, DecayLevel::Compressed);
    /// assert_eq!(target, DecayLevel::Gist);
    /// ```
    pub fn target_level(&self, score: f64, current: DecayLevel) -> DecayLevel {
        let target = if score >= self.config.threshold_full_to_compressed {
            DecayLevel::Full
        } else if score >= self.config.threshold_compressed_to_gist {
            DecayLevel::Compressed
        } else if score >= self.config.threshold_gist_to_tombstone {
            DecayLevel::Gist
        } else {
            DecayLevel::Tombstoned
        };

        // Never promote — only demote
        if target > current {
            current
        } else {
            target
        }
    }

    /// Pins a frame so it is never garbage collected.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::gc::GcEngine;
    ///
    /// let mut engine = GcEngine::with_defaults();
    /// engine.pin_frame(42);
    /// assert!(engine.is_pinned(42));
    /// ```
    pub fn pin_frame(&mut self, frame_id: u64) {
        self.pinned.insert(frame_id);
    }

    /// Unpins a frame, making it eligible for GC.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::gc::GcEngine;
    ///
    /// let mut engine = GcEngine::with_defaults();
    /// engine.pin_frame(42);
    /// engine.unpin_frame(42);
    /// assert!(!engine.is_pinned(42));
    /// ```
    pub fn unpin_frame(&mut self, frame_id: u64) {
        self.pinned.remove(&frame_id);
    }

    /// Checks whether a frame is pinned.
    pub fn is_pinned(&self, frame_id: u64) -> bool {
        self.pinned.contains(&frame_id)
    }

    /// Adds a reference to a frame (another frame cites it).
    pub fn add_reference(&mut self, frame_id: u64) {
        *self.ref_counts.entry(frame_id).or_insert(0) += 1;
    }

    /// Removes a reference from a frame.
    pub fn remove_reference(&mut self, frame_id: u64) {
        if let Some(count) = self.ref_counts.get_mut(&frame_id) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                self.ref_counts.remove(&frame_id);
            }
        }
    }

    /// Evaluates a batch of frames and returns their target decay levels.
    ///
    /// Only returns entries where the target level differs from the current level.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::gc::{GcEngine, FrameGcMeta, GcResult};
    /// use volt_db::compressed::DecayLevel;
    ///
    /// let engine = GcEngine::with_defaults();
    /// let now = 100 * 86_400_000_000u64; // 100 days
    ///
    /// let frames = vec![
    ///     FrameGcMeta {
    ///         frame_id: 1, strand_id: 0, created_at: 0,
    ///         global_certainty: 0.0, current_level: DecayLevel::Full,
    ///         reference_count: 0, is_pinned: false, is_wisdom: false,
    ///     },
    /// ];
    ///
    /// let demotions = engine.evaluate(&frames, now);
    /// assert!(!demotions.is_empty());
    /// ```
    pub fn evaluate(&self, frames: &[FrameGcMeta], now: u64) -> Vec<(u64, DecayLevel)> {
        let mut demotions = Vec::new();
        for meta in frames {
            let score = self.retention_score(meta, now);
            let target = self.target_level(score, meta.current_level);
            if target != meta.current_level {
                demotions.push((meta.frame_id, target));
            }
        }
        demotions
    }

    /// Returns a reference to the GC configuration.
    pub fn config(&self) -> &GcConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: microseconds for N days.
    fn days(n: u64) -> u64 {
        n * 86_400_000_000
    }

    fn fresh_frame(id: u64, gamma: f32) -> FrameGcMeta {
        FrameGcMeta {
            frame_id: id,
            strand_id: 0,
            created_at: days(100), // created at day 100
            global_certainty: gamma,
            current_level: DecayLevel::Full,
            reference_count: 0,
            is_pinned: false,
            is_wisdom: false,
        }
    }

    #[test]
    fn new_frame_high_retention() {
        let engine = GcEngine::with_defaults();
        // Frame created just now, high gamma
        let meta = fresh_frame(1, 0.9);
        let now = days(100); // same time as creation
        let score = engine.retention_score(&meta, now);
        // age=0 → exp(0)=1.0 → 0.40*1.0 + 0.35*0.9 + 0.15*ln(1) + 0.10*0
        // = 0.40 + 0.315 + 0 + 0 = 0.715
        assert!(score > 0.7, "score {score} should be > 0.7");
    }

    #[test]
    fn old_low_gamma_frame_tombstoned() {
        let engine = GcEngine::with_defaults();
        let meta = FrameGcMeta {
            frame_id: 1,
            strand_id: 0,
            created_at: 0,           // very old
            global_certainty: 0.0,   // zero gamma
            current_level: DecayLevel::Full,
            reference_count: 0,
            is_pinned: false,
            is_wisdom: false,
        };
        let now = days(365); // 1 year later
        let score = engine.retention_score(&meta, now);
        assert!(score < 0.1, "score {score} should be < 0.1 for old, low-gamma");

        let target = engine.target_level(score, DecayLevel::Full);
        assert_eq!(target, DecayLevel::Tombstoned);
    }

    #[test]
    fn pinned_frame_immortal() {
        let mut engine = GcEngine::with_defaults();
        engine.pin_frame(1);

        let meta = FrameGcMeta {
            frame_id: 1,
            strand_id: 0,
            created_at: 0,
            global_certainty: 0.0,
            current_level: DecayLevel::Full,
            reference_count: 0,
            is_pinned: false, // Will be checked via engine.pinned set
            is_wisdom: false,
        };
        let now = days(365);
        let score = engine.retention_score(&meta, now);
        assert_eq!(score, 1.0, "pinned frame should have max score");
    }

    #[test]
    fn high_gamma_immortal() {
        let engine = GcEngine::with_defaults();
        let meta = FrameGcMeta {
            frame_id: 1,
            strand_id: 0,
            created_at: 0,
            global_certainty: 1.0, // absolute certainty
            current_level: DecayLevel::Full,
            reference_count: 0,
            is_pinned: false,
            is_wisdom: false,
        };
        let now = days(365);
        let score = engine.retention_score(&meta, now);
        assert_eq!(score, 1.0);
    }

    #[test]
    fn wisdom_frame_immortal() {
        let engine = GcEngine::with_defaults();
        let meta = FrameGcMeta {
            frame_id: 1,
            strand_id: 0,
            created_at: 0,
            global_certainty: 0.0,
            current_level: DecayLevel::Full,
            reference_count: 0,
            is_pinned: false,
            is_wisdom: true,
        };
        let now = days(365);
        let score = engine.retention_score(&meta, now);
        assert_eq!(score, 1.0);
    }

    #[test]
    fn reference_count_boosts_retention() {
        let engine = GcEngine::with_defaults();

        let meta_no_refs = fresh_frame(1, 0.3);
        let meta_with_refs = FrameGcMeta {
            reference_count: 100,
            ..fresh_frame(2, 0.3)
        };

        let now = days(130); // 30 days after creation
        let score_no = engine.retention_score(&meta_no_refs, now);
        let score_with = engine.retention_score(&meta_with_refs, now);

        assert!(
            score_with > score_no,
            "refs should boost: {score_with} vs {score_no}"
        );
    }

    #[test]
    fn decay_only_decreases() {
        let engine = GcEngine::with_defaults();

        // Score would map to Full, but current is Compressed
        let target = engine.target_level(0.9, DecayLevel::Compressed);
        assert_eq!(target, DecayLevel::Compressed, "should not promote");

        // Score would map to Gist, but current is Tombstoned
        let target = engine.target_level(0.3, DecayLevel::Tombstoned);
        assert_eq!(target, DecayLevel::Tombstoned, "should not promote");
    }

    #[test]
    fn evaluate_returns_demotions_only() {
        let engine = GcEngine::with_defaults();
        let now = days(200); // 200 days

        let frames = vec![
            // Old, low gamma → should be demoted
            FrameGcMeta {
                frame_id: 1,
                strand_id: 0,
                created_at: 0,
                global_certainty: 0.0,
                current_level: DecayLevel::Full,
                reference_count: 0,
                is_pinned: false,
                is_wisdom: false,
            },
            // Recent, high gamma → should be preserved
            FrameGcMeta {
                frame_id: 2,
                strand_id: 0,
                created_at: days(200),
                global_certainty: 0.9,
                current_level: DecayLevel::Full,
                reference_count: 0,
                is_pinned: false,
                is_wisdom: false,
            },
        ];

        let demotions = engine.evaluate(&frames, now);

        // Frame 1 should be demoted, frame 2 should not
        assert!(
            demotions.iter().any(|&(id, _)| id == 1),
            "old frame should be demoted"
        );
        assert!(
            !demotions.iter().any(|&(id, _)| id == 2),
            "recent frame should be preserved"
        );
    }

    #[test]
    fn pin_unpin() {
        let mut engine = GcEngine::with_defaults();
        assert!(!engine.is_pinned(42));
        engine.pin_frame(42);
        assert!(engine.is_pinned(42));
        engine.unpin_frame(42);
        assert!(!engine.is_pinned(42));
    }

    #[test]
    fn add_remove_reference() {
        let mut engine = GcEngine::with_defaults();
        engine.add_reference(1);
        engine.add_reference(1);
        engine.add_reference(1);

        let meta = fresh_frame(1, 0.5);
        let now = days(100);
        let score_with = engine.retention_score(&meta, now);

        engine.remove_reference(1);
        engine.remove_reference(1);
        engine.remove_reference(1);

        let score_without = engine.retention_score(&meta, now);

        assert!(score_with > score_without);
    }

    #[test]
    fn threshold_boundaries() {
        let engine = GcEngine::with_defaults();

        // Exactly at threshold
        assert_eq!(engine.target_level(0.7, DecayLevel::Full), DecayLevel::Full);
        assert_eq!(
            engine.target_level(0.699, DecayLevel::Full),
            DecayLevel::Compressed
        );
        assert_eq!(
            engine.target_level(0.4, DecayLevel::Compressed),
            DecayLevel::Compressed
        );
        assert_eq!(
            engine.target_level(0.399, DecayLevel::Compressed),
            DecayLevel::Gist
        );
        assert_eq!(engine.target_level(0.1, DecayLevel::Gist), DecayLevel::Gist);
        assert_eq!(
            engine.target_level(0.099, DecayLevel::Gist),
            DecayLevel::Tombstoned
        );
    }
}
