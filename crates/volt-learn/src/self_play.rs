//! Self-play logic puzzle generation and grading.
//!
//! Generates simple logic puzzles using deterministic rules and grades
//! the Volt pipeline's answers by comparing output frames to expected
//! conclusion frames via cosine similarity.
//!
//! ## Puzzle Types
//!
//! - **Modus Ponens**: "if A then B, A" → "B"
//! - **Transitivity**: "if A then B, if B then C" → "C"
//! - **Modus Tollens**: "if A then B, not B" → "not A"
//! - **Conjunction**: "A and B" → "A"
//! - **Disjunction**: "A or B, not A" → "B"

use volt_core::{TensorFrame, SLOT_DIM};

/// The type of logic rule tested by a puzzle.
///
/// # Example
///
/// ```
/// use volt_learn::self_play::PuzzleType;
///
/// let pt = PuzzleType::ModusPonens;
/// assert_eq!(format!("{pt:?}"), "ModusPonens");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PuzzleType {
    /// If A then B. A. Therefore B.
    ModusPonens,
    /// If A then B. If B then C. Therefore A implies C.
    Transitivity,
    /// If A then B. Not B. Therefore not A.
    ModusTollens,
    /// A and B. Therefore A (conjunction elimination).
    Conjunction,
    /// A or B. Not A. Therefore B (disjunctive syllogism).
    Disjunction,
}

/// A self-play logic puzzle with premises and expected conclusion.
///
/// # Example
///
/// ```
/// use volt_learn::self_play::{LogicPuzzle, PuzzleType};
///
/// let puzzle = LogicPuzzle {
///     premises: "if alpha then beta alpha".to_string(),
///     conclusion: "beta".to_string(),
///     puzzle_type: PuzzleType::ModusPonens,
/// };
/// assert_eq!(puzzle.puzzle_type, PuzzleType::ModusPonens);
/// ```
#[derive(Debug, Clone)]
pub struct LogicPuzzle {
    /// The premise text to encode as input.
    pub premises: String,
    /// The expected conclusion text.
    pub conclusion: String,
    /// Which logic rule this puzzle tests.
    pub puzzle_type: PuzzleType,
}

/// Result of grading a single puzzle.
///
/// # Example
///
/// ```
/// use volt_learn::self_play::{PuzzleResult, PuzzleType};
///
/// let result = PuzzleResult {
///     puzzle_type: PuzzleType::ModusPonens,
///     correct: true,
///     similarity: 0.85,
///     gamma: 0.9,
/// };
/// assert!(result.correct);
/// ```
#[derive(Debug, Clone)]
pub struct PuzzleResult {
    /// Which logic rule was tested.
    pub puzzle_type: PuzzleType,
    /// Whether the output met the similarity threshold.
    pub correct: bool,
    /// Cosine similarity between output and expected conclusion.
    pub similarity: f32,
    /// The global certainty of the output frame.
    pub gamma: f32,
}

/// Atomic propositions used in puzzle generation.
const PROPOSITIONS: &[&str] = &[
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta",
    "theta", "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron",
    "pi", "rho", "sigma", "tau", "upsilon", "phi", "chi", "psi",
    "omega",
];

/// Simple PRNG for deterministic puzzle generation (splitmix64).
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Returns an index in [0, max).
    fn next_usize(&mut self, max: usize) -> usize {
        (self.next_u64() as usize) % max
    }
}

/// Generates a set of deterministic logic puzzles.
///
/// Creates `count` puzzles with a deterministic distribution of types
/// using the given seed. Each puzzle uses atomic propositions from a
/// fixed vocabulary.
///
/// # Example
///
/// ```
/// use volt_learn::self_play::generate_puzzles;
///
/// let puzzles = generate_puzzles(100, 42);
/// assert_eq!(puzzles.len(), 100);
///
/// // Deterministic: same seed produces same puzzles
/// let puzzles2 = generate_puzzles(100, 42);
/// for (a, b) in puzzles.iter().zip(puzzles2.iter()) {
///     assert_eq!(a.premises, b.premises);
///     assert_eq!(a.conclusion, b.conclusion);
/// }
/// ```
pub fn generate_puzzles(count: usize, seed: u64) -> Vec<LogicPuzzle> {
    let mut rng = Rng::new(seed);
    let mut puzzles = Vec::with_capacity(count);
    let types = [
        PuzzleType::ModusPonens,
        PuzzleType::Transitivity,
        PuzzleType::ModusTollens,
        PuzzleType::Conjunction,
        PuzzleType::Disjunction,
    ];

    for _ in 0..count {
        let puzzle_type = types[rng.next_usize(types.len())];
        let puzzle = generate_one_puzzle(&mut rng, puzzle_type);
        puzzles.push(puzzle);
    }

    puzzles
}

/// Generates a single puzzle of the given type.
fn generate_one_puzzle(rng: &mut Rng, puzzle_type: PuzzleType) -> LogicPuzzle {
    // Pick distinct propositions
    let n = PROPOSITIONS.len();
    let a_idx = rng.next_usize(n);
    let mut b_idx = rng.next_usize(n);
    while b_idx == a_idx {
        b_idx = rng.next_usize(n);
    }
    let mut c_idx = rng.next_usize(n);
    while c_idx == a_idx || c_idx == b_idx {
        c_idx = rng.next_usize(n);
    }

    let a = PROPOSITIONS[a_idx];
    let b = PROPOSITIONS[b_idx];
    let c = PROPOSITIONS[c_idx];

    match puzzle_type {
        PuzzleType::ModusPonens => LogicPuzzle {
            premises: format!("if {a} then {b} {a}"),
            conclusion: b.to_string(),
            puzzle_type,
        },
        PuzzleType::Transitivity => LogicPuzzle {
            premises: format!("if {a} then {b} if {b} then {c}"),
            conclusion: c.to_string(),
            puzzle_type,
        },
        PuzzleType::ModusTollens => LogicPuzzle {
            premises: format!("if {a} then {b} not {b}"),
            conclusion: format!("not {a}"),
            puzzle_type,
        },
        PuzzleType::Conjunction => LogicPuzzle {
            premises: format!("{a} and {b}"),
            conclusion: a.to_string(),
            puzzle_type,
        },
        PuzzleType::Disjunction => LogicPuzzle {
            premises: format!("{a} or {b} not {a}"),
            conclusion: b.to_string(),
            puzzle_type,
        },
    }
}

/// Grades a puzzle by comparing the output frame to the expected frame.
///
/// Computes the average cosine similarity between matching active R₀
/// slots of the output and expected frames. Returns `true` if the
/// similarity meets or exceeds the threshold.
///
/// # Example
///
/// ```
/// use volt_learn::self_play::grade_puzzle;
/// use volt_core::TensorFrame;
///
/// let output = TensorFrame::new();
/// let expected = TensorFrame::new();
/// // Empty frames have no active slots, so similarity = 0.0
/// assert!(!grade_puzzle(&output, &expected, 0.5));
/// ```
pub fn grade_puzzle(output: &TensorFrame, expected: &TensorFrame, threshold: f32) -> bool {
    let sim = frame_cosine_similarity(output, expected);
    sim >= threshold
}

/// Computes average cosine similarity between matching active R₀ slots.
fn frame_cosine_similarity(a: &TensorFrame, b: &TensorFrame) -> f32 {
    let mut total_sim = 0.0f32;
    let mut count = 0usize;

    for i in 0..a.slots.len() {
        if let Some(slot_a) = &a.slots[i]
            && let Some(slot_b) = &b.slots[i]
            && let Some(r0_a) = &slot_a.resolutions[0]
            && let Some(r0_b) = &slot_b.resolutions[0]
        {
            let sim = cosine_sim(r0_a, r0_b);
            total_sim += sim;
            count += 1;
        }
    }

    if count == 0 {
        return 0.0;
    }
    total_sim / count as f32
}

/// Cosine similarity between two SLOT_DIM vectors.
fn cosine_sim(a: &[f32; SLOT_DIM], b: &[f32; SLOT_DIM]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use volt_core::slot::{SlotData, SlotRole};

    #[test]
    fn generate_correct_count() {
        let puzzles = generate_puzzles(100, 42);
        assert_eq!(puzzles.len(), 100);
    }

    #[test]
    fn generate_zero_puzzles() {
        let puzzles = generate_puzzles(0, 42);
        assert!(puzzles.is_empty());
    }

    #[test]
    fn puzzles_are_deterministic() {
        let p1 = generate_puzzles(50, 42);
        let p2 = generate_puzzles(50, 42);
        for (a, b) in p1.iter().zip(p2.iter()) {
            assert_eq!(a.premises, b.premises);
            assert_eq!(a.conclusion, b.conclusion);
            assert_eq!(a.puzzle_type, b.puzzle_type);
        }
    }

    #[test]
    fn different_seeds_different_puzzles() {
        let p1 = generate_puzzles(50, 42);
        let p2 = generate_puzzles(50, 99);
        let same_count = p1.iter().zip(p2.iter())
            .filter(|(a, b)| a.premises == b.premises)
            .count();
        assert!(same_count < 50, "different seeds should mostly differ");
    }

    #[test]
    fn all_puzzle_types_generated() {
        let puzzles = generate_puzzles(1000, 42);
        let has_mp = puzzles.iter().any(|p| p.puzzle_type == PuzzleType::ModusPonens);
        let has_tr = puzzles.iter().any(|p| p.puzzle_type == PuzzleType::Transitivity);
        let has_mt = puzzles.iter().any(|p| p.puzzle_type == PuzzleType::ModusTollens);
        let has_cj = puzzles.iter().any(|p| p.puzzle_type == PuzzleType::Conjunction);
        let has_dj = puzzles.iter().any(|p| p.puzzle_type == PuzzleType::Disjunction);
        assert!(has_mp, "should have ModusPonens");
        assert!(has_tr, "should have Transitivity");
        assert!(has_mt, "should have ModusTollens");
        assert!(has_cj, "should have Conjunction");
        assert!(has_dj, "should have Disjunction");
    }

    #[test]
    fn modus_ponens_structure() {
        let puzzles = generate_puzzles(1000, 42);
        let mp: Vec<_> = puzzles.iter()
            .filter(|p| p.puzzle_type == PuzzleType::ModusPonens)
            .collect();
        assert!(!mp.is_empty());
        for p in &mp {
            assert!(p.premises.contains("if"), "MP should contain 'if': {}", p.premises);
            assert!(p.premises.contains("then"), "MP should contain 'then': {}", p.premises);
        }
    }

    #[test]
    fn transitivity_structure() {
        let puzzles = generate_puzzles(1000, 42);
        let tr: Vec<_> = puzzles.iter()
            .filter(|p| p.puzzle_type == PuzzleType::Transitivity)
            .collect();
        assert!(!tr.is_empty());
        for p in &tr {
            // Should have two "if ... then ..." clauses
            let if_count = p.premises.matches("if").count();
            assert_eq!(if_count, 2, "transitivity should have 2 'if': {}", p.premises);
        }
    }

    #[test]
    fn modus_tollens_structure() {
        let puzzles = generate_puzzles(1000, 42);
        let mt: Vec<_> = puzzles.iter()
            .filter(|p| p.puzzle_type == PuzzleType::ModusTollens)
            .collect();
        assert!(!mt.is_empty());
        for p in &mt {
            assert!(p.premises.contains("not"), "MT should contain 'not': {}", p.premises);
            assert!(p.conclusion.starts_with("not"), "MT conclusion should start with 'not': {}", p.conclusion);
        }
    }

    #[test]
    fn grade_identical_frames_passes() {
        let mut frame = TensorFrame::new();
        let mut r0 = [0.0f32; SLOT_DIM];
        r0[0] = 1.0;
        let mut slot = SlotData::new(SlotRole::Agent);
        slot.write_resolution(0, r0);
        frame.slots[0] = Some(slot);

        assert!(grade_puzzle(&frame, &frame, 0.99));
    }

    #[test]
    fn grade_empty_frames_fails() {
        let f1 = TensorFrame::new();
        let f2 = TensorFrame::new();
        assert!(!grade_puzzle(&f1, &f2, 0.1));
    }

    #[test]
    fn grade_orthogonal_frames_fails() {
        let mut f1 = TensorFrame::new();
        let mut f2 = TensorFrame::new();

        let mut r0_a = [0.0f32; SLOT_DIM];
        r0_a[0] = 1.0;
        let mut r0_b = [0.0f32; SLOT_DIM];
        r0_b[1] = 1.0;

        let mut slot_a = SlotData::new(SlotRole::Agent);
        slot_a.write_resolution(0, r0_a);
        let mut slot_b = SlotData::new(SlotRole::Agent);
        slot_b.write_resolution(0, r0_b);
        f1.slots[0] = Some(slot_a);
        f2.slots[0] = Some(slot_b);

        assert!(!grade_puzzle(&f1, &f2, 0.5));
    }

    #[test]
    fn puzzle_result_creation() {
        let result = PuzzleResult {
            puzzle_type: PuzzleType::ModusPonens,
            correct: true,
            similarity: 0.85,
            gamma: 0.9,
        };
        assert!(result.correct);
        assert_eq!(result.puzzle_type, PuzzleType::ModusPonens);
    }

    #[test]
    fn rng_deterministic() {
        let mut r1 = Rng::new(42);
        let mut r2 = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(r1.next_u64(), r2.next_u64());
        }
    }
}
