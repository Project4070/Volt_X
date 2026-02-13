//! Code Frame Pair Generation for Phase 2 Training.
//!
//! Converts [`CodeProblem`] instances into [`FramePair`] training data
//! for the Phase 2 scaled VFN training pipeline. Supports curriculum
//! staging to progressively increase training difficulty.
//!
//! ## Curriculum Stages
//!
//! Problems are classified into difficulty tiers based on their metadata
//! and code complexity:
//!
//! 1. **Simple Functions**: Single operation, no loops, ≤20 lines
//! 2. **Loops and Conditionals**: Contains for/while/if, ≤50 lines
//! 3. **Multi-function Programs**: Multiple def/fn, ≤100 lines
//! 4. **Algorithmic Reasoning**: Complex logic, sorting, searching
//!
//! ## Usage
//!
//! ```no_run
//! use volt_learn::code_frame_pairs::{generate_code_frame_pairs, CodeFramePairConfig};
//! use volt_learn::code_dataset::CodeDataset;
//!
//! let dataset = CodeDataset::from_file("humaneval.jsonl").unwrap();
//! let config = CodeFramePairConfig::default();
//! let pairs = generate_code_frame_pairs(&dataset, &config).unwrap();
//! ```

use volt_core::{SlotRole, TensorFrame, VoltError, SLOT_DIM};

use crate::code_dataset::{CodeDataset, CodeProblem};

/// A (question, answer) TensorFrame pair for VFN training.
///
/// Structurally identical to `volt_soft::training::FramePair` (when the
/// `gpu` feature is enabled on volt-soft), but defined here to avoid
/// requiring GPU dependencies for frame pair generation.
///
/// # Example
///
/// ```
/// use volt_learn::code_frame_pairs::CodeFramePair;
/// use volt_core::TensorFrame;
///
/// let pair = CodeFramePair {
///     question: TensorFrame::new(),
///     answer: TensorFrame::new(),
/// };
/// ```
#[derive(Debug, Clone)]
pub struct CodeFramePair {
    /// The question/input frame (encoded from problem description).
    pub question: TensorFrame,
    /// The answer/target frame (encoded from solution code).
    pub answer: TensorFrame,
}

/// Type alias for backward compatibility with callers.
pub type FramePair = CodeFramePair;

/// Configuration for code frame pair generation.
///
/// # Example
///
/// ```
/// use volt_learn::code_frame_pairs::CodeFramePairConfig;
///
/// let config = CodeFramePairConfig::default();
/// assert_eq!(config.resolution, 0);
/// ```
#[derive(Debug, Clone)]
pub struct CodeFramePairConfig {
    /// Resolution to write frame data at (default: 0 = R₀).
    pub resolution: usize,

    /// Random seed for deterministic encoding.
    pub seed: u64,

    /// Maximum number of pairs to generate (0 = unlimited).
    pub max_pairs: usize,

    /// Filter to a specific curriculum stage (None = all stages).
    pub stage_filter: Option<CurriculumStageFilter>,
}

impl Default for CodeFramePairConfig {
    fn default() -> Self {
        Self {
            resolution: 0,
            seed: 42,
            max_pairs: 0,
            stage_filter: None,
        }
    }
}

/// Curriculum stage filter for frame pair generation.
///
/// # Example
///
/// ```
/// use volt_learn::code_frame_pairs::CurriculumStageFilter;
///
/// let filter = CurriculumStageFilter::SimpleFunctions;
/// assert_eq!(filter.label(), "simple_functions");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CurriculumStageFilter {
    /// Stage 1: Simple single-operation functions.
    SimpleFunctions,
    /// Stage 2: Programs with loops and conditionals.
    LoopsAndConditionals,
    /// Stage 3: Multi-function programs.
    MultiFunctionPrograms,
    /// Stage 4: Algorithmic reasoning.
    AlgorithmicReasoning,
}

impl CurriculumStageFilter {
    /// Returns a human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            Self::SimpleFunctions => "simple_functions",
            Self::LoopsAndConditionals => "loops_conditionals",
            Self::MultiFunctionPrograms => "multi_function",
            Self::AlgorithmicReasoning => "algorithmic",
        }
    }
}

/// Classifies a code problem into a curriculum stage.
///
/// Uses simple heuristics based on code structure:
/// - Line count
/// - Presence of control flow keywords
/// - Number of function definitions
///
/// # Example
///
/// ```
/// use volt_learn::code_frame_pairs::{classify_problem, CurriculumStageFilter};
/// use volt_learn::code_dataset::CodeProblem;
///
/// let problem = CodeProblem {
///     id: "test/1".to_string(),
///     query: "Add two numbers".to_string(),
///     solution: "def add(a, b): return a + b".to_string(),
///     tests: vec![],
///     language: Some("python".to_string()),
///     difficulty: Some("easy".to_string()),
/// };
///
/// let stage = classify_problem(&problem);
/// assert_eq!(stage, CurriculumStageFilter::SimpleFunctions);
/// ```
pub fn classify_problem(problem: &CodeProblem) -> CurriculumStageFilter {
    let solution = &problem.solution;
    let lines: Vec<&str> = solution.lines().collect();
    let line_count = lines.len();
    let solution_lower = solution.to_lowercase();

    // Count function definitions
    let fn_count = lines
        .iter()
        .filter(|l| {
            let trimmed = l.trim();
            trimmed.starts_with("def ")
                || trimmed.starts_with("fn ")
                || trimmed.starts_with("func ")
                || trimmed.starts_with("function ")
        })
        .count();

    // Check for control flow
    let has_loop = solution_lower.contains("for ")
        || solution_lower.contains("while ")
        || solution_lower.contains("loop ");
    let has_conditional =
        solution_lower.contains("if ") || solution_lower.contains("match ");

    // Check explicit difficulty
    if let Some(ref diff) = problem.difficulty {
        let diff_lower = diff.to_lowercase();
        if diff_lower == "hard" || diff_lower == "competition" {
            return CurriculumStageFilter::AlgorithmicReasoning;
        }
    }

    // Classification heuristics
    if fn_count > 1 || line_count > 50 {
        if line_count > 100 || solution_lower.contains("sort") || solution_lower.contains("search")
        {
            CurriculumStageFilter::AlgorithmicReasoning
        } else {
            CurriculumStageFilter::MultiFunctionPrograms
        }
    } else if has_loop || (has_conditional && line_count > 10) {
        CurriculumStageFilter::LoopsAndConditionals
    } else {
        CurriculumStageFilter::SimpleFunctions
    }
}

/// Generates Phase 2 training FramePairs from a code dataset.
///
/// Each code problem is encoded into (query_frame, solution_frame) pairs
/// using deterministic hash-based encoding. The encoding places semantic
/// content into TensorFrame slots following the code attention mapping:
///
/// - S0 (Agent): Function identity
/// - S1 (Predicate): Operation type
/// - S2 (Patient): Arguments/parameters
/// - S3 (Location): Return value concept
///
/// # Errors
///
/// Returns [`VoltError::LearnError`] if the dataset is empty.
///
/// # Example
///
/// ```no_run
/// use volt_learn::code_frame_pairs::{generate_code_frame_pairs, CodeFramePairConfig};
/// use volt_learn::code_dataset::CodeDataset;
///
/// let dataset = CodeDataset::from_file("humaneval.jsonl").unwrap();
/// let config = CodeFramePairConfig::default();
/// let pairs = generate_code_frame_pairs(&dataset, &config).unwrap();
/// assert!(!pairs.is_empty());
/// ```
pub fn generate_code_frame_pairs(
    dataset: &CodeDataset,
    config: &CodeFramePairConfig,
) -> Result<Vec<FramePair>, VoltError> {
    if dataset.is_empty() {
        return Err(VoltError::LearnError {
            message: "generate_code_frame_pairs: empty dataset".to_string(),
        });
    }

    let mut pairs = Vec::new();
    let mut seed_offset = config.seed;

    for problem in dataset.iter() {
        // Apply curriculum filter
        if let Some(stage_filter) = &config.stage_filter {
            let stage = classify_problem(problem);
            if stage != *stage_filter {
                continue;
            }
        }

        // Encode query and solution into TensorFrames
        let query_frame = encode_text_to_frame(&problem.query, seed_offset, config.resolution)?;
        seed_offset = seed_offset.wrapping_add(1);
        let answer_frame =
            encode_text_to_frame(&problem.solution, seed_offset, config.resolution)?;
        seed_offset = seed_offset.wrapping_add(1);

        pairs.push(FramePair {
            question: query_frame,
            answer: answer_frame,
        });

        if config.max_pairs > 0 && pairs.len() >= config.max_pairs {
            break;
        }
    }

    if pairs.is_empty() {
        return Err(VoltError::LearnError {
            message: "generate_code_frame_pairs: no pairs generated (check curriculum filter)"
                .to_string(),
        });
    }

    Ok(pairs)
}

/// Encodes a text string into a TensorFrame using deterministic hashing.
///
/// This is a lightweight encoding that produces distinct frames for distinct
/// inputs without requiring a full neural encoder. Used for Phase 2 training
/// where the encoder is frozen and we focus on VFN training.
///
/// # Encoding Strategy
///
/// 1. Hash the text to produce a seed
/// 2. Generate normalized 256-dim vectors for 4 code slots:
///    - S0 (Agent): Overall text identity
///    - S1 (Predicate): Character bigram features
///    - S2 (Patient): Word-level features
///    - S3 (Location): Length/structure features
///
/// # Errors
///
/// Returns [`VoltError::Internal`] if frame construction fails.
fn encode_text_to_frame(
    text: &str,
    seed: u64,
    resolution: usize,
) -> Result<TensorFrame, VoltError> {
    let mut frame = TensorFrame::new();

    // Hash the text for deterministic encoding
    let text_hash = hash_text(text, seed);

    // S0 (Agent): Overall identity — hash-derived vector
    let s0 = generate_vector_from_hash(text_hash, 0);
    frame.write_at(0, resolution, SlotRole::Agent, s0)?;

    // S1 (Predicate): Character bigram features
    let s1 = generate_bigram_vector(text, text_hash);
    frame.write_at(1, resolution, SlotRole::Predicate, s1)?;

    // S2 (Patient): Word-level features
    let s2 = generate_word_vector(text, text_hash);
    frame.write_at(2, resolution, SlotRole::Patient, s2)?;

    // S3 (Location): Structure features (length, special chars)
    let s3 = generate_structure_vector(text, text_hash);
    frame.write_at(3, resolution, SlotRole::Location, s3)?;

    // Normalize all written slots
    for slot_idx in 0..4 {
        frame.normalize_slot(slot_idx, resolution)?;
    }

    Ok(frame)
}

/// Simple text hashing using FNV-1a.
fn hash_text(text: &str, seed: u64) -> u64 {
    let mut hash = 0xcbf29ce484222325u64 ^ seed;
    for &byte in text.as_bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Generates a normalized 256-dim vector from a hash seed.
fn generate_vector_from_hash(hash: u64, variant: u64) -> [f32; SLOT_DIM] {
    let mut rng = crate::nn_rng::SimpleRng::new(hash.wrapping_add(variant));
    let mut vec = [0.0f32; SLOT_DIM];
    for x in &mut vec {
        *x = rng.next_f32_range(-1.0, 1.0);
    }
    normalize_vec(&mut vec);
    vec
}

/// Generates a vector based on character bigram statistics.
fn generate_bigram_vector(text: &str, seed: u64) -> [f32; SLOT_DIM] {
    let mut vec = [0.0f32; SLOT_DIM];
    let bytes = text.as_bytes();

    // Accumulate bigram activations into vector dimensions
    for window in bytes.windows(2) {
        let idx = ((window[0] as usize) * 256 + window[1] as usize) % SLOT_DIM;
        vec[idx] += 1.0;
    }

    // Mix in seed-derived noise for uniqueness
    let mut rng = crate::nn_rng::SimpleRng::new(seed.wrapping_add(1));
    for x in &mut vec {
        *x += rng.next_f32_range(-0.1, 0.1);
    }

    normalize_vec(&mut vec);
    vec
}

/// Generates a vector based on word-level features.
fn generate_word_vector(text: &str, seed: u64) -> [f32; SLOT_DIM] {
    let mut vec = [0.0f32; SLOT_DIM];

    // Hash each word and scatter into the vector
    for word in text.split_whitespace() {
        let word_hash = hash_text(word, seed.wrapping_add(2));
        let idx = (word_hash as usize) % SLOT_DIM;
        vec[idx] += 1.0;
        // Also activate a few neighboring dimensions
        vec[(idx + 1) % SLOT_DIM] += 0.5;
        vec[(idx + 2) % SLOT_DIM] += 0.25;
    }

    // Add seed-based noise
    let mut rng = crate::nn_rng::SimpleRng::new(seed.wrapping_add(3));
    for x in &mut vec {
        *x += rng.next_f32_range(-0.05, 0.05);
    }

    normalize_vec(&mut vec);
    vec
}

/// Generates a vector based on structural features.
fn generate_structure_vector(text: &str, seed: u64) -> [f32; SLOT_DIM] {
    let mut vec = [0.0f32; SLOT_DIM];

    // Feature 0-7: length buckets
    let len = text.len();
    vec[0] = (len as f32 / 10.0).tanh();
    vec[1] = (len as f32 / 100.0).tanh();
    vec[2] = (len as f32 / 1000.0).tanh();
    vec[3] = text.lines().count() as f32 / 50.0;

    // Feature 8-31: character class counts
    let count = |pred: fn(char) -> bool| -> f32 {
        text.chars().filter(|c| pred(*c)).count() as f32 / len.max(1) as f32
    };
    vec[8] = count(|c| c.is_alphabetic());
    vec[9] = count(|c| c.is_numeric());
    vec[10] = count(|c| c.is_whitespace());
    vec[11] = count(|c| c == '(');
    vec[12] = count(|c| c == ')');
    vec[13] = count(|c| c == '{');
    vec[14] = count(|c| c == '}');
    vec[15] = count(|c| c == ':');

    // Fill remaining with seed-derived values
    let mut rng = crate::nn_rng::SimpleRng::new(seed.wrapping_add(4));
    for x in vec.iter_mut().skip(16) {
        *x = rng.next_f32_range(-0.5, 0.5);
    }

    normalize_vec(&mut vec);
    vec
}

/// L2-normalizes a vector in place.
fn normalize_vec(vec: &mut [f32; SLOT_DIM]) {
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for x in vec.iter_mut() {
            *x /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_dataset(n: usize) -> CodeDataset {
        use std::io::Write;
        let temp_dir = std::env::temp_dir();
        let tid = std::thread::current().id();
        let path = temp_dir.join(format!("cfp_test_{n}_{tid:?}.jsonl"));

        let mut file = std::fs::File::create(&path).unwrap();
        for i in 0..n {
            let diff = match i % 4 {
                0 => "easy",
                1 => "medium",
                2 => "hard",
                _ => "easy",
            };
            let solution = match i % 4 {
                0 => "def add(a, b): return a + b",
                1 => "def f(n):\\n    for i in range(n):\\n        if i > 0:\\n            pass",
                2 => "def f():\\n    pass\\ndef g():\\n    pass\\ndef h():\\n    pass",
                _ => "def sort_list(arr):\\n    return sorted(arr)",
            };
            writeln!(
                file,
                r#"{{"id":"test/{i}","query":"Problem {i}","solution":"{solution}","tests":[],"difficulty":"{diff}"}}"#,
            )
            .unwrap();
        }

        let dataset = CodeDataset::from_file(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        dataset
    }

    #[test]
    fn generate_pairs_from_dataset() {
        let dataset = make_dataset(10);
        let config = CodeFramePairConfig::default();
        let pairs = generate_code_frame_pairs(&dataset, &config).unwrap();
        assert_eq!(pairs.len(), 10);
    }

    #[test]
    fn pairs_have_active_slots() {
        let dataset = make_dataset(5);
        let config = CodeFramePairConfig::default();
        let pairs = generate_code_frame_pairs(&dataset, &config).unwrap();

        for pair in &pairs {
            assert!(pair.question.active_slot_count() >= 4);
            assert!(pair.answer.active_slot_count() >= 4);
        }
    }

    #[test]
    fn pairs_are_deterministic() {
        let dataset = make_dataset(5);
        let config = CodeFramePairConfig::default();
        let pairs1 = generate_code_frame_pairs(&dataset, &config).unwrap();
        let pairs2 = generate_code_frame_pairs(&dataset, &config).unwrap();

        assert_eq!(pairs1.len(), pairs2.len());
        for (a, b) in pairs1.iter().zip(pairs2.iter()) {
            assert_eq!(
                a.question.active_slot_count(),
                b.question.active_slot_count()
            );
        }
    }

    #[test]
    fn curriculum_filter_reduces_count() {
        let dataset = make_dataset(20);

        let all_config = CodeFramePairConfig::default();
        let all_pairs = generate_code_frame_pairs(&dataset, &all_config).unwrap();

        let simple_config = CodeFramePairConfig {
            stage_filter: Some(CurriculumStageFilter::SimpleFunctions),
            ..CodeFramePairConfig::default()
        };
        let simple_pairs = generate_code_frame_pairs(&dataset, &simple_config).unwrap();

        assert!(simple_pairs.len() < all_pairs.len());
    }

    #[test]
    fn max_pairs_limits_output() {
        let dataset = make_dataset(20);
        let config = CodeFramePairConfig {
            max_pairs: 5,
            ..CodeFramePairConfig::default()
        };
        let pairs = generate_code_frame_pairs(&dataset, &config).unwrap();
        assert_eq!(pairs.len(), 5);
    }

    #[test]
    fn classify_problem_simple() {
        let problem = CodeProblem {
            id: "test/1".to_string(),
            query: "Add two numbers".to_string(),
            solution: "def add(a, b): return a + b".to_string(),
            tests: vec![],
            language: None,
            difficulty: Some("easy".to_string()),
        };
        assert_eq!(
            classify_problem(&problem),
            CurriculumStageFilter::SimpleFunctions
        );
    }

    #[test]
    fn classify_problem_with_loop() {
        let problem = CodeProblem {
            id: "test/2".to_string(),
            query: "Sum a list".to_string(),
            solution: "def sum_list(lst):\n    total = 0\n    for x in lst:\n        total += x\n    return total".to_string(),
            tests: vec![],
            language: None,
            difficulty: None,
        };
        assert_eq!(
            classify_problem(&problem),
            CurriculumStageFilter::LoopsAndConditionals
        );
    }

    #[test]
    fn classify_problem_hard_explicit() {
        let problem = CodeProblem {
            id: "test/3".to_string(),
            query: "Sort array".to_string(),
            solution: "def merge_sort(arr): pass".to_string(),
            tests: vec![],
            language: None,
            difficulty: Some("hard".to_string()),
        };
        assert_eq!(
            classify_problem(&problem),
            CurriculumStageFilter::AlgorithmicReasoning
        );
    }

    #[test]
    fn encode_text_produces_normalized_frame() {
        let frame = encode_text_to_frame("def add(a, b): return a + b", 42, 0).unwrap();
        assert_eq!(frame.active_slot_count(), 4);

        for slot_idx in 0..4 {
            let slot = frame.read_slot(slot_idx).unwrap();
            let vec = slot.resolutions[0].as_ref().unwrap();
            let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-4,
                "slot {} not normalized: norm={}",
                slot_idx,
                norm
            );
        }
    }

    #[test]
    fn different_texts_produce_different_frames() {
        let f1 = encode_text_to_frame("def add(a, b): return a + b", 42, 0).unwrap();
        let f2 = encode_text_to_frame("def multiply(x, y): return x * y", 42, 0).unwrap();

        // Check that slot 0 vectors differ
        let s1 = f1.read_slot(0).unwrap().resolutions[0].unwrap();
        let s2 = f2.read_slot(0).unwrap().resolutions[0].unwrap();
        assert_ne!(s1, s2);
    }

    #[test]
    fn empty_dataset_errors() {
        use std::io::Write;
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("empty_cfp.jsonl");
        let mut file = std::fs::File::create(&path).unwrap();
        writeln!(file, r#"{{"id":"test/1","query":"Q","solution":"S","tests":[]}}"#).unwrap();

        let dataset = CodeDataset::from_file(&path).unwrap();
        let config = CodeFramePairConfig {
            stage_filter: Some(CurriculumStageFilter::AlgorithmicReasoning),
            ..CodeFramePairConfig::default()
        };

        // Simple problem filtered for algorithmic stage → no pairs
        let result = generate_code_frame_pairs(&dataset, &config);
        assert!(result.is_err());

        let _ = std::fs::remove_file(&path);
    }
}
