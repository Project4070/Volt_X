//! Code problem → FramePair conversion for VFN training.
//!
//! Encodes [`CodeProblem`]s through a [`Translator`] to produce
//! [`FramePair`]s suitable for flow matching training.
//!
//! ## Usage
//!
//! ```ignore
//! use volt_learn::code_pairs::{problems_to_frame_pairs, load_datasets};
//! use volt_translate::learned::LearnedTranslator;
//!
//! let dataset = load_datasets(&[
//!     "D:\\VoltData\\humaneval\\humaneval_problems.jsonl".into(),
//!     "D:\\VoltData\\mbpp\\mbpp_problems.jsonl".into(),
//! ]).unwrap();
//!
//! let translator = LearnedTranslator::load(
//!     "checkpoints/code_tokenizer.json".as_ref(),
//!     "checkpoints/code_encoder.safetensors".as_ref(),
//!     "checkpoints/code_decoder.safetensors".as_ref(),
//! ).unwrap();
//!
//! let pairs = problems_to_frame_pairs(dataset.iter().collect::<Vec<_>>().as_slice(), &translator);
//! println!("Generated {} training pairs", pairs.len());
//! ```

use std::path::PathBuf;

use volt_core::VoltError;
use volt_soft::training::FramePair;
use volt_translate::Translator;

use crate::code_dataset::{CodeDataset, CodeProblem};

/// Encodes code problems to flow matching training pairs.
///
/// For each problem, encodes the query (natural language description) and
/// solution (reference code) through the translator. Problems that fail
/// encoding are silently skipped with a summary warning at the end.
///
/// # Arguments
///
/// * `problems` — slice of code problems to encode
/// * `translator` — any [`Translator`] implementation (StubTranslator, LearnedTranslator, etc.)
///
/// # Returns
///
/// A `Vec<FramePair>` with one entry per successfully encoded problem.
/// Empty if all problems fail.
///
/// # Example
///
/// ```ignore
/// use volt_learn::code_pairs::problems_to_frame_pairs;
/// use volt_translate::stub::StubTranslator;
///
/// let problems = vec![/* ... */];
/// let translator = StubTranslator::new();
/// let pairs = problems_to_frame_pairs(&problems, &translator);
/// ```
pub fn problems_to_frame_pairs(
    problems: &[CodeProblem],
    translator: &dyn Translator,
) -> Vec<FramePair> {
    let mut pairs = Vec::with_capacity(problems.len());
    let mut skipped = 0usize;

    for problem in problems {
        let q_result = translator.encode(&problem.query);
        let a_result = translator.encode(&problem.solution);

        match (q_result, a_result) {
            (Ok(q_out), Ok(a_out)) => {
                pairs.push(FramePair {
                    question: q_out.frame,
                    answer: a_out.frame,
                });
            }
            _ => {
                skipped += 1;
            }
        }
    }

    if skipped > 0 {
        eprintln!(
            "[code_pairs] Skipped {skipped}/{} problems (encoding failed)",
            problems.len()
        );
    }

    pairs
}

/// Loads and merges multiple JSONL dataset files into one [`CodeDataset`].
///
/// Reads each file as a `CodeDataset` and concatenates all problems.
/// Files that fail to load are skipped with a warning.
///
/// # Errors
///
/// Returns [`VoltError::LearnError`] if no files could be loaded.
///
/// # Example
///
/// ```ignore
/// use volt_learn::code_pairs::load_datasets;
///
/// let dataset = load_datasets(&[
///     "humaneval.jsonl".into(),
///     "mbpp.jsonl".into(),
/// ]).unwrap();
/// println!("Total problems: {}", dataset.len());
/// ```
pub fn load_datasets(paths: &[PathBuf]) -> Result<CodeDataset, VoltError> {
    let mut all_problems = Vec::new();

    for path in paths {
        match CodeDataset::from_file(path) {
            Ok(ds) => {
                eprintln!(
                    "[code_pairs] Loaded {} problems from {}",
                    ds.len(),
                    path.display()
                );
                for problem in ds.iter() {
                    all_problems.push(problem.clone());
                }
            }
            Err(e) => {
                eprintln!(
                    "[code_pairs] WARNING: failed to load {}: {e}",
                    path.display()
                );
            }
        }
    }

    if all_problems.is_empty() {
        return Err(VoltError::LearnError {
            message: format!(
                "failed to load any problems from {} dataset files",
                paths.len()
            ),
        });
    }

    eprintln!("[code_pairs] Total: {} problems from {} files", all_problems.len(), paths.len());

    CodeDataset::from_problems(all_problems)
}

#[cfg(test)]
mod tests {
    use super::*;
    use volt_translate::stub::StubTranslator;

    fn make_problem(id: &str, query: &str, solution: &str) -> CodeProblem {
        CodeProblem {
            id: id.to_string(),
            query: query.to_string(),
            solution: solution.to_string(),
            tests: vec![],
            language: Some("python".to_string()),
            difficulty: None,
        }
    }

    #[test]
    fn problems_to_pairs_with_stub() {
        let problems = vec![
            make_problem("t/1", "Add two numbers", "def add(a, b): return a + b"),
            make_problem("t/2", "Reverse a string", "def rev(s): return s[::-1]"),
        ];

        let translator = StubTranslator::new();
        let pairs = problems_to_frame_pairs(&problems, &translator);

        assert_eq!(pairs.len(), 2);
        for pair in &pairs {
            assert!(pair.question.active_slot_count() > 0);
            assert!(pair.answer.active_slot_count() > 0);
        }
    }

    #[test]
    fn empty_problems_returns_empty() {
        let translator = StubTranslator::new();
        let pairs = problems_to_frame_pairs(&[], &translator);
        assert!(pairs.is_empty());
    }

    #[test]
    fn load_datasets_nonexistent_errors() {
        let result = load_datasets(&[PathBuf::from("nonexistent.jsonl")]);
        assert!(result.is_err());
    }

    #[test]
    fn load_datasets_from_temp_files() {
        use std::io::Write;
        let temp_dir = std::env::temp_dir();

        let path1 = temp_dir.join("code_pairs_test_1.jsonl");
        let path2 = temp_dir.join("code_pairs_test_2.jsonl");

        let mut f1 = std::fs::File::create(&path1).unwrap();
        writeln!(f1, r#"{{"id":"a/1","query":"Q1","solution":"S1","tests":[]}}"#).unwrap();
        writeln!(f1, r#"{{"id":"a/2","query":"Q2","solution":"S2","tests":[]}}"#).unwrap();

        let mut f2 = std::fs::File::create(&path2).unwrap();
        writeln!(f2, r#"{{"id":"b/1","query":"Q3","solution":"S3","tests":[]}}"#).unwrap();

        let dataset = load_datasets(&[path1.clone(), path2.clone()]).unwrap();
        assert_eq!(dataset.len(), 3);

        let _ = std::fs::remove_file(&path1);
        let _ = std::fs::remove_file(&path2);
    }
}
