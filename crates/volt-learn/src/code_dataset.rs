//! Code Dataset Pipeline (Phase 0.2)
//!
//! Unified dataset loader for code training. Supports streaming
//! (query, solution, tests) pairs from JSONL files and converting
//! them to TensorFrame pairs for training.
//!
//! ## Unified Format
//!
//! Each line in a `.jsonl` file contains one code problem:
//!
//! ```json
//! {
//!   "id": "HumanEval/0",
//!   "query": "Write a function to check if two words are anagrams.",
//!   "solution": "def are_anagrams(s1, s2):\n    return sorted(s1) == sorted(s2)",
//!   "tests": ["assert are_anagrams('listen', 'silent') == True", ...],
//!   "language": "python",
//!   "difficulty": "easy"
//! }
//! ```
//!
//! ## Usage
//!
//! ```no_run
//! use volt_learn::code_dataset::CodeDataset;
//!
//! let dataset = CodeDataset::from_file("path/to/humaneval.jsonl").unwrap();
//! for problem in dataset.iter().take(10) {
//!     println!("Problem {}: {}", problem.id, problem.query);
//! }
//! ```

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use serde::{Deserialize, Serialize};
use volt_core::{TensorFrame, VoltError};
use volt_translate::stub::StubTranslator;
use volt_translate::Translator;

/// A single code problem with query, solution, and tests.
///
/// This is the unified format for all code datasets (HumanEval, MBPP, APPS, etc.).
///
/// # Example
///
/// ```
/// use volt_learn::code_dataset::CodeProblem;
///
/// let problem = CodeProblem {
///     id: "test/1".to_string(),
///     query: "Write a function to add two numbers".to_string(),
///     solution: "def add(a, b): return a + b".to_string(),
///     tests: vec!["assert add(1, 2) == 3".to_string()],
///     language: Some("python".to_string()),
///     difficulty: Some("easy".to_string()),
/// };
/// assert_eq!(problem.id, "test/1");
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeProblem {
    /// Unique identifier (e.g., "HumanEval/0", "MBPP/1")
    pub id: String,
    /// Natural language description of the problem
    pub query: String,
    /// Reference solution code
    pub solution: String,
    /// Test cases (executable code snippets)
    pub tests: Vec<String>,
    /// Programming language (e.g., "python", "rust")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// Difficulty level (e.g., "easy", "medium", "hard")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub difficulty: Option<String>,
}

impl CodeProblem {
    /// Converts the query to a TensorFrame using the stub translator.
    ///
    /// The query (natural language description) is encoded as the input frame.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::TranslateError`] if encoding fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use volt_learn::code_dataset::CodeProblem;
    ///
    /// let problem = CodeProblem {
    ///     id: "test/1".to_string(),
    ///     query: "Add two numbers".to_string(),
    ///     solution: "def add(a, b): return a + b".to_string(),
    ///     tests: vec![],
    ///     language: Some("python".to_string()),
    ///     difficulty: None,
    /// };
    ///
    /// let translator = volt_translate::stub::StubTranslator::new();
    /// let query_frame = problem.to_query_frame(&translator).unwrap();
    /// ```
    pub fn to_query_frame(&self, translator: &StubTranslator) -> Result<TensorFrame, VoltError> {
        let output = translator.encode(&self.query)?;
        Ok(output.frame)
    }

    /// Converts the solution to a TensorFrame using the stub translator.
    ///
    /// The solution (reference code) is encoded as the target frame.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::TranslateError`] if encoding fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use volt_learn::code_dataset::CodeProblem;
    ///
    /// let problem = CodeProblem {
    ///     id: "test/1".to_string(),
    ///     query: "Add two numbers".to_string(),
    ///     solution: "def add(a, b): return a + b".to_string(),
    ///     tests: vec![],
    ///     language: Some("python".to_string()),
    ///     difficulty: None,
    /// };
    ///
    /// let translator = volt_translate::stub::StubTranslator::new();
    /// let solution_frame = problem.to_solution_frame(&translator).unwrap();
    /// ```
    pub fn to_solution_frame(
        &self,
        translator: &StubTranslator,
    ) -> Result<TensorFrame, VoltError> {
        let output = translator.encode(&self.solution)?;
        Ok(output.frame)
    }

    /// Converts problem to (query_frame, solution_frame) pair.
    ///
    /// This is the primary method used by training loops.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::TranslateError`] if encoding fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use volt_learn::code_dataset::CodeProblem;
    ///
    /// let problem = CodeProblem {
    ///     id: "test/1".to_string(),
    ///     query: "Add two numbers".to_string(),
    ///     solution: "def add(a, b): return a + b".to_string(),
    ///     tests: vec![],
    ///     language: Some("python".to_string()),
    ///     difficulty: None,
    /// };
    ///
    /// let translator = volt_translate::stub::StubTranslator::new();
    /// let (query, solution) = problem.to_frame_pair(&translator).unwrap();
    /// ```
    pub fn to_frame_pair(
        &self,
        translator: &StubTranslator,
    ) -> Result<(TensorFrame, TensorFrame), VoltError> {
        let query_frame = self.to_query_frame(translator)?;
        let solution_frame = self.to_solution_frame(translator)?;
        Ok((query_frame, solution_frame))
    }
}

/// Dataset of code problems loaded from a JSONL file.
///
/// Provides streaming iteration over problems without loading
/// the entire dataset into memory.
///
/// # Example
///
/// ```no_run
/// use volt_learn::code_dataset::CodeDataset;
///
/// let dataset = CodeDataset::from_file("humaneval.jsonl").unwrap();
/// println!("Loaded {} problems", dataset.len());
///
/// for problem in dataset.iter().take(5) {
///     println!("{}: {}", problem.id, problem.query);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct CodeDataset {
    /// All problems loaded from the file
    problems: Vec<CodeProblem>,
}

impl CodeDataset {
    /// Loads a code dataset from a JSONL file.
    ///
    /// Each line in the file must be a valid JSON object matching
    /// the [`CodeProblem`] format.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::LearnError`] if:
    /// - File cannot be opened
    /// - JSON parsing fails
    /// - File is empty
    ///
    /// # Example
    ///
    /// ```no_run
    /// use volt_learn::code_dataset::CodeDataset;
    ///
    /// let dataset = CodeDataset::from_file("humaneval.jsonl").unwrap();
    /// assert!(dataset.len() > 0);
    /// ```
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, VoltError> {
        let file = File::open(path.as_ref()).map_err(|e| VoltError::LearnError {
            message: format!("Failed to open dataset file: {}", e),
        })?;

        let reader = BufReader::new(file);
        let mut problems = Vec::new();

        for (line_num, line_result) in reader.lines().enumerate() {
            let line = line_result.map_err(|e| VoltError::LearnError {
                message: format!("Failed to read line {}: {}", line_num + 1, e),
            })?;

            // Skip empty lines
            if line.trim().is_empty() {
                continue;
            }

            let problem: CodeProblem =
                serde_json::from_str(&line).map_err(|e| VoltError::LearnError {
                    message: format!("Failed to parse JSON at line {}: {}", line_num + 1, e),
                })?;

            problems.push(problem);
        }

        if problems.is_empty() {
            return Err(VoltError::LearnError {
                message: "Dataset file is empty or contains no valid problems".to_string(),
            });
        }

        Ok(Self { problems })
    }

    /// Creates a dataset from an existing vector of problems.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::LearnError`] if the vector is empty.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_learn::code_dataset::{CodeDataset, CodeProblem};
    ///
    /// let problems = vec![CodeProblem {
    ///     id: "test/1".to_string(),
    ///     query: "Add two numbers".to_string(),
    ///     solution: "def add(a, b): return a + b".to_string(),
    ///     tests: vec![],
    ///     language: Some("python".to_string()),
    ///     difficulty: None,
    /// }];
    /// let dataset = CodeDataset::from_problems(problems).unwrap();
    /// assert_eq!(dataset.len(), 1);
    /// ```
    pub fn from_problems(problems: Vec<CodeProblem>) -> Result<Self, VoltError> {
        if problems.is_empty() {
            return Err(VoltError::LearnError {
                message: "Cannot create dataset from empty problem list".to_string(),
            });
        }
        Ok(Self { problems })
    }

    /// Returns the number of problems in the dataset.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use volt_learn::code_dataset::CodeDataset;
    ///
    /// let dataset = CodeDataset::from_file("humaneval.jsonl").unwrap();
    /// println!("Dataset contains {} problems", dataset.len());
    /// ```
    pub fn len(&self) -> usize {
        self.problems.len()
    }

    /// Returns `true` if the dataset contains no problems.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use volt_learn::code_dataset::CodeDataset;
    ///
    /// let dataset = CodeDataset::from_file("humaneval.jsonl").unwrap();
    /// assert!(!dataset.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.problems.is_empty()
    }

    /// Returns an iterator over problems.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use volt_learn::code_dataset::CodeDataset;
    ///
    /// let dataset = CodeDataset::from_file("humaneval.jsonl").unwrap();
    /// for problem in dataset.iter() {
    ///     println!("Problem: {}", problem.id);
    /// }
    /// ```
    pub fn iter(&self) -> impl Iterator<Item = &CodeProblem> {
        self.problems.iter()
    }

    /// Returns a reference to a specific problem by index.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use volt_learn::code_dataset::CodeDataset;
    ///
    /// let dataset = CodeDataset::from_file("humaneval.jsonl").unwrap();
    /// if let Some(problem) = dataset.get(0) {
    ///     println!("First problem: {}", problem.id);
    /// }
    /// ```
    pub fn get(&self, index: usize) -> Option<&CodeProblem> {
        self.problems.get(index)
    }

    /// Converts all problems to (query, solution) TensorFrame pairs.
    ///
    /// This is useful for batch training. For streaming training,
    /// use [`iter_frames`](#method.iter_frames) instead.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::TranslateError`] if any encoding fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use volt_learn::code_dataset::CodeDataset;
    /// use volt_translate::stub::StubTranslator;
    ///
    /// let dataset = CodeDataset::from_file("humaneval.jsonl").unwrap();
    /// let translator = StubTranslator::new();
    /// let pairs = dataset.to_frame_pairs(&translator).unwrap();
    /// println!("Created {} training pairs", pairs.len());
    /// ```
    pub fn to_frame_pairs(
        &self,
        translator: &StubTranslator,
    ) -> Result<Vec<(TensorFrame, TensorFrame)>, VoltError> {
        self.problems
            .iter()
            .map(|p| p.to_frame_pair(translator))
            .collect()
    }

    /// Returns an iterator that yields (query, solution) TensorFrame pairs.
    ///
    /// This is a streaming alternative to [`to_frame_pairs`](#method.to_frame_pairs)
    /// that doesn't load all frames into memory at once.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use volt_learn::code_dataset::CodeDataset;
    /// use volt_translate::stub::StubTranslator;
    ///
    /// let dataset = CodeDataset::from_file("humaneval.jsonl").unwrap();
    /// let translator = StubTranslator::new();
    ///
    /// for (i, result) in dataset.iter_frames(&translator).enumerate().take(10) {
    ///     match result {
    ///         Ok((query, solution)) => println!("Pair {}: encoded successfully", i),
    ///         Err(e) => eprintln!("Encoding error: {}", e),
    ///     }
    /// }
    /// ```
    pub fn iter_frames<'a>(
        &'a self,
        translator: &'a StubTranslator,
    ) -> impl Iterator<Item = Result<(TensorFrame, TensorFrame), VoltError>> + 'a {
        self.problems.iter().map(move |p| p.to_frame_pair(translator))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn code_problem_creation() {
        let problem = CodeProblem {
            id: "test/1".to_string(),
            query: "Write a function".to_string(),
            solution: "def f(): pass".to_string(),
            tests: vec!["assert f() is None".to_string()],
            language: Some("python".to_string()),
            difficulty: Some("easy".to_string()),
        };
        assert_eq!(problem.id, "test/1");
        assert_eq!(problem.tests.len(), 1);
    }

    #[test]
    fn code_problem_serialization() {
        let problem = CodeProblem {
            id: "test/1".to_string(),
            query: "Add two numbers".to_string(),
            solution: "def add(a, b): return a + b".to_string(),
            tests: vec!["assert add(1, 2) == 3".to_string()],
            language: Some("python".to_string()),
            difficulty: None,
        };

        let json = serde_json::to_string(&problem).unwrap();
        let deserialized: CodeProblem = serde_json::from_str(&json).unwrap();
        assert_eq!(problem.id, deserialized.id);
        assert_eq!(problem.query, deserialized.query);
    }

    #[test]
    #[ignore] // Requires larger stack on Windows
    fn code_problem_to_frame_pair() {
        let problem = CodeProblem {
            id: "test/1".to_string(),
            query: "Add numbers".to_string(),
            solution: "def add(a, b): return a + b".to_string(),
            tests: vec![],
            language: Some("python".to_string()),
            difficulty: None,
        };

        let translator = StubTranslator::new();
        let result = problem.to_frame_pair(&translator);
        assert!(result.is_ok());

        let (query_frame, solution_frame) = result.unwrap();
        // Verify encoding is deterministic
        let (query_frame2, solution_frame2) = problem.to_frame_pair(&translator).unwrap();
        assert_eq!(query_frame.frame_meta.frame_id, query_frame2.frame_meta.frame_id);
        assert_eq!(solution_frame.frame_meta.frame_id, solution_frame2.frame_meta.frame_id);
    }

    #[test]
    fn dataset_from_empty_file_errors() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("empty_dataset.jsonl");
        std::fs::File::create(&path).unwrap();

        let result = CodeDataset::from_file(&path);
        assert!(result.is_err());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn dataset_from_valid_file() {
        use std::io::Write;
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_dataset.jsonl");

        let mut file = std::fs::File::create(&path).unwrap();
        writeln!(
            file,
            r#"{{"id":"test/1","query":"Q1","solution":"S1","tests":["T1"],"language":"python"}}"#
        )
        .unwrap();
        writeln!(
            file,
            r#"{{"id":"test/2","query":"Q2","solution":"S2","tests":["T2","T3"]}}"#
        )
        .unwrap();

        let dataset = CodeDataset::from_file(&path).unwrap();
        assert_eq!(dataset.len(), 2);
        assert!(!dataset.is_empty());

        let first = dataset.get(0).unwrap();
        assert_eq!(first.id, "test/1");
        assert_eq!(first.tests.len(), 1);

        let second = dataset.get(1).unwrap();
        assert_eq!(second.id, "test/2");
        assert_eq!(second.tests.len(), 2);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn dataset_iteration() {
        use std::io::Write;
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("iter_dataset.jsonl");

        let mut file = std::fs::File::create(&path).unwrap();
        for i in 0..10 {
            writeln!(
                file,
                r#"{{"id":"test/{}","query":"Q{}","solution":"S{}","tests":[]}}"#,
                i, i, i
            )
            .unwrap();
        }

        let dataset = CodeDataset::from_file(&path).unwrap();
        assert_eq!(dataset.len(), 10);

        let ids: Vec<String> = dataset.iter().map(|p| p.id.clone()).collect();
        assert_eq!(ids.len(), 10);
        assert_eq!(ids[0], "test/0");
        assert_eq!(ids[9], "test/9");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    #[ignore] // Requires larger stack on Windows
    fn dataset_to_frame_pairs_deterministic() {
        use std::io::Write;
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("deterministic_dataset.jsonl");

        let mut file = std::fs::File::create(&path).unwrap();
        for i in 0..5 {
            writeln!(
                file,
                r#"{{"id":"test/{}","query":"Query {}","solution":"Solution {}","tests":[]}}"#,
                i, i, i
            )
            .unwrap();
        }

        let dataset = CodeDataset::from_file(&path).unwrap();
        let translator = StubTranslator::new();

        // Convert twice and verify deterministic
        let pairs1 = dataset.to_frame_pairs(&translator).unwrap();
        let pairs2 = dataset.to_frame_pairs(&translator).unwrap();

        assert_eq!(pairs1.len(), pairs2.len());
        for (i, ((q1, s1), (q2, s2))) in pairs1.iter().zip(pairs2.iter()).enumerate() {
            assert_eq!(
                q1.frame_meta.frame_id, q2.frame_meta.frame_id,
                "Query frame IDs differ at index {i}"
            );
            assert_eq!(
                s1.frame_meta.frame_id, s2.frame_meta.frame_id,
                "Solution frame IDs differ at index {i}"
            );
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    #[ignore] // Requires larger stack on Windows
    fn dataset_iter_frames() {
        use std::io::Write;
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("stream_dataset.jsonl");

        let mut file = std::fs::File::create(&path).unwrap();
        for i in 0..20 {
            writeln!(
                file,
                r#"{{"id":"test/{}","query":"Q{}","solution":"S{}","tests":[]}}"#,
                i, i, i
            )
            .unwrap();
        }

        let dataset = CodeDataset::from_file(&path).unwrap();
        let translator = StubTranslator::new();

        let mut count = 0;
        for result in dataset.iter_frames(&translator).take(10) {
            assert!(result.is_ok());
            count += 1;
        }
        assert_eq!(count, 10);

        let _ = std::fs::remove_file(&path);
    }
}
