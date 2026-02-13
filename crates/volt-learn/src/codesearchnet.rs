//! CodeSearchNet dataset loader for Phase 1 translator training.
//!
//! Reads CodeSearchNet Python JSONL files containing function-docstring
//! pairs. Used for contrastive training of the code encoder.
//!
//! # Data Format
//!
//! Each JSONL line:
//! ```json
//! {"code": "def add(a, b): ...", "docstring": "Add two numbers", "func_name": "add", "language": "python", "split": "train"}
//! ```
//!
//! # Example
//!
//! ```ignore
//! use volt_learn::codesearchnet::CsnDataset;
//!
//! let dataset = CsnDataset::from_file("data.jsonl").unwrap();
//! let (train, valid) = dataset.split_train_valid(0.9);
//! for record in train.records() {
//!     println!("{}: {}", record.func_name, record.docstring);
//! }
//! ```

use std::path::Path;
use volt_core::VoltError;

/// A single CodeSearchNet record (function + docstring).
///
/// # Example
///
/// ```
/// use volt_learn::codesearchnet::CsnRecord;
///
/// let record = CsnRecord {
///     code: "def add(a, b): return a + b".to_string(),
///     docstring: "Add two numbers".to_string(),
///     func_name: "add".to_string(),
///     language: "python".to_string(),
/// };
/// assert_eq!(record.func_name, "add");
/// ```
#[derive(Debug, Clone)]
pub struct CsnRecord {
    /// The full function source code.
    pub code: String,
    /// The function's docstring / natural language description.
    pub docstring: String,
    /// The function name (e.g., "Request.fresh").
    pub func_name: String,
    /// The programming language (e.g., "python").
    pub language: String,
}

/// A dataset of CodeSearchNet records.
///
/// Loaded from JSONL files, supports train/valid splitting.
///
/// # Example
///
/// ```ignore
/// use volt_learn::codesearchnet::CsnDataset;
///
/// let dataset = CsnDataset::from_file("codesearchnet.jsonl").unwrap();
/// assert!(dataset.len() > 0);
/// ```
#[derive(Debug, Clone)]
pub struct CsnDataset {
    records: Vec<CsnRecord>,
}

impl CsnDataset {
    /// Load a CodeSearchNet JSONL file.
    ///
    /// Each line must be a JSON object with at least `code` and `docstring` fields.
    /// Records with empty code or docstring are skipped.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::StorageError`] if the file cannot be read, or
    /// [`VoltError::LearnError`] if JSON parsing fails on all lines.
    pub fn from_file(path: &Path) -> Result<Self, VoltError> {
        use std::io::{BufRead, BufReader};

        let file = std::fs::File::open(path).map_err(|e| VoltError::StorageError {
            message: format!("failed to open CodeSearchNet file {}: {e}", path.display()),
        })?;
        let reader = BufReader::new(file);
        let mut records = Vec::new();
        let mut errors = 0usize;

        for line in reader.lines() {
            let line = line.map_err(|e| VoltError::StorageError {
                message: format!("failed to read line: {e}"),
            })?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let parsed: Result<serde_json::Value, _> = serde_json::from_str(line);
            let Ok(obj) = parsed else {
                errors += 1;
                continue;
            };

            let code = obj
                .get("code")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let docstring = obj
                .get("docstring")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let func_name = obj
                .get("func_name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let language = obj
                .get("language")
                .and_then(|v| v.as_str())
                .unwrap_or("python")
                .to_string();

            // Skip records with empty code or docstring
            if code.is_empty() || docstring.is_empty() {
                continue;
            }

            records.push(CsnRecord {
                code,
                docstring,
                func_name,
                language,
            });
        }

        if records.is_empty() {
            return Err(VoltError::LearnError {
                message: format!(
                    "no valid records in CodeSearchNet file {} ({errors} parse errors)",
                    path.display()
                ),
            });
        }

        Ok(Self { records })
    }

    /// Number of records in the dataset.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether the dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Access all records.
    pub fn records(&self) -> &[CsnRecord] {
        &self.records
    }

    /// Split into train and validation sets.
    ///
    /// `train_fraction` should be between 0.0 and 1.0 (e.g., 0.9 for 90% train).
    /// The split is deterministic (first N records = train, rest = valid).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let dataset = CsnDataset::from_file("data.jsonl").unwrap();
    /// let (train, valid) = dataset.split_train_valid(0.9);
    /// assert!(train.len() > valid.len());
    /// ```
    pub fn split_train_valid(&self, train_fraction: f64) -> (Self, Self) {
        let split_idx = (self.records.len() as f64 * train_fraction.clamp(0.0, 1.0)) as usize;
        let train = Self {
            records: self.records[..split_idx].to_vec(),
        };
        let valid = Self {
            records: self.records[split_idx..].to_vec(),
        };
        (train, valid)
    }

    /// Get a batch of records by indices.
    ///
    /// Returns references to the records at the given indices.
    /// Indices out of bounds are silently skipped.
    pub fn batch(&self, indices: &[usize]) -> Vec<&CsnRecord> {
        indices
            .iter()
            .filter_map(|&i| self.records.get(i))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_records() -> CsnDataset {
        CsnDataset {
            records: vec![
                CsnRecord {
                    code: "def add(a, b): return a + b".to_string(),
                    docstring: "Add two numbers".to_string(),
                    func_name: "add".to_string(),
                    language: "python".to_string(),
                },
                CsnRecord {
                    code: "def sub(a, b): return a - b".to_string(),
                    docstring: "Subtract two numbers".to_string(),
                    func_name: "sub".to_string(),
                    language: "python".to_string(),
                },
                CsnRecord {
                    code: "def mul(a, b): return a * b".to_string(),
                    docstring: "Multiply two numbers".to_string(),
                    func_name: "mul".to_string(),
                    language: "python".to_string(),
                },
            ],
        }
    }

    #[test]
    fn dataset_len_and_is_empty() {
        let ds = make_test_records();
        assert_eq!(ds.len(), 3);
        assert!(!ds.is_empty());
    }

    #[test]
    fn split_train_valid_90_10() {
        let ds = make_test_records();
        let (train, valid) = ds.split_train_valid(0.67);
        assert_eq!(train.len(), 2);
        assert_eq!(valid.len(), 1);
    }

    #[test]
    fn split_train_valid_full_train() {
        let ds = make_test_records();
        let (train, valid) = ds.split_train_valid(1.0);
        assert_eq!(train.len(), 3);
        assert_eq!(valid.len(), 0);
    }

    #[test]
    fn batch_returns_correct_records() {
        let ds = make_test_records();
        let batch = ds.batch(&[0, 2]);
        assert_eq!(batch.len(), 2);
        assert_eq!(batch[0].func_name, "add");
        assert_eq!(batch[1].func_name, "mul");
    }

    #[test]
    fn batch_skips_out_of_bounds() {
        let ds = make_test_records();
        let batch = ds.batch(&[0, 99]);
        assert_eq!(batch.len(), 1);
    }
}
