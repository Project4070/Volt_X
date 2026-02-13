//! Streaming JSONL reader for The Stack dataset.
//!
//! The Stack stores code files as JSONL with fields:
//! `{"content", "language", "path", "size"}`. This module provides
//! a streaming iterator that reads entries line-by-line without
//! loading the entire corpus into memory.
//!
//! # Example
//!
//! ```no_run
//! use volt_learn::stack_corpus::StackCorpusReader;
//!
//! let reader = StackCorpusReader::from_file("data/python_sample.jsonl")
//!     .expect("failed to open corpus");
//!
//! for entry in reader {
//!     let entry = entry.expect("failed to parse entry");
//!     println!("{}: {} bytes", entry.path, entry.content.len());
//! }
//! ```

use serde::Deserialize;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use volt_core::VoltError;

/// A single code file entry from The Stack JSONL format.
#[derive(Debug, Clone, Deserialize)]
pub struct StackEntry {
    /// Source code content.
    pub content: String,
    /// Programming language (e.g. "python").
    pub language: String,
    /// Repository-relative file path.
    #[serde(default)]
    pub path: String,
    /// File size in bytes.
    #[serde(default)]
    pub size: usize,
}

/// Streaming iterator over The Stack JSONL files.
///
/// Reads line-by-line from one or more JSONL files, yielding
/// [`StackEntry`] items without buffering the entire corpus.
pub struct StackCorpusReader {
    /// Current file reader.
    reader: BufReader<File>,
    /// Remaining file paths to process (for directory mode).
    remaining_files: Vec<PathBuf>,
    /// Reusable line buffer.
    line_buf: String,
    /// Current file path (for error messages).
    current_path: PathBuf,
    /// Line number within current file (for error messages).
    line_number: usize,
}

impl std::fmt::Debug for StackCorpusReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StackCorpusReader")
            .field("current_path", &self.current_path)
            .field("line_number", &self.line_number)
            .field("remaining_files", &self.remaining_files.len())
            .finish()
    }
}

impl StackCorpusReader {
    /// Open a single JSONL file for streaming.
    ///
    /// # Errors
    ///
    /// Returns `VoltError::LearnError` if the file cannot be opened.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use volt_learn::stack_corpus::StackCorpusReader;
    /// let reader = StackCorpusReader::from_file("corpus.jsonl").unwrap();
    /// ```
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, VoltError> {
        let path = path.as_ref();
        let file = File::open(path).map_err(|e| VoltError::LearnError {
            message: format!("failed to open corpus file {}: {e}", path.display()),
        })?;
        Ok(Self {
            reader: BufReader::new(file),
            remaining_files: Vec::new(),
            line_buf: String::new(),
            current_path: path.to_path_buf(),
            line_number: 0,
        })
    }

    /// Open all `*.jsonl` files in a directory for streaming.
    ///
    /// Files are processed in alphabetical order. Returns an error
    /// if the directory contains no JSONL files.
    ///
    /// # Errors
    ///
    /// Returns `VoltError::LearnError` if the directory cannot be read
    /// or contains no JSONL files.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use volt_learn::stack_corpus::StackCorpusReader;
    /// let reader = StackCorpusReader::from_directory("data/the_stack/").unwrap();
    /// ```
    pub fn from_directory(dir: impl AsRef<Path>) -> Result<Self, VoltError> {
        let dir = dir.as_ref();
        let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
            .map_err(|e| VoltError::LearnError {
                message: format!("failed to read corpus directory {}: {e}", dir.display()),
            })?
            .filter_map(|entry| {
                let entry = entry.ok()?;
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("jsonl") {
                    Some(path)
                } else {
                    None
                }
            })
            .collect();

        if paths.is_empty() {
            return Err(VoltError::LearnError {
                message: format!("no .jsonl files found in {}", dir.display()),
            });
        }

        // Sort alphabetically for deterministic ordering
        paths.sort();

        // Pop the first file to open, rest go into remaining_files
        let first_path = paths.remove(0);
        let file = File::open(&first_path).map_err(|e| VoltError::LearnError {
            message: format!("failed to open corpus file {}: {e}", first_path.display()),
        })?;

        Ok(Self {
            reader: BufReader::new(file),
            remaining_files: paths,
            line_buf: String::new(),
            current_path: first_path,
            line_number: 0,
        })
    }

    /// Try to advance to the next JSONL file in the queue.
    /// Returns `true` if a new file was opened, `false` if no files remain.
    fn advance_to_next_file(&mut self) -> Result<bool, VoltError> {
        if self.remaining_files.is_empty() {
            return Ok(false);
        }
        let next_path = self.remaining_files.remove(0);
        let file = File::open(&next_path).map_err(|e| VoltError::LearnError {
            message: format!("failed to open corpus file {}: {e}", next_path.display()),
        })?;
        self.reader = BufReader::new(file);
        self.current_path = next_path;
        self.line_number = 0;
        Ok(true)
    }
}

impl Iterator for StackCorpusReader {
    type Item = Result<StackEntry, VoltError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            self.line_buf.clear();
            match self.reader.read_line(&mut self.line_buf) {
                Ok(0) => {
                    // EOF on current file — try next file
                    match self.advance_to_next_file() {
                        Ok(true) => continue,
                        Ok(false) => return None,
                        Err(e) => return Some(Err(e)),
                    }
                }
                Ok(_) => {
                    self.line_number += 1;
                    let trimmed = self.line_buf.trim();
                    if trimmed.is_empty() {
                        continue; // skip blank lines
                    }
                    match serde_json::from_str::<StackEntry>(trimmed) {
                        Ok(entry) => return Some(Ok(entry)),
                        Err(e) => {
                            return Some(Err(VoltError::LearnError {
                                message: format!(
                                    "failed to parse JSONL at {}:{}: {e}",
                                    self.current_path.display(),
                                    self.line_number,
                                ),
                            }));
                        }
                    }
                }
                Err(e) => {
                    return Some(Err(VoltError::LearnError {
                        message: format!(
                            "I/O error reading {}:{}: {e}",
                            self.current_path.display(),
                            self.line_number,
                        ),
                    }));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn temp_jsonl(content: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f.flush().unwrap();
        f
    }

    #[test]
    fn parse_valid_entry() {
        let json = r#"{"content":"def foo(): pass","language":"python","path":"a/b.py","size":15}"#;
        let entry: StackEntry = serde_json::from_str(json).unwrap();
        assert_eq!(entry.content, "def foo(): pass");
        assert_eq!(entry.language, "python");
        assert_eq!(entry.path, "a/b.py");
        assert_eq!(entry.size, 15);
    }

    #[test]
    fn parse_missing_optional_fields() {
        let json = r#"{"content":"x = 1","language":"python"}"#;
        let entry: StackEntry = serde_json::from_str(json).unwrap();
        assert_eq!(entry.content, "x = 1");
        assert_eq!(entry.path, ""); // default
        assert_eq!(entry.size, 0); // default
    }

    #[test]
    fn iterator_yields_all_entries() {
        let data = r#"{"content":"a","language":"python","path":"a.py","size":1}
{"content":"b","language":"python","path":"b.py","size":1}
{"content":"c","language":"python","path":"c.py","size":1}
"#;
        let f = temp_jsonl(data);
        let reader = StackCorpusReader::from_file(f.path()).unwrap();
        let entries: Vec<_> = reader.collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].content, "a");
        assert_eq!(entries[2].content, "c");
    }

    #[test]
    fn iterator_skips_empty_lines() {
        let data = r#"{"content":"a","language":"python"}

{"content":"b","language":"python"}

"#;
        let f = temp_jsonl(data);
        let reader = StackCorpusReader::from_file(f.path()).unwrap();
        let entries: Vec<_> = reader.collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn iterator_errors_on_bad_json() {
        let data = r#"{"content":"a","language":"python"}
NOT VALID JSON
{"content":"b","language":"python"}
"#;
        let f = temp_jsonl(data);
        let reader = StackCorpusReader::from_file(f.path()).unwrap();
        let results: Vec<_> = reader.collect::<Vec<_>>();
        // Iterator yields all 3 items: Ok, Err, Ok
        assert_eq!(results.len(), 3);
        assert!(results[0].is_ok());
        assert!(results[1].is_err());
        assert!(results[2].is_ok()); // iterator continues past errors
    }

    #[test]
    fn from_file_nonexistent_errors() {
        let result = StackCorpusReader::from_file("/nonexistent/path.jsonl");
        assert!(result.is_err());
    }

    #[test]
    fn from_directory_no_jsonl_errors() {
        let dir = tempfile::tempdir().unwrap();
        // Empty directory has no .jsonl files
        let result = StackCorpusReader::from_directory(dir.path());
        assert!(result.is_err());
    }

    #[test]
    fn from_directory_reads_multiple_files() {
        let dir = tempfile::tempdir().unwrap();

        // Create two JSONL files
        let mut f1 = File::create(dir.path().join("01_data.jsonl")).unwrap();
        writeln!(f1, r#"{{"content":"a","language":"python"}}"#).unwrap();
        writeln!(f1, r#"{{"content":"b","language":"python"}}"#).unwrap();

        let mut f2 = File::create(dir.path().join("02_data.jsonl")).unwrap();
        writeln!(f2, r#"{{"content":"c","language":"python"}}"#).unwrap();

        let reader = StackCorpusReader::from_directory(dir.path()).unwrap();
        let entries: Vec<_> = reader.collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(entries.len(), 3);
        // Alphabetical order: 01_ first, then 02_
        assert_eq!(entries[0].content, "a");
        assert_eq!(entries[1].content, "b");
        assert_eq!(entries[2].content, "c");
    }
}
