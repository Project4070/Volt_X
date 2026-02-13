//! Heuristic role labeling for Python code tokens.
//!
//! Assigns BPE tokens to slot roles based on Python syntax patterns.
//! Used as supervision signal during Phase 1.3 role grounding training.
//!
//! # Role Mapping
//!
//! | Token Pattern | Slot | Role |
//! |---------------|------|------|
//! | `def`/`class` + name | S0 | Agent (function/class name) |
//! | method call verb | S1 | Predicate (operation) |
//! | args between `(` and `)` | S2 | Patient (arguments) |
//! | after `return` | S3 | Location (return value) |
//! | `for`/`while` | S4 | Time (execution order) |
//! | `if`/`elif`/`else` | S6 | Instrument (control flow) |
//! | `try`/`except`/`finally` | S7 | Cause (error handling) |
//! | everything else | 15 | Free(6) (no specific role) |
//!
//! # Example
//!
//! ```
//! use volt_learn::role_labels::label_code_tokens;
//!
//! let tokens = ["def", "add", "(", "a", ",", "b", ")", ":", "return", "a"];
//! let labels = label_code_tokens(&tokens);
//! assert_eq!(labels[0], 0); // "def" → Agent
//! assert_eq!(labels[1], 0); // "add" → Agent (function name)
//! assert_eq!(labels[3], 2); // "a" in args → Patient
//! assert_eq!(labels[8], 3); // "return" → Location
//! ```

use volt_core::MAX_SLOTS;

/// No specific role — maps to the last free slot (S15).
const NO_ROLE: u32 = (MAX_SLOTS - 1) as u32;

/// Assign slot role labels to a sequence of code tokens.
///
/// Uses heuristic pattern matching based on Python syntax. Each token
/// gets a label 0..15 corresponding to a slot index.
///
/// # Arguments
///
/// * `tokens` — Sequence of string tokens (can be BPE subwords or full words).
///
/// # Returns
///
/// Vec of role labels (u32), same length as `tokens`.
pub fn label_code_tokens(tokens: &[&str]) -> Vec<u32> {
    let mut labels = vec![NO_ROLE; tokens.len()];
    let mut state = LabelState::Normal;

    for (i, token) in tokens.iter().enumerate() {
        let t = token.trim().to_lowercase();

        match state {
            LabelState::Normal => {
                if t == "def" || t == "class" {
                    labels[i] = 0; // Agent: function/class keyword
                    state = LabelState::FuncName;
                } else if t == "return" {
                    labels[i] = 3; // Location: return
                    state = LabelState::ReturnExpr;
                } else if t == "for" || t == "while" {
                    labels[i] = 4; // Time: loop
                } else if t == "if" || t == "elif" || t == "else" {
                    labels[i] = 6; // Instrument: control flow
                } else if t == "try" || t == "except" || t == "finally" || t == "catch" {
                    labels[i] = 7; // Cause: error handling
                } else if t == "import" || t == "from" {
                    labels[i] = 5; // Manner: imports (algorithm pattern)
                } else if t == "(" {
                    // Might be function call args
                    state = LabelState::Args { depth: 1 };
                    labels[i] = 2; // Patient: args start
                } else if is_assignment_op(&t) {
                    labels[i] = 1; // Predicate: operation
                } else {
                    labels[i] = NO_ROLE;
                }
            }
            LabelState::FuncName => {
                labels[i] = 0; // Agent: function/class name
                if t == "(" {
                    state = LabelState::FuncArgs { depth: 1 };
                    labels[i] = 2; // Patient: args
                } else if t == ":" {
                    state = LabelState::Normal;
                }
                // Stay in FuncName until we see `(` or `:`
            }
            LabelState::FuncArgs { depth } => {
                labels[i] = 2; // Patient: argument tokens
                if t == "(" {
                    state = LabelState::FuncArgs { depth: depth + 1 };
                } else if t == ")" {
                    if depth <= 1 {
                        state = LabelState::Normal;
                    } else {
                        state = LabelState::FuncArgs { depth: depth - 1 };
                    }
                }
            }
            LabelState::Args { depth } => {
                labels[i] = 2; // Patient: argument tokens
                if t == "(" {
                    state = LabelState::Args { depth: depth + 1 };
                } else if t == ")" {
                    if depth <= 1 {
                        state = LabelState::Normal;
                    } else {
                        state = LabelState::Args { depth: depth - 1 };
                    }
                }
            }
            LabelState::ReturnExpr => {
                if t == "\n" || t == ";" || t.starts_with('#') {
                    state = LabelState::Normal;
                    labels[i] = NO_ROLE;
                } else {
                    labels[i] = 3; // Location: return expression
                }
            }
        }
    }

    labels
}

/// Internal state machine for token labeling.
#[derive(Debug, Clone, Copy)]
enum LabelState {
    /// Normal code context.
    Normal,
    /// Just saw `def`/`class`, expecting function/class name.
    FuncName,
    /// Inside function definition arguments.
    FuncArgs { depth: usize },
    /// Inside general parenthesized arguments.
    Args { depth: usize },
    /// Inside return expression.
    ReturnExpr,
}

/// Check if a token is an assignment or binary operator.
fn is_assignment_op(t: &str) -> bool {
    matches!(
        t,
        "=" | "+=" | "-=" | "*=" | "/=" | "//=" | "%=" | "**=" | "&=" | "|=" | "^=" | ">>="
            | "<<="
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn label_simple_function() {
        let tokens = ["def", "add", "(", "a", ",", "b", ")", ":", "return", "a"];
        let labels = label_code_tokens(&tokens);

        assert_eq!(labels[0], 0, "def → Agent");
        assert_eq!(labels[1], 0, "add → Agent (function name)");
        assert_eq!(labels[2], 2, "( → Patient (args start)");
        assert_eq!(labels[3], 2, "a → Patient (argument)");
        assert_eq!(labels[4], 2, ", → Patient (in args)");
        assert_eq!(labels[5], 2, "b → Patient (argument)");
        assert_eq!(labels[6], 2, ") → Patient (args end, then Normal)");
        // After ), state goes Normal
        assert_eq!(labels[8], 3, "return → Location");
        assert_eq!(labels[9], 3, "a → Location (return expr)");
    }

    #[test]
    fn label_control_flow() {
        let tokens = ["if", "x", ">", "0", ":", "for", "i", "in", "range"];
        let labels = label_code_tokens(&tokens);

        assert_eq!(labels[0], 6, "if → Instrument (control flow)");
        assert_eq!(labels[5], 4, "for → Time (execution order)");
    }

    #[test]
    fn label_error_handling() {
        let tokens = ["try", ":", "except", "Exception"];
        let labels = label_code_tokens(&tokens);

        assert_eq!(labels[0], 7, "try → Cause");
        assert_eq!(labels[2], 7, "except → Cause");
    }

    #[test]
    fn label_class_definition() {
        let tokens = ["class", "MyClass", "(", "Base", ")", ":"];
        let labels = label_code_tokens(&tokens);

        assert_eq!(labels[0], 0, "class → Agent");
        assert_eq!(labels[1], 0, "MyClass → Agent");
        assert_eq!(labels[2], 2, "( → Patient (bases)");
        assert_eq!(labels[3], 2, "Base → Patient");
    }

    #[test]
    fn empty_tokens() {
        let labels = label_code_tokens(&[]);
        assert!(labels.is_empty());
    }

    #[test]
    fn all_labels_in_range() {
        let tokens = [
            "def", "foo", "(", "x", ")", ":", "if", "x", ">", "0", ":", "return", "x", "+", "1",
        ];
        let labels = label_code_tokens(&tokens);
        for (i, &label) in labels.iter().enumerate() {
            assert!(
                (label as usize) < MAX_SLOTS,
                "label[{i}]={label} out of range"
            );
        }
    }
}
