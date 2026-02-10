//! PropBank ↔ SlotRole mapping.
//!
//! Maps PropBank semantic role labels (ARG0, ARG1, ARGM-LOC, etc.) to
//! Volt X's 16-slot TensorFrame roles. Used by the LLM translator to
//! convert PropBank-annotated training data into TensorFrame slot
//! assignments, and by the projection head classifier whose output
//! classes correspond to slot indices.
//!
//! ## Mapping Table
//!
//! | PropBank   | Slot | SlotRole      |
//! |------------|------|---------------|
//! | ARG0       | 0    | Agent         |
//! | V (verb)   | 1    | Predicate     |
//! | ARG1       | 2    | Patient       |
//! | ARGM-LOC   | 3    | Location      |
//! | ARGM-TMP   | 4    | Time          |
//! | ARGM-MNR   | 5    | Manner        |
//! | ARG2       | 6    | Instrument    |
//! | ARGM-CAU   | 7    | Cause         |
//! | ARG3/ARG4  | 8    | Result        |
//! | Other      | 9-15 | Free(0..6)    |

use volt_core::{SlotRole, VoltError};

/// Number of distinct role classes for the projection head classifier.
///
/// Matches [`volt_core::MAX_SLOTS`] — one class per slot index.
///
/// # Example
///
/// ```
/// use volt_translate::llm::roles::NUM_ROLE_CLASSES;
/// assert_eq!(NUM_ROLE_CLASSES, 16);
/// ```
pub const NUM_ROLE_CLASSES: usize = 16;

/// PropBank semantic role label.
///
/// Represents the standard PropBank argument labels used in semantic
/// role labeling. These are mapped to Volt X [`SlotRole`] values via
/// [`propbank_to_slot`].
///
/// # Example
///
/// ```
/// use volt_translate::llm::roles::PropBankRole;
/// let role = PropBankRole::Arg0;
/// assert_eq!(format!("{role:?}"), "Arg0");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PropBankRole {
    /// Verb / predicate.
    V,
    /// Proto-Agent (ARG0).
    Arg0,
    /// Proto-Patient (ARG1).
    Arg1,
    /// Indirect object, benefactive, or instrument (ARG2).
    Arg2,
    /// Start point or benefactive (ARG3).
    Arg3,
    /// End point (ARG4).
    Arg4,
    /// Location modifier (ARGM-LOC).
    ArgmLoc,
    /// Temporal modifier (ARGM-TMP).
    ArgmTmp,
    /// Manner modifier (ARGM-MNR).
    ArgmMnr,
    /// Cause modifier (ARGM-CAU).
    ArgmCau,
    /// Direction modifier (ARGM-DIR).
    ArgmDir,
    /// Adverbial modifier (ARGM-ADV).
    ArgmAdv,
    /// Purpose/not-cause (ARGM-PNC).
    ArgmPnc,
    /// Discourse marker (ARGM-DIS).
    ArgmDis,
    /// Negation (ARGM-NEG).
    ArgmNeg,
    /// Modal (ARGM-MOD).
    ArgmMod,
    /// No role (padding / outside any argument span).
    NoRole,
}

/// Maps a PropBank role to its slot index and [`SlotRole`].
///
/// Returns `(slot_index, role)` for use with
/// [`TensorFrame::write_at`](volt_core::TensorFrame::write_at).
///
/// # Example
///
/// ```
/// use volt_translate::llm::roles::{propbank_to_slot, PropBankRole};
/// use volt_core::SlotRole;
///
/// let (idx, role) = propbank_to_slot(PropBankRole::Arg0);
/// assert_eq!(idx, 0);
/// assert_eq!(role, SlotRole::Agent);
/// ```
pub fn propbank_to_slot(role: PropBankRole) -> (usize, SlotRole) {
    match role {
        PropBankRole::Arg0 => (0, SlotRole::Agent),
        PropBankRole::V => (1, SlotRole::Predicate),
        PropBankRole::Arg1 => (2, SlotRole::Patient),
        PropBankRole::ArgmLoc => (3, SlotRole::Location),
        PropBankRole::ArgmTmp => (4, SlotRole::Time),
        PropBankRole::ArgmMnr => (5, SlotRole::Manner),
        PropBankRole::Arg2 => (6, SlotRole::Instrument),
        PropBankRole::ArgmCau => (7, SlotRole::Cause),
        PropBankRole::Arg3 | PropBankRole::Arg4 => (8, SlotRole::Result),
        PropBankRole::ArgmDir => (9, SlotRole::Free(0)),
        PropBankRole::ArgmAdv => (10, SlotRole::Free(1)),
        PropBankRole::ArgmPnc => (11, SlotRole::Free(2)),
        PropBankRole::ArgmDis => (12, SlotRole::Free(3)),
        PropBankRole::ArgmNeg => (13, SlotRole::Free(4)),
        PropBankRole::ArgmMod => (14, SlotRole::Free(5)),
        PropBankRole::NoRole => (15, SlotRole::Free(6)),
    }
}

/// Maps a slot index (0–15) back to its primary [`PropBankRole`].
///
/// This is the inverse of [`propbank_to_slot`] for the primary role
/// assigned to each slot.
///
/// # Example
///
/// ```
/// use volt_translate::llm::roles::{slot_to_propbank, PropBankRole};
///
/// assert_eq!(slot_to_propbank(0), PropBankRole::Arg0);
/// assert_eq!(slot_to_propbank(1), PropBankRole::V);
/// ```
pub fn slot_to_propbank(index: usize) -> PropBankRole {
    match index {
        0 => PropBankRole::Arg0,
        1 => PropBankRole::V,
        2 => PropBankRole::Arg1,
        3 => PropBankRole::ArgmLoc,
        4 => PropBankRole::ArgmTmp,
        5 => PropBankRole::ArgmMnr,
        6 => PropBankRole::Arg2,
        7 => PropBankRole::ArgmCau,
        8 => PropBankRole::Arg3,
        9 => PropBankRole::ArgmDir,
        10 => PropBankRole::ArgmAdv,
        11 => PropBankRole::ArgmPnc,
        12 => PropBankRole::ArgmDis,
        13 => PropBankRole::ArgmNeg,
        14 => PropBankRole::ArgmMod,
        _ => PropBankRole::NoRole,
    }
}

/// Maps a slot index (0–15) to its [`SlotRole`].
///
/// Convenience function that combines [`slot_to_propbank`] and
/// [`propbank_to_slot`].
///
/// # Example
///
/// ```
/// use volt_translate::llm::roles::slot_to_role;
/// use volt_core::SlotRole;
///
/// assert_eq!(slot_to_role(0), SlotRole::Agent);
/// assert_eq!(slot_to_role(3), SlotRole::Location);
/// ```
pub fn slot_to_role(index: usize) -> SlotRole {
    propbank_to_slot(slot_to_propbank(index)).1
}

/// Parses a PropBank label string into a [`PropBankRole`].
///
/// Accepts standard PropBank notation: `"ARG0"`, `"ARG1"`, `"V"`,
/// `"ARGM-LOC"`, `"ARGM-TMP"`, etc. Case-insensitive.
///
/// # Errors
///
/// Returns [`VoltError::TranslateError`] if the label is not recognized.
///
/// # Example
///
/// ```
/// use volt_translate::llm::roles::{parse_propbank_label, PropBankRole};
///
/// assert_eq!(parse_propbank_label("ARG0").unwrap(), PropBankRole::Arg0);
/// assert_eq!(parse_propbank_label("argm-loc").unwrap(), PropBankRole::ArgmLoc);
/// assert_eq!(parse_propbank_label("V").unwrap(), PropBankRole::V);
/// assert!(parse_propbank_label("UNKNOWN").is_err());
/// ```
pub fn parse_propbank_label(label: &str) -> Result<PropBankRole, VoltError> {
    let upper = label.to_uppercase();
    match upper.as_str() {
        "V" | "REL" => Ok(PropBankRole::V),
        "ARG0" | "A0" => Ok(PropBankRole::Arg0),
        "ARG1" | "A1" => Ok(PropBankRole::Arg1),
        "ARG2" | "A2" => Ok(PropBankRole::Arg2),
        "ARG3" | "A3" => Ok(PropBankRole::Arg3),
        "ARG4" | "A4" => Ok(PropBankRole::Arg4),
        "ARGM-LOC" | "AM-LOC" => Ok(PropBankRole::ArgmLoc),
        "ARGM-TMP" | "AM-TMP" => Ok(PropBankRole::ArgmTmp),
        "ARGM-MNR" | "AM-MNR" => Ok(PropBankRole::ArgmMnr),
        "ARGM-CAU" | "AM-CAU" => Ok(PropBankRole::ArgmCau),
        "ARGM-DIR" | "AM-DIR" => Ok(PropBankRole::ArgmDir),
        "ARGM-ADV" | "AM-ADV" => Ok(PropBankRole::ArgmAdv),
        "ARGM-PNC" | "AM-PNC" => Ok(PropBankRole::ArgmPnc),
        "ARGM-DIS" | "AM-DIS" => Ok(PropBankRole::ArgmDis),
        "ARGM-NEG" | "AM-NEG" => Ok(PropBankRole::ArgmNeg),
        "ARGM-MOD" | "AM-MOD" => Ok(PropBankRole::ArgmMod),
        "O" | "_" | "NONE" => Ok(PropBankRole::NoRole),
        _ => Err(VoltError::TranslateError {
            message: format!("unrecognized PropBank label: {label}"),
        }),
    }
}

/// Returns the class index (0–15) for a PropBank role.
///
/// This is the integer label used by the projection head's role
/// classifier. It equals the slot index from [`propbank_to_slot`].
///
/// # Example
///
/// ```
/// use volt_translate::llm::roles::{role_to_class_index, PropBankRole};
///
/// assert_eq!(role_to_class_index(PropBankRole::Arg0), 0);
/// assert_eq!(role_to_class_index(PropBankRole::V), 1);
/// assert_eq!(role_to_class_index(PropBankRole::ArgmLoc), 3);
/// ```
pub fn role_to_class_index(role: PropBankRole) -> usize {
    propbank_to_slot(role).0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_core_roles_mapped() {
        assert_eq!(propbank_to_slot(PropBankRole::Arg0), (0, SlotRole::Agent));
        assert_eq!(propbank_to_slot(PropBankRole::V), (1, SlotRole::Predicate));
        assert_eq!(propbank_to_slot(PropBankRole::Arg1), (2, SlotRole::Patient));
        assert_eq!(propbank_to_slot(PropBankRole::ArgmLoc), (3, SlotRole::Location));
        assert_eq!(propbank_to_slot(PropBankRole::ArgmTmp), (4, SlotRole::Time));
        assert_eq!(propbank_to_slot(PropBankRole::ArgmMnr), (5, SlotRole::Manner));
        assert_eq!(propbank_to_slot(PropBankRole::Arg2), (6, SlotRole::Instrument));
        assert_eq!(propbank_to_slot(PropBankRole::ArgmCau), (7, SlotRole::Cause));
        assert_eq!(propbank_to_slot(PropBankRole::Arg3), (8, SlotRole::Result));
        assert_eq!(propbank_to_slot(PropBankRole::Arg4), (8, SlotRole::Result));
    }

    #[test]
    fn free_roles_mapped() {
        assert_eq!(propbank_to_slot(PropBankRole::ArgmDir).0, 9);
        assert_eq!(propbank_to_slot(PropBankRole::ArgmAdv).0, 10);
        assert_eq!(propbank_to_slot(PropBankRole::ArgmPnc).0, 11);
        assert_eq!(propbank_to_slot(PropBankRole::ArgmDis).0, 12);
        assert_eq!(propbank_to_slot(PropBankRole::ArgmNeg).0, 13);
        assert_eq!(propbank_to_slot(PropBankRole::ArgmMod).0, 14);
        assert_eq!(propbank_to_slot(PropBankRole::NoRole).0, 15);
    }

    #[test]
    fn slot_roundtrip() {
        for role in [
            PropBankRole::Arg0,
            PropBankRole::V,
            PropBankRole::Arg1,
            PropBankRole::ArgmLoc,
            PropBankRole::ArgmTmp,
            PropBankRole::ArgmMnr,
            PropBankRole::Arg2,
            PropBankRole::ArgmCau,
            PropBankRole::ArgmDir,
            PropBankRole::ArgmAdv,
            PropBankRole::ArgmPnc,
            PropBankRole::ArgmDis,
            PropBankRole::ArgmNeg,
            PropBankRole::ArgmMod,
            PropBankRole::NoRole,
        ] {
            let (idx, _slot_role) = propbank_to_slot(role);
            let back = slot_to_propbank(idx);
            assert_eq!(propbank_to_slot(back).0, idx, "round-trip failed for {role:?}");
        }
    }

    #[test]
    fn parse_standard_labels() {
        assert_eq!(parse_propbank_label("ARG0").unwrap(), PropBankRole::Arg0);
        assert_eq!(parse_propbank_label("ARG1").unwrap(), PropBankRole::Arg1);
        assert_eq!(parse_propbank_label("V").unwrap(), PropBankRole::V);
        assert_eq!(parse_propbank_label("ARGM-LOC").unwrap(), PropBankRole::ArgmLoc);
        assert_eq!(parse_propbank_label("ARGM-TMP").unwrap(), PropBankRole::ArgmTmp);
        assert_eq!(parse_propbank_label("ARGM-MNR").unwrap(), PropBankRole::ArgmMnr);
        assert_eq!(parse_propbank_label("ARGM-CAU").unwrap(), PropBankRole::ArgmCau);
    }

    #[test]
    fn parse_case_insensitive() {
        assert_eq!(parse_propbank_label("arg0").unwrap(), PropBankRole::Arg0);
        assert_eq!(parse_propbank_label("argm-loc").unwrap(), PropBankRole::ArgmLoc);
        assert_eq!(parse_propbank_label("v").unwrap(), PropBankRole::V);
    }

    #[test]
    fn parse_alternate_labels() {
        assert_eq!(parse_propbank_label("A0").unwrap(), PropBankRole::Arg0);
        assert_eq!(parse_propbank_label("A1").unwrap(), PropBankRole::Arg1);
        assert_eq!(parse_propbank_label("AM-LOC").unwrap(), PropBankRole::ArgmLoc);
        assert_eq!(parse_propbank_label("REL").unwrap(), PropBankRole::V);
        assert_eq!(parse_propbank_label("O").unwrap(), PropBankRole::NoRole);
    }

    #[test]
    fn parse_unknown_errors() {
        assert!(parse_propbank_label("UNKNOWN").is_err());
        assert!(parse_propbank_label("").is_err());
        assert!(parse_propbank_label("ARG99").is_err());
    }

    #[test]
    fn class_index_matches_slot() {
        for role in [
            PropBankRole::Arg0,
            PropBankRole::V,
            PropBankRole::Arg1,
            PropBankRole::ArgmLoc,
        ] {
            assert_eq!(role_to_class_index(role), propbank_to_slot(role).0);
        }
    }

    #[test]
    fn all_slots_covered() {
        for i in 0..NUM_ROLE_CLASSES {
            let role = slot_to_propbank(i);
            let (idx, _) = propbank_to_slot(role);
            assert_eq!(idx, i, "slot {i} not covered");
        }
    }

    #[test]
    fn slot_to_role_convenience() {
        assert_eq!(slot_to_role(0), SlotRole::Agent);
        assert_eq!(slot_to_role(1), SlotRole::Predicate);
        assert_eq!(slot_to_role(2), SlotRole::Patient);
        assert_eq!(slot_to_role(3), SlotRole::Location);
    }
}
