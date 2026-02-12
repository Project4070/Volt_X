//! Evaluation dataset for RLVF joint alignment.
//!
//! Provides 1000 deterministic (question, answer) pairs across four
//! categories: Math, Logic, Factual, and Creative. Each pair consists
//! of word sequences that the stub translator encodes into TensorFrames
//! via its hash-based `word_to_vector`.
//!
//! ## Categories
//!
//! - **Math** (250 pairs): arithmetic-style word patterns
//! - **Logic** (250 pairs): deductive reasoning patterns
//! - **Factual** (250 pairs): entity-attribute recall patterns
//! - **Creative** (250 pairs): word-association patterns

/// Category of an evaluation pair.
///
/// # Example
///
/// ```
/// use volt_learn::eval_dataset::EvalCategory;
///
/// let cat = EvalCategory::Math;
/// assert_eq!(format!("{cat:?}"), "Math");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EvalCategory {
    /// Arithmetic-style word patterns.
    Math,
    /// Deductive reasoning patterns.
    Logic,
    /// Entity-attribute recall patterns.
    Factual,
    /// Word-association patterns.
    Creative,
}

/// A single evaluation pair: question text mapped to verified answer text.
///
/// # Example
///
/// ```
/// use volt_learn::eval_dataset::{EvalPair, EvalCategory};
///
/// let pair = EvalPair {
///     question: "two plus three".to_string(),
///     answer: "five".to_string(),
///     category: EvalCategory::Math,
/// };
/// assert_eq!(pair.category, EvalCategory::Math);
/// ```
#[derive(Debug, Clone)]
pub struct EvalPair {
    /// The input question text.
    pub question: String,
    /// The verified correct answer text.
    pub answer: String,
    /// Which evaluation category this pair belongs to.
    pub category: EvalCategory,
}

/// Number words used by the math generator.
const NUMBERS: &[&str] = &[
    "zero", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
    "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
];

/// Operators for math patterns.
const MATH_OPS: &[&str] = &["plus", "minus", "times"];

/// Atomic propositions for logic patterns.
const ATOMS: &[&str] = &[
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta",
    "theta", "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron",
    "pi", "rho", "sigma", "tau", "upsilon",
];

/// Entity-attribute pairs for factual patterns.
const FACTUAL_ENTITIES: &[(&str, &str)] = &[
    ("sky", "blue"),
    ("grass", "green"),
    ("sun", "bright"),
    ("water", "clear"),
    ("fire", "hot"),
    ("ice", "cold"),
    ("night", "dark"),
    ("snow", "white"),
    ("blood", "red"),
    ("gold", "yellow"),
    ("iron", "strong"),
    ("silk", "smooth"),
    ("stone", "hard"),
    ("wind", "swift"),
    ("ocean", "deep"),
    ("mountain", "tall"),
    ("desert", "dry"),
    ("forest", "dense"),
    ("river", "flowing"),
    ("thunder", "loud"),
    ("honey", "sweet"),
    ("lemon", "sour"),
    ("pepper", "spicy"),
    ("salt", "savory"),
    ("feather", "light"),
];

/// Creative word associations.
const CREATIVE_PAIRS: &[(&str, &str)] = &[
    ("compose melody", "harmony rhythm"),
    ("paint canvas", "color texture"),
    ("write story", "narrative plot"),
    ("sculpt marble", "form shape"),
    ("dance rhythm", "movement grace"),
    ("sing harmony", "voice melody"),
    ("draw sketch", "line shadow"),
    ("weave fabric", "thread pattern"),
    ("forge metal", "hammer anvil"),
    ("carve wood", "chisel grain"),
    ("blend colors", "palette hue"),
    ("craft poem", "verse rhyme"),
    ("design space", "layout flow"),
    ("build bridge", "arch span"),
    ("cook feast", "flavor aroma"),
    ("brew potion", "herb essence"),
    ("play music", "note tempo"),
    ("grow garden", "seed bloom"),
    ("chart voyage", "compass star"),
    ("mold clay", "kiln glaze"),
    ("stitch quilt", "patch warmth"),
    ("tune string", "pitch resonance"),
    ("bake bread", "yeast crust"),
    ("arrange flowers", "petal stem"),
    ("knit sweater", "yarn loop"),
];

/// Generates the full evaluation dataset of 1000 pairs.
///
/// Produces exactly 250 pairs per category (Math, Logic, Factual,
/// Creative) using deterministic generation. The same call always
/// returns the same dataset.
///
/// # Example
///
/// ```
/// use volt_learn::eval_dataset::{generate_eval_dataset, EvalCategory};
///
/// let dataset = generate_eval_dataset();
/// assert_eq!(dataset.len(), 1000);
///
/// let math_count = dataset.iter()
///     .filter(|p| p.category == EvalCategory::Math)
///     .count();
/// assert_eq!(math_count, 250);
/// ```
pub fn generate_eval_dataset() -> Vec<EvalPair> {
    let mut pairs = Vec::with_capacity(1000);
    pairs.extend(generate_math_pairs());
    pairs.extend(generate_logic_pairs());
    pairs.extend(generate_factual_pairs());
    pairs.extend(generate_creative_pairs());
    pairs
}

/// Generates 250 math evaluation pairs.
#[allow(clippy::needless_range_loop)]
fn generate_math_pairs() -> Vec<EvalPair> {
    let mut pairs = Vec::with_capacity(250);

    // Addition: a plus b -> result (for small numbers)
    for a in 0..=10 {
        for b in 0..=10 {
            if pairs.len() >= 100 {
                break;
            }
            let sum = a + b;
            if sum < NUMBERS.len() {
                pairs.push(EvalPair {
                    question: format!("{} {} {}", NUMBERS[a], MATH_OPS[0], NUMBERS[b]),
                    answer: NUMBERS[sum].to_string(),
                    category: EvalCategory::Math,
                });
            }
        }
        if pairs.len() >= 100 {
            break;
        }
    }

    // Subtraction: a minus b -> result (only non-negative results)
    for a in 0..=20 {
        for b in 0..=a {
            if pairs.len() >= 180 {
                break;
            }
            let diff = a - b;
            if diff < NUMBERS.len() && a < NUMBERS.len() && b < NUMBERS.len() {
                pairs.push(EvalPair {
                    question: format!("{} {} {}", NUMBERS[a], MATH_OPS[1], NUMBERS[b]),
                    answer: NUMBERS[diff].to_string(),
                    category: EvalCategory::Math,
                });
            }
        }
        if pairs.len() >= 180 {
            break;
        }
    }

    // Multiplication: a times b -> result (small values)
    for a in 0..=10 {
        for b in 0..=10 {
            if pairs.len() >= 250 {
                break;
            }
            let product = a * b;
            if product < NUMBERS.len() {
                pairs.push(EvalPair {
                    question: format!("{} {} {}", NUMBERS[a], MATH_OPS[2], NUMBERS[b]),
                    answer: NUMBERS[product].to_string(),
                    category: EvalCategory::Math,
                });
            }
        }
        if pairs.len() >= 250 {
            break;
        }
    }

    // Pad if needed (unlikely but safe)
    while pairs.len() < 250 {
        let idx = pairs.len() % 100;
        pairs.push(pairs[idx].clone());
    }

    pairs.truncate(250);
    pairs
}

/// Generates 250 logic evaluation pairs.
#[allow(clippy::needless_range_loop)]
fn generate_logic_pairs() -> Vec<EvalPair> {
    let mut pairs = Vec::with_capacity(250);
    let n = ATOMS.len();

    // Modus Ponens: "if A then B A" -> "B" (100 pairs)
    for i in 0..n {
        for j in 0..n {
            if i == j || pairs.len() >= 100 {
                continue;
            }
            pairs.push(EvalPair {
                question: format!("if {} then {} {}", ATOMS[i], ATOMS[j], ATOMS[i]),
                answer: ATOMS[j].to_string(),
                category: EvalCategory::Logic,
            });
            if pairs.len() >= 100 {
                break;
            }
        }
        if pairs.len() >= 100 {
            break;
        }
    }

    // Transitivity: "if A then B if B then C" -> "C" (80 pairs)
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            for k in 0..n {
                if k == i || k == j || pairs.len() >= 180 {
                    continue;
                }
                pairs.push(EvalPair {
                    question: format!(
                        "if {} then {} if {} then {}",
                        ATOMS[i], ATOMS[j], ATOMS[j], ATOMS[k]
                    ),
                    answer: ATOMS[k].to_string(),
                    category: EvalCategory::Logic,
                });
                if pairs.len() >= 180 {
                    break;
                }
            }
            if pairs.len() >= 180 {
                break;
            }
        }
        if pairs.len() >= 180 {
            break;
        }
    }

    // Modus Tollens: "if A then B not B" -> "not A" (70 pairs)
    for i in 0..n {
        for j in 0..n {
            if i == j || pairs.len() >= 250 {
                continue;
            }
            pairs.push(EvalPair {
                question: format!("if {} then {} not {}", ATOMS[i], ATOMS[j], ATOMS[j]),
                answer: format!("not {}", ATOMS[i]),
                category: EvalCategory::Logic,
            });
            if pairs.len() >= 250 {
                break;
            }
        }
        if pairs.len() >= 250 {
            break;
        }
    }

    pairs.truncate(250);
    pairs
}

/// Generates 250 factual evaluation pairs.
fn generate_factual_pairs() -> Vec<EvalPair> {
    let mut pairs = Vec::with_capacity(250);

    // "color of sky" -> "blue", "property of iron" -> "strong", etc.
    let query_templates = &[
        "color of", "property of", "nature of", "quality of", "trait of",
        "aspect of", "feature of", "attribute of", "character of", "essence of",
    ];

    for &(entity, attribute) in FACTUAL_ENTITIES {
        for &template in query_templates {
            if pairs.len() >= 250 {
                break;
            }
            pairs.push(EvalPair {
                question: format!("{template} {entity}"),
                answer: attribute.to_string(),
                category: EvalCategory::Factual,
            });
        }
        if pairs.len() >= 250 {
            break;
        }
    }

    pairs.truncate(250);
    pairs
}

/// Generates 250 creative evaluation pairs.
fn generate_creative_pairs() -> Vec<EvalPair> {
    let mut pairs = Vec::with_capacity(250);

    // Base creative associations with variations
    let prefixes = &[
        "create", "imagine", "invent", "envision", "dream",
        "inspire", "explore", "express", "discover", "transform",
    ];

    for &(action, response) in CREATIVE_PAIRS {
        for &prefix in prefixes {
            if pairs.len() >= 250 {
                break;
            }
            pairs.push(EvalPair {
                question: format!("{prefix} {action}"),
                answer: response.to_string(),
                category: EvalCategory::Creative,
            });
        }
        if pairs.len() >= 250 {
            break;
        }
    }

    pairs.truncate(250);
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dataset_has_1000_pairs() {
        let dataset = generate_eval_dataset();
        assert_eq!(dataset.len(), 1000);
    }

    #[test]
    fn dataset_has_250_per_category() {
        let dataset = generate_eval_dataset();
        let math = dataset.iter().filter(|p| p.category == EvalCategory::Math).count();
        let logic = dataset.iter().filter(|p| p.category == EvalCategory::Logic).count();
        let factual = dataset.iter().filter(|p| p.category == EvalCategory::Factual).count();
        let creative = dataset.iter().filter(|p| p.category == EvalCategory::Creative).count();
        assert_eq!(math, 250);
        assert_eq!(logic, 250);
        assert_eq!(factual, 250);
        assert_eq!(creative, 250);
    }

    #[test]
    fn math_pairs_are_valid() {
        let dataset = generate_eval_dataset();
        let math_pairs: Vec<_> = dataset.iter()
            .filter(|p| p.category == EvalCategory::Math)
            .collect();
        // All questions should have an operator word
        for pair in &math_pairs {
            assert!(
                pair.question.contains("plus")
                    || pair.question.contains("minus")
                    || pair.question.contains("times"),
                "math question missing operator: {}",
                pair.question,
            );
        }
        // All answers should be number words
        for pair in &math_pairs {
            assert!(
                NUMBERS.contains(&pair.answer.as_str()),
                "math answer not a number word: {}",
                pair.answer,
            );
        }
    }

    #[test]
    fn logic_pairs_are_valid() {
        let dataset = generate_eval_dataset();
        let logic_pairs: Vec<_> = dataset.iter()
            .filter(|p| p.category == EvalCategory::Logic)
            .collect();
        // All logic questions should contain "if" and "then"
        for pair in &logic_pairs {
            assert!(
                pair.question.contains("if") && pair.question.contains("then"),
                "logic question missing if/then: {}",
                pair.question,
            );
        }
    }

    #[test]
    fn dataset_is_deterministic() {
        let d1 = generate_eval_dataset();
        let d2 = generate_eval_dataset();
        assert_eq!(d1.len(), d2.len());
        for (a, b) in d1.iter().zip(d2.iter()) {
            assert_eq!(a.question, b.question);
            assert_eq!(a.answer, b.answer);
            assert_eq!(a.category, b.category);
        }
    }

    #[test]
    fn eval_category_debug() {
        assert_eq!(format!("{:?}", EvalCategory::Math), "Math");
        assert_eq!(format!("{:?}", EvalCategory::Logic), "Logic");
        assert_eq!(format!("{:?}", EvalCategory::Factual), "Factual");
        assert_eq!(format!("{:?}", EvalCategory::Creative), "Creative");
    }

    #[test]
    fn eval_pair_clone() {
        let pair = EvalPair {
            question: "two plus three".to_string(),
            answer: "five".to_string(),
            category: EvalCategory::Math,
        };
        let cloned = pair.clone();
        assert_eq!(cloned.question, pair.question);
        assert_eq!(cloned.answer, pair.answer);
        assert_eq!(cloned.category, pair.category);
    }
}
