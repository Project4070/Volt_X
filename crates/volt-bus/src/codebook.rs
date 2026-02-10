//! VQ-VAE Codebook for vector quantization.
//!
//! Provides a codebook of up to 65,536 entries (addressable by `u16`), each being a
//! 256-dimensional vector. Uses an HNSW index for fast approximate nearest-neighbor
//! queries during quantization.
//!
//! # Binary Format
//!
//! The codebook persists as a binary file with this layout:
//! ```text
//! [4 bytes: magic "VXCB"]
//! [4 bytes: version u32 LE]
//! [4 bytes: entry_count u32 LE]
//! [4 bytes: dim u32 LE]
//! [entry_count * dim * 4 bytes: f32 LE data, row-major]
//! ```
//!
//! # Example
//!
//! ```
//! use volt_bus::codebook::{Codebook, CODEBOOK_CAPACITY};
//! use volt_core::SLOT_DIM;
//!
//! // Create a small codebook from random-ish entries
//! let mut entries = Vec::new();
//! for i in 0..64u16 {
//!     let mut v = [0.0f32; SLOT_DIM];
//!     v[i as usize % SLOT_DIM] = 1.0;
//!     entries.push(v);
//! }
//!
//! let codebook = Codebook::from_entries(entries).unwrap();
//! assert_eq!(codebook.len(), 64);
//!
//! // Quantize a vector
//! let mut query = [0.0f32; SLOT_DIM];
//! query[0] = 1.0;
//! let (id, quantized) = codebook.quantize(&query).unwrap();
//!
//! // Lookup by ID returns the same vector
//! let looked_up = codebook.lookup(id).unwrap();
//! assert_eq!(&quantized, looked_up);
//! ```

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use hnsw_rs::prelude::*;
use volt_core::{VoltError, SLOT_DIM};

/// Maximum number of codebook entries (2^16, addressable by `u16`).
///
/// # Example
///
/// ```
/// use volt_bus::codebook::CODEBOOK_CAPACITY;
/// assert_eq!(CODEBOOK_CAPACITY, 65536);
/// ```
pub const CODEBOOK_CAPACITY: usize = 65536;

/// Magic bytes identifying a codebook binary file.
const MAGIC: [u8; 4] = *b"VXCB";

/// Current binary format version.
const FORMAT_VERSION: u32 = 1;

// HNSW tuning parameters.
// M=24 balances recall vs build time for 256-dim vectors at 65K scale.
const HNSW_MAX_NB_CONNECTION: usize = 24;
const HNSW_MAX_LAYER: usize = 16;
const HNSW_EF_CONSTRUCTION: usize = 200;
const HNSW_EF_SEARCH: usize = 32;

/// A VQ-VAE codebook for vector quantization.
///
/// Contains up to [`CODEBOOK_CAPACITY`] (65,536) entries of dimension [`SLOT_DIM`] (256).
/// Each entry is addressable by a `u16` codebook ID.
///
/// Internally backed by an HNSW index ([`hnsw_rs`]) for sub-millisecond
/// approximate nearest-neighbor queries during [`quantize`](Codebook::quantize).
///
/// # Construction
///
/// Use [`Codebook::from_entries`] to build from a `Vec` of vectors, or
/// [`Codebook::load`] to deserialize from a binary file produced by
/// [`Codebook::save`] or the `tools/codebook_init.py` script.
pub struct Codebook {
    /// The codebook entry vectors, L2-normalized.
    entries: Vec<[f32; SLOT_DIM]>,
    /// HNSW index over entries for fast nearest-neighbor search.
    index: Hnsw<'static, f32, DistCosine>,
}

impl std::fmt::Debug for Codebook {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Codebook")
            .field("len", &self.entries.len())
            .field("dim", &SLOT_DIM)
            .finish()
    }
}

impl Codebook {
    /// Build a codebook from a set of entry vectors.
    ///
    /// Each entry is L2-normalized during construction to maintain consistency
    /// with the HDC convention of unit-norm slot vectors.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::BusError`] if:
    /// - `entries` is empty
    /// - `entries` has more than [`CODEBOOK_CAPACITY`] elements
    /// - Any entry is a zero vector or contains NaN/Inf
    ///
    /// # Example
    ///
    /// ```
    /// use volt_bus::codebook::Codebook;
    /// use volt_core::SLOT_DIM;
    ///
    /// let mut entries = Vec::new();
    /// for i in 0..16u16 {
    ///     let mut v = [0.0f32; SLOT_DIM];
    ///     v[i as usize] = 1.0;
    ///     entries.push(v);
    /// }
    /// let cb = Codebook::from_entries(entries).unwrap();
    /// assert_eq!(cb.len(), 16);
    /// ```
    pub fn from_entries(mut entries: Vec<[f32; SLOT_DIM]>) -> Result<Self, VoltError> {
        if entries.is_empty() {
            return Err(VoltError::BusError {
                message: "codebook cannot be empty".into(),
            });
        }
        if entries.len() > CODEBOOK_CAPACITY {
            return Err(VoltError::BusError {
                message: format!(
                    "codebook has {} entries, max is {}",
                    entries.len(),
                    CODEBOOK_CAPACITY
                ),
            });
        }

        // Validate and L2-normalize each entry
        for (i, entry) in entries.iter_mut().enumerate() {
            if entry.iter().any(|x| !x.is_finite()) {
                return Err(VoltError::BusError {
                    message: format!("codebook entry {} contains NaN or Inf", i),
                });
            }
            let norm: f32 = entry.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm < 1e-10 {
                return Err(VoltError::BusError {
                    message: format!("codebook entry {} is zero or near-zero", i),
                });
            }
            for x in entry.iter_mut() {
                *x /= norm;
            }
        }

        // Build HNSW index
        let index = Hnsw::<f32, DistCosine>::new(
            HNSW_MAX_NB_CONNECTION,
            entries.len(),
            HNSW_MAX_LAYER,
            HNSW_EF_CONSTRUCTION,
            DistCosine,
        );

        for (i, entry) in entries.iter().enumerate() {
            index.insert((&entry[..], i));
        }

        Ok(Self { entries, index })
    }

    /// Look up a codebook entry by its ID.
    ///
    /// Returns a reference to the L2-normalized entry vector.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::BusError`] if `id` is out of range.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_bus::codebook::Codebook;
    /// use volt_core::SLOT_DIM;
    ///
    /// let mut v = [0.0f32; SLOT_DIM];
    /// v[0] = 1.0;
    /// let cb = Codebook::from_entries(vec![v]).unwrap();
    /// let entry = cb.lookup(0).unwrap();
    /// assert!((entry[0] - 1.0).abs() < 1e-6);
    /// ```
    pub fn lookup(&self, id: u16) -> Result<&[f32; SLOT_DIM], VoltError> {
        let idx = id as usize;
        self.entries.get(idx).ok_or_else(|| VoltError::BusError {
            message: format!(
                "codebook ID {} out of range (codebook size {})",
                id,
                self.entries.len()
            ),
        })
    }

    /// Quantize a vector by finding the nearest codebook entry.
    ///
    /// Returns `(codebook_id, quantized_vector)` where `quantized_vector`
    /// is the L2-normalized codebook entry closest in cosine distance.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::BusError`] if:
    /// - The input contains NaN or Inf
    /// - The input is a zero vector
    /// - The HNSW search returns no results (should not happen for a valid codebook)
    ///
    /// # Example
    ///
    /// ```
    /// use volt_bus::codebook::Codebook;
    /// use volt_core::SLOT_DIM;
    ///
    /// // Build a codebook with one entry
    /// let mut v = [0.0f32; SLOT_DIM];
    /// v[0] = 1.0;
    /// let cb = Codebook::from_entries(vec![v]).unwrap();
    ///
    /// // Quantize a similar vector
    /// let mut query = [0.0f32; SLOT_DIM];
    /// query[0] = 0.9;
    /// query[1] = 0.1;
    /// let (id, quantized) = cb.quantize(&query).unwrap();
    /// assert_eq!(id, 0);
    /// assert!((quantized[0] - 1.0).abs() < 1e-6);
    /// ```
    pub fn quantize(&self, vector: &[f32; SLOT_DIM]) -> Result<(u16, [f32; SLOT_DIM]), VoltError> {
        // Validate input
        if vector.iter().any(|x| !x.is_finite()) {
            return Err(VoltError::BusError {
                message: "quantize: input contains NaN or Inf".into(),
            });
        }
        let norm_sq: f32 = vector.iter().map(|x| x * x).sum();
        if norm_sq < 1e-10 {
            return Err(VoltError::BusError {
                message: "quantize: input is zero or near-zero".into(),
            });
        }

        let results = self.index.search(vector.as_slice(), 1, HNSW_EF_SEARCH);

        if results.is_empty() {
            return Err(VoltError::BusError {
                message: "quantize: HNSW search returned no results".into(),
            });
        }

        let nearest = &results[0];
        let id = nearest.d_id as u16;
        let entry = self.entries[nearest.d_id];

        Ok((id, entry))
    }

    /// Number of entries in the codebook.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_bus::codebook::Codebook;
    /// use volt_core::SLOT_DIM;
    ///
    /// let v = [1.0f32; SLOT_DIM];
    /// let cb = Codebook::from_entries(vec![v]).unwrap();
    /// assert_eq!(cb.len(), 1);
    /// ```
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the codebook has no entries.
    ///
    /// In practice this is always `false` since construction rejects empty input.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_bus::codebook::Codebook;
    /// use volt_core::SLOT_DIM;
    ///
    /// let v = [1.0f32; SLOT_DIM];
    /// let cb = Codebook::from_entries(vec![v]).unwrap();
    /// assert!(!cb.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Save the codebook to a binary file.
    ///
    /// The HNSW index is NOT serialized; it is rebuilt when loading via
    /// [`Codebook::load`]. This keeps the format simple and portable
    /// (compatible with the Python initialization script).
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::BusError`] on I/O failure.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use volt_bus::codebook::Codebook;
    /// use std::path::Path;
    ///
    /// # let cb: Codebook = todo!();
    /// cb.save(Path::new("codebook.bin")).unwrap();
    /// ```
    pub fn save(&self, path: &Path) -> Result<(), VoltError> {
        let file = File::create(path).map_err(|e| VoltError::BusError {
            message: format!("failed to create codebook file {}: {}", path.display(), e),
        })?;
        let mut writer = BufWriter::new(file);

        // Header
        writer.write_all(&MAGIC).map_err(io_err)?;
        writer
            .write_all(&FORMAT_VERSION.to_le_bytes())
            .map_err(io_err)?;
        writer
            .write_all(&(self.entries.len() as u32).to_le_bytes())
            .map_err(io_err)?;
        writer
            .write_all(&(SLOT_DIM as u32).to_le_bytes())
            .map_err(io_err)?;

        // Entry data (row-major, f32 little-endian)
        for entry in &self.entries {
            for &val in entry {
                writer.write_all(&val.to_le_bytes()).map_err(io_err)?;
            }
        }

        Ok(())
    }

    /// Load a codebook from a binary file.
    ///
    /// Reads the binary format produced by [`Codebook::save`] or the
    /// `tools/codebook_init.py` script. The HNSW index is rebuilt from
    /// the loaded entries.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::BusError`] if:
    /// - The file cannot be opened or read
    /// - The magic bytes or version don't match
    /// - The dimension doesn't match [`SLOT_DIM`]
    /// - The entry count exceeds [`CODEBOOK_CAPACITY`]
    ///
    /// # Example
    ///
    /// ```no_run
    /// use volt_bus::codebook::Codebook;
    /// use std::path::Path;
    ///
    /// let cb = Codebook::load(Path::new("codebook.bin")).unwrap();
    /// println!("Loaded codebook with {} entries", cb.len());
    /// ```
    pub fn load(path: &Path) -> Result<Self, VoltError> {
        let file = File::open(path).map_err(|e| VoltError::BusError {
            message: format!("failed to open codebook file {}: {}", path.display(), e),
        })?;
        let mut reader = BufReader::new(file);

        // Read and validate magic
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic).map_err(io_err)?;
        if magic != MAGIC {
            return Err(VoltError::BusError {
                message: format!(
                    "invalid codebook magic: expected {:?}, got {:?}",
                    MAGIC, magic
                ),
            });
        }

        // Read and validate version
        let mut buf4 = [0u8; 4];
        reader.read_exact(&mut buf4).map_err(io_err)?;
        let version = u32::from_le_bytes(buf4);
        if version != FORMAT_VERSION {
            return Err(VoltError::BusError {
                message: format!(
                    "unsupported codebook version: expected {}, got {}",
                    FORMAT_VERSION, version
                ),
            });
        }

        // Read entry count
        reader.read_exact(&mut buf4).map_err(io_err)?;
        let entry_count = u32::from_le_bytes(buf4) as usize;
        if entry_count == 0 {
            return Err(VoltError::BusError {
                message: "codebook file contains 0 entries".into(),
            });
        }
        if entry_count > CODEBOOK_CAPACITY {
            return Err(VoltError::BusError {
                message: format!(
                    "codebook file has {} entries, max is {}",
                    entry_count, CODEBOOK_CAPACITY
                ),
            });
        }

        // Read and validate dimension
        reader.read_exact(&mut buf4).map_err(io_err)?;
        let dim = u32::from_le_bytes(buf4) as usize;
        if dim != SLOT_DIM {
            return Err(VoltError::BusError {
                message: format!(
                    "codebook dimension mismatch: file has {}, expected {}",
                    dim, SLOT_DIM
                ),
            });
        }

        // Read entries
        let mut entries = Vec::with_capacity(entry_count);
        for _ in 0..entry_count {
            let mut entry = [0.0f32; SLOT_DIM];
            for val in &mut entry {
                reader.read_exact(&mut buf4).map_err(io_err)?;
                *val = f32::from_le_bytes(buf4);
            }
            entries.push(entry);
        }

        // from_entries normalizes and builds the HNSW index
        Self::from_entries(entries)
    }
}

/// Convert an I/O error to a VoltError.
fn io_err(e: std::io::Error) -> VoltError {
    VoltError::BusError {
        message: format!("codebook I/O error: {}", e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a deterministic pseudo-random normalized vector from a seed.
    fn test_vector(seed: u64) -> [f32; SLOT_DIM] {
        let mut v = [0.0f32; SLOT_DIM];
        for i in 0..SLOT_DIM {
            let mut h = seed.wrapping_mul(0xd2b74407b1ce6e93);
            h = h.wrapping_add(i as u64);
            h ^= h >> 33;
            h = h.wrapping_mul(0xff51afd7ed558ccd);
            h ^= h >> 33;
            h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
            h ^= h >> 33;
            v[i] = ((h as f64 / u64::MAX as f64) * 2.0 - 1.0) as f32;
        }
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in &mut v {
            *x /= norm;
        }
        v
    }

    /// Cosine similarity between two vectors.
    fn cosine_sim(a: &[f32; SLOT_DIM], b: &[f32; SLOT_DIM]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a < 1e-10 || norm_b < 1e-10 {
            return 0.0;
        }
        dot / (norm_a * norm_b)
    }

    fn build_test_codebook(n: usize) -> Codebook {
        let entries: Vec<_> = (0..n).map(|i| test_vector(i as u64 + 1000)).collect();
        Codebook::from_entries(entries).expect("failed to build test codebook")
    }

    #[test]
    fn from_entries_rejects_empty() {
        let result = Codebook::from_entries(vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn from_entries_rejects_over_capacity() {
        // We can't actually allocate 65537 entries in a test, but verify the check
        let entries: Vec<_> = (0..CODEBOOK_CAPACITY + 1)
            .map(|i| test_vector(i as u64))
            .collect();
        let result = Codebook::from_entries(entries);
        assert!(result.is_err());
    }

    #[test]
    fn from_entries_rejects_zero_vector() {
        let zero = [0.0f32; SLOT_DIM];
        let result = Codebook::from_entries(vec![zero]);
        assert!(result.is_err());
    }

    #[test]
    fn from_entries_rejects_nan() {
        let mut v = test_vector(42);
        v[0] = f32::NAN;
        let result = Codebook::from_entries(vec![v]);
        assert!(result.is_err());
    }

    #[test]
    fn from_entries_normalizes() {
        let mut v = [0.0f32; SLOT_DIM];
        v[0] = 3.0;
        v[1] = 4.0;
        let cb = Codebook::from_entries(vec![v]).unwrap();
        let entry = cb.lookup(0).unwrap();
        let norm: f32 = entry.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-6,
            "entry should be unit-normalized, got norm {}",
            norm
        );
    }

    #[test]
    fn lookup_valid_id() {
        let cb = build_test_codebook(64);
        let entry = cb.lookup(0);
        assert!(entry.is_ok());
        let entry = cb.lookup(63);
        assert!(entry.is_ok());
    }

    #[test]
    fn lookup_out_of_range() {
        let cb = build_test_codebook(64);
        assert!(cb.lookup(64).is_err());
        assert!(cb.lookup(u16::MAX).is_err());
    }

    #[test]
    fn quantize_finds_exact_match() {
        let cb = build_test_codebook(64);
        // Quantize an entry that's already in the codebook
        let original = cb.lookup(10).unwrap().clone();
        let (id, quantized) = cb.quantize(&original).unwrap();
        assert_eq!(id, 10, "should find the exact matching entry");
        let sim = cosine_sim(&original, &quantized);
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "exact match should have similarity 1.0, got {}",
            sim
        );
    }

    #[test]
    fn quantize_noisy_vector_high_similarity() {
        let cb = build_test_codebook(256);
        // Take an entry and add small noise
        let original = cb.lookup(42).unwrap().clone();
        let mut noisy = original;
        for i in 0..SLOT_DIM {
            noisy[i] += 0.05 * ((i as f32 * 0.1).sin());
        }
        let (id, quantized) = cb.quantize(&noisy).unwrap();
        let sim = cosine_sim(&noisy, &quantized);
        assert!(
            sim > 0.85,
            "quantized vector should have similarity > 0.85 to input, got {} (id={})",
            sim,
            id
        );
    }

    #[test]
    fn quantize_rejects_nan() {
        let cb = build_test_codebook(16);
        let mut v = test_vector(42);
        v[0] = f32::NAN;
        assert!(cb.quantize(&v).is_err());
    }

    #[test]
    fn quantize_rejects_zero() {
        let cb = build_test_codebook(16);
        let zero = [0.0f32; SLOT_DIM];
        assert!(cb.quantize(&zero).is_err());
    }

    #[test]
    fn len_and_is_empty() {
        let cb = build_test_codebook(128);
        assert_eq!(cb.len(), 128);
        assert!(!cb.is_empty());
    }

    #[test]
    fn debug_format() {
        let cb = build_test_codebook(32);
        let debug = format!("{:?}", cb);
        assert!(debug.contains("Codebook"));
        assert!(debug.contains("32"));
    }

    #[test]
    fn save_load_roundtrip() {
        let cb = build_test_codebook(128);
        let dir = std::env::temp_dir().join("volt_codebook_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_codebook.bin");

        cb.save(&path).unwrap();
        let loaded = Codebook::load(&path).unwrap();

        assert_eq!(loaded.len(), cb.len());

        // Verify all entries match
        for i in 0..cb.len() {
            let orig = cb.lookup(i as u16).unwrap();
            let load = loaded.lookup(i as u16).unwrap();
            let sim = cosine_sim(orig, load);
            assert!(
                (sim - 1.0).abs() < 1e-6,
                "entry {} mismatch after load, similarity = {}",
                i,
                sim
            );
        }

        // Clean up
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn load_rejects_bad_magic() {
        let dir = std::env::temp_dir().join("volt_codebook_test_magic");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad_magic.bin");
        std::fs::write(&path, b"BAAD").unwrap();

        let result = Codebook::load(&path);
        assert!(result.is_err());

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn load_rejects_wrong_dim() {
        let dir = std::env::temp_dir().join("volt_codebook_test_dim");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("wrong_dim.bin");

        let mut data = Vec::new();
        data.extend_from_slice(&MAGIC);
        data.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        data.extend_from_slice(&1u32.to_le_bytes()); // 1 entry
        data.extend_from_slice(&128u32.to_le_bytes()); // wrong dim
        std::fs::write(&path, &data).unwrap();

        let result = Codebook::load(&path);
        assert!(result.is_err());

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }
}
