//! T2 Archive — LSM-Tree based compressed frame storage on disk.
//!
//! T2 provides persistent storage for frames that have aged out of T1.
//! Frames are stored in a Log-Structured Merge Tree:
//!
//! 1. **Memtable**: In-memory BTreeMap, absorbs writes
//! 2. **Sorted Runs**: Immutable binary files on disk, memory-mapped via `memmap2`
//! 3. **Compaction**: Merges runs at the same level into the next level
//! 4. **Bloom Filters**: Per-run filters for fast negative lookups
//!
//! ## Sorted Run File Format
//!
//! ```text
//! [magic: 4B "VXSR"][version: u32][entry_count: u32][bloom_bytes_len: u32]
//! [bloom_data: N bytes]
//! [index: entry_count × (frame_id: u64, offset: u32, length: u32, decay_level: u8)]
//! [frame_data: concatenated serialized entries]
//! ```

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::Write as IoWrite;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use volt_core::VoltError;

use crate::bloom::BloomFilter;
use crate::compressed::{DecayLevel, FrameEntry};

/// Magic bytes identifying a sorted run file.
const SORTED_RUN_MAGIC: [u8; 4] = *b"VXSR";

/// Current file format version.
const SORTED_RUN_VERSION: u32 = 1;

/// Size of the fixed header: magic(4) + version(4) + entry_count(4) + bloom_len(4).
const HEADER_SIZE: usize = 16;

/// Size of one index entry: frame_id(8) + offset(4) + length(4) + decay_level(1).
const INDEX_ENTRY_SIZE: usize = 17;

/// Configuration for T2 storage.
///
/// # Example
///
/// ```
/// use volt_db::tier2::T2Config;
/// use std::path::PathBuf;
///
/// let config = T2Config {
///     data_dir: PathBuf::from("/tmp/voltdb_t2"),
///     memtable_flush_threshold: 4 * 1024 * 1024,
///     max_runs_per_level: 4,
///     max_levels: 4,
/// };
/// ```
#[derive(Debug, Clone)]
pub struct T2Config {
    /// Directory for sorted run files.
    pub data_dir: PathBuf,
    /// Flush memtable when it exceeds this size in bytes (default 4MB).
    pub memtable_flush_threshold: usize,
    /// Maximum sorted runs per level before triggering compaction (default 4).
    pub max_runs_per_level: usize,
    /// Maximum levels in the LSM tree (default 4).
    pub max_levels: usize,
}

impl Default for T2Config {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("voltdb_t2"),
            memtable_flush_threshold: 4 * 1024 * 1024,
            max_runs_per_level: 4,
            max_levels: 4,
        }
    }
}

/// In-memory index entry pointing to frame data within a sorted run.
#[derive(Debug, Clone)]
struct IndexEntry {
    frame_id: u64,
    offset: u32,
    length: u32,
    decay_level: DecayLevel,
}

/// An immutable sorted run on disk, memory-mapped for fast reads.
struct SortedRun {
    level: usize,
    run_id: u64,
    bloom: BloomFilter,
    /// Memory-mapped file.
    #[allow(dead_code)]
    mmap: Mmap,
    /// In-memory index: sorted by frame_id for binary search.
    index: Vec<IndexEntry>,
    /// Start offset of frame data within the mmap.
    data_offset: usize,
    entry_count: usize,
    path: PathBuf,
}

impl std::fmt::Debug for SortedRun {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SortedRun")
            .field("level", &self.level)
            .field("run_id", &self.run_id)
            .field("entry_count", &self.entry_count)
            .field("path", &self.path)
            .finish()
    }
}

impl SortedRun {
    /// Creates a new sorted run from a set of frame entries.
    ///
    /// Writes the file, builds the bloom filter and index, then mmaps it.
    fn create(
        path: &Path,
        entries: &[(u64, Vec<u8>)], // (frame_id, serialized_frame_entry)
        level: usize,
        run_id: u64,
    ) -> Result<Self, VoltError> {
        let entry_count = entries.len();

        // Build bloom filter
        let mut bloom = BloomFilter::new(entry_count.max(1), 0.01);
        for &(frame_id, _) in entries {
            bloom.insert(frame_id);
        }
        let bloom_bytes = bloom.to_bytes();

        // Calculate offsets for frame data
        let index_start = HEADER_SIZE + bloom_bytes.len();
        let data_start = index_start + entry_count * INDEX_ENTRY_SIZE;

        let mut file = File::create(path).map_err(|e| VoltError::StorageError {
            message: format!("failed to create sorted run {}: {e}", path.display()),
        })?;

        // Write header
        file.write_all(&SORTED_RUN_MAGIC)
            .map_err(|e| VoltError::StorageError {
                message: format!("failed to write sorted run header: {e}"),
            })?;
        file.write_all(&SORTED_RUN_VERSION.to_le_bytes())
            .map_err(|e| VoltError::StorageError {
                message: format!("failed to write sorted run version: {e}"),
            })?;
        file.write_all(&(entry_count as u32).to_le_bytes())
            .map_err(|e| VoltError::StorageError {
                message: format!("failed to write sorted run entry count: {e}"),
            })?;
        file.write_all(&(bloom_bytes.len() as u32).to_le_bytes())
            .map_err(|e| VoltError::StorageError {
                message: format!("failed to write sorted run bloom size: {e}"),
            })?;

        // Write bloom filter
        file.write_all(&bloom_bytes)
            .map_err(|e| VoltError::StorageError {
                message: format!("failed to write bloom filter: {e}"),
            })?;

        // Build index entries and write frame data
        let mut index = Vec::with_capacity(entry_count);
        let mut data_buf = Vec::new();

        for &(frame_id, ref frame_bytes) in entries {
            let offset = data_buf.len() as u32;
            let length = frame_bytes.len() as u32;
            let decay_level = if frame_bytes.is_empty() {
                DecayLevel::Tombstoned
            } else {
                DecayLevel::from_tag(frame_bytes[0]).unwrap_or(DecayLevel::Tombstoned)
            };

            index.push(IndexEntry {
                frame_id,
                offset,
                length,
                decay_level,
            });
            data_buf.extend_from_slice(frame_bytes);
        }

        // Write index
        for idx in &index {
            file.write_all(&idx.frame_id.to_le_bytes())
                .map_err(|e| VoltError::StorageError {
                    message: format!("failed to write index entry: {e}"),
                })?;
            file.write_all(&idx.offset.to_le_bytes())
                .map_err(|e| VoltError::StorageError {
                    message: format!("failed to write index entry offset: {e}"),
                })?;
            file.write_all(&idx.length.to_le_bytes())
                .map_err(|e| VoltError::StorageError {
                    message: format!("failed to write index entry length: {e}"),
                })?;
            file.write_all(&[idx.decay_level.tag()])
                .map_err(|e| VoltError::StorageError {
                    message: format!("failed to write index entry level: {e}"),
                })?;
        }

        // Write frame data
        file.write_all(&data_buf)
            .map_err(|e| VoltError::StorageError {
                message: format!("failed to write frame data: {e}"),
            })?;

        file.sync_all().map_err(|e| VoltError::StorageError {
            message: format!("failed to sync sorted run: {e}"),
        })?;
        drop(file);

        // Memory-map the file
        let file = File::open(path).map_err(|e| VoltError::StorageError {
            message: format!("failed to reopen sorted run for mmap: {e}"),
        })?;
        // SAFETY: the file was just created and fsynced; we have exclusive access.
        let mmap = unsafe {
            Mmap::map(&file).map_err(|e| VoltError::StorageError {
                message: format!("failed to mmap sorted run: {e}"),
            })?
        };

        Ok(Self {
            level,
            run_id,
            bloom,
            mmap,
            index,
            data_offset: data_start,
            entry_count,
            path: path.to_path_buf(),
        })
    }

    /// Opens an existing sorted run file.
    fn open(path: &Path, level: usize, run_id: u64) -> Result<Self, VoltError> {
        let file = File::open(path).map_err(|e| VoltError::StorageError {
            message: format!("failed to open sorted run {}: {e}", path.display()),
        })?;

        // SAFETY: the file is immutable (sorted runs are never modified in place).
        let mmap = unsafe {
            Mmap::map(&file).map_err(|e| VoltError::StorageError {
                message: format!("failed to mmap sorted run {}: {e}", path.display()),
            })?
        };

        if mmap.len() < HEADER_SIZE {
            return Err(VoltError::StorageError {
                message: format!(
                    "sorted run {} too small: {} bytes",
                    path.display(),
                    mmap.len()
                ),
            });
        }

        // Parse header
        if &mmap[0..4] != SORTED_RUN_MAGIC.as_slice() {
            return Err(VoltError::StorageError {
                message: format!("sorted run {} has invalid magic bytes", path.display()),
            });
        }
        let version = u32::from_le_bytes(mmap[4..8].try_into().unwrap());
        if version != SORTED_RUN_VERSION {
            return Err(VoltError::StorageError {
                message: format!(
                    "sorted run {} has unsupported version {version}",
                    path.display()
                ),
            });
        }
        let entry_count = u32::from_le_bytes(mmap[8..12].try_into().unwrap()) as usize;
        let bloom_len = u32::from_le_bytes(mmap[12..16].try_into().unwrap()) as usize;

        // Parse bloom filter
        let bloom_start = HEADER_SIZE;
        let bloom_end = bloom_start + bloom_len;
        if mmap.len() < bloom_end {
            return Err(VoltError::StorageError {
                message: "sorted run bloom data truncated".to_string(),
            });
        }
        let bloom = BloomFilter::from_bytes(&mmap[bloom_start..bloom_end])?;

        // Parse index
        let index_start = bloom_end;
        let index_end = index_start + entry_count * INDEX_ENTRY_SIZE;
        if mmap.len() < index_end {
            return Err(VoltError::StorageError {
                message: "sorted run index data truncated".to_string(),
            });
        }

        let mut index = Vec::with_capacity(entry_count);
        for i in 0..entry_count {
            let base = index_start + i * INDEX_ENTRY_SIZE;
            let frame_id = u64::from_le_bytes(mmap[base..base + 8].try_into().unwrap());
            let offset = u32::from_le_bytes(mmap[base + 8..base + 12].try_into().unwrap());
            let length = u32::from_le_bytes(mmap[base + 12..base + 16].try_into().unwrap());
            let decay_level = DecayLevel::from_tag(mmap[base + 16]).ok_or_else(|| {
                VoltError::StorageError {
                    message: format!("invalid decay level tag in index entry {i}"),
                }
            })?;
            index.push(IndexEntry {
                frame_id,
                offset,
                length,
                decay_level,
            });
        }

        Ok(Self {
            level,
            run_id,
            bloom,
            mmap,
            index,
            data_offset: index_end,
            entry_count,
            path: path.to_path_buf(),
        })
    }

    /// Looks up a frame by ID using bloom filter + binary search.
    fn get(&self, frame_id: u64) -> Option<FrameEntry> {
        // Bloom filter fast rejection
        if !self.bloom.may_contain(frame_id) {
            return None;
        }

        // Binary search the index
        let pos = self
            .index
            .binary_search_by_key(&frame_id, |e| e.frame_id)
            .ok()?;

        let idx = &self.index[pos];
        let start = self.data_offset + idx.offset as usize;
        let end = start + idx.length as usize;

        if end > self.mmap.len() {
            return None;
        }

        FrameEntry::from_bytes(&self.mmap[start..end]).ok()
    }

    /// Returns all entries in this run for a given strand.
    fn scan_strand(&self, strand_id: u64) -> Vec<FrameEntry> {
        let mut entries = Vec::new();
        for idx in &self.index {
            let start = self.data_offset + idx.offset as usize;
            let end = start + idx.length as usize;
            if end > self.mmap.len() {
                continue;
            }
            if let Ok(entry) = FrameEntry::from_bytes(&self.mmap[start..end])
                && entry.strand_id() == strand_id
            {
                entries.push(entry);
            }
        }
        entries
    }

    /// Returns all entries in this run.
    fn scan_all(&self) -> Vec<(u64, FrameEntry)> {
        let mut entries = Vec::new();
        for idx in &self.index {
            let start = self.data_offset + idx.offset as usize;
            let end = start + idx.length as usize;
            if end > self.mmap.len() {
                continue;
            }
            if let Ok(entry) = FrameEntry::from_bytes(&self.mmap[start..end]) {
                entries.push((idx.frame_id, entry));
            }
        }
        entries
    }

    /// Returns the run's file path (needed for deletion during compaction).
    fn path(&self) -> &Path {
        &self.path
    }
}

/// T2 archive storage engine using LSM-Tree.
///
/// # Example
///
/// ```no_run
/// use volt_db::tier2::{Tier2Store, T2Config};
/// use volt_db::compressed::{FrameEntry, CompressedFrame, compress};
/// use volt_core::{TensorFrame, SlotData, SlotRole, SLOT_DIM};
/// use std::path::PathBuf;
///
/// let config = T2Config {
///     data_dir: PathBuf::from("/tmp/voltdb_t2_example"),
///     ..T2Config::default()
/// };
/// let mut store = Tier2Store::open(config).unwrap();
///
/// let mut frame = TensorFrame::new();
/// let mut slot = SlotData::new(SlotRole::Agent);
/// slot.write_resolution(0, [0.1; SLOT_DIM]);
/// frame.write_slot(0, slot).unwrap();
/// frame.frame_meta.frame_id = 1;
///
/// let compressed = compress(&frame);
/// store.insert(FrameEntry::Compressed(compressed)).unwrap();
/// ```
#[derive(Debug)]
pub struct Tier2Store {
    config: T2Config,
    /// In-memory buffer for incoming writes.
    memtable: BTreeMap<u64, Vec<u8>>, // frame_id -> serialized FrameEntry
    /// Approximate size of memtable in bytes.
    memtable_size: usize,
    /// Sorted runs organized by level.
    sorted_runs: Vec<Vec<SortedRun>>,
    /// Next run ID for file naming.
    next_run_id: u64,
}

impl Tier2Store {
    /// Opens (or creates) a T2 store in the configured directory.
    ///
    /// Scans for existing sorted run files and loads their indices.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::StorageError`] if directory creation or
    /// file loading fails.
    pub fn open(config: T2Config) -> Result<Self, VoltError> {
        fs::create_dir_all(&config.data_dir).map_err(|e| VoltError::StorageError {
            message: format!(
                "failed to create T2 data directory {}: {e}",
                config.data_dir.display()
            ),
        })?;

        let mut sorted_runs: Vec<Vec<SortedRun>> = (0..config.max_levels)
            .map(|_| Vec::new())
            .collect();
        let mut max_run_id = 0u64;

        // Discover existing run files: run_{id}_L{level}.vxr
        let entries = fs::read_dir(&config.data_dir).map_err(|e| VoltError::StorageError {
            message: format!(
                "failed to read T2 directory {}: {e}",
                config.data_dir.display()
            ),
        })?;

        for entry in entries {
            let entry = entry.map_err(|e| VoltError::StorageError {
                message: format!("failed to read T2 directory entry: {e}"),
            })?;
            let file_name = entry.file_name();
            let name = file_name.to_string_lossy();

            if let Some((run_id, level)) = parse_run_filename(&name)
                && level < config.max_levels
            {
                let run = SortedRun::open(&entry.path(), level, run_id)?;
                sorted_runs[level].push(run);
                max_run_id = max_run_id.max(run_id);
            }
        }

        // Sort runs within each level by run_id (newest first for query priority)
        for level_runs in &mut sorted_runs {
            level_runs.sort_by(|a, b| b.run_id.cmp(&a.run_id));
        }

        Ok(Self {
            config,
            memtable: BTreeMap::new(),
            memtable_size: 0,
            sorted_runs,
            next_run_id: max_run_id + 1,
        })
    }

    /// Inserts a frame entry into the memtable.
    ///
    /// If the memtable exceeds the flush threshold, it is automatically
    /// flushed to a sorted run on disk.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::StorageError`] if serialization or flush fails.
    pub fn insert(&mut self, entry: FrameEntry) -> Result<(), VoltError> {
        let frame_id = entry.frame_id();
        let bytes = entry.to_bytes()?;
        self.memtable_size += bytes.len();
        self.memtable.insert(frame_id, bytes);

        if self.memtable_size >= self.config.memtable_flush_threshold {
            self.flush_memtable()?;
        }

        Ok(())
    }

    /// Looks up a frame by ID across memtable and sorted runs.
    ///
    /// Search order: memtable (newest) -> level 0 runs -> level 1 -> ... -> level N.
    /// Bloom filters on runs allow fast skipping.
    pub fn get(&self, frame_id: u64) -> Option<FrameEntry> {
        // Check memtable first
        if let Some(bytes) = self.memtable.get(&frame_id) {
            return FrameEntry::from_bytes(bytes).ok();
        }

        // Check sorted runs from newest level (0) to oldest
        for level_runs in &self.sorted_runs {
            for run in level_runs {
                if let Some(entry) = run.get(frame_id) {
                    return Some(entry);
                }
            }
        }

        None
    }

    /// Returns all entries for a given strand across memtable and runs.
    pub fn scan_strand(&self, strand_id: u64) -> Vec<FrameEntry> {
        let mut entries = Vec::new();

        // Memtable entries
        for bytes in self.memtable.values() {
            if let Ok(entry) = FrameEntry::from_bytes(bytes)
                && entry.strand_id() == strand_id
            {
                entries.push(entry);
            }
        }

        // Sorted runs
        for level_runs in &self.sorted_runs {
            for run in level_runs {
                entries.extend(run.scan_strand(strand_id));
            }
        }

        entries
    }

    /// Returns all entries across memtable and runs.
    pub fn scan_all(&self) -> Vec<FrameEntry> {
        let mut entries = Vec::new();

        for bytes in self.memtable.values() {
            if let Ok(entry) = FrameEntry::from_bytes(bytes) {
                entries.push(entry);
            }
        }

        for level_runs in &self.sorted_runs {
            for run in level_runs {
                for (_id, entry) in run.scan_all() {
                    entries.push(entry);
                }
            }
        }

        entries
    }

    /// Flushes the memtable to a new level-0 sorted run on disk.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::StorageError`] if file creation fails.
    pub fn flush_memtable(&mut self) -> Result<(), VoltError> {
        if self.memtable.is_empty() {
            return Ok(());
        }

        let run_id = self.next_run_id;
        self.next_run_id += 1;

        let path = self
            .config
            .data_dir
            .join(format!("run_{run_id:04}_L0.vxr"));

        // Collect entries sorted by frame_id (BTreeMap is already sorted)
        let entries: Vec<(u64, Vec<u8>)> = self
            .memtable
            .iter()
            .map(|(&id, bytes)| (id, bytes.clone()))
            .collect();

        let run = SortedRun::create(&path, &entries, 0, run_id)?;

        // Ensure level 0 exists
        while self.sorted_runs.is_empty() {
            self.sorted_runs.push(Vec::new());
        }
        // Insert at the front (newest first)
        self.sorted_runs[0].insert(0, run);

        // Clear memtable
        self.memtable.clear();
        self.memtable_size = 0;

        Ok(())
    }

    /// Compacts runs at the given level into a single run at the next level.
    ///
    /// Only runs if the level exceeds `max_runs_per_level`.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::StorageError`] if file operations fail.
    pub fn compact(&mut self, level: usize) -> Result<(), VoltError> {
        if level >= self.sorted_runs.len() {
            return Ok(());
        }
        if self.sorted_runs[level].len() <= self.config.max_runs_per_level {
            return Ok(());
        }

        let next_level = level + 1;
        if next_level >= self.config.max_levels {
            return Ok(()); // Cannot compact beyond max level
        }

        // Collect all entries from runs at this level (merge sort by frame_id)
        let mut all_entries: BTreeMap<u64, Vec<u8>> = BTreeMap::new();
        for run in &self.sorted_runs[level] {
            for (frame_id, entry) in run.scan_all() {
                let bytes = entry.to_bytes()?;
                // Newer entries (from runs with higher run_id) take precedence.
                // Since runs are sorted newest-first and BTreeMap::insert overwrites,
                // we only insert if the key doesn't exist yet (preserve newest).
                all_entries.entry(frame_id).or_insert(bytes);
            }
        }

        // Create merged run at next level
        let run_id = self.next_run_id;
        self.next_run_id += 1;

        let path = self
            .config
            .data_dir
            .join(format!("run_{run_id:04}_L{next_level}.vxr"));

        let entries: Vec<(u64, Vec<u8>)> = all_entries.into_iter().collect();
        let merged_run = SortedRun::create(&path, &entries, next_level, run_id)?;

        // Delete old runs at this level
        let old_paths: Vec<PathBuf> = self.sorted_runs[level]
            .iter()
            .map(|r| r.path().to_path_buf())
            .collect();

        // Drop old runs first to release mmap handles (required on Windows)
        self.sorted_runs[level].clear();

        for old_path in &old_paths {
            let _ = fs::remove_file(old_path);
        }

        // Add merged run to next level
        while self.sorted_runs.len() <= next_level {
            self.sorted_runs.push(Vec::new());
        }
        self.sorted_runs[next_level].insert(0, merged_run);

        Ok(())
    }

    /// Automatically flushes and compacts if thresholds are exceeded.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::StorageError`] if any I/O operation fails.
    pub fn maybe_flush_and_compact(&mut self) -> Result<(), VoltError> {
        if self.memtable_size >= self.config.memtable_flush_threshold {
            self.flush_memtable()?;
        }

        for level in 0..self.config.max_levels.saturating_sub(1) {
            if level < self.sorted_runs.len()
                && self.sorted_runs[level].len() > self.config.max_runs_per_level
            {
                self.compact(level)?;
            }
        }

        Ok(())
    }

    /// Returns the total number of entries across memtable and all runs.
    pub fn total_entries(&self) -> usize {
        let mut count = self.memtable.len();
        for level_runs in &self.sorted_runs {
            for run in level_runs {
                count += run.entry_count;
            }
        }
        count
    }

    /// Returns the approximate disk size in bytes.
    pub fn disk_size_bytes(&self) -> u64 {
        let mut total = 0u64;
        for level_runs in &self.sorted_runs {
            for run in level_runs {
                total += run.mmap.len() as u64;
            }
        }
        total
    }

    /// Returns the number of entries in the memtable.
    pub fn memtable_len(&self) -> usize {
        self.memtable.len()
    }

    /// Returns the number of sorted runs at each level.
    pub fn runs_per_level(&self) -> Vec<usize> {
        self.sorted_runs.iter().map(|r| r.len()).collect()
    }

    /// Replaces a frame entry in the memtable (used by GC for in-place decay).
    ///
    /// If the frame is in the memtable, it is updated. If it's in a sorted
    /// run, the new version is written to the memtable (LSM merge semantics:
    /// newest version wins on compaction).
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::StorageError`] if serialization fails.
    pub fn update(&mut self, entry: FrameEntry) -> Result<(), VoltError> {
        let frame_id = entry.frame_id();
        let bytes = entry.to_bytes()?;
        let bytes_len = bytes.len();

        // Update memtable size tracking
        if let Some(old) = self.memtable.insert(frame_id, bytes) {
            // Replaced existing entry — adjust size
            self.memtable_size = self.memtable_size - old.len() + bytes_len;
        } else {
            self.memtable_size += bytes_len;
        }

        Ok(())
    }
}

/// Parses a sorted run filename into (run_id, level).
///
/// Expected format: `run_NNNN_LM.vxr`
fn parse_run_filename(name: &str) -> Option<(u64, usize)> {
    let name = name.strip_suffix(".vxr")?;
    let name = name.strip_prefix("run_")?;
    let parts: Vec<&str> = name.split("_L").collect();
    if parts.len() != 2 {
        return None;
    }
    let run_id = parts[0].parse::<u64>().ok()?;
    let level = parts[1].parse::<usize>().ok()?;
    Some((run_id, level))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compressed::{compress, to_gist_frame, to_tombstone};
    use volt_core::slot::{SlotData, SlotRole};
    use volt_core::{TensorFrame, SLOT_DIM};

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir()
            .join("volt_t2_test")
            .join(name)
            .join(format!("{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn make_test_frame(id: u64, strand: u64) -> TensorFrame {
        let mut frame = TensorFrame::new();
        let mut slot = SlotData::new(SlotRole::Agent);
        let val = (id as f32) * 0.01;
        slot.write_resolution(0, [val; SLOT_DIM]);
        frame.write_slot(0, slot).unwrap();
        frame.frame_meta.frame_id = id;
        frame.frame_meta.strand_id = strand;
        frame.frame_meta.created_at = id * 1000;
        frame
    }

    #[test]
    fn parse_run_filename_valid() {
        assert_eq!(parse_run_filename("run_0001_L0.vxr"), Some((1, 0)));
        assert_eq!(parse_run_filename("run_0042_L3.vxr"), Some((42, 3)));
    }

    #[test]
    fn parse_run_filename_invalid() {
        assert_eq!(parse_run_filename("not_a_run.vxr"), None);
        assert_eq!(parse_run_filename("run_abc_L0.vxr"), None);
        assert_eq!(parse_run_filename("run_0001_L0.txt"), None);
    }

    #[test]
    fn memtable_insert_and_get() {
        let dir = temp_dir("memtable_get");
        let config = T2Config {
            data_dir: dir.clone(),
            memtable_flush_threshold: 100 * 1024 * 1024, // Large, no auto-flush
            ..T2Config::default()
        };
        let mut store = Tier2Store::open(config).unwrap();

        let frame = make_test_frame(1, 0);
        store
            .insert(FrameEntry::Compressed(compress(&frame)))
            .unwrap();

        let result = store.get(1);
        assert!(result.is_some());
        assert_eq!(result.unwrap().frame_id(), 1);

        assert!(store.get(999).is_none());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn flush_creates_sorted_run() {
        let dir = temp_dir("flush_run");
        let config = T2Config {
            data_dir: dir.clone(),
            memtable_flush_threshold: 100 * 1024 * 1024,
            ..T2Config::default()
        };
        let mut store = Tier2Store::open(config).unwrap();

        for i in 1..=10u64 {
            let frame = make_test_frame(i, 0);
            store
                .insert(FrameEntry::Compressed(compress(&frame)))
                .unwrap();
        }
        assert_eq!(store.memtable_len(), 10);

        store.flush_memtable().unwrap();
        assert_eq!(store.memtable_len(), 0);
        assert_eq!(store.runs_per_level()[0], 1);

        // Entries should still be retrievable from the sorted run
        for i in 1..=10u64 {
            let result = store.get(i);
            assert!(result.is_some(), "frame {i} not found after flush");
            assert_eq!(result.unwrap().frame_id(), i);
        }

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn get_after_flush() {
        let dir = temp_dir("get_after_flush");
        let config = T2Config {
            data_dir: dir.clone(),
            memtable_flush_threshold: 100 * 1024 * 1024,
            ..T2Config::default()
        };
        let mut store = Tier2Store::open(config).unwrap();

        let frame = make_test_frame(42, 0);
        store
            .insert(FrameEntry::Compressed(compress(&frame)))
            .unwrap();
        store.flush_memtable().unwrap();

        let entry = store.get(42).unwrap();
        assert_eq!(entry.frame_id(), 42);
        assert_eq!(entry.decay_level(), DecayLevel::Compressed);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn bloom_prevents_scan() {
        let dir = temp_dir("bloom_scan");
        let config = T2Config {
            data_dir: dir.clone(),
            memtable_flush_threshold: 100 * 1024 * 1024,
            ..T2Config::default()
        };
        let mut store = Tier2Store::open(config).unwrap();

        // Insert frames 1-100
        for i in 1..=100u64 {
            let frame = make_test_frame(i, 0);
            store
                .insert(FrameEntry::Compressed(compress(&frame)))
                .unwrap();
        }
        store.flush_memtable().unwrap();

        // Query for frame IDs that were NOT inserted
        // Bloom filter should reject most (or all) of these
        let mut bloom_hits = 0;
        for id in 1000..1100u64 {
            if store.get(id).is_some() {
                bloom_hits += 1;
            }
        }
        // We expect 0 or very few false positives
        assert!(bloom_hits < 5, "too many bloom false positives: {bloom_hits}");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn compaction_merges() {
        let dir = temp_dir("compaction");
        let config = T2Config {
            data_dir: dir.clone(),
            memtable_flush_threshold: 100 * 1024 * 1024,
            max_runs_per_level: 2, // Compact after 3 runs at level 0
            max_levels: 4,
        };
        let mut store = Tier2Store::open(config).unwrap();

        // Create 3 L0 runs
        for batch in 0..3 {
            for i in 0..5u64 {
                let id = batch * 100 + i + 1;
                let frame = make_test_frame(id, 0);
                store
                    .insert(FrameEntry::Compressed(compress(&frame)))
                    .unwrap();
            }
            store.flush_memtable().unwrap();
        }
        assert_eq!(store.runs_per_level()[0], 3);

        // Compact
        store.compact(0).unwrap();

        // L0 should be empty, L1 should have 1 run
        assert_eq!(store.runs_per_level()[0], 0);
        assert_eq!(store.runs_per_level()[1], 1);

        // All entries should still be retrievable
        for batch in 0..3 {
            for i in 0..5u64 {
                let id = batch * 100 + i + 1;
                assert!(
                    store.get(id).is_some(),
                    "frame {id} not found after compaction"
                );
            }
        }

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn tombstone_stored_and_retrieved() {
        let dir = temp_dir("tombstone");
        let config = T2Config {
            data_dir: dir.clone(),
            ..T2Config::default()
        };
        let mut store = Tier2Store::open(config).unwrap();

        let ts = to_tombstone(42, 0, 5000, None);
        store.insert(FrameEntry::Tombstone(ts)).unwrap();

        let entry = store.get(42).unwrap();
        assert_eq!(entry.decay_level(), DecayLevel::Tombstoned);
        assert_eq!(entry.frame_id(), 42);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn scan_strand_filters() {
        let dir = temp_dir("scan_strand");
        let config = T2Config {
            data_dir: dir.clone(),
            ..T2Config::default()
        };
        let mut store = Tier2Store::open(config).unwrap();

        for i in 1..=5u64 {
            let frame = make_test_frame(i, 0);
            store
                .insert(FrameEntry::Compressed(compress(&frame)))
                .unwrap();
        }
        for i in 6..=10u64 {
            let frame = make_test_frame(i, 1);
            store
                .insert(FrameEntry::Compressed(compress(&frame)))
                .unwrap();
        }

        let strand_0 = store.scan_strand(0);
        let strand_1 = store.scan_strand(1);
        assert_eq!(strand_0.len(), 5);
        assert_eq!(strand_1.len(), 5);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn reopen_finds_existing_runs() {
        let dir = temp_dir("reopen");
        let config = T2Config {
            data_dir: dir.clone(),
            ..T2Config::default()
        };

        // Create and flush
        {
            let mut store = Tier2Store::open(config.clone()).unwrap();
            for i in 1..=10u64 {
                let frame = make_test_frame(i, 0);
                store
                    .insert(FrameEntry::Compressed(compress(&frame)))
                    .unwrap();
            }
            store.flush_memtable().unwrap();
        }

        // Reopen
        let store = Tier2Store::open(config).unwrap();
        for i in 1..=10u64 {
            assert!(store.get(i).is_some(), "frame {i} not found after reopen");
        }

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn auto_flush_on_threshold() {
        let dir = temp_dir("auto_flush");
        let config = T2Config {
            data_dir: dir.clone(),
            memtable_flush_threshold: 100, // Very small threshold
            ..T2Config::default()
        };
        let mut store = Tier2Store::open(config).unwrap();

        // Insert enough data to exceed the tiny threshold
        for i in 1..=20u64 {
            let frame = make_test_frame(i, 0);
            store
                .insert(FrameEntry::Compressed(compress(&frame)))
                .unwrap();
        }

        // Should have auto-flushed at least once
        assert!(store.runs_per_level()[0] > 0 || store.memtable_len() < 20);

        // All entries should be retrievable
        for i in 1..=20u64 {
            assert!(
                store.get(i).is_some(),
                "frame {i} not found after auto-flush"
            );
        }

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn update_replaces_in_memtable() {
        let dir = temp_dir("update");
        let config = T2Config {
            data_dir: dir.clone(),
            ..T2Config::default()
        };
        let mut store = Tier2Store::open(config).unwrap();

        let frame = make_test_frame(1, 0);
        store
            .insert(FrameEntry::Compressed(compress(&frame)))
            .unwrap();

        // Update to gist
        let compressed = compress(&frame);
        let gist = to_gist_frame(&compressed, [0.5; SLOT_DIM]);
        store.update(FrameEntry::Gist(gist)).unwrap();

        let entry = store.get(1).unwrap();
        assert_eq!(entry.decay_level(), DecayLevel::Gist);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn empty_flush_is_noop() {
        let dir = temp_dir("empty_flush");
        let config = T2Config {
            data_dir: dir.clone(),
            ..T2Config::default()
        };
        let mut store = Tier2Store::open(config).unwrap();

        store.flush_memtable().unwrap();
        assert_eq!(store.total_entries(), 0);
        assert!(store.runs_per_level().iter().all(|&n| n == 0));

        let _ = fs::remove_dir_all(&dir);
    }
}
