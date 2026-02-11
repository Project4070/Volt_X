//! Write-Ahead Log (WAL) for crash recovery.
//!
//! Each strand gets its own append-only binary log file. On crash,
//! the WAL is replayed to recover frames not yet flushed to T2.
//!
//! ## Entry Format
//!
//! ```text
//! [entry_len: u32][frame_id: u64][strand_id: u64][op: u8]
//! [payload_len: u32][payload: bytes][crc32: u32]
//! ```
//!
//! The CRC32 covers everything from `entry_len` through `payload`.
//! Corrupt or truncated entries at the tail are skipped on replay.

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::Write as IoWrite;
use std::path::{Path, PathBuf};

use crc32fast::Hasher;
use volt_core::VoltError;

/// WAL operation type.
///
/// # Example
///
/// ```
/// use volt_db::wal::WalOp;
///
/// assert_eq!(WalOp::from_tag(0), Some(WalOp::Store));
/// assert_eq!(WalOp::Tombstone.tag(), 3);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WalOp {
    /// A new frame was stored.
    Store = 0,
    /// A frame was compressed (GC demotion).
    Compress = 1,
    /// A frame was gisted (GC demotion).
    Gist = 2,
    /// A frame was tombstoned (GC demotion).
    Tombstone = 3,
}

impl WalOp {
    /// Converts a `u8` tag to a `WalOp`.
    pub fn from_tag(tag: u8) -> Option<Self> {
        match tag {
            0 => Some(Self::Store),
            1 => Some(Self::Compress),
            2 => Some(Self::Gist),
            3 => Some(Self::Tombstone),
            _ => None,
        }
    }

    /// Returns the `u8` tag for this operation.
    pub fn tag(self) -> u8 {
        self as u8
    }
}

/// A single WAL entry.
///
/// # Example
///
/// ```
/// use volt_db::wal::{WalEntry, WalOp};
///
/// let entry = WalEntry {
///     frame_id: 42,
///     strand_id: 1,
///     op: WalOp::Store,
///     payload: vec![1, 2, 3],
/// };
/// assert_eq!(entry.frame_id, 42);
/// ```
#[derive(Debug, Clone)]
pub struct WalEntry {
    /// Frame identifier.
    pub frame_id: u64,
    /// Strand identifier.
    pub strand_id: u64,
    /// The operation type.
    pub op: WalOp,
    /// Serialized frame data (e.g. from `FrameEntry::to_bytes()`).
    pub payload: Vec<u8>,
}

impl WalEntry {
    /// Serializes this entry to bytes including CRC32.
    fn to_bytes(&self) -> Vec<u8> {
        let payload_len = self.payload.len() as u32;
        // entry_len covers: frame_id(8) + strand_id(8) + op(1) + payload_len(4) + payload
        let entry_len: u32 = 8 + 8 + 1 + 4 + payload_len;

        let mut buf = Vec::with_capacity(4 + entry_len as usize + 4);

        // Header
        buf.extend_from_slice(&entry_len.to_le_bytes());
        buf.extend_from_slice(&self.frame_id.to_le_bytes());
        buf.extend_from_slice(&self.strand_id.to_le_bytes());
        buf.push(self.op.tag());
        buf.extend_from_slice(&payload_len.to_le_bytes());
        buf.extend_from_slice(&self.payload);

        // CRC32 over everything before this point
        let mut hasher = Hasher::new();
        hasher.update(&buf);
        let crc = hasher.finalize();
        buf.extend_from_slice(&crc.to_le_bytes());

        buf
    }

    /// Tries to read one entry from bytes at the given offset.
    ///
    /// Returns `(entry, bytes_consumed)` on success, or `None` if the
    /// data is truncated or the CRC doesn't match.
    fn from_bytes_at(data: &[u8], offset: usize) -> Option<(Self, usize)> {
        let remaining = data.len().checked_sub(offset)?;
        if remaining < 4 {
            return None;
        }

        let entry_len =
            u32::from_le_bytes(data[offset..offset + 4].try_into().ok()?) as usize;

        // Total: 4 (entry_len) + entry_len + 4 (crc)
        let total = 4 + entry_len + 4;
        if remaining < total {
            return None;
        }

        let entry_start = offset + 4;
        let entry_end = entry_start + entry_len;
        let crc_start = entry_end;

        // Verify CRC
        let stored_crc = u32::from_le_bytes(
            data[crc_start..crc_start + 4].try_into().ok()?,
        );
        let mut hasher = Hasher::new();
        hasher.update(&data[offset..entry_end]);
        let computed_crc = hasher.finalize();
        if stored_crc != computed_crc {
            return None;
        }

        // Parse entry
        if entry_len < 21 {
            // 8 + 8 + 1 + 4 = 21 minimum
            return None;
        }
        let mut pos = entry_start;
        let frame_id = u64::from_le_bytes(data[pos..pos + 8].try_into().ok()?);
        pos += 8;
        let strand_id = u64::from_le_bytes(data[pos..pos + 8].try_into().ok()?);
        pos += 8;
        let op = WalOp::from_tag(data[pos])?;
        pos += 1;
        let payload_len =
            u32::from_le_bytes(data[pos..pos + 4].try_into().ok()?) as usize;
        pos += 4;

        if pos + payload_len > entry_end {
            return None;
        }
        let payload = data[pos..pos + payload_len].to_vec();

        Some((
            Self {
                frame_id,
                strand_id,
                op,
                payload,
            },
            total,
        ))
    }
}

/// Per-strand WAL file.
///
/// Each strand gets its own `.wal` file in the WAL directory.
pub struct StrandWal {
    file: File,
    strand_id: u64,
    entry_count: usize,
    path: PathBuf,
}

impl std::fmt::Debug for StrandWal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StrandWal")
            .field("strand_id", &self.strand_id)
            .field("entry_count", &self.entry_count)
            .field("path", &self.path)
            .finish()
    }
}

impl StrandWal {
    /// Opens or creates a WAL file for the given strand.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::StorageError`] if the file cannot be opened.
    fn open_or_create(dir: &Path, strand_id: u64) -> Result<Self, VoltError> {
        let path = dir.join(format!("strand_{strand_id}.wal"));
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .read(true)
            .open(&path)
            .map_err(|e| VoltError::StorageError {
                message: format!("failed to open WAL for strand {strand_id}: {e}"),
            })?;

        Ok(Self {
            file,
            strand_id,
            entry_count: 0,
            path,
        })
    }

    /// Appends a WAL entry.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::StorageError`] if the write fails.
    fn append(&mut self, entry: &WalEntry) -> Result<(), VoltError> {
        let bytes = entry.to_bytes();
        self.file
            .write_all(&bytes)
            .map_err(|e| VoltError::StorageError {
                message: format!(
                    "failed to write WAL entry for strand {}: {e}",
                    self.strand_id
                ),
            })?;
        self.entry_count += 1;
        Ok(())
    }

    /// Flushes the WAL to disk (fsync).
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::StorageError`] if the sync fails.
    fn sync(&self) -> Result<(), VoltError> {
        self.file
            .sync_all()
            .map_err(|e| VoltError::StorageError {
                message: format!(
                    "failed to sync WAL for strand {}: {e}",
                    self.strand_id
                ),
            })
    }

    /// Replays all valid entries from this WAL file.
    ///
    /// Skips corrupt or truncated entries at the tail.
    fn replay(&self) -> Result<Vec<WalEntry>, VoltError> {
        let data = fs::read(&self.path).map_err(|e| VoltError::StorageError {
            message: format!(
                "failed to read WAL for strand {}: {e}",
                self.strand_id
            ),
        })?;

        let mut entries = Vec::new();
        let mut offset = 0;

        while offset < data.len() {
            match WalEntry::from_bytes_at(&data, offset) {
                Some((entry, consumed)) => {
                    entries.push(entry);
                    offset += consumed;
                }
                None => {
                    // Corrupt or truncated entry at tail — stop replay
                    break;
                }
            }
        }

        Ok(entries)
    }

    /// Truncates (clears) the WAL file after a successful checkpoint.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::StorageError`] if truncation fails.
    fn truncate(&mut self) -> Result<(), VoltError> {
        // Close and reopen with truncation
        self.file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .read(true)
            .open(&self.path)
            .map_err(|e| VoltError::StorageError {
                message: format!(
                    "failed to truncate WAL for strand {}: {e}",
                    self.strand_id
                ),
            })?;
        // Reopen in append mode
        self.file = OpenOptions::new()
            .append(true)
            .read(true)
            .open(&self.path)
            .map_err(|e| VoltError::StorageError {
                message: format!(
                    "failed to reopen WAL for strand {}: {e}",
                    self.strand_id
                ),
            })?;
        self.entry_count = 0;
        Ok(())
    }
}

/// Manages all per-strand WAL files in a directory.
///
/// # Example
///
/// ```no_run
/// use volt_db::wal::{WalManager, WalEntry, WalOp};
/// use std::path::Path;
///
/// let mut wal = WalManager::open(Path::new("/tmp/voltdb_wal")).unwrap();
/// wal.log_entry(WalEntry {
///     frame_id: 1,
///     strand_id: 0,
///     op: WalOp::Store,
///     payload: vec![],
/// }).unwrap();
/// wal.sync_all().unwrap();
/// ```
#[derive(Debug)]
pub struct WalManager {
    dir: PathBuf,
    wals: HashMap<u64, StrandWal>,
}

impl WalManager {
    /// Opens (or creates) the WAL directory and discovers existing WAL files.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::StorageError`] if the directory cannot be created.
    pub fn open(dir: &Path) -> Result<Self, VoltError> {
        fs::create_dir_all(dir).map_err(|e| VoltError::StorageError {
            message: format!("failed to create WAL directory {}: {e}", dir.display()),
        })?;

        let mut wals = HashMap::new();

        // Discover existing WAL files
        let entries = fs::read_dir(dir).map_err(|e| VoltError::StorageError {
            message: format!("failed to read WAL directory {}: {e}", dir.display()),
        })?;

        for entry in entries {
            let entry = entry.map_err(|e| VoltError::StorageError {
                message: format!("failed to read WAL directory entry: {e}"),
            })?;
            let file_name = entry.file_name();
            let name = file_name.to_string_lossy();
            if let Some(id_str) = name
                .strip_prefix("strand_")
                .and_then(|s| s.strip_suffix(".wal"))
                && let Ok(strand_id) = id_str.parse::<u64>()
            {
                let wal = StrandWal::open_or_create(dir, strand_id)?;
                wals.insert(strand_id, wal);
            }
        }

        Ok(Self {
            dir: dir.to_path_buf(),
            wals,
        })
    }

    /// Logs a WAL entry, creating the strand's WAL file if needed.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::StorageError`] if the write fails.
    pub fn log_entry(&mut self, entry: WalEntry) -> Result<(), VoltError> {
        let strand_id = entry.strand_id;
        let wal = self.get_or_create_wal(strand_id)?;
        wal.append(&entry)
    }

    /// Flushes all open WAL files to disk.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::StorageError`] if any sync fails.
    pub fn sync_all(&self) -> Result<(), VoltError> {
        for wal in self.wals.values() {
            wal.sync()?;
        }
        Ok(())
    }

    /// Replays all WAL files, returning entries grouped by strand.
    ///
    /// Corrupt entries at the tail of each WAL are silently skipped.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::StorageError`] if any file read fails.
    pub fn replay_all(&self) -> Result<HashMap<u64, Vec<WalEntry>>, VoltError> {
        let mut result = HashMap::new();
        for (&strand_id, wal) in &self.wals {
            let entries = wal.replay()?;
            if !entries.is_empty() {
                result.insert(strand_id, entries);
            }
        }
        Ok(result)
    }

    /// Truncates the WAL for a strand after a successful T2 flush.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::StorageError`] if truncation fails.
    pub fn checkpoint(&mut self, strand_id: u64) -> Result<(), VoltError> {
        if let Some(wal) = self.wals.get_mut(&strand_id) {
            wal.truncate()?;
        }
        Ok(())
    }

    /// Truncates all WAL files.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::StorageError`] if any truncation fails.
    pub fn checkpoint_all(&mut self) -> Result<(), VoltError> {
        let strand_ids: Vec<u64> = self.wals.keys().copied().collect();
        for strand_id in strand_ids {
            self.checkpoint(strand_id)?;
        }
        Ok(())
    }

    /// Returns the WAL directory path.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    fn get_or_create_wal(&mut self, strand_id: u64) -> Result<&mut StrandWal, VoltError> {
        if !self.wals.contains_key(&strand_id) {
            let wal = StrandWal::open_or_create(&self.dir, strand_id)?;
            self.wals.insert(strand_id, wal);
        }
        Ok(self.wals.get_mut(&strand_id).expect("just inserted"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir()
            .join("volt_wal_test")
            .join(name)
            .join(format!("{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn wal_op_roundtrip() {
        for op in [WalOp::Store, WalOp::Compress, WalOp::Gist, WalOp::Tombstone] {
            assert_eq!(WalOp::from_tag(op.tag()), Some(op));
        }
        assert_eq!(WalOp::from_tag(99), None);
    }

    #[test]
    fn wal_entry_bytes_roundtrip() {
        let entry = WalEntry {
            frame_id: 42,
            strand_id: 1,
            op: WalOp::Store,
            payload: vec![10, 20, 30, 40, 50],
        };
        let bytes = entry.to_bytes();
        let (restored, consumed) = WalEntry::from_bytes_at(&bytes, 0).unwrap();

        assert_eq!(consumed, bytes.len());
        assert_eq!(restored.frame_id, 42);
        assert_eq!(restored.strand_id, 1);
        assert_eq!(restored.op, WalOp::Store);
        assert_eq!(restored.payload, vec![10, 20, 30, 40, 50]);
    }

    #[test]
    fn append_and_replay() {
        let dir = temp_dir("append_and_replay");
        let mut wal = WalManager::open(&dir).unwrap();

        for i in 0..100u64 {
            wal.log_entry(WalEntry {
                frame_id: i,
                strand_id: 0,
                op: WalOp::Store,
                payload: vec![i as u8],
            })
            .unwrap();
        }
        wal.sync_all().unwrap();

        // Reopen and replay
        let wal2 = WalManager::open(&dir).unwrap();
        let entries = wal2.replay_all().unwrap();
        let strand_entries = &entries[&0];
        assert_eq!(strand_entries.len(), 100);
        for (i, e) in strand_entries.iter().enumerate() {
            assert_eq!(e.frame_id, i as u64);
            assert_eq!(e.payload, vec![i as u8]);
        }

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn corrupt_tail_recovery() {
        let dir = temp_dir("corrupt_tail");
        let mut wal = WalManager::open(&dir).unwrap();

        // Write 5 valid entries
        for i in 0..5u64 {
            wal.log_entry(WalEntry {
                frame_id: i,
                strand_id: 0,
                op: WalOp::Store,
                payload: vec![i as u8; 10],
            })
            .unwrap();
        }
        wal.sync_all().unwrap();

        // Append garbage to the WAL file to simulate partial write
        let wal_path = dir.join("strand_0.wal");
        let mut file = OpenOptions::new().append(true).open(&wal_path).unwrap();
        file.write_all(b"GARBAGE_PARTIAL_WRITE").unwrap();
        file.sync_all().unwrap();

        // Replay should recover the 5 valid entries
        let wal2 = WalManager::open(&dir).unwrap();
        let entries = wal2.replay_all().unwrap();
        let strand_entries = &entries[&0];
        assert_eq!(strand_entries.len(), 5);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn crc_mismatch_detection() {
        let dir = temp_dir("crc_mismatch");
        let mut wal = WalManager::open(&dir).unwrap();

        wal.log_entry(WalEntry {
            frame_id: 1,
            strand_id: 0,
            op: WalOp::Store,
            payload: vec![0; 20],
        })
        .unwrap();
        wal.log_entry(WalEntry {
            frame_id: 2,
            strand_id: 0,
            op: WalOp::Store,
            payload: vec![1; 20],
        })
        .unwrap();
        wal.sync_all().unwrap();

        // Corrupt a byte in the middle of the first entry
        let wal_path = dir.join("strand_0.wal");
        let mut data = fs::read(&wal_path).unwrap();
        // Flip a byte in the payload area of the first entry
        if data.len() > 25 {
            data[25] ^= 0xFF;
        }
        fs::write(&wal_path, &data).unwrap();

        // Replay should stop at the corrupt first entry
        let wal2 = WalManager::open(&dir).unwrap();
        let entries = wal2.replay_all().unwrap();
        // The first entry is corrupt, so replay stops — 0 entries recovered
        let strand_entries = entries.get(&0).map(|v| v.len()).unwrap_or(0);
        assert_eq!(strand_entries, 0);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn checkpoint_truncates() {
        let dir = temp_dir("checkpoint");
        let mut wal = WalManager::open(&dir).unwrap();

        for i in 0..10u64 {
            wal.log_entry(WalEntry {
                frame_id: i,
                strand_id: 0,
                op: WalOp::Store,
                payload: vec![],
            })
            .unwrap();
        }
        wal.sync_all().unwrap();

        wal.checkpoint(0).unwrap();

        // Reopen — should have 0 entries
        let wal2 = WalManager::open(&dir).unwrap();
        let entries = wal2.replay_all().unwrap();
        assert!(entries.is_empty() || entries[&0].is_empty());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn multiple_strands() {
        let dir = temp_dir("multi_strand");
        let mut wal = WalManager::open(&dir).unwrap();

        for i in 0..5u64 {
            wal.log_entry(WalEntry {
                frame_id: i,
                strand_id: 0,
                op: WalOp::Store,
                payload: vec![],
            })
            .unwrap();
        }
        for i in 10..15u64 {
            wal.log_entry(WalEntry {
                frame_id: i,
                strand_id: 1,
                op: WalOp::Store,
                payload: vec![],
            })
            .unwrap();
        }
        wal.sync_all().unwrap();

        let wal2 = WalManager::open(&dir).unwrap();
        let entries = wal2.replay_all().unwrap();
        assert_eq!(entries[&0].len(), 5);
        assert_eq!(entries[&1].len(), 5);
        assert_eq!(entries[&0][0].frame_id, 0);
        assert_eq!(entries[&1][0].frame_id, 10);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn empty_wal_replays_empty() {
        let dir = temp_dir("empty_replay");
        let wal = WalManager::open(&dir).unwrap();
        let entries = wal.replay_all().unwrap();
        assert!(entries.is_empty());

        let _ = fs::remove_dir_all(&dir);
    }
}
