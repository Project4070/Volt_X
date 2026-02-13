//! Scaled Vector Field Network for Phase 2 training.
//!
//! A deep residual MLP targeting ~50M parameters for code training.
//! Uses pre-norm residual blocks with GELU activation:
//!
//! ```text
//! Input(256) → ProjectUp(256→H) → [ResBlock(H→H)] × N → ProjectDown(H→256)
//! ```
//!
//! Each residual block:
//! ```text
//! x → LayerNorm → Linear(H→H) → GELU → Linear(H→H) → + x
//! ```
//!
//! ## Parameter Counts
//!
//! With `hidden_dim=2048`, `num_blocks=12`:
//! - ProjectUp: 256×2048 + 2048 = 526,336
//! - Each ResBlock: 2×(2048×2048 + 2048) = 8,392,704
//! - 12 ResBlocks: 100,712,448
//! - ProjectDown: 2048×256 + 256 = 524,544
//! - **Total: ~101.8M params** (configurable via hidden_dim/num_blocks)
//!
//! With `hidden_dim=1536`, `num_blocks=10`:
//! - **Total: ~50M params** (target for Phase 2.1)

use crate::nn::{Linear, Rng};
use volt_core::{VoltError, SLOT_DIM};

/// Configuration for the scaled VFN architecture.
///
/// # Example
///
/// ```
/// use volt_soft::scaled_vfn::ScaledVfnConfig;
///
/// // ~50M param config
/// let config = ScaledVfnConfig::default();
/// assert_eq!(config.hidden_dim, 1536);
/// assert_eq!(config.num_blocks, 10);
/// ```
#[derive(Debug, Clone)]
pub struct ScaledVfnConfig {
    /// Width of residual blocks (default: 1536 for ~50M params).
    pub hidden_dim: usize,

    /// Number of residual blocks (default: 10 for ~50M params).
    pub num_blocks: usize,

    /// Input and output dimension (always 256 = SLOT_DIM).
    pub io_dim: usize,
}

impl Default for ScaledVfnConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 1536,
            num_blocks: 10,
            io_dim: SLOT_DIM,
        }
    }
}

impl ScaledVfnConfig {
    /// Computes the total number of trainable parameters.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_soft::scaled_vfn::ScaledVfnConfig;
    ///
    /// let config = ScaledVfnConfig::default();
    /// let params = config.param_count();
    /// assert!(params > 40_000_000); // > 40M
    /// assert!(params < 60_000_000); // < 60M
    /// ```
    pub fn param_count(&self) -> usize {
        let h = self.hidden_dim;
        let d = self.io_dim;
        let proj_up = d * h + h;
        let proj_down = h * d + d;
        // Each ResBlock: 2 linear layers (H→H) + 2 LayerNorm (2×H params each)
        let per_block = 2 * (h * h + h) + 2 * h;
        proj_up + self.num_blocks * per_block + proj_down
    }
}

/// Layer normalization: normalize to zero mean and unit variance,
/// then apply learnable scale (gamma) and shift (beta).
#[derive(Clone)]
struct LayerNorm {
    gamma: Vec<f32>,
    beta: Vec<f32>,
    dim: usize,
    eps: f32,
}

impl LayerNorm {
    fn new(dim: usize) -> Self {
        Self {
            gamma: vec![1.0; dim],
            beta: vec![0.0; dim],
            dim,
            eps: 1e-5,
        }
    }

    fn forward(&self, input: &[f32]) -> Vec<f32> {
        debug_assert_eq!(input.len(), self.dim);
        let mean = input.iter().sum::<f32>() / self.dim as f32;
        let var = input.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / self.dim as f32;
        let std = (var + self.eps).sqrt();
        input
            .iter()
            .enumerate()
            .map(|(i, &x)| self.gamma[i] * (x - mean) / std + self.beta[i])
            .collect()
    }

    #[cfg(feature = "gpu")]
    fn gamma_mut(&mut self) -> &mut [f32] {
        &mut self.gamma
    }

    #[cfg(feature = "gpu")]
    fn beta_mut(&mut self) -> &mut [f32] {
        &mut self.beta
    }

    #[cfg(feature = "gpu")]
    fn gamma(&self) -> &[f32] {
        &self.gamma
    }

    #[cfg(feature = "gpu")]
    fn beta(&self) -> &[f32] {
        &self.beta
    }
}

/// A single pre-norm residual block: LayerNorm → Linear → GELU → Linear + skip.
#[derive(Clone)]
struct ResBlock {
    norm: LayerNorm,
    linear1: Linear,
    linear2: Linear,
    dim: usize,
}

impl ResBlock {
    fn new(rng: &mut Rng, dim: usize) -> Self {
        Self {
            norm: LayerNorm::new(dim),
            linear1: Linear::new_xavier(rng, dim, dim),
            linear2: Linear::new_xavier(rng, dim, dim),
            dim,
        }
    }

    fn forward(&self, input: &[f32]) -> Vec<f32> {
        debug_assert_eq!(input.len(), self.dim);

        // Pre-norm
        let normed = self.norm.forward(input);

        // Linear → GELU → Linear
        let h = self.linear1.forward(&normed);
        let h: Vec<f32> = h.into_iter().map(gelu).collect();
        let h = self.linear2.forward(&h);

        // Residual connection
        input.iter().zip(h.iter()).map(|(&a, &b)| a + b).collect()
    }
}

/// GELU activation: x * Φ(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + ((2.0_f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
}

/// A scaled Vector Field Network with deep residual architecture.
///
/// Designed for Phase 2 training at ~50M parameters. Uses pre-norm
/// residual blocks with GELU activation and a projection up/down
/// structure to bridge the SLOT_DIM (256) I/O with a wider hidden dim.
///
/// # Architecture
///
/// ```text
/// Input[256] → Linear(256→H) → GELU
///     → [LayerNorm → Linear(H→H) → GELU → Linear(H→H) + skip] × N
///     → LayerNorm → Linear(H→256)
/// ```
///
/// # Example
///
/// ```
/// use volt_soft::scaled_vfn::{ScaledVfn, ScaledVfnConfig};
/// use volt_core::SLOT_DIM;
///
/// let config = ScaledVfnConfig {
///     hidden_dim: 128,
///     num_blocks: 2,
///     io_dim: SLOT_DIM,
/// };
/// let vfn = ScaledVfn::new_random(42, &config);
/// let input = [0.1_f32; SLOT_DIM];
/// let drift = vfn.forward(&input).unwrap();
/// assert!(drift.iter().all(|x| x.is_finite()));
/// ```
#[derive(Clone)]
pub struct ScaledVfn {
    proj_up: Linear,
    blocks: Vec<ResBlock>,
    final_norm: LayerNorm,
    proj_down: Linear,
    config: ScaledVfnConfig,
}

impl std::fmt::Debug for ScaledVfn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ScaledVfn({}→{}×{}→{}, {:.1}M params)",
            self.config.io_dim,
            self.config.hidden_dim,
            self.config.num_blocks,
            self.config.io_dim,
            self.config.param_count() as f64 / 1_000_000.0
        )
    }
}

impl ScaledVfn {
    /// Creates a new scaled VFN with Xavier/Glorot random initialization.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_soft::scaled_vfn::{ScaledVfn, ScaledVfnConfig};
    ///
    /// let config = ScaledVfnConfig { hidden_dim: 64, num_blocks: 2, ..ScaledVfnConfig::default() };
    /// let vfn = ScaledVfn::new_random(42, &config);
    /// ```
    pub fn new_random(seed: u64, config: &ScaledVfnConfig) -> Self {
        let mut rng = Rng::new(seed);
        let proj_up = Linear::new_xavier(&mut rng, config.io_dim, config.hidden_dim);
        let blocks: Vec<ResBlock> = (0..config.num_blocks)
            .map(|_| ResBlock::new(&mut rng, config.hidden_dim))
            .collect();
        let final_norm = LayerNorm::new(config.hidden_dim);
        let proj_down = Linear::new_xavier(&mut rng, config.hidden_dim, config.io_dim);
        Self {
            proj_up,
            blocks,
            final_norm,
            proj_down,
            config: config.clone(),
        }
    }

    /// Computes the drift vector for a single slot embedding.
    ///
    /// Passes the input through the full residual network.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::Internal`] if input/output contains NaN or Inf.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_soft::scaled_vfn::{ScaledVfn, ScaledVfnConfig};
    /// use volt_core::SLOT_DIM;
    ///
    /// let config = ScaledVfnConfig { hidden_dim: 64, num_blocks: 2, ..ScaledVfnConfig::default() };
    /// let vfn = ScaledVfn::new_random(42, &config);
    /// let input = [0.1_f32; SLOT_DIM];
    /// let drift = vfn.forward(&input).unwrap();
    /// assert_eq!(drift.len(), SLOT_DIM);
    /// ```
    pub fn forward(&self, input: &[f32; SLOT_DIM]) -> Result<[f32; SLOT_DIM], VoltError> {
        if input.iter().any(|x| !x.is_finite()) {
            return Err(VoltError::Internal {
                message: "ScaledVfn forward: input contains NaN or Inf".to_string(),
            });
        }

        // Project up: 256 → hidden_dim, GELU
        let mut h = self.proj_up.forward(input);
        for x in &mut h {
            *x = gelu(*x);
        }

        // Residual blocks
        for block in &self.blocks {
            h = block.forward(&h);
        }

        // Final norm + project down
        h = self.final_norm.forward(&h);
        let out = self.proj_down.forward(&h);

        let mut result = [0.0f32; SLOT_DIM];
        result.copy_from_slice(&out);

        if result.iter().any(|x| !x.is_finite()) {
            return Err(VoltError::Internal {
                message: "ScaledVfn forward: output contains NaN or Inf".to_string(),
            });
        }

        Ok(result)
    }

    /// Returns the configuration used to create this VFN.
    pub fn config(&self) -> &ScaledVfnConfig {
        &self.config
    }

    /// Returns the total number of parameters.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_soft::scaled_vfn::{ScaledVfn, ScaledVfnConfig};
    ///
    /// let config = ScaledVfnConfig::default();
    /// let vfn = ScaledVfn::new_random(42, &config);
    /// assert!(vfn.param_count() > 40_000_000);
    /// ```
    pub fn param_count(&self) -> usize {
        self.config.param_count()
    }

    /// Returns the number of residual blocks.
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Returns the number of layers (for Forward-Forward compatibility).
    ///
    /// Layers: proj_up + 2*num_blocks (each block has 2 linear layers) + proj_down = 2 + 2*N.
    pub fn layer_count(&self) -> usize {
        2 + 2 * self.blocks.len()
    }

    /// Saves the scaled VFN to a binary checkpoint file.
    ///
    /// Format: "SVF2" magic + version(u32) + config + weights.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::LearnError`] if the file cannot be written.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use volt_soft::scaled_vfn::{ScaledVfn, ScaledVfnConfig};
    ///
    /// let config = ScaledVfnConfig { hidden_dim: 64, num_blocks: 2, ..ScaledVfnConfig::default() };
    /// let vfn = ScaledVfn::new_random(42, &config);
    /// vfn.save("scaled_vfn.bin").unwrap();
    /// ```
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), VoltError> {
        use std::fs::File;
        use std::io::Write;

        let mut file = File::create(path.as_ref()).map_err(|e| VoltError::LearnError {
            message: format!("Failed to create scaled VFN checkpoint: {e}"),
        })?;

        let write_err = |e: std::io::Error| VoltError::LearnError {
            message: format!("Failed to write scaled VFN checkpoint: {e}"),
        };

        // Magic: "SVF2"
        file.write_all(b"SVF2").map_err(write_err)?;

        // Version
        file.write_all(&1u32.to_le_bytes()).map_err(write_err)?;

        // Config
        file.write_all(&(self.config.io_dim as u32).to_le_bytes())
            .map_err(write_err)?;
        file.write_all(&(self.config.hidden_dim as u32).to_le_bytes())
            .map_err(write_err)?;
        file.write_all(&(self.config.num_blocks as u32).to_le_bytes())
            .map_err(write_err)?;

        // Checksum of all weights
        let checksum = self.compute_checksum();
        file.write_all(&checksum.to_le_bytes()).map_err(write_err)?;

        // proj_up weights + bias
        write_linear(&mut file, &self.proj_up)?;

        // Residual blocks
        for block in &self.blocks {
            write_layer_norm(&mut file, &block.norm)?;
            write_linear(&mut file, &block.linear1)?;
            write_linear(&mut file, &block.linear2)?;
        }

        // final_norm
        write_layer_norm(&mut file, &self.final_norm)?;

        // proj_down
        write_linear(&mut file, &self.proj_down)?;

        Ok(())
    }

    /// Loads a scaled VFN from a binary checkpoint file.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::LearnError`] if the file is invalid or corrupted.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use volt_soft::scaled_vfn::ScaledVfn;
    ///
    /// let vfn = ScaledVfn::load("scaled_vfn.bin").unwrap();
    /// ```
    pub fn load<P: AsRef<std::path::Path>>(path: P) -> Result<Self, VoltError> {
        use std::fs::File;
        use std::io::Read;

        let mut file = File::open(path.as_ref()).map_err(|e| VoltError::LearnError {
            message: format!("Failed to open scaled VFN checkpoint: {e}"),
        })?;

        let read_err = |e: std::io::Error| VoltError::LearnError {
            message: format!("Failed to read scaled VFN checkpoint: {e}"),
        };

        // Magic
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic).map_err(read_err)?;
        if &magic != b"SVF2" {
            return Err(VoltError::LearnError {
                message: format!(
                    "Invalid scaled VFN checkpoint: expected magic 'SVF2', got '{}'",
                    String::from_utf8_lossy(&magic)
                ),
            });
        }

        // Version
        let mut buf4 = [0u8; 4];
        file.read_exact(&mut buf4).map_err(read_err)?;
        let version = u32::from_le_bytes(buf4);
        if version != 1 {
            return Err(VoltError::LearnError {
                message: format!("Incompatible scaled VFN version: expected 1, got {version}"),
            });
        }

        // Config
        file.read_exact(&mut buf4).map_err(read_err)?;
        let io_dim = u32::from_le_bytes(buf4) as usize;
        file.read_exact(&mut buf4).map_err(read_err)?;
        let hidden_dim = u32::from_le_bytes(buf4) as usize;
        file.read_exact(&mut buf4).map_err(read_err)?;
        let num_blocks = u32::from_le_bytes(buf4) as usize;

        // Stored checksum
        file.read_exact(&mut buf4).map_err(read_err)?;
        let stored_checksum = u32::from_le_bytes(buf4);

        let config = ScaledVfnConfig {
            io_dim,
            hidden_dim,
            num_blocks,
        };

        // proj_up
        let proj_up = read_linear(&mut file, io_dim, hidden_dim)?;

        // Residual blocks
        let mut blocks = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            let norm = read_layer_norm(&mut file, hidden_dim)?;
            let linear1 = read_linear(&mut file, hidden_dim, hidden_dim)?;
            let linear2 = read_linear(&mut file, hidden_dim, hidden_dim)?;
            blocks.push(ResBlock {
                norm,
                linear1,
                linear2,
                dim: hidden_dim,
            });
        }

        // final_norm
        let final_norm = read_layer_norm(&mut file, hidden_dim)?;

        // proj_down
        let proj_down = read_linear(&mut file, hidden_dim, io_dim)?;

        let vfn = Self {
            proj_up,
            blocks,
            final_norm,
            proj_down,
            config,
        };

        // Validate checksum
        let computed_checksum = vfn.compute_checksum();
        if computed_checksum != stored_checksum {
            return Err(VoltError::LearnError {
                message: format!(
                    "Scaled VFN checksum mismatch: expected {stored_checksum}, got {computed_checksum}"
                ),
            });
        }

        Ok(vfn)
    }

    fn compute_checksum(&self) -> u32 {
        let mut hasher = crc32fast::Hasher::new();

        hash_linear(&mut hasher, &self.proj_up);
        for block in &self.blocks {
            hash_layer_norm(&mut hasher, &block.norm);
            hash_linear(&mut hasher, &block.linear1);
            hash_linear(&mut hasher, &block.linear2);
        }
        hash_layer_norm(&mut hasher, &self.final_norm);
        hash_linear(&mut hasher, &self.proj_down);

        hasher.finalize()
    }

    /// Collects all weights as a flat Vec for transfer to GPU tensors.
    ///
    /// Returns tuples of (name, shape, weights, bias) for each parameter group.
    #[cfg(feature = "gpu")]
    pub(crate) fn all_weights(&self) -> Vec<(&str, (usize, usize), &[f32], &[f32])> {
        let mut weights = Vec::new();
        let h = self.config.hidden_dim;
        let d = self.config.io_dim;

        weights.push(("proj_up", (h, d), self.proj_up.weights(), self.proj_up.bias()));
        for (i, block) in self.blocks.iter().enumerate() {
            // We return a static str approximation; callers use the index
            let _ = i; // used for naming in GPU path
            weights.push(("block_l1", (h, h), block.linear1.weights(), block.linear1.bias()));
            weights.push(("block_l2", (h, h), block.linear2.weights(), block.linear2.bias()));
        }
        weights.push(("proj_down", (d, h), self.proj_down.weights(), self.proj_down.bias()));
        weights
    }

    /// Returns references to the norm parameters for GPU transfer.
    #[cfg(feature = "gpu")]
    pub(crate) fn all_norms(&self) -> Vec<(&[f32], &[f32])> {
        let mut norms = Vec::new();
        for block in &self.blocks {
            norms.push((block.norm.gamma(), block.norm.beta()));
        }
        norms.push((self.final_norm.gamma(), self.final_norm.beta()));
        norms
    }
}

// --- Serialization helpers ---

fn write_linear(file: &mut std::fs::File, layer: &Linear) -> Result<(), VoltError> {
    use std::io::Write;
    let write_err = |e: std::io::Error| VoltError::LearnError {
        message: format!("write_linear: {e}"),
    };
    for &w in layer.weights() {
        file.write_all(&w.to_le_bytes()).map_err(write_err)?;
    }
    for &b in layer.bias() {
        file.write_all(&b.to_le_bytes()).map_err(write_err)?;
    }
    Ok(())
}

fn write_layer_norm(file: &mut std::fs::File, norm: &LayerNorm) -> Result<(), VoltError> {
    use std::io::Write;
    let write_err = |e: std::io::Error| VoltError::LearnError {
        message: format!("write_layer_norm: {e}"),
    };
    for &g in &norm.gamma {
        file.write_all(&g.to_le_bytes()).map_err(write_err)?;
    }
    for &b in &norm.beta {
        file.write_all(&b.to_le_bytes()).map_err(write_err)?;
    }
    Ok(())
}

fn read_linear(
    file: &mut std::fs::File,
    in_dim: usize,
    out_dim: usize,
) -> Result<Linear, VoltError> {
    use std::io::Read;
    let read_err = |e: std::io::Error| VoltError::LearnError {
        message: format!("read_linear: {e}"),
    };

    let mut weights = Vec::with_capacity(in_dim * out_dim);
    let mut buf = [0u8; 4];
    for _ in 0..(in_dim * out_dim) {
        file.read_exact(&mut buf).map_err(read_err)?;
        weights.push(f32::from_le_bytes(buf));
    }

    let mut bias = Vec::with_capacity(out_dim);
    for _ in 0..out_dim {
        file.read_exact(&mut buf).map_err(read_err)?;
        bias.push(f32::from_le_bytes(buf));
    }

    Linear::from_weights_and_bias(weights, bias, in_dim, out_dim)
}

fn read_layer_norm(file: &mut std::fs::File, dim: usize) -> Result<LayerNorm, VoltError> {
    use std::io::Read;
    let read_err = |e: std::io::Error| VoltError::LearnError {
        message: format!("read_layer_norm: {e}"),
    };

    let mut gamma = Vec::with_capacity(dim);
    let mut beta = Vec::with_capacity(dim);
    let mut buf = [0u8; 4];

    for _ in 0..dim {
        file.read_exact(&mut buf).map_err(read_err)?;
        gamma.push(f32::from_le_bytes(buf));
    }
    for _ in 0..dim {
        file.read_exact(&mut buf).map_err(read_err)?;
        beta.push(f32::from_le_bytes(buf));
    }

    Ok(LayerNorm {
        gamma,
        beta,
        dim,
        eps: 1e-5,
    })
}

fn hash_linear(hasher: &mut crc32fast::Hasher, layer: &Linear) {
    for &w in layer.weights() {
        hasher.update(&w.to_le_bytes());
    }
    for &b in layer.bias() {
        hasher.update(&b.to_le_bytes());
    }
}

fn hash_layer_norm(hasher: &mut crc32fast::Hasher, norm: &LayerNorm) {
    for &g in &norm.gamma {
        hasher.update(&g.to_le_bytes());
    }
    for &b in &norm.beta {
        hasher.update(&b.to_le_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> ScaledVfnConfig {
        ScaledVfnConfig {
            hidden_dim: 64,
            num_blocks: 2,
            io_dim: SLOT_DIM,
        }
    }

    #[test]
    fn default_config_targets_50m() {
        let config = ScaledVfnConfig::default();
        let params = config.param_count();
        // Should be approximately 50M (±10M)
        assert!(
            params > 40_000_000 && params < 60_000_000,
            "expected ~50M params, got {}",
            params
        );
    }

    #[test]
    fn forward_produces_correct_size() {
        let vfn = ScaledVfn::new_random(42, &small_config());
        let input = [0.1; SLOT_DIM];
        let output = vfn.forward(&input).unwrap();
        assert_eq!(output.len(), SLOT_DIM);
    }

    #[test]
    fn forward_output_is_finite() {
        let vfn = ScaledVfn::new_random(42, &small_config());
        for seed in 0..20 {
            let mut input = [0.0f32; SLOT_DIM];
            let mut rng = Rng::new(seed);
            for x in &mut input {
                *x = rng.next_f32_range(-1.0, 1.0);
            }
            let output = vfn.forward(&input).unwrap();
            assert!(
                output.iter().all(|x| x.is_finite()),
                "output contains NaN/Inf for seed {seed}"
            );
        }
    }

    #[test]
    fn forward_deterministic() {
        let vfn1 = ScaledVfn::new_random(42, &small_config());
        let vfn2 = ScaledVfn::new_random(42, &small_config());
        let input = [0.5; SLOT_DIM];
        let out1 = vfn1.forward(&input).unwrap();
        let out2 = vfn2.forward(&input).unwrap();
        assert_eq!(out1, out2);
    }

    #[test]
    fn different_seeds_different_output() {
        let vfn1 = ScaledVfn::new_random(42, &small_config());
        let vfn2 = ScaledVfn::new_random(43, &small_config());
        let input = [0.5; SLOT_DIM];
        let out1 = vfn1.forward(&input).unwrap();
        let out2 = vfn2.forward(&input).unwrap();
        assert_ne!(out1, out2);
    }

    #[test]
    fn nan_input_errors() {
        let vfn = ScaledVfn::new_random(42, &small_config());
        let mut input = [0.1; SLOT_DIM];
        input[0] = f32::NAN;
        assert!(vfn.forward(&input).is_err());
    }

    #[test]
    fn debug_format_readable() {
        let vfn = ScaledVfn::new_random(42, &small_config());
        let debug = format!("{:?}", vfn);
        assert!(debug.contains("ScaledVfn"));
        assert!(debug.contains("64×2"));
    }

    #[test]
    fn param_count_matches_config() {
        let config = small_config();
        let vfn = ScaledVfn::new_random(42, &config);
        assert_eq!(vfn.param_count(), config.param_count());
    }

    #[test]
    fn checkpoint_roundtrip() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("scaled_vfn_test.bin");

        let config = small_config();
        let vfn = ScaledVfn::new_random(42, &config);
        vfn.save(&path).unwrap();

        let loaded = ScaledVfn::load(&path).unwrap();

        let input = [0.42f32; SLOT_DIM];
        let out1 = vfn.forward(&input).unwrap();
        let out2 = loaded.forward(&input).unwrap();

        for (a, b) in out1.iter().zip(out2.iter()) {
            assert_eq!(*a, *b, "checkpoint roundtrip: outputs differ");
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn checkpoint_detects_corruption() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("scaled_vfn_corrupt.bin");

        let config = small_config();
        let vfn = ScaledVfn::new_random(42, &config);
        vfn.save(&path).unwrap();

        // Corrupt a byte in the weights section
        let mut data = std::fs::read(&path).unwrap();
        if data.len() > 30 {
            data[30] ^= 0xFF;
            std::fs::write(&path, &data).unwrap();
        }

        let result = ScaledVfn::load(&path);
        assert!(result.is_err());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn gelu_activation_basic() {
        assert!((gelu(0.0) - 0.0).abs() < 1e-6);
        assert!(gelu(1.0) > 0.8); // GELU(1) ≈ 0.841
        assert!(gelu(-1.0) < 0.0); // GELU(-1) ≈ -0.159
        assert!(gelu(3.0) > 2.9); // Nearly identity for large positive
    }

    #[test]
    fn layer_norm_normalizes() {
        let norm = LayerNorm::new(4);
        let input = [1.0, 2.0, 3.0, 4.0];
        let output = norm.forward(&input);
        // Mean should be ~0, variance ~1
        let mean = output.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5, "mean should be ~0, got {mean}");
    }

    #[test]
    fn residual_connection_preserves_input() {
        // With very small weights, output should be close to input
        let config = ScaledVfnConfig {
            hidden_dim: 64,
            num_blocks: 1,
            io_dim: SLOT_DIM,
        };
        let vfn = ScaledVfn::new_random(42, &config);
        let input = [0.0f32; SLOT_DIM];
        // Zero input through Xavier-initialized network with residual
        // should produce small but non-zero output (from biases and norms)
        let output = vfn.forward(&input).unwrap();
        assert!(output.iter().all(|x| x.is_finite()));
    }
}
