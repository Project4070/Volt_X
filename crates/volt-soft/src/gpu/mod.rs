//! GPU-accelerated RAR inference using candle.
//!
//! Provides GPU-parallel implementations of the VFN, cross-slot attention,
//! and RAR loop. All code is feature-gated behind `gpu`.
//!
//! The GPU path mirrors the CPU path in [`crate::rar`] but uses batched
//! tensor operations for ~10x speedup on CUDA-capable hardware.
//!
//! ## Usage
//!
//! ```bash
//! cargo test -p volt-soft --features gpu
//! ```

pub mod attention;
pub mod rar;
pub mod vfn;

use candle_core::Device;
use volt_core::VoltError;

/// Selects the best available compute device.
///
/// Tries CUDA first (device 0), falls back to CPU if unavailable.
///
/// # Example
///
/// ```ignore
/// use volt_soft::gpu::best_device;
///
/// let device = best_device().unwrap();
/// println!("Using device: {:?}", device);
/// ```
pub fn best_device() -> Result<Device, VoltError> {
    match Device::cuda_if_available(0) {
        Ok(dev) => Ok(dev),
        Err(e) => Err(VoltError::Internal {
            message: format!("failed to select compute device: {e}"),
        }),
    }
}

/// Returns true if CUDA is available on this system.
///
/// # Example
///
/// ```ignore
/// use volt_soft::gpu::cuda_available;
///
/// if cuda_available() {
///     println!("CUDA acceleration enabled");
/// }
/// ```
pub fn cuda_available() -> bool {
    candle_core::utils::cuda_is_available()
}
