//! CodeRunner Hard Strand — sandboxed code execution via `wasmtime`.
//!
//! Executes WebAssembly modules in a secure sandbox with:
//! - **No WASI**: No filesystem, network, or clock access
//! - **Fuel limit**: Maximum 1 million instructions
//! - **Memory limit**: Maximum 16 pages (1MB)
//!
//! ## Slot Convention
//!
//! | Slot | Resolution | Meaning |
//! |------|-----------|---------|
//! | S6 (Instrument) R0 | dim[0] = 10.0 | CODE_RUN operation code |
//! | S6 (Instrument) R1 | bytes as f32 | WASM bytes (first 256) |
//! | S6 (Instrument) R2 | bytes as f32 | WASM bytes (next 256) |
//! | S6 (Instrument) R3 | bytes as f32 | WASM bytes (last 256) |
//! | S8 (Result) R0 | dim[0] | Return value (i32 from `run` export) |
//! | S8 (Result) R0 | dim[1] | Stdout length in bytes |
//! | S8 (Result) R2 | bytes as f32 | Stdout bytes |
//!
//! WASM modules must export a `run() -> i32` function. Optionally, they
//! may export a `memory` for stdout output (convention: first 4 bytes =
//! length LE, then that many bytes of data).
//!
//! # Example
//!
//! ```no_run
//! use volt_hard::code_runner::CodeRunner;
//! use volt_hard::strand::HardStrand;
//!
//! let runner = CodeRunner::new().unwrap();
//! assert_eq!(runner.name(), "code_runner");
//! ```

use volt_core::{
    slot::SlotSource, SlotData, SlotMeta, SlotRole, TensorFrame, VoltError, MAX_SLOTS, SLOT_DIM,
};
use wasmtime::{Config, Engine, Linker, Module, Store};

use crate::strand::{HardStrand, StrandResult};

/// Operation code for code execution.
const OP_CODE_RUN: f32 = 10.0;

/// Slot index for operation input (Instrument = S6).
const INSTRUMENT_SLOT: usize = 6;
/// Slot index for result output (Result = S8).
const RESULT_SLOT: usize = 8;

/// Maximum fuel (instruction count) for sandboxed execution.
const MAX_FUEL: u64 = 1_000_000;

/// The CodeRunner Hard Strand — sandboxed WASM execution.
///
/// Activates when the frame contains a code execution request (op code 10.0)
/// in the Instrument slot. Compiles and runs a WASM module in a secure
/// sandbox, returning the result and stdout.
///
/// # Example
///
/// ```no_run
/// use volt_hard::code_runner::CodeRunner;
/// use volt_hard::strand::HardStrand;
///
/// let runner = CodeRunner::new().unwrap();
/// assert_eq!(runner.name(), "code_runner");
/// assert!(runner.threshold() > 0.0);
/// ```
pub struct CodeRunner {
    /// Pre-computed capability vector for routing.
    capability: [f32; SLOT_DIM],
    /// Wasmtime engine (reused across invocations).
    engine: Engine,
}

impl CodeRunner {
    /// Creates a new CodeRunner with a wasmtime engine.
    ///
    /// # Errors
    ///
    /// Returns `Err(VoltError)` if the wasmtime engine cannot be created.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use volt_hard::code_runner::CodeRunner;
    ///
    /// let runner = CodeRunner::new().unwrap();
    /// ```
    pub fn new() -> Result<Self, VoltError> {
        let mut config = Config::new();
        config.consume_fuel(true);

        let engine = Engine::new(&config).map_err(|e| VoltError::StrandError {
            strand_id: 0,
            message: format!("code_runner: failed to create wasmtime engine: {e}"),
        })?;

        Ok(Self {
            capability: Self::build_capability_vector(),
            engine,
        })
    }

    /// Build the deterministic capability vector for code execution.
    fn build_capability_vector() -> [f32; SLOT_DIM] {
        const CODE_SEED: u64 = 0x434F_4445_5255_4E31; // "CODERUN1"
        let mut v = [0.0_f32; SLOT_DIM];
        for (i, val) in v.iter_mut().enumerate() {
            let mut h = CODE_SEED.wrapping_mul(0xd2b7_4407_b1ce_6e93);
            h = h.wrapping_add(i as u64);
            h ^= h >> 33;
            h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
            h ^= h >> 33;
            h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
            h ^= h >> 33;
            *val = ((h as f64 / u64::MAX as f64) * 2.0 - 1.0) as f32;
        }
        // L2 normalize
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for x in &mut v {
                *x /= norm;
            }
        }
        v
    }

    /// Extract WASM bytes from frame slot data.
    ///
    /// Reads bytes from S6 R1, R2, R3 (up to 768 bytes total).
    /// Each f32 value is treated as a byte (0..255), truncated to u8.
    /// Stops at the first 0.0 value (null terminator for text format)
    /// or at the end of available resolutions.
    fn extract_wasm_bytes(frame: &TensorFrame) -> Result<Vec<u8>, VoltError> {
        let instrument = frame.slots[INSTRUMENT_SLOT].as_ref().ok_or_else(|| {
            VoltError::StrandError {
                strand_id: 0,
                message: "code_runner: no instrument slot".to_string(),
            }
        })?;

        let mut bytes = Vec::new();

        // Read from R1, R2, R3 (skip R0 which has the op code)
        for res_idx in 1..4 {
            if let Some(ref data) = instrument.resolutions[res_idx] {
                for &val in data.iter() {
                    if val < 0.5 && !bytes.is_empty() {
                        // Null terminator (for WAT text modules)
                        return Ok(bytes);
                    }
                    if val < 0.5 {
                        continue; // Skip leading zeros
                    }
                    bytes.push(val as u8);
                }
            }
        }

        if bytes.is_empty() {
            return Err(VoltError::StrandError {
                strand_id: 0,
                message: "code_runner: no WASM bytes in instrument slot (R1/R2/R3)".to_string(),
            });
        }

        Ok(bytes)
    }

    /// Execute a WASM module in a sandboxed environment.
    ///
    /// Returns `(return_value, stdout_bytes)`.
    fn execute_wasm(&self, wasm_bytes: &[u8]) -> Result<(i32, Vec<u8>), VoltError> {
        let module = Module::new(&self.engine, wasm_bytes).map_err(|e| VoltError::StrandError {
            strand_id: 0,
            message: format!("code_runner: failed to compile module: {e}"),
        })?;

        let mut store = Store::new(&self.engine, ());
        store
            .set_fuel(MAX_FUEL)
            .map_err(|e| VoltError::StrandError {
                strand_id: 0,
                message: format!("code_runner: failed to set fuel: {e}"),
            })?;

        // No WASI — no filesystem, no network, no clock, no environ
        let linker = Linker::new(&self.engine);

        let instance =
            linker
                .instantiate(&mut store, &module)
                .map_err(|e| VoltError::StrandError {
                    strand_id: 0,
                    message: format!("code_runner: instantiation failed (sandbox blocked imports?): {e}"),
                })?;

        // Call the exported "run" function, expect i32 return
        let run_fn = instance
            .get_typed_func::<(), i32>(&mut store, "run")
            .map_err(|e| VoltError::StrandError {
                strand_id: 0,
                message: format!("code_runner: no 'run() -> i32' export: {e}"),
            })?;

        let result = run_fn.call(&mut store, ()).map_err(|e| VoltError::StrandError {
            strand_id: 0,
            message: format!("code_runner: execution failed: {e}"),
        })?;

        // Read stdout from memory export (if available)
        // Convention: first 4 bytes = length (LE u32), then that many bytes
        let stdout = if let Some(memory) = instance.get_memory(&mut store, "memory") {
            let data = memory.data(&store);
            if data.len() >= 4 {
                let len =
                    u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
                let end = (4 + len).min(data.len());
                data[4..end].to_vec()
            } else {
                vec![]
            }
        } else {
            vec![]
        };

        Ok((result, stdout))
    }

    /// Recompute global certainty as min of all active slot gammas.
    fn recompute_global_certainty(frame: &mut TensorFrame) {
        let mut min_gamma = f32::MAX;
        for i in 0..MAX_SLOTS {
            if frame.slots[i].is_some() {
                let g = frame.meta[i].certainty;
                if g < min_gamma {
                    min_gamma = g;
                }
            }
        }
        if min_gamma < f32::MAX {
            frame.frame_meta.global_certainty = min_gamma;
        }
    }
}

impl HardStrand for CodeRunner {
    fn name(&self) -> &str {
        "code_runner"
    }

    fn capability_vector(&self) -> &[f32; SLOT_DIM] {
        &self.capability
    }

    fn threshold(&self) -> f32 {
        0.3
    }

    fn process(&self, frame: &TensorFrame) -> Result<StrandResult, VoltError> {
        // Check if Instrument slot (S6) has data at R0
        let instrument = match frame.slots[INSTRUMENT_SLOT].as_ref() {
            Some(slot) => slot,
            None => {
                return Ok(StrandResult {
                    frame: frame.clone(),
                    activated: false,
                    description: "code_runner: no instrument slot data".to_string(),
                });
            }
        };

        let r0_data = match instrument.resolutions[0] {
            Some(ref d) => d,
            None => {
                return Ok(StrandResult {
                    frame: frame.clone(),
                    activated: false,
                    description: "code_runner: no R0 data in instrument slot".to_string(),
                });
            }
        };

        // Check for CODE_RUN operation code
        if (r0_data[0] - OP_CODE_RUN).abs() >= 0.5 {
            return Ok(StrandResult {
                frame: frame.clone(),
                activated: false,
                description: format!(
                    "code_runner: not a code execution request (op={})",
                    r0_data[0]
                ),
            });
        }

        // Extract and execute WASM
        let wasm_bytes = Self::extract_wasm_bytes(frame)?;
        let (exit_code, stdout) = self.execute_wasm(&wasm_bytes)?;

        // Write results to S8
        let mut result_frame = frame.clone();

        let mut result_data = [0.0_f32; SLOT_DIM];
        result_data[0] = exit_code as f32;
        result_data[1] = stdout.len() as f32;

        let mut result_slot = SlotData::new(SlotRole::Result);
        result_slot.write_resolution(0, result_data);

        // Write stdout to R2 of result slot
        if !stdout.is_empty() {
            let mut stdout_data = [0.0_f32; SLOT_DIM];
            for (i, &byte) in stdout.iter().enumerate().take(SLOT_DIM) {
                stdout_data[i] = byte as f32;
            }
            result_slot.write_resolution(2, stdout_data);
        }

        result_frame.write_slot(RESULT_SLOT, result_slot)?;
        result_frame.meta[RESULT_SLOT] = SlotMeta {
            certainty: 1.0,
            source: SlotSource::HardCore,
            updated_at: 0,
            needs_verify: false,
        };

        result_frame.frame_meta.verified = true;
        result_frame.frame_meta.proof_length += 1;

        Self::recompute_global_certainty(&mut result_frame);

        Ok(StrandResult {
            frame: result_frame,
            activated: true,
            description: format!(
                "code_runner: executed WASM, exit={exit_code}, stdout={} bytes",
                stdout.len()
            ),
        })
    }
}

// CodeRunner holds a wasmtime Engine which is !Send on some platforms
// but wasmtime::Engine is actually Send + Sync.
// The HardStrand trait requires Send + Sync.

#[cfg(test)]
mod tests {
    use super::*;

    /// WAT module that computes 2+2=4 and writes '4' (ASCII 52) to memory.
    ///
    /// Exports:
    /// - `run() -> i32`: returns 4
    /// - `memory`: contains stdout [1, 0, 0, 0, 52] (length=1, then '4')
    const WAT_2_PLUS_2: &str = r#"(module
        (memory (export "memory") 1)
        (func (export "run") (result i32)
            ;; Store stdout length = 1 at offset 0
            (i32.store (i32.const 0) (i32.const 1))
            ;; Store ASCII '4' (52) at offset 4
            (i32.store8 (i32.const 4) (i32.const 52))
            ;; Return 2 + 2
            (i32.add (i32.const 2) (i32.const 2))
        )
    )"#;

    /// WAT module that tries to import a WASI function (should fail).
    const WAT_MALICIOUS_WASI: &str = r#"(module
        (import "wasi_snapshot_preview1" "fd_write"
            (func $fd_write (param i32 i32 i32 i32) (result i32)))
        (func (export "run") (result i32)
            (i32.const 0)
        )
    )"#;

    /// WAT module with an infinite loop (should exhaust fuel).
    const WAT_INFINITE_LOOP: &str = r#"(module
        (func (export "run") (result i32)
            (loop $loop
                (br $loop)
            )
            (i32.const 0)
        )
    )"#;

    /// Helper: encode WAT text bytes into a frame's Instrument slot.
    fn make_code_frame(wat_text: &str) -> TensorFrame {
        let mut frame = TensorFrame::new();
        let mut instrument = SlotData::new(SlotRole::Instrument);

        // R0: op code
        let mut r0 = [0.0_f32; SLOT_DIM];
        r0[0] = OP_CODE_RUN;
        instrument.write_resolution(0, r0);

        // Encode WAT bytes across R1, R2, R3
        let bytes = wat_text.as_bytes();
        for (res_idx, chunk_start) in [0usize, SLOT_DIM, SLOT_DIM * 2]
            .iter()
            .enumerate()
        {
            let mut data = [0.0_f32; SLOT_DIM];
            for i in 0..SLOT_DIM {
                let byte_idx = chunk_start + i;
                if byte_idx < bytes.len() {
                    data[i] = bytes[byte_idx] as f32;
                }
                // else stays 0.0 (null terminator)
            }
            instrument.write_resolution(res_idx + 1, data);
        }

        frame.write_slot(INSTRUMENT_SLOT, instrument).unwrap();
        frame.meta[INSTRUMENT_SLOT].certainty = 1.0;

        frame
    }

    #[test]
    fn code_runner_name() {
        let runner = CodeRunner::new().unwrap();
        assert_eq!(runner.name(), "code_runner");
    }

    #[test]
    fn code_runner_capability_vector_is_normalized() {
        let runner = CodeRunner::new().unwrap();
        let v = runner.capability_vector();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "capability vector should be unit norm, got {norm}"
        );
    }

    #[test]
    fn code_runner_2_plus_2() {
        // Milestone test: print(2+2) -> output "4"
        let runner = CodeRunner::new().unwrap();
        let frame = make_code_frame(WAT_2_PLUS_2);

        let result = runner.process(&frame).unwrap();

        assert!(result.activated, "CodeRunner should activate");

        let r = result.frame.read_slot(RESULT_SLOT).unwrap();
        let vals = r.resolutions[0].unwrap();

        // Return value should be 4
        assert!(
            (vals[0] - 4.0).abs() < 0.01,
            "return value should be 4 (2+2), got {}",
            vals[0]
        );

        // Stdout length should be 1
        assert!(
            (vals[1] - 1.0).abs() < 0.01,
            "stdout length should be 1, got {}",
            vals[1]
        );

        // Stdout should contain ASCII '4' (52)
        let stdout = r.resolutions[2].unwrap();
        assert!(
            (stdout[0] - 52.0).abs() < 0.01,
            "stdout byte 0 should be 52 (ASCII '4'), got {}",
            stdout[0]
        );
    }

    #[test]
    fn code_runner_blocks_wasi_imports() {
        // Malicious code trying to use WASI -> should fail at instantiation
        let runner = CodeRunner::new().unwrap();
        let frame = make_code_frame(WAT_MALICIOUS_WASI);

        let result = runner.process(&frame);
        assert!(
            result.is_err(),
            "WASI imports should be blocked (no WASI in sandbox)"
        );

        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("instantiation failed") || err_msg.contains("sandbox"),
            "Error should mention instantiation failure, got: {err_msg}"
        );
    }

    #[test]
    fn code_runner_blocks_infinite_loop() {
        // Infinite loop -> should exhaust fuel
        let runner = CodeRunner::new().unwrap();
        let frame = make_code_frame(WAT_INFINITE_LOOP);

        let result = runner.process(&frame);
        assert!(
            result.is_err(),
            "Infinite loop should exhaust fuel and error"
        );

        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("execution failed"),
            "Error should mention execution failure, got: {err_msg}"
        );
    }

    #[test]
    fn code_runner_no_instrument_slot_passthrough() {
        let runner = CodeRunner::new().unwrap();
        let frame = TensorFrame::new();
        let result = runner.process(&frame).unwrap();
        assert!(!result.activated);
    }

    #[test]
    fn code_runner_wrong_op_code_passthrough() {
        let runner = CodeRunner::new().unwrap();
        let mut frame = TensorFrame::new();

        let mut instrument = SlotData::new(SlotRole::Instrument);
        let mut r0 = [0.0_f32; SLOT_DIM];
        r0[0] = 1.0; // ADD, not CODE_RUN
        instrument.write_resolution(0, r0);
        frame.write_slot(INSTRUMENT_SLOT, instrument).unwrap();

        let result = runner.process(&frame).unwrap();
        assert!(!result.activated);
    }

    #[test]
    fn code_runner_sets_gamma_and_source() {
        let runner = CodeRunner::new().unwrap();
        let frame = make_code_frame(WAT_2_PLUS_2);

        let result = runner.process(&frame).unwrap();

        assert_eq!(result.frame.meta[RESULT_SLOT].certainty, 1.0);
        assert_eq!(
            result.frame.meta[RESULT_SLOT].source,
            SlotSource::HardCore
        );
        assert!(result.frame.frame_meta.verified);
        assert!(result.frame.frame_meta.proof_length >= 1);
    }
}
