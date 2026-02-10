//! Integration tests for the GPU-accelerated RAR loop (Milestone 2.4).
//!
//! All tests use `candle_core::Device::Cpu` so they run without CUDA hardware.
//! Feature-gated behind `gpu`.

#![cfg(feature = "gpu")]

use volt_core::{SlotRole, TensorFrame, SLOT_DIM, MAX_SLOTS};
use volt_soft::gpu::attention::GpuSlotAttention;
use volt_soft::gpu::rar::gpu_rar_loop;
use volt_soft::gpu::vfn::GpuVfn;
use volt_soft::rar::RarConfig;

fn normalized_vector(seed: u64) -> [f32; SLOT_DIM] {
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

fn make_frame(n: usize) -> TensorFrame {
    let roles = [
        SlotRole::Agent,
        SlotRole::Predicate,
        SlotRole::Patient,
        SlotRole::Location,
        SlotRole::Time,
        SlotRole::Manner,
        SlotRole::Instrument,
        SlotRole::Cause,
        SlotRole::Result,
        SlotRole::Free(0),
        SlotRole::Free(1),
        SlotRole::Free(2),
        SlotRole::Free(3),
        SlotRole::Free(4),
        SlotRole::Free(5),
        SlotRole::Free(6),
    ];
    let mut frame = TensorFrame::new();
    for i in 0..n.min(MAX_SLOTS) {
        frame
            .write_at(i, 0, roles[i], normalized_vector(i as u64 + 1000))
            .unwrap();
    }
    frame
}

/// GPU RAR matches CPU RAR within float32 precision.
#[test]
fn gpu_cpu_equivalence() {
    let cpu_vfn = volt_soft::vfn::Vfn::new_random(42);
    let cpu_attn = volt_soft::attention::SlotAttention::new_random(43);

    let device = candle_core::Device::Cpu;
    let gpu_vfn = GpuVfn::from_cpu_vfn(&cpu_vfn, &device).unwrap();
    let gpu_attn = GpuSlotAttention::from_cpu_attention(&cpu_attn, &device).unwrap();

    let config = RarConfig {
        epsilon: 0.01,
        max_iterations: 5,
        dt: 0.1,
        beta: 0.5,
        ..RarConfig::default()
    };

    let frame = make_frame(4);
    let cpu_result = volt_soft::rar::rar_loop(&frame, &cpu_vfn, &cpu_attn, &config).unwrap();
    let gpu_result = gpu_rar_loop(&frame, &gpu_vfn, &gpu_attn, &config).unwrap();

    assert_eq!(cpu_result.iterations, gpu_result.iterations);

    for i in 0..4 {
        let cpu_vec = cpu_result.frame.read_slot(i).unwrap().resolutions[0].unwrap();
        let gpu_vec = gpu_result.frame.read_slot(i).unwrap().resolutions[0].unwrap();
        for d in 0..SLOT_DIM {
            assert!(
                (cpu_vec[d] - gpu_vec[d]).abs() < 1e-3,
                "slot {} dim {} diverged: cpu={}, gpu={}",
                i, d, cpu_vec[d], gpu_vec[d],
            );
        }
    }
}

/// GPU RAR converges on multi-slot input.
#[test]
fn gpu_rar_converges() {
    let device = candle_core::Device::Cpu;
    let vfn = GpuVfn::new_random(42, &device).unwrap();
    let attn = GpuSlotAttention::new_random(43, &device).unwrap();

    let config = RarConfig {
        epsilon: 0.01,
        max_iterations: 50,
        ..RarConfig::default()
    };

    let frame = make_frame(4);
    let result = gpu_rar_loop(&frame, &vfn, &attn, &config).unwrap();

    assert!(result.iterations <= 50);
    assert_eq!(result.frame.active_slot_count(), 4);

    // All output should be unit-normalized
    for i in 0..4 {
        let vec = result.frame.read_slot(i).unwrap().resolutions[0].as_ref().unwrap();
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "slot {} should be normalized, got norm={}",
            i, norm,
        );
    }
}

/// GPU RAR respects iteration budget.
#[test]
fn gpu_rar_budget_enforcement() {
    let device = candle_core::Device::Cpu;
    let vfn = GpuVfn::new_random(42, &device).unwrap();
    let attn = GpuSlotAttention::new_random(43, &device).unwrap();

    let config = RarConfig {
        epsilon: 1e-10,
        max_iterations: 7,
        ..RarConfig::default()
    };

    let frame = make_frame(8);
    let result = gpu_rar_loop(&frame, &vfn, &attn, &config).unwrap();

    assert!(
        result.iterations <= 7,
        "should respect budget, got {} iterations",
        result.iterations,
    );
}

/// GPU RAR preserves inactive slots.
#[test]
fn gpu_rar_preserves_inactive_slots() {
    let device = candle_core::Device::Cpu;
    let vfn = GpuVfn::new_random(42, &device).unwrap();
    let attn = GpuSlotAttention::new_random(43, &device).unwrap();

    let config = RarConfig {
        max_iterations: 5,
        ..RarConfig::default()
    };

    let mut frame = TensorFrame::new();
    frame
        .write_at(0, 0, SlotRole::Agent, normalized_vector(42))
        .unwrap();

    let result = gpu_rar_loop(&frame, &vfn, &attn, &config).unwrap();

    for i in 1..MAX_SLOTS {
        assert!(
            result.frame.slots[i].is_none(),
            "slot {} should remain empty",
            i,
        );
    }
}

/// GPU RAR empty frame converges immediately.
#[test]
fn gpu_rar_empty_frame() {
    let device = candle_core::Device::Cpu;
    let vfn = GpuVfn::new_random(42, &device).unwrap();
    let attn = GpuSlotAttention::new_random(43, &device).unwrap();
    let config = RarConfig::default();
    let frame = TensorFrame::new();

    let result = gpu_rar_loop(&frame, &vfn, &attn, &config).unwrap();
    assert_eq!(result.iterations, 0);
    assert!(result.converged.iter().all(|&c| c));
}

/// GPU RAR with all 16 slots active.
#[test]
fn gpu_rar_full_slots() {
    let device = candle_core::Device::Cpu;
    let vfn = GpuVfn::new_random(42, &device).unwrap();
    let attn = GpuSlotAttention::new_random(43, &device).unwrap();

    let config = RarConfig {
        epsilon: 1e-10,
        max_iterations: 10,
        ..RarConfig::default()
    };

    let frame = make_frame(16);
    let result = gpu_rar_loop(&frame, &vfn, &attn, &config).unwrap();

    assert_eq!(result.iterations, 10);
    assert_eq!(result.frame.active_slot_count(), 16);
}
