//! Criterion benchmarks for RAR inference loop.
//!
//! Measures CPU and (optionally) GPU RAR performance to verify the
//! Milestone 2.4 requirement: GPU should be > 10x faster than CPU.
//!
//! ```bash
//! cargo bench -p volt-soft                  # CPU only
//! cargo bench -p volt-soft --features gpu   # CPU + GPU
//! ```

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use volt_core::{SlotRole, TensorFrame, MAX_SLOTS, SLOT_DIM};
use volt_soft::attention::SlotAttention;
use volt_soft::rar::{rar_loop, RarConfig};
use volt_soft::vfn::Vfn;

/// Deterministic normalized vector from a seed (splitmix64-based).
fn normalized_vector(seed: u64) -> [f32; SLOT_DIM] {
    let mut v = [0.0f32; SLOT_DIM];
    let mut h = seed;
    for x in v.iter_mut() {
        h = h.wrapping_add(0x9e3779b97f4a7c15);
        h = (h ^ (h >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        h = (h ^ (h >> 27)).wrapping_mul(0x94d049bb133111eb);
        h ^= h >> 31;
        *x = ((h as f64 / u64::MAX as f64) * 2.0 - 1.0) as f32;
    }
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
    v
}

/// Build a frame with `n` active slots at resolution 0.
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

fn bench_cpu_rar_16_slots_50_iter(c: &mut Criterion) {
    let vfn = Vfn::new_random(42);
    let attn = SlotAttention::new_random(43);
    let config = RarConfig {
        epsilon: 1e-10, // tiny epsilon to prevent early convergence
        max_iterations: 50,
        dt: 0.1,
        beta: 0.5,
        ..RarConfig::default()
    };
    let frame = make_frame(16);

    c.bench_function("cpu_rar_16slots_50iter", |b| {
        b.iter(|| black_box(rar_loop(&frame, &vfn, &attn, &config).unwrap()));
    });
}

fn bench_cpu_rar_4_slots_50_iter(c: &mut Criterion) {
    let vfn = Vfn::new_random(42);
    let attn = SlotAttention::new_random(43);
    let config = RarConfig {
        epsilon: 1e-10,
        max_iterations: 50,
        dt: 0.1,
        beta: 0.5,
        ..RarConfig::default()
    };
    let frame = make_frame(4);

    c.bench_function("cpu_rar_4slots_50iter", |b| {
        b.iter(|| black_box(rar_loop(&frame, &vfn, &attn, &config).unwrap()));
    });
}

#[cfg(feature = "gpu")]
fn bench_gpu_rar_16_slots_50_iter(c: &mut Criterion) {
    use volt_soft::gpu::attention::GpuSlotAttention;
    use volt_soft::gpu::rar::gpu_rar_loop;
    use volt_soft::gpu::vfn::GpuVfn;

    let device = volt_soft::gpu::best_device().unwrap();
    let vfn = GpuVfn::new_random(42, &device).unwrap();
    let attn = GpuSlotAttention::new_random(43, &device).unwrap();
    let config = RarConfig {
        epsilon: 1e-10,
        max_iterations: 50,
        dt: 0.1,
        beta: 0.5,
        ..RarConfig::default()
    };
    let frame = make_frame(16);

    c.bench_function("gpu_rar_16slots_50iter", |b| {
        b.iter(|| black_box(gpu_rar_loop(&frame, &vfn, &attn, &config).unwrap()));
    });
}

#[cfg(feature = "gpu")]
fn bench_gpu_rar_4_slots_50_iter(c: &mut Criterion) {
    use volt_soft::gpu::attention::GpuSlotAttention;
    use volt_soft::gpu::rar::gpu_rar_loop;
    use volt_soft::gpu::vfn::GpuVfn;

    let device = volt_soft::gpu::best_device().unwrap();
    let vfn = GpuVfn::new_random(42, &device).unwrap();
    let attn = GpuSlotAttention::new_random(43, &device).unwrap();
    let config = RarConfig {
        epsilon: 1e-10,
        max_iterations: 50,
        dt: 0.1,
        beta: 0.5,
        ..RarConfig::default()
    };
    let frame = make_frame(4);

    c.bench_function("gpu_rar_4slots_50iter", |b| {
        b.iter(|| black_box(gpu_rar_loop(&frame, &vfn, &attn, &config).unwrap()));
    });
}

#[cfg(not(feature = "gpu"))]
criterion_group!(
    benches,
    bench_cpu_rar_16_slots_50_iter,
    bench_cpu_rar_4_slots_50_iter,
);

#[cfg(feature = "gpu")]
criterion_group!(
    benches,
    bench_cpu_rar_16_slots_50_iter,
    bench_cpu_rar_4_slots_50_iter,
    bench_gpu_rar_16_slots_50_iter,
    bench_gpu_rar_4_slots_50_iter,
);

criterion_main!(benches);
