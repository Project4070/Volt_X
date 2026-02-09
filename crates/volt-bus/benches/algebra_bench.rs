//! Performance benchmarks for LLL Algebra operations.
//!
//! Critical requirement: bind on 256 dims must be < 10µs (PHASE-1.md line 66).

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use volt_bus::{bind, unbind, superpose, permute, similarity};
use volt_bus::{bind_frames, unbind_frames, similarity_frames};
use volt_core::{TensorFrame, SlotRole, SLOT_DIM};

/// Create a normalized test vector from seed (deterministic)
fn make_test_vec(seed: u64) -> [f32; SLOT_DIM] {
    let mut v = [0.0; SLOT_DIM];
    for i in 0..SLOT_DIM {
        v[i] = ((seed + i as u64) as f32 * 0.1).sin();
    }
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    for x in &mut v {
        *x /= norm;
    }
    v
}

fn bench_bind(c: &mut Criterion) {
    let a = make_test_vec(42);
    let b = make_test_vec(99);

    c.bench_function("bind (256 dims)", |bencher| {
        bencher.iter(|| {
            bind(black_box(&a), black_box(&b)).unwrap()
        });
    });
}

fn bench_unbind(c: &mut Criterion) {
    let a = make_test_vec(42);
    let b = make_test_vec(99);
    let bound = bind(&a, &b).unwrap();

    c.bench_function("unbind (256 dims)", |bencher| {
        bencher.iter(|| {
            unbind(black_box(&bound), black_box(&a)).unwrap()
        });
    });
}

fn bench_superpose(c: &mut Criterion) {
    let vecs: Vec<[f32; SLOT_DIM]> = (0..10).map(|i| make_test_vec(i)).collect();
    let refs: Vec<&[f32; SLOT_DIM]> = vecs.iter().collect();

    c.bench_function("superpose (10 vectors)", |bencher| {
        bencher.iter(|| {
            superpose(black_box(&refs)).unwrap()
        });
    });
}

fn bench_permute(c: &mut Criterion) {
    let a = make_test_vec(42);

    c.bench_function("permute", |bencher| {
        bencher.iter(|| {
            permute(black_box(&a), black_box(5))
        });
    });
}

fn bench_similarity(c: &mut Criterion) {
    let a = make_test_vec(42);
    let b = make_test_vec(99);

    c.bench_function("similarity", |bencher| {
        bencher.iter(|| {
            similarity(black_box(&a), black_box(&b))
        });
    });
}

fn bench_bind_frames(c: &mut Criterion) {
    let mut frame_a = TensorFrame::new();
    frame_a.write_at(0, 0, SlotRole::Agent, make_test_vec(1)).unwrap();
    frame_a.write_at(1, 0, SlotRole::Predicate, make_test_vec(2)).unwrap();
    frame_a.write_at(2, 0, SlotRole::Patient, make_test_vec(3)).unwrap();

    let mut frame_b = TensorFrame::new();
    frame_b.write_at(0, 0, SlotRole::Agent, make_test_vec(4)).unwrap();
    frame_b.write_at(1, 0, SlotRole::Predicate, make_test_vec(5)).unwrap();
    frame_b.write_at(2, 0, SlotRole::Patient, make_test_vec(6)).unwrap();

    c.bench_function("bind_frames (3 slots)", |bencher| {
        bencher.iter(|| {
            bind_frames(black_box(&frame_a), black_box(&frame_b)).unwrap()
        });
    });
}

fn bench_unbind_frames(c: &mut Criterion) {
    let mut frame_a = TensorFrame::new();
    frame_a.write_at(0, 0, SlotRole::Agent, make_test_vec(1)).unwrap();
    frame_a.write_at(1, 0, SlotRole::Predicate, make_test_vec(2)).unwrap();
    frame_a.write_at(2, 0, SlotRole::Patient, make_test_vec(3)).unwrap();

    let mut frame_b = TensorFrame::new();
    frame_b.write_at(0, 0, SlotRole::Agent, make_test_vec(4)).unwrap();
    frame_b.write_at(1, 0, SlotRole::Predicate, make_test_vec(5)).unwrap();
    frame_b.write_at(2, 0, SlotRole::Patient, make_test_vec(6)).unwrap();

    let bound = bind_frames(&frame_a, &frame_b).unwrap();

    c.bench_function("unbind_frames (3 slots)", |bencher| {
        bencher.iter(|| {
            unbind_frames(black_box(&bound), black_box(&frame_a)).unwrap()
        });
    });
}

fn bench_similarity_frames(c: &mut Criterion) {
    let mut frame_a = TensorFrame::new();
    frame_a.write_at(0, 0, SlotRole::Agent, make_test_vec(1)).unwrap();
    frame_a.write_at(1, 0, SlotRole::Predicate, make_test_vec(2)).unwrap();
    frame_a.write_at(2, 0, SlotRole::Patient, make_test_vec(3)).unwrap();

    let mut frame_b = TensorFrame::new();
    frame_b.write_at(0, 0, SlotRole::Agent, make_test_vec(4)).unwrap();
    frame_b.write_at(1, 0, SlotRole::Predicate, make_test_vec(5)).unwrap();
    frame_b.write_at(2, 0, SlotRole::Patient, make_test_vec(6)).unwrap();

    c.bench_function("similarity_frames (3 slots)", |bencher| {
        bencher.iter(|| {
            similarity_frames(black_box(&frame_a), black_box(&frame_b))
        });
    });
}

criterion_group!(
    benches,
    bench_bind,
    bench_unbind,
    bench_superpose,
    bench_permute,
    bench_similarity,
    bench_bind_frames,
    bench_unbind_frames,
    bench_similarity_frames,
);
criterion_main!(benches);
