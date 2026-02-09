use criterion::{black_box, criterion_group, criterion_main, Criterion};
use volt_core::{SlotData, SlotRole, TensorFrame, SLOT_DIM};

fn bench_frame_creation(c: &mut Criterion) {
    c.bench_function("TensorFrame::new", |b| {
        b.iter(|| black_box(TensorFrame::new()))
    });
}

fn bench_slot_write(c: &mut Criterion) {
    c.bench_function("write_slot", |b| {
        let mut frame = TensorFrame::new();
        let slot = SlotData::new(SlotRole::Agent);
        b.iter(|| {
            frame.write_slot(0, black_box(slot.clone())).unwrap();
        })
    });
}

fn bench_write_at(c: &mut Criterion) {
    c.bench_function("write_at", |b| {
        let mut frame = TensorFrame::new();
        let data = [0.42_f32; SLOT_DIM];
        b.iter(|| {
            frame
                .write_at(0, 0, SlotRole::Agent, black_box(data))
                .unwrap();
        })
    });
}

fn bench_active_slot_count(c: &mut Criterion) {
    c.bench_function("active_slot_count (4 slots)", |b| {
        let mut frame = TensorFrame::new();
        for i in 0..4 {
            frame
                .write_slot(i, SlotData::new(SlotRole::Free(i as u8)))
                .unwrap();
        }
        b.iter(|| black_box(frame.active_slot_count()))
    });
}

fn bench_data_size_bytes(c: &mut Criterion) {
    c.bench_function("data_size_bytes (4 slots, 2 res each)", |b| {
        let mut frame = TensorFrame::new();
        for i in 0..4 {
            let mut slot = SlotData::new(SlotRole::Free(i as u8));
            slot.write_resolution(0, [1.0; SLOT_DIM]);
            slot.write_resolution(1, [1.0; SLOT_DIM]);
            frame.write_slot(i, slot).unwrap();
        }
        b.iter(|| black_box(frame.data_size_bytes()))
    });
}

fn bench_merge_no_overlap(c: &mut Criterion) {
    c.bench_function("merge (no overlap, 4+4 slots)", |b| {
        let mut frame1 = TensorFrame::new();
        let mut frame2 = TensorFrame::new();

        for i in 0..4 {
            frame1
                .write_slot(i, SlotData::new(SlotRole::Free(i as u8)))
                .unwrap();
            frame2
                .write_slot(i + 4, SlotData::new(SlotRole::Free((i + 4) as u8)))
                .unwrap();
        }

        b.iter(|| black_box(frame1.clone().merge(frame2.clone())))
    });
}

fn bench_merge_full_overlap(c: &mut Criterion) {
    c.bench_function("merge (full overlap, 4 conflicts)", |b| {
        let mut frame1 = TensorFrame::new();
        let mut frame2 = TensorFrame::new();

        for i in 0..4 {
            frame1
                .write_slot(i, SlotData::new(SlotRole::Free(i as u8)))
                .unwrap();
            frame2
                .write_slot(i, SlotData::new(SlotRole::Free(i as u8)))
                .unwrap();
        }

        b.iter(|| black_box(frame1.clone().merge(frame2.clone())))
    });
}

fn bench_normalize_slot(c: &mut Criterion) {
    c.bench_function("normalize_slot", |b| {
        let mut frame = TensorFrame::new();
        frame
            .write_at(0, 0, SlotRole::Agent, [2.0; SLOT_DIM])
            .unwrap();

        b.iter(|| {
            let mut f = frame.clone();
            f.normalize_slot(0, 0).unwrap();
            black_box(f)
        })
    });
}

fn bench_normalize_all(c: &mut Criterion) {
    c.bench_function("normalize_all (4 slots, 2 res each)", |b| {
        let mut frame = TensorFrame::new();

        for i in 0..4 {
            let mut slot = SlotData::new(SlotRole::Free(i as u8));
            slot.write_resolution(0, [1.0; SLOT_DIM]);
            slot.write_resolution(1, [2.0; SLOT_DIM]);
            frame.write_slot(i, slot).unwrap();
        }

        b.iter(|| {
            let mut f = frame.clone();
            f.normalize_all().unwrap();
            black_box(f)
        })
    });
}

criterion_group!(
    benches,
    bench_frame_creation,
    bench_slot_write,
    bench_write_at,
    bench_active_slot_count,
    bench_data_size_bytes,
    bench_merge_no_overlap,
    bench_merge_full_overlap,
    bench_normalize_slot,
    bench_normalize_all,
);
criterion_main!(benches);
