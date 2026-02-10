//! Integration tests for the diffusion noise module.
//!
//! Tests noise generation, adaptive sigma, backward compatibility with
//! the RAR loop, and that diffusion-enabled RAR still converges.

use volt_core::{SlotRole, TensorFrame, SLOT_DIM, MAX_SLOTS};
use volt_soft::attention::SlotAttention;
use volt_soft::diffusion::{adapt_sigma, generate_noise, DiffusionConfig};
use volt_soft::rar::{rar_loop, RarConfig};
use volt_soft::vfn::Vfn;

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

fn make_frame(n: usize) -> TensorFrame {
    let roles = [
        SlotRole::Agent,
        SlotRole::Predicate,
        SlotRole::Patient,
        SlotRole::Location,
    ];
    let mut frame = TensorFrame::new();
    for i in 0..n.min(MAX_SLOTS) {
        frame
            .write_at(i, 0, roles[i % roles.len()], test_vector(i as u64 + 500))
            .unwrap();
    }
    frame
}

/// RAR with diffusion=None is backward compatible with M2.3.
#[test]
fn rar_without_diffusion_is_backward_compatible() {
    let vfn = Vfn::new_random(42);
    let attn = SlotAttention::new_random(43);
    let config = RarConfig {
        epsilon: 0.01,
        max_iterations: 10,
        ..RarConfig::default()
    };

    let frame = make_frame(4);
    let result = rar_loop(&frame, &vfn, &attn, &config).unwrap();
    assert!(result.iterations <= config.max_iterations);
    assert_eq!(result.frame.active_slot_count(), 4);
}

/// RAR with diffusion enabled still converges.
#[test]
fn rar_with_diffusion_converges() {
    let vfn = Vfn::new_random(42);
    let attn = SlotAttention::new_random(43);

    let diff = DiffusionConfig::uniform(0.01, 42);
    let config = RarConfig {
        epsilon: 0.5, // Generous threshold
        max_iterations: 50,
        dt: 0.1,
        beta: 0.5,
        diffusion: Some(diff),
        ..RarConfig::default()
    };

    let frame = make_frame(4);
    let result = rar_loop(&frame, &vfn, &attn, &config).unwrap();

    assert!(result.iterations <= 50);
    // Output should still be normalized
    for i in 0..4 {
        let slot = result.frame.read_slot(i).unwrap();
        let vec = slot.resolutions[0].as_ref().unwrap();
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "slot {} should be unit-normalized, got norm={}",
            i,
            norm
        );
    }
}

/// Diffusion noise changes the RAR trajectory.
#[test]
fn diffusion_changes_rar_trajectory() {
    let vfn = Vfn::new_random(42);
    let attn = SlotAttention::new_random(43);

    let config_no_noise = RarConfig {
        epsilon: 1e-10,
        max_iterations: 5,
        ..RarConfig::default()
    };

    let diff = DiffusionConfig::uniform(0.1, 42);
    let config_with_noise = RarConfig {
        epsilon: 1e-10,
        max_iterations: 5,
        diffusion: Some(diff),
        ..RarConfig::default()
    };

    let frame = make_frame(2);
    let result_no = rar_loop(&frame, &vfn, &attn, &config_no_noise).unwrap();
    let result_yes = rar_loop(&frame, &vfn, &attn, &config_with_noise).unwrap();

    // At least one slot should differ
    let any_different = (0..2).any(|i| {
        let s1 = result_no.frame.read_slot(i).unwrap().resolutions[0].unwrap();
        let s2 = result_yes.frame.read_slot(i).unwrap().resolutions[0].unwrap();
        s1 != s2
    });
    assert!(
        any_different,
        "diffusion noise should alter the RAR trajectory"
    );
}

/// Noise generation respects the active mask.
#[test]
fn noise_respects_active_mask() {
    let config = DiffusionConfig::uniform(1.0, 42);
    let mut mask = [false; MAX_SLOTS];
    mask[0] = true;
    mask[3] = true;

    let noise = generate_noise(&config, &mask, 1).unwrap();
    assert!(noise[0].is_some(), "active slot 0 should get noise");
    assert!(noise[3].is_some(), "active slot 3 should get noise");
    for i in 0..MAX_SLOTS {
        if !mask[i] {
            assert!(noise[i].is_none(), "inactive slot {} should get no noise", i);
        }
    }
}

/// Adaptive sigma decays for converging slots and grows for stuck slots.
#[test]
fn adaptive_sigma_integration() {
    let mut config = DiffusionConfig::uniform(0.5, 42);
    let deltas = [0.001f32; MAX_SLOTS]; // All slots close to convergence
    let converged = [false; MAX_SLOTS];
    let epsilon = 0.01;

    adapt_sigma(&mut config, &deltas, &converged, epsilon, 0.95, 1.05);

    // All slots have delta < epsilon, so sigma should decay
    for &s in &config.sigma {
        assert!(s < 0.5, "sigma should have decayed, got {}", s);
    }
}

/// Silent config produces identical results to no diffusion.
#[test]
fn silent_diffusion_matches_no_diffusion() {
    let vfn = Vfn::new_random(42);
    let attn = SlotAttention::new_random(43);

    let config_none = RarConfig {
        epsilon: 1e-10,
        max_iterations: 3,
        ..RarConfig::default()
    };

    let silent = DiffusionConfig::default(); // All zeros
    let config_silent = RarConfig {
        epsilon: 1e-10,
        max_iterations: 3,
        diffusion: Some(silent),
        ..RarConfig::default()
    };

    let frame = make_frame(2);
    let result_none = rar_loop(&frame, &vfn, &attn, &config_none).unwrap();
    let result_silent = rar_loop(&frame, &vfn, &attn, &config_silent).unwrap();

    assert_eq!(result_none.iterations, result_silent.iterations);
    for i in 0..2 {
        let s1 = result_none.frame.read_slot(i).unwrap().resolutions[0].unwrap();
        let s2 = result_silent.frame.read_slot(i).unwrap().resolutions[0].unwrap();
        assert_eq!(s1, s2, "silent diffusion should match no diffusion for slot {}", i);
    }
}
