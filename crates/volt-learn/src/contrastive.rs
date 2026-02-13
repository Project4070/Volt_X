//! InfoNCE contrastive loss for encoder training.
//!
//! Computes the symmetric InfoNCE loss between two sets of embeddings
//! (e.g., code summaries and docstring summaries). Positive pairs are
//! on the diagonal; all other pairs in the batch are negatives.
//!
//! # Example
//!
//! ```ignore
//! use volt_learn::contrastive::infonce_loss;
//! use candle_core::{Device, Tensor};
//!
//! let code_embeds = Tensor::randn(0.0, 1.0, (4, 256), &Device::Cpu).unwrap();
//! let doc_embeds = Tensor::randn(0.0, 1.0, (4, 256), &Device::Cpu).unwrap();
//! let loss = infonce_loss(&code_embeds, &doc_embeds, 0.07).unwrap();
//! ```

#[cfg(feature = "code-training")]
use candle_core::{DType, Tensor};
#[cfg(feature = "code-training")]
use volt_core::VoltError;

/// Computes symmetric InfoNCE contrastive loss.
///
/// Given code embeddings `C` and docstring embeddings `D` (both L2-normalized,
/// shape `[batch, dim]`), computes:
///
/// ```text
/// similarity = (C @ D.T) / temperature      [batch, batch]
/// loss_c2d = cross_entropy(similarity, labels)   (code → doc)
/// loss_d2c = cross_entropy(similarity.T, labels) (doc → code)
/// loss = (loss_c2d + loss_d2c) / 2
/// ```
///
/// where `labels = [0, 1, 2, ..., batch-1]` (diagonal = positive pairs).
///
/// # Arguments
///
/// * `code_embeds` — L2-normalized code summary vectors, shape `[batch, dim]`.
/// * `doc_embeds` — L2-normalized docstring summary vectors, shape `[batch, dim]`.
/// * `temperature` — Temperature scaling (typical: 0.07).
///
/// # Returns
///
/// Scalar loss tensor.
#[cfg(feature = "code-training")]
pub fn infonce_loss(
    code_embeds: &Tensor,
    doc_embeds: &Tensor,
    temperature: f64,
) -> Result<Tensor, VoltError> {
    let batch_size = code_embeds.dims()[0];

    // Similarity matrix: [batch, batch]
    let similarity = code_embeds
        .matmul(&doc_embeds.t().map_err(candle_err)?)
        .map_err(candle_err)?;

    // Scale by temperature
    let similarity = (similarity / temperature).map_err(candle_err)?;

    // Labels: [0, 1, 2, ..., batch-1]
    let labels = Tensor::arange(0u32, batch_size as u32, code_embeds.device())
        .map_err(candle_err)?;

    // Cross-entropy in both directions
    let loss_c2d = candle_nn::loss::cross_entropy(&similarity, &labels).map_err(candle_err)?;

    let sim_t = similarity.t().map_err(candle_err)?;
    let loss_d2c = candle_nn::loss::cross_entropy(&sim_t, &labels).map_err(candle_err)?;

    // Symmetric loss
    let loss = ((loss_c2d + loss_d2c).map_err(candle_err)? / 2.0).map_err(candle_err)?;

    Ok(loss)
}

/// Computes role classification cross-entropy loss.
///
/// # Arguments
///
/// * `role_probs` — Predicted role probabilities, shape `[batch, seq_len, 16]`.
/// * `role_labels` — Ground truth role indices, shape `[batch, seq_len]` (u32, 0..15).
///
/// # Returns
///
/// Scalar loss tensor.
#[cfg(feature = "code-training")]
pub fn role_classification_loss(
    role_logits: &Tensor,
    role_labels: &Tensor,
) -> Result<Tensor, VoltError> {
    // Reshape to [batch*seq_len, 16] and [batch*seq_len]
    let batch = role_logits.dims()[0];
    let seq_len = role_logits.dims()[1];
    let num_roles = role_logits.dims()[2];

    let logits_flat = role_logits
        .reshape((batch * seq_len, num_roles))
        .map_err(candle_err)?;
    let labels_flat = role_labels
        .reshape(batch * seq_len)
        .map_err(candle_err)?
        .to_dtype(DType::U32)
        .map_err(candle_err)?;

    candle_nn::loss::cross_entropy(&logits_flat, &labels_flat).map_err(candle_err)
}

#[cfg(feature = "code-training")]
fn candle_err(e: candle_core::Error) -> VoltError {
    VoltError::Internal {
        message: format!("candle error: {e}"),
    }
}

#[cfg(all(test, feature = "code-training"))]
mod tests {
    use super::*;
    use candle_core::{Device, D};

    fn l2_normalize(x: &Tensor) -> Tensor {
        let norm = x
            .sqr()
            .unwrap()
            .sum(D::Minus1)
            .unwrap()
            .sqrt()
            .unwrap()
            .clamp(1e-8, f64::INFINITY)
            .unwrap()
            .unsqueeze(D::Minus1)
            .unwrap();
        x.broadcast_div(&norm).unwrap()
    }

    #[test]
    fn infonce_loss_is_finite() {
        let code = Tensor::randn(0.0f32, 1.0, (4, 256), &Device::Cpu).unwrap();
        let doc = Tensor::randn(0.0f32, 1.0, (4, 256), &Device::Cpu).unwrap();
        let code = l2_normalize(&code);
        let doc = l2_normalize(&doc);

        let loss = infonce_loss(&code, &doc, 0.07).unwrap();
        let val: f32 = loss.to_scalar().unwrap();
        assert!(val.is_finite(), "loss should be finite, got {val}");
        assert!(val > 0.0, "loss should be positive, got {val}");
    }

    #[test]
    fn infonce_loss_identical_is_low() {
        // When code == doc, loss should be low (perfect matching)
        let embeds = Tensor::randn(0.0f32, 1.0, (4, 256), &Device::Cpu).unwrap();
        let embeds = l2_normalize(&embeds);

        let loss = infonce_loss(&embeds, &embeds, 0.07).unwrap();
        let val: f32 = loss.to_scalar().unwrap();
        assert!(
            val < 1.0,
            "identical embeddings should have low loss, got {val}"
        );
    }

    #[test]
    fn role_classification_loss_is_finite() {
        let logits = Tensor::randn(0.0f32, 1.0, (2, 10, 16), &Device::Cpu).unwrap();
        let labels = Tensor::zeros((2, 10), DType::U32, &Device::Cpu).unwrap();

        let loss = role_classification_loss(&logits, &labels).unwrap();
        let val: f32 = loss.to_scalar().unwrap();
        assert!(val.is_finite(), "role loss should be finite, got {val}");
    }
}
