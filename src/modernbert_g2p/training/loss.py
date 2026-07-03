"""Loss functions for Phase 2 pilots.

Implements `docs/design/phase2_tokenizer_pilots.md` §Trainer:

- Label-smoothed cross-entropy with per-row `sample_weight`
- BIO 3-class CE for accent-phrase boundary (APBP), scaled by ``alpha``

Both are subclasses of ``torch.nn.Module``. ``torch`` is imported lazily so
importing this module without touching the losses does not require torch.
Users access the classes as ``modernbert_g2p.training.loss.LabelSmoothingCELoss``
which triggers a first-touch materialization via PEP 562 ``__getattr__``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch  # noqa: F401

_LABEL_SMOOTHING_CLS: Any = None
_APBP_BIO_CLS: Any = None


def _build_label_smoothing_cls() -> Any:
    from torch import nn
    from torch.nn import functional as F  # noqa: N812

    class LabelSmoothingCELoss(nn.Module):
        """Label-smoothed CE with optional per-row ``sample_weight``.

        Per-token loss: ``L = -((1 - eps) * log p_y + eps / V * sum(log p_all))``.
        Padding positions (labels == ``ignore_index``) contribute zero. When
        ``sample_weight`` is a length-B tensor the per-token mean of each row
        is scaled by the row weight and the final reduction is the batch mean.
        """

        def __init__(self, smoothing: float = 0.1, ignore_index: int = -100) -> None:
            super().__init__()
            if not 0.0 <= smoothing < 1.0:
                raise ValueError(f"smoothing must be in [0, 1), got {smoothing}")
            self.smoothing = float(smoothing)
            self.ignore_index = int(ignore_index)

        def forward(
            self,
            logits: torch.Tensor,
            labels: torch.Tensor,
            sample_weight: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if logits.dim() < 2:
                raise ValueError(f"logits must have >= 2 dims, got {logits.dim()}")
            vocab_size = logits.size(-1)
            flat_logits = logits.reshape(-1, vocab_size)
            flat_labels = labels.reshape(-1)

            log_probs = F.log_softmax(flat_logits, dim=-1)
            valid_mask = flat_labels != self.ignore_index
            safe_labels = flat_labels.clamp(min=0)
            nll = -log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
            smooth = -log_probs.mean(dim=-1)
            per_token = (1.0 - self.smoothing) * nll + self.smoothing * smooth
            per_token = per_token * valid_mask.to(per_token.dtype)

            if sample_weight is None:
                denom = valid_mask.sum().clamp(min=1).to(per_token.dtype)
                return per_token.sum() / denom

            if sample_weight.dim() != 1:
                raise ValueError(
                    "sample_weight must be 1-D of shape (B,), got shape "
                    f"{tuple(sample_weight.shape)}"
                )
            batch_size = logits.size(0)
            if sample_weight.size(0) != batch_size:
                raise ValueError(
                    f"sample_weight length {sample_weight.size(0)} != batch size {batch_size}"
                )
            per_row_pt = per_token.view(batch_size, -1)
            per_row_valid = valid_mask.view(batch_size, -1).to(per_row_pt.dtype)
            per_row_sum = per_row_pt.sum(dim=-1)
            per_row_count = per_row_valid.sum(dim=-1).clamp(min=1)
            per_row_mean = per_row_sum / per_row_count
            weighted = per_row_mean * sample_weight.to(per_row_mean.dtype)
            return weighted.mean()

    return LabelSmoothingCELoss


def _build_apbp_bio_cls() -> Any:
    from torch import nn
    from torch.nn import functional as F  # noqa: N812

    class APBPBIOLoss(nn.Module):
        """3-class BIO CE for accent-phrase boundary, scaled by ``alpha``.

        Returns ``alpha * CE``; pass ``alpha=1.0`` to opt out of scaling and let
        the caller compose the multi-task weight explicitly.
        """

        def __init__(self, alpha: float = 0.2, ignore_index: int = -100) -> None:
            super().__init__()
            if alpha < 0.0:
                raise ValueError(f"alpha must be >= 0, got {alpha}")
            self.alpha = float(alpha)
            self.ignore_index = int(ignore_index)

        def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            if logits.size(-1) != 3:
                raise ValueError(f"APBP logits must have last dim=3, got {logits.size(-1)}")
            flat_logits = logits.reshape(-1, 3)
            flat_labels = labels.reshape(-1)
            ce = F.cross_entropy(
                flat_logits,
                flat_labels,
                ignore_index=self.ignore_index,
                reduction="mean",
            )
            return self.alpha * ce

    return APBPBIOLoss


def __getattr__(name: str) -> Any:
    global _LABEL_SMOOTHING_CLS, _APBP_BIO_CLS
    if name == "LabelSmoothingCELoss":
        if _LABEL_SMOOTHING_CLS is None:
            _LABEL_SMOOTHING_CLS = _build_label_smoothing_cls()
        return _LABEL_SMOOTHING_CLS
    if name == "APBPBIOLoss":
        if _APBP_BIO_CLS is None:
            _APBP_BIO_CLS = _build_apbp_bio_cls()
        return _APBP_BIO_CLS
    raise AttributeError(f"module 'modernbert_g2p.training.loss' has no attribute {name!r}")


__all__ = ["APBPBIOLoss", "LabelSmoothingCELoss"]  # noqa: F822 — PEP 562 __getattr__
