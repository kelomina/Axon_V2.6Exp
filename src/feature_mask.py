"""Utilities for applying exported PE/stat feature masks.

Feature masks are exported by ``scripts/search_feature_subset_ga.py``.  They are
input-side switches: selected scalar PE/stat features pass through, while
unselected scalar features are zeroed before the model sees them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch

from config import AxonExperimentConfig


def individual_from_mask_payload(payload: dict) -> list[bool]:
    """Return the boolean search vector stored in an exported mask payload."""
    if "individual" in payload:
        return [bool(value) for value in payload["individual"]]

    mask_spec = payload.get("mask_spec", {})
    pe_search_dim = int(mask_spec.get("pe_search_dim", 0))
    stat_feature_dim = int(mask_spec.get("stat_feature_dim", 0))
    if pe_search_dim <= 0 or stat_feature_dim <= 0:
        raise ValueError("Feature mask must contain individual or mask_spec dimensions")

    individual = [False] * (pe_search_dim + stat_feature_dim)
    for index in payload.get("selected_pe_indices", []):
        index = int(index)
        if index < 0 or index >= pe_search_dim:
            raise ValueError(f"PE feature index out of searched range: {index}")
        individual[index] = True
    for index in payload.get("selected_stat_indices", []):
        index = int(index)
        if index < 0 or index >= stat_feature_dim:
            raise ValueError(f"stat feature index out of range: {index}")
        individual[pe_search_dim + index] = True
    return individual


def load_feature_mask_payload(feature_mask_path: str | Path) -> dict:
    path = Path(feature_mask_path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_feature_mask_tensors(
    feature_mask_path: Optional[str | Path],
    config: AxonExperimentConfig,
    device: str | torch.device,
) -> Optional[tuple[torch.Tensor, torch.Tensor, dict]]:
    """Load an exported feature mask and expand it to model input dimensions."""
    if not feature_mask_path:
        return None

    payload = load_feature_mask_payload(feature_mask_path)
    mask_spec = payload.get("mask_spec", {})
    pe_search_dim = int(mask_spec.get("pe_search_dim", config.pe_feature_dim))
    pe_feature_dim = int(mask_spec.get("pe_feature_dim", config.pe_feature_dim))
    stat_feature_dim = int(mask_spec.get("stat_feature_dim", config.stat_feature_dim))

    if pe_feature_dim != config.pe_feature_dim:
        raise ValueError(
            f"Feature mask PE dim {pe_feature_dim} does not match model PE dim {config.pe_feature_dim}"
        )
    if stat_feature_dim != config.stat_feature_dim:
        raise ValueError(
            f"Feature mask stat dim {stat_feature_dim} does not match model stat dim {config.stat_feature_dim}"
        )
    if pe_search_dim <= 0 or pe_search_dim > config.pe_feature_dim:
        raise ValueError(f"Invalid feature mask pe_search_dim: {pe_search_dim}")

    individual = individual_from_mask_payload(payload)
    expected_len = pe_search_dim + config.stat_feature_dim
    if len(individual) != expected_len:
        raise ValueError(
            f"Feature mask individual length {len(individual)} does not match expected {expected_len}"
        )

    pe_mask = torch.zeros(config.pe_feature_dim, dtype=torch.float32, device=device)
    stat_mask = torch.zeros(config.stat_feature_dim, dtype=torch.float32, device=device)
    pe_bits = individual[:pe_search_dim]
    stat_bits = individual[pe_search_dim:]
    pe_mask[:pe_search_dim] = torch.tensor(pe_bits, dtype=torch.float32, device=device)
    stat_mask[:] = torch.tensor(stat_bits, dtype=torch.float32, device=device)
    return pe_mask, stat_mask, payload


def apply_feature_mask_to_tensors(
    pe_tensor: torch.Tensor,
    stat_tensor: torch.Tensor,
    feature_mask: Optional[tuple[torch.Tensor, torch.Tensor, dict]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply an expanded feature mask to one batch of PE/stat tensors."""
    if feature_mask is None:
        return pe_tensor, stat_tensor
    pe_mask, stat_mask, _payload = feature_mask
    return (
        pe_tensor * pe_mask.to(device=pe_tensor.device, dtype=pe_tensor.dtype).view(1, -1),
        stat_tensor * stat_mask.to(device=stat_tensor.device, dtype=stat_tensor.dtype).view(1, -1),
    )


class FeatureMaskedModel(torch.nn.Module):
    """Wrap a model and apply an exported PE/stat feature mask before forward."""

    def __init__(self, model: torch.nn.Module, pe_mask: torch.Tensor, stat_mask: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer("pe_mask", pe_mask.float().view(1, -1))
        self.register_buffer("stat_mask", stat_mask.float().view(1, -1))

    def forward(self, byte_seq, pe_features, stat_features=None, **kwargs):
        pe_features = pe_features * self.pe_mask.to(device=pe_features.device, dtype=pe_features.dtype)
        if stat_features is not None:
            stat_features = stat_features * self.stat_mask.to(
                device=stat_features.device,
                dtype=stat_features.dtype,
            )
        return self.model(byte_seq, pe_features, stat_features=stat_features, **kwargs)


def summarize_feature_mask(payload: dict) -> str:
    kept_total = payload.get("kept_total")
    kept_pe = payload.get("kept_pe")
    kept_stat = payload.get("kept_stat")
    return f"kept_total={kept_total}, kept_pe={kept_pe}, kept_stat={kept_stat}"
