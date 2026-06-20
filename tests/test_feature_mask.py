import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import AxonExperimentConfig  # noqa: E402
from feature_mask import FeatureMaskedModel, load_feature_mask_tensors  # noqa: E402


def test_load_feature_mask_tensors_expands_padding_and_stat_bits(tmp_path):
    mask_path = tmp_path / "mask.json"
    mask_path.write_text(
        json.dumps({
            "mask_spec": {
                "pe_feature_dim": 5,
                "pe_search_dim": 3,
                "stat_feature_dim": 49,
            },
            "individual": [True, False, True] + [False] * 48 + [True],
            "kept_total": 3,
            "kept_pe": 2,
            "kept_stat": 1,
        }),
        encoding="utf-8",
    )
    config = AxonExperimentConfig(pe_feature_dim=5, stat_feature_dim=49)

    pe_mask, stat_mask, payload = load_feature_mask_tensors(mask_path, config, "cpu")

    assert pe_mask.tolist() == [1.0, 0.0, 1.0, 0.0, 0.0]
    assert stat_mask[:48].sum().item() == 0.0
    assert stat_mask[48].item() == 1.0
    assert payload["kept_total"] == 3


def test_feature_masked_model_applies_masks_before_forward():
    class EchoModel(torch.nn.Module):
        def forward(self, byte_seq, pe_features, stat_features=None, **kwargs):
            return {
                "pe": pe_features,
                "stat": stat_features,
                "byte": byte_seq,
                "kwargs": kwargs,
            }

    wrapped = FeatureMaskedModel(
        EchoModel(),
        pe_mask=torch.tensor([1.0, 0.0, 1.0]),
        stat_mask=torch.tensor([0.0, 1.0]),
    )

    out = wrapped(
        torch.tensor([[1, 2]]),
        torch.tensor([[2.0, 3.0, 4.0]]),
        stat_features=torch.tensor([[5.0, 6.0]]),
        return_state=True,
    )

    assert out["pe"].tolist() == [[2.0, 0.0, 4.0]]
    assert out["stat"].tolist() == [[0.0, 6.0]]
    assert out["kwargs"]["return_state"] is True
