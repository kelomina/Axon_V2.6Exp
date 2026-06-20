import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from config import AxonExperimentConfig  # noqa: E402
from evaluate_feature_mask import build_summary, choose_best_rows, row_at_threshold  # noqa: E402
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


def _metric_row(
    threshold,
    *,
    f1,
    errors,
    false_positive,
    false_negative,
    recall=0.9,
    precision=0.9,
    accuracy=0.9,
):
    return {
        "threshold": threshold,
        "sample_count": 100,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": 0.95,
        "true_positive": 45,
        "true_negative": 45,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "errors": errors,
        "false_positive_rate": false_positive / 50,
        "false_negative_rate": false_negative / 50,
    }


def _threshold_row(threshold, full_metrics, mask_metrics):
    from evaluate_feature_mask import delta_metrics

    return {
        "threshold": threshold,
        "full": full_metrics,
        "mask": mask_metrics,
        "delta_mask_minus_full": delta_metrics(mask_metrics, full_metrics),
    }


def test_choose_best_rows_tracks_f1_and_error_optima_separately():
    rows = [
        _threshold_row(
            0.5,
            _metric_row(0.5, f1=0.90, errors=10, false_positive=4, false_negative=6),
            _metric_row(0.5, f1=0.92, errors=8, false_positive=5, false_negative=3),
        ),
        _threshold_row(
            0.55,
            _metric_row(0.55, f1=0.91, errors=9, false_positive=3, false_negative=6),
            _metric_row(0.55, f1=0.91, errors=6, false_positive=2, false_negative=4),
        ),
    ]

    best = choose_best_rows(rows, "mask")

    assert best["best_f1"]["threshold"] == 0.5
    assert best["best_errors"]["threshold"] == 0.55


def test_build_summary_compares_recommendations_to_full_baseline_threshold():
    rows = [
        _threshold_row(
            0.5,
            _metric_row(0.5, f1=0.90, errors=10, false_positive=4, false_negative=6),
            _metric_row(0.5, f1=0.93, errors=7, false_positive=5, false_negative=2),
        ),
        _threshold_row(
            0.55,
            _metric_row(0.55, f1=0.91, errors=9, false_positive=3, false_negative=6),
            _metric_row(0.55, f1=0.92, errors=6, false_positive=2, false_negative=4),
        ),
    ]

    summary = build_summary(rows, baseline_threshold=0.5)

    assert summary["baseline_full"]["metrics"]["errors"] == 10
    assert summary["best_mask_f1"]["threshold"] == 0.5
    assert summary["best_mask_f1"]["delta_vs_baseline_full"]["f1"] > 0
    assert summary["best_mask_errors"]["threshold"] == 0.55
    assert summary["best_mask_errors"]["delta_vs_baseline_full"]["errors"] == -4


def test_row_at_threshold_reports_available_values_when_missing():
    rows = [
        _threshold_row(
            0.5,
            _metric_row(0.5, f1=0.90, errors=10, false_positive=4, false_negative=6),
            _metric_row(0.5, f1=0.93, errors=7, false_positive=5, false_negative=2),
        )
    ]

    try:
        row_at_threshold(rows, 0.6)
    except ValueError as exc:
        message = str(exc)
        assert "baseline threshold 0.6" in message
        assert "0.5" in message
    else:
        raise AssertionError("Expected a missing baseline threshold to raise")
