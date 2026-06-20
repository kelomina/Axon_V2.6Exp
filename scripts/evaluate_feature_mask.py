#!/usr/bin/env python3
"""Evaluate an exported feature mask against the full-feature baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import AxonExperimentConfig  # noqa: E402
from dataset import FeatureCacheDataset  # noqa: E402
from feature_mask import load_feature_mask_tensors, summarize_feature_mask  # noqa: E402
from model import AxonMalwareModel  # noqa: E402
from security import load_safe_checkpoint  # noqa: E402


def parse_thresholds(text: str) -> list[float]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise ValueError("at least one threshold is required")
    return values


def _safe_auc(labels: np.ndarray, probs: np.ndarray) -> Optional[float]:
    try:
        return float(roc_auc_score(labels, probs))
    except ValueError:
        return None


def metrics_at_threshold(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    preds = (probs >= threshold).astype(np.int64)
    labels = labels.astype(np.int64)
    tp = int(((labels == 1) & (preds == 1)).sum())
    tn = int(((labels == 0) & (preds == 0)).sum())
    fp = int(((labels == 0) & (preds == 1)).sum())
    fn = int(((labels == 1) & (preds == 0)).sum())
    return {
        "threshold": float(threshold),
        "sample_count": int(labels.shape[0]),
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "auc": _safe_auc(labels, probs),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "errors": int(fp + fn),
        "false_positive_rate": float(fp / max(1, int((labels == 0).sum()))),
        "false_negative_rate": float(fn / max(1, int((labels == 1).sum()))),
    }


@torch.no_grad()
def collect_probs(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    feature_mask=None,
) -> tuple[np.ndarray, np.ndarray]:
    probs = []
    labels_all = []
    model.eval()
    for byte_seq, pe_features, stat_features, labels in loader:
        byte_seq = byte_seq.to(device, non_blocking=True)
        pe_features = pe_features.to(device, non_blocking=True)
        stat_features = stat_features.to(device, non_blocking=True)
        if feature_mask is not None:
            pe_mask, stat_mask, _payload = feature_mask
            pe_features = pe_features * pe_mask.to(device=device, dtype=pe_features.dtype).view(1, -1)
            stat_features = stat_features * stat_mask.to(device=device, dtype=stat_features.dtype).view(1, -1)
        logits = model(byte_seq, pe_features, stat_features=stat_features)["logits"]
        probs.extend(torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy().tolist())
        labels_all.extend(labels.detach().cpu().numpy().astype(np.int64).tolist())
    return np.asarray(labels_all, dtype=np.int64), np.asarray(probs, dtype=np.float64)


def delta_metrics(mask_row: dict, full_row: dict) -> dict:
    keys = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "false_positive",
        "false_negative",
        "errors",
        "false_positive_rate",
        "false_negative_rate",
    ]
    return {key: mask_row[key] - full_row[key] for key in keys}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a feature mask vs full-feature baseline.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--feature-mask", type=Path, required=True)
    parser.add_argument("--samples-per-class", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--thresholds",
        default="0.45,0.50,0.55,0.60,0.65,0.70",
        help="Comma-separated thresholds to compare.",
    )
    parser.add_argument("--output-json", type=Path, default=Path("reports/feature_mask_eval.json"))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    checkpoint = load_safe_checkpoint(args.checkpoint, map_location=device)
    config = AxonExperimentConfig.from_dict(checkpoint["config"])

    model = AxonMalwareModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    feature_mask = load_feature_mask_tensors(args.feature_mask, config, device)
    assert feature_mask is not None
    print(f"[Feature Mask] {args.feature_mask} ({summarize_feature_mask(feature_mask[2])})")

    dataset = FeatureCacheDataset(
        data_dir=args.data_dir,
        max_byte_length=config.max_byte_length,
        pe_feature_dim=config.pe_feature_dim,
        stat_feature_dim=config.stat_feature_dim,
        max_samples_per_class=args.samples_per_class,
        axon_config=config,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    print(f"[Eval] samples={len(dataset)}, device={device}")

    labels, full_probs = collect_probs(model, loader, device, feature_mask=None)
    labels_mask, mask_probs = collect_probs(model, loader, device, feature_mask=feature_mask)
    if not np.array_equal(labels, labels_mask):
        raise RuntimeError("full and mask evaluation labels differ")

    thresholds = parse_thresholds(args.thresholds)
    rows = []
    for threshold in thresholds:
        full_row = metrics_at_threshold(labels, full_probs, threshold)
        mask_row = metrics_at_threshold(labels, mask_probs, threshold)
        rows.append({
            "threshold": float(threshold),
            "full": full_row,
            "mask": mask_row,
            "delta_mask_minus_full": delta_metrics(mask_row, full_row),
        })

    best_mask = max(rows, key=lambda row: (row["mask"]["f1"], -row["mask"]["errors"]))
    payload = {
        "checkpoint": str(args.checkpoint),
        "data_dir": args.data_dir,
        "feature_mask": str(args.feature_mask),
        "feature_mask_summary": {
            "kept_total": feature_mask[2].get("kept_total"),
            "kept_pe": feature_mask[2].get("kept_pe"),
            "kept_stat": feature_mask[2].get("kept_stat"),
            "note": feature_mask[2].get("note"),
        },
        "samples": int(labels.shape[0]),
        "thresholds": thresholds,
        "rows": rows,
        "best_mask_threshold": best_mask["threshold"],
        "best_mask_metrics": best_mask["mask"],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("threshold | full_f1 | mask_f1 | full_fp/fn | mask_fp/fn | mask_errors")
    for row in rows:
        print(
            f"{row['threshold']:.2f} | {row['full']['f1']:.4f} | {row['mask']['f1']:.4f} | "
            f"{row['full']['false_positive']}/{row['full']['false_negative']} | "
            f"{row['mask']['false_positive']}/{row['mask']['false_negative']} | "
            f"{row['mask']['errors']}"
        )
    print(f"JSON: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
