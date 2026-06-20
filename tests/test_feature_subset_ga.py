import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from search_feature_subset_ga import (  # noqa: E402
    FeatureMaskSpec,
    FitnessConfig,
    compute_fitness,
    choose_best_metrics,
    parse_thresholds,
    repair_individual,
    run_genetic_search,
    split_batches_stratified,
)


def test_parse_thresholds_rejects_empty_values():
    assert parse_thresholds("0.4, 0.5,0.6") == [0.4, 0.5, 0.6]

    try:
        parse_thresholds("")
    except ValueError as exc:
        assert "at least one threshold" in str(exc)
    else:
        raise AssertionError("Expected empty thresholds to raise")


def test_feature_mask_spec_expands_search_bits_and_leaves_padding_zero():
    spec = FeatureMaskSpec(pe_feature_dim=5, stat_feature_dim=2, pe_search_dim=3)
    individual = np.array([1, 0, 1, 0, 1], dtype=bool)

    pe_mask, stat_mask = spec.to_model_masks(individual)

    assert pe_mask.tolist() == [1.0, 0.0, 1.0, 0.0, 0.0]
    assert stat_mask.tolist() == [0.0, 1.0]
    assert spec.selected_indices(individual) == ([0, 2], [1])


def test_choose_best_metrics_prefers_best_f1_threshold():
    labels = np.array([1, 1, 0, 0], dtype=np.int64)
    probs = np.array([0.9, 0.6, 0.55, 0.1], dtype=np.float64)

    best = choose_best_metrics(labels, probs, thresholds=[0.5, 0.7])

    assert best["threshold"] == 0.5
    assert best["f1"] > 0.7
    assert best["fp"] == 1


def test_compute_fitness_penalizes_larger_feature_sets_when_metrics_match():
    config = FitnessConfig(objective="f1", feature_penalty=0.1)
    metrics = {"f1": 0.95, "sample_count": 10, "fp": 0, "fn": 0}

    compact = compute_fitness(metrics, kept_ratio=0.25, fitness_config=config)
    large = compute_fitness(metrics, kept_ratio=0.75, fitness_config=config)

    assert compact > large


def test_compute_fitness_respects_min_objective_guard():
    config = FitnessConfig(
        objective="f1",
        feature_penalty=0.1,
        min_objective=0.95,
        below_min_penalty=5.0,
    )
    protected = compute_fitness(
        {"f1": 0.95, "sample_count": 10, "fp": 0, "fn": 0},
        kept_ratio=0.8,
        fitness_config=config,
    )
    too_low = compute_fitness(
        {"f1": 0.90, "sample_count": 10, "fp": 0, "fn": 0},
        kept_ratio=0.1,
        fitness_config=config,
    )

    assert protected > too_low


def test_repair_individual_enforces_minimum_pe_and_stat_features():
    spec = FeatureMaskSpec(pe_feature_dim=4, stat_feature_dim=3, pe_search_dim=4)
    empty = np.zeros(spec.search_dim, dtype=bool)

    repaired = repair_individual(
        empty,
        spec,
        rng=__import__("random").Random(7),
        min_pe_features=2,
        min_stat_features=1,
    )

    pe_bits, stat_bits = spec.split_individual(repaired)
    assert pe_bits.sum() >= 2
    assert stat_bits.sum() >= 1


def test_run_genetic_search_accepts_synthetic_evaluator():
    spec = FeatureMaskSpec(pe_feature_dim=4, stat_feature_dim=2, pe_search_dim=4)

    def evaluate(individual: np.ndarray) -> dict:
        # 这个合成任务只奖励第 1 和第 4 个搜索位，用来验证 GA 管线能工作。
        score = float(individual[[1, 4]].sum() / 2)
        return {
            "threshold": 0.5,
            "sample_count": 4,
            "accuracy": score,
            "precision": score,
            "recall": score,
            "f1": score,
            "auc": score,
            "tp": int(score * 2),
            "tn": 2,
            "fp": 0,
            "fn": int((1 - score) * 2),
            "errors": int((1 - score) * 2),
            "prob_mean": 0.5,
            "prob_std": 0.0,
        }

    result = run_genetic_search(
        spec=spec,
        ga_config=__import__("search_feature_subset_ga").GeneticConfig(
            population_size=8,
            generations=2,
            elite_size=2,
            mutation_rate=0.05,
            min_pe_features=1,
            min_stat_features=1,
        ),
        fitness_config=FitnessConfig(objective="f1", feature_penalty=0.0),
        evaluate_individual=evaluate,
        seed=3,
    )

    assert result["best"]["metrics"]["f1"] == 1.0
    assert result["evaluated_candidates"] >= 1


def test_split_batches_stratified_creates_balanced_holdout():
    torch = __import__("torch")
    byte_seq = torch.arange(8 * 4).reshape(8, 4)
    pe_features = torch.arange(8 * 3, dtype=torch.float32).reshape(8, 3)
    stat_features = torch.arange(8 * 2, dtype=torch.float32).reshape(8, 2)
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)

    search_batches, search_labels, holdout_batches, holdout_labels = split_batches_stratified(
        [(byte_seq, pe_features, stat_features)],
        labels,
        holdout_ratio=0.25,
        seed=11,
    )

    assert len(search_batches) == 1
    assert len(holdout_batches) == 1
    assert search_labels.shape[0] == 6
    assert holdout_labels.shape[0] == 2
    assert search_labels.tolist().count(0) == 3
    assert search_labels.tolist().count(1) == 3
    assert holdout_labels.tolist().count(0) == 1
    assert holdout_labels.tolist().count(1) == 1
