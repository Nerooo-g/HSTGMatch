"""
Evaluation metrics for map-matching: Precision, Recall, and F1.

Map-matching evaluation treats each predicted route as a set (or multi-set)
of road segments and compares it to the ground-truth route.

Precision = |pred ∩ truth| / |pred|
Recall    = |pred ∩ truth| / |truth|
F1        = 2 * P * R / (P + R)

Both segment-level (exact match) and sequence-level variants are supported.
"""

from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Per-sample metrics
# ---------------------------------------------------------------------------

def _precision_recall_f1_single(
    pred: List[int],
    truth: List[int],
) -> Tuple[float, float, float]:
    """
    Compute Precision, Recall, F1 for one predicted vs. ground-truth route.

    Uses multiset intersection: counts duplicate segment IDs correctly.
    """
    if len(pred) == 0 and len(truth) == 0:
        return 1.0, 1.0, 1.0
    if len(pred) == 0:
        return 0.0, 0.0, 0.0
    if len(truth) == 0:
        return 0.0, 0.0, 0.0

    # Multiset intersection
    from collections import Counter
    pred_counter = Counter(pred)
    truth_counter = Counter(truth)

    intersection = 0
    for seg_id, cnt in pred_counter.items():
        intersection += min(cnt, truth_counter.get(seg_id, 0))

    precision = intersection / len(pred)
    recall = intersection / len(truth)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


# ---------------------------------------------------------------------------
# Batch metrics
# ---------------------------------------------------------------------------

def compute_precision_recall_f1(
    predictions: List[List[int]],
    ground_truths: List[List[int]],
) -> Tuple[float, float, float]:
    """
    Compute macro-averaged Precision, Recall, F1 over a list of route pairs.

    Args:
        predictions:   List of predicted seg_id sequences.
        ground_truths: List of ground-truth seg_id sequences.

    Returns:
        (mean_precision, mean_recall, mean_f1)
    """
    assert len(predictions) == len(ground_truths), (
        "predictions and ground_truths must have the same length"
    )

    precisions, recalls, f1s = [], [], []
    for pred, truth in zip(predictions, ground_truths):
        p, r, f = _precision_recall_f1_single(pred, truth)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    return float(np.mean(precisions)), float(np.mean(recalls)), float(np.mean(f1s))


# ---------------------------------------------------------------------------
# RouteEvaluator — accumulates metrics across batches
# ---------------------------------------------------------------------------

class RouteEvaluator:
    """
    Stateful evaluator that accumulates per-sample metrics across batches
    and computes final macro-average scores.

    Usage:
        evaluator = RouteEvaluator()
        for pred_batch, truth_batch in ...:
            evaluator.update(pred_batch, truth_batch)
        p, r, f1 = evaluator.compute()
        evaluator.reset()
    """

    def __init__(self) -> None:
        self.precisions: List[float] = []
        self.recalls: List[float] = []
        self.f1s: List[float] = []

    def update(
        self,
        predictions: List[List[int]],
        ground_truths: List[List[int]],
    ) -> None:
        for pred, truth in zip(predictions, ground_truths):
            p, r, f = _precision_recall_f1_single(pred, truth)
            self.precisions.append(p)
            self.recalls.append(r)
            self.f1s.append(f)

    def compute(self) -> Tuple[float, float, float]:
        """Return (mean_precision, mean_recall, mean_f1)."""
        if not self.precisions:
            return 0.0, 0.0, 0.0
        return (
            float(np.mean(self.precisions)),
            float(np.mean(self.recalls)),
            float(np.mean(self.f1s)),
        )

    def reset(self) -> None:
        self.precisions.clear()
        self.recalls.clear()
        self.f1s.clear()

    def __str__(self) -> str:
        p, r, f = self.compute()
        return f"Precision={p:.4f}  Recall={r:.4f}  F1={f:.4f}  (n={len(self.f1s)})"
