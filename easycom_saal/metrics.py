from __future__ import annotations


def binary_auroc(scores: list[float], labels: list[bool]) -> float:
    positives = [score for score, label in zip(scores, labels) if label]
    negatives = [score for score, label in zip(scores, labels) if not label]
    if not positives or not negatives:
        return float("nan")
    wins = sum(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positives
        for negative in negatives
    )
    return wins / (len(positives) * len(negatives))


def binary_average_precision(scores: list[float], labels: list[bool]) -> float:
    positive_count = sum(labels)
    if positive_count == 0:
        return float("nan")
    ordered = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    true_positives = 0
    precision_sum = 0.0
    for rank, (_, label) in enumerate(ordered, start=1):
        if label:
            true_positives += 1
            precision_sum += true_positives / rank
    return precision_sum / positive_count
