from __future__ import annotations

from bisect import bisect_left
import csv
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO / "reports" / "easycom_official_spatial_speaker_v1"
VALIDATION_DIR = REPORT_DIR / "validation_step100"
PREDICTIONS = VALIDATION_DIR / "response_predictions.csv"
RAW_METRICS = VALIDATION_DIR / "response_metrics.json"
OUTPUT = REPORT_DIR / "selection.json"
GOALS = {"recall": 0.75, "correct": 0.50, "nts": 0.50, "silence": 0.85}


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def response_count(scores: list[float], threshold: float) -> int:
    return len(scores) - bisect_left(scores, threshold)


def score(row: dict[str, str]) -> float:
    return float(row["second_candidate_probability"]) + float(row["candidate_margin"])


def search(rows: list[dict[str, str]]) -> tuple[dict[str, Any], dict[str, Any]]:
    for row in rows:
        row["session_group"] = row["session_id"].split("/", 1)[0]
    sessions = sorted({row["session_group"] for row in rows})
    candidate_thresholds = sorted({0.0, 1.0, *(score(row) for row in rows)})
    vad_thresholds = sorted({0.0, 1.0, *(float(row["vad_probability"]) for row in rows)})
    counts = {
        session: {
            "addressed": sum(
                as_bool(row["should_respond"])
                for row in rows
                if row["session_group"] == session
            ),
            "nts": sum(
                row["sample_type"] == "non_target_speech"
                for row in rows
                if row["session_group"] == session
            ),
            "silence": sum(
                row["sample_type"] == "silence"
                for row in rows
                if row["session_group"] == session
            ),
        }
        for session in sessions
    }
    best_feasible = None
    best_feasible_key = None
    best_comprehensive = None
    best_comprehensive_key = None
    for vad_threshold in vad_thresholds:
        active: dict[str, dict[str, list[float]]] = {}
        for session in sessions:
            selected = [
                row
                for row in rows
                if row["session_group"] == session
                and float(row["vad_probability"]) >= vad_threshold
            ]
            active[session] = {
                "addressed": sorted(score(row) for row in selected if as_bool(row["should_respond"])),
                "correct": sorted(
                    score(row)
                    for row in selected
                    if as_bool(row["should_respond"]) and as_bool(row["best_candidate_correct"])
                ),
                "nts": sorted(score(row) for row in selected if row["sample_type"] == "non_target_speech"),
                "silence": sorted(score(row) for row in selected if row["sample_type"] == "silence"),
            }
        for candidate_threshold in candidate_thresholds:
            per_session = {}
            for session in sessions:
                values = active[session]
                total = counts[session]
                per_session[session] = {
                    "addressed_response_recall": response_count(
                        values["addressed"], candidate_threshold
                    ) / total["addressed"],
                    "addressed_correct_speaker_rate": response_count(
                        values["correct"], candidate_threshold
                    ) / total["addressed"],
                    "non_target_speech_rejection_rate": 1.0
                    - response_count(values["nts"], candidate_threshold) / total["nts"],
                    "silence_rejection_rate": 1.0
                    - response_count(values["silence"], candidate_threshold) / total["silence"],
                }
            normalized = []
            for values in per_session.values():
                normalized.extend(
                    [
                        min(values["addressed_response_recall"] / GOALS["recall"], 1.0),
                        min(values["addressed_correct_speaker_rate"] / GOALS["correct"], 1.0),
                        min(values["non_target_speech_rejection_rate"] / GOALS["nts"], 1.0),
                        min(values["silence_rejection_rate"] / GOALS["silence"], 1.0),
                    ]
                )
            harmonic = len(normalized) / sum(1.0 / max(value, 1e-6) for value in normalized)
            point = {
                "candidate_threshold": candidate_threshold,
                "vad_threshold": vad_threshold,
                "per_session": per_session,
                "worst_session_addressed_recall": min(
                    values["addressed_response_recall"] for values in per_session.values()
                ),
                "goal_normalized_harmonic_score": harmonic,
            }
            comprehensive_key = (harmonic, min(normalized))
            if best_comprehensive_key is None or comprehensive_key > best_comprehensive_key:
                best_comprehensive = point
                best_comprehensive_key = comprehensive_key
            feasible = all(
                values["addressed_correct_speaker_rate"] >= 0.50
                and values["non_target_speech_rejection_rate"] >= 0.50
                and values["silence_rejection_rate"] >= 0.85
                for values in per_session.values()
            )
            if feasible:
                feasible_key = (
                    point["worst_session_addressed_recall"],
                    min(values["addressed_correct_speaker_rate"] for values in per_session.values()),
                    harmonic,
                )
                if best_feasible_key is None or feasible_key > best_feasible_key:
                    best_feasible = point
                    best_feasible_key = feasible_key
    if best_feasible is None or best_comprehensive is None:
        raise RuntimeError("Expected both feasible and comprehensive operating points.")
    return best_feasible, best_comprehensive


def main() -> None:
    with PREDICTIONS.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    raw_metrics = json.loads(RAW_METRICS.read_text(encoding="utf-8"))
    feasible, comprehensive = search(rows)
    report = {
        "status": "complete_feasible_validation_operating_point",
        "experiment": "signed GCC-PHAT spatial speaker identity with set-max conditional addressing gate",
        "decision": "VAD gate AND max candidate addressing gate; spatial matcher selects speaker identity",
        "base_checkpoint": str(
            REPO / "checkpoints" / "easycom_official_speaker_residual_v1_stage_step100.pt"
        ),
        "spatial_speaker_checkpoint": str(
            REPO / "reports" / "easycom_gcc_bbox_speaker_pilot_v1" / "spatial_matcher.pt"
        ),
        "predictions": str(PREDICTIONS),
        "threshold_free_per_session": {
            session: {
                "addressed_speaker_top1": values["addressed_speaker_top1"],
                "target_auroc": values["target_auroc"],
                "speech_only_target_auroc": values["speech_only_target_auroc"],
                "vad_auroc": values["vad_auroc"],
            }
            for session, values in raw_metrics["per_session"].items()
        },
        "selection_rule": (
            "correct-speaker >= 0.50, NTS rejection >= 0.50, and silence rejection >= 0.85 "
            "separately on Sessions 4 and 12; maximize worst-session addressed recall"
        ),
        "operating_point": feasible,
        "comprehensive_operating_point": comprehensive,
        "test_sessions_used": [],
    }
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
