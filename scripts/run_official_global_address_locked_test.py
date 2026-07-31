from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[1]
SOURCE_REPORT_DIR = REPO / "reports" / "easycom_official_global_address_overnight_v1"
DEPLOYMENT_MANIFEST = SOURCE_REPORT_DIR / "deployment_manifest.json"
SELECTION_REPORT = SOURCE_REPORT_DIR / "selection.json"
TEST_MANIFEST = REPO / "manifests" / "easycom_official_split_test_10_11.json"
REPORT_DIR = SOURCE_REPORT_DIR / "test_10_11"
METRICS_PATH = REPORT_DIR / "response_metrics.json"
PREDICTIONS_PATH = REPORT_DIR / "response_predictions.csv"
LOCKED_TEST_REPORT = REPORT_DIR / "locked_test_report.json"
STDOUT_LOG = REPO / "logs" / "easycom_official_global_address_overnight_v1_test_10_11.out.log"
STDERR_LOG = REPO / "logs" / "easycom_official_global_address_overnight_v1_test_10_11.err.log"
EXPECTED_TEST_SESSIONS = {"Session_10", "Session_11"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen global-addressedness model on the official test split."
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate and print the locked test configuration without evaluating.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, values: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(values, indent=2), encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_sessions(path: Path) -> set[str]:
    manifest = read_json(path)
    return {
        str(session["session_id"]).split("/", 1)[0]
        for session in manifest.get("sessions", [])
    }


def locked_configuration() -> dict[str, Any]:
    deployment = read_json(DEPLOYMENT_MANIFEST)
    selection = read_json(SELECTION_REPORT)
    if deployment.get("test_sessions_used") != []:
        raise RuntimeError("Deployment thresholds were not selected without test sessions.")
    if selection.get("test_sessions_used") != []:
        raise RuntimeError("Model selection was not isolated from the test sessions.")

    selected = selection.get("selected_stage")
    if not isinstance(selected, dict):
        raise RuntimeError("Selection report does not contain a selected stage.")
    point = selected.get("operating_point")
    if not isinstance(point, dict):
        raise RuntimeError("Selected stage does not contain an operating point.")

    candidate_threshold = float(deployment["candidate_threshold"])
    vad_threshold = float(deployment["vad_threshold"])
    if not math.isclose(candidate_threshold, float(point["candidate_threshold"])):
        raise RuntimeError("Deployment candidate threshold differs from model selection.")
    if not math.isclose(vad_threshold, float(point["vad_threshold"])):
        raise RuntimeError("Deployment VAD threshold differs from model selection.")

    model_checkpoint = Path(deployment["model_checkpoint"])
    spatial_checkpoint = Path(deployment["spatial_speaker_checkpoint"])
    required_paths = (
        model_checkpoint,
        spatial_checkpoint,
        TEST_MANIFEST,
    )
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Locked test inputs are missing: {missing}")

    sessions = manifest_sessions(TEST_MANIFEST)
    if sessions != EXPECTED_TEST_SESSIONS:
        raise RuntimeError(
            f"Expected test sessions {sorted(EXPECTED_TEST_SESSIONS)}, got {sorted(sessions)}."
        )

    return {
        "model_checkpoint": str(model_checkpoint),
        "spatial_speaker_checkpoint": str(spatial_checkpoint),
        "test_manifest": str(TEST_MANIFEST),
        "test_sessions": sorted(sessions),
        "candidate_threshold": candidate_threshold,
        "vad_threshold": vad_threshold,
        "decision": deployment["decision"],
        "selected_stage": selected["name"],
    }


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def per_session_operating_metrics(path: Path) -> dict[str, dict[str, float | int]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    sessions = sorted({row["session_id"].split("/", 1)[0] for row in rows})
    result: dict[str, dict[str, float | int]] = {}
    for session in sessions:
        session_rows = [
            row for row in rows if row["session_id"].split("/", 1)[0] == session
        ]
        addressed = [row for row in session_rows if as_bool(row["should_respond"])]
        non_target_speech = [
            row for row in session_rows if row["sample_type"] == "non_target_speech"
        ]
        silence = [row for row in session_rows if row["sample_type"] == "silence"]
        result[session] = {
            "samples": len(session_rows),
            "addressed_samples": len(addressed),
            "non_target_speech_samples": len(non_target_speech),
            "silence_samples": len(silence),
            "addressed_response_recall": sum(
                as_bool(row["predicted_should_respond"]) for row in addressed
            )
            / len(addressed),
            "addressed_correct_speaker_rate": sum(
                as_bool(row["correct_action"]) for row in addressed
            )
            / len(addressed),
            "non_target_speech_rejection_rate": sum(
                as_bool(row["correct_action"]) for row in non_target_speech
            )
            / len(non_target_speech),
            "silence_rejection_rate": sum(
                as_bool(row["correct_action"]) for row in silence
            )
            / len(silence),
        }
    return result


def completed_report(configuration: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "complete_locked_test",
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "configuration": configuration,
        "input_sha256": {
            "model_checkpoint": sha256(Path(configuration["model_checkpoint"])),
            "spatial_speaker_checkpoint": sha256(
                Path(configuration["spatial_speaker_checkpoint"])
            ),
            "test_manifest": sha256(TEST_MANIFEST),
        },
        "artifacts": {
            "metrics": str(METRICS_PATH),
            "predictions": str(PREDICTIONS_PATH),
            "stdout_log": str(STDOUT_LOG),
            "stderr_log": str(STDERR_LOG),
        },
        "overall_operating_metrics": {
            key: metrics[key]
            for key in (
                "samples",
                "addressed_response_recall",
                "addressed_correct_speaker_rate",
                "non_target_speech_rejection_rate",
                "silence_rejection_rate",
                "target_detection_auroc",
                "speech_only_target_detection_auroc",
            )
        },
        "per_session_operating_metrics": per_session_operating_metrics(PREDICTIONS_PATH),
        "threshold_free_per_session": metrics["per_session"],
        "timing": metrics["timing"],
        "test_data_was_used_for_selection_or_tuning": False,
    }


def run_locked_test(configuration: dict[str, Any]) -> dict[str, Any]:
    final_artifacts = (METRICS_PATH, PREDICTIONS_PATH, LOCKED_TEST_REPORT)
    existing = [path for path in final_artifacts if path.exists()]
    if len(existing) == len(final_artifacts):
        return read_json(LOCKED_TEST_REPORT)
    if existing:
        raise RuntimeError(
            "Partial final test artifacts exist; refusing to overwrite a held-out evaluation: "
            f"{[str(path) for path in existing]}"
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    STDOUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    temporary_metrics = REPORT_DIR / "response_metrics.in_progress.json"
    temporary_predictions = REPORT_DIR / "response_predictions.in_progress.csv"
    command = [
        sys.executable,
        "-u",
        "-m",
        "easycom_saal.eval_response",
        "--easycom-root",
        r"D:\EasyComDataset_reextract",
        "--manifest",
        configuration["test_manifest"],
        "--checkpoint",
        configuration["model_checkpoint"],
        "--spatial-speaker-checkpoint",
        configuration["spatial_speaker_checkpoint"],
        "--candidate-target-threshold",
        str(configuration["candidate_threshold"]),
        "--vad-response-gate-threshold",
        str(configuration["vad_threshold"]),
        "--output-json",
        str(temporary_metrics),
        "--output-csv",
        str(temporary_predictions),
        "--batch-size",
        "4",
        "--num-workers",
        "0",
        "--log-every",
        "20",
    ]
    with (
        STDOUT_LOG.open("a", encoding="utf-8", errors="replace") as stdout_handle,
        STDERR_LOG.open("a", encoding="utf-8", errors="replace") as stderr_handle,
    ):
        completed = subprocess.run(
            command,
            cwd=REPO,
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            check=False,
            text=True,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Locked test evaluation failed with exit code {completed.returncode}; "
            f"see {STDERR_LOG}."
        )
    if not temporary_metrics.is_file() or not temporary_predictions.is_file():
        raise RuntimeError("Evaluator exited successfully without producing both test artifacts.")

    temporary_metrics.replace(METRICS_PATH)
    temporary_predictions.replace(PREDICTIONS_PATH)
    metrics = read_json(METRICS_PATH)
    report = completed_report(configuration, metrics)
    write_json_atomic(LOCKED_TEST_REPORT, report)
    return report


def main() -> None:
    args = parse_args()
    configuration = locked_configuration()
    if args.check_only:
        print(json.dumps({"status": "locked_test_configuration_valid", **configuration}, indent=2))
        return
    report = run_locked_test(configuration)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
