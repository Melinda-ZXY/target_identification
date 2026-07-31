from __future__ import annotations

from bisect import bisect_left
import csv
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

import torch


REPO = Path(r"C:\Users\melin\easycom_saal")
PYTHON = REPO / ".venv" / "Scripts" / "python.exe"
PREFIX = "easycom_official_candidate_ranking_pilot_v1"
LOGS = REPO / "logs"
CHECKPOINTS = REPO / "checkpoints"
REPORT_DIR = REPO / "reports" / PREFIX
PIPELINE_LOG = LOGS / f"{PREFIX}_pipeline.log"
SUMMARY = REPORT_DIR / "selection.json"
BASE_CHECKPOINT = CHECKPOINTS / "easycom_relation_adaptation_full_v1_best.pt"
BASE_PREDICTIONS = (
    REPO
    / "reports"
    / "easycom_official_split_relation_adaptation_v1"
    / "validation_step0"
    / "response_predictions.csv"
)
BASE_METRICS = BASE_PREDICTIONS.with_name("response_metrics.json")
TRAIN_MANIFEST = (
    REPO / "manifests" / "easycom_official_split_train_1_2_3_5_6_7_8_9.json"
)
VALIDATION_MANIFEST = (
    REPO / "manifests" / "easycom_official_split_validation_4_12.json"
)
BEST_CHECKPOINT = CHECKPOINTS / f"{PREFIX}_best.pt"
DEPLOYMENT_CHECKPOINT = CHECKPOINTS / f"{PREFIX}_best_calibrated.pt"
STAGE_ENDS = (200, 400)
GRADIENT_ACCUMULATION_STEPS = 2
MIN_CORRECT_SPEAKER_RATE = 0.50
MIN_NON_TARGET_SPEECH_REJECTION = 0.50
MIN_SILENCE_REJECTION = 0.85


def log(message: str) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    with PIPELINE_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}\n")


def checkpoint_step(path: Path) -> int:
    return int(path.stem.rsplit("step", 1)[1])


def latest_interim(output: Path) -> Path | None:
    candidates = list(output.parent.glob(f"{output.stem}_epoch1_step*.pt"))
    return max(candidates, key=checkpoint_step) if candidates else None


def run_logged(name: str, arguments: list[str], stdout: Path, stderr: Path) -> int:
    log(f"start {name}")
    with (
        stdout.open("a", encoding="utf-8", errors="replace") as stdout_handle,
        stderr.open("a", encoding="utf-8", errors="replace") as stderr_handle,
    ):
        completed = subprocess.run(
            [str(PYTHON), "-u", "-m", *arguments],
            cwd=REPO,
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            check=False,
            text=True,
        )
    log(f"finish {name} exit={completed.returncode}")
    return completed.returncode


def training_arguments(
    resume: Path,
    output: Path,
    progress: Path,
    step_offset: int,
    max_steps: int,
    reset_state: bool,
) -> list[str]:
    arguments = [
        "easycom_saal.train",
        "--easycom-root", r"D:\EasyComDataset_reextract",
        "--manifest", str(TRAIN_MANIFEST),
        "--resume-checkpoint", str(resume),
        "--epochs", "1",
        "--batch-size", "4",
        "--gradient-accumulation-steps", str(GRADIENT_ACCUMULATION_STEPS),
        "--lr", "1e-5",
        "--num-workers", "0",
        "--no-temporal-loss",
        "--use-timeline-multilabels",
        "--lambda-heatmap", "0",
        "--lambda-candidate-target-bce", "1",
        "--lambda-candidate-speaker-bce", "0",
        "--lambda-vad-bce", "0",
        "--lambda-candidate-classifier-ce", "0.5",
        "--candidate-target-threshold", "0.05",
        "--candidate-classifier-type", "temporal",
        "--use-relation-candidate-head",
        "--use-candidate-identity-memory",
        "--use-causal-memory",
        "--use-frame-aligned-audio",
        "--use-masked-head-video",
        "--preserve-audio-energy",
        "--streaming-smoothing-windows", "3",
        "--audio-global-gain-augmentation-db", "0",
        "--addressing-hard-negative-finetune",
        "--addressing-negative-weight", "1.25",
        "--addressing-hard-negative-fraction", "0.5",
        "--addressing-hard-negative-weight", "2",
        "--max-steps", str(max_steps),
        "--step-offset", str(step_offset),
        "--log-every", "10",
        "--checkpoint-every", "100",
        "--progress-path", str(progress),
        "--output", str(output),
    ]
    if reset_state:
        arguments.append("--reset-stateful-progress")
    return arguments


def train_stage(initial_checkpoint: Path, initial_step: int, target_step: int) -> Path:
    output = CHECKPOINTS / f"{PREFIX}_stage_step{target_step}.pt"
    if output.exists():
        log(f"skip completed training stage={target_step}")
        return output
    stdout = LOGS / f"{PREFIX}_stage_step{target_step}_train.out.log"
    stderr = LOGS / f"{PREFIX}_stage_step{target_step}_train.err.log"
    progress = LOGS / f"{PREFIX}_stage_step{target_step}.progress.json"
    for attempt in range(1, 4):
        interim = latest_interim(output)
        resume = interim or initial_checkpoint
        step_offset = checkpoint_step(interim) if interim is not None else initial_step
        remaining = target_step - step_offset
        if remaining <= 0:
            raise RuntimeError(f"Stage {target_step} has no final checkpoint")
        exit_code = run_logged(
            f"candidate-ranking training stage={target_step} attempt={attempt} offset={step_offset}",
            training_arguments(
                resume,
                output,
                progress,
                step_offset,
                remaining,
                reset_state=interim is None and initial_step == 0,
            ),
            stdout,
            stderr,
        )
        if exit_code == 0:
            return output
        if attempt < 3:
            time.sleep(30)
    raise RuntimeError(f"Candidate-ranking stage {target_step} failed three times")


def evaluate(checkpoint: Path, step: int) -> tuple[dict[str, Any], Path]:
    directory = REPORT_DIR / f"validation_step{step}"
    metrics = directory / "response_metrics.json"
    predictions = directory / "response_predictions.csv"
    if metrics.exists() and predictions.exists():
        return json.loads(metrics.read_text(encoding="utf-8")), predictions
    directory.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 4):
        exit_code = run_logged(
            f"Sessions 4+12 validation step={step} attempt={attempt}",
            [
                "easycom_saal.eval_response",
                "--easycom-root", r"D:\EasyComDataset_reextract",
                "--manifest", str(VALIDATION_MANIFEST),
                "--checkpoint", str(checkpoint),
                "--output-json", str(metrics),
                "--output-csv", str(predictions),
                "--batch-size", "4",
                "--num-workers", "0",
                "--log-every", "20",
            ],
            LOGS / f"{PREFIX}_validation_step{step}.out.log",
            LOGS / f"{PREFIX}_validation_step{step}.err.log",
        )
        if exit_code == 0:
            return json.loads(metrics.read_text(encoding="utf-8")), predictions
        if attempt < 3:
            time.sleep(30)
    raise RuntimeError(f"Validation step {step} failed three times")


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def response_count(scores: list[float], threshold: float) -> int:
    return len(scores) - bisect_left(scores, threshold)


def select_operating_point(predictions: Path) -> dict[str, Any] | None:
    with predictions.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["session_group"] = row["session_id"].split("/", 1)[0]
    sessions = sorted({row["session_group"] for row in rows})
    candidate_thresholds = sorted(
        {0.0, 1.0, *(float(row["predicted_probability"]) for row in rows)}
    )
    vad_thresholds = sorted({0.0, 1.0, *(float(row["vad_probability"]) for row in rows)})
    counts = {
        session: {
            "addressed": sum(
                as_bool(row["should_respond"])
                for row in rows
                if row["session_group"] == session
            ),
            "non_target_speech": sum(
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
    best: dict[str, Any] | None = None
    best_key: tuple[float, ...] | None = None
    for vad_threshold in vad_thresholds:
        active_scores: dict[str, dict[str, list[float]]] = {}
        for session in sessions:
            active = [
                row
                for row in rows
                if row["session_group"] == session
                and float(row["vad_probability"]) >= vad_threshold
            ]
            active_scores[session] = {
                "addressed": sorted(
                    float(row["predicted_probability"])
                    for row in active
                    if as_bool(row["should_respond"])
                ),
                "correct": sorted(
                    float(row["predicted_probability"])
                    for row in active
                    if as_bool(row["should_respond"])
                    and as_bool(row["best_candidate_correct"])
                ),
                "non_target_speech": sorted(
                    float(row["predicted_probability"])
                    for row in active
                    if row["sample_type"] == "non_target_speech"
                ),
                "silence": sorted(
                    float(row["predicted_probability"])
                    for row in active
                    if row["sample_type"] == "silence"
                ),
            }
        for candidate_threshold in candidate_thresholds:
            per_session: dict[str, dict[str, float]] = {}
            for session in sessions:
                scores = active_scores[session]
                addressed = counts[session]["addressed"]
                non_target = counts[session]["non_target_speech"]
                silence = counts[session]["silence"]
                per_session[session] = {
                    "addressed_response_recall": response_count(
                        scores["addressed"], candidate_threshold
                    )
                    / addressed,
                    "addressed_correct_speaker_rate": response_count(
                        scores["correct"], candidate_threshold
                    )
                    / addressed,
                    "non_target_speech_rejection_rate": 1.0
                    - response_count(scores["non_target_speech"], candidate_threshold)
                    / non_target,
                    "silence_rejection_rate": 1.0
                    - response_count(scores["silence"], candidate_threshold) / silence,
                }
            if any(
                values["addressed_correct_speaker_rate"] + 1e-12
                < MIN_CORRECT_SPEAKER_RATE
                or values["non_target_speech_rejection_rate"] + 1e-12
                < MIN_NON_TARGET_SPEECH_REJECTION
                or values["silence_rejection_rate"] + 1e-12 < MIN_SILENCE_REJECTION
                for values in per_session.values()
            ):
                continue
            worst_recall = min(
                values["addressed_response_recall"] for values in per_session.values()
            )
            point = {
                "candidate_threshold": candidate_threshold,
                "vad_threshold": vad_threshold,
                "per_session": per_session,
                "worst_session_addressed_recall": worst_recall,
            }
            key = (
                worst_recall,
                min(
                    values["addressed_correct_speaker_rate"]
                    for values in per_session.values()
                ),
                min(
                    values["non_target_speech_rejection_rate"]
                    for values in per_session.values()
                ),
                min(
                    values["silence_rejection_rate"] for values in per_session.values()
                ),
                -vad_threshold,
            )
            if best_key is None or key > best_key:
                best = point
                best_key = key
    return best


def stage_record(
    step: int,
    checkpoint: Path,
    metrics: dict[str, Any],
    predictions: Path,
) -> dict[str, Any]:
    return {
        "microstep": step,
        "optimizer_step": step // GRADIENT_ACCUMULATION_STEPS,
        "checkpoint": str(checkpoint),
        "predictions": str(predictions),
        "target_auroc": metrics.get("target_detection_auroc"),
        "speech_only_target_auroc": metrics.get("speech_only_target_detection_auroc"),
        "vad_auroc": metrics.get("vad_task", {}).get("auroc"),
        "operating_point": select_operating_point(predictions),
    }


def write_summary(values: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    temporary = SUMMARY.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(values, indent=2), encoding="utf-8")
    temporary.replace(SUMMARY)


def make_deployment_checkpoint(source: Path, selected: dict[str, Any]) -> None:
    checkpoint = torch.load(source, map_location="cpu")
    config = dict(checkpoint.get("config", {}))
    point = selected["operating_point"]
    config["candidate_target_threshold"] = float(point["candidate_threshold"])
    config["vad_threshold"] = float(point["vad_threshold"])
    config["use_vad_response_gate"] = True
    checkpoint["config"] = config
    checkpoint["validation_selection"] = selected
    temporary = DEPLOYMENT_CHECKPOINT.with_suffix(".pt.tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(DEPLOYMENT_CHECKPOINT)


def main() -> None:
    log("start official candidate-ranking pilot v1")
    base_metrics = json.loads(BASE_METRICS.read_text(encoding="utf-8"))
    stages = [stage_record(0, BASE_CHECKPOINT, base_metrics, BASE_PREDICTIONS)]
    checkpoint = BASE_CHECKPOINT
    previous_step = 0
    for target_step in STAGE_ENDS:
        checkpoint = train_stage(checkpoint, previous_step, target_step)
        metrics, predictions = evaluate(checkpoint, target_step)
        stages.append(stage_record(target_step, checkpoint, metrics, predictions))
        write_summary({"status": "running", "stages": stages})
        previous_step = target_step

    eligible = [stage for stage in stages if stage["operating_point"] is not None]
    selected = (
        max(
            eligible,
            key=lambda stage: (
                stage["operating_point"]["worst_session_addressed_recall"],
                float(stage["speech_only_target_auroc"] or float("-inf")),
            ),
        )
        if eligible
        else None
    )
    if selected is not None:
        shutil.copy2(Path(selected["checkpoint"]), BEST_CHECKPOINT)
        make_deployment_checkpoint(BEST_CHECKPOINT, selected)
    report = {
        "status": "complete" if selected is not None else "complete_no_feasible_checkpoint",
        "selection_rule": (
            "enforce correct-speaker >= 0.50, non-target-speech rejection >= 0.50, "
            "and silence rejection >= 0.85 separately on Sessions 4 and 12; "
            "then maximize worst-session addressed recall"
        ),
        "training": {
            "starting_checkpoint": str(BASE_CHECKPOINT),
            "microsteps": STAGE_ENDS[-1],
            "optimizer_updates": STAGE_ENDS[-1] // GRADIENT_ACCUMULATION_STEPS,
            "learning_rate": 1e-5,
            "candidate_classifier_ce_weight": 0.5,
            "candidate_target_bce_weight": 1.0,
            "trainable_components": [
                "candidate_relation_head.addressing_classifier"
            ],
            "frozen_components": [
                "VAD head",
                "speaking head",
                "audiovisual encoders",
                "fusion",
                "relation transformer",
                "identity-memory GRU",
            ],
        },
        "stages": stages,
        "selected_stage": selected,
        "best_checkpoint": str(BEST_CHECKPOINT) if selected is not None else None,
        "deployment_checkpoint": (
            str(DEPLOYMENT_CHECKPOINT) if selected is not None else None
        ),
        "test_sessions_used": [],
    }
    write_summary(report)
    log("official candidate-ranking pilot v1 complete")


if __name__ == "__main__":
    main()
