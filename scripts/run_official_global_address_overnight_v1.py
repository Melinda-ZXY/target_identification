from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any


REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from run_official_candidate_ranking_pilot_v1 import select_operating_point


PYTHON = REPO / ".venv" / "Scripts" / "python.exe"
PREFIX = "easycom_official_global_address_overnight_v1"
LOG_DIR = REPO / "logs"
CHECKPOINT_DIR = REPO / "checkpoints"
REPORT_DIR = REPO / "reports" / PREFIX
PIPELINE_LOG = LOG_DIR / f"{PREFIX}_pipeline.log"
SUMMARY = REPORT_DIR / "selection.json"
DEPLOYMENT = REPORT_DIR / "deployment_manifest.json"
TRAIN_MANIFEST = REPO / "manifests" / "easycom_official_split_train_1_2_3_5_6_7_8_9.json"
VALIDATION_MANIFEST = REPO / "manifests" / "easycom_official_split_validation_4_12.json"
BASE_CHECKPOINT = CHECKPOINT_DIR / "easycom_official_short_full_finetune_v1_stage_step200.pt"
SPATIAL_CHECKPOINT = REPO / "reports" / "easycom_gcc_bbox_speaker_pilot_v1" / "spatial_matcher.pt"
BEST_CHECKPOINT = CHECKPOINT_DIR / f"{PREFIX}_best.pt"
PREVIOUS_RECALL = {"Session_4": 0.6637168141592921, "Session_12": 0.6681818181818182}


def log(message: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with PIPELINE_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}\n")


def checkpoint_step(path: Path) -> int:
    return int(path.stem.rsplit("step", 1)[1])


def latest_interim(output: Path) -> Path | None:
    candidates = list(output.parent.glob(f"{output.stem}_epoch1_step*.pt"))
    return max(candidates, key=checkpoint_step) if candidates else None


def run_logged(name: str, arguments: list[str], stdout: Path, stderr: Path) -> None:
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
    if completed.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {completed.returncode}")


def train_stage(
    stage: str,
    resume: Path,
    target_steps: int,
    learning_rate: float,
    positive_weight: float,
    adapter: bool,
) -> Path:
    output = CHECKPOINT_DIR / f"{PREFIX}_{stage}_step{target_steps}.pt"
    if output.exists():
        log(f"skip completed training stage={stage} steps={target_steps}")
        return output
    for attempt in range(1, 4):
        interim = latest_interim(output)
        active_resume = interim or resume
        step_offset = checkpoint_step(interim) if interim is not None else 0
        remaining = target_steps - step_offset
        if remaining <= 0:
            raise RuntimeError(f"Stage {stage} has interim state but no final checkpoint.")
        arguments = [
            "easycom_saal.train",
            "--easycom-root", r"D:\EasyComDataset_reextract",
            "--manifest", str(TRAIN_MANIFEST),
            "--resume-checkpoint", str(active_resume),
            "--epochs", "1",
            "--batch-size", "4",
            "--gradient-accumulation-steps", "2",
            "--lr", str(learning_rate),
            "--num-workers", "0",
            "--no-temporal-loss",
            "--use-timeline-multilabels",
            "--lambda-heatmap", "0",
            "--lambda-candidate-target-bce", "1",
            "--lambda-candidate-speaker-bce", "0",
            "--lambda-candidate-speaker-ce", "0",
            "--lambda-vad-bce", "0",
            "--lambda-candidate-classifier-ce", "0",
            "--candidate-target-threshold", "0.5",
            "--candidate-classifier-type", "temporal",
            "--use-relation-candidate-head",
            "--use-candidate-identity-memory",
            "--use-causal-memory",
            "--use-global-address-gate",
            "--use-frame-aligned-audio",
            "--use-masked-head-video",
            "--preserve-audio-energy",
            "--streaming-smoothing-windows", "3",
            "--global-address-positive-weight", str(positive_weight),
            "--global-address-adapter-finetune" if adapter else "--global-address-finetune",
            "--max-steps", str(remaining),
            "--step-offset", str(step_offset),
            "--log-every", "20",
            "--checkpoint-every", "100",
            "--progress-path", str(LOG_DIR / f"{output.stem}.progress.json"),
            "--output", str(output),
        ]
        if interim is None:
            arguments.append("--reset-stateful-progress")
        try:
            run_logged(
                f"training stage={stage} attempt={attempt} offset={step_offset} remaining={remaining}",
                arguments,
                LOG_DIR / f"{output.stem}_train.out.log",
                LOG_DIR / f"{output.stem}_train.err.log",
            )
            return output
        except RuntimeError:
            if attempt == 3:
                raise
            time.sleep(30)
    raise RuntimeError(f"Training stage {stage} failed.")


def evaluate_stage(stage: str, checkpoint: Path) -> tuple[dict[str, Any], Path]:
    directory = REPORT_DIR / stage
    metrics_path = directory / "response_metrics.json"
    predictions_path = directory / "response_predictions.csv"
    if not (metrics_path.exists() and predictions_path.exists()):
        directory.mkdir(parents=True, exist_ok=True)
        run_logged(
            f"validation stage={stage}",
            [
                "easycom_saal.eval_response",
                "--easycom-root", r"D:\EasyComDataset_reextract",
                "--manifest", str(VALIDATION_MANIFEST),
                "--checkpoint", str(checkpoint),
                "--spatial-speaker-checkpoint", str(SPATIAL_CHECKPOINT),
                "--output-json", str(metrics_path),
                "--output-csv", str(predictions_path),
                "--batch-size", "4",
                "--num-workers", "0",
                "--log-every", "20",
            ],
            LOG_DIR / f"{PREFIX}_{stage}_validation.out.log",
            LOG_DIR / f"{PREFIX}_{stage}_validation.err.log",
        )
    return json.loads(metrics_path.read_text(encoding="utf-8")), predictions_path


def accepted(point: dict[str, Any] | None) -> bool:
    if point is None:
        return False
    return all(
        point["per_session"][session]["addressed_response_recall"] > baseline
        for session, baseline in PREVIOUS_RECALL.items()
    )


def stage_record(
    name: str,
    checkpoint: Path,
    metrics: dict[str, Any],
    predictions: Path,
) -> dict[str, Any]:
    point = select_operating_point(predictions)
    return {
        "name": name,
        "checkpoint": str(checkpoint),
        "predictions": str(predictions),
        "threshold_free_per_session": {
            session: {
                "addressed_speaker_top1": values["addressed_speaker_top1"],
                "target_auroc": values["target_auroc"],
                "speech_only_target_auroc": values["speech_only_target_auroc"],
                "vad_auroc": values["vad_auroc"],
            }
            for session, values in metrics["per_session"].items()
        },
        "operating_point": point,
        "outperforms_previous": accepted(point),
    }


def selection_key(stage: dict[str, Any]) -> tuple[float, ...]:
    point = stage["operating_point"]
    recalls = [
        point["per_session"][session]["addressed_response_recall"]
        for session in sorted(PREVIOUS_RECALL)
    ]
    correct = [
        point["per_session"][session]["addressed_correct_speaker_rate"]
        for session in sorted(PREVIOUS_RECALL)
    ]
    return min(recalls), sum(recalls), min(correct)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    log("start automatic global addressedness overnight pipeline")

    checkpoint_a = train_stage(
        "stage_a_head",
        BASE_CHECKPOINT,
        target_steps=2326,
        learning_rate=1e-4,
        positive_weight=1.0,
        adapter=False,
    )
    metrics_a, predictions_a = evaluate_stage("stage_a_head", checkpoint_a)
    stage_a = stage_record("stage_a_head", checkpoint_a, metrics_a, predictions_a)

    if stage_a["outperforms_previous"]:
        branch = "improvement_refinement"
        stage_b_steps = 600
        stage_b_lr = 3e-5
        stage_b_positive_weight = 1.15
    else:
        branch = "recovery_full_adapter"
        stage_b_steps = 2326
        stage_b_lr = 5e-5
        stage_b_positive_weight = 1.35
    log(
        f"automatic branch={branch} stage_a_outperforms={stage_a['outperforms_previous']} "
        f"stage_b_steps={stage_b_steps}"
    )

    checkpoint_b = train_stage(
        f"stage_b_{branch}",
        checkpoint_a,
        target_steps=stage_b_steps,
        learning_rate=stage_b_lr,
        positive_weight=stage_b_positive_weight,
        adapter=True,
    )
    metrics_b, predictions_b = evaluate_stage(f"stage_b_{branch}", checkpoint_b)
    stage_b = stage_record(f"stage_b_{branch}", checkpoint_b, metrics_b, predictions_b)

    eligible = [stage for stage in (stage_a, stage_b) if stage["outperforms_previous"]]
    selected = max(eligible, key=selection_key) if eligible else None
    if selected is not None:
        shutil.copy2(Path(selected["checkpoint"]), BEST_CHECKPOINT)
        point = selected["operating_point"]
        deployment = {
            "model_checkpoint": str(BEST_CHECKPOINT),
            "spatial_speaker_checkpoint": str(SPATIAL_CHECKPOINT),
            "candidate_threshold": point["candidate_threshold"],
            "vad_threshold": point["vad_threshold"],
            "decision": "VAD gate AND global addressedness gate; spatial matcher selects speaker",
            "test_sessions_used": [],
        }
        DEPLOYMENT.write_text(json.dumps(deployment, indent=2), encoding="utf-8")

    report = {
        "status": "complete_new_best" if selected is not None else "complete_no_improvement",
        "experiment": "frozen-backbone global addressedness head with automatic isolated-adapter follow-up",
        "previous_recall_per_session": PREVIOUS_RECALL,
        "acceptance": (
            "Both Session 4 and Session 12 recall strictly exceed the previous values while "
            "correct-speaker >= 0.50, NTS rejection >= 0.50, and silence rejection >= 0.85 "
            "separately on each session."
        ),
        "automatic_branch": branch,
        "stage_a": stage_a,
        "stage_b": stage_b,
        "selected_stage": selected,
        "best_checkpoint": str(BEST_CHECKPOINT) if selected is not None else None,
        "deployment_manifest": str(DEPLOYMENT) if selected is not None else None,
        "test_sessions_used": [],
    }
    SUMMARY.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"pipeline complete status={report['status']}")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
