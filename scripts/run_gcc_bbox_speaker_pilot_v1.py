from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from easycom_saal.config import EasyComSAALConfig
from easycom_saal.datasets import EasyComAnnotations, EasyComSAALDataset
from easycom_saal.datasets.audio_features import extract_audio_clip, load_audio_6ch


ROOT = Path(r"D:\EasyComDataset_reextract")
TRAIN_MANIFEST = REPO / "manifests" / "easycom_official_split_train_1_2_3_5_6_7_8_9.json"
VALIDATION_MANIFEST = REPO / "manifests" / "easycom_official_split_validation_4_12.json"
REPORT_DIR = REPO / "reports" / "easycom_gcc_bbox_speaker_pilot_v1"
TRAIN_CACHE = REPORT_DIR / "train_features.npz"
VALIDATION_CACHE = REPORT_DIR / "validation_features.npz"
REPORT_PATH = REPORT_DIR / "selection.json"
MODEL_PATH = REPORT_DIR / "spatial_matcher.pt"

TARGET_SAMPLE_RATE = 16_000
MAX_LAG_MS = 1.0
PAIR_INDICES = tuple((left, right) for left in range(6) for right in range(left + 1, 6))
EPOCH_OPTIONS = (5, 10, 20, 40)
BASELINE_ADDRESS_TOP1 = {"Session_4": 0.5796460176991151, "Session_12": 0.85}


def signed_gcc_phat(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    """Return magnitude-invariant GCC curves with negative and positive lags."""
    if waveform.shape[0] != 6 or waveform.ndim != 2:
        raise ValueError(f"Expected [6,N] waveform, got {waveform.shape}")
    centered = waveform.astype(np.float32, copy=False) - waveform.mean(axis=1, keepdims=True)
    centered = centered * np.hanning(centered.shape[1]).astype(np.float32)[None]
    fft_size = 1 << int(math.ceil(math.log2(max(2, centered.shape[1]))))
    spectrum = np.fft.rfft(centered, n=fft_size, axis=1)
    left = spectrum[[pair[0] for pair in PAIR_INDICES]]
    right = spectrum[[pair[1] for pair in PAIR_INDICES]]
    cross_power = left * np.conj(right)
    cross_power /= np.maximum(np.abs(cross_power), 1e-8)
    correlation = np.fft.irfft(cross_power, n=fft_size, axis=1)
    max_lag = max(1, int(round(MAX_LAG_MS * sample_rate / 1000.0)))
    signed = np.concatenate(
        [correlation[:, -max_lag:], correlation[:, : max_lag + 1]],
        axis=1,
    ).astype(np.float32)
    signed /= np.maximum(np.linalg.norm(signed, axis=1, keepdims=True), 1e-6)
    return signed.reshape(-1)


def bbox_motion_features(sequence: torch.Tensor, heatmap_size: tuple[int, int]) -> np.ndarray:
    height, width = heatmap_size
    boxes = sequence.detach().cpu().numpy().astype(np.float32)
    boxes[..., [0, 2]] /= float(width)
    boxes[..., [1, 3]] /= float(height)
    centers = np.stack(
        [(boxes[..., 0] + boxes[..., 2]) * 0.5, (boxes[..., 1] + boxes[..., 3]) * 0.5],
        axis=-1,
    )
    sizes = np.stack(
        [boxes[..., 2] - boxes[..., 0], boxes[..., 3] - boxes[..., 1]],
        axis=-1,
    )
    geometry = np.concatenate([centers, sizes], axis=-1)
    middle = geometry[:, geometry.shape[1] // 2]
    mean = geometry.mean(axis=1)
    center_std = centers.std(axis=1)
    center_delta = centers[:, -1] - centers[:, 0]
    return np.concatenate([middle, mean, center_std, center_delta], axis=1).astype(np.float32)


def make_dataset(manifest: Path) -> EasyComSAALDataset:
    annotations = EasyComAnnotations.from_manifest(manifest, root=ROOT)
    config = EasyComSAALConfig(
        easycom_root=str(ROOT),
        annotation_manifest=str(manifest),
        use_timeline_multilabels=True,
        keep_only_tos_to_camera_wearer=False,
    )
    return EasyComSAALDataset(
        annotations,
        config,
        load_media=False,
        return_previous=False,
    )


def extract_features(manifest: Path, output: Path) -> None:
    started = time.perf_counter()
    dataset = make_dataset(manifest)
    records: list[dict[str, Any]] = []
    by_audio: dict[str, list[dict[str, Any]]] = {}
    for sample in dataset.samples:
        (
            _,
            candidate_sequences,
            speaker_labels,
            _,
            _,
            candidate_ids,
        ) = dataset._candidate_tensors(sample)
        if int((speaker_labels > 0.5).sum().item()) != 1:
            continue
        record = {
            "sample": sample,
            "bbox": bbox_motion_features(candidate_sequences, dataset.config.heatmap_size),
            "target": int(torch.argmax(speaker_labels).item()),
            "candidate_ids": candidate_ids,
        }
        records.append(record)
        by_audio.setdefault(sample.audio_path, []).append(record)

    audio_features: list[np.ndarray] = []
    bbox_features: list[np.ndarray] = []
    targets: list[int] = []
    candidate_counts: list[int] = []
    session_groups: list[str] = []
    recordings: list[str] = []
    timestamps: list[float] = []
    addressed: list[bool] = []
    candidate_ids_all: list[str] = []
    processed = 0
    for recording_index, (audio_path, grouped) in enumerate(sorted(by_audio.items()), start=1):
        waveform, sample_rate = load_audio_6ch(
            audio_path,
            expected_channels=6,
            target_sample_rate=TARGET_SAMPLE_RATE,
        )
        for record in grouped:
            sample = record["sample"]
            clip = extract_audio_clip(
                waveform,
                start_time=sample.clip_start_time,
                end_time=sample.clip_end_time,
                sample_rate=sample_rate,
                target_num_samples=TARGET_SAMPLE_RATE,
            )
            audio_features.append(signed_gcc_phat(clip.numpy(), sample_rate))
            bbox_features.append(record["bbox"])
            targets.append(record["target"])
            candidate_counts.append(record["bbox"].shape[0])
            session_groups.append(sample.session_id.split("/", 1)[0])
            recordings.append(sample.session_id)
            timestamps.append(float(sample.timestamp))
            addressed.append(bool(sample.should_respond))
            candidate_ids_all.append("|".join(str(value) for value in record["candidate_ids"]))
            processed += 1
        if recording_index % 20 == 0 or recording_index == len(by_audio):
            print(
                f"features recordings={recording_index}/{len(by_audio)} "
                f"windows={processed}/{len(records)} elapsed_s={time.perf_counter() - started:.1f}",
                flush=True,
            )

    max_candidates = max(candidate_counts)
    bbox_dim = bbox_features[0].shape[1]
    padded_bbox = np.zeros((len(records), max_candidates, bbox_dim), dtype=np.float32)
    candidate_mask = np.zeros((len(records), max_candidates), dtype=np.bool_)
    for index, features in enumerate(bbox_features):
        padded_bbox[index, : features.shape[0]] = features
        candidate_mask[index, : features.shape[0]] = True

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        audio=np.stack(audio_features).astype(np.float32),
        bbox=padded_bbox,
        candidate_mask=candidate_mask,
        target=np.asarray(targets, dtype=np.int64),
        session_group=np.asarray(session_groups),
        recording=np.asarray(recordings),
        timestamp=np.asarray(timestamps, dtype=np.float32),
        addressed=np.asarray(addressed, dtype=np.bool_),
        candidate_ids=np.asarray(candidate_ids_all),
    )
    print(f"saved features={output} windows={len(records)}", flush=True)


def load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


class SpatialMatcher(nn.Module):
    def __init__(self, audio_dim: int, bbox_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.audio = nn.Sequential(
            nn.Linear(audio_dim, 128),
            nn.GELU(),
            nn.Linear(128, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.candidate = nn.Sequential(
            nn.Linear(bbox_dim, 32),
            nn.GELU(),
            nn.Linear(32, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.geometry_bias = nn.Linear(bbox_dim, 1)

    def forward(
        self,
        audio: torch.Tensor,
        bbox: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        audio_embedding = self.audio(audio)
        candidate_embedding = self.candidate(bbox)
        scores = torch.einsum("bd,bkd->bk", audio_embedding, candidate_embedding)
        scores = scores / math.sqrt(float(audio_embedding.shape[-1]))
        scores = scores + self.geometry_bias(bbox).squeeze(-1)
        return scores.masked_fill(~candidate_mask, -1e4)


def prepare_tensors(
    data: dict[str, np.ndarray],
    audio_mean: np.ndarray,
    audio_std: np.ndarray,
    device: torch.device,
    geometry_only: bool = False,
) -> dict[str, Any]:
    audio = (data["audio"] - audio_mean) / audio_std
    if geometry_only:
        audio = np.zeros_like(audio)
    return {
        "audio": torch.from_numpy(audio.astype(np.float32)).to(device),
        "bbox": torch.from_numpy(data["bbox"]).to(device),
        "candidate_mask": torch.from_numpy(data["candidate_mask"]).to(device),
        "target": torch.from_numpy(data["target"]).to(device),
        "session_group": data["session_group"],
        "recording": data["recording"],
        "addressed": data["addressed"],
    }


@torch.no_grad()
def accuracy(
    model: SpatialMatcher,
    data: dict[str, Any],
    indices: np.ndarray,
) -> dict[str, float]:
    selected = torch.from_numpy(indices).to(data["audio"].device)
    scores = model(
        data["audio"].index_select(0, selected),
        data["bbox"].index_select(0, selected),
        data["candidate_mask"].index_select(0, selected),
    )
    correct = scores.argmax(dim=1).eq(data["target"].index_select(0, selected)).cpu().numpy()
    addressed = data["addressed"][indices]
    return {
        "all_speech_top1": float(correct.mean()),
        "addressed_speaker_top1": float(correct[addressed].mean()) if addressed.any() else float("nan"),
        "windows": int(len(indices)),
        "addressed_windows": int(addressed.sum()),
    }


def train_snapshots(
    data: dict[str, Any],
    train_indices: np.ndarray,
    epochs: tuple[int, ...],
    seed: int,
) -> dict[int, dict[str, torch.Tensor]]:
    torch.manual_seed(seed)
    model = SpatialMatcher(data["audio"].shape[1], data["bbox"].shape[2]).to(data["audio"].device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    group_names = data["session_group"][train_indices]
    unique, counts = np.unique(group_names, return_counts=True)
    group_weights = {group: 1.0 / count for group, count in zip(unique, counts)}
    weights = torch.tensor(
        [group_weights[group] for group in group_names],
        dtype=torch.float32,
        device=data["audio"].device,
    )
    train_tensor = torch.from_numpy(train_indices).to(data["audio"].device)
    batch_size = min(256, len(train_indices))
    steps_per_epoch = max(1, math.ceil(len(train_indices) / batch_size))
    snapshots: dict[int, dict[str, torch.Tensor]] = {}
    model.train()
    for epoch in range(1, max(epochs) + 1):
        for _ in range(steps_per_epoch):
            positions = torch.multinomial(weights, batch_size, replacement=True)
            selected = train_tensor.index_select(0, positions)
            scores = model(
                data["audio"].index_select(0, selected),
                data["bbox"].index_select(0, selected),
                data["candidate_mask"].index_select(0, selected),
            )
            loss = F.cross_entropy(scores, data["target"].index_select(0, selected))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if epoch in epochs:
            snapshots[epoch] = copy.deepcopy(model.state_dict())
    return snapshots


def evaluate_loso(
    train: dict[str, Any],
    device: torch.device,
) -> tuple[int, dict[str, Any]]:
    groups = sorted(np.unique(train["session_group"]))
    by_epoch: dict[int, dict[str, dict[str, float]]] = {epoch: {} for epoch in EPOCH_OPTIONS}
    for fold, held_out in enumerate(groups):
        train_indices = np.flatnonzero(train["session_group"] != held_out)
        held_indices = np.flatnonzero(train["session_group"] == held_out)
        snapshots = train_snapshots(train, train_indices, EPOCH_OPTIONS, seed=20260728 + fold)
        model = SpatialMatcher(train["audio"].shape[1], train["bbox"].shape[2]).to(device).eval()
        for epoch, state in snapshots.items():
            model.load_state_dict(state)
            by_epoch[epoch][held_out] = accuracy(model, train, held_indices)
        print(f"loso held_out={held_out} done", flush=True)

    ranking = []
    for epoch, per_group in by_epoch.items():
        addressed_scores = [metrics["addressed_speaker_top1"] for metrics in per_group.values()]
        all_scores = [metrics["all_speech_top1"] for metrics in per_group.values()]
        ranking.append(
            (
                float(np.mean(addressed_scores)),
                float(np.min(addressed_scores)),
                float(np.mean(all_scores)),
                epoch,
            )
        )
    selected_epoch = max(ranking)[-1]
    report = {
        "selected_epoch": selected_epoch,
        "selection": "maximize mean held-session addressed top1; tie-break worst session then all speech",
        "by_epoch": {
            str(epoch): {
                "mean_addressed_top1": float(
                    np.mean([value["addressed_speaker_top1"] for value in per_group.values()])
                ),
                "worst_addressed_top1": float(
                    np.min([value["addressed_speaker_top1"] for value in per_group.values()])
                ),
                "mean_all_speech_top1": float(
                    np.mean([value["all_speech_top1"] for value in per_group.values()])
                ),
                "per_session": per_group,
            }
            for epoch, per_group in by_epoch.items()
        },
    }
    return selected_epoch, report


def validation_report(
    model: SpatialMatcher,
    validation: dict[str, Any],
) -> dict[str, Any]:
    per_session = {
        session: accuracy(
            model,
            validation,
            np.flatnonzero(validation["session_group"] == session),
        )
        for session in sorted(np.unique(validation["session_group"]))
    }
    per_recording = {
        recording: accuracy(
            model,
            validation,
            np.flatnonzero(validation["recording"] == recording),
        )
        for recording in sorted(np.unique(validation["recording"]))
    }
    return {"per_session": per_session, "per_recording": per_recording}


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if not TRAIN_CACHE.exists():
        extract_features(TRAIN_MANIFEST, TRAIN_CACHE)
    if not VALIDATION_CACHE.exists():
        extract_features(VALIDATION_MANIFEST, VALIDATION_CACHE)

    raw_train = load_cache(TRAIN_CACHE)
    raw_validation = load_cache(VALIDATION_CACHE)
    audio_mean = raw_train["audio"].mean(axis=0, keepdims=True)
    audio_std = raw_train["audio"].std(axis=0, keepdims=True).clip(min=1e-4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = prepare_tensors(raw_train, audio_mean, audio_std, device)
    validation = prepare_tensors(raw_validation, audio_mean, audio_std, device)

    selected_epoch, loso = evaluate_loso(train, device)
    snapshots = train_snapshots(
        train,
        np.arange(len(raw_train["target"])),
        (selected_epoch,),
        seed=20260728,
    )
    model = SpatialMatcher(train["audio"].shape[1], train["bbox"].shape[2]).to(device).eval()
    model.load_state_dict(snapshots[selected_epoch])
    validation_metrics = validation_report(model, validation)
    torch.save(
        {
            "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "audio_mean": torch.from_numpy(audio_mean.astype(np.float32)),
            "audio_std": torch.from_numpy(audio_std.astype(np.float32)),
            "audio_dim": int(raw_train["audio"].shape[1]),
            "bbox_dim": int(raw_train["bbox"].shape[2]),
            "sample_rate": TARGET_SAMPLE_RATE,
            "max_lag_ms": MAX_LAG_MS,
            "selected_epoch": selected_epoch,
        },
        MODEL_PATH,
    )

    geometry_train = prepare_tensors(raw_train, audio_mean, audio_std, device, geometry_only=True)
    geometry_validation = prepare_tensors(
        raw_validation,
        audio_mean,
        audio_std,
        device,
        geometry_only=True,
    )
    geometry_state = train_snapshots(
        geometry_train,
        np.arange(len(raw_train["target"])),
        (selected_epoch,),
        seed=20260728,
    )[selected_epoch]
    geometry_model = SpatialMatcher(
        geometry_train["audio"].shape[1],
        geometry_train["bbox"].shape[2],
    ).to(device).eval()
    geometry_model.load_state_dict(geometry_state)
    geometry_metrics = validation_report(geometry_model, geometry_validation)

    passed_gate = (
        validation_metrics["per_session"]["Session_4"]["addressed_speaker_top1"] >= 0.65
        and validation_metrics["per_session"]["Session_12"]["addressed_speaker_top1"] >= 0.80
    )
    report = {
        "status": "passed_integration_gate" if passed_gate else "stopped_before_full_integration",
        "method": "signed GCC-PHAT audio direction to candidate bbox compatibility",
        "test_sessions_used": [],
        "feature": {
            "sample_rate": TARGET_SAMPLE_RATE,
            "max_lag_ms": MAX_LAG_MS,
            "microphone_pairs": len(PAIR_INDICES),
            "audio_dimension": int(raw_train["audio"].shape[1]),
            "bbox_dimension": int(raw_train["bbox"].shape[2]),
        },
        "train_windows": int(len(raw_train["target"])),
        "validation_windows": int(len(raw_validation["target"])),
        "loso": loso,
        "validation": validation_metrics,
        "geometry_only_validation": geometry_metrics,
        "current_model_addressed_top1": BASELINE_ADDRESS_TOP1,
        "integration_gate": {
            "Session_4_min_addressed_top1": 0.65,
            "Session_12_min_addressed_top1": 0.80,
            "passed": passed_gate,
        },
        "spatial_matcher_checkpoint": str(MODEL_PATH),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "selected_epoch": selected_epoch,
        "validation": validation_metrics["per_session"],
        "geometry_only": geometry_metrics["per_session"],
        "report": str(REPORT_PATH),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
