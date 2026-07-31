from __future__ import annotations

import argparse
from collections import deque
import csv
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from easycom_saal.config import EasyComSAALConfig
from easycom_saal.datasets import EasyComAnnotations, EasyComSAALDataset, saal_collate_fn
from easycom_saal.models import SAALModel
from easycom_saal.models.spatial_speaker import load_spatial_speaker_matcher
from easycom_saal.utils.stateful_memory import IdentityStateCache, SessionStreamBatchSampler
from easycom_saal.utils.turn_graph import TurnGraphCache
from easycom_saal.metrics import binary_auroc, binary_average_precision


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate respond versus non-target EasyCom decisions")
    parser.add_argument("--easycom-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--latency-samples", type=int, default=50)
    parser.add_argument("--candidate-target-threshold", type=float, default=None)
    parser.add_argument("--vad-response-gate-threshold", type=float, default=None)
    parser.add_argument("--audio-gain-db", type=float, default=0.0)
    parser.add_argument("--audio-noise-snr-db", type=float, default=None)
    parser.add_argument("--audio-channel-gain-std-db", type=float, default=0.0)
    parser.add_argument("--audio-perturbation-seed", type=int, default=0)
    parser.add_argument("--spatial-speaker-checkpoint", default=None)
    parser.add_argument(
        "--spatial-response-gate",
        choices=("selected", "set_max"),
        default="selected",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def load_model(checkpoint_path: str, config: EasyComSAALConfig, device: torch.device) -> SAALModel:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_config = EasyComSAALConfig.from_dict(checkpoint.get("config", config.to_dict()))
    model = SAALModel(
        num_audio_channels=checkpoint_config.num_audio_channels,
        num_corr_channels=checkpoint_config.num_audio_channels * (checkpoint_config.num_audio_channels - 1) // 2,
        heatmap_size=checkpoint_config.heatmap_size,
        head_crop_feature_dim=checkpoint_config.head_crop_feature_dim,
        frame_aligned_audio=checkpoint_config.use_frame_aligned_audio,
    )
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    if any(key.startswith(("head_crop_encoder.", "head_crop_temporal_candidate_head.")) for key in missing):
        model.initialize_head_crop_branch_from_temporal()
    if any(
        key.startswith(("candidate_relation_head.input_norm.", "candidate_relation_head.temporal."))
        for key in missing
    ):
        model.initialize_relation_branch_from_temporal()
    if any(
        key.startswith(("independent_candidate_head.input_norm.", "independent_candidate_head.temporal."))
        for key in missing
    ):
        model.initialize_independent_branch_from_temporal()
    if missing:
        print(f"checkpoint missing keys initialized from scratch: {missing}", flush=True)
    if unexpected:
        print(f"checkpoint unexpected keys ignored: {unexpected}", flush=True)
    return model.to(device).eval()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _smooth_candidate_probabilities(
    candidate_ids: list[Any] | tuple[Any, ...],
    probabilities: torch.Tensor,
    history: deque[dict[Any, float]],
) -> torch.Tensor:
    current = {
        candidate_id: float(probability.item())
        for candidate_id, probability in zip(candidate_ids, probabilities)
    }
    history.append(current)
    smoothed = []
    for candidate_id in candidate_ids:
        values = [entry[candidate_id] for entry in history if candidate_id in entry]
        smoothed.append(sum(values) / len(values))
    return probabilities.new_tensor(smoothed)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    addressed = [row for row in rows if bool(row["should_respond"])]
    non_target = [row for row in rows if not bool(row["should_respond"])]
    non_target_speech = [row for row in rows if row.get("sample_type") == "non_target_speech"]
    silence = [row for row in rows if row.get("sample_type") == "silence"]
    addressed_responses = sum(bool(row["predicted_should_respond"]) for row in addressed)
    addressed_correct = sum(bool(row["correct_action"]) for row in addressed)
    non_target_correct = sum(bool(row["correct_action"]) for row in non_target)
    non_target_speech_correct = sum(bool(row["correct_action"]) for row in non_target_speech)
    silence_correct = sum(bool(row["correct_action"]) for row in silence)
    overall_correct = addressed_correct + non_target_correct
    scores = [float(row["predicted_probability"]) for row in rows]
    labels = [bool(row["should_respond"]) for row in rows]
    speech_rows = [row for row in rows if row.get("sample_type") != "silence"]
    speech_scores = [float(row["predicted_probability"]) for row in speech_rows]
    speech_labels = [bool(row["should_respond"]) for row in speech_rows]
    return {
        "samples": len(rows),
        "addressed_samples": len(addressed),
        "non_target_samples": len(non_target),
        "non_target_speech_samples": len(non_target_speech),
        "silence_samples": len(silence),
        "overall_action_accuracy": overall_correct / len(rows) if rows else float("nan"),
        "addressed_response_recall": addressed_responses / len(addressed) if addressed else float("nan"),
        "addressed_correct_speaker_rate": addressed_correct / len(addressed) if addressed else float("nan"),
        "non_target_rejection_rate": non_target_correct / len(non_target) if non_target else float("nan"),
        "non_target_false_response_rate": 1.0 - non_target_correct / len(non_target) if non_target else float("nan"),
        "non_target_speech_rejection_rate": (
            non_target_speech_correct / len(non_target_speech) if non_target_speech else float("nan")
        ),
        "silence_rejection_rate": silence_correct / len(silence) if silence else float("nan"),
        "target_detection_auroc": binary_auroc(scores, labels),
        "target_detection_average_precision": binary_average_precision(scores, labels),
        "target_detection_positive_rate": sum(labels) / len(labels) if labels else float("nan"),
        "speech_only_target_detection_auroc": binary_auroc(speech_scores, speech_labels),
        "speech_only_target_detection_average_precision": binary_average_precision(
            speech_scores,
            speech_labels,
        ),
        "speech_only_target_detection_positive_rate": (
            sum(speech_labels) / len(speech_labels) if speech_labels else float("nan")
        ),
        "counts": {
            "addressed_correct_speaker": addressed_correct,
            "addressed_wrong_speaker_or_rejected": len(addressed) - addressed_correct,
            "non_target_rejected": non_target_correct,
            "non_target_false_response": len(non_target) - non_target_correct,
            "non_target_speech_rejected": non_target_speech_correct,
            "non_target_speech_false_response": len(non_target_speech) - non_target_speech_correct,
            "silence_rejected": silence_correct,
            "silence_false_response": len(silence) - silence_correct,
        },
    }


def summarize_candidate_threshold(
    rows: list[dict[str, Any]],
    threshold: float,
    vad_gate_threshold: float | None = None,
) -> dict[str, float]:
    addressed = [row for row in rows if bool(row["should_respond"])]
    non_target = [row for row in rows if not bool(row["should_respond"])]
    non_target_speech = [row for row in rows if row.get("sample_type") == "non_target_speech"]
    silence = [row for row in rows if row.get("sample_type") == "silence"]
    def responds(row: dict[str, Any]) -> bool:
        candidate_passes = float(row["predicted_probability"]) >= threshold
        vad_probability = row.get("vad_probability")
        vad_passes = (
            vad_gate_threshold is None
            or vad_probability is None
            or float(vad_probability) >= vad_gate_threshold
        )
        return candidate_passes and vad_passes

    addressed_detected = sum(responds(row) for row in addressed)
    addressed_correct = sum(
        responds(row) and bool(row["best_candidate_correct"])
        for row in addressed
    )
    non_target_rejected = sum(not responds(row) for row in non_target)
    non_target_speech_rejected = sum(not responds(row) for row in non_target_speech)
    silence_rejected = sum(not responds(row) for row in silence)
    return {
        "threshold": threshold,
        "addressed_response_recall": addressed_detected / len(addressed) if addressed else float("nan"),
        "addressed_correct_speaker_rate": addressed_correct / len(addressed) if addressed else float("nan"),
        "non_target_rejection_rate": non_target_rejected / len(non_target) if non_target else float("nan"),
        "non_target_false_response_rate": 1.0 - non_target_rejected / len(non_target) if non_target else float("nan"),
        "non_target_speech_rejection_rate": (
            non_target_speech_rejected / len(non_target_speech) if non_target_speech else float("nan")
        ),
        "silence_rejection_rate": silence_rejected / len(silence) if silence else float("nan"),
    }


def summarize_sessions(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        session = str(row["session_id"]).split("/", 1)[0]
        grouped.setdefault(session, []).append(row)
    result: dict[str, dict[str, Any]] = {}
    for session, session_rows in sorted(grouped.items()):
        addressed = [row for row in session_rows if bool(row["should_respond"])]
        speech = [row for row in session_rows if bool(row["speech_present"])]
        vad_rows = [row for row in session_rows if row.get("vad_probability") is not None]
        result[session] = {
            "samples": len(session_rows),
            "addressed_speaker_top1": (
                sum(bool(row["speaker_best_candidate_correct"]) for row in addressed) / len(addressed)
                if addressed
                else None
            ),
            "target_auroc": binary_auroc(
                [float(row["predicted_probability"]) for row in session_rows],
                [bool(row["should_respond"]) for row in session_rows],
            ),
            "speech_only_target_auroc": binary_auroc(
                [float(row["predicted_probability"]) for row in speech],
                [bool(row["should_respond"]) for row in speech],
            ),
            "vad_auroc": (
                binary_auroc(
                    [float(row["vad_probability"]) for row in vad_rows],
                    [bool(row["speech_present"]) for row in vad_rows],
                )
                if vad_rows
                else None
            ),
        }
    return result


def _latency_summary(seconds: list[float]) -> dict[str, float | int]:
    if not seconds:
        return {"samples": 0}
    ordered = sorted(seconds)
    p95_index = min(len(ordered) - 1, max(0, int(round(0.95 * len(ordered) + 0.5)) - 1))
    return {
        "samples": len(seconds),
        "mean_ms": 1000.0 * sum(seconds) / len(seconds),
        "median_ms": 1000.0 * ordered[len(ordered) // 2],
        "p95_ms": 1000.0 * ordered[p95_index],
        "min_ms": 1000.0 * ordered[0],
        "max_ms": 1000.0 * ordered[-1],
    }


def model_compute_summary(
    model: SAALModel,
    config: EasyComSAALConfig,
    mean_candidates: float,
    checkpoint_path: str,
    use_candidate_target: bool = False,
) -> dict[str, Any]:
    def conv_out(size: int, kernel: int, stride: int, padding: int) -> int:
        return (size + 2 * padding - kernel) // stride + 1

    steps = max(1, int(round(config.clip_duration_sec * config.video_fps)))
    video_h, video_w = config.video_size
    video_h1, video_w1 = conv_out(video_h, 7, 2, 3), conv_out(video_w, 7, 2, 3)
    video_h2, video_w2 = conv_out(video_h1, 3, 2, 1), conv_out(video_w1, 3, 2, 1)
    video_macs = (
        32 * steps * video_h1 * video_w1 * (3 * 3 * 7 * 7)
        + 64 * steps * video_h2 * video_w2 * (32 * 3 * 3 * 3)
        + 128 * steps * video_h2 * video_w2 * (64 * 3 * 3 * 3)
    )

    frequency_bins = config.n_fft // 2 + 1
    time_bins = int(round(config.clip_duration_sec * config.sample_rate)) // config.hop_length + 1
    audio_h1, audio_w1 = conv_out(frequency_bins, 3, 2, 1), conv_out(time_bins, 3, 2, 1)
    audio_h2, audio_w2 = conv_out(audio_h1, 3, 2, 1), conv_out(audio_w1, 3, 2, 1)
    audio_channels = config.num_audio_channels + config.num_audio_channels * (config.num_audio_channels - 1) // 2
    audio_macs = (
        64 * audio_h1 * audio_w1 * (audio_channels * 3 * 3)
        + 128 * audio_h2 * audio_w2 * (64 * 3 * 3)
        + 128 * audio_h2 * audio_w2 * (128 * 3 * 3)
    )

    fused_h, fused_w = video_h2, video_w2
    fusion_macs = (
        steps * 128 * fused_h * fused_w * (256 * 3 * 3)
        + steps * 128 * fused_h * fused_w * (128 * 3 * 3)
        + steps * 128 * 128
    )
    candidate_input_dim = 260 + (config.head_crop_feature_dim if config.use_head_crop_branch else 0)
    candidate_macs = mean_candidates * (
        steps * 3 * (candidate_input_dim * 128 + 128 * 128)
        + 132 * 128
        + 128
    )
    relation_tokens = mean_candidates + 1.0
    relation_macs = (
        mean_candidates * steps * 3 * (260 * 128 + 128 * 128)
        + mean_candidates * 132 * 128
        + 2 * (
            4 * relation_tokens * 128 * 128
            + 2 * relation_tokens * relation_tokens * 128
            + 2 * relation_tokens * 128 * 256
        )
        + mean_candidates * 2 * 128
    )
    independent_macs = (
        mean_candidates * steps * 3 * (260 * 128 + 128 * 128)
        + mean_candidates * 132 * 128
        + mean_candidates * 2 * 128
    )
    head_crop_macs = mean_candidates * steps * (
        16 * 56 * 56 * (3 * 5 * 5)
        + 32 * 28 * 28 * (16 * 3 * 3)
        + config.head_crop_feature_dim * 14 * 14 * (32 * 3 * 3)
    )
    no_response_macs = steps * 3 * (128 * 128 + 128 * 128) + 128 * 128 + 128
    component_macs = {
        "video_encoder": video_macs,
        "audio_encoder": audio_macs,
        "fusion": fusion_macs,
        (
            "candidate_relation_head"
            if config.use_relation_candidate_head
            else "independent_candidate_head"
            if config.use_independent_candidate_head
            else "temporal_candidate_head"
        ): (
            relation_macs
            if config.use_relation_candidate_head
            else independent_macs
            if config.use_independent_candidate_head
            else candidate_macs
        ),
    }
    if config.use_head_crop_branch:
        component_macs["head_crop_encoder"] = head_crop_macs
    if not use_candidate_target:
        component_macs["no_response_head"] = no_response_macs
    total_macs = sum(component_macs.values())
    parameter_counts = {
        name: sum(parameter.numel() for parameter in getattr(model, name).parameters())
        for name in (
            "video_encoder",
            "audio_encoder",
            "fusion",
            "heatmap_head",
            "candidate_head",
            "temporal_candidate_head",
            "independent_candidate_head",
            "head_crop_encoder",
            "head_crop_temporal_candidate_head",
            "candidate_relation_head",
            "no_response_head",
        )
    }
    return {
        "parameters": {
            "total": sum(parameter.numel() for parameter in model.parameters()),
            "response_training_trainable": (
                (
                    parameter_counts["head_crop_encoder"] + parameter_counts["head_crop_temporal_candidate_head"]
                    if config.use_head_crop_branch
                    else parameter_counts["candidate_relation_head"]
                    if config.use_relation_candidate_head
                    else parameter_counts["independent_candidate_head"]
                    if config.use_independent_candidate_head
                    else parameter_counts["temporal_candidate_head"]
                )
                + (0 if use_candidate_target else parameter_counts["no_response_head"])
            ),
            "by_component": parameter_counts,
        },
        "checkpoint_size_mb": Path(checkpoint_path).stat().st_size / (1024**2),
        "one_second_window": {
            "mean_candidates": mean_candidates,
            "approximate_gmacs": total_macs / 1e9,
            "approximate_gflops": 2.0 * total_macs / 1e9,
            "component_gmacs": {name: value / 1e9 for name, value in component_macs.items()},
            "notes": "Convolution, linear, and GRU multiply-accumulates; excludes STFT, normalization, activations, pooling, and head tracking.",
        },
    }


def benchmark_streaming_latency(
    dataset: EasyComSAALDataset,
    model: SAALModel,
    device: torch.device,
    sample_count: int,
    use_candidate_target: bool = False,
) -> dict[str, Any]:
    count = min(len(dataset), max(0, int(sample_count)))
    preprocessing_times: list[float] = []
    inference_times: list[float] = []
    end_to_end_times: list[float] = []
    warmup = min(5, count)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for index in range(count):
            sample_start = time.perf_counter()
            item = dataset[index]
            preprocessing_end = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_start = time.perf_counter()
            fused_temporal_features = model.encode_temporal_features(
                item["video_clip"].unsqueeze(0).to(device),
                item["audio_spectrogram"].unsqueeze(0).to(device),
                item["audio_channel_correlation"].unsqueeze(0).to(device),
            )
            candidate_sequences = [item["candidate_bboxes_sequence_heatmap"].to(device)]
            if dataset.config.use_candidate_identity_memory:
                if dataset.config.use_turn_graph:
                    model.temporal_identity_turn_graph_relation_logits(
                        fused_temporal_features,
                        candidate_sequences,
                        [tuple(item["candidate_ids"])],
                        [
                            fused_temporal_features.new_zeros(
                                (len(item["candidate_ids"]), TurnGraphCache.feature_dim)
                            )
                        ],
                    )
                elif dataset.config.use_global_address_gate:
                    model.temporal_identity_global_relation_logits(
                        fused_temporal_features,
                        candidate_sequences,
                        [tuple(item["candidate_ids"])],
                    )
                else:
                    model.temporal_identity_memory_relation_logits(
                        fused_temporal_features,
                        candidate_sequences,
                        [tuple(item["candidate_ids"])],
                    )
            elif dataset.config.use_candidate_address_memory:
                model.temporal_memory_relation_logits(fused_temporal_features, candidate_sequences)
            elif dataset.config.use_global_address_gate:
                model.temporal_global_relation_logits(fused_temporal_features, candidate_sequences)
            elif dataset.config.use_independent_candidate_head:
                model.temporal_independent_candidate_logits(fused_temporal_features, candidate_sequences)
            elif dataset.config.use_relation_candidate_head:
                model.temporal_relation_logits(fused_temporal_features, candidate_sequences)
            elif use_candidate_target:
                if dataset.config.use_head_crop_branch:
                    model.temporal_head_crop_candidate_logits(
                        fused_temporal_features,
                        candidate_sequences,
                        [item["candidate_head_crops"].to(device)],
                    )
                else:
                    model.temporal_candidate_logits(fused_temporal_features, candidate_sequences)
            else:
                model.temporal_response_logits(fused_temporal_features, candidate_sequences)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_end = time.perf_counter()
            if index >= warmup:
                preprocessing_times.append(preprocessing_end - sample_start)
                inference_times.append(inference_end - inference_start)
                end_to_end_times.append(inference_end - sample_start)

    result = {
        "batch_size": 1,
        "warmup_samples": warmup,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "preprocessing": _latency_summary(preprocessing_times),
        "model_inference": _latency_summary(inference_times),
        "end_to_end": _latency_summary(end_to_end_times),
    }
    if device.type == "cuda":
        result["peak_cuda_memory_mb"] = torch.cuda.max_memory_allocated(device) / (1024**2)
        result["peak_cuda_reserved_mb"] = torch.cuda.max_memory_reserved(device) / (1024**2)
    return result


def evaluate(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = EasyComSAALConfig.from_dict(checkpoint.get("config", {}))
    config.easycom_root = args.easycom_root
    config.annotation_manifest = args.manifest
    config.keep_only_tos_to_camera_wearer = False
    config.use_temporal_loss = False
    config.batch_size = args.batch_size
    config.num_workers = args.num_workers
    config.audio_fixed_gain_db = float(args.audio_gain_db)
    config.audio_global_gain_augmentation_db = 0.0
    config.audio_noise_snr_db = args.audio_noise_snr_db
    config.audio_channel_gain_std_db = float(args.audio_channel_gain_std_db)
    config.audio_perturbation_seed = int(args.audio_perturbation_seed)
    use_candidate_target = (
        config.lambda_candidate_target_bce > 0.0
        or config.use_shared_response_gate
        or config.use_scene_response_head
    )
    prediction_mode = (
        'scene_response_head'
        if config.use_scene_response_head
        else 'shared_response_gate'
        if config.use_shared_response_gate
        else
        "identity_turn_graph"
        if use_candidate_target and config.use_turn_graph
        else "selected_speaker_conditional_addressing"
        if use_candidate_target and config.use_selected_speaker_addressing
        else "identity_global_response_gate"
        if use_candidate_target
        and config.use_candidate_identity_memory
        and config.use_global_address_gate
        else "candidate_target_sigmoid"
        if use_candidate_target
        else "n_plus_one_softmax"
    )
    if args.spatial_speaker_checkpoint is not None:
        prediction_mode = f"{prediction_mode}_spatial_speaker"
        if args.spatial_response_gate == "set_max":
            prediction_mode = f"{prediction_mode}_set_max_address_gate"
    elif args.spatial_response_gate != "selected":
        raise ValueError("The spatial set-level response gate requires a spatial speaker checkpoint.")
    candidate_target_threshold = (
        config.candidate_target_threshold
        if args.candidate_target_threshold is None
        else float(args.candidate_target_threshold)
    )
    vad_response_gate_threshold = (
        float(args.vad_response_gate_threshold)
        if args.vad_response_gate_threshold is not None
        else float(config.vad_threshold)
        if config.use_vad_response_gate
        else None
    )
    stateful_identity_memory = bool(
        config.use_candidate_identity_memory and config.use_causal_memory
    )
    if config.use_causal_channel_rms_normalization and config.num_workers != 0:
        raise ValueError("Causal channel RMS normalization requires --num-workers 0.")

    annotations = EasyComAnnotations.from_manifest(config.annotation_manifest, root=config.easycom_root)
    dataset = EasyComSAALDataset(
        annotations,
        config=config,
        load_media=True,
        return_previous=config.use_causal_memory and not stateful_identity_memory,
    )
    if args.max_samples is not None:
        dataset.samples = dataset.samples[: max(0, int(args.max_samples))]
    if not dataset.samples:
        raise RuntimeError("No response-evaluation samples were built.")

    stateful_batch_sampler = (
        SessionStreamBatchSampler(dataset, batch_size=config.batch_size)
        if stateful_identity_memory
        else None
    )
    dataloader = (
        DataLoader(
            dataset,
            batch_sampler=stateful_batch_sampler,
            num_workers=config.num_workers,
            collate_fn=saal_collate_fn,
            pin_memory=torch.device(args.device).type == "cuda",
            persistent_workers=config.num_workers > 0,
        )
        if stateful_batch_sampler is not None
        else DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=saal_collate_fn,
            pin_memory=torch.device(args.device).type == "cuda",
            persistent_workers=config.num_workers > 0,
        )
    )
    device = torch.device(args.device)
    model = load_model(args.checkpoint, config, device)
    spatial_speaker = (
        load_spatial_speaker_matcher(args.spatial_speaker_checkpoint, device)
        if args.spatial_speaker_checkpoint is not None
        else None
    )
    sample_counts = dataset.sample_type_counts()
    print(
        f"start response_eval samples={len(dataset)} addressed={sample_counts.get('addressed_speech', 0)} "
        f"non_target={sample_counts.get('non_target_speech', 0)} "
        f"silence={sample_counts.get('silence', 0)} device={device} "
        f"mode={prediction_mode} "
        f"threshold={candidate_target_threshold if use_candidate_target else None} "
        f"stateful_identity_memory={stateful_identity_memory}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    speaker_scores: list[float] = []
    speaker_labels_flat: list[bool] = []
    vad_scores: list[float] = []
    vad_labels: list[bool] = []
    self_speaker_scores: list[float] = []
    self_speaker_labels: list[bool] = []
    smoothing_windows = max(1, int(config.streaming_smoothing_windows))
    speaker_histories: dict[str, deque[dict[Any, float]]] = {}
    addressing_histories: dict[str, deque[dict[Any, float]]] = {}
    processed = 0
    evaluation_start = time.perf_counter()
    previous_batch_end = evaluation_start
    data_seconds = 0.0
    model_seconds = 0.0
    identity_state_cache = (
        IdentityStateCache(config.temporal_max_gap_sec)
        if stateful_identity_memory
        else None
    )
    turn_graph_cache = (
        TurnGraphCache(
            decay_seconds=config.turn_graph_decay_sec,
            reset_gap_seconds=config.turn_graph_reset_gap_sec,
            speaker_threshold=config.speaker_threshold,
        )
        if config.use_turn_graph
        else None
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for batch in dataloader:
            batch_received = time.perf_counter()
            data_seconds += batch_received - previous_batch_end
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            model_start = time.perf_counter()
            fused_temporal_features = model.encode_temporal_features(
                batch["video_clip"].to(device),
                batch["audio_spectrogram"].to(device),
                batch["audio_channel_correlation"].to(device),
            )
            candidate_sequences = [boxes.to(device) for boxes in batch["candidate_bboxes_sequence_heatmap"]]
            relation_speaking_logits = None
            global_addressing_logits = None
            vad_logits = None
            self_speaking_logits = None
            self_target_logits = None
            spatial_speaking_logits = None
            shared_response_logits = None
            scene_response_logits = None
            if config.use_candidate_identity_memory:
                previous_states: list[dict[object, torch.Tensor]] | None = None
                if stateful_identity_memory:
                    if identity_state_cache is None:
                        raise RuntimeError("Stateful identity evaluation requires an initialized state cache.")
                    previous_states = identity_state_cache.previous_states(
                        batch["session_id"],
                        batch["timestamp"],
                    )
                elif config.use_causal_memory:
                    temporal_mask = batch["apply_temporal_loss"].to(device)
                    previous_states = [{} for _ in range(fused_temporal_features.shape[0])]
                    if temporal_mask.any():
                        previous_features = model.encode_temporal_features(
                            batch["prev_video_clip"][temporal_mask.cpu()].to(device),
                            batch["prev_audio_spectrogram"][temporal_mask.cpu()].to(device),
                            batch["prev_audio_channel_correlation"][temporal_mask.cpu()].to(device),
                        )
                        valid_indices = temporal_mask.nonzero(as_tuple=False).flatten()
                        previous_boxes = [
                            batch["prev_candidate_bboxes_sequence_heatmap"][int(index.item())].to(device)
                            for index in valid_indices
                        ]
                        previous_ids = [
                            tuple(batch["prev_candidate_ids"][int(index.item())])
                            for index in valid_indices
                        ]
                        if config.use_global_address_gate:
                            _, _, _, valid_states = model.temporal_identity_global_relation_logits(
                                previous_features,
                                previous_boxes,
                                previous_ids,
                            )
                        else:
                            _, _, _, valid_states = model.temporal_identity_memory_relation_logits(
                                previous_features,
                                previous_boxes,
                                previous_ids,
                            )
                        for index, state in zip(valid_indices.tolist(), valid_states):
                            previous_states[index] = state
                if config.use_scene_response_head:
                    (
                        relation_speaking_logits,
                        conditional_addressing_logits,
                        scene_response_logits,
                        vad_logits,
                        updated_identity_states,
                    ) = model.temporal_identity_scene_response_logits(
                        fused_temporal_features,
                        candidate_sequences,
                        [tuple(ids) for ids in batch['candidate_ids']],
                        previous_states=previous_states,
                    )
                elif config.use_turn_graph:
                    if turn_graph_cache is None:
                        raise RuntimeError("Turn-graph evaluation requires an initialized graph cache.")
                    turn_features = turn_graph_cache.features(
                        batch["session_id"],
                        batch["timestamp"],
                        batch["candidate_ids"],
                        fused_temporal_features,
                    )
                    (
                        relation_speaking_logits,
                        conditional_addressing_logits,
                        vad_logits,
                        self_speaking_logits,
                        self_target_logits,
                        updated_identity_states,
                    ) = model.temporal_identity_turn_graph_relation_logits(
                        fused_temporal_features,
                        candidate_sequences,
                        [tuple(ids) for ids in batch["candidate_ids"]],
                        turn_features,
                        previous_states=previous_states,
                    )
                elif config.use_global_address_gate:
                    (
                        relation_speaking_logits,
                        global_addressing_logits,
                        vad_logits,
                        updated_identity_states,
                    ) = model.temporal_identity_global_relation_logits(
                        fused_temporal_features,
                        candidate_sequences,
                        [tuple(ids) for ids in batch["candidate_ids"]],
                        previous_states=previous_states,
                    )
                else:
                    (
                        relation_speaking_logits,
                        conditional_addressing_logits,
                        vad_logits,
                        updated_identity_states,
                    ) = model.temporal_identity_memory_relation_logits(
                        fused_temporal_features,
                        candidate_sequences,
                        [tuple(ids) for ids in batch["candidate_ids"]],
                        previous_states=previous_states,
                    )
                if identity_state_cache is not None:
                    identity_state_cache.update(
                        batch["session_id"],
                        batch["timestamp"],
                        updated_identity_states,
                    )
                if turn_graph_cache is not None:
                    if self_speaking_logits is None or self_target_logits is None:
                        raise RuntimeError("Turn-graph evaluation requires SELF prediction heads.")
                    turn_graph_cache.update(
                        batch["session_id"],
                        batch["timestamp"],
                        batch["candidate_ids"],
                        relation_speaking_logits,
                        conditional_addressing_logits,
                        self_speaking_logits,
                        self_target_logits,
                    )
                if global_addressing_logits is not None:
                    response_logits = SAALModel.combine_global_address_logits(
                        relation_speaking_logits,
                        global_addressing_logits,
                    )
                else:
                    response_logits = (
                        conditional_addressing_logits
                        if config.use_selected_speaker_addressing
                        else SAALModel.combine_vad_logits(
                            SAALModel.combine_relation_logits(
                                relation_speaking_logits,
                                conditional_addressing_logits,
                            ),
                            vad_logits,
                        )
                    )
            elif config.use_candidate_address_memory:
                previous_memory = None
                if config.use_causal_memory:
                    temporal_mask = batch["apply_temporal_loss"].to(device)
                    previous_memory = fused_temporal_features.new_zeros(
                        (fused_temporal_features.shape[0], 128)
                    )
                    if temporal_mask.any():
                        previous_features = model.encode_temporal_features(
                            batch["prev_video_clip"][temporal_mask.cpu()].to(device),
                            batch["prev_audio_spectrogram"][temporal_mask.cpu()].to(device),
                            batch["prev_audio_channel_correlation"][temporal_mask.cpu()].to(device),
                        )
                        valid_indices = temporal_mask.nonzero(as_tuple=False).flatten()
                        previous_boxes = [
                            batch["prev_candidate_bboxes_sequence_heatmap"][int(index.item())].to(device)
                            for index in valid_indices
                        ]
                        _, _, valid_memory = model.temporal_memory_relation_logits(
                            previous_features,
                            previous_boxes,
                        )
                        previous_memory = previous_memory.index_copy(0, valid_indices, valid_memory)
                (
                    relation_speaking_logits,
                    conditional_addressing_logits,
                    _,
                ) = model.temporal_memory_relation_logits(
                    fused_temporal_features,
                    candidate_sequences,
                    previous_memory=previous_memory,
                )
                response_logits = (
                    conditional_addressing_logits
                    if config.use_selected_speaker_addressing
                    else SAALModel.combine_relation_logits(
                        relation_speaking_logits,
                        conditional_addressing_logits,
                    )
                )
            elif config.use_global_address_gate:
                previous_memory = None
                if config.use_causal_memory:
                    temporal_mask = batch["apply_temporal_loss"].to(device)
                    previous_memory = fused_temporal_features.new_zeros(
                        (fused_temporal_features.shape[0], 128)
                    )
                    if temporal_mask.any():
                        previous_features = model.encode_temporal_features(
                            batch["prev_video_clip"][temporal_mask.cpu()].to(device),
                            batch["prev_audio_spectrogram"][temporal_mask.cpu()].to(device),
                            batch["prev_audio_channel_correlation"][temporal_mask.cpu()].to(device),
                        )
                        valid_indices = temporal_mask.nonzero(as_tuple=False).flatten()
                        previous_boxes = [
                            batch["prev_candidate_bboxes_sequence_heatmap"][int(index.item())].to(device)
                            for index in valid_indices
                        ]
                        _, _, valid_memory = model.temporal_global_relation_logits(
                            previous_features,
                            previous_boxes,
                        )
                        previous_memory = previous_memory.index_copy(0, valid_indices, valid_memory)
                (
                    relation_speaking_logits,
                    global_addressing_logits,
                    _,
                ) = model.temporal_global_relation_logits(
                    fused_temporal_features,
                    candidate_sequences,
                    previous_memory=previous_memory,
                )
                response_logits = SAALModel.combine_global_address_logits(
                    relation_speaking_logits,
                    global_addressing_logits,
                )
            elif config.use_independent_candidate_head:
                relation_speaking_logits, conditional_addressing_logits = (
                    model.temporal_independent_candidate_logits(
                        fused_temporal_features,
                        candidate_sequences,
                    )
                )
                response_logits = (
                    conditional_addressing_logits
                    if config.use_selected_speaker_addressing
                    else SAALModel.combine_relation_logits(
                        relation_speaking_logits,
                        conditional_addressing_logits,
                    )
                )
            elif config.use_relation_candidate_head:
                relation_speaking_logits, conditional_addressing_logits = model.temporal_relation_logits(
                    fused_temporal_features,
                    candidate_sequences,
                )
                response_logits = (
                    conditional_addressing_logits
                    if config.use_selected_speaker_addressing
                    else SAALModel.combine_relation_logits(
                        relation_speaking_logits,
                        conditional_addressing_logits,
                    )
                )
            else:
                response_logits = (
                model.temporal_head_crop_candidate_logits(
                    fused_temporal_features,
                    candidate_sequences,
                    [crops.to(device) for crops in batch["candidate_head_crops"]],
                )
                if use_candidate_target and config.use_head_crop_branch
                else model.temporal_candidate_logits(fused_temporal_features, candidate_sequences)
                if use_candidate_target
                else model.temporal_response_logits(fused_temporal_features, candidate_sequences)
                )
            if config.use_shared_response_gate:
                if vad_logits is None or conditional_addressing_logits is None:
                    raise RuntimeError('Shared response evaluation requires VAD and addressing logits.')
                maximum_address = torch.stack([values.max() for values in conditional_addressing_logits])
                candidate_counts = vad_logits.new_tensor([len(ids) for ids in batch['candidate_ids']])
                audio_waveform = batch['audio_waveform'].to(device).float()
                audio_rms = audio_waveform.square().flatten(1).mean(1).sqrt().clamp_min(1e-8)
                shared_response_logits = model.shared_response_logits(
                    vad_logits, maximum_address, candidate_counts, 20.0 * torch.log10(audio_rms),
                )
            if spatial_speaker is not None:
                spatial_speaking_logits = spatial_speaker.forward_waveform(
                    batch["audio_waveform"].to(device),
                    candidate_sequences,
                    config.heatmap_size,
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            model_seconds += time.perf_counter() - model_start
            for index, logits in enumerate(response_logits):
                candidate_ids = batch["candidate_ids"][index]
                candidate_count = len(candidate_ids)
                should_respond = bool(batch["should_respond"][index].item())
                speech_present = bool(batch["speech_present"][index].item())
                vad_probability = (
                    float(torch.sigmoid(vad_logits[index]).item()) if vad_logits is not None else None
                )
                audio_rms = batch["audio_waveform"][index].float().square().mean().sqrt().clamp_min(1e-8)
                audio_log_rms_db = float((20.0 * torch.log10(audio_rms)).item())
                self_speaking_probability = (
                    float(torch.sigmoid(self_speaking_logits[index]).item())
                    if self_speaking_logits is not None
                    else None
                )
                if self_speaking_probability is not None:
                    self_speaker_scores.append(self_speaking_probability)
                    self_speaker_labels.append(bool(batch["wearer_speaking"][index].item()))
                if vad_probability is not None:
                    vad_scores.append(vad_probability)
                    vad_labels.append(speech_present)
                response_labels = batch["response_candidate_labels"][index]
                if relation_speaking_logits is not None:
                    sample_speaker_probabilities = (
                        torch.softmax(spatial_speaking_logits[index], dim=0)
                        if spatial_speaking_logits is not None
                        else torch.sigmoid(relation_speaking_logits[index])
                    )
                    if config.use_selected_speaker_addressing:
                        session_id = str(batch["session_id"][index])
                        speaker_history = speaker_histories.setdefault(
                            session_id,
                            deque(maxlen=smoothing_windows),
                        )
                        sample_speaker_probabilities = _smooth_candidate_probabilities(
                            candidate_ids,
                            sample_speaker_probabilities,
                            speaker_history,
                        )
                    speaker_scores.extend(float(value.item()) for value in sample_speaker_probabilities)
                    speaker_labels_flat.extend(
                        bool(value.item()) for value in batch["speaker_candidate_labels"][index]
                    )
                    speaker_predicted_index = int(sample_speaker_probabilities.argmax().item())
                    selected_speaker_probability = float(
                        sample_speaker_probabilities[speaker_predicted_index].item()
                    )
                    speaker_best_candidate_correct = bool(
                        batch["speaker_candidate_labels"][index][speaker_predicted_index].item()
                    )
                else:
                    selected_speaker_probability = None
                    speaker_best_candidate_correct = None
                if use_candidate_target:
                    probabilities = torch.sigmoid(logits)
                    if config.use_selected_speaker_addressing:
                        session_id = str(batch["session_id"][index])
                        addressing_history = addressing_histories.setdefault(
                            session_id,
                            deque(maxlen=smoothing_windows),
                        )
                        probabilities = _smooth_candidate_probabilities(
                            candidate_ids,
                            probabilities,
                            addressing_history,
                        )
                    ranked_probabilities = probabilities.sort(descending=True).values
                    second_candidate_probability = (
                        float(ranked_probabilities[1].item())
                        if ranked_probabilities.numel() > 1
                        else 0.0
                    )
                    candidate_margin = float(
                        (ranked_probabilities[0] - second_candidate_probability).item()
                    )
                    predicted_index = (
                        int(sample_speaker_probabilities.argmax().item())
                        if global_addressing_logits is not None
                        or scene_response_logits is not None
                        or config.use_selected_speaker_addressing
                        else int(probabilities.argmax().item())
                    )
                    predicted_probability = (
                        float(torch.softmax(scene_response_logits[index], dim=0)[0].item())
                        if scene_response_logits is not None
                        else float(torch.sigmoid(shared_response_logits[index]).item())
                        if shared_response_logits is not None
                        else float(torch.sigmoid(global_addressing_logits[index]).item())
                        if global_addressing_logits is not None
                        else float(probabilities.max().item())
                        if spatial_speaking_logits is not None
                        and args.spatial_response_gate == "set_max"
                        else float(probabilities[predicted_index].item())
                    )
                    predicted_should_respond = predicted_probability >= candidate_target_threshold
                    predicted_candidate_id = candidate_ids[predicted_index] if predicted_should_respond else None
                    target_indices = (
                        response_labels.bool().nonzero(as_tuple=False).flatten()
                        if should_respond
                        else response_labels.new_empty((0,), dtype=torch.long)
                    )
                    best_candidate_correct = bool(
                        target_indices.numel() > 0 and response_labels.bool()[predicted_index].item()
                    )
                    correct_action = (
                        predicted_should_respond and best_candidate_correct
                        if should_respond
                        else not predicted_should_respond
                    )
                    no_response_probability = None
                else:
                    probabilities = torch.softmax(logits, dim=0)
                    candidate_probabilities = probabilities[:candidate_count]
                    ranked_probabilities = candidate_probabilities.sort(descending=True).values
                    second_candidate_probability = (
                        float(ranked_probabilities[1].item())
                        if ranked_probabilities.numel() > 1
                        else 0.0
                    )
                    candidate_margin = float(
                        (ranked_probabilities[0] - second_candidate_probability).item()
                    )
                    predicted_index = int(probabilities.argmax().item())
                    predicted_probability = float(probabilities[predicted_index].item())
                    predicted_should_respond = predicted_index < candidate_count
                    predicted_candidate_id = candidate_ids[predicted_index] if predicted_should_respond else None
                    target_indices = (
                        response_labels.bool().nonzero(as_tuple=False).flatten()
                        if should_respond
                        else response_labels.new_tensor([candidate_count], dtype=torch.long)
                    )
                    best_candidate_correct = bool(
                        should_respond
                        and predicted_index < candidate_count
                        and response_labels.bool()[predicted_index].item()
                    )
                    correct_action = bool((target_indices == predicted_index).any().item())
                    no_response_probability = float(probabilities[-1].item())
                if (
                    vad_response_gate_threshold is not None
                    and vad_probability is not None
                    and vad_probability < vad_response_gate_threshold
                ):
                    predicted_should_respond = False
                    predicted_candidate_id = None
                    correct_action = not should_respond
                rows.append(
                    {
                        "sample_index": processed,
                        "session_id": batch["session_id"][index],
                        "timestamp": float(batch["timestamp"][index].item()),
                        "sample_type": batch["sample_type"][index],
                        "source_speaker_id": batch["source_speaker_id"][index],
                        "speech_target_id": batch["speech_target_id"][index],
                        "should_respond": should_respond,
                        "speech_present": speech_present,
                        "vad_probability": vad_probability,
                        "audio_log_rms_db": audio_log_rms_db,
                        "wearer_speaking_probability": self_speaking_probability,
                        "global_address_probability": (
                            float(torch.sigmoid(global_addressing_logits[index]).item())
                            if global_addressing_logits is not None
                            else None
                        ),
                        "predicted_should_respond": predicted_should_respond,
                        "predicted_candidate_id": predicted_candidate_id,
                        "correct_action": correct_action,
                        "best_candidate_correct": best_candidate_correct,
                        "no_response_probability": no_response_probability,
                        "predicted_probability": predicted_probability,
                        "second_candidate_probability": second_candidate_probability,
                        "candidate_margin": candidate_margin,
                        "candidate_count": candidate_count,
                        "speaker_best_candidate_correct": speaker_best_candidate_correct,
                        "selected_speaker_probability": selected_speaker_probability,
                    }
                )
                processed += 1
            if processed == len(dataset) or processed % max(1, args.log_every) < config.batch_size:
                print(f"response_eval step={processed}/{len(dataset)}", flush=True)
            previous_batch_end = time.perf_counter()

    evaluation_wall_seconds = time.perf_counter() - evaluation_start
    batched_peak_allocated = (
        torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else None
    )
    batched_peak_reserved = (
        torch.cuda.max_memory_reserved(device) / (1024**2) if device.type == "cuda" else None
    )
    metrics = summarize(rows)
    if speaker_scores:
        threshold = float(config.speaker_threshold)
        predictions = [score >= threshold for score in speaker_scores]
        true_positive = sum(prediction and label for prediction, label in zip(predictions, speaker_labels_flat))
        false_positive = sum(prediction and not label for prediction, label in zip(predictions, speaker_labels_flat))
        true_negative = sum(not prediction and not label for prediction, label in zip(predictions, speaker_labels_flat))
        false_negative = sum(not prediction and label for prediction, label in zip(predictions, speaker_labels_flat))
        metrics["speaker_task"] = {
            "threshold": threshold,
            "candidate_accuracy": (true_positive + true_negative) / len(speaker_scores),
            "speaker_recall": true_positive / (true_positive + false_negative) if true_positive + false_negative else None,
            "non_speaker_rejection": true_negative / (true_negative + false_positive) if true_negative + false_positive else None,
            "best_candidate_accuracy": sum(bool(row["speaker_best_candidate_correct"]) for row in rows) / len(rows),
        }
    if vad_scores:
        threshold = float(config.vad_threshold)
        predictions = [score >= threshold for score in vad_scores]
        true_positive = sum(prediction and label for prediction, label in zip(predictions, vad_labels))
        false_positive = sum(prediction and not label for prediction, label in zip(predictions, vad_labels))
        true_negative = sum(not prediction and not label for prediction, label in zip(predictions, vad_labels))
        false_negative = sum(not prediction and label for prediction, label in zip(predictions, vad_labels))
        metrics["vad_task"] = {
            "threshold": threshold,
            "auroc": binary_auroc(vad_scores, vad_labels),
            "average_precision": binary_average_precision(vad_scores, vad_labels),
            "speech_recall": true_positive / (true_positive + false_negative) if true_positive + false_negative else None,
            "silence_rejection": true_negative / (true_negative + false_positive) if true_negative + false_positive else None,
        }
    if self_speaker_scores:
        metrics["self_speaker_task"] = {
            "auroc": binary_auroc(self_speaker_scores, self_speaker_labels),
            "average_precision": binary_average_precision(self_speaker_scores, self_speaker_labels),
        }
    metrics["prediction_mode"] = prediction_mode
    metrics["per_session"] = summarize_sessions(rows)
    metrics["vad_response_gate_threshold"] = vad_response_gate_threshold
    metrics["stateful_identity_memory"] = stateful_identity_memory
    metrics["candidate_target_threshold"] = candidate_target_threshold if use_candidate_target else None
    if use_candidate_target:
        thresholds = sorted({candidate_target_threshold, *(index / 20.0 for index in range(1, 20))})
        metrics["threshold_curve"] = [
            summarize_candidate_threshold(rows, threshold, vad_response_gate_threshold)
            for threshold in thresholds
        ]
    mean_candidates = sum(int(row["candidate_count"]) for row in rows) / len(rows)
    metrics["compute"] = model_compute_summary(
        model,
        config,
        mean_candidates,
        args.checkpoint,
        use_candidate_target=use_candidate_target,
    )
    metrics["timing"] = {
        "batched_evaluation": {
            "batch_size": config.batch_size,
            "samples": processed,
            "wall_seconds": evaluation_wall_seconds,
            "end_to_end_ms_per_sample": 1000.0 * evaluation_wall_seconds / processed,
            "data_ms_per_sample": 1000.0 * data_seconds / processed,
            "model_ms_per_sample": 1000.0 * model_seconds / processed,
            "end_to_end_samples_per_second": processed / evaluation_wall_seconds,
            "model_samples_per_second": processed / model_seconds,
            "peak_cuda_memory_mb": batched_peak_allocated,
            "peak_cuda_reserved_mb": batched_peak_reserved,
        },
        "streaming_batch1": benchmark_streaming_latency(
            dataset,
            model,
            device,
            sample_count=args.latency_samples,
            use_candidate_target=use_candidate_target,
        ),
    }
    return metrics, rows


def main() -> None:
    args = build_arg_parser().parse_args()
    metrics, rows = evaluate(args)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if args.output_csv:
        _write_csv(Path(args.output_csv), rows)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
