from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from easycom_saal.config import EasyComSAALConfig
from easycom_saal.datasets import EasyComAnnotations, EasyComSAALDataset, saal_collate_fn
from easycom_saal.models import SAALModel
from easycom_saal.utils.stateful_memory import IdentityStateCache, SessionStreamBatchSampler
from easycom_saal.utils.turn_graph import TurnGraphCache


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train SAAL-style EasyCom heatmap model")
    parser.add_argument("--easycom-root", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--output", type=str, default="checkpoints/easycom_saal.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-temporal-loss", action="store_true")
    parser.add_argument("--include-non-target-speech", action="store_true")
    parser.add_argument("--use-timeline-multilabels", action="store_true")
    parser.add_argument("--label-min-overlap-sec", type=float, default=None)
    parser.add_argument("--exclude-silence-windows", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--progress-path", type=str, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--resume-checkpoint", type=str, default=None)
    parser.add_argument("--reset-stateful-progress", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--step-offset", type=int, default=0)
    parser.add_argument("--sample-start", type=int, default=None)
    parser.add_argument("--sample-end", type=int, default=None)
    parser.add_argument("--sample-order-seed", type=int, default=0)
    parser.add_argument("--lambda-candidate-ranking", type=float, default=None)
    parser.add_argument("--lambda-candidate-ce", type=float, default=None)
    parser.add_argument("--lambda-candidate-classifier-ce", type=float, default=None)
    parser.add_argument("--lambda-candidate-target-bce", type=float, default=None)
    parser.add_argument("--lambda-candidate-speaker-bce", type=float, default=None)
    parser.add_argument("--lambda-candidate-speaker-ce", type=float, default=None)
    parser.add_argument("--weight-speaker-ce-by-candidate-count", action="store_true")
    parser.add_argument("--lambda-vad-bce", type=float, default=None)
    parser.add_argument("--candidate-target-threshold", type=float, default=None)
    parser.add_argument("--vad-threshold", type=float, default=None)
    parser.add_argument("--lambda-response-ce", type=float, default=None)
    parser.add_argument("--response-addressed-weight", type=float, default=None)
    parser.add_argument("--lambda-heatmap", type=float, default=None)
    parser.add_argument("--candidate-ranking-margin", type=float, default=None)
    parser.add_argument("--candidate-ce-temperature", type=float, default=None)
    parser.add_argument("--candidate-topk-fraction", type=float, default=None)
    parser.add_argument("--candidate-topk-min-pixels", type=int, default=None)
    parser.add_argument("--candidate-classifier-type", choices=("spatial", "temporal"), default=None)
    parser.add_argument("--freeze-base-for-candidate-classifier", action="store_true")
    parser.add_argument("--unfreeze-fusion-for-candidate-classifier", action="store_true")
    parser.add_argument("--unfreeze-video-last-block-for-candidate-classifier", action="store_true")
    parser.add_argument("--unfreeze-audio-encoder-for-candidate-classifier", action="store_true")
    parser.add_argument("--fusion-lr", type=float, default=None)
    parser.add_argument("--video-last-block-lr", type=float, default=None)
    parser.add_argument("--audio-encoder-lr", type=float, default=None)
    parser.add_argument("--use-head-crop-branch", action="store_true")
    parser.add_argument("--use-relation-candidate-head", action="store_true")
    parser.add_argument("--use-independent-candidate-head", action="store_true")
    parser.add_argument("--use-global-address-gate", action="store_true")
    parser.add_argument("--use-selected-speaker-addressing", action="store_true")
    parser.add_argument("--use-candidate-address-memory", action="store_true")
    parser.add_argument("--use-candidate-identity-memory", action="store_true")
    parser.add_argument("--identity-bptt-windows", type=int, default=None)
    parser.add_argument("--use-turn-graph", action="store_true")
    parser.add_argument("--turn-graph-decay-sec", type=float, default=None)
    parser.add_argument("--turn-graph-reset-gap-sec", type=float, default=None)
    parser.add_argument("--turn-graph-teacher-forcing", type=float, default=None)
    parser.add_argument("--turn-graph-other-only", action="store_true")
    parser.add_argument("--lambda-self-speaker-bce", type=float, default=None)
    parser.add_argument("--lambda-self-target-bce", type=float, default=None)
    parser.add_argument("--use-causal-memory", action="store_true")
    parser.add_argument("--use-frame-aligned-audio", action="store_true")
    parser.add_argument("--preserve-audio-energy", action="store_true")
    parser.add_argument("--audio-global-gain-augmentation-db", type=float, default=None)
    parser.add_argument("--audio-channel-gain-std-db", type=float, default=None)
    parser.add_argument("--audio-channel-gain-per-recording", action="store_true")
    parser.add_argument("--audio-perturbation-seed", type=int, default=None)
    parser.add_argument("--use-causal-channel-rms-normalization", action="store_true")
    parser.add_argument("--audio-normalization-tau-sec", type=float, default=None)
    parser.add_argument("--audio-normalization-target-rms-db", type=float, default=None)
    parser.add_argument("--audio-normalization-max-gain-db", type=float, default=None)
    parser.add_argument("--audio-robustness-heads-only", action="store_true")
    parser.add_argument("--speaker-representation-finetune", action="store_true")
    parser.add_argument("--speaker-residual-finetune", action="store_true")
    parser.add_argument("--global-address-finetune", action="store_true")
    parser.add_argument("--global-address-adapter-finetune", action="store_true")
    parser.add_argument("--global-address-positive-weight", type=float, default=None)
    parser.add_argument("--addressing-hard-negative-finetune", action="store_true")
    parser.add_argument("--addressing-negative-weight", type=float, default=None)
    parser.add_argument("--addressing-hard-negative-fraction", type=float, default=None)
    parser.add_argument("--addressing-hard-negative-weight", type=float, default=None)
    parser.add_argument("--relation-adaptation-finetune", action="store_true")
    parser.add_argument("--relation-aux-lr", type=float, default=None)
    parser.add_argument("--use-masked-head-video", action="store_true")
    parser.add_argument("--streaming-smoothing-windows", type=int, default=None)
    parser.add_argument('--lambda-shared-response-bce', type=float, default=None)
    parser.add_argument('--use-shared-response-gate', action='store_true')
    parser.add_argument('--shared-response-finetune', action='store_true')
    parser.add_argument('--session-category-balanced-response-loss', action='store_true')
    parser.add_argument('--lambda-scene-response-ce', type=float, default=None)
    parser.add_argument('--use-scene-response-head', action='store_true')
    parser.add_argument('--scene-response-finetune', action='store_true')
    return parser


def heatmap_loss(pred: torch.Tensor, target: torch.Tensor, loss_name: str) -> torch.Tensor:
    if target.ndim == 3:
        target = target.unsqueeze(1)
    if loss_name == "mse":
        pred_prob = torch.sigmoid(pred)
        return F.mse_loss(pred_prob, target)
    if loss_name == "bce":
        return F.binary_cross_entropy_with_logits(pred, target)
    raise ValueError(f"Unsupported heatmap loss: {loss_name}")


def temporal_loss(pred: torch.Tensor, pred_prev: torch.Tensor, loss_name: str) -> torch.Tensor:
    pred = torch.sigmoid(pred)
    pred_prev = torch.sigmoid(pred_prev)
    if loss_name == "l1":
        return F.l1_loss(pred, pred_prev)
    if loss_name == "mse":
        return F.mse_loss(pred, pred_prev)
    raise ValueError(f"Unsupported temporal loss: {loss_name}")


def candidate_box_score(
    heatmap: torch.Tensor,
    bbox: torch.Tensor,
    topk_fraction: float,
    min_pixels: int,
) -> torch.Tensor:
    height, width = heatmap.shape
    x1, y1, x2, y2 = bbox.unbind()
    x_start = int(torch.floor(torch.minimum(x1, x2)).clamp(0, width - 1).item())
    y_start = int(torch.floor(torch.minimum(y1, y2)).clamp(0, height - 1).item())
    x_end = int(torch.ceil(torch.maximum(x1, x2)).clamp(x_start + 1, width).item())
    y_end = int(torch.ceil(torch.maximum(y1, y2)).clamp(y_start + 1, height).item())
    region = heatmap[y_start:y_end, x_start:x_end].flatten()
    if region.numel() == 0:
        return heatmap.new_tensor(0.0)
    k = max(int(min_pixels), int(round(region.numel() * float(topk_fraction))))
    k = max(1, min(k, region.numel()))
    return region.topk(k).values.mean()


def candidate_ranking_loss(
    pred: torch.Tensor,
    candidate_bboxes: list[torch.Tensor],
    candidate_labels: list[torch.Tensor],
    margin: float,
    topk_fraction: float,
    min_pixels: int,
) -> torch.Tensor:
    pred_prob = torch.sigmoid(pred)
    if pred_prob.ndim == 4:
        pred_prob = pred_prob[:, 0]

    losses = []
    for heatmap, boxes, labels in zip(pred_prob, candidate_bboxes, candidate_labels):
        if boxes.numel() == 0 or labels.numel() == 0:
            continue
        labels = labels.bool()
        if not labels.any() or labels.all():
            continue
        scores = torch.stack(
            [candidate_box_score(heatmap, box, topk_fraction, min_pixels) for box in boxes],
        )
        positive_scores = scores[labels]
        negative_scores = scores[~labels]
        losses.append(F.relu(float(margin) - positive_scores[:, None] + negative_scores[None, :]).mean())

    if not losses:
        return pred.new_tensor(0.0)
    return torch.stack(losses).mean()


def candidate_softmax_ce_loss(
    pred: torch.Tensor,
    candidate_bboxes: list[torch.Tensor],
    candidate_labels: list[torch.Tensor],
    topk_fraction: float,
    min_pixels: int,
    temperature: float,
) -> torch.Tensor:
    pred_prob = torch.sigmoid(pred)
    if pred_prob.ndim == 4:
        pred_prob = pred_prob[:, 0]

    losses = []
    temperature = max(float(temperature), 1e-6)
    for heatmap, boxes, labels in zip(pred_prob, candidate_bboxes, candidate_labels):
        if boxes.numel() == 0 or labels.numel() == 0:
            continue
        labels = labels.bool()
        if boxes.shape[0] < 2 or labels.sum().item() != 1:
            continue
        scores = torch.stack(
            [candidate_box_score(heatmap, box, topk_fraction, min_pixels) for box in boxes],
        )
        target = labels.nonzero(as_tuple=False).flatten()
        losses.append(F.cross_entropy((scores / temperature).unsqueeze(0), target[:1]))

    if not losses:
        return pred.new_tensor(0.0)
    return torch.stack(losses).mean()


def candidate_classifier_ce_loss(
    candidate_logits: list[torch.Tensor],
    candidate_labels: list[torch.Tensor],
) -> torch.Tensor:
    losses = []
    for logits, labels in zip(candidate_logits, candidate_labels):
        if logits.numel() == 0 or labels.numel() == 0:
            continue
        labels = labels.bool()
        if logits.shape[0] < 2 or labels.sum().item() != 1:
            continue
        target = labels.nonzero(as_tuple=False).flatten()[:1]
        losses.append(F.cross_entropy(logits.unsqueeze(0), target))

    if not losses:
        if candidate_logits and any(logits.numel() > 0 for logits in candidate_logits):
            return sum(logits.sum() * 0.0 for logits in candidate_logits if logits.numel() > 0)
        if candidate_logits:
            return candidate_logits[0].sum() * 0.0
        return torch.tensor(0.0)
    return torch.stack(losses).mean()


def candidate_speaker_ce_loss(
    speaking_logits: list[torch.Tensor],
    speaker_candidate_labels: list[torch.Tensor],
    weight_by_candidate_count: bool = False,
) -> torch.Tensor:
    """Categorical source-speaker supervision for single-speaker windows."""
    losses = []
    weights = []
    for logits, labels in zip(speaking_logits, speaker_candidate_labels):
        if logits.numel() < 2 or labels.numel() == 0:
            continue
        labels = labels.bool()
        if labels.sum().item() != 1:
            continue
        target = labels.nonzero(as_tuple=False).flatten()[:1]
        losses.append(F.cross_entropy(logits.unsqueeze(0), target))
        weights.append(float(max(1, logits.numel() - 1)) if weight_by_candidate_count else 1.0)

    if not losses:
        if speaking_logits:
            return sum(logits.sum() * 0.0 for logits in speaking_logits)
        return torch.tensor(0.0)
    loss_tensor = torch.stack(losses)
    weight_tensor = loss_tensor.new_tensor(weights)
    return (loss_tensor * weight_tensor).sum() / weight_tensor.sum()


def candidate_classifier_top1(
    candidate_logits: list[torch.Tensor],
    candidate_labels: list[torch.Tensor],
) -> float | None:
    correct = 0
    total = 0
    for logits, labels in zip(candidate_logits, candidate_labels):
        if logits.numel() == 0 or labels.numel() == 0:
            continue
        labels = labels.bool()
        if logits.shape[0] < 2 or labels.sum().item() != 1:
            continue
        pred_index = int(logits.argmax().detach().cpu().item())
        correct += int(bool(labels[pred_index].detach().cpu().item()))
        total += 1
    if total == 0:
        return None
    return correct / total


def candidate_target_bce_loss(
    candidate_logits: list[torch.Tensor],
    response_candidate_labels: list[torch.Tensor],
) -> torch.Tensor:
    """Balanced per-sample BCE for independent candidate target decisions."""
    losses = []
    for logits, labels in zip(candidate_logits, response_candidate_labels):
        if logits.numel() == 0 or labels.numel() == 0:
            continue
        if logits.shape != labels.shape:
            raise ValueError(f"Candidate logits {tuple(logits.shape)} do not match labels {tuple(labels.shape)}")
        positive = labels.bool()
        positive_loss = F.softplus(-logits[positive]).mean() if positive.any() else None
        negative_loss = F.softplus(logits[~positive]).mean() if (~positive).any() else None
        if positive_loss is not None and negative_loss is not None:
            losses.append(0.5 * (positive_loss + negative_loss))
        elif positive_loss is not None:
            losses.append(positive_loss)
        elif negative_loss is not None:
            losses.append(negative_loss)

    if not losses:
        if candidate_logits:
            return sum(logits.sum() * 0.0 for logits in candidate_logits)
        return torch.tensor(0.0)
    return torch.stack(losses).mean()


def conditional_addressing_bce_loss(
    addressing_logits: list[torch.Tensor],
    response_candidate_labels: list[torch.Tensor],
    speaker_candidate_labels: list[torch.Tensor],
    negative_weight: float = 1.0,
    hard_negative_fraction: float = 0.0,
    hard_negative_weight: float = 1.0,
) -> torch.Tensor:
    """Balanced BCE over active speakers only: P(addressing robot | speaking)."""
    active_logits = []
    active_labels = []
    for logits, response_labels, speaker_labels in zip(
        addressing_logits,
        response_candidate_labels,
        speaker_candidate_labels,
    ):
        if logits.shape != response_labels.shape or logits.shape != speaker_labels.shape:
            raise ValueError("Addressing logits, response labels, and speaker labels must have matching shapes.")
        active = speaker_labels.bool()
        if active.any():
            active_logits.append(logits[active])
            active_labels.append(response_labels[active].bool())

    if not active_logits:
        return sum(logits.sum() * 0.0 for logits in addressing_logits)
    logits = torch.cat(active_logits)
    labels = torch.cat(active_labels)
    positive_loss = F.softplus(-logits[labels]).mean() if labels.any() else None
    negative_loss = None
    if (~labels).any():
        negative_logits = logits[~labels]
        negative_losses = F.softplus(negative_logits)
        weights = torch.ones_like(negative_losses)
        if hard_negative_fraction > 0.0 and hard_negative_weight > 1.0:
            hard_count = max(
                1,
                int(torch.ceil(
                    negative_logits.new_tensor(negative_logits.numel() * hard_negative_fraction)
                ).item()),
            )
            hard_count = min(hard_count, negative_logits.numel())
            hard_indices = torch.topk(negative_logits.detach(), hard_count).indices
            weights[hard_indices] = hard_negative_weight
        negative_loss = (negative_losses * weights).sum() / weights.sum().clamp_min(1.0)
    if positive_loss is not None and negative_loss is not None:
        return (positive_loss + negative_weight * negative_loss) / (1.0 + negative_weight)
    return positive_loss if positive_loss is not None else negative_loss


def balanced_binary_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    positive_weight: float = 1.0,
) -> torch.Tensor:
    labels = labels.bool()
    positive_loss = F.softplus(-logits[labels]).mean() if labels.any() else None
    negative_loss = F.softplus(logits[~labels]).mean() if (~labels).any() else None
    if positive_loss is not None and negative_loss is not None:
        return (float(positive_weight) * positive_loss + negative_loss) / (
            float(positive_weight) + 1.0
        )
    return positive_loss if positive_loss is not None else negative_loss


def session_class_balanced_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    session_ids: list[str],
) -> torch.Tensor:
    labels = labels.bool()
    session_groups = [str(value).split('/', 1)[0] for value in session_ids]
    losses = []
    for session_group in sorted(set(session_groups)):
        mask = torch.tensor(
            [value == session_group for value in session_groups],
            device=logits.device,
            dtype=torch.bool,
        )
        group_logits = logits[mask]
        group_labels = labels[mask]
        losses.append(balanced_binary_bce_loss(group_logits, group_labels))
    return torch.stack(losses).mean()


def session_category_balanced_response_bce_loss(
    logits: torch.Tensor, labels: torch.Tensor, session_ids: list[str], sample_types: list[str],
) -> torch.Tensor:
    labels = labels.float()
    groups = [str(value).split('/', 1)[0] for value in session_ids]
    losses = []
    for group in sorted(set(groups)):
        for category in ('addressed_speech', 'non_target_speech', 'silence'):
            mask = torch.tensor(
                [g == group and t == category for g, t in zip(groups, sample_types)],
                device=logits.device, dtype=torch.bool,
            )
            if mask.any():
                losses.append(F.binary_cross_entropy_with_logits(logits[mask], labels[mask]))
    return torch.stack(losses).mean()


def session_category_balanced_scene_ce_loss(
    logits: torch.Tensor, session_ids: list[str], sample_types: list[str],
) -> torch.Tensor:
    category_ids = {'addressed_speech': 0, 'non_target_speech': 1, 'silence': 2}
    targets = torch.tensor([category_ids[value] for value in sample_types], device=logits.device)
    groups = [str(value).split('/', 1)[0] for value in session_ids]
    losses = []
    for group in sorted(set(groups)):
        for category, category_id in category_ids.items():
            mask = torch.tensor(
                [g == group and t == category for g, t in zip(groups, sample_types)],
                device=logits.device, dtype=torch.bool,
            )
            if mask.any():
                losses.append(F.cross_entropy(logits[mask], targets[mask]))
    return torch.stack(losses).mean()


def shared_response_gate_loss(model: SAALModel, vad_logits: torch.Tensor, address_logits: list[torch.Tensor], batch: dict[str, Any]) -> torch.Tensor:
    maximum_address = torch.stack([values.max() for values in address_logits])
    candidate_counts = vad_logits.new_tensor([len(ids) for ids in batch['candidate_ids']])
    audio_rms = batch['audio_waveform'].float().square().flatten(1).mean(1).sqrt().clamp_min(1e-8)
    audio_log_rms_db = 20.0 * torch.log10(audio_rms)
    logits = model.shared_response_logits(vad_logits.detach(), maximum_address.detach(), candidate_counts, audio_log_rms_db)
    return session_class_balanced_bce_loss(logits, batch['should_respond'], batch['session_id'])


def wearer_target_bce_loss(
    candidate_logits: list[torch.Tensor],
    candidate_labels: list[torch.Tensor],
    wearer_speaking: torch.Tensor,
) -> torch.Tensor:
    selected_logits = []
    selected_labels = []
    for logits, labels, speaking in zip(candidate_logits, candidate_labels, wearer_speaking):
        if bool(speaking.item()) and labels.bool().any():
            selected_logits.append(logits)
            selected_labels.append(labels)
    if selected_logits:
        return candidate_target_bce_loss(selected_logits, selected_labels)
    if candidate_logits:
        return sum(logits.sum() * 0.0 for logits in candidate_logits)
    return wearer_speaking.float().sum() * 0.0


def candidate_binary_metrics(
    candidate_logits: list[torch.Tensor],
    candidate_labels: list[torch.Tensor],
    threshold: float = 0.5,
) -> dict[str, float | None]:
    true_positive = false_positive = true_negative = false_negative = 0
    for logits, labels in zip(candidate_logits, candidate_labels):
        if logits.numel() == 0:
            continue
        predictions = torch.sigmoid(logits) >= threshold
        labels = labels.bool()
        true_positive += int((predictions & labels).sum().item())
        false_positive += int((predictions & ~labels).sum().item())
        true_negative += int((~predictions & ~labels).sum().item())
        false_negative += int((~predictions & labels).sum().item())
    total = true_positive + false_positive + true_negative + false_negative
    return {
        "accuracy": (true_positive + true_negative) / total if total else None,
        "recall": true_positive / (true_positive + false_negative) if true_positive + false_negative else None,
        "specificity": true_negative / (true_negative + false_positive) if true_negative + false_positive else None,
    }


def candidate_target_metrics(
    candidate_logits: list[torch.Tensor],
    response_candidate_labels: list[torch.Tensor],
    should_respond: torch.Tensor,
    threshold: float,
) -> dict[str, float | None]:
    addressed_detected = 0
    addressed_correct = 0
    addressed_total = 0
    non_target_rejected = 0
    non_target_total = 0
    for logits, labels, respond in zip(candidate_logits, response_candidate_labels, should_respond):
        if logits.numel() == 0:
            continue
        probabilities = torch.sigmoid(logits)
        predicted_index = int(probabilities.argmax().detach().cpu().item())
        predicted_should_respond = float(probabilities[predicted_index].detach().cpu().item()) >= float(threshold)
        if bool(respond.item()):
            labels = labels.bool()
            if not labels.any():
                continue
            addressed_total += 1
            addressed_detected += int(predicted_should_respond)
            addressed_correct += int(predicted_should_respond and bool(labels[predicted_index].item()))
        else:
            non_target_total += 1
            non_target_rejected += int(not predicted_should_respond)

    total = addressed_total + non_target_total
    return {
        "top1": (addressed_correct + non_target_rejected) / total if total else None,
        "addressed_detected": addressed_detected / addressed_total if addressed_total else None,
        "addressed_correct": addressed_correct / addressed_total if addressed_total else None,
        "non_target_rejection": non_target_rejected / non_target_total if non_target_total else None,
    }


def global_response_metrics(
    global_response_logits: torch.Tensor,
    candidate_ranking_logits: list[torch.Tensor],
    response_candidate_labels: list[torch.Tensor],
    should_respond: torch.Tensor,
    threshold: float,
) -> dict[str, float | None]:
    addressed_detected = addressed_correct = addressed_total = 0
    non_target_rejected = non_target_total = 0
    for gate_logit, ranking_logits, labels, respond in zip(
        global_response_logits,
        candidate_ranking_logits,
        response_candidate_labels,
        should_respond,
    ):
        if ranking_logits.numel() == 0:
            continue
        predicted_should_respond = bool(torch.sigmoid(gate_logit).detach().item() >= threshold)
        predicted_index = int(ranking_logits.argmax().detach().cpu().item())
        if bool(respond.item()):
            labels = labels.bool()
            if not labels.any():
                continue
            addressed_total += 1
            addressed_detected += int(predicted_should_respond)
            addressed_correct += int(predicted_should_respond and bool(labels[predicted_index].item()))
        else:
            non_target_total += 1
            non_target_rejected += int(not predicted_should_respond)

    total = addressed_total + non_target_total
    return {
        "top1": (addressed_correct + non_target_rejected) / total if total else None,
        "addressed_detected": addressed_detected / addressed_total if addressed_total else None,
        "addressed_correct": addressed_correct / addressed_total if addressed_total else None,
        "non_target_rejection": non_target_rejected / non_target_total if non_target_total else None,
    }


def response_action_ce_loss(
    response_logits: list[torch.Tensor],
    response_candidate_labels: list[torch.Tensor],
    should_respond: torch.Tensor,
    addressed_weight: float = 1.0,
) -> torch.Tensor:
    losses = []
    for logits, labels, respond in zip(response_logits, response_candidate_labels, should_respond):
        if logits.numel() == 0:
            continue
        candidate_count = logits.shape[0] - 1
        labels = labels.bool()
        if bool(respond.item()):
            if not labels.any():
                continue
            log_probabilities = F.log_softmax(logits, dim=0)
            loss = -torch.logsumexp(log_probabilities[:candidate_count][labels], dim=0)
        else:
            target = logits.new_tensor(candidate_count, dtype=torch.long)
            loss = F.cross_entropy(logits.unsqueeze(0), target.reshape(1))
        losses.append(loss * float(addressed_weight) if bool(respond.item()) else loss)

    if not losses:
        if response_logits:
            return sum(logits.sum() * 0.0 for logits in response_logits)
        return torch.tensor(0.0)
    return torch.stack(losses).mean()


def response_action_metrics(
    response_logits: list[torch.Tensor],
    response_candidate_labels: list[torch.Tensor],
    should_respond: torch.Tensor,
) -> dict[str, float | None]:
    addressed_correct = 0
    addressed_total = 0
    non_target_correct = 0
    non_target_total = 0
    for logits, labels, respond in zip(response_logits, response_candidate_labels, should_respond):
        if logits.numel() == 0:
            continue
        candidate_count = logits.shape[0] - 1
        predicted = int(logits.argmax().detach().cpu().item())
        if bool(respond.item()):
            labels = labels.bool()
            if not labels.any():
                continue
            addressed_total += 1
            addressed_correct += int(predicted < candidate_count and bool(labels[predicted].item()))
        else:
            non_target_total += 1
            non_target_correct += int(predicted == candidate_count)

    total = addressed_total + non_target_total
    return {
        "top1": (addressed_correct + non_target_correct) / total if total else None,
        "addressed_correct": addressed_correct / addressed_total if addressed_total else None,
        "non_target_rejection": non_target_correct / non_target_total if non_target_total else None,
    }


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    def move_value(value: Any) -> Any:
        if torch.is_tensor(value):
            return value.to(device)
        if isinstance(value, list):
            return [move_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(move_value(item) for item in value)
        if isinstance(value, dict):
            return {key: move_value(item) for key, item in value.items()}
        return value

    return {key: move_value(value) for key, value in batch.items()}


def load_model_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
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
    return checkpoint


def load_optimizer_checkpoint(
    checkpoint: dict[str, Any],
    optimizer: torch.optim.Optimizer,
) -> None:
    if "optimizer" not in checkpoint:
        return
    learning_rates = [group["lr"] for group in optimizer.param_groups]
    try:
        optimizer.load_state_dict(checkpoint["optimizer"])
    except ValueError as exc:
        print(f"skipped optimizer state: {exc}", flush=True)
        return
    for group, learning_rate in zip(optimizer.param_groups, learning_rates):
        group["lr"] = learning_rate


def freeze_base_model_for_candidate_classifier(
    model: SAALModel,
    classifier_type: str,
    train_no_response_head: bool = False,
    unfreeze_fusion: bool = False,
    unfreeze_video_last_block: bool = False,
    unfreeze_audio_encoder: bool = False,
    use_head_crop_branch: bool = False,
    use_relation_candidate_head: bool = False,
    use_independent_candidate_head: bool = False,
) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    head = (
        model.candidate_relation_head
        if use_relation_candidate_head
        else model.independent_candidate_head
        if use_independent_candidate_head
        else model.head_crop_temporal_candidate_head
        if classifier_type == "temporal" and use_head_crop_branch
        else model.temporal_candidate_head
        if classifier_type == "temporal"
        else model.candidate_head
    )
    for parameter in head.parameters():
        parameter.requires_grad = True
    if use_head_crop_branch:
        for parameter in model.head_crop_encoder.parameters():
            parameter.requires_grad = True
    if train_no_response_head:
        for parameter in model.no_response_head.parameters():
            parameter.requires_grad = True
    if unfreeze_fusion:
        for parameter in model.fusion.parameters():
            parameter.requires_grad = True
    if unfreeze_video_last_block:
        for module in model.video_encoder.net[6:9]:
            for parameter in module.parameters():
                parameter.requires_grad = True
    if unfreeze_audio_encoder:
        for parameter in model.audio_encoder.parameters():
            parameter.requires_grad = True


def freeze_for_addressing_hard_negative_finetune(model: SAALModel) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.candidate_relation_head.addressing_classifier.parameters():
        parameter.requires_grad = True


def relation_adaptation_parameter_groups(
    model: SAALModel,
    relation_learning_rate: float,
    auxiliary_learning_rate: float,
    fusion_learning_rate: float,
    video_last_block_learning_rate: float,
    audio_encoder_learning_rate: float,
) -> list[dict[str, Any]]:
    for parameter in model.parameters():
        parameter.requires_grad = False

    relation = model.candidate_relation_head
    relation_core_modules = [
        relation.input_norm,
        relation.temporal,
        relation.token_projection,
        relation.scene_projection,
        relation.relation,
        relation.output_norm,
        relation.address_scene_norm,
        relation.candidate_memory,
        relation.candidate_memory_norm,
        relation.addressing_classifier,
    ]
    auxiliary_modules = [relation.speaking_classifier, relation.vad_classifier]

    def enable_modules(modules: list[torch.nn.Module]) -> list[torch.nn.Parameter]:
        parameters = [parameter for module in modules for parameter in module.parameters()]
        for parameter in parameters:
            parameter.requires_grad = True
        return parameters

    relation_parameters = enable_modules(relation_core_modules)
    relation.camera_token.requires_grad = True
    relation_parameters.append(relation.camera_token)
    auxiliary_parameters = enable_modules(auxiliary_modules)
    fusion_parameters = enable_modules([model.fusion])
    video_parameters = enable_modules(list(model.video_encoder.net[6:9]))
    audio_parameters = enable_modules([model.audio_encoder])

    return [
        {
            "params": relation_parameters,
            "lr": relation_learning_rate,
            "group_name": "relation_memory_addressing",
        },
        {
            "params": auxiliary_parameters,
            "lr": auxiliary_learning_rate,
            "group_name": "speaker_vad_heads",
        },
        {
            "params": fusion_parameters,
            "lr": fusion_learning_rate,
            "group_name": "fusion",
        },
        {
            "params": video_parameters,
            "lr": video_last_block_learning_rate,
            "group_name": "video_last_block",
        },
        {
            "params": audio_parameters,
            "lr": audio_encoder_learning_rate,
            "group_name": "audio_encoder",
        },
    ]


def speaker_representation_parameter_groups(
    model: SAALModel,
    speaker_learning_rate: float,
    relation_learning_rate: float,
    fusion_learning_rate: float,
    audio_encoder_learning_rate: float,
) -> list[dict[str, Any]]:
    """Train spatial speaker features while preserving response and memory behavior."""
    for parameter in model.parameters():
        parameter.requires_grad = False

    relation = model.candidate_relation_head
    relation_modules = [
        relation.input_norm,
        relation.temporal,
        relation.token_projection,
        relation.scene_projection,
        relation.relation,
        relation.output_norm,
    ]

    def enable(modules: list[torch.nn.Module]) -> list[torch.nn.Parameter]:
        parameters = [parameter for module in modules for parameter in module.parameters()]
        for parameter in parameters:
            parameter.requires_grad = True
        return parameters

    relation_parameters = enable(relation_modules)
    relation.camera_token.requires_grad = True
    relation_parameters.append(relation.camera_token)
    speaker_parameters = enable([relation.speaking_classifier])
    fusion_parameters = enable([model.fusion])
    audio_parameters = enable([model.audio_encoder])
    return [
        {
            "params": speaker_parameters,
            "lr": speaker_learning_rate,
            "group_name": "speaker_classifier",
        },
        {
            "params": relation_parameters,
            "lr": relation_learning_rate,
            "group_name": "speaker_relation_representation",
        },
        {
            "params": fusion_parameters,
            "lr": fusion_learning_rate,
            "group_name": "speaker_fusion",
        },
        {
            "params": audio_parameters,
            "lr": audio_encoder_learning_rate,
            "group_name": "speaker_audio_encoder",
        },
    ]


def speaker_residual_parameter_groups(
    model: SAALModel,
    learning_rate: float,
) -> list[dict[str, Any]]:
    """Train only a zero-initialized correction to frozen speaker logits."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    parameters = list(model.candidate_relation_head.speaker_residual_adapter.parameters())
    for parameter in parameters:
        parameter.requires_grad = True
    return [
        {
            "params": parameters,
            "lr": learning_rate,
            "group_name": "speaker_residual_adapter",
        }
    ]


def scene_response_parameter_groups(model: SAALModel, learning_rate: float) -> list[dict[str, Any]]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    modules = [
        model.candidate_relation_head.scene_response_classifier,
        model.candidate_relation_head.global_address_adapter,
    ]
    parameters = [parameter for module in modules for parameter in module.parameters()]
    for parameter in parameters:
        parameter.requires_grad = True
    return [{'params': parameters, 'lr': learning_rate, 'group_name': 'scene_response_head'}]


def shared_response_parameter_groups(
    model: SAALModel,
    learning_rate: float,
) -> list[dict[str, Any]]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    parameters = list(model.shared_response_gate.parameters())
    for parameter in parameters:
        parameter.requires_grad = True
    return [{
        'params': parameters,
        'lr': learning_rate,
        'group_name': 'shared_response_gate',
    }]


def global_address_parameter_groups(
    model: SAALModel,
    learning_rate: float,
    train_adapter: bool = False,
) -> list[dict[str, Any]]:
    """Train only the set-level response path on frozen causal AV features."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    relation = model.candidate_relation_head
    modules = [relation.global_address_classifier]
    if train_adapter:
        modules.append(relation.global_address_adapter)
    parameters = [parameter for module in modules for parameter in module.parameters()]
    for parameter in parameters:
        parameter.requires_grad = True
    return [
        {
            "params": parameters,
            "lr": learning_rate,
            "group_name": (
                "global_address_head_and_adapter" if train_adapter else "global_address_head"
            ),
        }
    ]


def candidate_finetune_parameter_groups(
    model: SAALModel,
    classifier_type: str,
    head_learning_rate: float,
    fusion_learning_rate: float,
    video_last_block_learning_rate: float,
    audio_encoder_learning_rate: float = 1e-5,
    train_no_response_head: bool = False,
    train_fusion: bool = False,
    train_video_last_block: bool = False,
    train_audio_encoder: bool = False,
    use_head_crop_branch: bool = False,
    use_relation_candidate_head: bool = False,
    use_independent_candidate_head: bool = False,
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []

    def add_group(name: str, modules: list[torch.nn.Module], learning_rate: float) -> None:
        parameters = [
            parameter
            for module in modules
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        if parameters:
            groups.append({"params": parameters, "lr": float(learning_rate), "group_name": name})

    head = (
        model.candidate_relation_head
        if use_relation_candidate_head
        else model.independent_candidate_head
        if use_independent_candidate_head
        else model.head_crop_temporal_candidate_head
        if classifier_type == "temporal" and use_head_crop_branch
        else model.temporal_candidate_head
        if classifier_type == "temporal"
        else model.candidate_head
    )
    head_modules = [head, model.head_crop_encoder] if use_head_crop_branch else [head]
    add_group("candidate_head", head_modules, head_learning_rate)
    if train_no_response_head:
        add_group("no_response_head", [model.no_response_head], head_learning_rate)
    if train_fusion:
        add_group("fusion", [model.fusion], fusion_learning_rate)
    if train_video_last_block:
        add_group("video_last_block", list(model.video_encoder.net[6:9]), video_last_block_learning_rate)
    if train_audio_encoder:
        add_group("audio_encoder", [model.audio_encoder], audio_encoder_learning_rate)
    return groups


def set_base_model_eval_for_candidate_classifier(
    model: SAALModel,
    classifier_type: str,
    train_no_response_head: bool = False,
    train_fusion: bool = False,
    train_video_last_block: bool = False,
    train_audio_encoder: bool = False,
    use_head_crop_branch: bool = False,
    use_relation_candidate_head: bool = False,
    use_independent_candidate_head: bool = False,
) -> None:
    model.video_encoder.eval()
    model.audio_encoder.train(train_audio_encoder)
    model.fusion.eval()
    model.heatmap_head.eval()
    model.candidate_head.train(classifier_type == "spatial")
    model.temporal_candidate_head.train(
        classifier_type == "temporal"
        and not use_head_crop_branch
        and not use_relation_candidate_head
        and not use_independent_candidate_head
    )
    model.head_crop_encoder.train(classifier_type == "temporal" and use_head_crop_branch)
    model.head_crop_temporal_candidate_head.train(classifier_type == "temporal" and use_head_crop_branch)
    model.candidate_relation_head.train(use_relation_candidate_head)
    model.independent_candidate_head.train(use_independent_candidate_head)
    model.no_response_head.train(train_no_response_head)
    if train_fusion:
        model.fusion.train()
    if train_video_last_block:
        for module in model.video_encoder.net[6:9]:
            module.train()


def write_json_atomic(path: Path, values: dict[str, Any]) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(values, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def save_checkpoint_atomic(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: EasyComSAALConfig,
    epoch: int,
    step: int,
    elapsed_seconds: float,
    identity_state_cache: IdentityStateCache | None = None,
    stateful_epoch_index: int | None = None,
    stateful_batches_completed: int | None = None,
    turn_graph_cache: TurnGraphCache | None = None,
) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp")
    values = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config.to_dict(),
        "epoch": epoch,
        "step": step,
        "elapsed_seconds": elapsed_seconds,
    }
    if identity_state_cache is not None:
        values["identity_state_cache"] = identity_state_cache.state_dict()
        values["stateful_epoch_index"] = stateful_epoch_index
        values["stateful_batches_completed"] = stateful_batches_completed
    if turn_graph_cache is not None:
        values["turn_graph_cache"] = turn_graph_cache.state_dict()
    torch.save(values, tmp_path)
    tmp_path.replace(path)


def train(
    config: EasyComSAALConfig,
    output_path: str,
    device: str,
    log_every: int = 10,
    progress_path: str | None = None,
    checkpoint_every: int = 0,
    resume_checkpoint: str | None = None,
    max_steps: int | None = None,
    step_offset: int = 0,
    sample_start: int | None = None,
    sample_end: int | None = None,
    sample_order_seed: int = 0,
    reset_stateful_progress: bool = False,
) -> None:
    torch_device = torch.device(device)
    stateful_identity_memory = bool(
        config.use_candidate_identity_memory and config.use_causal_memory
    )
    annotations = EasyComAnnotations.from_manifest(config.annotation_manifest, root=config.easycom_root)
    dataset = EasyComSAALDataset(
        annotations,
        config=config,
        load_media=True,
        return_previous=(
            config.use_temporal_loss
            or (config.use_causal_memory and not stateful_identity_memory)
        ),
    )
    if len(dataset) == 0:
        raise RuntimeError("No valid EasyCom SAAL samples were built. Check ToS filtering, media paths, and bboxes.")

    full_sample_count = len(dataset)
    partitioned = sample_start is not None or sample_end is not None
    partition_start = 0 if sample_start is None else int(sample_start)
    partition_end = full_sample_count if sample_end is None else int(sample_end)
    if partitioned:
        if not 0 <= partition_start < partition_end <= full_sample_count:
            raise ValueError(
                f"Invalid sample partition [{partition_start}, {partition_end}) for {full_sample_count} samples."
            )
        generator = torch.Generator().manual_seed(int(sample_order_seed))
        permutation = torch.randperm(full_sample_count, generator=generator).tolist()
        dataset = Subset(dataset, permutation[partition_start:partition_end])

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
            pin_memory=torch_device.type == "cuda",
            persistent_workers=config.num_workers > 0,
        )
        if stateful_batch_sampler is not None
        else DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=not partitioned,
            num_workers=config.num_workers,
            collate_fn=saal_collate_fn,
            pin_memory=torch_device.type == "cuda",
            persistent_workers=config.num_workers > 0,
        )
    )

    model = SAALModel(
        num_audio_channels=config.num_audio_channels,
        num_corr_channels=config.num_audio_channels * (config.num_audio_channels - 1) // 2,
        heatmap_size=config.heatmap_size,
        head_crop_feature_dim=config.head_crop_feature_dim,
        frame_aligned_audio=config.use_frame_aligned_audio,
    ).to(torch_device)
    checkpoint = None
    if resume_checkpoint is not None:
        checkpoint = load_model_checkpoint(resume_checkpoint, model, torch_device)
        print(
            f"resumed checkpoint={resume_checkpoint} "
            f"checkpoint_epoch={checkpoint.get('epoch')} checkpoint_step={checkpoint.get('step')} "
            f"lr={config.learning_rate}",
            flush=True,
        )
    if config.scene_response_finetune:
        scene_response_parameter_groups(model, learning_rate=config.learning_rate)
        print('configured frozen-backbone scene response fine-tuning', flush=True)
    elif config.shared_response_finetune:
        shared_response_parameter_groups(model, learning_rate=config.learning_rate)
        print('configured frozen-backbone shared response-gate fine-tuning', flush=True)
    elif config.relation_adaptation_finetune:
        print("configured full relation/memory adaptation fine-tuning", flush=True)
    elif config.global_address_adapter_finetune or config.global_address_finetune:
        global_address_parameter_groups(
            model,
            learning_rate=config.learning_rate,
            train_adapter=config.global_address_adapter_finetune,
        )
        print(
            "configured frozen-backbone global addressedness fine-tuning "
            f"adapter={config.global_address_adapter_finetune}",
            flush=True,
        )
    elif config.speaker_residual_finetune:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.candidate_relation_head.speaker_residual_adapter.parameters():
            parameter.requires_grad = True
        print("configured frozen-backbone speaker residual fine-tuning", flush=True)
    elif config.speaker_representation_finetune:
        print("configured cross-session speaker representation fine-tuning", flush=True)
    elif config.addressing_hard_negative_finetune:
        freeze_for_addressing_hard_negative_finetune(model)
        print("configured conditional-addressing hard-negative fine-tuning", flush=True)
    elif config.audio_robustness_heads_only:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for module in (
            model.candidate_relation_head.speaking_classifier,
            model.candidate_relation_head.vad_classifier,
        ):
            for parameter in module.parameters():
                parameter.requires_grad = True
        print("configured audio-robustness VAD/speaker heads-only fine-tuning", flush=True)
    elif config.turn_graph_other_only:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.candidate_relation_head.turn_graph_other_residual.parameters():
            parameter.requires_grad = True
        print("configured turn-graph OTHER residual-only fine-tuning", flush=True)
    elif config.freeze_base_for_candidate_classifier:
        freeze_base_model_for_candidate_classifier(
            model,
            config.candidate_classifier_type,
            train_no_response_head=config.lambda_response_ce > 0.0,
            unfreeze_fusion=config.unfreeze_fusion_for_candidate_classifier,
            unfreeze_video_last_block=config.unfreeze_video_last_block_for_candidate_classifier,
            unfreeze_audio_encoder=config.unfreeze_audio_encoder_for_candidate_classifier,
            use_head_crop_branch=config.use_head_crop_branch,
            use_relation_candidate_head=config.use_relation_candidate_head,
            use_independent_candidate_head=config.use_independent_candidate_head,
        )
        print(
            f"configured candidate fine-tuning head={config.candidate_classifier_type} "
            f"fusion={config.unfreeze_fusion_for_candidate_classifier} "
            f"video_last_block={config.unfreeze_video_last_block_for_candidate_classifier}",
            f" head_crop_branch={config.use_head_crop_branch}",
            flush=True,
        )
    if config.scene_response_finetune:
        optimizer_groups = scene_response_parameter_groups(model, config.learning_rate)
        trainable_parameters = [parameter for group in optimizer_groups for parameter in group['params']]
    elif config.shared_response_finetune:
        optimizer_groups = shared_response_parameter_groups(model, config.learning_rate)
        trainable_parameters = [parameter for group in optimizer_groups for parameter in group['params']]
    elif config.relation_adaptation_finetune:
        optimizer_groups = relation_adaptation_parameter_groups(
            model,
            relation_learning_rate=config.learning_rate,
            auxiliary_learning_rate=config.relation_aux_learning_rate,
            fusion_learning_rate=config.fusion_learning_rate,
            video_last_block_learning_rate=config.video_last_block_learning_rate,
            audio_encoder_learning_rate=config.audio_encoder_learning_rate,
        )
        trainable_parameters = [
            parameter for group in optimizer_groups for parameter in group["params"]
        ]
    elif config.global_address_adapter_finetune or config.global_address_finetune:
        optimizer_groups = global_address_parameter_groups(
            model,
            learning_rate=config.learning_rate,
            train_adapter=config.global_address_adapter_finetune,
        )
        trainable_parameters = [
            parameter for group in optimizer_groups for parameter in group["params"]
        ]
    elif config.speaker_residual_finetune:
        optimizer_groups = speaker_residual_parameter_groups(
            model,
            learning_rate=config.learning_rate,
        )
        trainable_parameters = [
            parameter for group in optimizer_groups for parameter in group["params"]
        ]
    elif config.speaker_representation_finetune:
        optimizer_groups = speaker_representation_parameter_groups(
            model,
            speaker_learning_rate=config.learning_rate,
            relation_learning_rate=config.relation_aux_learning_rate,
            fusion_learning_rate=config.fusion_learning_rate,
            audio_encoder_learning_rate=config.audio_encoder_learning_rate,
        )
        trainable_parameters = [
            parameter for group in optimizer_groups for parameter in group["params"]
        ]
    elif config.addressing_hard_negative_finetune:
        trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer_groups = [
            {
                "params": trainable_parameters,
                "lr": config.learning_rate,
                "group_name": "addressing_classifier",
            }
        ]
    elif config.audio_robustness_heads_only:
        trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer_groups = [
            {
                "params": trainable_parameters,
                "lr": config.learning_rate,
                "group_name": "audio_robustness_heads",
            }
        ]
    elif config.turn_graph_other_only:
        trainable_parameters = [
            parameter
            for parameter in model.candidate_relation_head.turn_graph_other_residual.parameters()
            if parameter.requires_grad
        ]
        optimizer_groups = [
            {
                "params": trainable_parameters,
                "lr": config.learning_rate,
                "group_name": "turn_graph_other_residual",
            }
        ]
    elif config.freeze_base_for_candidate_classifier:
        optimizer_groups = candidate_finetune_parameter_groups(
            model,
            classifier_type=config.candidate_classifier_type,
            head_learning_rate=config.learning_rate,
            fusion_learning_rate=config.fusion_learning_rate,
            video_last_block_learning_rate=config.video_last_block_learning_rate,
            audio_encoder_learning_rate=config.audio_encoder_learning_rate,
            train_no_response_head=config.lambda_response_ce > 0.0,
            train_fusion=config.unfreeze_fusion_for_candidate_classifier,
            train_video_last_block=config.unfreeze_video_last_block_for_candidate_classifier,
            train_audio_encoder=config.unfreeze_audio_encoder_for_candidate_classifier,
            use_head_crop_branch=config.use_head_crop_branch,
            use_relation_candidate_head=config.use_relation_candidate_head,
            use_independent_candidate_head=config.use_independent_candidate_head,
        )
        trainable_parameters = [parameter for group in optimizer_groups for parameter in group["params"]]
    else:
        trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer_groups = [{"params": trainable_parameters, "lr": config.learning_rate, "group_name": "all"}]
    if not trainable_parameters:
        raise RuntimeError("No trainable model parameters remain.")
    optimizer = torch.optim.AdamW(optimizer_groups, lr=config.learning_rate)
    print(
        "optimizer_groups="
        + ",".join(
            f"{group.get('group_name')}:{group['lr']}:{sum(parameter.numel() for parameter in group['params'])}"
            for group in optimizer.param_groups
        ),
        flush=True,
    )
    if checkpoint is not None and not config.shared_response_finetune and not config.scene_response_finetune:
        load_optimizer_checkpoint(checkpoint, optimizer)

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
    stateful_resume_epoch = 0
    stateful_resume_batches = 0
    if (
        identity_state_cache is not None
        and checkpoint is not None
        and "identity_state_cache" in checkpoint
        and not reset_stateful_progress
    ):
        identity_state_cache.load_state_dict(checkpoint["identity_state_cache"], torch_device)
        stateful_resume_epoch = int(checkpoint.get("stateful_epoch_index", 0) or 0)
        stateful_resume_batches = int(checkpoint.get("stateful_batches_completed", 0) or 0)
        if stateful_batch_sampler is None:
            raise RuntimeError("Stateful identity checkpoint requires a session stream batch sampler.")
        stateful_batch_sampler.start_batch = stateful_resume_batches
        print(
            f"restored stateful identity cache sessions={len(identity_state_cache.states)} "
            f"epoch_index={stateful_resume_epoch} batches_completed={stateful_resume_batches}",
            flush=True,
        )
    if turn_graph_cache is not None and checkpoint is not None and "turn_graph_cache" in checkpoint:
        if not reset_stateful_progress:
            turn_graph_cache.load_state_dict(checkpoint["turn_graph_cache"])
            print(
                f"restored turn graph sessions={len(turn_graph_cache.states)}",
                flush=True,
            )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    progress_output = Path(progress_path) if progress_path is not None else output.with_suffix(".progress.json")
    progress_output.parent.mkdir(parents=True, exist_ok=True)
    total_steps = (
        stateful_batch_sampler.total_batches
        if stateful_batch_sampler is not None
        else len(dataloader)
    )
    planned_steps = config.epochs * total_steps if max_steps is None else max(0, int(max_steps))
    start_time = time.perf_counter()

    print(
        f"start device={torch_device} samples={len(dataset)} epochs={config.epochs} "
        f"batch_size={config.batch_size} steps_per_epoch={total_steps} num_workers={config.num_workers} "
        f"gradient_accumulation_steps={config.gradient_accumulation_steps} "
        f"lambda_candidate_ranking={config.lambda_candidate_ranking} "
        f"lambda_candidate_ce={config.lambda_candidate_ce} "
        f"lambda_candidate_classifier_ce={config.lambda_candidate_classifier_ce} "
        f"lambda_candidate_target_bce={config.lambda_candidate_target_bce} "
        f"lambda_candidate_speaker_bce={config.lambda_candidate_speaker_bce} "
        f"lambda_candidate_speaker_ce={config.lambda_candidate_speaker_ce} "
        f"candidate_target_threshold={config.candidate_target_threshold} "
        f"lambda_response_ce={config.lambda_response_ce} "
        f"response_addressed_weight={config.response_addressed_weight} "
        f"lambda_heatmap={config.lambda_heatmap} "
        f"candidate_classifier_type={config.candidate_classifier_type} "
        f"freeze_base_for_candidate_classifier={config.freeze_base_for_candidate_classifier} "
        f"unfreeze_fusion={config.unfreeze_fusion_for_candidate_classifier} "
        f"unfreeze_video_last_block={config.unfreeze_video_last_block_for_candidate_classifier} "
        f"unfreeze_audio_encoder={config.unfreeze_audio_encoder_for_candidate_classifier} "
        f"head_crop_branch={config.use_head_crop_branch} "
        f"relation_head={config.use_relation_candidate_head} "
        f"independent_head={config.use_independent_candidate_head} "
        f"global_address_gate={config.use_global_address_gate} "
        f"candidate_address_memory={config.use_candidate_address_memory} "
        f"candidate_identity_memory={config.use_candidate_identity_memory} "
        f"identity_bptt_windows={config.identity_bptt_windows} "
        f"turn_graph={config.use_turn_graph} "
        f"turn_graph_teacher_forcing={config.turn_graph_teacher_forcing} "
        f"lambda_self_speaker_bce={config.lambda_self_speaker_bce} "
        f"lambda_self_target_bce={config.lambda_self_target_bce} "
        f"causal_memory={config.use_causal_memory} "
        f"stateful_identity_memory={stateful_identity_memory} "
        f"preserve_audio_energy={config.preserve_audio_energy} "
        f"addressing_hard_negative_finetune={config.addressing_hard_negative_finetune} "
        f"addressing_negative_weight={config.addressing_negative_weight} "
        f"addressing_hard_negative_fraction={config.addressing_hard_negative_fraction} "
        f"addressing_hard_negative_weight={config.addressing_hard_negative_weight} "
        f"relation_adaptation_finetune={config.relation_adaptation_finetune} "
        f"relation_aux_lr={config.relation_aux_learning_rate} "
        f"lambda_vad_bce={config.lambda_vad_bce} "
        f"frame_aligned_audio={config.use_frame_aligned_audio} "
        f"masked_head_video={config.use_masked_head_video} "
        f"max_steps={max_steps} step_offset={step_offset} "
        f"sample_partition={partition_start}:{partition_end}/{full_sample_count} "
        f"sample_order_seed={sample_order_seed}",
        flush=True,
    )

    model.train()
    if config.scene_response_finetune:
        model.eval()
        model.candidate_relation_head.scene_response_classifier.train()
        model.candidate_relation_head.global_address_adapter.train()
    elif config.shared_response_finetune:
        model.eval()
        model.shared_response_gate.train()
    elif config.relation_adaptation_finetune:
        model.eval()
        model.candidate_relation_head.train()
        model.fusion.train()
        model.audio_encoder.train()
        for module in model.video_encoder.net[6:9]:
            module.train()
    elif config.global_address_adapter_finetune or config.global_address_finetune:
        model.eval()
        model.candidate_relation_head.global_address_classifier.train()
        if config.global_address_adapter_finetune:
            model.candidate_relation_head.global_address_adapter.train()
    elif config.speaker_residual_finetune:
        model.eval()
        model.candidate_relation_head.speaker_residual_adapter.train()
    elif config.addressing_hard_negative_finetune:
        model.eval()
        model.candidate_relation_head.addressing_classifier.train()
    elif config.audio_robustness_heads_only:
        model.eval()
        model.candidate_relation_head.speaking_classifier.train()
        model.candidate_relation_head.vad_classifier.train()
    elif config.turn_graph_other_only:
        model.eval()
        model.candidate_relation_head.turn_graph_other_residual.train()
    elif config.freeze_base_for_candidate_classifier:
        set_base_model_eval_for_candidate_classifier(
            model,
            config.candidate_classifier_type,
            train_no_response_head=config.lambda_response_ce > 0.0,
            train_fusion=config.unfreeze_fusion_for_candidate_classifier,
            train_video_last_block=config.unfreeze_video_last_block_for_candidate_classifier,
            train_audio_encoder=config.unfreeze_audio_encoder_for_candidate_classifier,
            use_head_crop_branch=config.use_head_crop_branch,
            use_relation_candidate_head=config.use_relation_candidate_head,
            use_independent_candidate_head=config.use_independent_candidate_head,
        )
    completed_steps = 0
    last_global_step = int(step_offset)
    last_stateful_epoch_index = stateful_resume_epoch
    last_stateful_batches_completed = stateful_resume_batches
    running_loss = 0.0
    use_identity_tbptt = bool(stateful_identity_memory and config.identity_bptt_windows > 1)
    pending_tbptt_losses: list[torch.Tensor] = []
    accumulated_microsteps = 0
    optimizer_steps_completed = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(config.epochs):
        if stateful_batch_sampler is not None:
            if epoch < stateful_resume_epoch:
                continue
            if epoch == stateful_resume_epoch:
                stateful_batch_sampler.start_batch = stateful_resume_batches
            else:
                stateful_batch_sampler.start_batch = 0
                if identity_state_cache is not None:
                    identity_state_cache.clear()
                if turn_graph_cache is not None:
                    turn_graph_cache.clear()
        previous_step_end = time.perf_counter()
        for step, batch in enumerate(dataloader, start=1):
            if max_steps is not None and completed_steps >= max_steps:
                break
            completed_steps += 1
            global_step = int(step_offset) + completed_steps
            last_global_step = global_step
            if stateful_batch_sampler is not None:
                last_stateful_epoch_index = epoch
                last_stateful_batches_completed = stateful_batch_sampler.start_batch + step
            step_start = time.perf_counter()
            data_seconds = step_start - previous_step_end
            batch = move_batch_to_device(batch, torch_device)
            fused_temporal_features = None
            if config.candidate_classifier_type == "temporal":
                pred, fused_features, fused_temporal_features = model.forward_with_temporal_features(
                    batch["video_clip"],
                    batch["audio_spectrogram"],
                    batch["audio_channel_correlation"],
                )
            else:
                pred, fused_features = model.forward_with_features(
                    batch["video_clip"],
                    batch["audio_spectrogram"],
                    batch["audio_channel_correlation"],
                )
            loss_main = heatmap_loss(pred, batch["target_heatmap"], config.heatmap_loss)
            loss_candidate = pred.new_tensor(0.0)
            loss_candidate_ce = pred.new_tensor(0.0)
            loss_candidate_classifier_ce = pred.new_tensor(0.0)
            loss_candidate_target_bce = pred.new_tensor(0.0)
            loss_candidate_speaker_bce = pred.new_tensor(0.0)
            loss_candidate_speaker_ce = pred.new_tensor(0.0)
            loss_vad_bce = pred.new_tensor(0.0)
            loss_shared_response_bce = pred.new_tensor(0.0)
            loss_scene_response_ce = pred.new_tensor(0.0)
            loss_self_speaker_bce = pred.new_tensor(0.0)
            loss_self_target_bce = pred.new_tensor(0.0)
            loss_response_ce = pred.new_tensor(0.0)
            candidate_classifier_batch_top1 = None
            response_batch_metrics = {
                "top1": None,
                "addressed_correct": None,
                "non_target_rejection": None,
            }
            target_batch_metrics = {
                "top1": None,
                "addressed_detected": None,
                "addressed_correct": None,
                "non_target_rejection": None,
            }
            speaker_batch_metrics = {"accuracy": None, "recall": None, "specificity": None}
            if config.lambda_candidate_ranking > 0.0:
                loss_candidate = candidate_ranking_loss(
                    pred,
                    batch["candidate_bboxes_heatmap"],
                    batch["candidate_labels"],
                    margin=config.candidate_ranking_margin,
                    topk_fraction=config.candidate_topk_fraction,
                    min_pixels=config.candidate_topk_min_pixels,
                )
            if config.lambda_candidate_ce > 0.0:
                loss_candidate_ce = candidate_softmax_ce_loss(
                    pred,
                    batch["candidate_bboxes_heatmap"],
                    batch["candidate_labels"],
                    topk_fraction=config.candidate_topk_fraction,
                    min_pixels=config.candidate_topk_min_pixels,
                    temperature=config.candidate_ce_temperature,
                )
            response_logits = None
            classifier_logits = None
            if config.lambda_response_ce > 0.0:
                if config.candidate_classifier_type != "temporal" or fused_temporal_features is None:
                    raise RuntimeError("Response action training requires temporal candidate features.")
                response_logits = model.temporal_response_logits(
                    fused_temporal_features,
                    batch["candidate_bboxes_sequence_heatmap"],
                )
                loss_response_ce = response_action_ce_loss(
                    response_logits,
                    batch["response_candidate_labels"],
                    batch["should_respond"],
                    addressed_weight=config.response_addressed_weight,
                )
                response_batch_metrics = response_action_metrics(
                    response_logits,
                    batch["response_candidate_labels"],
                    batch["should_respond"],
                )

            relation_speaking_logits = None
            relation_addressing_logits = None
            relation_global_addressing_logits = None
            vad_logits = None
            scene_response_logits = None
            self_speaking_logits = None
            self_target_logits = None
            identity_updated_states: list[dict[object, torch.Tensor]] | None = None
            if (config.use_relation_candidate_head or config.use_independent_candidate_head) and (
                config.lambda_candidate_target_bce > 0.0
                or config.lambda_candidate_speaker_bce > 0.0
                or config.lambda_candidate_speaker_ce > 0.0
                or config.lambda_vad_bce > 0.0
                or config.lambda_shared_response_bce > 0.0
                or config.lambda_scene_response_ce > 0.0
            ):
                if fused_temporal_features is None:
                    raise RuntimeError("Multitask candidate training requires temporal candidate features.")
                if config.use_candidate_identity_memory:
                    previous_states: list[dict[object, torch.Tensor]] | None = None
                    if stateful_identity_memory:
                        if identity_state_cache is None:
                            raise RuntimeError("Stateful identity memory requires an initialized state cache.")
                        previous_states = identity_state_cache.previous_states(
                            batch["session_id"],
                            batch["timestamp"],
                        )
                    elif config.use_causal_memory:
                        temporal_mask = batch["apply_temporal_loss"]
                        previous_states = [{} for _ in range(fused_temporal_features.shape[0])]
                        if temporal_mask.any():
                            previous_features = model.encode_temporal_features(
                                batch["prev_video_clip"][temporal_mask],
                                batch["prev_audio_spectrogram"][temporal_mask],
                                batch["prev_audio_channel_correlation"][temporal_mask],
                            )
                            valid_indices = temporal_mask.nonzero(as_tuple=False).flatten()
                            previous_boxes = [
                                batch["prev_candidate_bboxes_sequence_heatmap"][int(index.item())]
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
                            relation_addressing_logits,
                            scene_response_logits,
                            vad_logits,
                            identity_updated_states,
                        ) = model.temporal_identity_scene_response_logits(
                            fused_temporal_features,
                            batch['candidate_bboxes_sequence_heatmap'],
                            [tuple(ids) for ids in batch['candidate_ids']],
                            previous_states=previous_states,
                        )
                    elif config.use_turn_graph:
                        if turn_graph_cache is None:
                            raise RuntimeError("Turn-graph mode requires an initialized graph cache.")
                        turn_features = turn_graph_cache.features(
                            batch["session_id"],
                            batch["timestamp"],
                            batch["candidate_ids"],
                            fused_temporal_features,
                        )
                        (
                            relation_speaking_logits,
                            relation_addressing_logits,
                            vad_logits,
                            self_speaking_logits,
                            self_target_logits,
                            identity_updated_states,
                        ) = model.temporal_identity_turn_graph_relation_logits(
                            fused_temporal_features,
                            batch["candidate_bboxes_sequence_heatmap"],
                            [tuple(ids) for ids in batch["candidate_ids"]],
                            turn_features,
                            previous_states=previous_states,
                        )
                    elif config.use_global_address_gate:
                        (
                            relation_speaking_logits,
                            relation_global_addressing_logits,
                            vad_logits,
                            identity_updated_states,
                        ) = model.temporal_identity_global_relation_logits(
                            fused_temporal_features,
                            batch["candidate_bboxes_sequence_heatmap"],
                            [tuple(ids) for ids in batch["candidate_ids"]],
                            previous_states=previous_states,
                        )
                    else:
                        (
                            relation_speaking_logits,
                            relation_addressing_logits,
                            vad_logits,
                            identity_updated_states,
                        ) = model.temporal_identity_memory_relation_logits(
                            fused_temporal_features,
                            batch["candidate_bboxes_sequence_heatmap"],
                            [tuple(ids) for ids in batch["candidate_ids"]],
                            previous_states=previous_states,
                        )
                elif config.use_independent_candidate_head:
                    (
                        relation_speaking_logits,
                        relation_addressing_logits,
                    ) = model.temporal_independent_candidate_logits(
                        fused_temporal_features,
                        batch["candidate_bboxes_sequence_heatmap"],
                    )
                elif config.use_candidate_address_memory:
                    previous_memory = None
                    if config.use_causal_memory:
                        temporal_mask = batch["apply_temporal_loss"]
                        previous_memory = fused_temporal_features.new_zeros(
                            (fused_temporal_features.shape[0], 128)
                        )
                        if temporal_mask.any():
                            previous_features = model.encode_temporal_features(
                                batch["prev_video_clip"][temporal_mask],
                                batch["prev_audio_spectrogram"][temporal_mask],
                                batch["prev_audio_channel_correlation"][temporal_mask],
                            )
                            valid_indices = temporal_mask.nonzero(as_tuple=False).flatten()
                            previous_boxes = [
                                batch["prev_candidate_bboxes_sequence_heatmap"][int(index.item())]
                                for index in valid_indices
                            ]
                            _, _, valid_memory = model.temporal_memory_relation_logits(
                                previous_features,
                                previous_boxes,
                            )
                            previous_memory = previous_memory.index_copy(0, valid_indices, valid_memory)
                    (
                        relation_speaking_logits,
                        relation_addressing_logits,
                        _,
                    ) = model.temporal_memory_relation_logits(
                        fused_temporal_features,
                        batch["candidate_bboxes_sequence_heatmap"],
                        previous_memory=previous_memory,
                    )
                elif config.use_global_address_gate:
                    previous_memory = None
                    if config.use_causal_memory:
                        temporal_mask = batch["apply_temporal_loss"]
                        previous_memory = fused_temporal_features.new_zeros(
                            (fused_temporal_features.shape[0], 128)
                        )
                        if temporal_mask.any():
                            previous_features = model.encode_temporal_features(
                                batch["prev_video_clip"][temporal_mask],
                                batch["prev_audio_spectrogram"][temporal_mask],
                                batch["prev_audio_channel_correlation"][temporal_mask],
                            )
                            valid_indices = temporal_mask.nonzero(as_tuple=False).flatten()
                            previous_boxes = [
                                batch["prev_candidate_bboxes_sequence_heatmap"][int(index.item())]
                                for index in valid_indices
                            ]
                            _, _, valid_memory = model.temporal_global_relation_logits(
                                previous_features,
                                previous_boxes,
                            )
                            previous_memory = previous_memory.index_copy(0, valid_indices, valid_memory)
                    (
                        relation_speaking_logits,
                        relation_global_addressing_logits,
                        _,
                    ) = model.temporal_global_relation_logits(
                        fused_temporal_features,
                        batch["candidate_bboxes_sequence_heatmap"],
                        previous_memory=previous_memory,
                    )
                else:
                    relation_speaking_logits, relation_addressing_logits = model.temporal_relation_logits(
                        fused_temporal_features,
                        batch["candidate_bboxes_sequence_heatmap"],
                    )

            if config.lambda_candidate_target_bce > 0.0:
                if config.candidate_classifier_type != "temporal" or fused_temporal_features is None:
                    raise RuntimeError("Candidate target BCE training requires temporal candidate features.")
                classifier_logits = (
                    model.combine_global_address_logits(
                        relation_speaking_logits,
                        relation_global_addressing_logits,
                    )
                    if relation_global_addressing_logits is not None and relation_speaking_logits is not None
                    else
                    model.combine_relation_logits(relation_speaking_logits, relation_addressing_logits)
                    if relation_addressing_logits is not None and relation_speaking_logits is not None
                    else (
                    model.temporal_head_crop_candidate_logits(
                        fused_temporal_features,
                        batch["candidate_bboxes_sequence_heatmap"],
                        batch["candidate_head_crops"],
                    )
                    if config.use_head_crop_branch
                    else model.temporal_candidate_logits(
                        fused_temporal_features,
                        batch["candidate_bboxes_sequence_heatmap"],
                    )
                    )
                )
                loss_candidate_target_bce = (
                    session_category_balanced_response_bce_loss(
                        relation_global_addressing_logits,
                        batch['should_respond'],
                        batch['session_id'],
                        batch['sample_type'],
                    )
                    if relation_global_addressing_logits is not None
                    and config.use_session_category_balanced_response_loss
                    else
                    balanced_binary_bce_loss(
                        relation_global_addressing_logits,
                        batch["should_respond"],
                        positive_weight=config.global_address_positive_weight,
                    )
                    if relation_global_addressing_logits is not None
                    else
                    conditional_addressing_bce_loss(
                        relation_addressing_logits,
                        batch["response_candidate_labels"],
                        batch["speaker_candidate_labels"],
                        negative_weight=(
                            config.addressing_negative_weight
                            if (
                                config.addressing_hard_negative_finetune
                                or config.relation_adaptation_finetune
                            )
                            else 1.0
                        ),
                        hard_negative_fraction=(
                            config.addressing_hard_negative_fraction
                            if (
                                config.addressing_hard_negative_finetune
                                or config.relation_adaptation_finetune
                            )
                            else 0.0
                        ),
                        hard_negative_weight=(
                            config.addressing_hard_negative_weight
                            if (
                                config.addressing_hard_negative_finetune
                                or config.relation_adaptation_finetune
                            )
                            else 1.0
                        ),
                    )
                    if relation_addressing_logits is not None
                    else candidate_target_bce_loss(
                        classifier_logits,
                        batch["response_candidate_labels"],
                    )
                )
                target_batch_metrics = (
                    global_response_metrics(
                        relation_global_addressing_logits,
                        relation_speaking_logits,
                        batch["response_candidate_labels"],
                        batch["should_respond"],
                        threshold=config.candidate_target_threshold,
                    )
                    if relation_global_addressing_logits is not None
                    and relation_speaking_logits is not None
                    else candidate_target_metrics(
                        classifier_logits,
                        batch["response_candidate_labels"],
                        batch["should_respond"],
                        threshold=config.candidate_target_threshold,
                    )
                )

            if config.lambda_candidate_speaker_bce > 0.0:
                if relation_speaking_logits is None:
                    raise RuntimeError("Speaker auxiliary loss requires the relation candidate head.")
                loss_candidate_speaker_bce = candidate_target_bce_loss(
                    relation_speaking_logits,
                    batch["speaker_candidate_labels"],
                )
                speaker_batch_metrics = candidate_binary_metrics(
                    relation_speaking_logits,
                    batch["speaker_candidate_labels"],
                    threshold=config.speaker_threshold,
                )

            if config.lambda_candidate_speaker_ce > 0.0:
                if relation_speaking_logits is None:
                    raise RuntimeError("Speaker CE requires the relation candidate head.")
                loss_candidate_speaker_ce = candidate_speaker_ce_loss(
                    relation_speaking_logits,
                    batch["speaker_candidate_labels"],
                    weight_by_candidate_count=config.weight_speaker_ce_by_candidate_count,
                )

            if config.lambda_vad_bce > 0.0:
                if vad_logits is None:
                    raise RuntimeError("VAD loss requires the candidate identity-memory relation mode.")
                loss_vad_bce = balanced_binary_bce_loss(vad_logits, batch["speech_present"])

            if config.lambda_scene_response_ce > 0.0:
                if scene_response_logits is None:
                    raise RuntimeError('Scene response loss requires the identity-scene response head.')
                loss_scene_response_ce = session_category_balanced_scene_ce_loss(
                    scene_response_logits, batch['session_id'], batch['sample_type'],
                )

            if config.lambda_shared_response_bce > 0.0:
                if vad_logits is None or relation_addressing_logits is None:
                    raise RuntimeError('Shared response loss requires identity-memory VAD and addressing logits.')
                loss_shared_response_bce = shared_response_gate_loss(
                    model, vad_logits, relation_addressing_logits, batch,
                )

            if config.lambda_self_speaker_bce > 0.0:
                if self_speaking_logits is None:
                    raise RuntimeError("SELF-speaker loss requires turn-graph mode.")
                loss_self_speaker_bce = balanced_binary_bce_loss(
                    self_speaking_logits,
                    batch["wearer_speaking"],
                )

            if config.lambda_self_target_bce > 0.0:
                if self_target_logits is None:
                    raise RuntimeError("SELF-target loss requires turn-graph mode.")
                loss_self_target_bce = wearer_target_bce_loss(
                    self_target_logits,
                    batch["wearer_target_candidate_labels"],
                    batch["wearer_speaking"],
                )

            if config.lambda_candidate_classifier_ce > 0.0:
                if config.candidate_classifier_type == "temporal":
                    if fused_temporal_features is None:
                        raise RuntimeError("Temporal candidate classifier requested without temporal features.")
                    if classifier_logits is None:
                        classifier_logits = (
                            [logits[:-1] for logits in response_logits]
                            if response_logits is not None
                            else model.temporal_head_crop_candidate_logits(
                                fused_temporal_features,
                                batch["candidate_bboxes_sequence_heatmap"],
                                batch["candidate_head_crops"],
                            )
                            if config.use_head_crop_branch
                            else model.temporal_candidate_logits(
                                fused_temporal_features,
                                batch["candidate_bboxes_sequence_heatmap"],
                            )
                        )
                else:
                    classifier_logits = model.candidate_logits(fused_features, batch["candidate_bboxes_heatmap"])
                candidate_selection_logits = (
                    relation_speaking_logits
                    if relation_global_addressing_logits is not None
                    and relation_speaking_logits is not None
                    else classifier_logits
                )
                loss_candidate_classifier_ce = candidate_classifier_ce_loss(
                    candidate_selection_logits,
                    batch["candidate_labels"],
                )
                candidate_classifier_batch_top1 = candidate_classifier_top1(
                    candidate_selection_logits,
                    batch["candidate_labels"],
                )

            loss_temp = pred.new_tensor(0.0)
            temporal_mask = batch["apply_temporal_loss"]
            if config.use_temporal_loss and temporal_mask.any():
                pred_prev = model(
                    batch["prev_video_clip"][temporal_mask],
                    batch["prev_audio_spectrogram"][temporal_mask],
                    batch["prev_audio_channel_correlation"][temporal_mask],
                )
                loss_temp = temporal_loss(pred[temporal_mask], pred_prev, config.temporal_loss)

            loss = (
                config.lambda_heatmap * loss_main
                + config.lambda_temporal * loss_temp
                + config.lambda_candidate_ranking * loss_candidate
                + config.lambda_candidate_ce * loss_candidate_ce
                + config.lambda_candidate_classifier_ce * loss_candidate_classifier_ce
                + config.lambda_candidate_target_bce * loss_candidate_target_bce
                + config.lambda_candidate_speaker_bce * loss_candidate_speaker_bce
                + config.lambda_candidate_speaker_ce * loss_candidate_speaker_ce
                + config.lambda_vad_bce * loss_vad_bce
                + config.lambda_shared_response_bce * loss_shared_response_bce
                + config.lambda_scene_response_ce * loss_scene_response_ce
                + config.lambda_self_speaker_bce * loss_self_speaker_bce
                + config.lambda_self_target_bce * loss_self_target_bce
                + config.lambda_response_ce * loss_response_ce
            )

            optimizer_updated = False
            if use_identity_tbptt:
                if identity_state_cache is None or identity_updated_states is None:
                    raise RuntimeError("Identity TBPTT requires updated streaming identity states.")
                identity_state_cache.update(
                    batch["session_id"],
                    batch["timestamp"],
                    identity_updated_states,
                    detach=False,
                )
                pending_tbptt_losses.append(loss)
                reached_run_end = max_steps is not None and completed_steps >= max_steps
                reached_epoch_end = step == len(dataloader)
                reached_segment_end = len(pending_tbptt_losses) >= config.identity_bptt_windows
                if reached_segment_end or reached_epoch_end or reached_run_end:
                    segment_loss = torch.stack(pending_tbptt_losses).mean()
                    optimizer.zero_grad(set_to_none=True)
                    if segment_loss.requires_grad:
                        segment_loss.backward()
                        optimizer.step()
                    identity_state_cache.detach()
                    pending_tbptt_losses.clear()
                    optimizer_updated = True
            else:
                if loss.requires_grad:
                    loss.backward()
                accumulated_microsteps += 1
                if identity_state_cache is not None and identity_updated_states is not None:
                    identity_state_cache.update(
                        batch["session_id"],
                        batch["timestamp"],
                        identity_updated_states,
                    )
                reached_run_end = max_steps is not None and completed_steps >= max_steps
                reached_epoch_end = step == len(dataloader)
                reached_accumulation_end = accumulated_microsteps >= config.gradient_accumulation_steps
                if reached_accumulation_end or reached_epoch_end or reached_run_end:
                    if accumulated_microsteps > 1:
                        for parameter in trainable_parameters:
                            if parameter.grad is not None:
                                parameter.grad.div_(float(accumulated_microsteps))
                    if loss.requires_grad:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    accumulated_microsteps = 0
                    optimizer_updated = True
                    optimizer_steps_completed += 1

            if turn_graph_cache is not None:
                if (
                    relation_speaking_logits is None
                    or relation_addressing_logits is None
                    or self_speaking_logits is None
                    or self_target_logits is None
                ):
                    raise RuntimeError("Turn-graph state update requires all graph prediction heads.")
                turn_graph_cache.update(
                    batch["session_id"],
                    batch["timestamp"],
                    batch["candidate_ids"],
                    relation_speaking_logits,
                    relation_addressing_logits,
                    self_speaking_logits,
                    self_target_logits,
                    speaker_labels=batch["speaker_candidate_labels"],
                    response_labels=batch["response_candidate_labels"],
                    self_speaking_labels=batch["wearer_speaking"],
                    self_target_labels=batch["wearer_target_candidate_labels"],
                    teacher_forcing=config.turn_graph_teacher_forcing,
                )

            running_loss += float(loss.detach().cpu())
            if torch_device.type == "cuda":
                torch.cuda.synchronize(torch_device)

            step_seconds = time.perf_counter() - step_start
            previous_step_end = time.perf_counter()
            is_log_step = (
                completed_steps == 1
                or completed_steps == planned_steps
                or step == total_steps
                or completed_steps % max(1, log_every) == 0
            )
            if is_log_step:
                elapsed_seconds = time.perf_counter() - start_time
                progress = {
                    "epoch": epoch + 1,
                    "epochs": config.epochs,
                    "step": global_step,
                    "run_step": completed_steps,
                    "steps_per_epoch": total_steps,
                    "planned_steps": planned_steps,
                    "step_offset": int(step_offset),
                    "samples": len(dataset),
                    "batch_size": config.batch_size,
                    "gradient_accumulation_steps": config.gradient_accumulation_steps,
                    "optimizer_step": (
                        int(step_offset) // config.gradient_accumulation_steps
                        + optimizer_steps_completed
                    ),
                    "loss": running_loss / completed_steps,
                    "heatmap": float(loss_main.detach().cpu()),
                    "temporal": float(loss_temp.detach().cpu()),
                    "candidate": float(loss_candidate.detach().cpu()),
                    "candidate_ce": float(loss_candidate_ce.detach().cpu()),
                    "candidate_classifier_ce": float(loss_candidate_classifier_ce.detach().cpu()),
                    "candidate_classifier_top1": candidate_classifier_batch_top1,
                    "candidate_target_bce": float(loss_candidate_target_bce.detach().cpu()),
                    "candidate_speaker_bce": float(loss_candidate_speaker_bce.detach().cpu()),
                    "candidate_speaker_ce": float(loss_candidate_speaker_ce.detach().cpu()),
                    "vad_bce": float(loss_vad_bce.detach().cpu()),
                    "self_speaker_bce": float(loss_self_speaker_bce.detach().cpu()),
                    "self_target_bce": float(loss_self_target_bce.detach().cpu()),
                    "speaker_accuracy": speaker_batch_metrics["accuracy"],
                    "speaker_recall": speaker_batch_metrics["recall"],
                    "speaker_specificity": speaker_batch_metrics["specificity"],
                    "target_top1": target_batch_metrics["top1"],
                    "target_addressed_detected": target_batch_metrics["addressed_detected"],
                    "target_addressed_correct": target_batch_metrics["addressed_correct"],
                    "target_non_target_rejection": target_batch_metrics["non_target_rejection"],
                    "response_ce": float(loss_response_ce.detach().cpu()),
                    "response_top1": response_batch_metrics["top1"],
                    "addressed_correct": response_batch_metrics["addressed_correct"],
                    "non_target_rejection": response_batch_metrics["non_target_rejection"],
                    "candidate_classifier_type": config.candidate_classifier_type,
                    "data_seconds": data_seconds,
                    "step_seconds": step_seconds,
                    "elapsed_seconds": elapsed_seconds,
                    "device": str(torch_device),
                    "output": str(output),
                }
                write_json_atomic(progress_output, progress)
                print(
                    f"epoch={epoch + 1}/{config.epochs} step={global_step}/{total_steps} "
                    f"run_step={completed_steps}/{planned_steps} "
                    f"loss={running_loss / completed_steps:.4f} "
                    f"heatmap={float(loss_main.detach().cpu()):.4f} "
                    f"temporal={float(loss_temp.detach().cpu()):.4f} "
                    f"candidate={float(loss_candidate.detach().cpu()):.4f} "
                    f"candidate_ce={float(loss_candidate_ce.detach().cpu()):.4f} "
                    f"candidate_classifier_ce={float(loss_candidate_classifier_ce.detach().cpu()):.4f} "
                    f"candidate_classifier_top1={candidate_classifier_batch_top1} "
                    f"candidate_target_bce={float(loss_candidate_target_bce.detach().cpu()):.4f} "
                    f"candidate_speaker_bce={float(loss_candidate_speaker_bce.detach().cpu()):.4f} "
                    f"candidate_speaker_ce={float(loss_candidate_speaker_ce.detach().cpu()):.4f} "
                    f"vad_bce={float(loss_vad_bce.detach().cpu()):.4f} "
                    f"self_speaker_bce={float(loss_self_speaker_bce.detach().cpu()):.4f} "
                    f"self_target_bce={float(loss_self_target_bce.detach().cpu()):.4f} "
                    f"speaker_recall={speaker_batch_metrics['recall']} "
                    f"target_top1={target_batch_metrics['top1']} "
                    f"target_addressed_detected={target_batch_metrics['addressed_detected']} "
                    f"target_addressed_correct={target_batch_metrics['addressed_correct']} "
                    f"target_non_target_rejection={target_batch_metrics['non_target_rejection']} "
                    f"response_ce={float(loss_response_ce.detach().cpu()):.4f} "
                    f"response_top1={response_batch_metrics['top1']} "
                    f"addressed_correct={response_batch_metrics['addressed_correct']} "
                    f"non_target_rejection={response_batch_metrics['non_target_rejection']} "
                    f"data_s={data_seconds:.2f} step_s={step_seconds:.2f} elapsed_s={elapsed_seconds:.1f}",
                    flush=True,
                )

            if checkpoint_every > 0 and global_step % checkpoint_every == 0 and optimizer_updated:
                checkpoint = output.with_name(f"{output.stem}_epoch{epoch + 1}_step{global_step}{output.suffix}")
                elapsed_seconds = time.perf_counter() - start_time
                save_checkpoint_atomic(
                    checkpoint,
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    epoch=epoch + 1,
                    step=global_step,
                    elapsed_seconds=elapsed_seconds,
                    identity_state_cache=identity_state_cache,
                    stateful_epoch_index=last_stateful_epoch_index,
                    stateful_batches_completed=last_stateful_batches_completed,
                    turn_graph_cache=turn_graph_cache,
                )
                print(f"saved interim checkpoint={checkpoint}", flush=True)
        if max_steps is not None and completed_steps >= max_steps:
            break

    save_checkpoint_atomic(
        output,
        model=model,
        optimizer=optimizer,
        config=config,
        epoch=config.epochs,
        step=last_global_step,
        elapsed_seconds=time.perf_counter() - start_time,
        identity_state_cache=identity_state_cache,
        stateful_epoch_index=last_stateful_epoch_index,
        stateful_batches_completed=last_stateful_batches_completed,
        turn_graph_cache=turn_graph_cache,
    )
    print(f"saved checkpoint={output}", flush=True)


def main() -> None:
    args = build_arg_parser().parse_args()
    config = EasyComSAALConfig(
        easycom_root=args.easycom_root,
        annotation_manifest=args.manifest,
        use_temporal_loss=not args.no_temporal_loss,
        keep_only_tos_to_camera_wearer=not (
            args.include_non_target_speech or args.use_timeline_multilabels
        ),
        use_timeline_multilabels=args.use_timeline_multilabels,
        include_silence_windows=not args.exclude_silence_windows,
    )
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.gradient_accumulation_steps is not None:
        config.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.label_min_overlap_sec is not None:
        config.label_min_overlap_sec = args.label_min_overlap_sec
    if args.lambda_candidate_ranking is not None:
        config.lambda_candidate_ranking = args.lambda_candidate_ranking
    if args.lambda_candidate_ce is not None:
        config.lambda_candidate_ce = args.lambda_candidate_ce
    if args.lambda_candidate_classifier_ce is not None:
        config.lambda_candidate_classifier_ce = args.lambda_candidate_classifier_ce
    if args.lambda_candidate_target_bce is not None:
        config.lambda_candidate_target_bce = args.lambda_candidate_target_bce
    if args.lambda_candidate_speaker_bce is not None:
        config.lambda_candidate_speaker_bce = args.lambda_candidate_speaker_bce
    if args.lambda_candidate_speaker_ce is not None:
        config.lambda_candidate_speaker_ce = args.lambda_candidate_speaker_ce
    if args.weight_speaker_ce_by_candidate_count:
        config.weight_speaker_ce_by_candidate_count = True
    if args.lambda_vad_bce is not None:
        config.lambda_vad_bce = args.lambda_vad_bce
    if args.lambda_shared_response_bce is not None:
        config.lambda_shared_response_bce = args.lambda_shared_response_bce
    if args.use_shared_response_gate:
        config.use_shared_response_gate = True
    if args.shared_response_finetune:
        config.shared_response_finetune = True
    if args.session_category_balanced_response_loss:
        config.use_session_category_balanced_response_loss = True
    if args.lambda_scene_response_ce is not None:
        config.lambda_scene_response_ce = args.lambda_scene_response_ce
    if args.use_scene_response_head:
        config.use_scene_response_head = True
    if args.scene_response_finetune:
        config.scene_response_finetune = True
    if args.candidate_target_threshold is not None:
        config.candidate_target_threshold = args.candidate_target_threshold
    if args.vad_threshold is not None:
        config.vad_threshold = args.vad_threshold
    if args.lambda_response_ce is not None:
        config.lambda_response_ce = args.lambda_response_ce
    if args.response_addressed_weight is not None:
        config.response_addressed_weight = args.response_addressed_weight
    if args.lambda_heatmap is not None:
        config.lambda_heatmap = args.lambda_heatmap
    if args.candidate_ranking_margin is not None:
        config.candidate_ranking_margin = args.candidate_ranking_margin
    if args.candidate_ce_temperature is not None:
        config.candidate_ce_temperature = args.candidate_ce_temperature
    if args.candidate_topk_fraction is not None:
        config.candidate_topk_fraction = args.candidate_topk_fraction
    if args.candidate_topk_min_pixels is not None:
        config.candidate_topk_min_pixels = args.candidate_topk_min_pixels
    if args.candidate_classifier_type is not None:
        config.candidate_classifier_type = args.candidate_classifier_type
    if args.freeze_base_for_candidate_classifier:
        config.freeze_base_for_candidate_classifier = True
    if args.unfreeze_fusion_for_candidate_classifier:
        config.unfreeze_fusion_for_candidate_classifier = True
    if args.unfreeze_video_last_block_for_candidate_classifier:
        config.unfreeze_video_last_block_for_candidate_classifier = True
    if args.unfreeze_audio_encoder_for_candidate_classifier:
        config.unfreeze_audio_encoder_for_candidate_classifier = True
    if args.fusion_lr is not None:
        config.fusion_learning_rate = args.fusion_lr
    if args.video_last_block_lr is not None:
        config.video_last_block_learning_rate = args.video_last_block_lr
    if args.audio_encoder_lr is not None:
        config.audio_encoder_learning_rate = args.audio_encoder_lr
    if args.use_head_crop_branch:
        config.use_head_crop_branch = True
    if args.use_relation_candidate_head:
        config.use_relation_candidate_head = True
    if args.use_independent_candidate_head:
        config.use_independent_candidate_head = True
    if args.use_global_address_gate:
        config.use_global_address_gate = True
    if args.use_selected_speaker_addressing:
        config.use_selected_speaker_addressing = True
    if args.use_candidate_address_memory:
        config.use_candidate_address_memory = True
    if args.use_candidate_identity_memory:
        config.use_candidate_identity_memory = True
    if args.identity_bptt_windows is not None:
        config.identity_bptt_windows = args.identity_bptt_windows
    if args.use_turn_graph:
        config.use_turn_graph = True
    if args.turn_graph_decay_sec is not None:
        config.turn_graph_decay_sec = args.turn_graph_decay_sec
    if args.turn_graph_reset_gap_sec is not None:
        config.turn_graph_reset_gap_sec = args.turn_graph_reset_gap_sec
    if args.turn_graph_teacher_forcing is not None:
        config.turn_graph_teacher_forcing = args.turn_graph_teacher_forcing
    if args.turn_graph_other_only:
        config.turn_graph_other_only = True
    if args.lambda_self_speaker_bce is not None:
        config.lambda_self_speaker_bce = args.lambda_self_speaker_bce
    if args.lambda_self_target_bce is not None:
        config.lambda_self_target_bce = args.lambda_self_target_bce
    if args.use_causal_memory:
        config.use_causal_memory = True
    if args.use_frame_aligned_audio:
        config.use_frame_aligned_audio = True
    if args.preserve_audio_energy:
        config.preserve_audio_energy = True
    if args.audio_global_gain_augmentation_db is not None:
        config.audio_global_gain_augmentation_db = args.audio_global_gain_augmentation_db
    if args.audio_channel_gain_std_db is not None:
        config.audio_channel_gain_std_db = args.audio_channel_gain_std_db
    if args.audio_channel_gain_per_recording:
        config.audio_channel_gain_per_recording = True
    if args.audio_perturbation_seed is not None:
        config.audio_perturbation_seed = args.audio_perturbation_seed
    if args.use_causal_channel_rms_normalization:
        config.use_causal_channel_rms_normalization = True
    if args.audio_normalization_tau_sec is not None:
        config.audio_normalization_tau_sec = args.audio_normalization_tau_sec
    if args.audio_normalization_target_rms_db is not None:
        config.audio_normalization_target_rms_db = args.audio_normalization_target_rms_db
    if args.audio_normalization_max_gain_db is not None:
        config.audio_normalization_max_gain_db = args.audio_normalization_max_gain_db
    if args.audio_robustness_heads_only:
        config.audio_robustness_heads_only = True
    if args.speaker_representation_finetune:
        config.speaker_representation_finetune = True
    if args.speaker_residual_finetune:
        config.speaker_residual_finetune = True
    if args.global_address_finetune:
        config.global_address_finetune = True
    if args.global_address_adapter_finetune:
        config.global_address_adapter_finetune = True
    if args.global_address_positive_weight is not None:
        config.global_address_positive_weight = args.global_address_positive_weight
    if args.addressing_hard_negative_finetune:
        config.addressing_hard_negative_finetune = True
    if args.addressing_negative_weight is not None:
        config.addressing_negative_weight = args.addressing_negative_weight
    if args.addressing_hard_negative_fraction is not None:
        config.addressing_hard_negative_fraction = args.addressing_hard_negative_fraction
    if args.addressing_hard_negative_weight is not None:
        config.addressing_hard_negative_weight = args.addressing_hard_negative_weight
    if args.relation_adaptation_finetune:
        config.relation_adaptation_finetune = True
    if args.relation_aux_lr is not None:
        config.relation_aux_learning_rate = args.relation_aux_lr
    if args.use_masked_head_video:
        config.use_masked_head_video = True
    if args.streaming_smoothing_windows is not None:
        config.streaming_smoothing_windows = args.streaming_smoothing_windows
    if config.lambda_candidate_target_bce > 0.0 and config.lambda_response_ce > 0.0:
        raise ValueError("Candidate target BCE and N+1 response CE are alternative response objectives.")
    if config.use_head_crop_branch and config.candidate_classifier_type != "temporal":
        raise ValueError("The head-crop branch requires the temporal candidate classifier.")
    if config.use_relation_candidate_head and config.candidate_classifier_type != "temporal":
        raise ValueError("The relation candidate head requires the temporal candidate classifier.")
    if config.use_independent_candidate_head and config.candidate_classifier_type != "temporal":
        raise ValueError("The independent candidate head requires the temporal candidate classifier.")
    if config.use_relation_candidate_head and config.use_independent_candidate_head:
        raise ValueError("Relation and independent candidate heads are alternative modes.")
    if config.use_relation_candidate_head and config.use_head_crop_branch:
        raise ValueError("The relation and isolated head-crop branches are alternative candidate representations.")
    if config.use_independent_candidate_head and config.use_head_crop_branch:
        raise ValueError("The independent and isolated head-crop branches are alternative candidate representations.")
    if config.lambda_candidate_speaker_bce > 0.0 and not (
        config.use_relation_candidate_head or config.use_independent_candidate_head
    ):
        raise ValueError("Speaker auxiliary loss requires a relation or independent multitask candidate head.")
    if config.lambda_candidate_speaker_ce > 0.0 and not (
        config.use_relation_candidate_head or config.use_independent_candidate_head
    ):
        raise ValueError("Speaker CE requires a relation or independent multitask candidate head.")
    if config.use_selected_speaker_addressing and not (
        config.use_relation_candidate_head or config.use_independent_candidate_head
    ):
        raise ValueError("Selected-speaker addressing requires a relation candidate head.")
    if config.use_selected_speaker_addressing and config.use_global_address_gate:
        raise ValueError("Selected-speaker conditional addressing and the global address gate are alternative modes.")
    if config.use_global_address_gate and not config.use_relation_candidate_head:
        raise ValueError("The global address gate requires --use-relation-candidate-head.")
    if config.use_candidate_address_memory and not config.use_relation_candidate_head:
        raise ValueError("Candidate address memory requires --use-relation-candidate-head.")
    if config.use_candidate_identity_memory and not config.use_relation_candidate_head:
        raise ValueError("Candidate identity memory requires --use-relation-candidate-head.")
    if config.use_candidate_address_memory and (
        config.use_global_address_gate or config.use_candidate_identity_memory
    ):
        raise ValueError("Shared candidate memory cannot be combined with global or identity memory.")
    if config.use_causal_memory and not (
        config.use_global_address_gate
        or config.use_candidate_address_memory
        or config.use_candidate_identity_memory
    ):
        raise ValueError("Causal memory requires a global or candidate address-memory mode.")
    if config.use_candidate_identity_memory and not config.use_causal_memory:
        raise ValueError("Candidate identity memory requires --use-causal-memory.")
    if config.use_turn_graph and not (
        config.use_candidate_identity_memory and config.use_causal_memory
    ):
        raise ValueError("Turn graph requires stateful candidate identity memory.")
    if config.use_turn_graph and config.use_global_address_gate:
        raise ValueError("Turn graph and global address gate are alternative addressing modes.")
    if config.use_turn_graph and not config.use_timeline_multilabels:
        raise ValueError("Turn graph requires timeline multi-label samples.")
    if config.turn_graph_decay_sec <= 0.0 or config.turn_graph_reset_gap_sec <= 0.0:
        raise ValueError("Turn-graph decay and reset gaps must be positive.")
    if not 0.0 <= config.turn_graph_teacher_forcing <= 1.0:
        raise ValueError("Turn-graph teacher forcing must be between 0 and 1.")
    if config.turn_graph_other_only and not config.use_turn_graph:
        raise ValueError("Turn-graph OTHER-only fine-tuning requires --use-turn-graph.")
    if (config.lambda_self_speaker_bce > 0.0 or config.lambda_self_target_bce > 0.0) and not config.use_turn_graph:
        raise ValueError("SELF speaker/target supervision requires --use-turn-graph.")
    if config.identity_bptt_windows < 1:
        raise ValueError("Identity BPTT windows must be at least 1.")
    if config.gradient_accumulation_steps < 1:
        raise ValueError("Gradient accumulation steps must be at least 1.")
    if config.gradient_accumulation_steps > 1 and config.identity_bptt_windows > 1:
        raise ValueError("Gradient accumulation and identity BPTT cannot both exceed 1.")
    if config.identity_bptt_windows > 1 and not (
        config.use_candidate_identity_memory and config.use_causal_memory
    ):
        raise ValueError("Identity BPTT requires stateful candidate identity memory.")
    if config.lambda_vad_bce > 0.0 and not config.use_candidate_identity_memory:
        raise ValueError("VAD supervision requires --use-candidate-identity-memory.")
    if config.streaming_smoothing_windows < 1:
        raise ValueError("Streaming smoothing windows must be at least 1.")
    if config.audio_global_gain_augmentation_db < 0.0:
        raise ValueError("Global gain augmentation range must be non-negative.")
    if config.audio_channel_gain_std_db < 0.0:
        raise ValueError("Channel gain augmentation standard deviation must be non-negative.")
    if config.audio_normalization_tau_sec <= 0.0:
        raise ValueError("Audio normalization time constant must be positive.")
    if config.audio_normalization_max_gain_db < 0.0:
        raise ValueError("Audio normalization maximum gain must be non-negative.")
    if config.use_causal_channel_rms_normalization and config.num_workers != 0:
        raise ValueError("Causal channel RMS normalization requires --num-workers 0.")
    if config.addressing_negative_weight <= 0.0:
        raise ValueError("Addressing negative weight must be positive.")
    if not 0.0 <= config.addressing_hard_negative_fraction <= 1.0:
        raise ValueError("Addressing hard-negative fraction must be between 0 and 1.")
    if config.addressing_hard_negative_weight < 1.0:
        raise ValueError("Addressing hard-negative weight must be at least 1.")
    if config.relation_aux_learning_rate <= 0.0:
        raise ValueError("Relation auxiliary-head learning rate must be positive.")
    if config.global_address_positive_weight <= 0.0:
        raise ValueError("Global addressedness positive weight must be positive.")
    if config.global_address_finetune and config.global_address_adapter_finetune:
        raise ValueError("Choose either global head-only or global adapter fine-tuning.")
    if (config.global_address_finetune or config.global_address_adapter_finetune) and not (
        config.use_relation_candidate_head
        and config.use_candidate_identity_memory
        and config.use_causal_memory
        and config.use_global_address_gate
        and config.lambda_candidate_target_bce > 0.0
    ):
        raise ValueError(
            "Global addressedness fine-tuning requires causal identity relation global-gate training."
        )
    if config.addressing_hard_negative_finetune and not (
        config.use_relation_candidate_head
        and config.use_candidate_identity_memory
        and config.use_causal_memory
        and config.lambda_candidate_target_bce > 0.0
    ):
        raise ValueError(
            "Addressing hard-negative fine-tuning requires causal identity-relation target BCE training."
        )
    if config.addressing_hard_negative_finetune and (
        config.audio_robustness_heads_only
        or config.speaker_representation_finetune
        or config.speaker_residual_finetune
        or config.turn_graph_other_only
        or config.relation_adaptation_finetune
    ):
        raise ValueError("Addressing hard-negative fine-tuning cannot be combined with another heads-only mode.")
    if config.scene_response_finetune and not (
        config.use_scene_response_head
        and config.use_relation_candidate_head
        and config.use_candidate_identity_memory
        and config.use_causal_memory
        and config.lambda_scene_response_ce > 0.0
    ):
        raise ValueError('Scene response fine-tuning requires causal identity memory and scene-response CE.')
    if config.use_scene_response_head and config.lambda_scene_response_ce <= 0.0:
        raise ValueError('The scene response head requires a positive scene-response CE weight.')
    if config.shared_response_finetune and not (
        config.use_shared_response_gate
        and config.use_relation_candidate_head
        and config.use_candidate_identity_memory
        and config.use_causal_memory
        and config.use_selected_speaker_addressing
        and config.lambda_shared_response_bce > 0.0
    ):
        raise ValueError('Shared response fine-tuning requires the causal identity relation addressing path and its BCE loss.')
    if config.use_shared_response_gate and config.lambda_shared_response_bce <= 0.0:
        raise ValueError('The shared response gate requires a positive shared-response BCE weight.')
    if config.relation_adaptation_finetune and not (
        config.use_relation_candidate_head
        and config.use_candidate_identity_memory
        and config.use_causal_memory
        and config.lambda_candidate_target_bce > 0.0
        and config.lambda_candidate_speaker_bce > 0.0
        and config.lambda_vad_bce > 0.0
    ):
        raise ValueError(
            "Relation adaptation requires causal identity-relation target, speaker, and VAD supervision."
        )
    if config.relation_adaptation_finetune and (
        config.audio_robustness_heads_only
        or config.speaker_representation_finetune
        or config.speaker_residual_finetune
        or config.turn_graph_other_only
    ):
        raise ValueError("Relation adaptation cannot be combined with another specialized fine-tuning mode.")
    if config.audio_robustness_heads_only and not (
        config.use_relation_candidate_head
        and config.use_candidate_identity_memory
        and config.lambda_vad_bce > 0.0
        and config.lambda_candidate_speaker_bce > 0.0
    ):
        raise ValueError("Audio robustness heads-only mode requires identity relation VAD/speaker training.")
    if config.speaker_representation_finetune and not (
        config.use_relation_candidate_head
        and config.use_candidate_identity_memory
        and config.use_causal_memory
        and config.lambda_candidate_speaker_ce > 0.0
    ):
        raise ValueError(
            "Speaker representation fine-tuning requires causal identity relation speaker CE training."
        )
    if config.speaker_representation_finetune and (
        config.audio_robustness_heads_only
        or config.addressing_hard_negative_finetune
        or config.speaker_residual_finetune
        or config.turn_graph_other_only
    ):
        raise ValueError("Speaker representation fine-tuning cannot be combined with another specialized mode.")
    if config.speaker_residual_finetune and not (
        config.use_relation_candidate_head
        and config.use_candidate_identity_memory
        and config.use_causal_memory
        and config.lambda_candidate_speaker_ce > 0.0
    ):
        raise ValueError(
            "Speaker residual fine-tuning requires causal identity relation speaker CE training."
        )
    if config.speaker_residual_finetune and (
        config.audio_robustness_heads_only
        or config.addressing_hard_negative_finetune
        or config.turn_graph_other_only
        or config.relation_adaptation_finetune
    ):
        raise ValueError("Speaker residual fine-tuning cannot be combined with another specialized mode.")
    train(
        config,
        output_path=args.output,
        device=args.device,
        log_every=args.log_every,
        progress_path=args.progress_path,
        checkpoint_every=args.checkpoint_every,
        resume_checkpoint=args.resume_checkpoint,
        max_steps=args.max_steps,
        step_offset=args.step_offset,
        sample_start=args.sample_start,
        sample_end=args.sample_end,
        sample_order_seed=args.sample_order_seed,
        reset_stateful_progress=args.reset_stateful_progress,
    )


if __name__ == "__main__":
    main()
