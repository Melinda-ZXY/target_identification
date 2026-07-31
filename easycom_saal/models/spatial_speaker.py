from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn


PAIR_INDICES = tuple((left, right) for left in range(6) for right in range(left + 1, 6))


class SpatialSpeakerMatcher(nn.Module):
    """Match signed multichannel time delays to candidate image geometry."""

    def __init__(
        self,
        audio_dim: int,
        bbox_dim: int,
        audio_mean: torch.Tensor | None = None,
        audio_std: torch.Tensor | None = None,
        hidden_dim: int = 64,
        sample_rate: int = 16_000,
        max_lag_ms: float = 1.0,
    ) -> None:
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.max_lag_ms = float(max_lag_ms)
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
        self.register_buffer(
            "audio_mean",
            torch.zeros(1, audio_dim) if audio_mean is None else audio_mean.float().reshape(1, -1),
        )
        self.register_buffer(
            "audio_std",
            torch.ones(1, audio_dim) if audio_std is None else audio_std.float().reshape(1, -1),
        )

    def score_features(
        self,
        audio: torch.Tensor,
        bbox: torch.Tensor,
        candidate_mask: torch.Tensor,
        normalize_audio: bool = True,
    ) -> torch.Tensor:
        if normalize_audio:
            audio = (audio - self.audio_mean) / self.audio_std.clamp_min(1e-4)
        audio_embedding = self.audio(audio)
        candidate_embedding = self.candidate(bbox)
        scores = torch.einsum("bd,bkd->bk", audio_embedding, candidate_embedding)
        scores = scores / math.sqrt(float(audio_embedding.shape[-1]))
        scores = scores + self.geometry_bias(bbox).squeeze(-1)
        return scores.masked_fill(~candidate_mask, -1e4)

    def forward_waveform(
        self,
        waveform: torch.Tensor,
        candidate_sequences: list[torch.Tensor],
        heatmap_size: tuple[int, int],
    ) -> list[torch.Tensor]:
        audio = self.signed_gcc_phat(waveform)
        bbox, candidate_mask = self.bbox_motion_features(candidate_sequences, heatmap_size)
        scores = self.score_features(audio, bbox, candidate_mask)
        return [row[: sequence.shape[0]] for row, sequence in zip(scores, candidate_sequences)]

    def signed_gcc_phat(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 3 or waveform.shape[1] != 6:
            raise ValueError(f"Expected waveform [B,6,N], got {tuple(waveform.shape)}")
        centered = waveform.float() - waveform.float().mean(dim=-1, keepdim=True)
        window = torch.hann_window(
            centered.shape[-1],
            periodic=False,
            device=centered.device,
            dtype=centered.dtype,
        )
        centered = centered * window
        fft_size = 1 << int(math.ceil(math.log2(max(2, centered.shape[-1]))))
        spectrum = torch.fft.rfft(centered, n=fft_size, dim=-1)
        left = spectrum[:, [pair[0] for pair in PAIR_INDICES]]
        right = spectrum[:, [pair[1] for pair in PAIR_INDICES]]
        cross_power = left * right.conj()
        cross_power = cross_power / cross_power.abs().clamp_min(1e-8)
        correlation = torch.fft.irfft(cross_power, n=fft_size, dim=-1)
        max_lag = max(1, int(round(self.max_lag_ms * self.sample_rate / 1000.0)))
        signed = torch.cat(
            [correlation[..., -max_lag:], correlation[..., : max_lag + 1]],
            dim=-1,
        )
        signed = signed / torch.linalg.vector_norm(signed, dim=-1, keepdim=True).clamp_min(1e-6)
        return signed.flatten(start_dim=1)

    @staticmethod
    def bbox_motion_features(
        candidate_sequences: list[torch.Tensor],
        heatmap_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not candidate_sequences:
            raise ValueError("At least one candidate-sequence batch is required.")
        height, width = heatmap_size
        max_candidates = max(sequence.shape[0] for sequence in candidate_sequences)
        bbox_dim = 12
        output = candidate_sequences[0].new_zeros(
            (len(candidate_sequences), max_candidates, bbox_dim)
        )
        mask = torch.zeros(
            (len(candidate_sequences), max_candidates),
            dtype=torch.bool,
            device=output.device,
        )
        for index, sequence in enumerate(candidate_sequences):
            boxes = sequence.float().clone()
            boxes[..., [0, 2]] /= float(width)
            boxes[..., [1, 3]] /= float(height)
            centers = torch.stack(
                [(boxes[..., 0] + boxes[..., 2]) * 0.5, (boxes[..., 1] + boxes[..., 3]) * 0.5],
                dim=-1,
            )
            sizes = torch.stack(
                [boxes[..., 2] - boxes[..., 0], boxes[..., 3] - boxes[..., 1]],
                dim=-1,
            )
            geometry = torch.cat([centers, sizes], dim=-1)
            middle = geometry[:, geometry.shape[1] // 2]
            mean = geometry.mean(dim=1)
            center_std = centers.std(dim=1, unbiased=False)
            center_delta = centers[:, -1] - centers[:, 0]
            features = torch.cat([middle, mean, center_std, center_delta], dim=1)
            output[index, : sequence.shape[0]] = features
            mask[index, : sequence.shape[0]] = True
        return output, mask


def load_spatial_speaker_matcher(
    checkpoint_path: str | Path,
    device: torch.device,
) -> SpatialSpeakerMatcher:
    checkpoint: dict[str, Any] = torch.load(checkpoint_path, map_location="cpu")
    model = SpatialSpeakerMatcher(
        audio_dim=int(checkpoint["audio_dim"]),
        bbox_dim=int(checkpoint["bbox_dim"]),
        audio_mean=torch.as_tensor(checkpoint["audio_mean"]),
        audio_std=torch.as_tensor(checkpoint["audio_std"]),
        sample_rate=int(checkpoint["sample_rate"]),
        max_lag_ms=float(checkpoint["max_lag_ms"]),
    )
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    allowed_missing = {"audio_mean", "audio_std"}
    if set(missing) - allowed_missing or unexpected:
        raise RuntimeError(
            f"Invalid spatial speaker checkpoint missing={missing} unexpected={unexpected}"
        )
    return model.to(device).eval()
