from __future__ import annotations

from itertools import combinations
from pathlib import Path
import math

import torch
import torch.nn.functional as F


class CausalChannelRMSNormalizer:
    """Slowly equalize recording and channel gain using past windows only."""

    def __init__(
        self,
        num_channels: int,
        tau_sec: float = 30.0,
        target_rms_db: float = -44.0,
        max_gain_db: float = 6.0,
        eps: float = 1e-8,
    ) -> None:
        self.num_channels = int(num_channels)
        self.tau_sec = float(tau_sec)
        self.target_rms_db = float(target_rms_db)
        self.max_gain_db = float(max_gain_db)
        self.eps = float(eps)
        self.baseline_db: torch.Tensor | None = None
        self.last_timestamp: float | None = None

    def reset(self) -> None:
        self.baseline_db = None
        self.last_timestamp = None

    def __call__(self, waveform: torch.Tensor, timestamp: float | None = None) -> torch.Tensor:
        if waveform.ndim != 2 or waveform.shape[0] != self.num_channels:
            raise ValueError(
                f"Expected [{self.num_channels},samples] waveform, got {tuple(waveform.shape)}"
            )
        current_db = 20.0 * torch.log10(
            waveform.square().mean(dim=-1).sqrt().clamp_min(self.eps)
        )
        timestamp_value = None if timestamp is None else float(timestamp)
        if (
            self.last_timestamp is not None
            and timestamp_value is not None
            and timestamp_value <= self.last_timestamp
        ):
            self.reset()

        baseline = current_db if self.baseline_db is None else self.baseline_db.to(current_db)
        gain_db = (self.target_rms_db - baseline).clamp(
            min=-self.max_gain_db,
            max=self.max_gain_db,
        )
        normalized = waveform * torch.pow(10.0, gain_db[:, None] / 20.0)

        if self.baseline_db is None:
            updated = current_db
        else:
            delta_sec = (
                max(0.0, timestamp_value - self.last_timestamp)
                if timestamp_value is not None and self.last_timestamp is not None
                else 1.0
            )
            alpha = 1.0 if self.tau_sec <= 0.0 else 1.0 - math.exp(-delta_sec / self.tau_sec)
            updated = baseline + alpha * (current_db - baseline)
        self.baseline_db = updated.detach().cpu()
        self.last_timestamp = timestamp_value
        return normalized.contiguous()


def channel_pairs(num_channels: int) -> list[tuple[int, int]]:
    return list(combinations(range(num_channels), 2))


def _load_with_torchaudio(path: Path) -> tuple[torch.Tensor, int]:
    import torchaudio

    waveform, sample_rate = torchaudio.load(str(path))
    return waveform, int(sample_rate)


def _load_with_soundfile(path: Path) -> tuple[torch.Tensor, int]:
    import soundfile as sf

    samples, sample_rate = sf.read(str(path), always_2d=True)
    waveform = torch.from_numpy(samples).float().transpose(0, 1)
    return waveform, int(sample_rate)


def resample_audio(waveform: torch.Tensor, source_rate: int, target_rate: int) -> torch.Tensor:
    if source_rate == target_rate:
        return waveform
    try:
        import torchaudio

        return torchaudio.functional.resample(waveform, source_rate, target_rate)
    except Exception:
        old_len = waveform.shape[-1]
        new_len = int(round(old_len * float(target_rate) / float(source_rate)))
        resized = F.interpolate(
            waveform.unsqueeze(0),
            size=new_len,
            mode="linear",
            align_corners=False,
        )
        return resized.squeeze(0)


def load_audio_6ch(
    path: str | Path,
    expected_channels: int = 6,
    target_sample_rate: int | None = None,
) -> tuple[torch.Tensor, int]:
    path = Path(path)
    try:
        waveform, sample_rate = _load_with_soundfile(path)
    except Exception:
        waveform, sample_rate = _load_with_torchaudio(path)

    if waveform.ndim != 2:
        raise ValueError(f"Expected [channels, samples] audio, got shape {tuple(waveform.shape)}")
    if waveform.shape[0] < expected_channels:
        raise ValueError(
            f"Expected at least {expected_channels} audio channels, got {waveform.shape[0]} from {path}"
        )
    if waveform.shape[0] > expected_channels:
        waveform = waveform[:expected_channels]

    waveform = waveform.float()
    if target_sample_rate is not None:
        waveform = resample_audio(waveform, sample_rate, target_sample_rate)
        sample_rate = int(target_sample_rate)
    return waveform.contiguous(), sample_rate


def normalize_audio(waveform: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    peak = waveform.abs().amax(dim=-1, keepdim=True).clamp_min(eps)
    return waveform / peak


def prepare_audio_for_features(waveform: torch.Tensor, preserve_energy: bool = False) -> torch.Tensor:
    """Keep recorded levels for VAD-aware models; retain legacy peak normalization otherwise."""
    return waveform if preserve_energy else normalize_audio(waveform)


def extract_audio_clip(
    waveform: torch.Tensor,
    start_time: float,
    end_time: float,
    sample_rate: int,
    pad: bool = True,
    target_num_samples: int | None = None,
) -> torch.Tensor:
    if end_time <= start_time:
        raise ValueError("Audio clip end_time must be greater than start_time")

    start_sample = int(round(float(start_time) * sample_rate))
    end_sample = int(round(float(end_time) * sample_rate))
    target_len = max(1, end_sample - start_sample)
    if target_num_samples is not None:
        target_len = max(1, int(target_num_samples))

    clip_start = max(0, start_sample)
    clip_end = min(waveform.shape[-1], end_sample)
    clip = waveform[:, clip_start:clip_end]

    if pad and clip.shape[-1] < target_len:
        left_pad = max(0, -start_sample)
        right_pad = target_len - clip.shape[-1] - left_pad
        clip = F.pad(clip, (left_pad, max(0, right_pad)))
    if clip.shape[-1] > target_len:
        clip = clip[:, :target_len]

    return clip.contiguous()


def compute_multichannel_spectrogram(
    audio_clip: torch.Tensor,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
    eps: float = 1e-8,
) -> torch.Tensor:
    stft = _compute_stft(audio_clip, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    power = stft.abs().pow(2.0)
    return torch.log(power + eps)


def compute_channel_correlation(
    audio_clip: torch.Tensor,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
    pairs: list[tuple[int, int]] | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    stft = _compute_stft(audio_clip, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    return _channel_correlation_from_stft(stft, pairs=pairs, eps=eps)


def compute_multichannel_audio_features(
    audio_clip: torch.Tensor,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
    pairs: list[tuple[int, int]] | None = None,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    stft = _compute_stft(audio_clip, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    spectrogram = torch.log(stft.abs().pow(2.0) + eps)
    correlation = _channel_correlation_from_stft(stft, pairs=pairs, eps=eps)
    return spectrogram, correlation


def _compute_stft(
    audio_clip: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
) -> torch.Tensor:
    window = torch.hann_window(win_length, device=audio_clip.device, dtype=audio_clip.dtype)
    return torch.stft(
        audio_clip,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
    )


def _channel_correlation_from_stft(
    stft: torch.Tensor,
    pairs: list[tuple[int, int]] | None,
    eps: float,
) -> torch.Tensor:
    pairs = pairs or channel_pairs(stft.shape[0])
    channel_i, channel_j = zip(*pairs)
    left = stft[list(channel_i)]
    right = stft[list(channel_j)]
    cross_power = left * right.conj()
    denom = left.abs() * right.abs()
    return (cross_power.real / denom.clamp_min(eps)).float()
