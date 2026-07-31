from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class VideoEncoder(nn.Module):
    def __init__(self, out_channels: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 7, 7), padding=(1, 3, 3), stride=(1, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, padding=1, stride=(1, 2, 2)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, video_clip: torch.Tensor) -> torch.Tensor:
        return self.forward_temporal(video_clip).mean(dim=1)

    def forward_temporal(self, video_clip: torch.Tensor) -> torch.Tensor:
        if video_clip.ndim != 5:
            raise ValueError(f"Expected video shape [B,T,C,H,W] or [B,C,T,H,W], got {tuple(video_clip.shape)}")
        if video_clip.shape[2] == 3:
            video_clip = video_clip.permute(0, 2, 1, 3, 4)
        features = self.net(video_clip)
        return features.permute(0, 2, 1, 3, 4).contiguous()


class AudioEncoder(nn.Module):
    def __init__(self, num_audio_channels: int = 6, num_corr_channels: int = 15, out_dim: int = 128) -> None:
        super().__init__()
        in_channels = num_audio_channels + num_corr_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, spectrogram: torch.Tensor, channel_correlation: torch.Tensor) -> torch.Tensor:
        features = self.forward_map(spectrogram, channel_correlation)
        return self.pool(features).flatten(1)

    def forward_map(self, spectrogram: torch.Tensor, channel_correlation: torch.Tensor) -> torch.Tensor:
        if spectrogram.ndim != 4:
            raise ValueError(f"Expected spectrogram shape [B,C,F,T], got {tuple(spectrogram.shape)}")
        if channel_correlation.ndim != 4:
            raise ValueError(
                f"Expected channel_correlation shape [B,P,F,T], got {tuple(channel_correlation.shape)}"
            )
        audio = torch.cat([spectrogram, channel_correlation], dim=1)
        return self.conv(audio)

    def forward_temporal(
        self,
        spectrogram: torch.Tensor,
        channel_correlation: torch.Tensor,
        output_steps: int,
    ) -> torch.Tensor:
        """Return frame-aligned audio features as [B,T,C]."""
        features = self.forward_map(spectrogram, channel_correlation).mean(dim=2)
        features = F.interpolate(features, size=output_steps, mode="linear", align_corners=False)
        return features.transpose(1, 2).contiguous()


class AudioVisualFusion(nn.Module):
    def __init__(self, video_channels: int = 128, audio_dim: int = 128, fused_channels: int = 128) -> None:
        super().__init__()
        self.audio_projection = nn.Linear(audio_dim, audio_dim)
        self.fusion = nn.Sequential(
            nn.Conv2d(video_channels + audio_dim, fused_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(fused_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(fused_channels, fused_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(fused_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, video_features: torch.Tensor, audio_features: torch.Tensor) -> torch.Tensor:
        audio_features = self.audio_projection(audio_features)
        audio_map = audio_features[:, :, None, None].expand(
            -1,
            -1,
            video_features.shape[-2],
            video_features.shape[-1],
        )
        return self.fusion(torch.cat([video_features, audio_map], dim=1))

    def forward_temporal(self, video_features: torch.Tensor, audio_features: torch.Tensor) -> torch.Tensor:
        if video_features.ndim != 5:
            raise ValueError(f"Expected temporal video features [B,T,C,H,W], got {tuple(video_features.shape)}")
        batch, steps, channels, height, width = video_features.shape
        flattened_video = video_features.reshape(batch * steps, channels, height, width)
        if audio_features.ndim == 2:
            repeated_audio = audio_features[:, None, :].expand(-1, steps, -1)
        elif audio_features.ndim == 3 and audio_features.shape[:2] == (batch, steps):
            repeated_audio = audio_features
        else:
            raise ValueError(
                f"Expected audio features [B,C] or [B,T,C] aligned to {(batch, steps)}, "
                f"got {tuple(audio_features.shape)}"
            )
        repeated_audio = repeated_audio.reshape(batch * steps, -1)
        fused = self.forward(flattened_video, repeated_audio)
        return fused.reshape(batch, steps, fused.shape[1], fused.shape[2], fused.shape[3])


class HeadCropEncoder(nn.Module):
    def __init__(self, out_dim: int = 64) -> None:
        super().__init__()
        self.out_dim = int(out_dim)
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, candidate_head_crops: list[torch.Tensor]) -> list[torch.Tensor]:
        counts = [crops.shape[0] for crops in candidate_head_crops]
        nonempty = [crops for crops in candidate_head_crops if crops.numel() > 0]
        if not nonempty:
            return [crops.new_empty((0, crops.shape[1], self.out_dim)) for crops in candidate_head_crops]
        steps = nonempty[0].shape[1]
        for crops in nonempty:
            if crops.ndim != 5 or crops.shape[1] != steps or crops.shape[2] != 3:
                raise ValueError(f"Expected head crops [N,{steps},3,H,W], got {tuple(crops.shape)}")
        flattened = torch.cat([crops.reshape(-1, *crops.shape[2:]) for crops in nonempty], dim=0)
        encoded = self.net(flattened).flatten(1)
        encoded_by_sample = []
        offset = 0
        for crops, count in zip(candidate_head_crops, counts):
            if count == 0:
                encoded_by_sample.append(crops.new_empty((0, crops.shape[1], encoded.shape[1])))
                continue
            values = encoded[offset : offset + count * steps].reshape(count, steps, -1)
            encoded_by_sample.append(values)
            offset += count * steps
        return encoded_by_sample


class HeatmapHead(nn.Module):
    def __init__(self, in_channels: int = 128, heatmap_size: tuple[int, int] = (56, 56)) -> None:
        super().__init__()
        self.heatmap_size = heatmap_size
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        heatmap = self.head(features)
        if heatmap.shape[-2:] != self.heatmap_size:
            heatmap = F.interpolate(heatmap, size=self.heatmap_size, mode="bilinear", align_corners=False)
        return heatmap


class CandidateHead(nn.Module):
    def __init__(self, in_channels: int = 128, hidden_dim: int = 128) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_channels * 2 + 4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor, candidate_bboxes: list[torch.Tensor]) -> list[torch.Tensor]:
        if features.ndim != 4:
            raise ValueError(f"Expected feature shape [B,C,H,W], got {tuple(features.shape)}")
        if len(candidate_bboxes) != features.shape[0]:
            raise ValueError(
                f"Expected {features.shape[0]} candidate box tensors, got {len(candidate_bboxes)}"
            )

        logits = []
        for feature_map, boxes in zip(features, candidate_bboxes):
            if boxes.numel() == 0:
                logits.append(feature_map.new_empty((0,)))
                continue
            pooled = torch.stack([self._pool_box(feature_map, box) for box in boxes])
            logits.append(self.classifier(pooled).squeeze(-1))
        return logits

    @staticmethod
    def _pool_box(
        feature_map: torch.Tensor,
        bbox: torch.Tensor,
        include_geometry: bool = True,
    ) -> torch.Tensor:
        channels, height, width = feature_map.shape
        x1, y1, x2, y2 = bbox.unbind()
        x_start = int(torch.floor(torch.minimum(x1, x2)).clamp(0, width - 1).item())
        y_start = int(torch.floor(torch.minimum(y1, y2)).clamp(0, height - 1).item())
        x_end = int(torch.ceil(torch.maximum(x1, x2)).clamp(x_start + 1, width).item())
        y_end = int(torch.ceil(torch.maximum(y1, y2)).clamp(y_start + 1, height).item())

        region = feature_map[:, y_start:y_end, x_start:x_end].flatten(1)
        if region.numel() == 0:
            mean_features = feature_map.new_zeros((channels,))
            max_features = feature_map.new_zeros((channels,))
        else:
            mean_features = region.mean(dim=1)
            max_features = region.max(dim=1).values

        box_width = max(1.0, float(x_end - x_start))
        box_height = max(1.0, float(y_end - y_start))
        geometry = feature_map.new_tensor(
            [
                (x_start + x_end) / (2.0 * width),
                (y_start + y_end) / (2.0 * height),
                box_width / float(width),
                box_height / float(height),
            ]
        )
        if not include_geometry:
            geometry.zero_()
        return torch.cat([mean_features, max_features, geometry], dim=0)


class TemporalCandidateHead(nn.Module):
    def __init__(self, in_channels: int = 128, hidden_dim: int = 128, extra_dim: int = 0) -> None:
        super().__init__()
        self.base_pooled_dim = in_channels * 2 + 4
        self.extra_dim = int(extra_dim)
        pooled_dim = self.base_pooled_dim + self.extra_dim
        self.input_norm = nn.LayerNorm(pooled_dim)
        self.temporal = nn.GRU(
            input_size=pooled_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim + 4),
            nn.Linear(hidden_dim + 4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        temporal_features: torch.Tensor,
        candidate_bbox_sequences: list[torch.Tensor],
        include_geometry: bool = True,
        candidate_extra_sequences: list[torch.Tensor] | None = None,
    ) -> list[torch.Tensor]:
        if temporal_features.ndim != 5:
            raise ValueError(f"Expected temporal features [B,T,C,H,W], got {tuple(temporal_features.shape)}")
        if len(candidate_bbox_sequences) != temporal_features.shape[0]:
            raise ValueError(
                f"Expected {temporal_features.shape[0]} candidate sequence tensors, got {len(candidate_bbox_sequences)}"
            )

        logits = []
        steps = temporal_features.shape[1]
        if self.extra_dim > 0:
            if candidate_extra_sequences is None or len(candidate_extra_sequences) != temporal_features.shape[0]:
                raise ValueError("Candidate extra sequences are required for this temporal head.")
        elif candidate_extra_sequences is not None:
            raise ValueError("Candidate extra sequences were provided to a temporal head without extra inputs.")

        for sample_index, (feature_sequence, box_sequence) in enumerate(
            zip(temporal_features, candidate_bbox_sequences)
        ):
            if box_sequence.numel() == 0:
                logits.append(feature_sequence.new_empty((0,)))
                continue
            if box_sequence.ndim == 2:
                box_sequence = box_sequence[:, None, :].expand(-1, steps, -1)
            if box_sequence.ndim == 3 and box_sequence.shape[1] != steps:
                indices = torch.linspace(
                    0,
                    box_sequence.shape[1] - 1,
                    steps,
                    device=box_sequence.device,
                ).round().long()
                box_sequence = box_sequence.index_select(1, indices)
            if box_sequence.ndim != 3 or box_sequence.shape[1] != steps:
                raise ValueError(
                    f"Expected candidate boxes [N,{steps},4], got {tuple(box_sequence.shape)}"
                )

            candidate_sequences = []
            for candidate_boxes in box_sequence:
                pooled_steps = torch.stack(
                    [
                        CandidateHead._pool_box(
                            feature_sequence[step_index],
                            candidate_boxes[step_index],
                            include_geometry=include_geometry,
                        )
                        for step_index in range(steps)
                    ]
                )
                candidate_sequences.append(pooled_steps)
            sequence_tensor = torch.stack(candidate_sequences)
            if self.extra_dim > 0:
                extra_sequence = candidate_extra_sequences[sample_index]
                if extra_sequence.shape != (sequence_tensor.shape[0], steps, self.extra_dim):
                    raise ValueError(
                        f"Expected candidate extras {tuple(sequence_tensor.shape[:2]) + (self.extra_dim,)}, "
                        f"got {tuple(extra_sequence.shape)}"
                    )
                sequence_tensor = torch.cat([sequence_tensor, extra_sequence], dim=-1)
            sequence_tensor = self.input_norm(sequence_tensor)
            _, hidden = self.temporal(sequence_tensor)
            embeddings = hidden[-1]
            center_geometry = box_sequence[:, steps // 2, :]
            if not include_geometry:
                center_geometry = torch.zeros_like(center_geometry)
            logits.append(self.classifier(torch.cat([embeddings, center_geometry], dim=1)).squeeze(-1))
        return logits


class IndependentCandidateHead(nn.Module):
    """Classify speaking and addressing independently after per-candidate temporal encoding."""

    def __init__(self, in_channels: int = 128, hidden_dim: int = 128) -> None:
        super().__init__()
        pooled_dim = in_channels * 2 + 4
        self.input_norm = nn.LayerNorm(pooled_dim)
        self.temporal = nn.GRU(pooled_dim, hidden_dim, batch_first=True)
        self.token_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim + 4),
            nn.Linear(hidden_dim + 4, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.speaking_classifier = nn.Linear(hidden_dim, 1)
        self.addressing_classifier = nn.Linear(hidden_dim, 1)

    def initialize_temporal_encoder(self, source: TemporalCandidateHead) -> None:
        with torch.no_grad():
            self.input_norm.load_state_dict(source.input_norm.state_dict())
            self.temporal.load_state_dict(source.temporal.state_dict())

    def forward(
        self,
        temporal_features: torch.Tensor,
        candidate_bbox_sequences: list[torch.Tensor],
        include_geometry: bool = True,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if temporal_features.ndim != 5:
            raise ValueError(f"Expected temporal features [B,T,C,H,W], got {tuple(temporal_features.shape)}")
        if len(candidate_bbox_sequences) != temporal_features.shape[0]:
            raise ValueError(
                f"Expected {temporal_features.shape[0]} candidate sequence tensors, "
                f"got {len(candidate_bbox_sequences)}"
            )

        speaking_logits: list[torch.Tensor] = []
        addressing_logits: list[torch.Tensor] = []
        steps = temporal_features.shape[1]
        for feature_sequence, box_sequence in zip(temporal_features, candidate_bbox_sequences):
            if box_sequence.numel() == 0:
                speaking_logits.append(feature_sequence.new_empty((0,)))
                addressing_logits.append(feature_sequence.new_empty((0,)))
                continue
            box_sequence = CandidateRelationHead._align_boxes(box_sequence, steps)
            candidate_sequences = torch.stack(
                [
                    torch.stack(
                        [
                            CandidateHead._pool_box(
                                feature_sequence[step],
                                candidate_boxes[step],
                                include_geometry=include_geometry,
                            )
                            for step in range(steps)
                        ]
                    )
                    for candidate_boxes in box_sequence
                ]
            )
            _, hidden = self.temporal(self.input_norm(candidate_sequences))
            geometry = CandidateRelationHead._normalized_geometry(
                box_sequence[:, steps // 2],
                feature_sequence.shape[-2:],
            )
            if not include_geometry:
                geometry.zero_()
            candidate_tokens = self.output_norm(
                self.token_projection(torch.cat([hidden[-1], geometry], dim=1))
            )
            speaking_logits.append(self.speaking_classifier(candidate_tokens).squeeze(-1))
            addressing_logits.append(self.addressing_classifier(candidate_tokens).squeeze(-1))
        return speaking_logits, addressing_logits


class CandidateRelationHead(nn.Module):
    """Jointly classify speaking and robot-addressing after temporal candidate encoding."""

    def __init__(
        self,
        in_channels: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        pooled_dim = in_channels * 2 + 4
        self.input_norm = nn.LayerNorm(pooled_dim)
        self.temporal = nn.GRU(pooled_dim, hidden_dim, batch_first=True)
        self.token_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim + 4),
            nn.Linear(hidden_dim + 4, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.scene_projection = nn.Linear(in_channels, hidden_dim)
        self.camera_token = nn.Parameter(torch.zeros(1, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.relation = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.speaking_classifier = nn.Linear(hidden_dim, 1)
        self.speaker_residual_adapter = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.zeros_(self.speaker_residual_adapter[-1].weight)
        nn.init.zeros_(self.speaker_residual_adapter[-1].bias)
        self.addressing_classifier = nn.Linear(hidden_dim, 1)
        self.address_scene_norm = nn.LayerNorm(hidden_dim)
        self.causal_memory = nn.GRUCell(hidden_dim, hidden_dim)
        self.candidate_memory = nn.GRUCell(hidden_dim, hidden_dim)
        self.candidate_memory_norm = nn.LayerNorm(hidden_dim)
        self.turn_graph_projection = nn.Sequential(
            nn.LayerNorm(5),
            nn.Linear(5, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.turn_graph_other_residual = nn.Sequential(
            nn.LayerNorm(3),
            nn.Linear(3, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
        )
        nn.init.zeros_(self.turn_graph_other_residual[-1].weight)
        nn.init.zeros_(self.turn_graph_other_residual[-1].bias)
        self.turn_graph_norm = nn.LayerNorm(hidden_dim)
        self.self_speaking_classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, 1),
        )
        self.self_target_classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, 1),
        )
        self.global_address_classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, 1),
        )
        self.scene_response_classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim // 2, 3),
        )
        self.global_address_adapter = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )
        nn.init.zeros_(self.global_address_adapter[-1].weight)
        nn.init.zeros_(self.global_address_adapter[-1].bias)
        self.vad_classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, 1),
        )

    def initialize_temporal_encoder(self, source: TemporalCandidateHead) -> None:
        with torch.no_grad():
            self.input_norm.load_state_dict(source.input_norm.state_dict())
            self.temporal.load_state_dict(source.temporal.state_dict())

    def forward(
        self,
        temporal_features: torch.Tensor,
        candidate_bbox_sequences: list[torch.Tensor],
        include_geometry: bool = True,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        speaking, addressing, _, _ = self._forward_all(
            temporal_features,
            candidate_bbox_sequences,
            include_geometry=include_geometry,
            previous_memory=None,
        )
        return speaking, addressing

    def forward_with_global_gate(
        self,
        temporal_features: torch.Tensor,
        candidate_bbox_sequences: list[torch.Tensor],
        previous_memory: torch.Tensor | None = None,
        include_geometry: bool = True,
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        speaking, _, global_addressing, updated_memory = self._forward_all(
            temporal_features,
            candidate_bbox_sequences,
            include_geometry=include_geometry,
            previous_memory=previous_memory,
        )
        return speaking, global_addressing, updated_memory

    def forward_with_candidate_memory(
        self,
        temporal_features: torch.Tensor,
        candidate_bbox_sequences: list[torch.Tensor],
        previous_memory: torch.Tensor | None = None,
        include_geometry: bool = True,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
        speaking, addressing, memory_bias, updated_memory = self._forward_all(
            temporal_features,
            candidate_bbox_sequences,
            include_geometry=include_geometry,
            previous_memory=previous_memory,
        )
        memory_conditioned = [
            candidate_logits + memory_bias[index]
            for index, candidate_logits in enumerate(addressing)
        ]
        return speaking, memory_conditioned, updated_memory

    def forward_with_candidate_identity_memory(
        self,
        temporal_features: torch.Tensor,
        candidate_bbox_sequences: list[torch.Tensor],
        candidate_ids: list[tuple[object, ...]],
        previous_states: list[dict[object, torch.Tensor]] | None = None,
        include_geometry: bool = True,
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        torch.Tensor,
        list[dict[object, torch.Tensor]],
    ]:
        (
            speaking_logits,
            addressing_logits,
            _,
            vad_logits,
            updated_states,
            _,
            _,
        ) = self._forward_with_candidate_identity_memory(
            temporal_features,
            candidate_bbox_sequences,
            candidate_ids,
            previous_states=previous_states,
            include_geometry=include_geometry,
        )
        return speaking_logits, addressing_logits, vad_logits, updated_states

    def forward_with_candidate_identity_global_gate(
        self,
        temporal_features: torch.Tensor,
        candidate_bbox_sequences: list[torch.Tensor],
        candidate_ids: list[tuple[object, ...]],
        previous_states: list[dict[object, torch.Tensor]] | None = None,
        include_geometry: bool = True,
    ) -> tuple[
        list[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        list[dict[object, torch.Tensor]],
    ]:
        (
            speaking_logits,
            _,
            global_addressing_logits,
            vad_logits,
            updated_states,
            _,
            _,
        ) = self._forward_with_candidate_identity_memory(
            temporal_features,
            candidate_bbox_sequences,
            candidate_ids,
            previous_states=previous_states,
            include_geometry=include_geometry,
        )
        return speaking_logits, global_addressing_logits, vad_logits, updated_states

    def forward_with_candidate_identity_scene_response(
        self,
        temporal_features: torch.Tensor,
        candidate_bbox_sequences: list[torch.Tensor],
        candidate_ids: list[tuple[object, ...]],
        previous_states: list[dict[object, torch.Tensor]] | None = None,
        include_geometry: bool = True,
    ):
        values = self._forward_with_candidate_identity_memory(
            temporal_features, candidate_bbox_sequences, candidate_ids,
            previous_states=previous_states, include_geometry=include_geometry,
        )
        speaking, addressing, _, vad, states, _, scene = values
        scene = scene + self.global_address_adapter(scene)
        return speaking, addressing, self.scene_response_classifier(scene), vad, states

    def forward_with_candidate_identity_turn_graph(
        self,
        temporal_features: torch.Tensor,
        candidate_bbox_sequences: list[torch.Tensor],
        candidate_ids: list[tuple[object, ...]],
        turn_graph_features: list[torch.Tensor],
        previous_states: list[dict[object, torch.Tensor]] | None = None,
        include_geometry: bool = True,
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        list[torch.Tensor],
        list[dict[object, torch.Tensor]],
    ]:
        (
            speaking_logits,
            _,
            _,
            vad_logits,
            updated_states,
            remembered_batches,
            identity_scene_batch,
        ) = self._forward_with_candidate_identity_memory(
            temporal_features,
            candidate_bbox_sequences,
            candidate_ids,
            previous_states=previous_states,
            include_geometry=include_geometry,
        )
        if len(turn_graph_features) != len(remembered_batches):
            raise ValueError("Turn-graph features must match the identity-memory batch size.")

        addressing_logits: list[torch.Tensor] = []
        self_target_logits: list[torch.Tensor] = []
        for remembered, graph_features, scene in zip(
            remembered_batches,
            turn_graph_features,
            identity_scene_batch,
        ):
            expected_shape = (remembered.shape[0], 8)
            if graph_features.shape != expected_shape:
                raise ValueError(
                    f"Expected turn-graph features {expected_shape}, "
                    f"got {tuple(graph_features.shape)}"
                )
            if remembered.numel() == 0:
                addressing_logits.append(remembered.new_empty((0,)))
                self_target_logits.append(remembered.new_empty((0,)))
                continue
            base_graph_features = graph_features[:, :5]
            other_graph_features = graph_features[:, 5:]
            graph_embedding = self.turn_graph_projection(base_graph_features)
            conditioned = self.turn_graph_norm(remembered + graph_embedding)
            addressing_logits.append(
                self.addressing_classifier(conditioned).squeeze(-1)
                + self.turn_graph_other_residual(other_graph_features).squeeze(-1)
            )
            scene_expanded = scene.unsqueeze(0).expand(conditioned.shape[0], -1)
            self_target_logits.append(
                self.self_target_classifier(torch.cat([conditioned, scene_expanded], dim=1)).squeeze(-1)
            )

        self_speaking_logits = self.self_speaking_classifier(identity_scene_batch).squeeze(-1)
        return (
            speaking_logits,
            addressing_logits,
            vad_logits,
            self_speaking_logits,
            self_target_logits,
            updated_states,
        )

    def _forward_with_candidate_identity_memory(
        self,
        temporal_features: torch.Tensor,
        candidate_bbox_sequences: list[torch.Tensor],
        candidate_ids: list[tuple[object, ...]],
        previous_states: list[dict[object, torch.Tensor]] | None,
        include_geometry: bool,
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        list[dict[object, torch.Tensor]],
        list[torch.Tensor],
        torch.Tensor,
    ]:
        related_tokens, scene_batch = self._encode_relation_tokens(
            temporal_features,
            candidate_bbox_sequences,
            include_geometry=include_geometry,
        )
        if len(candidate_ids) != len(related_tokens):
            raise ValueError("Candidate IDs and relation-token batches must have the same length.")
        if previous_states is None:
            previous_states = [{} for _ in related_tokens]
        if len(previous_states) != len(related_tokens):
            raise ValueError("Previous candidate-state batches must match the input batch size.")

        speaking_logits: list[torch.Tensor] = []
        addressing_logits: list[torch.Tensor] = []
        identity_scene_vectors: list[torch.Tensor] = []
        remembered_batches: list[torch.Tensor] = []
        updated_states: list[dict[object, torch.Tensor]] = []
        for tokens, scene_vector, ids, previous in zip(
            related_tokens,
            scene_batch,
            candidate_ids,
            previous_states,
        ):
            ids = tuple(ids)
            if len(ids) != tokens.shape[0]:
                raise ValueError(f"Expected {tokens.shape[0]} candidate IDs, got {len(ids)}.")
            if tokens.numel() == 0:
                speaking_logits.append(tokens.new_empty((0,)))
                addressing_logits.append(tokens.new_empty((0,)))
                identity_scene_vectors.append(scene_vector)
                remembered_batches.append(tokens)
                updated_states.append({})
                continue
            prior = torch.stack(
                [previous.get(candidate_id, tokens.new_zeros(tokens.shape[1])) for candidate_id in ids]
            )
            remembered = self.candidate_memory(tokens, prior)
            remembered = self.candidate_memory_norm(remembered)
            remembered_batches.append(remembered)
            base_speaking_logits = self.speaking_classifier(remembered).squeeze(-1)
            sample_speaking_logits = (
                base_speaking_logits
                + self.speaker_residual_adapter(remembered).squeeze(-1)
            )
            speaking_logits.append(sample_speaking_logits)
            addressing_logits.append(self.addressing_classifier(remembered).squeeze(-1))
            # Preserve the frozen checkpoint's scene/addressing path; the adapter
            # only corrects which participant is selected as the speaker.
            speaker_weights = torch.sigmoid(base_speaking_logits)
            speaker_context = (
                (speaker_weights[:, None] * remembered).sum(dim=0)
                / speaker_weights.sum().clamp_min(1e-6)
            )
            identity_scene_vectors.append(self.address_scene_norm(scene_vector + speaker_context))
            updated_states.append(
                {candidate_id: remembered[index] for index, candidate_id in enumerate(ids)}
            )

        global_input = torch.stack(identity_scene_vectors)
        global_input = global_input + self.global_address_adapter(global_input)
        global_addressing_logits = self.global_address_classifier(global_input).squeeze(-1)
        vad_logits = self.vad_classifier(scene_batch).squeeze(-1)
        return (
            speaking_logits,
            addressing_logits,
            global_addressing_logits,
            vad_logits,
            updated_states,
            remembered_batches,
            torch.stack(identity_scene_vectors),
        )

    def _encode_relation_tokens(
        self,
        temporal_features: torch.Tensor,
        candidate_bbox_sequences: list[torch.Tensor],
        include_geometry: bool,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        if temporal_features.ndim != 5:
            raise ValueError(f"Expected temporal features [B,T,C,H,W], got {tuple(temporal_features.shape)}")
        if len(candidate_bbox_sequences) != temporal_features.shape[0]:
            raise ValueError(
                f"Expected {temporal_features.shape[0]} candidate sequence tensors, "
                f"got {len(candidate_bbox_sequences)}"
            )

        related_tokens: list[torch.Tensor] = []
        scene_vectors: list[torch.Tensor] = []
        steps = temporal_features.shape[1]
        hidden_dim = int(self.output_norm.normalized_shape[0])
        for feature_sequence, box_sequence in zip(temporal_features, candidate_bbox_sequences):
            if box_sequence.numel() == 0:
                related_tokens.append(feature_sequence.new_empty((0, hidden_dim)))
                scene_vectors.append(self.scene_projection(feature_sequence.mean(dim=(0, 2, 3))))
                continue
            box_sequence = self._align_boxes(box_sequence, steps)
            candidate_sequences = torch.stack(
                [
                    torch.stack(
                        [
                            CandidateHead._pool_box(
                                feature_sequence[step],
                                candidate_boxes[step],
                                include_geometry=include_geometry,
                            )
                            for step in range(steps)
                        ]
                    )
                    for candidate_boxes in box_sequence
                ]
            )
            _, hidden = self.temporal(self.input_norm(candidate_sequences))
            geometry = self._normalized_geometry(box_sequence[:, steps // 2], feature_sequence.shape[-2:])
            if not include_geometry:
                geometry.zero_()
            candidate_tokens = self.token_projection(torch.cat([hidden[-1], geometry], dim=1))
            scene_token = self.camera_token + self.scene_projection(feature_sequence.mean(dim=(0, 2, 3)))[None]
            relation_output = self.relation(torch.cat([scene_token, candidate_tokens], dim=0).unsqueeze(0))[0]
            camera_output = self.output_norm(relation_output[0])
            related = self.output_norm(relation_output[1:])
            preliminary_speaking = self.speaking_classifier(related).squeeze(-1)
            speaker_weights = torch.sigmoid(preliminary_speaking)
            speaker_context = (speaker_weights[:, None] * related).sum(dim=0) / speaker_weights.sum().clamp_min(1e-6)
            related_tokens.append(related)
            scene_vectors.append(self.address_scene_norm(camera_output + speaker_context))
        return related_tokens, torch.stack(scene_vectors)

    def _forward_all(
        self,
        temporal_features: torch.Tensor,
        candidate_bbox_sequences: list[torch.Tensor],
        include_geometry: bool,
        previous_memory: torch.Tensor | None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor]:
        related_tokens, scene_batch = self._encode_relation_tokens(
            temporal_features,
            candidate_bbox_sequences,
            include_geometry=include_geometry,
        )
        speaking_logits = [
            self.speaking_classifier(tokens).squeeze(-1)
            + self.speaker_residual_adapter(tokens).squeeze(-1)
            for tokens in related_tokens
        ]
        addressing_logits = [self.addressing_classifier(tokens).squeeze(-1) for tokens in related_tokens]
        if previous_memory is None:
            previous_memory = scene_batch.new_zeros(scene_batch.shape)
        if previous_memory.shape != scene_batch.shape:
            raise ValueError(
                f"Expected causal memory {tuple(scene_batch.shape)}, got {tuple(previous_memory.shape)}"
            )
        updated_memory = self.causal_memory(scene_batch, previous_memory)
        global_input = updated_memory + self.global_address_adapter(updated_memory)
        global_addressing = self.global_address_classifier(global_input).squeeze(-1)
        return speaking_logits, addressing_logits, global_addressing, updated_memory

    @staticmethod
    def _align_boxes(box_sequence: torch.Tensor, steps: int) -> torch.Tensor:
        if box_sequence.ndim == 2:
            box_sequence = box_sequence[:, None, :].expand(-1, steps, -1)
        if box_sequence.ndim == 3 and box_sequence.shape[1] != steps:
            indices = torch.linspace(0, box_sequence.shape[1] - 1, steps, device=box_sequence.device).round().long()
            box_sequence = box_sequence.index_select(1, indices)
        if box_sequence.ndim != 3 or box_sequence.shape[1] != steps:
            raise ValueError(f"Expected candidate boxes [N,{steps},4], got {tuple(box_sequence.shape)}")
        return box_sequence

    @staticmethod
    def _normalized_geometry(boxes: torch.Tensor, spatial_size: tuple[int, int]) -> torch.Tensor:
        height, width = spatial_size
        x1, y1, x2, y2 = boxes.unbind(dim=1)
        return torch.stack(
            [
                (x1 + x2) / (2.0 * width),
                (y1 + y2) / (2.0 * height),
                (x2 - x1).abs() / float(width),
                (y2 - y1).abs() / float(height),
            ],
            dim=1,
        )


class NoResponseHead(nn.Module):
    def __init__(self, in_channels: int = 128, hidden_dim: int = 128) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(in_channels)
        self.temporal = nn.GRU(
            input_size=in_channels,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, temporal_features: torch.Tensor) -> torch.Tensor:
        if temporal_features.ndim != 5:
            raise ValueError(f"Expected temporal features [B,T,C,H,W], got {tuple(temporal_features.shape)}")
        sequence = temporal_features.mean(dim=(-2, -1))
        _, hidden = self.temporal(self.input_norm(sequence))
        return self.classifier(hidden[-1]).squeeze(-1)


class SharedResponseGate(nn.Module):
    def __init__(self, hidden_dim: int = 16) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.LayerNorm(4), nn.Linear(4, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def forward(self, vad_logits: torch.Tensor, address_logits: torch.Tensor, candidate_counts: torch.Tensor, audio_log_rms_db: torch.Tensor) -> torch.Tensor:
        shapes = {vad_logits.shape, address_logits.shape, candidate_counts.shape, audio_log_rms_db.shape}
        if len(shapes) != 1:
            raise ValueError('Shared response-gate inputs must have matching batch shapes.')
        features = torch.stack([
            vad_logits,
            address_logits,
            torch.log1p(candidate_counts.float()),
            (audio_log_rms_db.float() + 44.0) / 10.0,
        ], dim=1)
        return self.network(features).squeeze(-1)


class SAALModel(nn.Module):
    def __init__(
        self,
        num_audio_channels: int = 6,
        num_corr_channels: int = 15,
        heatmap_size: tuple[int, int] = (56, 56),
        head_crop_feature_dim: int = 64,
        frame_aligned_audio: bool = False,
    ) -> None:
        super().__init__()
        self.video_encoder = VideoEncoder(out_channels=128)
        self.frame_aligned_audio = bool(frame_aligned_audio)
        self.audio_encoder = AudioEncoder(
            num_audio_channels=num_audio_channels,
            num_corr_channels=num_corr_channels,
            out_dim=128,
        )
        self.fusion = AudioVisualFusion(video_channels=128, audio_dim=128, fused_channels=128)
        self.heatmap_head = HeatmapHead(in_channels=128, heatmap_size=heatmap_size)
        self.candidate_head = CandidateHead(in_channels=128, hidden_dim=128)
        self.temporal_candidate_head = TemporalCandidateHead(in_channels=128, hidden_dim=128)
        self.independent_candidate_head = IndependentCandidateHead(in_channels=128, hidden_dim=128)
        self.candidate_relation_head = CandidateRelationHead(in_channels=128, hidden_dim=128)
        self.head_crop_encoder = HeadCropEncoder(out_dim=head_crop_feature_dim)
        self.head_crop_temporal_candidate_head = TemporalCandidateHead(
            in_channels=128,
            hidden_dim=128,
            extra_dim=head_crop_feature_dim,
        )
        self.no_response_head = NoResponseHead(in_channels=128, hidden_dim=128)
        self.shared_response_gate = SharedResponseGate()

    def shared_response_logits(self, vad_logits: torch.Tensor, address_logits: torch.Tensor, candidate_counts: torch.Tensor, audio_log_rms_db: torch.Tensor) -> torch.Tensor:
        return self.shared_response_gate(vad_logits, address_logits, candidate_counts, audio_log_rms_db)

    def initialize_head_crop_branch_from_temporal(self) -> None:
        source = self.temporal_candidate_head
        target = self.head_crop_temporal_candidate_head
        with torch.no_grad():
            target.input_norm.weight[: source.input_norm.weight.numel()].copy_(source.input_norm.weight)
            target.input_norm.bias[: source.input_norm.bias.numel()].copy_(source.input_norm.bias)
            target.temporal.weight_ih_l0[:, : source.temporal.input_size].copy_(source.temporal.weight_ih_l0)
            target.temporal.weight_hh_l0.copy_(source.temporal.weight_hh_l0)
            target.temporal.bias_ih_l0.copy_(source.temporal.bias_ih_l0)
            target.temporal.bias_hh_l0.copy_(source.temporal.bias_hh_l0)
            target.classifier.load_state_dict(source.classifier.state_dict())

    def initialize_relation_branch_from_temporal(self) -> None:
        self.candidate_relation_head.initialize_temporal_encoder(self.temporal_candidate_head)

    def initialize_independent_branch_from_temporal(self) -> None:
        self.independent_candidate_head.initialize_temporal_encoder(self.temporal_candidate_head)

    def forward_with_features(
        self,
        video_clip: torch.Tensor,
        audio_spectrogram: torch.Tensor,
        audio_channel_correlation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_features = self.video_encoder(video_clip)
        audio_features = self.audio_encoder(audio_spectrogram, audio_channel_correlation)
        fused_features = self.fusion(video_features, audio_features)
        return self.heatmap_head(fused_features), fused_features

    def forward_with_temporal_features(
        self,
        video_clip: torch.Tensor,
        audio_spectrogram: torch.Tensor,
        audio_channel_correlation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fused_temporal_features = self.encode_temporal_features(
            video_clip,
            audio_spectrogram,
            audio_channel_correlation,
        )
        fused_features = fused_temporal_features.mean(dim=1)
        return self.heatmap_head(fused_features), fused_features, fused_temporal_features

    def encode_temporal_features(
        self,
        video_clip: torch.Tensor,
        audio_spectrogram: torch.Tensor,
        audio_channel_correlation: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a temporal window without executing the optional heatmap head."""
        video_features = self.video_encoder.forward_temporal(video_clip)
        audio_features = (
            self.audio_encoder.forward_temporal(
                audio_spectrogram,
                audio_channel_correlation,
                output_steps=video_features.shape[1],
            )
            if self.frame_aligned_audio
            else self.audio_encoder(audio_spectrogram, audio_channel_correlation)
        )
        return self.fusion.forward_temporal(video_features, audio_features)

    def candidate_logits(self, fused_features: torch.Tensor, candidate_bboxes: list[torch.Tensor]) -> list[torch.Tensor]:
        return self.candidate_head(fused_features, candidate_bboxes)

    def temporal_candidate_logits(
        self,
        fused_temporal_features: torch.Tensor,
        candidate_bbox_sequences: list[torch.Tensor],
        include_geometry: bool = True,
    ) -> list[torch.Tensor]:
        return self.temporal_candidate_head(
            fused_temporal_features,
            candidate_bbox_sequences,
            include_geometry=include_geometry,
        )

    def temporal_response_logits(
        self,
        fused_temporal_features: torch.Tensor,
        candidate_bbox_sequences: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        candidate_logits = self.temporal_candidate_logits(fused_temporal_features, candidate_bbox_sequences)
        no_response_logits = self.no_response_head(fused_temporal_features)
        return [
            torch.cat([logits, no_response_logits[index : index + 1]], dim=0)
            for index, logits in enumerate(candidate_logits)
        ]

    def temporal_relation_logits(
        self,
        fused_temporal_features: torch.Tensor,
        candidate_bbox_sequences: list[torch.Tensor],
        include_geometry: bool = True,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        return self.candidate_relation_head(
            fused_temporal_features,
            candidate_bbox_sequences,
            include_geometry=include_geometry,
        )

    def temporal_independent_candidate_logits(
        self,
        fused_temporal_features: torch.Tensor,
        candidate_bbox_sequences: list[torch.Tensor],
        include_geometry: bool = True,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        return self.independent_candidate_head(
            fused_temporal_features,
            candidate_bbox_sequences,
            include_geometry=include_geometry,
        )

    def temporal_global_relation_logits(
        self,
        fused_temporal_features: torch.Tensor,
        candidate_bbox_sequences: list[torch.Tensor],
        previous_memory: torch.Tensor | None = None,
        include_geometry: bool = True,
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        return self.candidate_relation_head.forward_with_global_gate(
            fused_temporal_features,
            candidate_bbox_sequences,
            previous_memory=previous_memory,
            include_geometry=include_geometry,
        )

    def temporal_memory_relation_logits(
        self,
        fused_temporal_features: torch.Tensor,
        candidate_bbox_sequences: list[torch.Tensor],
        previous_memory: torch.Tensor | None = None,
        include_geometry: bool = True,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
        return self.candidate_relation_head.forward_with_candidate_memory(
            fused_temporal_features,
            candidate_bbox_sequences,
            previous_memory=previous_memory,
            include_geometry=include_geometry,
        )

    def temporal_identity_memory_relation_logits(
        self,
        fused_temporal_features: torch.Tensor,
        candidate_bbox_sequences: list[torch.Tensor],
        candidate_ids: list[tuple[object, ...]],
        previous_states: list[dict[object, torch.Tensor]] | None = None,
        include_geometry: bool = True,
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        torch.Tensor,
        list[dict[object, torch.Tensor]],
    ]:
        return self.candidate_relation_head.forward_with_candidate_identity_memory(
            fused_temporal_features,
            candidate_bbox_sequences,
            candidate_ids,
            previous_states=previous_states,
            include_geometry=include_geometry,
        )

    def temporal_identity_global_relation_logits(
        self,
        fused_temporal_features: torch.Tensor,
        candidate_bbox_sequences: list[torch.Tensor],
        candidate_ids: list[tuple[object, ...]],
        previous_states: list[dict[object, torch.Tensor]] | None = None,
        include_geometry: bool = True,
    ) -> tuple[
        list[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        list[dict[object, torch.Tensor]],
    ]:
        return self.candidate_relation_head.forward_with_candidate_identity_global_gate(
            fused_temporal_features,
            candidate_bbox_sequences,
            candidate_ids,
            previous_states=previous_states,
            include_geometry=include_geometry,
        )

    def temporal_identity_scene_response_logits(
        self,
        fused_temporal_features: torch.Tensor,
        candidate_bbox_sequences: list[torch.Tensor],
        candidate_ids: list[tuple[object, ...]],
        previous_states: list[dict[object, torch.Tensor]] | None = None,
        include_geometry: bool = True,
    ):
        return self.candidate_relation_head.forward_with_candidate_identity_scene_response(
            fused_temporal_features, candidate_bbox_sequences, candidate_ids,
            previous_states=previous_states, include_geometry=include_geometry,
        )

    def temporal_identity_turn_graph_relation_logits(
        self,
        fused_temporal_features: torch.Tensor,
        candidate_bbox_sequences: list[torch.Tensor],
        candidate_ids: list[tuple[object, ...]],
        turn_graph_features: list[torch.Tensor],
        previous_states: list[dict[object, torch.Tensor]] | None = None,
        include_geometry: bool = True,
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        list[torch.Tensor],
        list[dict[object, torch.Tensor]],
    ]:
        return self.candidate_relation_head.forward_with_candidate_identity_turn_graph(
            fused_temporal_features,
            candidate_bbox_sequences,
            candidate_ids,
            turn_graph_features,
            previous_states=previous_states,
            include_geometry=include_geometry,
        )

    @staticmethod
    def combine_vad_logits(
        candidate_logits: list[torch.Tensor],
        vad_logits: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Return logits for P(candidate target) * P(speech present)."""
        if len(candidate_logits) != vad_logits.shape[0]:
            raise ValueError("Candidate-logit and VAD batches must have the same length.")
        combined = []
        for index, logits in enumerate(candidate_logits):
            probability = torch.sigmoid(logits) * torch.sigmoid(vad_logits[index])
            combined.append(torch.logit(probability.clamp(1e-6, 1.0 - 1e-6)))
        return combined

    @staticmethod
    def combine_relation_logits(
        speaking_logits: list[torch.Tensor],
        conditional_addressing_logits: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Return logits for P(speaking) * P(addressing robot | speaking)."""
        if len(speaking_logits) != len(conditional_addressing_logits):
            raise ValueError("Speaking and addressing batches must have the same length.")
        joint_logits = []
        for speaking, addressing in zip(speaking_logits, conditional_addressing_logits):
            if speaking.shape != addressing.shape:
                raise ValueError(
                    f"Speaking logits {tuple(speaking.shape)} do not match addressing logits "
                    f"{tuple(addressing.shape)}"
                )
            probability = torch.sigmoid(speaking) * torch.sigmoid(addressing)
            joint_logits.append(torch.logit(probability.clamp(1e-6, 1.0 - 1e-6)))
        return joint_logits

    @staticmethod
    def combine_global_address_logits(
        speaking_logits: list[torch.Tensor],
        global_addressing_logits: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Return candidate logits for P(speaking_i) * P(scene addresses robot)."""
        if len(speaking_logits) != global_addressing_logits.shape[0]:
            raise ValueError("Speaking and global-address batches must have the same length.")
        joint_logits = []
        for index, speaking in enumerate(speaking_logits):
            probability = torch.sigmoid(speaking) * torch.sigmoid(global_addressing_logits[index])
            joint_logits.append(torch.logit(probability.clamp(1e-6, 1.0 - 1e-6)))
        return joint_logits

    def temporal_head_crop_candidate_logits(
        self,
        fused_temporal_features: torch.Tensor,
        candidate_bbox_sequences: list[torch.Tensor],
        candidate_head_crops: list[torch.Tensor],
        include_geometry: bool = True,
    ) -> list[torch.Tensor]:
        crop_features = self.head_crop_encoder(candidate_head_crops)
        return self.head_crop_temporal_candidate_head(
            fused_temporal_features,
            candidate_bbox_sequences,
            include_geometry=include_geometry,
            candidate_extra_sequences=crop_features,
        )

    def forward(
        self,
        video_clip: torch.Tensor,
        audio_spectrogram: torch.Tensor,
        audio_channel_correlation: torch.Tensor,
    ) -> torch.Tensor:
        heatmap, _ = self.forward_with_features(video_clip, audio_spectrogram, audio_channel_correlation)
        return heatmap
