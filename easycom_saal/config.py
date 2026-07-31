from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class EasyComSAALConfig:
    easycom_root: str | None = None
    annotation_manifest: str | None = None

    sample_rate: int = 16000
    num_audio_channels: int = 6

    video_fps: float = 20.0
    clip_duration_sec: float = 1.0
    clip_stride_sec: float = 0.5

    # Tuples are in (height, width) order.
    video_size: tuple[int, int] = (224, 224)
    heatmap_size: tuple[int, int] = (56, 56)
    use_head_crop_branch: bool = False
    use_masked_head_video: bool = False
    head_crop_size: tuple[int, int] = (112, 112)
    head_crop_feature_dim: int = 64
    head_crop_padding: float = 0.18
    # EasyCom's compressed videos prepend a participant-thumbnail sidebar to
    # the 1920x1080 scene described by the manifest and bbox annotations.
    crop_composite_video_to_manifest: bool = True
    # Never keep a participant alive by borrowing an arbitrarily old box.
    head_bbox_max_time_delta_sec: float = 0.08

    use_transcription: bool = False
    use_beamforming: bool = False
    use_denoising: bool = False
    use_source_separation: bool = False

    label_source: str = "easycom_tos"
    # Set false to retain non-target speech for respond/no-response modeling.
    keep_only_tos_to_camera_wearer: bool = True
    # Build one sample per fixed streaming window and derive multi-label targets
    # from every speech annotation that intersects that window.
    use_timeline_multilabels: bool = False
    label_min_overlap_sec: float = 0.1
    include_silence_windows: bool = True
    require_visible_source_head_bbox: bool = True
    require_media_exists: bool = True

    heatmap_sigma_mode: str = "bbox_relative"
    heatmap_sigma_ratio: float = 0.25
    heatmap_fixed_sigma: float = 5.0

    use_temporal_loss: bool = True
    temporal_loss: str = "l1"
    lambda_heatmap: float = 1.0
    lambda_temporal: float = 0.05
    temporal_max_gap_sec: float = 1.0

    lambda_candidate_ranking: float = 0.0
    lambda_candidate_ce: float = 0.0
    lambda_candidate_classifier_ce: float = 0.0
    lambda_candidate_target_bce: float = 0.0
    lambda_candidate_speaker_bce: float = 0.0
    lambda_candidate_speaker_ce: float = 0.0
    weight_speaker_ce_by_candidate_count: bool = False
    candidate_target_threshold: float = 0.5
    speaker_threshold: float = 0.5
    use_relation_candidate_head: bool = False
    use_independent_candidate_head: bool = False
    use_global_address_gate: bool = False
    use_candidate_address_memory: bool = False
    use_causal_memory: bool = False
    use_frame_aligned_audio: bool = False
    preserve_audio_energy: bool = False
    audio_fixed_gain_db: float = 0.0
    audio_global_gain_augmentation_db: float = 0.0
    audio_noise_snr_db: float | None = None
    audio_channel_gain_std_db: float = 0.0
    audio_channel_gain_per_recording: bool = False
    audio_perturbation_seed: int = 0
    use_causal_channel_rms_normalization: bool = False
    audio_normalization_tau_sec: float = 30.0
    audio_normalization_target_rms_db: float = -44.0
    audio_normalization_max_gain_db: float = 6.0
    audio_robustness_heads_only: bool = False
    speaker_representation_finetune: bool = False
    speaker_residual_finetune: bool = False
    global_address_finetune: bool = False
    global_address_adapter_finetune: bool = False
    global_address_positive_weight: float = 1.0
    addressing_hard_negative_finetune: bool = False
    addressing_negative_weight: float = 1.0
    addressing_hard_negative_fraction: float = 0.0
    addressing_hard_negative_weight: float = 1.0
    relation_adaptation_finetune: bool = False
    relation_aux_learning_rate: float = 3e-5
    use_candidate_identity_memory: bool = False
    identity_bptt_windows: int = 1
    use_turn_graph: bool = False
    turn_graph_decay_sec: float = 5.0
    turn_graph_reset_gap_sec: float = 30.0
    turn_graph_teacher_forcing: float = 0.5
    turn_graph_other_only: bool = False
    lambda_self_speaker_bce: float = 0.0
    lambda_self_target_bce: float = 0.0
    lambda_vad_bce: float = 0.0
    vad_threshold: float = 0.5
    use_selected_speaker_addressing: bool = False
    use_vad_response_gate: bool = False
    use_shared_response_gate: bool = False
    shared_response_finetune: bool = False
    lambda_shared_response_bce: float = 0.0
    use_session_category_balanced_response_loss: bool = False
    use_scene_response_head: bool = False
    scene_response_finetune: bool = False
    lambda_scene_response_ce: float = 0.0
    streaming_smoothing_windows: int = 1
    lambda_response_ce: float = 0.0
    response_addressed_weight: float = 1.0
    candidate_classifier_type: str = "spatial"
    candidate_ranking_margin: float = 0.10
    candidate_ce_temperature: float = 0.25
    candidate_topk_fraction: float = 0.20
    candidate_topk_min_pixels: int = 1
    freeze_base_for_candidate_classifier: bool = False
    unfreeze_fusion_for_candidate_classifier: bool = False
    unfreeze_video_last_block_for_candidate_classifier: bool = False
    fusion_learning_rate: float = 3e-5
    video_last_block_learning_rate: float = 1e-5
    unfreeze_audio_encoder_for_candidate_classifier: bool = False
    audio_encoder_learning_rate: float = 1e-5

    skip_ambiguous_multi_target: bool = True

    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160

    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    num_workers: int = 0
    learning_rate: float = 1e-4
    epochs: int = 10
    heatmap_loss: str = "mse"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "EasyComSAALConfig":
        normalized = dict(values)
        for key in ("video_size", "heatmap_size", "head_crop_size"):
            if key in normalized and isinstance(normalized[key], list):
                normalized[key] = tuple(normalized[key])
        known_fields = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in normalized.items() if key in known_fields})

    def resolve_path(self, path: str | Path | None) -> Path | None:
        if path is None:
            return None
        resolved = Path(path)
        if resolved.is_absolute():
            return resolved
        if self.easycom_root is None:
            return resolved
        return Path(self.easycom_root) / resolved


DEFAULT_CONFIG = EasyComSAALConfig()
