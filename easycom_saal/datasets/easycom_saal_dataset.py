from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from easycom_saal.config import EasyComSAALConfig
from easycom_saal.datasets.audio_features import (
    CausalChannelRMSNormalizer,
    compute_multichannel_audio_features,
    extract_audio_clip,
    load_audio_6ch,
    prepare_audio_for_features,
)
from easycom_saal.datasets.easycom_annotations import EasyComAnnotations, ParticipantId
from easycom_saal.datasets.heatmap import bbox_center_on_heatmap, make_heatmap_from_bbox
from easycom_saal.utils.bbox_utils import bbox_inside_frame, bbox_is_visible, bbox_to_xyxy, scale_bbox
from easycom_saal.utils.temporal_pairs import build_temporal_pairs
from easycom_saal.utils.time_alignment import clip_window_around, seconds_to_frame_id, segment_to_clip_centers


@dataclass
class EasyComSAALSample:
    session_id: str
    timestamp: float
    clip_start_time: float
    clip_end_time: float
    center_frame_id: int
    attended_speaker_id: ParticipantId | None
    attended_speaker_ids: tuple[ParticipantId, ...]
    source_speaker_id: ParticipantId | None
    speech_target_id: ParticipantId | None
    camera_wearer_id: ParticipantId
    source_bbox: tuple[float, float, float, float] | None
    video_size: tuple[int, int]
    audio_path: str
    video_path: str
    should_respond: bool
    sample_type: str
    active_source_speaker_ids: tuple[ParticipantId, ...]
    wearer_speaking: bool = False
    wearer_target_ids: tuple[ParticipantId, ...] = ()
    is_overlap: bool = False
    prev_valid_index: int | None = None

    @property
    def target_bbox(self) -> tuple[float, float, float, float] | None:
        """Legacy alias for source-localization and addressed-only reports."""
        return self.source_bbox

    @property
    def should_attend(self) -> bool:
        return self.should_respond


class EasyComSAALDataset(Dataset):
    def __init__(
        self,
        annotations: EasyComAnnotations,
        config: EasyComSAALConfig | None = None,
        session_ids: list[str] | None = None,
        load_media: bool = True,
        return_previous: bool = True,
    ) -> None:
        self.annotations = annotations
        self.config = config or EasyComSAALConfig()
        self.session_ids = session_ids or annotations.iter_session_ids()
        self.load_media = load_media
        self.return_previous = return_previous
        self.skipped_ambiguous_response_windows = 0
        self._video_geometry_cache: dict[str, tuple[tuple[int, int], tuple[int, int], int, int]] = {}
        self._audio_normalizers: dict[str, CausalChannelRMSNormalizer] = {}

        self.samples = self._build_samples()
        prev_indices = build_temporal_pairs(
            self.samples,
            self.config.temporal_max_gap_sec,
            pair_by_source=not self.config.use_timeline_multilabels,
        )
        for sample, prev_index in zip(self.samples, prev_indices):
            sample.prev_valid_index = prev_index

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        item = self._materialize_sample(sample)

        previous_item = None
        apply_temporal_loss = False
        if self.return_previous and sample.prev_valid_index is not None:
            previous = self.samples[sample.prev_valid_index]
            previous_item = self._materialize_sample(previous)
            apply_temporal_loss = True

        item["sample_prev"] = previous_item
        item["apply_temporal_loss"] = apply_temporal_loss
        return item

    def _build_samples(self) -> list[EasyComSAALSample]:
        if self.config.use_timeline_multilabels:
            return self._build_timeline_samples()

        samples: list[EasyComSAALSample] = []
        for session_id in self.session_ids:
            session = self.annotations.get_session_info(session_id)
            camera_wearer_id = self.annotations.get_camera_wearer_id(session_id)
            video_fps = session.video_fps or self.config.video_fps
            video_size = self._session_video_size(session)
            audio_path = self.annotations.resolve_path(session.audio_path)
            video_path = self.annotations.resolve_path(session.video_path)

            if not self._media_paths_are_valid(audio_path, video_path):
                continue

            for segment in self.annotations.get_speech_segments(session_id):
                source_id = self.annotations.get_source_speaker(segment)
                speech_target_id = self.annotations.get_speech_target(segment)

                if not self._is_valid_segment(segment, source_id, speech_target_id, camera_wearer_id):
                    continue

                should_respond = speech_target_id == camera_wearer_id
                if self.config.keep_only_tos_to_camera_wearer and not should_respond:
                    continue

                centers = segment_to_clip_centers(
                    segment.start_time,
                    segment.end_time,
                    self.config.clip_duration_sec,
                    self.config.clip_stride_sec,
                )

                for center_time in centers:
                    clip_start, clip_end = clip_window_around(center_time, self.config.clip_duration_sec)
                    center_frame_id = seconds_to_frame_id(center_time, video_fps)
                    bbox = self.annotations.get_head_bbox(
                        session_id=session_id,
                        participant_id=source_id,
                        frame_id=center_frame_id,
                        timestamp=center_time,
                        fps=video_fps,
                        max_time_delta_sec=self.config.head_bbox_max_time_delta_sec,
                    )
                    if bbox is None or not bbox_is_visible(bbox):
                        continue
                    scene_bbox = self._bbox_in_scene(session_id, bbox, video_size)
                    if not bbox_inside_frame(scene_bbox, video_size, require_center=True):
                        continue

                    samples.append(
                        EasyComSAALSample(
                            session_id=session_id,
                            timestamp=center_time,
                            clip_start_time=clip_start,
                            clip_end_time=clip_end,
                            center_frame_id=center_frame_id,
                            attended_speaker_id=source_id if should_respond else None,
                            attended_speaker_ids=(source_id,) if should_respond else (),
                            source_speaker_id=source_id,
                            speech_target_id=speech_target_id,
                            camera_wearer_id=camera_wearer_id,
                            source_bbox=scene_bbox,
                            video_size=video_size,
                            audio_path=str(audio_path),
                            video_path=str(video_path),
                            should_respond=should_respond,
                            sample_type="addressed_speech" if should_respond else "non_target_speech",
                            active_source_speaker_ids=(source_id,),
                            is_overlap=len(segment.active_speaker_ids) > 1,
                        )
                    )
        if self.config.keep_only_tos_to_camera_wearer:
            return samples
        return self._consolidate_response_samples(samples)

    def _build_timeline_samples(self) -> list[EasyComSAALSample]:
        if self.config.clip_duration_sec <= 0 or self.config.clip_stride_sec <= 0:
            raise ValueError("Clip duration and stride must be positive.")
        if self.config.label_min_overlap_sec < 0:
            raise ValueError("Label minimum overlap must be non-negative.")

        samples: list[EasyComSAALSample] = []
        half_window = self.config.clip_duration_sec / 2.0
        for session_id in self.session_ids:
            session = self.annotations.get_session_info(session_id)
            camera_wearer_id = self.annotations.get_camera_wearer_id(session_id)
            video_fps = session.video_fps or self.config.video_fps
            video_size = self._session_video_size(session)
            audio_path = self.annotations.resolve_path(session.audio_path)
            video_path = self.annotations.resolve_path(session.video_path)
            if not self._media_paths_are_valid(audio_path, video_path):
                continue

            all_segments = self.annotations.get_speech_segments(session_id)
            labelable_segments = []
            for segment in all_segments:
                source_id = self.annotations.get_source_speaker(segment)
                target_id = self.annotations.get_speech_target(segment)
                if self._is_labelable_timeline_segment(segment, source_id, target_id):
                    labelable_segments.append(segment)

            duration = self._session_label_duration(session, all_segments)
            if duration < self.config.clip_duration_sec:
                continue

            center_time = half_window
            while center_time + half_window <= duration + 1e-9:
                clip_start, clip_end = clip_window_around(center_time, self.config.clip_duration_sec)
                center_frame_id = seconds_to_frame_id(center_time, video_fps)
                raw_intersections = [
                    segment
                    for segment in all_segments
                    if self._segment_overlap_sec(segment, clip_start, clip_end) >= self.config.label_min_overlap_sec
                ]
                labelable_intersections = [
                    segment
                    for segment in labelable_segments
                    if self._segment_overlap_sec(segment, clip_start, clip_end) >= self.config.label_min_overlap_sec
                ]

                visible_candidates = self._visible_candidate_bboxes(
                    session_id,
                    center_time,
                    center_frame_id,
                    video_fps,
                    video_size,
                )
                if not visible_candidates:
                    center_time += self.config.clip_stride_sec
                    continue

                # Keep only windows whose intersecting speech is fully labelable.
                # Otherwise an omitted event would become a false negative.
                if len(raw_intersections) != len(labelable_intersections):
                    center_time += self.config.clip_stride_sec
                    continue

                visible_segments = [
                    segment
                    for segment in labelable_intersections
                    if self.annotations.get_source_speaker(segment) != camera_wearer_id
                    and self.annotations.get_source_speaker(segment) in visible_candidates
                ]
                has_unlocalizable_attention_target = any(
                    self.annotations.get_speech_target(segment) == camera_wearer_id
                    and self.annotations.get_source_speaker(segment) != camera_wearer_id
                    and self.annotations.get_source_speaker(segment) not in visible_candidates
                    for segment in labelable_intersections
                )
                if has_unlocalizable_attention_target:
                    center_time += self.config.clip_stride_sec
                    continue

                active_sources = tuple(
                    sorted(
                        {
                            self.annotations.get_source_speaker(segment)
                            for segment in visible_segments
                            if self.annotations.get_source_speaker(segment) is not None
                        },
                        key=str,
                    )
                )
                attended_sources = tuple(
                    sorted(
                        {
                            self.annotations.get_source_speaker(segment)
                            for segment in visible_segments
                            if self.annotations.get_speech_target(segment) == camera_wearer_id
                            and self.annotations.get_source_speaker(segment) is not None
                        },
                        key=str,
                    )
                )
                wearer_segments = [
                    segment
                    for segment in labelable_intersections
                    if self.annotations.get_source_speaker(segment) == camera_wearer_id
                ]
                wearer_targets = tuple(
                    sorted(
                        {
                            self.annotations.get_speech_target(segment)
                            for segment in wearer_segments
                            if self.annotations.get_speech_target(segment) in visible_candidates
                        },
                        key=str,
                    )
                )
                should_respond = bool(attended_sources)
                if self.config.keep_only_tos_to_camera_wearer and not should_respond:
                    center_time += self.config.clip_stride_sec
                    continue
                if not labelable_intersections and not self.config.include_silence_windows:
                    center_time += self.config.clip_stride_sec
                    continue

                representative_id = attended_sources[0] if attended_sources else active_sources[0] if active_sources else None
                representative_segment = next(
                    (
                        segment
                        for segment in visible_segments
                        if self.annotations.get_source_speaker(segment) == representative_id
                    ),
                    None,
                )
                representative_target = (
                    self.annotations.get_speech_target(representative_segment)
                    if representative_segment is not None
                    else None
                )
                representative_bbox = visible_candidates.get(representative_id) if representative_id is not None else None
                sample_type = (
                    "addressed_speech"
                    if should_respond
                    else "non_target_speech"
                    if labelable_intersections
                    else "silence"
                )
                samples.append(
                    EasyComSAALSample(
                        session_id=session_id,
                        timestamp=center_time,
                        clip_start_time=clip_start,
                        clip_end_time=clip_end,
                        center_frame_id=center_frame_id,
                        attended_speaker_id=attended_sources[0] if len(attended_sources) == 1 else None,
                        attended_speaker_ids=attended_sources,
                        source_speaker_id=representative_id,
                        speech_target_id=representative_target,
                        camera_wearer_id=camera_wearer_id,
                        source_bbox=representative_bbox,
                        video_size=video_size,
                        audio_path=str(audio_path),
                        video_path=str(video_path),
                        should_respond=should_respond,
                        sample_type=sample_type,
                        active_source_speaker_ids=active_sources,
                        wearer_speaking=bool(wearer_segments),
                        wearer_target_ids=wearer_targets,
                        is_overlap=self._has_simultaneous_speech(labelable_intersections, clip_start, clip_end),
                    )
                )
                center_time += self.config.clip_stride_sec
        return samples

    def _is_labelable_timeline_segment(
        self,
        segment: Any,
        source_id: ParticipantId | None,
        speech_target_id: ParticipantId | None,
    ) -> bool:
        if segment.ambiguous and self.config.skip_ambiguous_multi_target:
            return False
        if source_id is None or speech_target_id is None:
            return False
        return bool(segment.source_speaker_is_active)

    def _session_label_duration(self, session: Any, segments: list[Any]) -> float:
        media_durations = [
            float(value)
            for value in (session.audio_duration_sec, session.video_duration_sec)
            if value is not None
        ]
        annotation_duration = max((float(segment.end_time) for segment in segments), default=0.0)
        if media_durations:
            media_duration = min(media_durations)
            return min(media_duration, annotation_duration) if annotation_duration > 0 else media_duration
        return annotation_duration

    @staticmethod
    def _segment_overlap_sec(segment: Any, clip_start: float, clip_end: float) -> float:
        return max(0.0, min(float(segment.end_time), clip_end) - max(float(segment.start_time), clip_start))

    @staticmethod
    def _has_simultaneous_speech(segments: list[Any], clip_start: float, clip_end: float) -> bool:
        clipped = sorted(
            (
                max(float(segment.start_time), clip_start),
                min(float(segment.end_time), clip_end),
            )
            for segment in segments
        )
        latest_end = float("-inf")
        for start, end in clipped:
            if start < latest_end:
                return True
            latest_end = max(latest_end, end)
        return False

    def _visible_candidate_bboxes(
        self,
        session_id: str,
        timestamp: float,
        frame_id: int,
        fps: float,
        video_size: tuple[int, int],
    ) -> dict[ParticipantId, tuple[float, float, float, float]]:
        visible = {}
        for participant_id in self.annotations.iter_participant_ids(session_id):
            bbox = self.annotations.get_head_bbox(
                session_id=session_id,
                participant_id=participant_id,
                frame_id=frame_id,
                timestamp=timestamp,
                fps=fps,
                max_time_delta_sec=self.config.head_bbox_max_time_delta_sec,
            )
            if bbox is None or not bbox_is_visible(bbox):
                continue
            scene_bbox = self._bbox_in_scene(session_id, bbox, video_size)
            if not bbox_inside_frame(scene_bbox, video_size, require_center=True):
                continue
            visible[participant_id] = scene_bbox
        return visible

    def sample_type_counts(self) -> dict[str, int]:
        return dict(Counter(sample.sample_type for sample in self.samples))

    def _consolidate_response_samples(self, samples: list[EasyComSAALSample]) -> list[EasyComSAALSample]:
        grouped: dict[tuple[str, float, float], list[EasyComSAALSample]] = defaultdict(list)
        for sample in samples:
            key = (sample.session_id, round(sample.clip_start_time, 9), round(sample.clip_end_time, 9))
            grouped[key].append(sample)

        consolidated: list[EasyComSAALSample] = []
        for group in grouped.values():
            addressed_by_source = {
                str(sample.source_speaker_id): sample
                for sample in group
                if sample.should_respond
            }
            if len(addressed_by_source) > 1:
                self.skipped_ambiguous_response_windows += 1
                continue

            chosen = next(iter(addressed_by_source.values()), group[0])
            active_sources = tuple(
                sorted(
                    {source_id for sample in group for source_id in sample.active_source_speaker_ids},
                    key=str,
                )
            )
            consolidated.append(
                replace(
                    chosen,
                    active_source_speaker_ids=active_sources,
                    is_overlap=chosen.is_overlap or len(active_sources) > 1,
                )
            )
        return consolidated

    def _is_valid_segment(
        self,
        segment: Any,
        source_id: ParticipantId | None,
        speech_target_id: ParticipantId | None,
        camera_wearer_id: ParticipantId,
    ) -> bool:
        if segment.ambiguous and self.config.skip_ambiguous_multi_target:
            return False
        if speech_target_id is None:
            return False
        if source_id is None:
            return False
        if source_id == camera_wearer_id:
            return False
        if not segment.source_speaker_is_active:
            return False
        return True

    def _session_video_size(self, session: Any) -> tuple[int, int]:
        if session.video_height is not None and session.video_width is not None:
            return int(session.video_height), int(session.video_width)
        return self.config.video_size

    def _video_geometry(
        self,
        session_id: str,
        scene_size: tuple[int, int],
    ) -> tuple[tuple[int, int], tuple[int, int], int, int]:
        cached = self._video_geometry_cache.get(session_id)
        if cached is not None:
            return cached

        session = self.annotations.get_session_info(session_id)
        source_size = scene_size
        if self.config.crop_composite_video_to_manifest:
            video_path = self.annotations.resolve_path(session.video_path)
            if video_path is not None and video_path.exists():
                probed_size = _probe_video_size_cv2(str(video_path))
                if probed_size is not None:
                    source_size = probed_size
        crop_top, crop_left = _scene_crop_offset(source_size, scene_size)
        geometry = source_size, scene_size, crop_top, crop_left
        self._video_geometry_cache[session_id] = geometry
        return geometry

    def _bbox_in_scene(
        self,
        session_id: str,
        bbox: Any,
        scene_size: tuple[int, int],
    ) -> tuple[float, float, float, float]:
        _, _, crop_top, crop_left = self._video_geometry(session_id, scene_size)
        return _translate_bbox_to_scene(bbox, crop_top=crop_top, crop_left=crop_left)

    def _media_paths_are_valid(self, audio_path: Path | None, video_path: Path | None) -> bool:
        if not self.config.require_media_exists:
            return audio_path is not None and video_path is not None
        return audio_path is not None and audio_path.exists() and video_path is not None and video_path.exists()

    def _materialize_sample(self, sample: EasyComSAALSample) -> dict[str, Any]:
        if sample.source_bbox is None:
            source_heatmap = torch.zeros(self.config.heatmap_size, dtype=torch.float32)
            source_center = torch.zeros(2, dtype=torch.float32)
            source_bbox = torch.zeros(4, dtype=torch.float32)
        else:
            source_heatmap = make_heatmap_from_bbox(
                sample.source_bbox,
                video_size=sample.video_size,
                heatmap_size=self.config.heatmap_size,
                sigma_mode=self.config.heatmap_sigma_mode,
                sigma_ratio=self.config.heatmap_sigma_ratio,
                fixed_sigma=self.config.heatmap_fixed_sigma,
            )
            source_center = torch.tensor(
                bbox_center_on_heatmap(sample.source_bbox, sample.video_size, self.config.heatmap_size),
                dtype=torch.float32,
            )
            source_bbox = torch.tensor(sample.source_bbox, dtype=torch.float32)
        (
            candidate_bboxes,
            candidate_bbox_sequences,
            speaker_labels,
            response_labels,
            wearer_target_labels,
            candidate_ids,
        ) = self._candidate_tensors(sample)

        if self.load_media:
            video_clip, candidate_head_crops = self._load_video_clip_and_head_crops(
                sample,
                candidate_bbox_sequences,
            )
            audio_waveform, audio_spectrogram, audio_channel_correlation = self._load_audio_features(sample)
        else:
            video_clip = torch.empty(0)
            candidate_head_crops = torch.empty(0)
            audio_waveform = torch.empty(0)
            audio_spectrogram = torch.empty(0)
            audio_channel_correlation = torch.empty(0)

        return {
            "video_clip": video_clip,
            "candidate_head_crops": candidate_head_crops,
            "audio_waveform": audio_waveform,
            "audio_spectrogram": audio_spectrogram,
            "audio_channel_correlation": audio_channel_correlation,
            # Legacy names continue to represent source-speaker localization.
            "target_heatmap": source_heatmap,
            "source_heatmap": source_heatmap,
            "attended_speaker_id": sample.attended_speaker_id,
            "attended_speaker_ids": sample.attended_speaker_ids,
            "source_speaker_id": sample.source_speaker_id,
            "speech_target_id": sample.speech_target_id,
            "camera_wearer_id": sample.camera_wearer_id,
            "target_bbox": source_bbox,
            "source_bbox": source_bbox,
            "target_center": source_center,
            "source_center": source_center,
            "candidate_bboxes_heatmap": candidate_bboxes,
            "candidate_bboxes_sequence_heatmap": candidate_bbox_sequences,
            "candidate_ids": candidate_ids,
            # Historical candidate losses rank the person to respond to.
            "candidate_labels": response_labels,
            "attention_candidate_labels": response_labels,
            "speaker_candidate_labels": speaker_labels,
            "response_candidate_labels": response_labels,
            "wearer_target_candidate_labels": wearer_target_labels,
            "wearer_speaking": torch.tensor(sample.wearer_speaking, dtype=torch.bool),
            "attention_label": torch.tensor(float(sample.should_attend), dtype=torch.float32),
            "response_label": torch.tensor(float(sample.should_respond), dtype=torch.float32),
            "should_attend": sample.should_attend,
            "should_respond": sample.should_respond,
            "speech_present": torch.tensor(sample.sample_type != "silence", dtype=torch.bool),
            "sample_type": sample.sample_type,
            "active_source_speaker_ids": sample.active_source_speaker_ids,
            "timestamp": torch.tensor(sample.timestamp, dtype=torch.float32),
            "session_id": sample.session_id,
            "valid": True,
            "is_overlap": sample.is_overlap,
        }

    def _candidate_tensors(
        self,
        sample: EasyComSAALSample,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[ParticipantId, ...],
    ]:
        session = self.annotations.get_session_info(sample.session_id)
        video_fps = session.video_fps or self.config.video_fps
        frame_ids = self._clip_frame_ids(sample, video_fps)
        candidate_bboxes = []
        candidate_bbox_sequences = []
        speaker_labels = []
        response_labels = []
        wearer_target_labels = []
        candidate_ids = []
        for participant_id in self.annotations.iter_participant_ids(sample.session_id):
            bbox = self.annotations.get_head_bbox(
                session_id=sample.session_id,
                participant_id=participant_id,
                frame_id=sample.center_frame_id,
                timestamp=sample.timestamp,
                fps=session.video_fps,
                max_time_delta_sec=self.config.head_bbox_max_time_delta_sec,
            )
            if bbox is None or not bbox_is_visible(bbox):
                continue
            center_bbox = self._bbox_in_scene(sample.session_id, bbox, sample.video_size)
            if not bbox_inside_frame(center_bbox, sample.video_size, require_center=True):
                continue
            candidate_bboxes.append(scale_bbox(center_bbox, sample.video_size, self.config.heatmap_size))
            candidate_ids.append(participant_id)
            candidate_bbox_sequences.append(
                [
                    scale_bbox(temporal_bbox, sample.video_size, self.config.heatmap_size)
                    for temporal_bbox in self._candidate_bbox_sequence(
                        sample,
                        participant_id=participant_id,
                        frame_ids=frame_ids,
                        fps=video_fps,
                        fallback_bbox=center_bbox,
                    )
                ]
            )
            is_active_source = participant_id in sample.active_source_speaker_ids
            is_response_target = participant_id in sample.attended_speaker_ids
            speaker_labels.append(1.0 if is_active_source else 0.0)
            response_labels.append(1.0 if is_response_target else 0.0)
            wearer_target_labels.append(1.0 if participant_id in sample.wearer_target_ids else 0.0)

        if not candidate_bboxes:
            frames = max(1, int(round(self.config.clip_duration_sec * self.config.video_fps)))
            return (
                torch.empty((0, 4), dtype=torch.float32),
                torch.empty((0, frames, 4), dtype=torch.float32),
                torch.empty((0,), dtype=torch.float32),
                torch.empty((0,), dtype=torch.float32),
                torch.empty((0,), dtype=torch.float32),
                (),
            )
        return (
            torch.tensor(candidate_bboxes, dtype=torch.float32),
            torch.tensor(candidate_bbox_sequences, dtype=torch.float32),
            torch.tensor(speaker_labels, dtype=torch.float32),
            torch.tensor(response_labels, dtype=torch.float32),
            torch.tensor(wearer_target_labels, dtype=torch.float32),
            tuple(candidate_ids),
        )

    def _clip_frame_ids(self, sample: EasyComSAALSample, fps: float) -> list[int]:
        expected_frames = max(1, int(round(self.config.clip_duration_sec * self.config.video_fps)))
        start_frame = max(0, int(round(float(sample.clip_start_time) * float(fps))))
        end_frame = max(start_frame + 1, int(round(float(sample.clip_end_time) * float(fps))))
        return torch.linspace(start_frame, end_frame - 1, expected_frames).round().long().tolist()

    def _candidate_bbox_sequence(
        self,
        sample: EasyComSAALSample,
        participant_id: ParticipantId,
        frame_ids: list[int],
        fps: float,
        fallback_bbox: tuple[float, float, float, float],
    ) -> list[tuple[float, float, float, float]]:
        sequence = []
        last_bbox = fallback_bbox
        for frame_id in frame_ids:
            bbox = self.annotations.get_head_bbox(
                session_id=sample.session_id,
                participant_id=participant_id,
                frame_id=int(frame_id),
                timestamp=float(frame_id) / float(fps),
                fps=fps,
                max_time_delta_sec=self.config.head_bbox_max_time_delta_sec,
            )
            if bbox is not None and bbox_is_visible(bbox):
                scene_bbox = self._bbox_in_scene(sample.session_id, bbox, sample.video_size)
                if bbox_inside_frame(scene_bbox, sample.video_size, require_center=True):
                    last_bbox = scene_bbox
            sequence.append(last_bbox)
        return sequence

    def _load_audio_features(self, sample: EasyComSAALSample) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        waveform, sample_rate = load_audio_6ch(
            sample.audio_path,
            expected_channels=self.config.num_audio_channels,
            target_sample_rate=self.config.sample_rate,
        )
        clip = extract_audio_clip(
            waveform,
            start_time=sample.clip_start_time,
            end_time=sample.clip_end_time,
            sample_rate=sample_rate,
            target_num_samples=max(1, int(round(self.config.clip_duration_sec * sample_rate))),
        )
        clip = perturb_audio_clip(
            clip,
            fixed_gain_db=(
                self.config.audio_fixed_gain_db
                + _stable_clip_gain_db(
                    sample.session_id,
                    self.config.audio_global_gain_augmentation_db,
                    self.config.audio_perturbation_seed,
                )
            ),
            noise_snr_db=self.config.audio_noise_snr_db,
            channel_gain_std_db=self.config.audio_channel_gain_std_db,
            seed=(
                _stable_recording_audio_seed(
                    sample.session_id,
                    self.config.audio_perturbation_seed,
                )
                if self.config.audio_channel_gain_per_recording
                else _stable_audio_seed(sample, self.config.audio_perturbation_seed)
            ),
        )
        if self.config.use_causal_channel_rms_normalization:
            normalizer = self._audio_normalizers.get(sample.session_id)
            if normalizer is None:
                normalizer = CausalChannelRMSNormalizer(
                    num_channels=self.config.num_audio_channels,
                    tau_sec=self.config.audio_normalization_tau_sec,
                    target_rms_db=self.config.audio_normalization_target_rms_db,
                    max_gain_db=self.config.audio_normalization_max_gain_db,
                )
                self._audio_normalizers[sample.session_id] = normalizer
            clip = normalizer(clip, timestamp=sample.timestamp)
        clip = prepare_audio_for_features(clip, preserve_energy=self.config.preserve_audio_energy)
        spectrogram, correlation = compute_multichannel_audio_features(
            clip,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
        )
        return clip, spectrogram, correlation

    def _load_video_clip_and_head_crops(
        self,
        sample: EasyComSAALSample,
        candidate_bbox_sequences: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            import cv2
        except Exception as exc:
            raise ImportError("opencv-python is required for video loading") from exc

        expected_frames = max(1, int(round(self.config.clip_duration_sec * self.config.video_fps)))
        frames = _read_video_clip_cv2(
            sample.video_path,
            start_time=sample.clip_start_time,
            end_time=sample.clip_end_time,
            target_frames=expected_frames,
            scene_size=(
                sample.video_size
                if self.config.crop_composite_video_to_manifest
                else None
            ),
        )
        frames = _resample_frames(frames, expected_frames)
        candidate_head_crops = (
            extract_candidate_head_crops(
                frames,
                candidate_bbox_sequences,
                bbox_size=self.config.heatmap_size,
                output_size=self.config.head_crop_size,
                padding=self.config.head_crop_padding,
            )
            if self.config.use_head_crop_branch
            else torch.empty(0)
        )
        if self.config.use_masked_head_video:
            frames = mask_frames_to_candidate_heads(
                frames,
                candidate_bbox_sequences,
                bbox_size=self.config.heatmap_size,
                padding=self.config.head_crop_padding,
            )
        resized_frames = F.interpolate(
            frames,
            size=self.config.video_size,
            mode="bilinear",
            align_corners=False,
        )
        return resized_frames.contiguous(), candidate_head_crops

    def _load_video_clip(self, sample: EasyComSAALSample) -> torch.Tensor:
        video_clip, _ = self._load_video_clip_and_head_crops(sample, torch.empty(0))
        return video_clip


def _stable_audio_seed(sample: EasyComSAALSample, base_seed: int) -> int:
    key = f"{sample.session_id}|{sample.timestamp:.6f}|{int(base_seed)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little") % (2**31)


def _stable_recording_audio_seed(session_id: str, base_seed: int) -> int:
    key = f"{session_id}|channel-gain|{int(base_seed)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little") % (2**31)


def _stable_clip_gain_db(session_id: str, maximum_db: float, base_seed: int) -> float:
    if maximum_db <= 0.0:
        return 0.0
    key = f"{session_id}|gain|{int(base_seed)}".encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
    unit = integer / float(2**64 - 1)
    return (2.0 * unit - 1.0) * float(maximum_db)


def perturb_audio_clip(
    clip: torch.Tensor,
    *,
    fixed_gain_db: float = 0.0,
    noise_snr_db: float | None = None,
    channel_gain_std_db: float = 0.0,
    seed: int = 0,
) -> torch.Tensor:
    """Apply deterministic gain/noise perturbations before feature extraction."""
    if fixed_gain_db == 0.0 and noise_snr_db is None and channel_gain_std_db == 0.0:
        return clip
    generator = torch.Generator(device=clip.device).manual_seed(int(seed))
    perturbed = clip * (10.0 ** (float(fixed_gain_db) / 20.0))
    if channel_gain_std_db > 0.0:
        channel_gain_db = torch.randn(
            (perturbed.shape[0], 1),
            generator=generator,
            device=perturbed.device,
            dtype=perturbed.dtype,
        ) * float(channel_gain_std_db)
        perturbed = perturbed * torch.pow(10.0, channel_gain_db / 20.0)
    if noise_snr_db is not None:
        reference_rms = perturbed.square().mean().sqrt().clamp_min(1e-4)
        noise_rms = reference_rms / (10.0 ** (float(noise_snr_db) / 20.0))
        noise = torch.randn(
            perturbed.shape,
            generator=generator,
            device=perturbed.device,
            dtype=perturbed.dtype,
        )
        perturbed = perturbed + noise * noise_rms
    return perturbed.clamp(-1.0, 1.0)


def _read_video_clip_cv2(
    video_path: str,
    start_time: float,
    end_time: float,
    target_frames: int,
    scene_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 20.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start_frame = max(0, int(round(float(start_time) * float(fps))))
    end_frame = max(start_frame + 1, int(round(float(end_time) * float(fps))))
    if frame_count > 0:
        end_frame = min(end_frame, frame_count)

    if end_frame <= start_frame:
        capture.release()
        raise RuntimeError(f"No video frames available for {video_path} at {start_time:.3f}-{end_time:.3f}s")

    positions = torch.linspace(start_frame, end_frame - 1, target_frames).round().long().tolist()
    wanted_positions = set(positions)
    decoded: dict[int, torch.Tensor] = {}
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for frame_id in range(start_frame, end_frame):
        ok, frame_bgr = capture.read()
        if not ok:
            continue
        if frame_id not in wanted_positions:
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if scene_size is not None:
            frame_rgb = _crop_composite_frame_to_scene(frame_rgb, scene_size)
        decoded[frame_id] = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
    capture.release()

    frames = [decoded[frame_id] for frame_id in positions if frame_id in decoded]
    if not frames:
        raise RuntimeError(f"No video frames decoded for {video_path} at {start_time:.3f}-{end_time:.3f}s")
    return torch.stack(frames, dim=0)


def _probe_video_size_cv2(video_path: str) -> tuple[int, int] | None:
    try:
        import cv2
    except Exception:
        return None

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0))
    capture.release()
    if width <= 0 or height <= 0:
        return None
    return height, width


def _scene_crop_offset(
    source_size: tuple[int, int],
    scene_size: tuple[int, int],
) -> tuple[int, int]:
    """Return the top/left offset of a right-aligned scene in a composite frame."""
    source_height, source_width = source_size
    scene_height, scene_width = scene_size
    if source_height == scene_height and source_width >= scene_width:
        return 0, source_width - scene_width
    return 0, 0


def _translate_bbox_to_scene(
    bbox: Any,
    crop_top: int,
    crop_left: int,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox_to_xyxy(bbox)
    return x1 - crop_left, y1 - crop_top, x2 - crop_left, y2 - crop_top


def _crop_composite_frame_to_scene(frame: Any, scene_size: tuple[int, int]) -> Any:
    """Remove EasyCom's left thumbnail sidebar while preserving the RGB scene."""
    scene_height, scene_width = scene_size
    source_height, source_width = frame.shape[:2]
    crop_top, crop_left = _scene_crop_offset(
        (source_height, source_width),
        (scene_height, scene_width),
    )
    if crop_top == 0 and crop_left == 0:
        return frame
    return frame[crop_top : crop_top + scene_height, crop_left : crop_left + scene_width]


def _resample_frames(frames: torch.Tensor, target_frames: int) -> torch.Tensor:
    if frames.shape[0] == target_frames:
        return frames
    positions = torch.linspace(0, frames.shape[0] - 1, target_frames)
    indices = positions.round().long().clamp(0, frames.shape[0] - 1)
    return frames.index_select(0, indices)


def extract_candidate_head_crops(
    frames: torch.Tensor,
    candidate_bbox_sequences: torch.Tensor,
    bbox_size: tuple[int, int],
    output_size: tuple[int, int],
    padding: float = 0.18,
) -> torch.Tensor:
    """Crop tracked candidates from source-resolution frames as [N,T,C,H,W]."""
    if frames.ndim != 4:
        raise ValueError(f"Expected source frames [T,C,H,W], got {tuple(frames.shape)}")
    steps, channels, frame_height, frame_width = frames.shape
    if candidate_bbox_sequences.numel() == 0:
        return frames.new_empty((0, steps, channels, *output_size))
    if candidate_bbox_sequences.ndim != 3 or candidate_bbox_sequences.shape[1] != steps:
        raise ValueError(
            f"Expected candidate boxes [N,{steps},4], got {tuple(candidate_bbox_sequences.shape)}"
        )

    bbox_height, bbox_width = bbox_size
    candidate_crops = []
    for box_sequence in candidate_bbox_sequences:
        temporal_crops = []
        for frame, bbox in zip(frames, box_sequence):
            x1, y1, x2, y2 = bbox.unbind()
            x1 = float(x1.item()) * frame_width / float(bbox_width)
            x2 = float(x2.item()) * frame_width / float(bbox_width)
            y1 = float(y1.item()) * frame_height / float(bbox_height)
            y2 = float(y2.item()) * frame_height / float(bbox_height)
            box_width_pixels = max(1.0, abs(x2 - x1))
            box_height_pixels = max(1.0, abs(y2 - y1))
            x_start = max(0, int(round(min(x1, x2) - padding * box_width_pixels)))
            x_end = min(frame_width, int(round(max(x1, x2) + padding * box_width_pixels)))
            y_start = max(0, int(round(min(y1, y2) - padding * box_height_pixels)))
            y_end = min(frame_height, int(round(max(y1, y2) + padding * box_height_pixels)))
            if x_end <= x_start or y_end <= y_start:
                crop = frame.new_zeros((1, channels, *output_size))
            else:
                crop = F.interpolate(
                    frame[:, y_start:y_end, x_start:x_end].unsqueeze(0),
                    size=output_size,
                    mode="bilinear",
                    align_corners=False,
                )
            temporal_crops.append(crop.squeeze(0))
        candidate_crops.append(torch.stack(temporal_crops))
    return torch.stack(candidate_crops).contiguous()


def mask_frames_to_candidate_heads(
    frames: torch.Tensor,
    candidate_bbox_sequences: torch.Tensor,
    bbox_size: tuple[int, int],
    padding: float = 0.18,
) -> torch.Tensor:
    """Keep all tracked heads at their scene positions and black out the background."""
    if frames.ndim != 4:
        raise ValueError(f"Expected source frames [T,C,H,W], got {tuple(frames.shape)}")
    if candidate_bbox_sequences.numel() == 0:
        return torch.zeros_like(frames)
    if candidate_bbox_sequences.ndim != 3 or candidate_bbox_sequences.shape[1] != frames.shape[0]:
        raise ValueError(
            f"Expected candidate boxes [N,{frames.shape[0]},4], got {tuple(candidate_bbox_sequences.shape)}"
        )

    _, _, frame_height, frame_width = frames.shape
    bbox_height, bbox_width = bbox_size
    mask = frames.new_zeros((frames.shape[0], 1, frame_height, frame_width))
    for candidate_boxes in candidate_bbox_sequences:
        for step, bbox in enumerate(candidate_boxes):
            x1, y1, x2, y2 = bbox.unbind()
            x1 = float(x1.item()) * frame_width / float(bbox_width)
            x2 = float(x2.item()) * frame_width / float(bbox_width)
            y1 = float(y1.item()) * frame_height / float(bbox_height)
            y2 = float(y2.item()) * frame_height / float(bbox_height)
            box_width_pixels = max(1.0, abs(x2 - x1))
            box_height_pixels = max(1.0, abs(y2 - y1))
            x_start = max(0, int(round(min(x1, x2) - padding * box_width_pixels)))
            x_end = min(frame_width, int(round(max(x1, x2) + padding * box_width_pixels)))
            y_start = max(0, int(round(min(y1, y2) - padding * box_height_pixels)))
            y_end = min(frame_height, int(round(max(y1, y2) + padding * box_height_pixels)))
            if x_end > x_start and y_end > y_start:
                mask[step, :, y_start:y_end, x_start:x_end] = 1.0
    return (frames * mask).contiguous()


def _prepare_for_collate(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().clone().contiguous()
    if isinstance(value, dict):
        return {key: _prepare_for_collate(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_prepare_for_collate(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_prepare_for_collate(item) for item in value)
    return value


def saal_collate_fn(items: list[dict[str, Any]]) -> dict[str, Any]:
    list_keys = {
        "active_source_speaker_ids",
        "attention_candidate_labels",
        "attended_speaker_id",
        "attended_speaker_ids",
        "camera_wearer_id",
        "candidate_bboxes_heatmap",
        "candidate_bboxes_sequence_heatmap",
        "candidate_head_crops",
        "candidate_ids",
        "candidate_labels",
        "response_candidate_labels",
        "sample_type",
        "session_id",
        "source_speaker_id",
        "speaker_candidate_labels",
        "speech_target_id",
        "wearer_target_candidate_labels",
    }
    current_items = [
        {
            key: _prepare_for_collate(value)
            for key, value in item.items()
            if key not in {"sample_prev", "apply_temporal_loss", *list_keys}
        }
        for item in items
    ]
    batch = default_collate(current_items)
    for key in list_keys:
        batch[key] = [_prepare_for_collate(item[key]) for item in items]

    temporal_mask = torch.tensor([bool(item["apply_temporal_loss"]) for item in items], dtype=torch.bool)
    previous_items = []
    for current, item in zip(current_items, items):
        if item["sample_prev"] is None:
            previous_items.append(current)
        else:
            previous_items.append(
                {
                    key: _prepare_for_collate(value)
                    for key, value in item["sample_prev"].items()
                    if key not in list_keys
                }
            )

    previous_batch = default_collate(previous_items)
    for key, value in previous_batch.items():
        batch[f"prev_{key}"] = value
    for key in list_keys:
        batch[f"prev_{key}"] = [
            _prepare_for_collate((item["sample_prev"] if item["sample_prev"] is not None else item)[key])
            for item in items
        ]
    batch["apply_temporal_loss"] = temporal_mask
    return batch
