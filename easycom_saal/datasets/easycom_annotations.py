from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ParticipantId = int | str


@dataclass
class SpeechSegment:
    session_id: str
    start_time: float
    end_time: float
    source_speaker_id: ParticipantId | None
    speech_target_id: ParticipantId | None
    source_speaker_is_active: bool = True
    segment_id: str | None = None
    ambiguous: bool = False
    active_speaker_ids: tuple[ParticipantId, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HeadBBox:
    session_id: str
    participant_id: ParticipantId
    x1: float
    y1: float
    x2: float
    y2: float
    frame_id: int | None = None
    timestamp: float | None = None
    visible: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionInfo:
    session_id: str
    camera_wearer_id: ParticipantId
    audio_path: str | None = None
    video_path: str | None = None
    video_width: int | None = None
    video_height: int | None = None
    video_fps: float | None = None
    audio_duration_sec: float | None = None
    video_duration_sec: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _first(row: dict[str, Any], aliases: tuple[str, ...], default: Any = None) -> Any:
    for alias in aliases:
        if alias in row and row[alias] not in ("", None):
            return row[alias]
    lower = {str(key).lower(): value for key, value in row.items()}
    for alias in aliases:
        value = lower.get(alias.lower())
        if value not in ("", None):
            return value
    return default


def _as_bool(value: Any, default: bool = False) -> bool:
    if value in ("", None):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "visible", "active"}


def _as_id(value: Any) -> ParticipantId | None:
    if value in ("", None, "None", "none", "null", "nan"):
        return None
    if isinstance(value, (list, tuple, set)):
        values = [_as_id(item) for item in value]
        values = [item for item in values if item is not None]
        if len(values) == 1:
            return values[0]
        return None
    try:
        as_float = float(value)
        as_int = int(as_float)
        if as_float == as_int:
            return as_int
    except (TypeError, ValueError):
        pass
    return str(value)


def _as_id_tuple(value: Any) -> tuple[ParticipantId, ...]:
    if value in ("", None):
        return ()
    if isinstance(value, str):
        value = value.replace(";", ",").replace("|", ",")
        return tuple(item for item in (_as_id(part.strip()) for part in value.split(",")) if item is not None)
    if isinstance(value, (list, tuple, set)):
        return tuple(item for item in (_as_id(part) for part in value) if item is not None)
    single = _as_id(value)
    return () if single is None else (single,)


def _as_float(value: Any, default: float | None = None) -> float | None:
    if value in ("", None):
        return default
    return float(value)


def _as_int(value: Any, default: int | None = None) -> int | None:
    if value in ("", None):
        return default
    return int(float(value))


def _load_table(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    raise ValueError(f"Unsupported annotation file type: {path}")


class EasyComAnnotations:
    def __init__(
        self,
        sessions: list[SessionInfo],
        speech_segments: list[SpeechSegment],
        head_bboxes: list[HeadBBox],
        root: str | Path | None = None,
    ) -> None:
        self.root = Path(root) if root is not None else None
        self.sessions = {session.session_id: session for session in sessions}
        self.speech_segments = sorted(speech_segments, key=lambda item: (item.session_id, item.start_time))

        self._segments_by_session: dict[str, list[SpeechSegment]] = defaultdict(list)
        for segment in self.speech_segments:
            self._segments_by_session[segment.session_id].append(segment)

        self._bboxes_by_key: dict[tuple[str, ParticipantId], list[HeadBBox]] = defaultdict(list)
        self._bboxes_by_frame: dict[tuple[str, ParticipantId], dict[int, HeadBBox]] = defaultdict(dict)
        self._participant_ids_by_session: dict[str, set[ParticipantId]] = defaultdict(set)
        for bbox in head_bboxes:
            key = (bbox.session_id, bbox.participant_id)
            self._bboxes_by_key[key].append(bbox)
            self._participant_ids_by_session[bbox.session_id].add(bbox.participant_id)
            if bbox.frame_id is not None:
                self._bboxes_by_frame[key][bbox.frame_id] = bbox
        for bboxes in self._bboxes_by_key.values():
            bboxes.sort(key=lambda item: (item.frame_id if item.frame_id is not None else 10**12, item.timestamp or 0.0))

    @classmethod
    def from_manifest(cls, manifest_path: str | Path, root: str | Path | None = None) -> "EasyComAnnotations":
        manifest_path = Path(manifest_path)
        manifest = _load_table(manifest_path)
        if not isinstance(manifest, dict):
            raise ValueError("Manifest must be a JSON object with sessions, speech_segments, and head_bboxes")

        manifest_root = root if root is not None else manifest.get("easycom_root")
        sessions = [_normalize_session(row) for row in manifest.get("sessions", [])]
        speech_segments = [_normalize_speech_segment(row) for row in manifest.get("speech_segments", [])]
        head_bboxes = [_normalize_head_bbox(row) for row in manifest.get("head_bboxes", [])]

        return cls(sessions=sessions, speech_segments=speech_segments, head_bboxes=head_bboxes, root=manifest_root)

    @classmethod
    def from_files(
        cls,
        sessions_path: str | Path,
        speech_segments_path: str | Path,
        head_bboxes_path: str | Path,
        root: str | Path | None = None,
    ) -> "EasyComAnnotations":
        sessions = [_normalize_session(row) for row in _load_table(Path(sessions_path))]
        speech_segments = [_normalize_speech_segment(row) for row in _load_table(Path(speech_segments_path))]
        head_bboxes = [_normalize_head_bbox(row) for row in _load_table(Path(head_bboxes_path))]
        return cls(sessions=sessions, speech_segments=speech_segments, head_bboxes=head_bboxes, root=root)

    def iter_session_ids(self) -> list[str]:
        return sorted(self.sessions)

    def get_session_info(self, session_id: str) -> SessionInfo:
        return self.sessions[session_id]

    def get_speech_segments(self, session_id: str) -> list[SpeechSegment]:
        return list(self._segments_by_session.get(session_id, []))

    def get_camera_wearer_id(self, session_id: str) -> ParticipantId:
        return self.sessions[session_id].camera_wearer_id

    def iter_participant_ids(self, session_id: str | None = None) -> list[ParticipantId]:
        participant_ids = (
            set().union(*self._participant_ids_by_session.values())
            if session_id is None and self._participant_ids_by_session
            else self._participant_ids_by_session.get(session_id, set())
        )
        return sorted(participant_ids, key=str)

    def get_source_speaker(self, segment: SpeechSegment) -> ParticipantId | None:
        return segment.source_speaker_id

    def get_speech_target(self, segment: SpeechSegment) -> ParticipantId | None:
        return segment.speech_target_id

    def get_head_bbox(
        self,
        session_id: str,
        participant_id: ParticipantId,
        frame_id: int | None = None,
        timestamp: float | None = None,
        fps: float | None = None,
        max_time_delta_sec: float = 0.08,
    ) -> HeadBBox | None:
        if max_time_delta_sec < 0:
            raise ValueError("max_time_delta_sec must be non-negative")
        bboxes = self._bboxes_by_key.get((session_id, participant_id), [])
        if not bboxes:
            return None

        if frame_id is not None:
            exact = self._bboxes_by_frame.get((session_id, participant_id), {}).get(frame_id)
            if exact is not None:
                return exact

        if timestamp is None and frame_id is not None and fps is not None:
            timestamp = frame_id / float(fps)

        if timestamp is not None:
            candidates = [bbox for bbox in bboxes if bbox.timestamp is not None]
            if candidates:
                best = min(candidates, key=lambda bbox: abs(float(bbox.timestamp) - float(timestamp)))
                if abs(float(best.timestamp) - float(timestamp)) <= max_time_delta_sec:
                    return best

        if frame_id is not None:
            with_frame = [bbox for bbox in bboxes if bbox.frame_id is not None]
            if with_frame:
                best = min(with_frame, key=lambda bbox: abs(int(bbox.frame_id) - int(frame_id)))
                if fps is not None and fps > 0:
                    frame_delta_sec = abs(int(best.frame_id) - int(frame_id)) / float(fps)
                    if frame_delta_sec <= max_time_delta_sec:
                        return best
        return None

    def resolve_path(self, path: str | None) -> Path | None:
        if path is None:
            return None
        resolved = Path(path)
        if resolved.is_absolute() or self.root is None:
            return resolved
        return self.root / resolved


def _normalize_session(row: dict[str, Any]) -> SessionInfo:
    session_id = str(_first(row, ("session_id", "session", "recording_id", "video_id", "conversation_id")))
    return SessionInfo(
        session_id=session_id,
        camera_wearer_id=_as_id(_first(row, ("camera_wearer_id", "wearer_id", "ego_participant_id", "ego_id"))),
        audio_path=_first(row, ("audio_path", "audio", "mic_array_audio_path", "microphone_array_audio_path")),
        video_path=_first(row, ("video_path", "video", "rgb_video_path", "egocentric_video_path")),
        video_width=_as_int(_first(row, ("video_width", "width", "frame_width"))),
        video_height=_as_int(_first(row, ("video_height", "height", "frame_height"))),
        video_fps=_as_float(_first(row, ("video_fps", "fps", "frame_rate"))),
        audio_duration_sec=_as_float(_first(row, ("audio_duration_sec", "audio_duration"))),
        video_duration_sec=_as_float(_first(row, ("video_duration_sec", "video_duration"))),
        metadata=dict(row),
    )


def _normalize_speech_segment(row: dict[str, Any]) -> SpeechSegment:
    return SpeechSegment(
        session_id=str(_first(row, ("session_id", "session", "recording_id", "video_id", "conversation_id"))),
        start_time=float(_first(row, ("start_time", "start_time_sec", "start_sec", "start", "t_start"))),
        end_time=float(_first(row, ("end_time", "end_time_sec", "end_sec", "end", "t_end"))),
        source_speaker_id=_as_id(
            _first(row, ("source_speaker_id", "speaker_id", "source_id", "talker_id", "source_speaker"))
        ),
        speech_target_id=_as_id(
            _first(row, ("speech_target_id", "target_id", "target_of_speech_id", "tos_id", "addressee_id"))
        ),
        source_speaker_is_active=_as_bool(
            _first(row, ("source_speaker_is_active", "is_active", "speaking", "speech_activity"), True),
            default=True,
        ),
        segment_id=str(_first(row, ("segment_id", "utterance_id", "id"), "")) or None,
        ambiguous=_as_bool(_first(row, ("ambiguous", "is_ambiguous", "multi_target_ambiguous"), False)),
        active_speaker_ids=_as_id_tuple(_first(row, ("active_speaker_ids", "active_speakers", "overlap_speakers"))),
        metadata=dict(row),
    )


def _normalize_head_bbox(row: dict[str, Any]) -> HeadBBox:
    x1 = _as_float(_first(row, ("x1", "left")))
    y1 = _as_float(_first(row, ("y1", "top")))
    x2 = _as_float(_first(row, ("x2", "right")))
    y2 = _as_float(_first(row, ("y2", "bottom")))

    if x1 is None or y1 is None or x2 is None or y2 is None:
        x = float(_first(row, ("x", "bbox_x")))
        y = float(_first(row, ("y", "bbox_y")))
        width = float(_first(row, ("w", "width", "bbox_width")))
        height = float(_first(row, ("h", "height", "bbox_height")))
        x1, y1, x2, y2 = x, y, x + width, y + height

    return HeadBBox(
        session_id=str(_first(row, ("session_id", "session", "recording_id", "video_id", "conversation_id"))),
        participant_id=_as_id(_first(row, ("participant_id", "person_id", "speaker_id", "track_id"))),
        frame_id=_as_int(_first(row, ("frame_id", "frame", "frame_number", "image_id"))),
        timestamp=_as_float(_first(row, ("timestamp", "time", "time_sec"))),
        x1=float(x1),
        y1=float(y1),
        x2=float(x2),
        y2=float(y2),
        visible=_as_bool(_first(row, ("visible", "is_visible", "visibility"), True), default=True),
        metadata=dict(row),
    )
