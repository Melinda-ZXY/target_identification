from __future__ import annotations


def seconds_to_frame_id(timestamp: float, fps: float) -> int:
    return int(round(float(timestamp) * float(fps)))


def frame_id_to_seconds(frame_id: int, fps: float) -> float:
    return float(frame_id) / float(fps)


def clip_window_around(center_time: float, duration_sec: float) -> tuple[float, float]:
    half = float(duration_sec) / 2.0
    return max(0.0, float(center_time) - half), float(center_time) + half


def segment_to_clip_centers(
    start_time: float,
    end_time: float,
    clip_duration_sec: float,
    clip_stride_sec: float,
) -> list[float]:
    start_time = float(start_time)
    end_time = float(end_time)
    if end_time <= start_time:
        return []

    half = float(clip_duration_sec) / 2.0
    first_center = start_time + half
    last_center = end_time - half

    if first_center > last_center:
        return [(start_time + end_time) / 2.0]

    centers: list[float] = []
    current = first_center
    while current <= last_center + 1e-9:
        centers.append(current)
        current += float(clip_stride_sec)
    return centers


def has_valid_clip_alignment(
    clip_start_time: float,
    clip_end_time: float,
    audio_duration_sec: float | None = None,
    video_duration_sec: float | None = None,
) -> bool:
    if clip_end_time <= clip_start_time:
        return False
    if audio_duration_sec is not None and clip_end_time > audio_duration_sec:
        return False
    if video_duration_sec is not None and clip_end_time > video_duration_sec:
        return False
    return True

