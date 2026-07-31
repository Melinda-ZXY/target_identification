from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterator, Sequence
from typing import Any

import torch
from torch.utils.data import Sampler, Subset


def _sample_metadata(dataset: Any, local_index: int) -> tuple[str, float]:
    if isinstance(dataset, Subset):
        source_index = int(dataset.indices[local_index])
        sample = dataset.dataset.samples[source_index]
    else:
        sample = dataset.samples[local_index]
    return str(sample.session_id), float(sample.timestamp)


class SessionStreamBatchSampler(Sampler[list[int]]):
    """Round-robin chronological session streams with one window per session in a batch."""

    def __init__(self, dataset: Any, batch_size: int, start_batch: int = 0) -> None:
        if batch_size < 1:
            raise ValueError("Stateful stream batch size must be at least 1.")
        grouped: dict[str, list[tuple[float, int]]] = defaultdict(list)
        for local_index in range(len(dataset)):
            session_id, timestamp = _sample_metadata(dataset, local_index)
            grouped[session_id].append((timestamp, local_index))
        streams = {
            session_id: deque(index for _, index in sorted(entries))
            for session_id, entries in sorted(grouped.items())
        }
        active = deque((session_id, indices) for session_id, indices in streams.items() if indices)
        batches: list[list[int]] = []
        while active:
            batch: list[int] = []
            for _ in range(min(int(batch_size), len(active))):
                session_id, indices = active.popleft()
                batch.append(indices.popleft())
                if indices:
                    active.append((session_id, indices))
            batches.append(batch)
        self.batches = batches
        self.start_batch = int(start_batch)
        self._validate_start_batch()

    @property
    def total_batches(self) -> int:
        return len(self.batches)

    def _validate_start_batch(self) -> None:
        if not 0 <= self.start_batch <= self.total_batches:
            raise ValueError(
                f"Stateful start batch {self.start_batch} is outside [0, {self.total_batches}]."
            )

    def __iter__(self) -> Iterator[list[int]]:
        self._validate_start_batch()
        yield from self.batches[self.start_batch :]

    def __len__(self) -> int:
        self._validate_start_batch()
        return self.total_batches - self.start_batch


class IdentityStateCache:
    """Detached participant states carried independently for each chronological session stream."""

    def __init__(self, max_gap_seconds: float) -> None:
        self.max_gap_seconds = float(max_gap_seconds)
        self.states: dict[str, dict[object, torch.Tensor]] = {}
        self.timestamps: dict[str, float] = {}

    def clear(self) -> None:
        self.states.clear()
        self.timestamps.clear()

    @staticmethod
    def _timestamps(values: torch.Tensor | Sequence[float]) -> list[float]:
        if torch.is_tensor(values):
            return [float(value) for value in values.detach().cpu().flatten().tolist()]
        return [float(value) for value in values]

    def previous_states(
        self,
        session_ids: Sequence[str],
        timestamps: torch.Tensor | Sequence[float],
    ) -> list[dict[object, torch.Tensor]]:
        if len(set(session_ids)) != len(session_ids):
            raise ValueError("A stateful batch cannot contain two windows from the same session.")
        timestamp_values = self._timestamps(timestamps)
        previous: list[dict[object, torch.Tensor]] = []
        for session_id, timestamp in zip(session_ids, timestamp_values):
            session_id = str(session_id)
            prior_timestamp = self.timestamps.get(session_id)
            gap = None if prior_timestamp is None else timestamp - prior_timestamp
            if gap is None or gap <= 0.0 or gap > self.max_gap_seconds:
                self.states.pop(session_id, None)
                self.timestamps.pop(session_id, None)
            previous.append(self.states.get(session_id, {}))
        return previous

    def update(
        self,
        session_ids: Sequence[str],
        timestamps: torch.Tensor | Sequence[float],
        updated_states: Sequence[dict[object, torch.Tensor]],
        *,
        detach: bool = True,
    ) -> None:
        timestamp_values = self._timestamps(timestamps)
        if not (len(session_ids) == len(timestamp_values) == len(updated_states)):
            raise ValueError("Session IDs, timestamps, and identity states must have equal lengths.")
        for session_id, timestamp, states in zip(session_ids, timestamp_values, updated_states):
            session_id = str(session_id)
            self.states[session_id] = {
                participant_id: state.detach() if detach else state
                for participant_id, state in states.items()
            }
            self.timestamps[session_id] = timestamp

    def detach(self) -> None:
        self.states = {
            session_id: {
                participant_id: state.detach()
                for participant_id, state in states.items()
            }
            for session_id, states in self.states.items()
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "max_gap_seconds": self.max_gap_seconds,
            "timestamps": dict(self.timestamps),
            "states": {
                session_id: {
                    participant_id: state.detach().cpu()
                    for participant_id, state in states.items()
                }
                for session_id, states in self.states.items()
            },
        }

    def load_state_dict(self, values: dict[str, Any], device: torch.device) -> None:
        self.max_gap_seconds = float(values.get("max_gap_seconds", self.max_gap_seconds))
        self.timestamps = {
            str(session_id): float(timestamp)
            for session_id, timestamp in values.get("timestamps", {}).items()
        }
        self.states = {
            str(session_id): {
                participant_id: state.detach().to(device)
                for participant_id, state in states.items()
            }
            for session_id, states in values.get("states", {}).items()
        }
