import numpy as np
import torch

from easycom_saal.models.spatial_speaker import SpatialSpeakerMatcher
from scripts.run_gcc_bbox_speaker_pilot_v1 import bbox_motion_features, signed_gcc_phat


def test_runtime_signed_gcc_matches_cached_feature_extractor() -> None:
    generator = np.random.default_rng(7)
    waveform = generator.normal(size=(6, 16_000)).astype(np.float32)
    waveform[1] = np.roll(waveform[0], 5)
    expected = signed_gcc_phat(waveform, 16_000)
    model = SpatialSpeakerMatcher(audio_dim=expected.size, bbox_dim=12)

    actual = model.signed_gcc_phat(torch.from_numpy(waveform).unsqueeze(0))[0]

    assert np.allclose(actual.numpy(), expected, atol=2e-5)


def test_runtime_bbox_features_match_cached_feature_extractor() -> None:
    sequence = torch.tensor(
        [
            [[2.0, 4.0, 8.0, 12.0], [3.0, 5.0, 9.0, 13.0]],
            [[20.0, 10.0, 28.0, 24.0], [18.0, 8.0, 26.0, 22.0]],
        ]
    )
    expected = bbox_motion_features(sequence, (56, 56))

    actual, mask = SpatialSpeakerMatcher.bbox_motion_features([sequence], (56, 56))

    assert np.allclose(actual[0].numpy(), expected, atol=1e-7)
    assert mask.tolist() == [[True, True]]
