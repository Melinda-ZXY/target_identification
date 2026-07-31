import torch

from easycom_saal.eval_ablations import binary_auroc, binary_average_precision
from easycom_saal.eval_response import summarize
from easycom_saal.models import SAALModel
from easycom_saal.models.saal_model import CandidateHead
from easycom_saal.train import (
    candidate_finetune_parameter_groups,
    candidate_target_bce_loss,
    candidate_target_metrics,
    candidate_speaker_ce_loss,
    conditional_addressing_bce_loss,
    freeze_for_addressing_hard_negative_finetune,
    relation_adaptation_parameter_groups,
    freeze_base_model_for_candidate_classifier,
    global_response_metrics,
    global_address_parameter_groups,
    response_action_ce_loss,
    response_action_metrics,
    speaker_residual_parameter_groups,
    speaker_representation_parameter_groups,
    session_class_balanced_bce_loss,
    session_category_balanced_response_bce_loss,
    session_category_balanced_scene_ce_loss,
    scene_response_parameter_groups,
    shared_response_parameter_groups,
)


def test_speaker_ce_trains_categorical_selection_on_single_speaker_windows() -> None:
    logits = [
        torch.tensor([3.0, -2.0], requires_grad=True),
        torch.tensor([-1.0, 4.0], requires_grad=True),
        torch.tensor([0.0, 0.0], requires_grad=True),
    ]
    labels = [
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.0, 1.0]),
        torch.tensor([0.0, 0.0]),
    ]

    loss = candidate_speaker_ce_loss(logits, labels)
    loss.backward()

    assert loss.item() < 0.02
    assert logits[0].grad is not None
    assert logits[1].grad is not None
    assert logits[2].grad is None


def test_speaker_ce_weights_crowded_candidate_sets_more_heavily() -> None:
    logits = [torch.tensor([4.0, -4.0]), torch.tensor([0.0, 0.0, 0.0, 0.0])]
    labels = [torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0, 0.0, 0.0])]

    unweighted = candidate_speaker_ce_loss(logits, labels)
    weighted = candidate_speaker_ce_loss(logits, labels, weight_by_candidate_count=True)

    assert weighted.item() > unweighted.item()


def test_speaker_representation_finetune_preserves_response_memory_and_vad() -> None:
    model = SAALModel(frame_aligned_audio=True)

    groups = speaker_representation_parameter_groups(
        model,
        speaker_learning_rate=1e-4,
        relation_learning_rate=3e-5,
        fusion_learning_rate=1e-5,
        audio_encoder_learning_rate=1e-5,
    )

    assert [group["group_name"] for group in groups] == [
        "speaker_classifier",
        "speaker_relation_representation",
        "speaker_fusion",
        "speaker_audio_encoder",
    ]
    relation = model.candidate_relation_head
    assert relation.speaking_classifier.weight.requires_grad
    assert next(relation.relation.parameters()).requires_grad
    assert next(model.fusion.parameters()).requires_grad
    assert next(model.audio_encoder.parameters()).requires_grad
    assert not relation.addressing_classifier.weight.requires_grad
    assert not next(relation.vad_classifier.parameters()).requires_grad
    assert not relation.candidate_memory.weight_hh.requires_grad
    assert not next(model.video_encoder.parameters()).requires_grad


def test_speaker_residual_is_zero_initialized_and_only_trainable_component() -> None:
    model = SAALModel(frame_aligned_audio=True)
    relation = model.candidate_relation_head
    tokens = torch.randn(5, 128)

    residual = relation.speaker_residual_adapter(tokens).squeeze(-1)
    groups = speaker_residual_parameter_groups(model, learning_rate=1e-3)

    assert torch.equal(residual, torch.zeros_like(residual))
    assert [group["group_name"] for group in groups] == ["speaker_residual_adapter"]
    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable_names
    assert all(
        name.startswith("candidate_relation_head.speaker_residual_adapter.")
        for name in trainable_names
    )
    assert not relation.speaking_classifier.weight.requires_grad
    assert not relation.addressing_classifier.weight.requires_grad
    assert not relation.candidate_memory.weight_hh.requires_grad
    assert not next(relation.vad_classifier.parameters()).requires_grad


def test_global_address_finetune_is_isolated_from_speaker_vad_and_memory() -> None:
    model = SAALModel(frame_aligned_audio=True)
    relation = model.candidate_relation_head

    groups = global_address_parameter_groups(model, learning_rate=1e-4)

    assert [group["group_name"] for group in groups] == ["global_address_head"]
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable
    assert all(
        name.startswith("candidate_relation_head.global_address_classifier.")
        for name in trainable
    )
    assert not relation.speaking_classifier.weight.requires_grad
    assert not relation.addressing_classifier.weight.requires_grad
    assert not relation.candidate_memory.weight_hh.requires_grad
    assert not next(relation.vad_classifier.parameters()).requires_grad


def test_global_address_adapter_starts_as_identity_and_stays_isolated() -> None:
    model = SAALModel(frame_aligned_audio=True)
    relation = model.candidate_relation_head
    features = torch.randn(4, 128)

    residual = relation.global_address_adapter(features)
    groups = global_address_parameter_groups(model, learning_rate=3e-5, train_adapter=True)

    assert torch.equal(residual, torch.zeros_like(residual))
    assert [group["group_name"] for group in groups] == ["global_address_head_and_adapter"]
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert all(
        name.startswith("candidate_relation_head.global_address_classifier.")
        or name.startswith("candidate_relation_head.global_address_adapter.")
        for name in trainable
    )


def test_session_category_balanced_losses_backpropagate() -> None:
    binary_logits = torch.tensor([2.0, -1.0, -2.0], requires_grad=True)
    scene_logits = torch.randn(3, 3, requires_grad=True)
    sessions = ['Session_1/a', 'Session_1/b', 'Session_2/a']
    categories = ['addressed_speech', 'non_target_speech', 'silence']
    binary = session_category_balanced_response_bce_loss(
        binary_logits, torch.tensor([1.0, 0.0, 0.0]), sessions, categories,
    )
    scene = session_category_balanced_scene_ce_loss(scene_logits, sessions, categories)

    (binary + scene).backward()
    assert binary_logits.grad is not None
    assert scene_logits.grad is not None


def test_session_class_balanced_response_loss_backpropagates() -> None:
    logits = torch.tensor([2.0, -2.0, -1.0, 1.0], requires_grad=True)
    labels = torch.tensor([True, False, True, False])
    loss = session_class_balanced_bce_loss(
        logits, labels, ['Session_1/a', 'Session_1/b', 'Session_2/a', 'Session_2/b'],
    )

    loss.backward()
    assert loss.item() > 0.0
    assert logits.grad is not None


def test_scene_response_finetune_is_isolated_and_outputs_three_classes() -> None:
    model = SAALModel(frame_aligned_audio=True)
    groups = scene_response_parameter_groups(model, learning_rate=3e-5)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    relation = model.candidate_relation_head

    assert [group['group_name'] for group in groups] == ['scene_response_head']
    assert all(
        name.startswith('candidate_relation_head.scene_response_classifier.')
        or name.startswith('candidate_relation_head.global_address_adapter.')
        for name in trainable
    )
    assert relation.scene_response_classifier(torch.randn(2, 128)).shape == (2, 3)
    assert not relation.global_address_classifier[-1].weight.requires_grad


def test_shared_response_gate_is_the_only_trainable_finetune_component() -> None:
    model = SAALModel(frame_aligned_audio=True)
    groups = shared_response_parameter_groups(model, learning_rate=1e-3)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}

    assert [group['group_name'] for group in groups] == ['shared_response_gate']
    assert trainable
    assert all(name.startswith('shared_response_gate.') for name in trainable)
    logits = model.shared_response_logits(
        torch.tensor([2.0, -1.0]),
        torch.tensor([1.0, 0.5]),
        torch.tensor([2.0, 4.0]),
        torch.tensor([-42.0, -48.0]),
    )
    assert logits.shape == (2,)


def test_response_loss_and_metrics_handle_candidate_and_no_response_targets() -> None:
    logits = [
        torch.tensor([4.0, 0.0, -1.0]),
        torch.tensor([-1.0, 0.0, 4.0]),
    ]
    labels = [
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.0, 0.0]),
    ]
    should_respond = torch.tensor([True, False])

    loss = response_action_ce_loss(logits, labels, should_respond)
    metrics = response_action_metrics(logits, labels, should_respond)

    assert loss.item() < 0.05
    assert metrics == {
        "top1": 1.0,
        "addressed_correct": 1.0,
        "non_target_rejection": 1.0,
    }

    weighted_loss = response_action_ce_loss(logits, labels, should_respond, addressed_weight=4.0)
    assert weighted_loss.item() > loss.item()


def test_response_loss_and_metrics_accept_multiple_valid_attention_targets() -> None:
    logits = [torch.tensor([3.0, 2.0, -2.0, -3.0])]
    labels = [torch.tensor([1.0, 1.0, 0.0])]
    should_respond = torch.tensor([True])

    loss = response_action_ce_loss(logits, labels, should_respond)
    metrics = response_action_metrics(logits, labels, should_respond)

    assert loss.item() < 0.05
    assert metrics["addressed_correct"] == 1.0


def test_model_appends_one_no_response_logit_per_sample() -> None:
    model = SAALModel()
    temporal_features = torch.randn((2, 4, 128, 8, 8))
    candidate_sequences = [
        torch.tensor([[[1, 1, 4, 4]] * 4, [[3, 2, 7, 6]] * 4], dtype=torch.float32),
        torch.tensor([[[2, 2, 6, 6]] * 4], dtype=torch.float32),
    ]

    logits = model.temporal_response_logits(temporal_features, candidate_sequences)

    assert [tuple(values.shape) for values in logits] == [(3,), (2,)]


def test_relation_head_emits_speaking_and_addressing_logits_per_candidate() -> None:
    model = SAALModel(frame_aligned_audio=True)
    temporal_features = torch.randn((2, 4, 128, 8, 8))
    candidate_sequences = [
        torch.tensor([[[1, 1, 4, 4]] * 4, [[3, 2, 7, 6]] * 4], dtype=torch.float32),
        torch.tensor([[[2, 2, 6, 6]] * 4], dtype=torch.float32),
    ]

    speaking, addressing = model.temporal_relation_logits(temporal_features, candidate_sequences)

    assert [tuple(values.shape) for values in speaking] == [(2,), (1,)]
    assert [tuple(values.shape) for values in addressing] == [(2,), (1,)]


def test_audio_encoder_can_preserve_temporal_steps() -> None:
    model = SAALModel(frame_aligned_audio=True)
    spectrogram = torch.randn((2, 6, 257, 101))
    correlation = torch.randn((2, 15, 257, 101))

    features = model.audio_encoder.forward_temporal(spectrogram, correlation, output_steps=20)

    assert features.shape == (2, 20, 128)


def test_candidate_target_sigmoid_loss_and_threshold_metrics() -> None:
    logits = [torch.tensor([-4.0, 4.0]), torch.tensor([-4.0, -3.0])]
    labels = [torch.tensor([0.0, 1.0]), torch.tensor([0.0, 0.0])]
    should_respond = torch.tensor([True, False])

    loss = candidate_target_bce_loss(logits, labels)
    metrics = candidate_target_metrics(logits, labels, should_respond, threshold=0.5)

    assert loss.item() < 0.05
    assert metrics == {
        "top1": 1.0,
        "addressed_detected": 1.0,
        "addressed_correct": 1.0,
        "non_target_rejection": 1.0,
    }


def test_candidate_target_metrics_accept_multiple_valid_attention_targets() -> None:
    metrics = candidate_target_metrics(
        [torch.tensor([2.0, 3.0, -3.0])],
        [torch.tensor([1.0, 1.0, 0.0])],
        torch.tensor([True]),
        threshold=0.5,
    )

    assert metrics["addressed_detected"] == 1.0
    assert metrics["addressed_correct"] == 1.0


def test_candidate_target_loss_balances_one_positive_against_multiple_negatives() -> None:
    one_negative = candidate_target_bce_loss(
        [torch.tensor([0.0, 0.0])],
        [torch.tensor([1.0, 0.0])],
    )
    three_negatives = candidate_target_bce_loss(
        [torch.tensor([0.0, 0.0, 0.0, 0.0])],
        [torch.tensor([1.0, 0.0, 0.0, 0.0])],
    )

    assert torch.allclose(one_negative, three_negatives)


def test_conditional_addressing_loss_ignores_silent_candidates() -> None:
    labels = [torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 0.0])]
    speakers = [torch.tensor([1.0, 0.0, 0.0]), torch.tensor([1.0, 0.0])]
    baseline = conditional_addressing_bce_loss(
        [torch.tensor([0.0, 0.0, 0.0]), torch.tensor([0.0, 0.0])],
        labels,
        speakers,
    )
    changed_silent_candidates = conditional_addressing_bce_loss(
        [torch.tensor([0.0, 100.0, -100.0]), torch.tensor([0.0, 100.0])],
        labels,
        speakers,
    )

    assert torch.allclose(baseline, torch.tensor(0.6931472), atol=1e-6)
    assert torch.allclose(baseline, changed_silent_candidates)


def test_conditional_addressing_hard_negatives_receive_more_gradient() -> None:
    baseline_logits = torch.tensor([0.2, 1.5, -1.5], requires_grad=True)
    hard_logits = baseline_logits.detach().clone().requires_grad_(True)
    response_labels = [torch.tensor([1.0, 0.0, 0.0])]
    speaker_labels = [torch.ones(3)]

    baseline_loss = conditional_addressing_bce_loss(
        [baseline_logits], response_labels, speaker_labels
    )
    baseline_loss.backward()
    hard_loss = conditional_addressing_bce_loss(
        [hard_logits],
        response_labels,
        speaker_labels,
        negative_weight=1.5,
        hard_negative_fraction=0.5,
        hard_negative_weight=2.0,
    )
    hard_loss.backward()

    assert hard_logits.grad is not None
    assert baseline_logits.grad is not None
    assert hard_logits.grad[1] > baseline_logits.grad[1]
    assert hard_logits.grad[1] > hard_logits.grad[2]


def test_addressing_hard_negative_finetune_only_unfreezes_address_classifier() -> None:
    model = SAALModel()
    freeze_for_addressing_hard_negative_finetune(model)

    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable
    assert all(
        name.startswith("candidate_relation_head.addressing_classifier.")
        for name in trainable
    )


def test_relation_adaptation_uses_disjoint_discriminative_parameter_groups() -> None:
    model = SAALModel()
    groups = relation_adaptation_parameter_groups(
        model,
        relation_learning_rate=1e-4,
        auxiliary_learning_rate=3e-5,
        fusion_learning_rate=1e-5,
        video_last_block_learning_rate=3e-6,
        audio_encoder_learning_rate=3e-6,
    )

    assert {group["group_name"]: group["lr"] for group in groups} == {
        "relation_memory_addressing": 1e-4,
        "speaker_vad_heads": 3e-5,
        "fusion": 1e-5,
        "video_last_block": 3e-6,
        "audio_encoder": 3e-6,
    }
    assigned = [id(parameter) for group in groups for parameter in group["params"]]
    assert len(assigned) == len(set(assigned))
    assert all(parameter.requires_grad for parameter in model.candidate_relation_head.relation.parameters())
    assert all(parameter.requires_grad for parameter in model.candidate_relation_head.candidate_memory.parameters())
    assert all(parameter.requires_grad for parameter in model.candidate_relation_head.addressing_classifier.parameters())
    assert all(parameter.requires_grad for parameter in model.candidate_relation_head.speaking_classifier.parameters())
    assert all(parameter.requires_grad for parameter in model.candidate_relation_head.vad_classifier.parameters())
    assert all(parameter.requires_grad for parameter in model.fusion.parameters())
    assert all(parameter.requires_grad for parameter in model.audio_encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.video_encoder.net[6].parameters())
    assert not any(parameter.requires_grad for parameter in model.video_encoder.net[0].parameters())
    assert not any(
        parameter.requires_grad
        for parameter in model.candidate_relation_head.global_address_classifier.parameters()
    )


def test_relation_joint_probability_multiplies_speaking_and_addressing() -> None:
    speaking = [torch.tensor([0.0, 2.0])]
    addressing = [torch.tensor([2.0, 0.0])]

    joint_logits = SAALModel.combine_relation_logits(speaking, addressing)

    expected = torch.sigmoid(speaking[0]) * torch.sigmoid(addressing[0])
    assert torch.allclose(torch.sigmoid(joint_logits[0]), expected)


def test_global_address_gate_updates_causal_memory_and_combines_with_speakers() -> None:
    model = SAALModel()
    temporal_features = torch.randn((2, 4, 128, 8, 8))
    candidate_sequences = [
        torch.tensor([[[1, 1, 4, 4]] * 4, [[3, 2, 7, 6]] * 4], dtype=torch.float32),
        torch.tensor([[[2, 2, 6, 6]] * 4], dtype=torch.float32),
    ]

    speaking, global_addressing, memory = model.temporal_global_relation_logits(
        temporal_features,
        candidate_sequences,
    )
    joint = model.combine_global_address_logits(speaking, global_addressing)

    assert global_addressing.shape == (2,)
    assert memory.shape == (2, 128)
    assert [tuple(values.shape) for values in joint] == [(2,), (1,)]
    assert torch.allclose(
        torch.sigmoid(joint[0]),
        torch.sigmoid(speaking[0]) * torch.sigmoid(global_addressing[0]),
    )

    _, _, next_memory = model.temporal_global_relation_logits(
        temporal_features,
        candidate_sequences,
        previous_memory=memory,
    )
    assert not torch.allclose(memory, next_memory)


def test_candidate_address_memory_keeps_candidate_specific_scores() -> None:
    model = SAALModel()
    temporal_features = torch.randn((1, 4, 128, 8, 8))
    candidate_sequences = [
        torch.tensor([[[1, 1, 4, 4]] * 4, [[3, 2, 7, 6]] * 4], dtype=torch.float32),
    ]

    speaking, addressing, memory = model.temporal_memory_relation_logits(
        temporal_features,
        candidate_sequences,
    )
    _, next_addressing, next_memory = model.temporal_memory_relation_logits(
        temporal_features,
        candidate_sequences,
        previous_memory=memory,
    )

    assert speaking[0].shape == (2,)
    assert addressing[0].shape == (2,)
    assert memory.shape == (1, 128)
    assert not torch.allclose(memory, next_memory)
    assert not torch.allclose(addressing[0], next_addressing[0])


def test_candidate_identity_memory_tracks_participant_ids_and_reports_vad() -> None:
    model = SAALModel()
    temporal_features = torch.randn((1, 4, 128, 8, 8))
    candidate_sequences = [
        torch.tensor([[[1, 1, 4, 4]] * 4, [[3, 2, 7, 6]] * 4], dtype=torch.float32),
    ]

    speaking, addressing, vad, states = model.temporal_identity_memory_relation_logits(
        temporal_features,
        candidate_sequences,
        [(2, 3)],
    )
    _, next_addressing, next_vad, next_states = model.temporal_identity_memory_relation_logits(
        temporal_features,
        candidate_sequences,
        [(2, 3)],
        previous_states=states,
    )

    assert speaking[0].shape == (2,)
    assert addressing[0].shape == (2,)
    assert vad.shape == (1,)
    assert next_vad.shape == (1,)
    assert set(states[0]) == {2, 3}
    assert set(next_states[0]) == {2, 3}
    assert not torch.allclose(addressing[0], next_addressing[0])


def test_candidate_identity_memory_can_drive_a_separate_global_response_gate() -> None:
    model = SAALModel()
    temporal_features = torch.randn((1, 4, 128, 8, 8))
    candidate_sequences = [
        torch.tensor([[[1, 1, 4, 4]] * 4, [[3, 2, 7, 6]] * 4], dtype=torch.float32),
    ]

    speaking, global_response, vad, states = model.temporal_identity_global_relation_logits(
        temporal_features,
        candidate_sequences,
        [(2, 3)],
    )

    assert speaking[0].shape == (2,)
    assert global_response.shape == (1,)
    assert vad.shape == (1,)
    assert set(states[0]) == {2, 3}


def test_candidate_identity_memory_can_be_conditioned_by_turn_graph_features() -> None:
    model = SAALModel()
    temporal_features = torch.randn((1, 4, 128, 8, 8))
    candidate_sequences = [
        torch.tensor([[[1, 1, 4, 4]] * 4, [[3, 2, 7, 6]] * 4], dtype=torch.float32),
    ]
    graph_features = [
        torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0, 0.2, 0.1, 0.5], [0.0] * 8])
    ]

    speaking, addressing, vad, self_speaking, self_targets, states = (
        model.temporal_identity_turn_graph_relation_logits(
            temporal_features,
            candidate_sequences,
            [(2, 3)],
            graph_features,
        )
    )

    assert speaking[0].shape == addressing[0].shape == self_targets[0].shape == (2,)
    assert vad.shape == self_speaking.shape == (1,)
    assert set(states[0]) == {2, 3}


def test_other_turn_graph_residual_starts_behaviorally_neutral() -> None:
    model = SAALModel().eval()
    temporal_features = torch.randn((1, 4, 128, 8, 8))
    candidate_sequences = [
        torch.tensor([[[1, 1, 4, 4]] * 4, [[3, 2, 7, 6]] * 4], dtype=torch.float32),
    ]
    without_other = [torch.zeros((2, 8))]
    with_other = [torch.zeros((2, 8))]
    with_other[0][:, 5:] = torch.tensor([[0.9, 0.1, 0.8], [0.1, 0.9, 0.2]])

    with torch.inference_mode():
        first = model.temporal_identity_turn_graph_relation_logits(
            temporal_features,
            candidate_sequences,
            [(2, 3)],
            without_other,
        )[1]
        second = model.temporal_identity_turn_graph_relation_logits(
            temporal_features,
            candidate_sequences,
            [(2, 3)],
            with_other,
        )[1]

    assert torch.equal(first[0], second[0])


def test_global_response_metrics_threshold_gate_and_candidate_rank_separately() -> None:
    metrics = global_response_metrics(
        torch.tensor([2.0, -2.0]),
        [torch.tensor([-4.0, -5.0]), torch.tensor([5.0, 4.0])],
        [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 0.0])],
        torch.tensor([True, False]),
        threshold=0.5,
    )

    assert metrics == {
        "top1": 1.0,
        "addressed_detected": 1.0,
        "addressed_correct": 1.0,
        "non_target_rejection": 1.0,
    }


def test_vad_probability_gates_candidate_probabilities() -> None:
    candidate_logits = [torch.tensor([0.0, 2.0])]
    vad_logits = torch.tensor([0.0])

    combined = SAALModel.combine_vad_logits(candidate_logits, vad_logits)

    expected = torch.sigmoid(candidate_logits[0]) * 0.5
    assert torch.allclose(torch.sigmoid(combined[0]), expected)


def test_partial_feature_finetuning_uses_discriminative_learning_rates() -> None:
    model = SAALModel()
    freeze_base_model_for_candidate_classifier(
        model,
        classifier_type="temporal",
        unfreeze_fusion=True,
        unfreeze_video_last_block=True,
    )
    groups = candidate_finetune_parameter_groups(
        model,
        classifier_type="temporal",
        head_learning_rate=1e-4,
        fusion_learning_rate=3e-5,
        video_last_block_learning_rate=1e-5,
        train_fusion=True,
        train_video_last_block=True,
    )

    assert {group["group_name"]: group["lr"] for group in groups} == {
        "candidate_head": 1e-4,
        "fusion": 3e-5,
        "video_last_block": 1e-5,
    }
    assert all(parameter.requires_grad for parameter in model.temporal_candidate_head.parameters())
    assert all(parameter.requires_grad for parameter in model.fusion.parameters())
    assert all(parameter.requires_grad for parameter in model.video_encoder.net[6].parameters())
    assert not any(parameter.requires_grad for parameter in model.video_encoder.net[0].parameters())
    assert not any(parameter.requires_grad for parameter in model.audio_encoder.parameters())
    assigned = [id(parameter) for group in groups for parameter in group["params"]]
    assert len(assigned) == len(set(assigned))


def test_geometry_ablation_preserves_pooled_features_and_zeros_coordinates() -> None:
    feature_map = torch.arange(2 * 4 * 4, dtype=torch.float32).reshape(2, 4, 4)
    bbox = torch.tensor([1.0, 1.0, 3.0, 3.0])

    normal = CandidateHead._pool_box(feature_map, bbox, include_geometry=True)
    ablated = CandidateHead._pool_box(feature_map, bbox, include_geometry=False)

    assert torch.equal(normal[:-4], ablated[:-4])
    assert torch.count_nonzero(normal[-4:]).item() == 4
    assert torch.count_nonzero(ablated[-4:]).item() == 0


def test_ablation_ranking_metrics() -> None:
    scores = [0.9, 0.8, 0.2, 0.1]
    labels = [True, False, True, False]

    assert binary_auroc(scores, labels) == 0.75
    assert binary_average_precision(scores, labels) == (1.0 + 2.0 / 3.0) / 2.0


def test_independent_candidate_head_returns_both_tasks() -> None:
    model = SAALModel()
    features = torch.randn(2, 3, 128, 8, 8)
    boxes = [
        torch.tensor(
            [
                [[1.0, 1.0, 4.0, 4.0]] * 3,
                [[4.0, 2.0, 7.0, 6.0]] * 3,
            ]
        ).squeeze(2),
        torch.tensor([[[2.0, 1.0, 6.0, 5.0]] * 3]).squeeze(2),
    ]

    speaking, addressing = model.temporal_independent_candidate_logits(features, boxes)

    assert [tuple(logits.shape) for logits in speaking] == [(2,), (1,)]
    assert [tuple(logits.shape) for logits in addressing] == [(2,), (1,)]


def test_independent_candidate_finetuning_only_unfreezes_baseline_modules() -> None:
    model = SAALModel()
    freeze_base_model_for_candidate_classifier(
        model,
        "temporal",
        unfreeze_fusion=True,
        unfreeze_video_last_block=True,
        use_independent_candidate_head=True,
    )
    groups = candidate_finetune_parameter_groups(
        model,
        classifier_type="temporal",
        head_learning_rate=1e-4,
        fusion_learning_rate=3e-5,
        video_last_block_learning_rate=1e-5,
        train_fusion=True,
        train_video_last_block=True,
        use_independent_candidate_head=True,
    )

    assert {group["group_name"]: group["lr"] for group in groups} == {
        "candidate_head": 1e-4,
        "fusion": 3e-5,
        "video_last_block": 1e-5,
    }
    assert all(parameter.requires_grad for parameter in model.independent_candidate_head.parameters())
    assert not any(parameter.requires_grad for parameter in model.candidate_relation_head.parameters())


def test_response_summary_includes_threshold_independent_separation() -> None:
    rows = [
        {
            "should_respond": True,
            "predicted_should_respond": True,
            "correct_action": True,
            "sample_type": "addressed_speech",
            "predicted_probability": 0.9,
        },
        {
            "should_respond": True,
            "predicted_should_respond": False,
            "correct_action": False,
            "sample_type": "addressed_speech",
            "predicted_probability": 0.2,
        },
        {
            "should_respond": False,
            "predicted_should_respond": True,
            "correct_action": False,
            "sample_type": "non_target_speech",
            "predicted_probability": 0.8,
        },
        {
            "should_respond": False,
            "predicted_should_respond": False,
            "correct_action": True,
            "sample_type": "silence",
            "predicted_probability": 0.1,
        },
    ]

    metrics = summarize(rows)

    assert metrics["target_detection_auroc"] == 0.75
    assert metrics["target_detection_average_precision"] == (1.0 + 2.0 / 3.0) / 2.0
    assert metrics["speech_only_target_detection_auroc"] == 0.5
