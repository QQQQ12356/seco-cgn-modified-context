import json
from types import SimpleNamespace

import pytest
import torch

from context_compression import ContextCompressor, mass_segments, semantic_blocks
from compressor_config import check_route
from scripts.prepare_data import normalize, partition


@pytest.mark.parametrize("length,budget", [(0, 0), (1, 1), (31, 0), (31, 1), (31, 7), (31, 31), (31, 40), (129, 13)])
@pytest.mark.parametrize("query_length", [0, 7])
def test_budget_coverage_gradient(length, budget, query_length):
    torch.manual_seed(42)
    compressor = ContextCompressor(32, relevance_dim=16, num_slots=4, block_width=16)
    hidden = torch.randn(length, 32, requires_grad=True)
    query = torch.randn(query_length, 32)
    memory, trace = compressor(hidden, query, budget, return_trace=True)
    assert memory.shape == (min(length, budget), 32)
    assert torch.isfinite(memory).all()
    if 0 < budget < length:
        segments = trace["segment_ranges"]
        assert len(segments) == budget
        assert [position for start, end in segments for position in range(start, end)] == list(range(length))
        assert sum(trace["block_allocation"]) == budget
        for weight in trace["merge_weights"].values():
            assert torch.allclose(weight.sum(), torch.tensor(1.0))
            assert (weight > 0).all()
        memory.square().mean().backward()
        assert (hidden.grad.abs().sum(-1) > 0).all()
        assert compressor.scorer.mlp[-1].weight.grad.abs().sum() > 0
        assert compressor.merger.raw_temperature.grad.abs() > 0


def test_boundaries_seek_local_semantic_change():
    unit = torch.zeros(20, 2)
    unit[:12, 0] = 1
    unit[12:, 1] = 1
    assert semantic_blocks(unit, 2) == [(0, 12), (12, 20)]
    assert semantic_blocks(unit, 2, mode="uniform") == [(0, 10), (10, 20)]


def test_mass_segments_remain_nonempty_under_extreme_mass():
    segments = mass_segments(torch.tensor([1e10, 1., 1., 1., 1.]), 4)
    assert all(end > start for start, end in segments)
    assert segments[0][0] == 0 and segments[-1][1] == 5


@pytest.mark.parametrize("mode", ["hybrid", "mean", "anchor"])
def test_ablation_modes_and_bfloat16(mode):
    compressor = ContextCompressor(32, relevance_dim=16, num_slots=4, merge_mode=mode)
    hidden = torch.randn(21, 32).bfloat16()
    memory, _ = compressor(hidden, None, 4)
    assert memory.dtype == hidden.dtype and torch.isfinite(memory).all()


def test_context_checkpoint_requires_metadata(tmp_path):
    args = SimpleNamespace(compressor_version="context_adaptive_v1")
    with pytest.raises(RuntimeError, match="metadata"):
        check_route(str(tmp_path / "model.safetensors"), args)
    (tmp_path / "compressor_config.json").write_text(json.dumps({"compressor_version": "legacy"}))
    with pytest.raises(RuntimeError):
        check_route(str(tmp_path / "model.safetensors"), args)


def test_context_route_roundtrip_and_mismatch(tmp_path):
    from compressor_config import write_config
    from modeling_seco_cluster import ModelArguments

    args = ModelArguments()
    write_config(str(tmp_path), args)
    path = str(tmp_path / "model.safetensors")
    assert check_route(path, args)["compressor_version"] == "context_adaptive_v1"
    args.context_anchor_weight = 0.6
    with pytest.raises(RuntimeError, match="context_anchor_weight"):
        check_route(path, args)


def test_preprocessing_schema_and_group_split():
    row = normalize({"context": "same context", "question": "who?", "answers": {"text": ["me"]}})
    assert row["answer"] == ["me"]
    duplicate = dict(row, prompt="when?", input="same   context")
    assert partition(row, 42, 0.1) == partition(duplicate, 42, 0.1)
    with pytest.raises(ValueError):
        normalize(dict(row, answer=[]))


def test_context_model_forward(tiny_model_dir):
    from test_model_integration import build_model, make_batch

    model = build_model(tiny_model_dir, compressor_version="context_adaptive_v1")
    outputs = model(*make_batch(25, 5))
    assert torch.isfinite(outputs.loss)
    outputs.loss.backward()
    assert model.compressor.merger.raw_temperature.grad is not None
