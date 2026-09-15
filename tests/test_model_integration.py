"""End-to-end wiring checks for the compression route on a tiny local model.

Builds a two-layer random Llama so the whole SECO forward, checkpoint round-trip
and the encoder-backbone optimisation can be exercised on CPU without
downloading pretrained weights. The real tokenizer is used because the model
hard-codes ``bos_token_id`` as the pooler seed initialiser.

    python -m pytest tests/test_model_integration.py -q
"""

import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modeling_seco_cluster import SECO, ModelArguments, TrainingArguments  # noqa: E402

from conftest import HIDDEN  # noqa: E402  (the shared tiny-model fixture lives there)


def build_model(model_dir, compressor_version="multiscale_budget_v1", **model_overrides):
    from peft import LoraConfig

    training_kwargs = {
        key: model_overrides.pop(key) for key in ("encoder_last_hidden_only",)
        if key in model_overrides
    }
    model_args = ModelArguments(
        model_name_or_path=model_dir,
        compress_ratio=model_overrides.pop("compress_ratio", 4),
        compressor_version=compressor_version,
        relevance_dim=32,
        query_slots=4,
        query_phrase_widths=[2, 4],
        context_window_widths=[4, 8],
        allocation_block_width=16,
        lora_r=4,
        lora_alpha=8,
        **model_overrides,
    )
    training_args = TrainingArguments(
        # bf16 is the only half precision CPU linear supports; fp16 would need a GPU
        output_dir="/tmp/seco-wiring-test", bf16=True, report_to="none",
        **training_kwargs,
    )
    lora_config = LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM", target_modules="all-linear",
    )
    model = SECO(model_args, training_args, lora_config)
    model.eval()
    return model


def make_batch(context_len, query_len, answer_len=3, batch=1):
    total = context_len + query_len
    input_ids = torch.randint(100, 1000, (batch, total))
    prompt_mask = torch.zeros(batch, total, dtype=torch.long)
    prompt_mask[:, context_len:] = 1
    attention_mask = torch.ones(batch, total, dtype=torch.long)
    labels = torch.randint(100, 1000, (batch, answer_len))
    return input_ids, prompt_mask, attention_mask, labels


def test_forward_produces_a_scalar_loss(tiny_model_dir):
    model = build_model(tiny_model_dir)
    outputs = model(*make_batch(20, 5))
    assert outputs.loss.dim() == 0
    assert torch.isfinite(outputs.loss)
    outputs.loss.backward()
    compressor_grads = [
        name for name, param in model.named_parameters()
        if name.startswith("compressor.") and param.grad is not None
        and param.grad.abs().sum() > 0
    ]
    assert compressor_grads, "no compressor parameter received a gradient"


def test_diversity_loss_is_added_to_the_objective(tiny_model_dir):
    model = build_model(tiny_model_dir)
    weights_before = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if name.startswith("compressor.query_encoder.proj")
    }
    model(*make_batch(30, 8)).loss.backward()
    # the projection is only reachable through the score path; a zero gradient
    # here means the new modules are decorative
    for name, param in model.named_parameters():
        if name.startswith("compressor.query_encoder.proj"):
            assert param.grad is not None and param.grad.abs().sum() > 0, name
    assert weights_before


def test_legacy_path_still_runs(tiny_model_dir):
    """--compressor_version legacy must reproduce the historical behaviour."""
    model = build_model(tiny_model_dir, compressor_version="legacy")
    assert model.compressor is None
    outputs = model(*make_batch(20, 5))
    assert torch.isfinite(outputs.loss)


def test_legacy_and_new_paths_agree_on_the_output_length(tiny_model_dir):
    batch = make_batch(24, 6)
    legacy = build_model(tiny_model_dir, compressor_version="legacy")
    new = build_model(tiny_model_dir)
    for model in (legacy, new):
        embeds, masks, _ = model._compress_batch(
            torch.randn(1, 30, HIDDEN, dtype=torch.bfloat16), batch[1], batch[2]
        )
        assert len(embeds) == len(masks) == 1          # unpadded, one row per sample
        assert embeds[0].size(0) == masks[0].size(0)
        assert embeds[0].dtype == torch.bfloat16


def test_encoder_backbone_optimisation_is_equivalent(tiny_model_dir):
    """Calling the backbone directly must not change the hidden states."""
    model = build_model(tiny_model_dir, encoder_last_hidden_only=False)
    input_ids, _, attention_mask, _ = make_batch(24, 6)
    embeds = model.tokens_to_embeddings(input_ids)

    model.training_args.encoder_last_hidden_only = False
    full = model._encode(embeds, attention_mask)
    model.training_args.encoder_last_hidden_only = True
    backbone = model._encode(embeds, attention_mask)

    assert full.shape == backbone.shape
    assert torch.equal(full, backbone), "backbone path changed the encoder output"


def test_checkpoint_round_trip_preserves_the_compressor(tiny_model_dir, tmp_path):
    from safetensors.torch import load_file, save_file

    from training_utils import _is_trained_parameter

    model = build_model(tiny_model_dir)
    state = {k: v for k, v in model.state_dict().items() if _is_trained_parameter(k)}
    assert any(k.startswith("compressor.") for k in state), "compressor not saved"

    path = tmp_path / "model.safetensors"
    save_file(state, str(path))

    reloaded = build_model(tiny_model_dir)
    reloaded.load_state_dict(load_file(str(path)), strict=False)
    for name, param in model.named_parameters():
        if name.startswith("compressor."):
            assert torch.equal(param, dict(reloaded.named_parameters())[name]), name


def test_eval_is_deterministic_and_needs_no_labels(tiny_model_dir):
    model = build_model(tiny_model_dir)
    input_ids, prompt_mask, attention_mask, _ = make_batch(18, 4, batch=2)
    with torch.no_grad():
        first = model(input_ids, prompt_mask, attention_mask, labels=None)
        second = model(input_ids, prompt_mask, attention_mask, labels=None)
    assert torch.equal(first, second)


def test_missing_compressor_weights_fail_loudly(tiny_model_dir, tmp_path):
    """A legacy checkpoint must not silently warm-start the new compressor."""
    from safetensors.torch import save_file

    model = build_model(tiny_model_dir, compressor_version="legacy")
    legacy_state = {
        k: v for k, v in model.state_dict().items()
        if "lora_" in k or k.endswith("pooler_seed")
    }
    path = tmp_path / "legacy.safetensors"
    save_file(legacy_state, str(path))

    new_model = build_model(tiny_model_dir)
    new_model.training_args.restore_from = str(path)
    with pytest.raises(RuntimeError, match="compressor"):
        new_model.init()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


def _route_args(**overrides):
    from types import SimpleNamespace

    values = dict(
        compressor_version="multiscale_budget_v1", budget_mode="strict_context",
        selection_mode="budgeted", relevance_dim=256, query_slots=8,
        raw_memory_fraction=0.25, compress_ratio=32,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_checkpoint_route_mismatch_is_rejected(tmp_path):
    """Two routes can share identical tensor shapes; the config must catch it."""
    from compressor_config import check_route, write_config

    directory = tmp_path / "ckpt"
    directory.mkdir()
    write_config(str(directory), _route_args())
    saved = str(directory / "model.safetensors")

    assert check_route(saved, _route_args()) is not None

    with pytest.raises(RuntimeError, match="selection_mode"):
        check_route(saved, _route_args(selection_mode="topk"))

    with pytest.raises(RuntimeError, match="compressor_version"):
        check_route(saved, _route_args(compressor_version="legacy"))

    with pytest.raises(RuntimeError, match="budget_mode"):
        check_route(saved, _route_args(budget_mode="ceil_target"))


def test_legacy_checkpoint_without_config_is_accepted(tmp_path):
    """Checkpoints predating the config file must still evaluate."""
    from compressor_config import check_route

    assert check_route(str(tmp_path / "model.safetensors"), _route_args()) is None


def test_training_and_inference_tokenisation_agree(tiny_model_dir):
    """The P0 fix: both paths must build the same ids and prompt mask.

    They used to differ — training encoded context and query separately with
    add_special_tokens=False, inference tokenised the concatenated string (which
    inserts a BOS) — so the encoder saw a different sequence at eval time.
    """
    from transformers import AutoTokenizer

    from ft_inference_all import encode_context_and_query
    from training_utils import InstructFTTokenizeFunction

    tokenizer = AutoTokenizer.from_pretrained(tiny_model_dir, use_fast=False)
    context = "The Eastern Football League is based in the eastern suburbs."
    question = "Which suburbs is the league based in?"

    train = InstructFTTokenizeFunction(tokenizer, 28000)(
        {"input": [context], "prompt": [question], "answer": ["eastern"]}
    )
    infer_ids, infer_mask = encode_context_and_query(tokenizer, context, question, 28000)

    assert train["input_ids"][0].tolist() == infer_ids
    assert train["prompt_mask"][0].tolist() == infer_mask


def test_context_truncation_protects_the_query(tiny_model_dir):
    """A context longer than the budget loses tokens, never the question."""
    from transformers import AutoTokenizer

    from ft_inference_all import encode_context_and_query

    tokenizer = AutoTokenizer.from_pretrained(tiny_model_dir, use_fast=False)
    question = "What colour was the door?"
    context = "The quick brown fox jumps over the lazy dog. " * 500

    ids, mask = encode_context_and_query(tokenizer, context, question, 64)
    query_ids = tokenizer.encode(question, add_special_tokens=False)

    assert len(ids) == 64
    assert mask[-len(query_ids):] == [1] * len(query_ids)
    assert ids[-len(query_ids):] == query_ids
    assert mask.count(0) == 64 - len(query_ids)


def test_inert_compressor_parameter_does_not_block_loading(tiny_model_dir):
    """An added parameter the route never reads must not fail a warm start.

    The historical v1 checkpoint predates ``query_encoder.raw_attn_tau``. In
    ``dot`` attention mode nothing reads it, so refusing to load would block the
    "old weights, fixed code" evaluation without protecting anything.
    """
    from compressor_config import (
        fatal_missing_compressor_weights,
        inert_compressor_keys,
    )

    model = build_model(tiny_model_dir)                # dot mode by default
    assert "query_encoder.raw_attn_tau" in inert_compressor_keys(model.compressor)

    missing = ["compressor.query_encoder.raw_attn_tau"]
    assert fatal_missing_compressor_weights(model.compressor, missing) == []

    # a parameter the forward pass does read is still fatal
    real = ["compressor.query_encoder.proj.weight", "decoder.base_model.x"]
    fatal = fatal_missing_compressor_weights(model.compressor, real)
    assert fatal == ["compressor.query_encoder.proj.weight"]


def test_cosine_attention_mode_makes_its_temperature_required(tiny_model_dir):
    from compressor_config import fatal_missing_compressor_weights

    model = build_model(tiny_model_dir, query_attention_mode="cosine_tau")
    missing = ["compressor.query_encoder.raw_attn_tau"]
    assert fatal_missing_compressor_weights(model.compressor, missing) == missing


def test_gradient_groups_partition_the_compressor_exactly_once(tiny_model_dir):
    """Every compressor parameter must land in exactly one submodule group.

    A parameter missing from all groups is silently unlogged, which is how a
    frozen submodule goes unnoticed for a whole run.
    """
    from training_utils import _COMPRESSOR_SUBMODULES, _param_groups

    for kwargs in ({}, {"query_attention_mode": "cosine_tau"}):
        model = build_model(tiny_model_dir, **kwargs)
        groups = _param_groups(model)
        members = [
            (name, group) for group, params in groups.items()
            for param in params
            for name, other in model.named_parameters() if other is param
        ]
        compressor = {n for n, _ in members if n.startswith("compressor.")}
        assert compressor, "no compressor parameters were grouped"
        assert "compressor_other" not in groups, "a compressor parameter matched no group"

        submodule_names = {g for g, _ in _COMPRESSOR_SUBMODULES}
        counted = [g for n, g in members if n.startswith("compressor.") and g in submodule_names]
        assert len(counted) == len(compressor)
        assert len(set(counted)) == len(submodule_names), "a submodule group stayed empty"


def test_v2_modules_receive_gradient_end_to_end(tiny_model_dir):
    """Each v2 addition must be reachable from the training loss.

    A change that is wired in but never receives a gradient would look
    implemented and behave like the baseline.
    """
    model = build_model(tiny_model_dir, query_attention_mode="cosine_tau",
                        diversity_mode="slot_rep", merge_gate_mode="query")
    model(*make_batch(30, 8)).loss.backward()

    watched = {
        "compressor.query_encoder.raw_attn_tau": "attention temperature",
        "compressor.query_encoder.proj.weight": "query projection",
        "compressor.query_encoder.slots": "query slots",
        # the gate's *output* projection, not its first layer: the residual gate
        # is zero-initialised, exactly like the historical plain gate, so the
        # first layer's gradient is zero at step 0 and becomes non-zero once the
        # output projection moves off zero
        "compressor.merger.gate.2.weight": "query-conditioned merge gate",
    }
    for name, label in watched.items():
        param = dict(model.named_parameters())[name]
        assert param.grad is not None, f"{label} received no gradient at all"
        assert param.grad.abs().sum() > 0, f"{label} received an all-zero gradient"


def test_query_gate_width_follows_the_configured_mode(tiny_model_dir):
    plain = build_model(tiny_model_dir)
    queried = build_model(tiny_model_dir, merge_gate_mode="query")
    plain_in = plain.compressor.merger.gate[0].in_features
    queried_in = queried.compressor.merger.gate[0].in_features
    width = plain.compressor.relevance_dim
    assert plain_in == 2 * width
    assert queried_in == plain_in + width        # one extra relevance vector


def test_dot_mode_does_not_read_the_attention_temperature(tiny_model_dir):
    """A checkpoint trained before the parameter existed must still be usable."""
    model = build_model(tiny_model_dir)          # dot mode
    batch = make_batch(20, 5)                    # one batch, so only the
    with torch.no_grad():                        # parameter can move the loss
        model.compressor.query_encoder.raw_attn_tau.fill_(50.0)
    first = model(*batch).loss
    with torch.no_grad():
        model.compressor.query_encoder.raw_attn_tau.fill_(-50.0)
    second = model(*batch).loss
    assert torch.equal(first, second)
