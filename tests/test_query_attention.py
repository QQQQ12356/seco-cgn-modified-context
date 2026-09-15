"""Tests for the calibrated query attention and the v2 aggregation changes.

The saved v1 checkpoint scores query tokens with ``K.slot / sqrt(p)`` where the
keys are unit norm and the slots are free vectors the optimiser keeps at
``||slot|| ~ 0.34``; the resulting softmax can differ between its largest and
smallest entry by at most ~4.3%, i.e. it is a near-uniform average. These tests
pin the replacement (``cosine_tau``) and the two aggregation changes that build
on a genuinely selective query representation.

    python -m pytest tests/test_query_attention.py -q
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compression_modules import (  # noqa: E402
    TAU_ATTN_RANGE,
    EvidenceMerger,
    MultiScaleQuery,
    MultiscaleCompressor,
)

D_MODEL = 32
RELEVANCE = 32
SLOTS = 4


def make_query(length=12, seed=0, dim=D_MODEL):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(length, dim, generator=generator)


def build_query_encoder(**overrides):
    torch.manual_seed(0)
    kwargs = dict(
        d_model=D_MODEL, relevance_dim=RELEVANCE, num_slots=SLOTS,
        phrase_widths=(2, 4),
    )
    kwargs.update(overrides)
    return MultiScaleQuery(**kwargs)


def token_logit_spread(encoder, query):
    _, aux = encoder(query)
    logits = aux["token_logits"]
    return float(logits.max() - logits.min())


def test_dot_attention_cannot_separate_query_tokens():
    """Reproduce the diagnosed failure: the historical scale is near-uniform."""
    encoder = build_query_encoder(attention_mode="dot")
    query = make_query()

    _, aux = encoder(query)
    attention = aux["token_attn"]
    # ||slots|| ~ 0.02*sqrt(p) at init, so both the logits and the induced
    # probabilities live in a narrow band: the slots cannot pick anything out
    assert token_logit_spread(encoder, query) < 0.1
    assert float(attention.max() / attention.min()) < 1.1


def test_cosine_temperature_makes_the_slots_selective():
    """The calibrated scale separates query tokens by a wide margin."""
    dot = build_query_encoder(attention_mode="dot")
    cosine = build_query_encoder(attention_mode="cosine_tau", attn_tau_init=0.1)

    query = make_query()
    assert token_logit_spread(cosine, query) > 10 * token_logit_spread(dot, query)

    _, aux = cosine(query)
    assert float(aux["token_attn"].max() / aux["token_attn"].min()) > 3.0


def test_attention_temperature_stays_inside_its_range():
    encoder = build_query_encoder(attention_mode="cosine_tau")
    lo, hi = TAU_ATTN_RANGE
    for raw in (-1e6, -50.0, 0.0, 50.0, 1e6):
        with torch.no_grad():
            encoder.raw_attn_tau.fill_(raw)
        assert lo - 1e-6 <= float(encoder.attn_tau()) <= hi + 1e-6


def test_attention_temperature_receives_gradient():
    encoder = build_query_encoder(attention_mode="cosine_tau")
    _, aux = encoder(make_query())
    # a differentiable route from the temperature to the loss the caller builds
    aux["token_attn"].sum().backward()
    assert encoder.raw_attn_tau.grad is not None
    assert encoder.raw_attn_tau.grad.abs().item() > 0


def test_cosine_mode_normalises_both_sides():
    """Scaling the slots must not change the attention at all."""
    encoder = build_query_encoder(attention_mode="cosine_tau")
    query = make_query()
    _, before = encoder(query)
    with torch.no_grad():
        encoder.slots.mul_(25.0)
    _, after = encoder(query)
    assert torch.allclose(before["token_attn"], after["token_attn"], atol=1e-5)


def test_mean_query_mode_returns_one_slot_and_no_attention():
    encoder = build_query_encoder(query_mode="mean")
    slots, aux = encoder(make_query())
    assert slots.shape == (1, RELEVANCE)
    assert aux["token_attn"] is None
    # a single slot is the normalised mean of the projected query tokens
    assert float(slots.norm(dim=-1)) == pytest.approx(1.0, abs=1e-4)


def test_mean_query_equals_the_projected_query_mean():
    encoder = build_query_encoder(query_mode="mean")
    query = make_query()
    slots, _ = encoder(query)
    with torch.no_grad():
        zq = torch.nn.functional.normalize(
            encoder.proj(encoder.q_norm(query.float())), dim=-1
        )
    expected = torch.nn.functional.normalize(zq.mean(dim=0, keepdim=True), dim=-1)
    assert torch.allclose(slots, expected, atol=1e-5)


def test_slot_diversity_is_zero_for_orthogonal_slots_and_one_when_identical():
    orthogonal = torch.eye(SLOTS, RELEVANCE)
    assert float(MultiScaleQuery.slot_diversity_loss(orthogonal)) == pytest.approx(
        0.0, abs=1e-6
    )
    identical = torch.ones(SLOTS, RELEVANCE)
    assert float(MultiScaleQuery.slot_diversity_loss(identical)) == pytest.approx(
        1.0, abs=1e-6
    )
    # a single slot (the mean-query control) has no pair to separate
    assert MultiScaleQuery.slot_diversity_loss(torch.ones(1, RELEVANCE)) is None


def test_slot_diversity_is_scale_invariant_and_differentiable():
    slots = torch.randn(SLOTS, RELEVANCE, requires_grad=True)
    loss = MultiScaleQuery.slot_diversity_loss(slots)
    scaled = MultiScaleQuery.slot_diversity_loss(slots.detach() * 100.0)
    assert float(loss) == pytest.approx(float(scaled), abs=1e-5)
    loss.backward()
    assert slots.grad is not None and slots.grad.abs().sum() > 0


def test_attention_diversity_has_no_signal_for_flat_attention_but_slot_rep_does():
    """The motivating contrast: why the attention-overlap term was replaced."""
    flat = torch.full((SLOTS, 10), 1.0 / 10.0)
    assert float(MultiScaleQuery.diversity_loss(flat)) == pytest.approx(0.1, abs=1e-5)

    # four slots that are all the same direction: attention overlap is already
    # at its floor, so it cannot report the collapse
    collapsed = torch.ones(SLOTS, RELEVANCE)
    assert float(MultiScaleQuery.slot_diversity_loss(collapsed)) == pytest.approx(1.0)


def build_merger(gate_mode):
    torch.manual_seed(0)
    merger = EvidenceMerger(relevance_dim=RELEVANCE, raw_fraction=0.0, gate_mode=gate_mode)
    with torch.no_grad():                      # zero-init gate has no effect yet
        merger.gate[-1].weight.normal_(0.0, 0.2)
    return merger


def merge_alpha(merger, hidden, unit, query_repr):
    length = hidden.size(0)
    score = torch.linspace(-1.0, 1.0, length)
    qsim = torch.rand(length, query_repr.size(0), generator=torch.Generator().manual_seed(3))
    memory, info = merger(
        hidden, unit, score, qsim, anchors=[0],
        merge_anchor_blocks={0: 0}, block_ranges=[(0, length)],
        return_trace=True, query_repr=query_repr,
    )
    return info["merge_alpha"][0]


def test_query_gate_mode_changes_alpha_when_the_query_changes():
    torch.manual_seed(1)
    hidden = torch.randn(6, RELEVANCE)
    unit = torch.nn.functional.normalize(hidden, dim=-1)
    merger = build_merger("query")

    first = merge_alpha(merger, hidden, unit, torch.randn(2, RELEVANCE))
    second = merge_alpha(merger, hidden, unit, torch.randn(2, RELEVANCE) * 5.0)
    assert first != pytest.approx(second, abs=1e-4)


def test_plain_gate_mode_ignores_the_query():
    torch.manual_seed(1)
    hidden = torch.randn(6, RELEVANCE)
    unit = torch.nn.functional.normalize(hidden, dim=-1)
    merger = build_merger("plain")

    first = merge_alpha(merger, hidden, unit, torch.randn(2, RELEVANCE))
    second = merge_alpha(merger, hidden, unit, torch.randn(2, RELEVANCE) * 5.0)
    assert first == pytest.approx(second, abs=1e-6)


def test_query_gate_mode_survives_an_all_zero_relevance_anchor():
    """A zero qsim row must not divide by zero when forming the query context."""
    torch.manual_seed(1)
    hidden = torch.randn(6, RELEVANCE)
    unit = torch.nn.functional.normalize(hidden, dim=-1)
    merger = build_merger("query")
    merger.eval()

    score = torch.linspace(-1.0, 1.0, 6)
    _, info = merger(
        hidden, unit, score, torch.zeros(6, 2), anchors=[0],
        merge_anchor_blocks={0: 0}, block_ranges=[(0, 6)],
        return_trace=True, query_repr=torch.randn(2, RELEVANCE),
    )
    assert info["merge_alpha"] and all(alpha == alpha for alpha in info["merge_alpha"])


def test_compressor_selects_the_configured_diversity_term():
    compressor = MultiscaleCompressor(
        D_MODEL, relevance_dim=RELEVANCE, num_slots=SLOTS, phrase_widths=(2, 4),
        window_widths=(4, 8), block_width=8, raw_fraction=0.25,
        attention_mode="cosine_tau", diversity_mode="slot_rep",
    )
    context = torch.randn(24, D_MODEL)
    query = make_query()
    _, trace = compressor(context, query, budget=6, return_trace=True)
    assert trace["div_loss"] is not None
    assert trace["slot_rep_diversity"] is not None
    assert trace["attn_tau"] is not None


def test_compressor_trace_reports_the_attention_spread():
    compressor = MultiscaleCompressor(
        D_MODEL, relevance_dim=RELEVANCE, num_slots=SLOTS, phrase_widths=(2, 4),
        window_widths=(4, 8), block_width=8, raw_fraction=0.25,
        attention_mode="cosine_tau",
    )
    _, trace = compressor(torch.randn(24, D_MODEL), make_query(), budget=6, return_trace=True)
    assert trace["attn_logit_std"]["0"] > 0
    assert 0.0 <= trace["token_attn_entropy"] <= 1.0
    assert 0.0 < trace["token_attn_max_prob"] <= 1.0


@pytest.mark.parametrize("gate_mode", ["plain", "query"])
def test_query_repr_is_accepted_by_the_compressor(gate_mode):
    compressor = MultiscaleCompressor(
        D_MODEL, relevance_dim=RELEVANCE, num_slots=SLOTS, phrase_widths=(2, 4),
        window_widths=(4, 8), block_width=8, raw_fraction=0.5, merge_gate_mode=gate_mode,
    )
    memory, _ = compressor(torch.randn(24, D_MODEL), make_query(), budget=6)
    assert memory.shape == (6, D_MODEL)
    assert torch.isfinite(memory).all()


def test_mean_query_control_produces_a_complete_trace():
    """The single-slot control must record None, not crash on an undefined term."""
    compressor = MultiscaleCompressor(
        D_MODEL, relevance_dim=RELEVANCE, num_slots=1, phrase_widths=(2, 4),
        window_widths=(4, 8), block_width=8, raw_fraction=0.25, query_mode="mean",
    )
    _, trace = compressor(torch.randn(24, D_MODEL), make_query(), budget=6, return_trace=True)
    assert trace["div_loss"] is None
    assert trace["slot_attn_overlap"] is None
    assert trace["slot_rep_diversity"] is None
    assert trace["token_attn_entropy"] is None
    assert trace["token_attn"] is None
    assert trace["slot_repr"].shape == (1, RELEVANCE)
    assert trace["num_query_slots"] == 1


def test_slot_rep_diversity_mode_is_undefined_for_one_slot_without_crashing():
    compressor = MultiscaleCompressor(
        D_MODEL, relevance_dim=RELEVANCE, num_slots=1, phrase_widths=(2, 4),
        window_widths=(4, 8), block_width=8, raw_fraction=0.25,
        query_mode="mean", diversity_mode="slot_rep",
    )
    _, trace = compressor(torch.randn(24, D_MODEL), make_query(), budget=6, return_trace=True)
    assert trace["div_loss"] is None
    assert trace["slot_rep_diversity"] is None
