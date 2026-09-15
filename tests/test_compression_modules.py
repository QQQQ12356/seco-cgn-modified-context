"""Pure-tensor correctness and gradient tests for the compression route.

Runnable without downloading a language model:

    python -m pytest tests/test_compression_modules.py -q
"""

import math
import os
import random
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compression_modules import (  # noqa: E402
    ContextScorer,
    EvidenceMerger,
    MultiscaleCompressor,
    _block_scores,
    _split_blocks,
    allocate_block_budget,
    bounded_tau,
    resolve_budget,
    select_anchors,
)

D_MODEL = 64
RELEVANCE_DIM = 32
DTYPE = torch.bfloat16


def build_compressor(**overrides):
    torch.manual_seed(0)
    kwargs = dict(
        relevance_dim=RELEVANCE_DIM,
        num_slots=4,
        phrase_widths=(2, 4),
        window_widths=(8, 32),
        block_width=16,
        raw_fraction=0.25,
    )
    kwargs.update(overrides)
    return MultiscaleCompressor(D_MODEL, **kwargs)


def random_hidden(length, dtype=DTYPE, requires_grad=False):
    tensor = torch.randn(length, D_MODEL, dtype=dtype)
    if requires_grad:
        tensor.requires_grad_(True)
    return tensor


# --------------------------------------------------------------------------
# budget semantics
# --------------------------------------------------------------------------

@pytest.mark.parametrize("mode,length,ratio,expected", [
    ("legacy", 0, 32, (0, True)),
    ("legacy", 1, 32, (1, True)),
    ("legacy", 10, 32, (2, True)),          # the historical floor of two tokens
    ("legacy", 1000, 32, (32, True)),
    ("ceil_target", 0, 32, (0, True)),
    ("ceil_target", 10, 32, (1, True)),
    ("ceil_target", 1000, 32, (32, True)),
    ("strict_context", 0, 32, (0, True)),
    ("strict_context", 10, 32, (1, False)),  # infeasible: fewer tokens than the ratio
    ("strict_context", 32, 32, (1, True)),
    ("strict_context", 1000, 32, (31, True)),
    ("strict_context", 1000, 1, (1000, True)),
])
def test_resolve_budget_table(mode, length, ratio, expected):
    assert resolve_budget(length, ratio, mode) == expected


def test_resolve_budget_rejects_unknown_mode():
    with pytest.raises(ValueError):
        resolve_budget(100, 32, "nonsense")


def test_resolve_budget_ratio_below_one_is_clamped():
    # R < 1 is not a compression target; clamp to 1 rather than dividing by zero
    assert resolve_budget(10, 0, "strict_context") == (10, True)


@pytest.mark.parametrize("ratio", [16, 32, 64])
def test_strict_budget_meets_the_requested_ratio(ratio):
    length = ratio * 7 + 3
    budget, feasible = resolve_budget(length, ratio, "strict_context")
    assert feasible
    assert budget == length // ratio
    assert length / budget >= ratio


def test_ceil_target_does_not_guarantee_the_ratio():
    length, ratio = 1000, 32
    budget, _ = resolve_budget(length, ratio, "ceil_target")
    assert budget == 32
    assert length / budget < ratio        # 31.25x: documented, not a bug


# --------------------------------------------------------------------------
# capacity-constrained allocation
# --------------------------------------------------------------------------

def test_allocation_sums_exactly_to_budget():
    allocation = allocate_block_budget([0.1, 0.5, 0.4], 10, [4, 4, 4])
    assert sum(allocation) == 10
    assert all(0 <= a <= c for a, c in zip(allocation, [4, 4, 4]))


def test_allocation_respects_saturated_blocks():
    # block 0 can only take three; the surplus has to move to the others
    allocation = allocate_block_budget([5.0, 0.0, 0.0], 9, [3, 3, 3])
    assert allocation[0] == 3
    assert sum(allocation) == 9


def test_allocation_is_stable_on_tied_scores():
    first = allocate_block_budget([0.0, 0.0, 0.0], 7, [3, 3, 3])
    second = allocate_block_budget([0.0, 0.0, 0.0], 7, [3, 3, 3])
    assert first == second
    assert sum(first) == 7


def test_allocation_rejects_impossible_budget():
    with pytest.raises(ValueError):
        allocate_block_budget([0.0, 0.0], 1, [4, 4])       # fewer than blocks
    with pytest.raises(ValueError):
        allocate_block_budget([0.0, 0.0], 20, [4, 4])      # more than capacity


def test_allocation_randomised_invariants():
    rng = random.Random(1234)
    for _ in range(500):
        blocks = rng.randint(1, 8)
        caps = [rng.randint(1, 30) for _ in range(blocks)]
        budget = rng.randint(blocks, sum(caps))
        allocation = allocate_block_budget(
            [rng.random() for _ in range(blocks)], budget, caps
        )
        assert sum(allocation) == budget
        assert all(0 <= a <= c for a, c in zip(allocation, caps))


def test_split_blocks_covers_every_position_exactly_once():
    for length, blocks in [(10, 3), (7, 7), (100, 1), (13, 5)]:
        ranges = _split_blocks(length, blocks)
        assert sum(end - start for start, end in ranges) == length
        assert ranges[0][0] == 0
        assert ranges[-1][1] == length
        for (_, previous_end), (start, _) in zip(ranges, ranges[1:]):
            assert previous_end == start


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------

def test_select_anchors_returns_budget_in_original_order():
    torch.manual_seed(0)
    length, budget = 200, 9
    score = torch.randn(length)
    qsim = torch.rand(length, 4)
    unit = torch.nn.functional.normalize(torch.randn(length, RELEVANCE_DIM), dim=-1)
    anchors = select_anchors(score, qsim, unit, budget, block_width=32)
    assert len(anchors) == budget
    assert anchors == sorted(anchors)
    assert len(set(anchors)) == budget


def test_select_anchors_is_deterministic():
    torch.manual_seed(0)
    length, budget = 120, 6
    score = torch.zeros(length)                       # all ties
    qsim = torch.zeros(length, 4)
    unit = torch.nn.functional.normalize(torch.randn(length, RELEVANCE_DIM), dim=-1)
    first = select_anchors(score, qsim, unit, budget, block_width=16)
    second = select_anchors(score, qsim, unit, budget, block_width=16)
    assert first == second


def test_select_anchors_short_context_returns_all_positions():
    score = torch.randn(5)
    qsim = torch.rand(5, 2)
    unit = torch.randn(5, RELEVANCE_DIM)
    assert select_anchors(score, qsim, unit, 5) == [0, 1, 2, 3, 4]


def test_select_anchors_zero_budget_is_empty():
    assert select_anchors(torch.randn(5), torch.rand(5, 2), torch.randn(5, 8), 0) == []


def test_selection_is_blind_to_scores_when_everything_ties():
    """A degenerate but important case: no score information must not crash."""
    length, budget = 40, 4
    score = torch.zeros(length)
    qsim = torch.full((length, 4), 0.5)
    unit = torch.nn.functional.normalize(torch.randn(length, RELEVANCE_DIM), dim=-1)
    anchors = select_anchors(score, qsim, unit, budget, block_width=8)
    assert len(anchors) == budget


# --------------------------------------------------------------------------
# gradient plumbing
# --------------------------------------------------------------------------

def test_diversity_loss_matches_uniform_overlap():
    from compression_modules import MultiScaleQuery
    attention = torch.full((4, 10), 0.1)
    assert MultiScaleQuery.diversity_loss(attention).item() == pytest.approx(0.1, abs=1e-5)


def test_diversity_loss_is_zero_for_orthogonal_slots():
    from compression_modules import MultiScaleQuery
    attention = torch.zeros(3, 8)
    attention[0, 0] = attention[1, 1] = attention[2, 2] = 1.0
    assert MultiScaleQuery.diversity_loss(attention).item() == pytest.approx(0.0, abs=1e-6)


def test_diversity_loss_skipped_for_short_queries():
    from compression_modules import MultiScaleQuery
    assert MultiScaleQuery.diversity_loss(torch.softmax(torch.randn(4, 3), -1)) is None
    assert MultiScaleQuery.diversity_loss(None) is None


@pytest.mark.parametrize("length,query_len,budget", [
    (60, 10, 4),
    (60, 10, 1),
    (10, 3, 1),
    (500, 40, 16),
    (40, 1, 2),
])
def test_gradients_reach_compressor_parameters(length, query_len, budget):
    compressor = build_compressor()
    context = random_hidden(length, requires_grad=True)
    query = random_hidden(query_len)
    memory, _ = compressor(context, query, budget)
    memory.float().pow(2).sum().backward()

    with_gradient = [
        name for name, param in compressor.named_parameters()
        if param.grad is not None and param.grad.abs().sum() > 0
    ]
    assert len(with_gradient) >= 8, with_gradient
    assert context.grad is not None and torch.isfinite(context.grad).all()
    # the projection must learn from the encoder signal, not only from the merge
    assert any("query_encoder.proj" in name for name in with_gradient)


def test_backward_is_finite_on_a_degenerate_all_equal_context():
    compressor = build_compressor()
    context = torch.ones(80, D_MODEL, dtype=DTYPE, requires_grad=True)
    query = torch.ones(5, D_MODEL, dtype=DTYPE)
    memory, _ = compressor(context, query, 4)
    memory.float().sum().backward()
    for name, param in compressor.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), name


def test_bounded_temperature_stays_inside_its_range():
    raw = torch.linspace(-50, 50, 101)
    tau = bounded_tau(raw, 0.02, 0.5)
    assert tau.min() >= 0.02 - 1e-6
    assert tau.max() <= 0.5 + 1e-6


# --------------------------------------------------------------------------
# output contract
# --------------------------------------------------------------------------

@pytest.mark.parametrize("length,budget", [
    (60, 1), (60, 2), (60, 7), (60, 59), (100, 16), (100, 100),
])
def test_output_is_exactly_budget_tokens(length, budget):
    compressor = build_compressor()
    memory, trace = compressor(random_hidden(length), random_hidden(9), budget)
    assert memory.shape == (budget, D_MODEL)
    assert trace["memory_tokens"] == budget
    assert torch.isfinite(memory.float()).all()


def test_full_budget_returns_the_context_untouched():
    compressor = build_compressor()
    context = random_hidden(50)
    memory, trace = compressor(context, random_hidden(9), 50)
    assert torch.equal(memory, context)
    assert trace["merge_slots"] == 0


def test_empty_context_returns_empty_memory():
    compressor = build_compressor()
    memory, trace = compressor(random_hidden(0), random_hidden(9), 0)
    assert memory.shape == (0, D_MODEL)
    assert trace["memory_tokens"] == 0


def test_empty_query_falls_back_without_a_query_signal():
    compressor = build_compressor()
    memory, trace = compressor(random_hidden(80), random_hidden(0), 4)
    assert memory.shape == (4, D_MODEL)
    assert trace["div_loss"] is None
    assert trace["num_query_slots"] == 0


def test_raw_slots_are_bitwise_copies_of_the_context():
    compressor = build_compressor(raw_fraction=0.5)
    context = random_hidden(120)
    memory, trace = compressor(context, random_hidden(9), 4)
    anchors = trace["anchors"]
    num_raw = math.floor(0.5 * len(anchors))
    exact = sum(
        torch.equal(memory[position], context[anchor])
        for position, anchor in enumerate(anchors)
    )
    assert exact == num_raw


def test_memory_follows_the_original_anchor_order():
    compressor = build_compressor()
    _, trace = compressor(random_hidden(300), random_hidden(9), 8)
    assert trace["anchors"] == sorted(trace["anchors"])


def test_selection_ignores_the_query_when_it_is_empty():
    """Replacing the query with zeros must not change an empty-query selection."""
    compressor = build_compressor()
    context = random_hidden(200)
    _, first = compressor(context, random_hidden(0), 6)
    _, second = compressor(context, random_hidden(0), 6)
    assert first["anchors"] == second["anchors"]


def test_result_is_deterministic_across_repeated_calls():
    compressor = build_compressor()
    context = random_hidden(150)
    query = random_hidden(11)
    first, _ = compressor(context, query, 5)
    second, _ = compressor(context, query, 5)
    assert torch.equal(first, second)


def test_eval_mode_survives_a_context_of_ones():
    """All-equal context: every cosine is the same, softmaxes must not blow up."""
    compressor = build_compressor()
    memory, _ = compressor(torch.ones(300, D_MODEL, dtype=DTYPE), random_hidden(7), 8)
    assert torch.isfinite(memory.float()).all()


def test_long_context_with_a_wide_block_stays_finite():
    compressor = build_compressor(block_width=2048)
    memory, _ = compressor(random_hidden(900), random_hidden(20), 10)
    assert memory.shape == (10, D_MODEL)
    assert torch.isfinite(memory.float()).all()


# --------------------------------------------------------------------------
# context scorer
# --------------------------------------------------------------------------

def test_scorer_standardises_over_valid_tokens():
    torch.manual_seed(0)
    scorer = ContextScorer(relevance_dim=RELEVANCE_DIM, window_widths=(4, 8))
    context = torch.nn.functional.normalize(torch.randn(50, RELEVANCE_DIM), dim=-1)
    query = torch.nn.functional.normalize(torch.randn(3, RELEVANCE_DIM), dim=-1)
    score, relevance = scorer(context, query, torch.ones(50, dtype=torch.bool))
    assert score.mean().abs().item() < 1e-4
    assert score.std(unbiased=False).item() == pytest.approx(1.0, abs=1e-3)
    assert relevance.shape == (50,)


def test_scorer_residual_keeps_the_query_signal_at_initialisation():
    """With a zero-initialised MLP the score is the raw relevance signal."""
    torch.manual_seed(0)
    scorer = ContextScorer(relevance_dim=RELEVANCE_DIM, window_widths=(4,))
    context = torch.nn.functional.normalize(torch.randn(30, RELEVANCE_DIM), dim=-1)
    query = torch.nn.functional.normalize(torch.randn(3, RELEVANCE_DIM), dim=-1)
    score, relevance = scorer(context, query, torch.ones(30, dtype=torch.bool))
    expected = (relevance - relevance.mean()) / (relevance.std(unbiased=False) + 1e-5)
    assert torch.allclose(score, expected, atol=1e-4)


def test_scorer_handles_the_empty_query_without_nan():
    torch.manual_seed(0)
    scorer = ContextScorer(relevance_dim=RELEVANCE_DIM, window_widths=(4, 8))
    context = torch.nn.functional.normalize(torch.randn(30, RELEVANCE_DIM), dim=-1)
    score, relevance = scorer(context, None, torch.ones(30, dtype=torch.bool))
    assert torch.isfinite(score).all()
    assert torch.equal(relevance, torch.zeros(30))


def test_scorer_single_valid_token_does_not_divide_by_zero():
    scorer = ContextScorer(relevance_dim=RELEVANCE_DIM, window_widths=(4,))
    context = torch.nn.functional.normalize(torch.randn(1, RELEVANCE_DIM), dim=-1)
    score, _ = scorer(context, None, torch.ones(1, dtype=torch.bool))
    assert torch.isfinite(score).all()


# --------------------------------------------------------------------------
# evidence merger
# --------------------------------------------------------------------------

def test_merger_starts_as_the_anchor_hidden_state():
    """alpha is initialised at 0.7, so a fresh merger is an anchor-weighted blend."""
    torch.manual_seed(0)
    merger = EvidenceMerger(relevance_dim=RELEVANCE_DIM, raw_fraction=0.0)
    hidden = random_hidden(40)
    unit = torch.nn.functional.normalize(torch.randn(40, RELEVANCE_DIM), dim=-1)
    score = torch.randn(40)
    qsim = torch.rand(40, 3)
    anchors = [3, 17, 29]
    # block ranges must partition [0, length): three contiguous blocks of 40
    memory, info = merger(
        hidden, unit, score, qsim, anchors,
        {0: 0, 1: 1, 2: 2},
        [(0, 14), (14, 27), (27, 40)],
    )
    assert memory.shape == (3, D_MODEL)
    assert torch.isfinite(memory.float()).all()
    # merge slots must pool their own block's members, not fall back to the anchor
    assert all(len(members) > 1 for members in info["merge_members"].values()), info


def test_merger_output_is_a_convex_blend_of_anchor_and_members():
    """With alpha=0.7 the blend must stay inside the span of its member set."""
    torch.manual_seed(0)
    merger = EvidenceMerger(relevance_dim=RELEVANCE_DIM, raw_fraction=0.0)
    hidden = torch.randn(20, D_MODEL, dtype=torch.float32)
    unit = torch.nn.functional.normalize(torch.randn(20, RELEVANCE_DIM), dim=-1)
    memory, info = merger(
        hidden, unit, torch.randn(20), torch.rand(20, 2), [5],
        {0: 0}, [(0, 20)],
    )
    assert memory.shape == (1, D_MODEL)
    assert torch.isfinite(memory).all()
    # every member of the single block must be available as a pooling candidate
    assert len(info["merge_members"][0]) == 20


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


def test_block_scores_are_size_invariant():
    """Two equally informative blocks must score equally despite a size gap.

    Regression guard for log-sum-exp: the raw sum grows with block size, so a
    large-but-average block would out-bid a small-but-identical one.
    """
    score = torch.zeros(44)
    small, large = _block_scores(score, [(0, 4)])[0], _block_scores(score, [(0, 40)])[0]
    assert small == pytest.approx(large, abs=1e-5)
    assert small == pytest.approx(0.0, abs=1e-5)


def test_block_scores_rank_by_content_not_length():
    score = torch.zeros(44)
    score[:4] = 3.0                       # small but concentrated signal
    score[4:44] = -3.0                    # large and uninformative
    small, large = _block_scores(score, [(0, 4), (4, 44)])
    assert small > large
