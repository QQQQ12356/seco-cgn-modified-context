"""Context-first, ordered compression with a fixed query representation."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from compression_modules import (
    ContextScorer, MultiScaleQuery, MultiscaleCompressor, allocate_block_budget,
)


def semantic_blocks(unit, count, radius=0.25, mode="semantic"):
    length = unit.size(0)
    if not 1 <= count <= length:
        raise ValueError("block count must be between one and context length")
    change = (1 - (unit[1:] * unit[:-1]).sum(-1)).clamp(0, 2)
    boundaries = [0]
    margin = int(length / count * radius)
    for index in range(1, count):
        center = length * index // count
        lower = max(boundaries[-1] + 1, center - margin)
        upper = min(length - (count - index), center + margin)
        boundary = center
        if mode == "semantic":
            boundary = lower + int(change[lower - 1:upper].argmax())
        boundaries.append(boundary)
    boundaries.append(length)
    return list(zip(boundaries[:-1], boundaries[1:]))


def mass_segments(mass, count, offset=0):
    length = mass.numel()
    if not 1 <= count <= length:
        raise ValueError("segment count must be between one and block length")
    cumulative = mass.detach().float().cumsum(0)
    targets = cumulative[-1] * torch.arange(
        1, count, device=mass.device, dtype=torch.float32
    ) / count
    proposed = (torch.searchsorted(cumulative, targets) + 1).tolist()
    boundaries = [0]
    for index, boundary in enumerate(proposed, 1):
        boundaries.append(max(boundaries[-1] + 1, min(boundary, length - count + index)))
    boundaries.append(length)
    return [(offset + start, offset + end)
            for start, end in zip(boundaries[:-1], boundaries[1:])]


class ContiguousMerger(nn.Module):
    def __init__(self, anchor_weight=0.35, mode="hybrid"):
        super().__init__()
        self.anchor_weight = anchor_weight
        self.mode = mode
        self.raw_temperature = nn.Parameter(torch.tensor(0.0))

    def forward(self, hidden, score, segments):
        memories, anchors, weights = [], [], []
        temperature = 0.1 + 1.9 * self.raw_temperature.sigmoid()
        for start, end in segments:
            local = hidden[start:end].float()
            local_score = score[start:end]
            anchor = start + int(local_score.detach().argmax())
            weight = torch.softmax(local_score / temperature, dim=0)
            if self.mode == "mean":
                weight = torch.ones_like(weight) / weight.numel()
            elif self.mode == "anchor":
                weight = F.one_hot(
                    local_score.detach().argmax(), end - start
                ).to(weight.dtype)
            else:
                weight = (1 - self.anchor_weight) * weight
                weight = weight + self.anchor_weight * F.one_hot(
                    local_score.detach().argmax(), end - start
                ).to(weight.dtype)
            memories.append((weight[:, None] * local).sum(0))
            anchors.append(anchor)
            weights.append(weight)
        return torch.stack(memories).to(hidden.dtype), anchors, weights


class ContextCompressor(MultiscaleCompressor):
    """Reuse query encoding, replace context selection and aggregation only."""

    def __init__(self, d_model, relevance_dim=256, num_slots=8,
                 phrase_widths=(2, 4), window_widths=(8, 32), block_width=128,
                 fusion_tau_init=0.1, attention_mode="dot", attn_tau_init=0.1,
                 query_mode="multi_slot", diversity_mode="attention",
                 boundary_mode="semantic", context_budget_mode="adaptive",
                 merge_mode="hybrid", novelty_weight=0.5,
                 boundary_radius=0.25, anchor_weight=0.35):
        nn.Module.__init__(self)
        if boundary_mode not in ("semantic", "uniform"):
            raise ValueError("unknown context boundary mode")
        if context_budget_mode not in ("adaptive", "uniform"):
            raise ValueError("unknown context budget mode")
        if merge_mode not in ("hybrid", "mean", "anchor"):
            raise ValueError("unknown context merge mode")
        if diversity_mode not in ("attention", "slot_rep"):
            raise ValueError("unknown diversity mode")
        if block_width < 1 or not window_widths or min(window_widths) < 1:
            raise ValueError("context widths must be positive")
        if not 0 <= novelty_weight <= 1 or not 0 <= boundary_radius < 0.5:
            raise ValueError("invalid novelty weight or boundary radius")
        if not 0 <= anchor_weight < 1:
            raise ValueError("anchor weight must lie in [0, 1)")
        self.block_width = block_width
        self.boundary_mode = boundary_mode
        self.context_budget_mode = context_budget_mode
        self.novelty_weight = novelty_weight
        self.boundary_radius = boundary_radius
        self.diversity_mode = diversity_mode
        self.query_encoder = MultiScaleQuery(
            d_model, relevance_dim=relevance_dim, num_slots=num_slots,
            phrase_widths=phrase_widths, attention_mode=attention_mode,
            attn_tau_init=attn_tau_init, query_mode=query_mode,
        )
        self.scorer = ContextScorer(
            relevance_dim=relevance_dim, window_widths=window_widths,
            fusion_tau_init=fusion_tau_init,
        )
        self.merger = ContiguousMerger(anchor_weight, merge_mode)

    def forward(self, context_hidden, query_hidden, budget, block_width=None,
                return_trace=False):
        length = context_hidden.size(0)
        if isinstance(budget, bool) or int(budget) != budget or budget < 0:
            raise ValueError("budget must be a nonnegative integer")
        budget = min(int(budget), length)
        trace = {
            "context_tokens": length, "memory_tokens": budget,
            "selection_mode": "context_adaptive_v1", "div_loss": None,
            "budget_feasible": True,
        }
        if budget == 0:
            return context_hidden[:0], trace
        if budget == length:
            return context_hidden, trace
        unit = self.encode_context(context_hidden)
        query_repr, query_aux = None, None
        if query_hidden is not None and query_hidden.size(0):
            query_repr, query_aux = self.query_encoder(query_hidden)
        trace["div_loss"] = self.diversity_term(query_aux, query_repr)
        valid = torch.ones(length, dtype=torch.bool, device=unit.device)
        score, relevance = self.scorer(unit, query_repr, valid)
        local_mean = F.avg_pool1d(
            unit.t().unsqueeze(0), kernel_size=5, stride=1,
            padding=2, count_include_pad=False,
        ).squeeze(0).t()
        novelty = (1 - F.cosine_similarity(unit, local_mean, dim=-1)).clamp(0, 2) / 2
        mass = 0.1 + self.novelty_weight * novelty + (1 - self.novelty_weight) * score.sigmoid()
        width = self.block_width if block_width is None else int(block_width)
        if width < 1:
            raise ValueError("block width must be positive")
        blocks = semantic_blocks(
            unit.detach(), min(budget, math.ceil(length / width)),
            self.boundary_radius, self.boundary_mode,
        )
        allocation_mass = mass if self.context_budget_mode == "adaptive" else torch.ones_like(mass)
        block_scores = [float(allocation_mass[start:end].detach().sum().log())
                        for start, end in blocks]
        allocation = allocate_block_budget(
            block_scores, budget, [end - start for start, end in blocks]
        )
        segments, anchor_blocks = [], {}
        for block_index, ((start, end), count) in enumerate(zip(blocks, allocation)):
            for segment in mass_segments(allocation_mass[start:end], count, start):
                anchor_blocks[len(segments)] = block_index
                segments.append(segment)
        merge_score = score + self.novelty_weight * novelty
        memory, anchors, weights = self.merger(context_hidden, merge_score, segments)
        trace.update({
            "num_blocks": len(blocks), "block_ranges": blocks,
            "block_allocation": allocation, "segment_ranges": segments,
            "anchors": anchors, "anchor_blocks": anchor_blocks,
            "anchor_types": ["raw" if end - start == 1 else self.merger.mode
                             for start, end in segments],
            "raw_positions": [start for start, end in segments if end - start == 1],
            "raw_slots": sum(end - start == 1 for start, end in segments),
            "merge_slots": sum(end - start > 1 for start, end in segments),
            "num_query_slots": 0 if query_repr is None else query_repr.size(0),
        })
        if return_trace:
            trace.update({
                "score": score.detach().cpu(), "relevance": relevance.detach().cpu(),
                "novelty": novelty.detach().cpu(), "information_mass": mass.detach().cpu(),
                "merge_members": {anchor: list(range(start, end))
                                  for anchor, (start, end) in zip(anchors, segments)},
                "merge_weights": {anchor: weight.detach().cpu()
                                  for anchor, weight in zip(anchors, weights)},
            })
        return memory, trace
