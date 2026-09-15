"""Compressor components for the multiscale budgeted compression route.

Implements the design in ``Docx/SECO压缩优化方案与实施交接.md`` section 5:

* :func:`resolve_budget` -- explicit compression budget semantics.
* :class:`MultiScaleQuery` -- token / phrase / global query representation with
  learnable slots and a scale gate (module A).
* :class:`ContextScorer` -- query-guided context informativeness score built on
  local windows of the low-dimensional relevance space (module B).
* :func:`allocate_block_budget` / :func:`select_anchors` -- deterministic,
  budget-exact coverage + de-duplication selection (module C).
* :class:`EvidenceMerger` -- raw evidence slots plus anchor-residual soft
  aggregation (module D).

Every component is a plain ``nn.Module``/function over tensors so it can be unit
tested without loading a language model.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

BUDGET_MODES = ("legacy", "ceil_target", "strict_context")

# how the query slots read the query candidate lists. "dot" is the historical
# unnormalised Q.K/sqrt(p) score; "cosine_tau" normalises both sides and divides
# by a learnable temperature, which is the only way to escape the near-uniform
# regime the saved v1 checkpoint sits in (see the v1 review doc, section 5.2).
QUERY_ATTENTION_MODES = ("dot", "cosine_tau")
QUERY_MODES = ("multi_slot", "mean")
DIVERSITY_MODES = ("attention", "slot_rep")
MERGE_GATE_MODES = ("plain", "query")

# every temperature is parameterised as lo + (hi - lo) * sigmoid(raw)
TAU_QUERY_RANGE = (0.02, 0.5)
TAU_ASSIGN_RANGE = (0.02, 0.5)
TAU_MERGE_RANGE = (0.2, 5.0)
# the attention temperature is *not* the fusion temperature: it scales the
# slot-to-candidate logits before the softmax, not the fusion over slots
TAU_ATTN_RANGE = (0.03, 1.0)

_EPS = 1e-5


def inverse_sigmoid(value, lo, hi):
    """Raw parameter value whose bounded sigmoid parameterisation hits ``value``."""
    z = (float(value) - lo) / (hi - lo)
    z = min(max(z, 1e-4), 1 - 1e-4)
    return math.log(z / (1 - z))


def bounded_tau(raw, lo, hi):
    return lo + (hi - lo) * torch.sigmoid(raw)


def resolve_budget(length, ratio, mode="strict_context"):
    """Return ``(K, feasible)`` for a context of ``length`` valid tokens.

    ``legacy`` preserves the historical ``min(L, max(2, ceil(L/R)))`` behaviour
    (note the floor of two tokens). ``ceil_target`` is ``max(1, ceil(L/R))``.
    ``strict_context`` is ``floor(L/R)`` and is infeasible for ``0 < L < R``,
    where it falls back to a single memory slot and reports ``feasible=False``.
    """
    if mode not in BUDGET_MODES:
        raise ValueError(f"unknown budget mode: {mode!r}")
    length = int(length)
    if length <= 0:
        return 0, True
    ratio = max(1, int(ratio))
    if mode == "legacy":
        return min(length, max(2, math.ceil(length / ratio))), True
    if mode == "ceil_target":
        return max(1, math.ceil(length / ratio)), True
    # strict_context
    if length >= ratio:
        return length // ratio, True
    return 1, False


def _optional_scalar(value):
    """Detach a tensor to a float, passing ``None`` straight through.

    An objective that is undefined for this configuration returns ``None``; the
    trace has to record that, not crash on it.
    """
    if value is None:
        return None
    return float(value.detach())


def _normalised_entropy(attention):
    """Mean row entropy of a ``(M, n)`` attention map, divided by ``log n``.

    ``n == 1`` has no defined normalisation and returns 0.0 rather than a
    division by zero. A value near 1.0 means the distribution is flat.
    """
    probabilities = attention.detach().float().clamp_min(1e-12)
    entropy = -(probabilities * probabilities.log()).sum(dim=-1)
    width = attention.size(-1)
    if width <= 1:
        return 0.0
    return float((entropy / math.log(width)).mean())


def _masked_window_mean(z, valid, width):
    """Mean of ``z`` over a centred window of ``width`` clipped to valid entries.

    ``z`` is ``(L, p)`` and ``valid`` an ``(L,)`` float mask. Invalid entries
    never contribute to either the numerator or the denominator, so padding and
    query tokens cannot leak into a context local summary.
    """
    length = z.size(0)
    width = min(int(width), length)
    if width <= 1:
        return z
    masked = z * valid.unsqueeze(-1)
    cum_sum = torch.cumsum(masked, dim=0)
    cum_zero = z.new_zeros(1, z.size(1))
    cum_sum = torch.cat([cum_zero, cum_sum], dim=0)          # (L+1, p)
    cum_cnt = torch.cumsum(valid, dim=0)
    cum_cnt = torch.cat([valid.new_zeros(1), cum_cnt], dim=0)  # (L+1,)

    index = torch.arange(length, device=z.device)
    # exactly `width` entries centred as evenly as possible: the previous
    # `i - width//2 .. i + width//2 + 1` realised width+1 for every even width,
    # so a configured 8/32 silently pooled 9/33 tokens
    lo = (index - (width - 1) // 2).clamp_min(0)
    hi = (index - (width - 1) // 2 + width).clamp_max(length)
    window_sum = cum_sum[hi] - cum_sum[lo]
    count = (cum_cnt[hi] - cum_cnt[lo]).clamp_min(1.0).unsqueeze(-1)
    return window_sum / count


class MultiScaleQuery(nn.Module):
    """Fuse query hidden states into ``M`` low-dimensional slot representations.

    The query is represented at three scales (per token, short phrases, global);
    each scale is attended by its own set of learnable slots so that a long
    phrase list cannot dominate a short token list through sheer candidate
    count. A per-slot scale gate mixes the scales and every scale keeps a global
    residual so no slot can collapse onto a single phrase.

    ``attention_mode`` selects how a slot scores a candidate:

    * ``dot``         -- historical ``K.slot / sqrt(p)``. The token scale's keys
      are unit norm but the slots are free vectors that the optimiser keeps
      small, so the logits are bounded by ``2*||slot||/sqrt(p)`` and the softmax
      is near-uniform (<= 1.043 max/min probability ratio on the saved v1
      checkpoint). Eight slots then average the query tokens instead of
      selecting different ones.
    * ``cosine_tau``  -- normalise both sides and divide by a learnable
      temperature. The logit range stops depending on an unconstrained norm
      product, so the slot can actually commit to a subset of the query.

    ``query_mode="mean"`` drops the slot set entirely and returns the normalised
    query mean as a single slot, which is the cheap control for "do the extra
    slots buy anything".
    """

    def __init__(self, d_model, relevance_dim=256, num_slots=8, phrase_widths=(2, 4),
                 attention_mode="dot", attn_tau_init=0.1, query_mode="multi_slot"):
        super().__init__()
        if attention_mode not in QUERY_ATTENTION_MODES:
            raise ValueError(f"unknown query attention mode: {attention_mode!r}")
        if query_mode not in QUERY_MODES:
            raise ValueError(f"unknown query mode: {query_mode!r}")
        self.relevance_dim = relevance_dim
        self.num_slots = num_slots
        self.phrase_widths = tuple(int(w) for w in phrase_widths)
        self.attention_mode = attention_mode
        self.query_mode = query_mode
        # one slot set per candidate list: token, each phrase width, global
        self.num_scales = 1 + len(self.phrase_widths) + 1

        self.q_norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, relevance_dim)
        self.phrase_proj = nn.Linear(relevance_dim, relevance_dim)
        self.global_proj = nn.Linear(relevance_dim, relevance_dim)
        self.res_proj = nn.Linear(relevance_dim, relevance_dim)

        # small independent noise per scale -- deliberately not a shared BOS copy
        self.slots = nn.Parameter(
            torch.randn(self.num_scales, num_slots, relevance_dim) * 0.02
        )
        self.scale_gate = nn.Parameter(torch.zeros(num_slots, self.num_scales))
        self.raw_attn_tau = nn.Parameter(
            torch.tensor(inverse_sigmoid(attn_tau_init, *TAU_ATTN_RANGE))
        )

    def attn_tau(self):
        return bounded_tau(self.raw_attn_tau, *TAU_ATTN_RANGE)

    def _slot_logits(self, keys, slots):
        """``(M, n, p)`` candidate keys and ``(M, p)`` slots -> ``(M, n)`` logits."""
        if self.attention_mode == "cosine_tau":
            unit_keys = F.normalize(keys, dim=-1)
            unit_slots = F.normalize(slots, dim=-1)
            cosine = (unit_keys * unit_slots.unsqueeze(1)).sum(dim=-1)
            return cosine / self.attn_tau()
        return (
            torch.bmm(keys, slots.unsqueeze(-1)).squeeze(-1)
            / math.sqrt(self.relevance_dim)
        )

    def _scale_candidates(self, zq):
        """Per-scale query candidate lists; ``None`` where the scale is skipped."""
        candidates = [zq]                                # token scale
        for width in self.phrase_widths:
            if zq.size(0) >= width:
                pooled = zq.unfold(0, width, 1).mean(dim=-1)   # (Q-w+1, p)
                candidates.append(self.phrase_proj(pooled))
            else:
                candidates.append(None)
        # global scale: masked mean and the last valid token
        global_repr = torch.cat([zq.mean(dim=0, keepdim=True), zq[-1:]], dim=0)
        candidates.append(self.global_proj(global_repr))
        return candidates

    def forward(self, query_hidden):
        """``query_hidden``: ``(Q, d)`` valid query tokens, ``Q >= 1``.

        Returns ``(U, aux)`` with ``U`` the ``(M, p)`` L2-normalised slot
        representation; ``aux`` carries the token-scale attention (diversity
        loss) and the per-scale logit spread the probe reports.
        """
        zq = F.normalize(self.proj(self.q_norm(query_hidden.float())), dim=-1)
        if self.query_mode == "mean":
            # single-slot control: no learned reader at all, just the query mean
            mean_slot = F.normalize(zq.mean(dim=0, keepdim=True), dim=-1)
            return mean_slot, {
                "token_attn": None, "token_logits": None, "scale_logit_std": {},
            }

        candidates = self._scale_candidates(zq)

        scale = zq.new_zeros(self.num_scales, self.num_slots, self.relevance_dim)
        token_attn = None
        token_logits = None
        scale_logit_std = {}
        for s, cand in enumerate(candidates):
            if cand is None:
                continue                    # scale skipped: contributes nothing
            keys = cand.unsqueeze(0).expand(self.num_slots, -1, -1)   # (M, n_s, p)
            scores = self._slot_logits(keys, self.slots[s])           # (M, n_s)
            attn = torch.softmax(scores, dim=-1)
            scale[s] = torch.bmm(attn.unsqueeze(1), keys).squeeze(1)
            scale_logit_std[str(s)] = float(scores.detach().float().std())
            if s == 0:
                token_attn = attn                                       # (M, Q)
                token_logits = scores

        global_ctx = scale[-1]                                          # (M, p)
        scale = scale + self.res_proj(global_ctx).unsqueeze(0)          # global residual
        gate = torch.softmax(self.scale_gate, dim=-1).t()               # (num_scales, M)
        fused = (scale * gate.unsqueeze(-1)).sum(dim=0)                 # (M, p)
        return F.normalize(fused, dim=-1), {
            "token_attn": token_attn,
            "token_logits": token_logits,
            "scale_logit_std": scale_logit_std,
        }

    @staticmethod
    def diversity_loss(token_attn, min_query_len=4):
        """Mean pairwise overlap of the local slot attention distributions.

        Returns ``None`` when the query is too short for the constraint to mean
        anything, so the caller can skip the term instead of forcing slots to be
        orthogonal over two or three tokens.
        """
        if token_attn is None or token_attn.size(-1) < min_query_len:
            return None
        slots = token_attn.size(0)
        if slots < 2:
            return None
        overlap = torch.mm(token_attn, token_attn.t())                   # (M, M)
        off_diagonal = overlap.sum() - torch.diagonal(overlap).sum()
        return off_diagonal / (slots * (slots - 1))

    @staticmethod
    def slot_diversity_loss(slot_repr, min_slots=2):
        """Penalise squared cosine between slot *representations*.

        The attention-overlap objective above is ~1/Q for every flat attention
        distribution, so it carries almost no gradient exactly when the slots
        have collapsed onto the same query average. This term acts on the fused
        slot vectors instead and is informative whether or not the attention is
        sharp: it is 0 for mutually orthogonal slots and 1 when they coincide.
        """
        if slot_repr is None or slot_repr.size(0) < min_slots:
            return None
        unit = F.normalize(slot_repr.float(), dim=-1)
        gram = torch.mm(unit, unit.t())
        off_diagonal = gram.pow(2).sum() - torch.diagonal(gram).pow(2).sum()
        return off_diagonal / (unit.size(0) * (unit.size(0) - 1))


class ContextScorer(nn.Module):
    """Score each context token by query relevance plus local context evidence."""

    def __init__(self, relevance_dim=256, window_widths=(8, 32), hidden=256,
                 fusion_tau_init=0.1):
        super().__init__()
        self.window_widths = tuple(int(w) for w in window_widths)
        self.raw_tau = nn.Parameter(
            torch.tensor(inverse_sigmoid(fusion_tau_init, *TAU_QUERY_RANGE))
        )
        # features: token relevance, one relevance per window, novelty, norm. position
        num_features = 2 + len(self.window_widths) + 1
        self.mlp = nn.Sequential(
            nn.Linear(num_features, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        # residual scorer: start from the raw query relevance signal
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def tau(self):
        return bounded_tau(self.raw_tau, *TAU_QUERY_RANGE)

    @staticmethod
    def fuse_slots(scores, tau):
        """Smooth "at least one query aspect is relevant" score over ``M`` slots."""
        slots = scores.size(-1)
        return tau * torch.logsumexp(scores / tau, dim=-1) - tau * math.log(slots)

    def forward(self, context_rel, query_repr, valid_mask):
        """``context_rel``: ``(L, p)`` normalised; ``query_repr``: ``(M, p)``.

        Returns ``(s_standardised, relevance)`` in float32. ``s_standardised``
        is zero-mean unit-variance over the valid tokens of this sample.
        """
        length = context_rel.size(0)
        device = context_rel.device
        tau = self.tau()

        if query_repr is None or query_repr.size(0) == 0:
            # empty query: fall back to query-free local evidence only
            relevance = context_rel.new_zeros(length)
        else:
            sim = torch.mm(context_rel, query_repr.t())                  # (L, M)
            relevance = self.fuse_slots(sim, tau)

        valid = valid_mask.to(context_rel.dtype)
        features = [relevance]
        for width in self.window_widths:
            local = _masked_window_mean(context_rel, valid, width)
            local = F.normalize(local, dim=-1)
            if query_repr is None or query_repr.size(0) == 0:
                features.append(local.new_zeros(length))
            else:
                features.append(self.fuse_slots(torch.mm(local, query_repr.t()), tau))

        # novelty: how much the token deviates from its own local summary
        primary = _masked_window_mean(context_rel, valid, self.window_widths[0])
        novelty = 1.0 - F.cosine_similarity(
            context_rel, F.normalize(primary, dim=-1), dim=-1
        )
        features.append(novelty)

        position = torch.arange(length, device=device, dtype=context_rel.dtype)
        position = position / max(1, length - 1) if length > 1 else position * 0
        features.append(position)

        stacked = torch.stack(features, dim=-1)                          # (L, F)
        score = relevance + self.mlp(stacked).squeeze(-1)

        masked = score[valid_mask]
        if masked.numel() > 1:
            mean = masked.mean()
            std = masked.std(unbiased=False)
            score = (score - mean) / (std + 1e-5)
        else:
            score = score * 0.0
        return score, relevance


def allocate_block_budget(block_scores, budget, capacities):
    """Capacity-constrained largest-remainder allocation summing exactly to budget.

    Every block gets one slot first (so no region of the context is silently
    dropped), then the remaining budget is handed out proportionally to a
    softmax over block scores, respecting the per-block capacity. Ties in the
    fractional remainder go to the earliest block.
    """
    num_blocks = len(capacities)
    if num_blocks == 0:
        return []
    budget = int(budget)
    if budget < num_blocks:
        raise ValueError(
            f"budget {budget} smaller than number of blocks {num_blocks}"
        )
    if budget > sum(capacities):
        raise ValueError("budget exceeds total capacity")

    allocation = [1] * num_blocks
    remaining = budget - num_blocks
    scores = torch.as_tensor(block_scores, dtype=torch.float64)

    while remaining > 0:
        free = [b for b in range(num_blocks) if allocation[b] < capacities[b]]
        if not free:
            raise RuntimeError("no capacity left but budget remains")
        weights = torch.softmax(scores[free], dim=-1)
        quota = weights * remaining
        floor_quota = torch.floor(quota)
        granted = 0
        for pos, block in enumerate(free):
            room = capacities[block] - allocation[block]
            add = min(int(floor_quota[pos].item()), room)
            allocation[block] += add
            granted += add
        if granted > 0:
            remaining -= granted
            continue
        # nothing could be granted by floor: largest remainder wins
        remainder = quota - floor_quota
        best = int(torch.argmax(remainder).item())
        allocation[free[best]] += 1
        remaining -= 1
    return allocation


def _split_blocks(length, num_blocks):
    """Split ``range(length)`` into ``num_blocks`` contiguous near-equal blocks."""
    base, extra = divmod(length, num_blocks)
    blocks = []
    start = 0
    for b in range(num_blocks):
        size = base + (1 if b < extra else 0)
        blocks.append((start, start + size))
        start += size
    return blocks


def _block_scores(score, blocks):
    """Log-mean-exp informativeness per block.

    Log-*mean*-exp rather than log-sum-exp: the sum grows with the block size, so
    a large but unremarkable block would out-bid a small concentrated one. The
    mean-exp scale makes the comparison about content, not length.
    """
    return [
        float(torch.logsumexp(score[start:end], dim=0).item() - math.log(end - start))
        for start, end in blocks
    ]


def _greedy_in_block(score, qsim, unit, budget, coverage_weight, redundancy_weight):
    """Pick ``budget`` in-block anchors by relevance + coverage - redundancy."""
    size = score.size(0)
    if budget >= size:
        return list(range(size))
    selected = [int(torch.argmax(score).item())]
    while len(selected) < budget:
        index = torch.tensor(selected, device=score.device)
        coverage = qsim[index].max(dim=0).values                       # (M,)
        gain = torch.clamp(qsim - coverage.unsqueeze(0), min=0).mean(dim=-1)
        redundancy = torch.mm(unit, unit[index].t()).clamp_min(0).max(dim=-1).values
        candidate = score + coverage_weight * gain - redundancy_weight * redundancy
        candidate[index] = float("-inf")
        selected.append(int(torch.argmax(candidate).item()))
    return selected


def select_anchors(score, qsim, unit, budget, block_width=128,
                   coverage_weight=0.2, redundancy_weight=0.2):
    """Return ``budget`` anchor indices in original order.

    ``score`` is the standardised context informativeness ``(L,)``; ``qsim`` is
    the per-slot relevance in ``[0, 1]`` ``(L, M)``; ``unit`` are the normalised
    low-dimensional context vectors ``(L, p)``.
    """
    length = score.size(0)
    budget = int(budget)
    if budget <= 0:
        return []
    if budget >= length:
        return list(range(length))

    num_blocks = min(budget, max(1, math.ceil(length / block_width)))
    blocks = _split_blocks(length, num_blocks)
    capacities = [end - start for start, end in blocks]
    block_scores = _block_scores(score, blocks)
    allocation = allocate_block_budget(block_scores, budget, capacities)

    anchors = []
    for (start, end), take in zip(blocks, allocation):
        if take <= 0:
            continue
        local = _greedy_in_block(
            score[start:end], qsim[start:end], unit[start:end], take,
            coverage_weight, redundancy_weight,
        )
        anchors.extend(start + offset for offset in local)
    anchors.sort()
    return anchors


def _select_evidence_slots(score, unit, qsim, anchors, num_raw, coverage_weight=0.2):
    """Pick the anchors that stay as raw evidence: high score, spread out, diversified."""
    if num_raw <= 0:
        return set()
    remaining = list(anchors)
    chosen = []
    coverage = qsim.new_zeros(qsim.size(-1))
    while len(chosen) < num_raw and remaining:
        index = torch.tensor(remaining, device=score.device)
        gain = torch.clamp(qsim[index] - coverage.unsqueeze(0), min=0).mean(dim=-1)
        candidate = score[index] + coverage_weight * gain
        if chosen:
            chosen_index = torch.tensor(chosen, device=score.device)
            redundancy = torch.mm(unit[index], unit[chosen_index].t()).clamp_min(0)
            candidate = candidate - 0.5 * redundancy.max(dim=-1).values
            span = index.float().unsqueeze(1) - chosen_index.float().unsqueeze(0)
            candidate = candidate - 0.5 * (1.0 / (span.abs().min(dim=-1).values + 1.0))
        pick = int(torch.argmax(candidate).item())
        anchor = remaining.pop(pick)
        chosen.append(anchor)
        coverage = torch.maximum(coverage, qsim[anchor])
    return set(chosen)


class EvidenceMerger(nn.Module):
    """Keep a fraction of anchors as raw evidence and merge the rest.

    Merged slots are a residual blend of the anchor's own hidden state and a
    soft-pooled summary of its block, where the pooling weights combine
    relevance, position proximity and the context informativeness score.
    """

    def __init__(self, relevance_dim=256, raw_fraction=0.25, assignment_tau_init=0.1,
                 merge_tau_init=1.0, position_penalty=0.2, anchor_residual_init=0.7,
                 gate_mode="plain"):
        super().__init__()
        if gate_mode not in MERGE_GATE_MODES:
            raise ValueError(f"unknown merge gate mode: {gate_mode!r}")
        self.raw_fraction = float(raw_fraction)
        self.position_penalty = float(position_penalty)
        self.gate_mode = gate_mode
        self.raw_assign_tau = nn.Parameter(
            torch.tensor(inverse_sigmoid(assignment_tau_init, *TAU_ASSIGN_RANGE))
        )
        self.raw_merge_tau = nn.Parameter(
            torch.tensor(inverse_sigmoid(merge_tau_init, *TAU_MERGE_RANGE))
        )
        # "query" widens the gate input by the anchor's query context, so how
        # much raw evidence survives depends on what was asked, not only on how
        # the anchor and its summary relate to each other
        gate_width = 3 * relevance_dim if gate_mode == "query" else 2 * relevance_dim
        self.gate = nn.Sequential(
            nn.Linear(gate_width, relevance_dim),
            nn.GELU(),
            nn.Linear(relevance_dim, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.raw_gate_bias = nn.Parameter(
            torch.tensor(math.log(anchor_residual_init / (1 - anchor_residual_init)))
        )

    def assign_tau(self):
        return bounded_tau(self.raw_assign_tau, *TAU_ASSIGN_RANGE)

    def merge_tau(self):
        return bounded_tau(self.raw_merge_tau, *TAU_MERGE_RANGE)

    def _gate_input(self, anchor_unit, pooled_unit, anchor, qsim, query_repr):
        parts = [anchor_unit.unsqueeze(0), pooled_unit.unsqueeze(0)]
        if self.gate_mode == "query" and query_repr is not None:
            weights = qsim[anchor].clamp_min(0)
            total = weights.sum()
            # a uniform qsim row would divide by ~M; the clamp keeps a
            # zero-relevance anchor from producing a NaN "query context"
            context = (weights / total.clamp_min(_EPS)) @ query_repr
            parts.append(F.normalize(context, dim=-1).unsqueeze(0))
        return torch.cat(parts, dim=-1)

    def forward(self, hidden, unit, score, qsim, anchors, merge_anchor_blocks,
                block_ranges, return_trace=False, query_repr=None):
        """Build ``(K, d)`` memory from ``K`` anchors.

        ``merge_anchor_blocks`` maps each merge anchor index (into ``anchors``)
        to the index of the block whose tokens it summarises. With
        ``return_trace`` the real aggregation weights and entropies are returned
        alongside, detached to CPU, for the diagnostic probe.
        """
        dtype = hidden.dtype
        num_raw = int(math.floor(self.raw_fraction * len(anchors)))
        if len(anchors) <= 1:
            num_raw = 0
        raw_set = _select_evidence_slots(score, unit, qsim, anchors, num_raw)

        # merge anchors only: raw evidence slots keep their hidden state verbatim
        # and must not act as summary targets
        merges_per_block = {}
        for position, block in merge_anchor_blocks.items():
            if anchors[position] in raw_set:
                continue
            merges_per_block.setdefault(block, []).append(position)

        merged = {}
        # cheap record kept on every call: the doc requires anchor positions and
        # types in the trace. The heavy per-slot weights and entropies are only
        # materialised on request, since each one costs a host synchronisation.
        info = {"raw_positions": sorted(raw_set), "merge_members": {},
                "merge_entropy": [], "merge_entropy_normalised": [],
                "merge_alpha": [], "merge_weights": {}}
        for block, positions in merges_per_block.items():
            start, end = block_ranges[block]
            members = [
                offset for offset in range(start, end) if offset not in raw_set
            ]
            if not members:
                members = [anchors[positions[0]]]
            member_index = torch.tensor(members, device=hidden.device)
            member_unit = unit[member_index]                            # (n, p)
            anchor_local = torch.tensor(
                [anchors[position] for position in positions], device=hidden.device
            )
            # token -> merge-anchor assignment
            similarity = torch.mm(member_unit, unit[anchor_local].t())
            distance = (
                member_index.float().unsqueeze(1)
                - anchor_local.float().unsqueeze(0)
            ).abs() / max(1, end - start)
            energy = similarity / self.assign_tau() - self.position_penalty * distance
            assignment = torch.softmax(energy, dim=-1)                  # (n, k_b)

            log_weight = torch.log(assignment + _EPS)
            log_weight = log_weight + score[member_index].unsqueeze(-1) / self.merge_tau()
            weight = torch.softmax(log_weight, dim=0)                   # (n, k_b)

            pooled = torch.mm(weight.t(), hidden[member_index].float())  # (k_b, d)
            # gate works in the low-dimensional relevance space: same statistic,
            # 1/p the cost of the d-dimensional value pooling above
            pooled_unit = F.normalize(torch.mm(weight.t(), member_unit), dim=-1)
            for slot, position in enumerate(positions):
                anchor = anchors[position]
                gate_in = self._gate_input(
                    unit[anchor], pooled_unit[slot], anchor, qsim, query_repr
                )
                alpha = torch.sigmoid(self.gate(gate_in) + self.raw_gate_bias).squeeze()
                merged[position] = alpha * hidden[anchor].float() + (1 - alpha) * pooled[slot]

                info["merge_members"][position] = members
                if return_trace:
                    column = weight[:, slot].detach().float().cpu()
                    member_count = column.numel()
                    entropy = -(column * column.clamp_min(1e-12).log()).sum().item()
                    info["merge_entropy"].append(entropy)
                    # H/log(n) so a slot with a single member reads as 0, not as
                    # a falsely healthy normalised value
                    info["merge_entropy_normalised"].append(
                        entropy / math.log(member_count) if member_count > 1 else 0.0
                    )
                    info["merge_alpha"].append(float(alpha.detach()))
                    info["merge_weights"][position] = column

        output = []
        for position, anchor in enumerate(anchors):
            if anchor in raw_set:
                output.append(hidden[anchor])
            else:
                output.append(merged[position].to(dtype))
        if not output:
            memory = hidden.new_zeros(0, hidden.size(-1))
        else:
            memory = torch.stack(output, dim=0)
        return memory, info


class MultiscaleCompressor(nn.Module):
    """Assemble modules A-D behind a single ``forward`` for the SECO model."""

    def __init__(self, d_model, relevance_dim=256, num_slots=8, phrase_widths=(2, 4),
                 window_widths=(8, 32), block_width=128, raw_fraction=0.25,
                 fusion_tau_init=0.1, assignment_tau_init=0.1, merge_tau_init=1.0,
                 position_penalty=0.2, anchor_residual_init=0.7,
                 coverage_weight=0.2, redundancy_weight=0.2, selection_mode="budgeted",
                 attention_mode="dot", attn_tau_init=0.1, query_mode="multi_slot",
                 diversity_mode="attention", merge_gate_mode="plain"):
        super().__init__()
        if selection_mode not in ("budgeted", "topk"):
            raise ValueError(f"unknown selection mode: {selection_mode!r}")
        if diversity_mode not in DIVERSITY_MODES:
            raise ValueError(f"unknown diversity mode: {diversity_mode!r}")
        self.relevance_dim = relevance_dim
        self.selection_mode = selection_mode
        self.diversity_mode = diversity_mode
        self.block_width = int(block_width)
        self.coverage_weight = float(coverage_weight)
        self.redundancy_weight = float(redundancy_weight)

        # the query-fusion temperature lives in ContextScorer (it scales the
        # context-to-slot relevance fusion); the query *attention* scale is a
        # separate parameter owned by MultiScaleQuery
        self.query_encoder = MultiScaleQuery(
            d_model, relevance_dim=relevance_dim, num_slots=num_slots,
            phrase_widths=phrase_widths, attention_mode=attention_mode,
            attn_tau_init=attn_tau_init, query_mode=query_mode,
        )
        self.scorer = ContextScorer(
            relevance_dim=relevance_dim, window_widths=window_widths,
            fusion_tau_init=fusion_tau_init,
        )
        self.merger = EvidenceMerger(
            relevance_dim=relevance_dim, raw_fraction=raw_fraction,
            assignment_tau_init=assignment_tau_init,
            merge_tau_init=merge_tau_init,
            position_penalty=position_penalty,
            anchor_residual_init=anchor_residual_init,
            gate_mode=merge_gate_mode,
        )

    def diversity_term(self, query_aux, query_repr):
        """The configured slot-diversification objective, or ``None``."""
        if query_aux is None:
            return None
        if self.diversity_mode == "slot_rep":
            return MultiScaleQuery.slot_diversity_loss(query_repr)
        return MultiScaleQuery.diversity_loss(query_aux["token_attn"])

    def encode_context(self, context_hidden):
        """Shared relevance space for queries and contexts."""
        projected = self.query_encoder.proj(
            self.query_encoder.q_norm(context_hidden.float())
        )
        return F.normalize(projected, dim=-1)

    def forward(self, context_hidden, query_hidden, budget, block_width=None,
                return_trace=False):
        """Compress ``(L, d)`` context hidden states into ``(K, d)`` memory.

        Returns ``(memory, trace)``. ``trace`` always carries cheap statistics
        and the differentiable diversity term; with ``return_trace`` it also
        carries the real selection and aggregation record (anchor positions,
        raw/merge split, block allocation, per-slot entropy, slot attention),
        detached to CPU so the diagnostic probe reads the algorithm that
        actually ran instead of re-deriving a second one.
        """
        length = context_hidden.size(0)
        budget = int(budget)
        trace = {"context_tokens": length, "memory_tokens": budget}
        if length == 0 or budget <= 0:
            return context_hidden.new_zeros(0, context_hidden.size(-1)), trace

        context_unit = self.encode_context(context_hidden)
        valid = torch.ones(length, device=context_hidden.device, dtype=torch.bool)
        query_aux = None
        if query_hidden is not None and query_hidden.size(0) > 0:
            query_repr, query_aux = self.query_encoder(query_hidden)
            div_loss = self.diversity_term(query_aux, query_repr)
        else:
            # empty query: no query signal, fall back to local context evidence
            query_repr, div_loss = None, None
        score, relevance = self.scorer(context_unit, query_repr, valid)
        trace["div_loss"] = div_loss
        trace["num_query_slots"] = 0 if query_repr is None else query_repr.size(0)

        if budget >= length:
            trace["budget_feasible"] = True
            trace["raw_slots"] = length
            trace["merge_slots"] = 0
            return context_hidden, trace

        if query_repr is None:
            qsim = context_unit.new_full((length, 1), 0.5)              # neutral coverage
        else:
            qsim = (1.0 + torch.mm(context_unit, query_repr.t())) / 2.0  # (L, M)
        width = int(block_width or self.block_width)
        allocation = None
        if self.selection_mode == "topk":
            # ablation: global top-k on the scorer, no block forcing or coverage
            anchors = sorted(
                torch.topk(score, budget, largest=True).indices.tolist()
            )
            ranges = [(0, length)]
            num_blocks = 1
            anchor_blocks = {position: 0 for position in range(len(anchors))}
        else:
            anchors = select_anchors(
                score, qsim, context_unit, budget, block_width=width,
                coverage_weight=self.coverage_weight,
                redundancy_weight=self.redundancy_weight,
            )
            num_blocks = min(budget, max(1, math.ceil(length / width)))
            ranges = _split_blocks(length, num_blocks)
            anchor_blocks = {}
            for position, anchor in enumerate(anchors):
                for block, (start, end) in enumerate(ranges):
                    if start <= anchor < end:
                        anchor_blocks[position] = block
                        break

        memory, merger_info = self.merger(
            context_hidden, context_unit, score, qsim, anchors,
            anchor_blocks, ranges, return_trace=return_trace, query_repr=query_repr,
        )

        raw_positions = set(merger_info["raw_positions"])
        trace.update({
            "num_blocks": num_blocks,
            "budget_feasible": True,
            "selection_mode": self.selection_mode,
            "anchors": anchors,
            "anchor_blocks": anchor_blocks,
            "anchor_types": ["raw" if a in raw_positions else "merge" for a in anchors],
            "block_ranges": ranges,
            "block_allocation": allocation,
            "raw_positions": merger_info["raw_positions"],
            "merge_members": merger_info["merge_members"],
        })
        if return_trace:
            trace.update({
                "score": score.detach().float().cpu(),
                "relevance": relevance.detach().float().cpu(),
                "qsim": qsim.detach().float().cpu(),
                "merge_entropy": merger_info["merge_entropy"],
                "merge_entropy_normalised": merger_info["merge_entropy_normalised"],
                "merge_alpha": merger_info["merge_alpha"],
                "merge_weights": merger_info["merge_weights"],
                # None whenever the objective is undefined (mean-query control,
                # or a query too short for the constraint to mean anything)
                "slot_attn_overlap": _optional_scalar(
                    MultiScaleQuery.diversity_loss(
                        query_aux["token_attn"] if query_aux is not None else None
                    )
                ),
                # undefined for a single-slot query, exactly like the
                # attention-overlap term; records None rather than crashing
                "slot_rep_diversity": _optional_scalar(
                    MultiScaleQuery.slot_diversity_loss(query_repr)
                ),
                # attention diagnostics: a flat softmax is the failure mode this
                # route has to rule out, so the spread is recorded, not assumed
                "attn_tau": float(self.query_encoder.attn_tau().detach()),
                "attn_logit_std": query_aux["scale_logit_std"] if query_aux else None,
                "token_attn_entropy": (
                    float(_normalised_entropy(query_aux["token_attn"]))
                    if query_aux is not None and query_aux["token_attn"] is not None
                    else None
                ),
                "token_attn_max_prob": (
                    float(query_aux["token_attn"].detach().max())
                    if query_aux is not None and query_aux["token_attn"] is not None
                    else None
                ),
                # full query-side record, small enough to always attach: (M, Q)
                # attention and the (M, p) slot vectors it reads through
                "token_attn": (
                    query_aux["token_attn"].detach().float().cpu()
                    if query_aux is not None and query_aux["token_attn"] is not None
                    else None
                ),
                "slot_repr": (
                    query_repr.detach().float().cpu() if query_repr is not None else None
                ),
            })
        return memory, trace
