"""Regression tests for the training-side prefix/answer alignment bug.

``_compress_batch`` used to right-pad every sample's compressed prefix to the
batch maximum and ``forward`` appended the whole batch's answers afterwards. For
any sample shorter than the batch maximum the first answer token therefore sat
in a padding slot: it was predicted from the *padding* position
``max_prefix - 1``, whose position id is wrong and which was built as if extra
tokens followed the query. These tests pin the corrected contract: each sample
keeps its own valid prefix, the answer is glued directly to it, and only the
concatenated sequence is padded -- at the end.

    python -m pytest tests/test_training_alignment.py -q
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HIDDEN = 64

from test_model_integration import build_model  # noqa: E402


def make_prefix(length, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(length, HIDDEN, dtype=torch.bfloat16, generator=generator)


def first_answer_position(full_labels, row=0):
    """Index of the first supervised label token in ``row``."""
    supervised = (full_labels[row] != -100).nonzero(as_tuple=True)[0]
    assert supervised.numel() > 0, "row has no supervised labels"
    return int(supervised[0])


def first_answer_logits(model, inputs_embeds, attention_mask, full_labels, row=0):
    position = first_answer_position(full_labels, row)
    outputs = model.decoder(
        inputs_embeds=inputs_embeds, attention_mask=attention_mask, return_dict=True
    )
    # causal shift: the token at `position` is predicted by the logits before it
    return outputs.logits[row, position - 1]


def mixed_length_batch(context_lens, query_len=5, answer_len=3):
    """Right-padded batch where each sample has its own context length."""
    batch = len(context_lens)
    total = max(context_lens) + query_len
    input_ids = torch.randint(100, 1000, (batch, total))
    prompt_mask = torch.zeros(batch, total, dtype=torch.long)
    attention_mask = torch.zeros(batch, total, dtype=torch.long)
    for row, context_len in enumerate(context_lens):
        prompt_mask[row, context_len:context_len + query_len] = 1
        attention_mask[row, :context_len + query_len] = 1
    labels = torch.randint(100, 1000, (batch, answer_len))
    return input_ids, prompt_mask, attention_mask, labels


def test_answer_is_glued_to_its_own_prefix_not_to_the_batch_padding(tiny_model_dir):
    """The short sample's first answer token must sit right after its own prefix."""
    model = build_model(tiny_model_dir)
    prefixes = [make_prefix(6, 0), make_prefix(11, 1)]
    masks = [torch.ones(6, dtype=torch.bool), torch.ones(11, dtype=torch.bool)]
    labels = torch.tensor([[7, 8], [9, 10]])

    _, attention, full_labels = model._build_training_batch(prefixes, masks, labels)

    assert first_answer_position(full_labels, 0) == 6
    assert first_answer_position(full_labels, 1) == 11
    # every real token is attended, every pad slot is not
    assert attention[0, :8].all() and not attention[0, 8:].any()
    assert attention[1, :13].all() and not attention[1, 13:].any()
    assert full_labels[0, :6].eq(-100).all()
    assert full_labels[0, 6:8].tolist() == [7, 8]


def test_short_sample_logits_are_independent_of_its_neighbour(tiny_model_dir):
    """Adding a longer sample must not move where the short sample is predicted from."""
    model = build_model(tiny_model_dir)
    short_labels = torch.tensor([[7, 8]])
    prefixes = [make_prefix(6, 0), make_prefix(11, 1)]
    masks = [torch.ones(6, dtype=torch.bool), torch.ones(11, dtype=torch.bool)]

    alone = model._build_training_batch([prefixes[0]], [masks[0]], short_labels)
    paired = model._build_training_batch(
        prefixes, masks, torch.tensor([[7, 8, -100], [9, 10, 11]])
    )

    alone_logits = first_answer_logits(model, *alone, row=0)
    paired_logits = first_answer_logits(model, *paired, row=0)
    assert torch.allclose(alone_logits.float(), paired_logits.float(), atol=0.25)

    # the test has power: decoding from the *padding* slot gives something else
    buggy_position = int(prefixes[1].size(0)) - 1
    outputs = model.decoder(
        inputs_embeds=paired[0], attention_mask=paired[1], return_dict=True
    )
    assert not torch.allclose(
        alone_logits.float(), outputs.logits[0, buggy_position].float(), atol=0.25
    ), "logits at the batch-padding slot coincide with the correct first-answer logits"


def test_real_forward_keeps_each_sample_aligned(tiny_model_dir):
    """End-to-end: a batch of mixed-length samples stays per-sample aligned."""
    model = build_model(tiny_model_dir)
    input_ids, prompt_mask, attention_mask, labels = mixed_length_batch([20, 12])

    hidden = model._encode(model.tokens_to_embeddings(input_ids), attention_mask)
    prefixes, masks, _ = model._compress_batch(hidden, prompt_mask, attention_mask)
    inputs_embeds, full_attention, full_labels = model._build_training_batch(
        prefixes, masks, labels
    )

    assert full_attention.shape == full_labels.shape == inputs_embeds.shape[:2]
    for row, prefix in enumerate(prefixes):
        real = int(prefix.size(0)) + int((labels[row] != -100).sum())
        assert first_answer_position(full_labels, row) == int(prefix.size(0))
        assert full_attention[row, :real].all()
        assert not full_attention[row, real:].any()


def test_generation_batch_still_pads_to_a_rectangle(tiny_model_dir):
    """The eval path needs a dense tensor again: pad, but only at the end."""
    model = build_model(tiny_model_dir)
    input_ids, prompt_mask, attention_mask, _ = mixed_length_batch([20, 12])

    hidden = model._encode(model.tokens_to_embeddings(input_ids), attention_mask)
    prefixes, masks, _ = model._compress_batch(hidden, prompt_mask, attention_mask)
    embeds, dense_mask = model._pad_compressed_batch(prefixes, masks)

    assert embeds.shape[0] == 2
    assert embeds.shape[1] == max(prefix.size(0) for prefix in prefixes)
    assert embeds.shape[:2] == dense_mask.shape
    for row, prefix in enumerate(prefixes):
        assert dense_mask[row, :prefix.size(0)].all()
        assert not dense_mask[row, prefix.size(0):].any()
        assert torch.equal(embeds[row, :prefix.size(0)], prefix)


def test_empty_answer_row_still_aligns(tmp_path):
    """A row whose answer is entirely padding must not shift its neighbour."""
    from test_model_integration import build_model

    model = build_model(_tiny_dir(tmp_path))
    labels = torch.tensor([[-100, -100], [9, 10]])
    _, attention, full_labels = model._build_training_batch(
        [make_prefix(6, 0), make_prefix(11, 1)],
        [torch.ones(6, dtype=torch.bool), torch.ones(11, dtype=torch.bool)],
        labels,
    )
    # row 0 contributes no supervised tokens at all
    assert first_answer_position(full_labels, 1) == 11
    assert full_labels[0].eq(-100).all()
    assert not attention[0, 6:].any()
    assert attention[0, :6].all()


def test_single_sample_batch_matches_the_previous_behaviour(tmp_path):
    """Batch size one was never affected; the fix must not change it either."""
    from test_model_integration import build_model

    model = build_model(_tiny_dir(tmp_path))
    labels = torch.tensor([[7, 8]])
    embeds, attention, full_labels = model._build_training_batch(
        [make_prefix(6, 0)], [torch.ones(6, dtype=torch.bool)], labels
    )
    assert embeds.size(0) == 1
    assert embeds.size(1) == 8                       # prefix 6 + answer 2, no pad
    assert attention.all()
    assert full_labels.tolist() == [[-100] * 6 + [7, 8]]


_TINY_DIR_CACHE = {}


def _tiny_dir(tmp_path):
    """Build (once) the tiny llama used by the shared fixture, outside pytest."""
    if "dir" in _TINY_DIR_CACHE:
        return _TINY_DIR_CACHE["dir"]
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

    directory = str(tmp_path / "tiny")
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct", use_fast=False
    )
    tokenizer.save_pretrained(directory)
    torch.manual_seed(0)
    LlamaForCausalLM(LlamaConfig(
        vocab_size=128256, hidden_size=HIDDEN, intermediate_size=HIDDEN * 2,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=4096, tie_word_embeddings=False,
    )).save_pretrained(directory)
    _TINY_DIR_CACHE["dir"] = directory
    return directory
