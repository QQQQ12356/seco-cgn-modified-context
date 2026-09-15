"""Shared fixtures for the SECO test suite.

The tiny two-layer random Llama lives here rather than in one test module so
every test file that needs a real ``SECO`` forward can use it. The real
tokenizer is required because the model hard-codes ``bos_token_id`` as the
pooler seed initialiser.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TOKENIZER = "meta-llama/Llama-3.2-1B-Instruct"
HIDDEN = 64
VOCAB = 128256          # must cover bos_token_id = 128000


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    transformers = pytest.importorskip("transformers")
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, use_fast=False)
    except OSError as error:                       # pragma: no cover - env dependent
        pytest.skip(f"tokenizer unavailable: {error}")

    directory = tmp_path_factory.mktemp("tiny-llama")
    # SECO resolves both the model and the tokenizer from model_name_or_path
    tokenizer.save_pretrained(directory)
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=HIDDEN * 2,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=4096, tie_word_embeddings=False,
    )
    LlamaForCausalLM(config).save_pretrained(directory)
    return str(directory)
