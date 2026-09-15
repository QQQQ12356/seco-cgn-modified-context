"""Route metadata stored next to a checkpoint.

Five compression routes can produce checkpoints with byte-identical tensor
shapes (legacy/multiscale x budget modes x selection modes), so a wrong route
argument loads silently and reports a number for an algorithm the weights were
never trained with. The route arguments are written to ``compressor_config.json``
on save and re-checked on load.
"""

import json
import os

CONFIG_NAME = "compressor_config.json"

# arguments that change what the compressor computes; a mismatch means the
# checkpoint does not describe the model being built
ROUTE_KEYS = (
    "compressor_version",
    "budget_mode",
    "legacy_budget_mode",
    "selection_mode",
    "relevance_dim",
    "query_slots",
    "raw_memory_fraction",
    "compress_ratio",
    "query_attention_mode",
    "query_mode",
    "diversity_mode",
    "merge_gate_mode",
    "context_boundary_mode",
    "context_budget_mode",
    "context_merge_mode",
    "context_novelty_weight",
    "context_boundary_radius",
    "context_anchor_weight",
    "allocation_block_width",
    "context_window_widths",
    "query_phrase_widths",
)


def inert_compressor_keys(compressor):
    """Compressor parameter names this configuration never reads.

    A later code version can add a parameter that the route it is running does
    not use -- the query attention temperature exists only for the ``cosine_tau``
    score. Requiring it to be present when loading a checkpoint trained before
    it existed would block exactly the "old weights, fixed code" evaluation the
    review asks to keep possible, while proving nothing about correctness.
    """
    if compressor is None:
        return set()
    inert = set()
    encoder = getattr(compressor, "query_encoder", None)
    if encoder is not None and getattr(encoder, "attention_mode", "dot") != "cosine_tau":
        inert.add("query_encoder.raw_attn_tau")
    return inert


def fatal_missing_compressor_weights(compressor, missing):
    """``missing`` keys that must be present for the reported metric to be real.

    A randomly initialised parameter the forward pass actually multiplies by is
    fatal; one the configuration never reads is not.
    """
    prefix = "compressor."
    inert = inert_compressor_keys(compressor)
    return [
        key for key in missing
        if key.startswith(prefix) and key[len(prefix):] not in inert
    ]


def write_config(directory, model_args, extra=None):
    config = {key: value for key, value in vars(model_args).items()
              if not key.startswith("_")}
    if extra:
        config.update(extra)
    path = os.path.join(directory, CONFIG_NAME)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2, default=str)
    return path


def load_config(checkpoint_path):
    """Read the route config sitting next to a checkpoint, or ``None``."""
    directory = os.path.dirname(os.path.abspath(checkpoint_path))
    path = os.path.join(directory, CONFIG_NAME)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def check_route(checkpoint_path, model_args):
    """Raise if the checkpoint was produced by a different compression route.

    A checkpoint with no config predates this file (legacy-only era) and is
    accepted, but a config that disagrees is always an error: the evaluation
    would silently measure the wrong algorithm.
    """
    saved = load_config(checkpoint_path)
    if getattr(model_args, "compressor_version", None) == "context_adaptive_v1":
        required = {"compressor_version", "allocation_block_width",
                    "context_window_widths", "query_phrase_widths"}
        required.update(key for key in ROUTE_KEYS if key.startswith("context_"))
        if saved is None or not required.issubset(saved):
            raise RuntimeError("context checkpoints require complete route metadata")
    if saved is None:
        return None
    mismatches = []
    for key in ROUTE_KEYS:
        if key not in saved:
            continue
        expected = getattr(model_args, key, None)
        if expected is None:
            continue
        if str(saved[key]) != str(expected):
            mismatches.append(f"{key}: checkpoint={saved[key]!r} but running={expected!r}")
    if mismatches:
        raise RuntimeError(
            f"{checkpoint_path} was produced by a different compression route:\n  "
            + "\n  ".join(mismatches)
            + "\nPass the matching route arguments, or the reported metric will not "
              "describe the algorithm these weights were trained with."
        )
    return saved
