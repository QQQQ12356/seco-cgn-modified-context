"""Stable identity for an evaluation row.

Several scripts need to align the same question across runs: the paired
comparison, the frozen dev split, and the inference output. Deriving the id from
``(subset, question, context)`` rather than from the row index keeps that
alignment correct when the file is shuffled, when a run samples a different
subset, and when a duplicate row appears at two positions -- which the historical
2000-row evaluation does.
"""

import hashlib

ID_SEPARATOR = "\x1f"       # unit separator: cannot occur in the source text


def stable_sample_id(sample):
    """Short hash of a row's identity, or its explicit ``sample_id`` if present."""
    if sample.get("sample_id"):
        return sample["sample_id"]
    payload = ID_SEPARATOR.join(
        [
            str(sample.get("subset", "")),
            str(sample.get("prompt", "")),
            str(sample.get("input", "")),
        ]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
