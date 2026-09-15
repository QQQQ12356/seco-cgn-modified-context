"""Normalize QA JSONL and create context-disjoint train/dev partitions."""

import argparse
import hashlib
import json
from pathlib import Path


def normalize(row):
    context = row.get("input", row.get("context"))
    question = row.get("prompt", row.get("question"))
    answers = row.get("answer", row.get("answers"))
    if isinstance(answers, dict):
        answers = answers.get("text")
    if isinstance(answers, str):
        answers = [answers]
    if not isinstance(context, str) or not context.strip():
        raise ValueError("missing nonempty input/context")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("missing nonempty prompt/question")
    if not isinstance(answers, list) or not answers or any(
        not isinstance(answer, str) or not answer.strip() for answer in answers
    ):
        raise ValueError("answer/answers must contain nonempty strings")
    result = dict(row)
    result.update(input=context, prompt=question, answer=answers)
    return result


def partition(row, seed, dev_fraction):
    context = " ".join(row["input"].split())
    digest = hashlib.sha256(f"{seed}\0{context}".encode()).digest()
    return "dev" if int.from_bytes(digest[:8], "big") / 2**64 < dev_fraction else "train"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-fraction", type=float, default=0.1)
    args = parser.parse_args()
    if not 0 < args.dev_fraction < 1:
        parser.error("--dev-fraction must lie in (0, 1)")
    rows = {"train": [], "dev": []}
    identities = set()
    with args.input.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = normalize(json.loads(line))
            except (ValueError, TypeError, AttributeError) as error:
                raise ValueError(f"{args.input}:{line_number}: {error}") from error
            identity = (row["input"], row["prompt"], row.get("subset", ""))
            if identity in identities:
                continue
            identities.add(identity)
            rows[partition(row, args.seed, args.dev_fraction)].append(row)
    if any(not split for split in rows.values()):
        raise ValueError("empty train/dev split; provide more contexts or change seed/fraction")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, split in rows.items():
        with (args.output_dir / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in split:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "source": str(args.input.resolve()),
        "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "seed": args.seed, "dev_fraction": args.dev_fraction,
        "counts": {name: len(split) for name, split in rows.items()},
        "policy": "normalized-context-disjoint hash split; training source only",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
