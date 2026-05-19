"""Convert Nvidia Nemotron-PII dataset to SFT JSONL for leap-finetune.

Downloads from HuggingFace, replaces all PHI spans with [REDACTED],
formats as SFT messages, samples 50K rows stratified by domain, and
splits 80/20. By default this only saves local JSONL files; HuggingFace
uploads are opt-in and require an explicit repo ID.

Usage:
    python prepare_data.py
    python prepare_data.py --source nvidia/Nemotron-PII --split train --sample 50000
    python prepare_data.py --push --dataset-repo your-org/phi-redaction-sft
    PHI_FIREWALL_HF_DATASET_REPO=your-org/phi-redaction-sft python prepare_data.py --push
    python prepare_data.py --input /path/to/local.parquet
"""

import argparse
import ast
import json
import os
import random
from collections import defaultdict
from pathlib import Path

try:
    import pyarrow.parquet as pq
except ImportError:
    raise ImportError("pyarrow is required: pip install pyarrow")

SYSTEM_PROMPT = (
    "Replace all names, SSNs, DOBs, phone numbers, emails, addresses, "
    "medical record numbers, and IDs with [REDACTED]. Output only the "
    "redacted text, nothing else."
)

HF_SOURCE_DATASET = "nvidia/Nemotron-PII"
HF_SOURCE_SPLIT = "train"
HF_DATASET_REPO_ENV = "PHI_FIREWALL_HF_DATASET_REPO"
OUTPUT_DIR = Path(__file__).parent / "data"
SEED = 42
MAX_TEXT_CHARS = 2000


def parse_spans(spans_raw: str) -> list[dict]:
    try:
        return ast.literal_eval(spans_raw)
    except (ValueError, SyntaxError):
        return []


def redact_text(text: str, spans: list[dict]) -> str:
    sorted_spans = sorted(spans, key=lambda s: s["start"], reverse=True)
    result = text
    for span in sorted_spans:
        start = span["start"]
        end = span["end"]
        result = result[:start] + "[REDACTED]" + result[end:]
    return result


def build_example(text: str, spans: list[dict]) -> dict | None:
    if not spans:
        return None
    if len(text) > MAX_TEXT_CHARS:
        return None

    redacted = redact_text(text, spans)

    if redacted == text:
        return None

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
            {"role": "assistant", "content": redacted},
        ],
        "_domain": None,
    }


def stratified_sample(examples: list[dict], n: int, seed: int = SEED) -> list[dict]:
    rng = random.Random(seed)
    by_domain = defaultdict(list)
    for ex in examples:
        by_domain[ex["_domain"]].append(ex)

    domains = sorted(by_domain.keys())
    n_domains = len(domains)
    per_domain = n // n_domains
    remainder = n % n_domains

    sampled = []
    for i, domain in enumerate(domains):
        items = by_domain[domain]
        rng.shuffle(items)
        count = per_domain + (1 if i < remainder else 0)
        sampled.extend(items[:count])

    rng.shuffle(sampled)
    return sampled


def stratified_split(
    examples: list[dict], train_ratio: float = 0.8, seed: int = SEED
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    by_domain = defaultdict(list)
    for ex in examples:
        by_domain[ex["_domain"]].append(ex)

    train, test = [], []
    for domain in sorted(by_domain.keys()):
        items = by_domain[domain]
        rng.shuffle(items)
        n_train = round(len(items) * train_ratio)
        train.extend(items[:n_train])
        test.extend(items[n_train:])

    return train, test


def strip_meta(examples: list[dict]) -> list[dict]:
    return [{"messages": ex["messages"]} for ex in examples]


def save_jsonl(examples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Saved {len(examples)} examples to {path}")


def validate_example(ex: dict) -> bool:
    msgs = ex["messages"]
    if len(msgs) != 3:
        return False
    if msgs[0]["role"] != "system":
        return False
    if msgs[1]["role"] != "user":
        return False
    if msgs[2]["role"] != "assistant":
        return False
    if not msgs[1]["content"].strip():
        return False
    if not msgs[2]["content"].strip():
        return False
    return True


def load_table(source: str, split: str) -> "pyarrow.Table":
    print(f"Downloading {source} (split: {split}) from HuggingFace...")
    from datasets import load_dataset

    ds = load_dataset(source, split=split)
    print(f"Loaded {len(ds)} rows")
    table = ds.with_format("arrow").data.table
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare PHI redaction SFT dataset")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Local parquet file (overrides --source)",
    )
    parser.add_argument(
        "--source", type=str, default=HF_SOURCE_DATASET, help="HuggingFace dataset ID"
    )
    parser.add_argument(
        "--split", type=str, default=HF_SOURCE_SPLIT, help="Dataset split to use"
    )
    parser.add_argument("--sample", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--push", action="store_true", help="Upload the generated dataset to HuggingFace"
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Skip HuggingFace upload (default; retained for compatibility)",
    )
    parser.add_argument(
        "--dataset-repo",
        type=str,
        default=os.environ.get(HF_DATASET_REPO_ENV),
        help=(
            "HuggingFace dataset repo to push to, for example "
            f"'your-org/phi-redaction-sft'. Can also be set with {HF_DATASET_REPO_ENV}."
        ),
    )
    args = parser.parse_args()

    if args.input:
        print(f"Loading local file {args.input}...")
        table = pq.read_table(args.input)
        print(f"Loaded {table.num_rows} rows")
    else:
        table = load_table(args.source, args.split)

    print("Converting to SFT examples...")
    examples = []
    skipped = 0
    for i in range(table.num_rows):
        text = table.column("text")[i].as_py()
        spans_raw = table.column("spans")[i].as_py()
        domain = table.column("domain")[i].as_py()
        spans = parse_spans(spans_raw)

        ex = build_example(text, spans)
        if ex is None:
            skipped += 1
            continue
        ex["_domain"] = domain
        examples.append(ex)

    print(f"Converted {len(examples)} examples (skipped {skipped} rows)")

    print(f"Sampling {args.sample} rows stratified by domain...")
    if len(examples) < args.sample:
        print(f"Warning: only {len(examples)} valid examples, using all")
        sampled = examples
    else:
        sampled = stratified_sample(examples, args.sample, args.seed)

    domain_counts = defaultdict(int)
    for ex in sampled:
        domain_counts[ex["_domain"]] += 1
    print(f"Sampled {len(sampled)} examples across {len(domain_counts)} domains")
    for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {domain}: {count}")
    print(f"  ... ({len(domain_counts)} total domains)")

    print("Splitting 80/20...")
    train, test = stratified_split(sampled, train_ratio=0.8, seed=args.seed)
    print(f"Train: {len(train)}, Test: {len(test)}")

    train_clean = [ex for ex in strip_meta(train) if validate_example(ex)]
    test_clean = [ex for ex in strip_meta(test) if validate_example(ex)]
    print(f"After validation: Train: {len(train_clean)}, Test: {len(test_clean)}")

    print("Saving local copies...")
    save_jsonl(train_clean, OUTPUT_DIR / "train.jsonl")
    save_jsonl(test_clean, OUTPUT_DIR / "test.jsonl")

    should_push = args.push and not args.no_push
    if should_push and not args.dataset_repo:
        raise SystemExit(
            "Refusing to push without a HuggingFace dataset repo. "
            "Pass --dataset-repo your-org/phi-redaction-sft or set "
            f"{HF_DATASET_REPO_ENV}."
        )

    if should_push:
        try:
            from datasets import Dataset, DatasetDict

            print(f"\nUploading to HuggingFace Hub as '{args.dataset_repo}'...")
            ds = DatasetDict(
                {
                    "train": Dataset.from_list(train_clean),
                    "test": Dataset.from_list(test_clean),
                }
            )
            ds.push_to_hub(args.dataset_repo, private=False)
            print(
                f"Done. Dataset at: https://huggingface.co/datasets/{args.dataset_repo}"
            )
        except ImportError:
            print("\n'huggingface datasets' not installed. Skipping upload.")
            print("Install with: pip install datasets")
            print(
                f"Then run: python {' '.join([str(args.input)])} --sample {args.sample}"
            )
    else:
        print("\nSkipping HuggingFace upload. Pass --push with --dataset-repo to upload.")

    print("\nDone!")
    print(f"  Train: {len(train_clean)} examples -> {OUTPUT_DIR / 'train.jsonl'}")
    print(f"  Test:  {len(test_clean)} examples -> {OUTPUT_DIR / 'test.jsonl'}")


if __name__ == "__main__":
    main()
