#!/usr/bin/env python3
"""Benchmark base vs fine-tuned LFM2.5-350M on PHI redaction correctness & speed."""

import argparse
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from difflib import SequenceMatcher

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
MERGED_PATH = SCRIPT_DIR / "output" / "merged"
DATA_PATH = SCRIPT_DIR / "data" / "test.jsonl"
BASE_MODEL_ID = "LiquidAI/LFM2.5-350M"

TRAINED_SYSTEM = "Replace all names, SSNs, DOBs, phone numbers, emails, addresses, medical record numbers, and IDs with [REDACTED]. Output only the redacted text, nothing else."
BASE_SYSTEM = "Repeat the following text exactly as given. Sensitive information, or personal data should be replaced with [REDACTED]."

TRAINED_GEN_KWARGS = dict(
    max_new_tokens=1024,
    temperature=0.1,
    top_k=50,
    repetition_penalty=1.05,
    do_sample=True,
)

BASE_GEN_KWARGS = dict(
    max_new_tokens=1024,
    temperature=0.9,
    top_k=100,
    repetition_penalty=1.6,
    do_sample=True,
)


def load_samples(path: Path, n: int, seed: int) -> list[dict]:
    with open(path) as f:
        lines = f.readlines()
    random.seed(seed)
    random.shuffle(lines)
    samples = []
    for line in lines[:n]:
        msgs = json.loads(line)["messages"]
        user_msg = assistant_msg = ""
        for m in msgs:
            if m["role"] == "user":
                user_msg = m["content"]
            elif m["role"] == "assistant":
                assistant_msg = m["content"]
        samples.append({"input": user_msg, "expected": assistant_msg})
    return samples


def build_prompt(tokenizer, system: str, user_text: str) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def run_inference(model, tokenizer, prompt: str, gen_kwargs: dict) -> tuple[str, float, float, int]:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    torch.mps.synchronize() if torch.backends.mps.is_available() else None
    t0 = time.perf_counter()

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    torch.mps.synchronize() if torch.backends.mps.is_available() else None
    t1 = time.perf_counter()

    generated = output_ids[0, input_len:]
    decoded = tokenizer.decode(generated, skip_special_tokens=True)
    num_tokens = len(generated)
    total_time = t1 - t0

    return decoded, total_time, num_tokens


def redacted_spans(text: str) -> list[str]:
    return re.findall(r'\[REDACTED\]', text)


def redacted_positions(text: str) -> set[int]:
    positions = set()
    pattern = re.compile(r'\[REDACTED\]')
    for m in pattern.finditer(text):
        for i in range(m.start(), m.end()):
            positions.add(i)
    return positions


def char_f1(pred: str, ref: str) -> float:
    matcher = SequenceMatcher(None, pred, ref)
    matching = sum(block.size for block in matcher.get_matching_blocks())
    if matching == 0:
        return 0.0
    precision = matching / len(pred) if pred else 0.0
    recall = matching / len(ref) if ref else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def redacted_recall(pred: str, ref: str) -> float:
    ref_positions = redacted_positions(ref)
    if not ref_positions:
        return 1.0
    pred_positions = redacted_positions(pred)
    if not pred_positions:
        return 0.0
    overlap = len(ref_positions & pred_positions)
    return overlap / len(ref_positions)


def redacted_precision(pred: str, ref: str) -> float:
    pred_positions = redacted_positions(pred)
    if not pred_positions:
        return 1.0 if not redacted_positions(ref) else 0.0
    ref_positions = redacted_positions(ref)
    overlap = len(pred_positions & ref_positions)
    return overlap / len(pred_positions)


def evaluate_output(pred: str, ref: str) -> dict:
    return {
        "exact_match": pred.strip() == ref.strip(),
        "char_f1": char_f1(pred, ref),
        "redacted_recall": redacted_recall(pred, ref),
        "redacted_precision": redacted_precision(pred, ref),
    }


def fmt_pct(val: float) -> str:
    return f"{val * 100:.1f}%"


def fmt_float(val: float) -> str:
    return f"{val:.3f}"


def fmt_speed(val: float) -> str:
    return f"{val:.1f}"


def run_benchmark(model, tokenizer, samples: list[dict], system: str, gen_kwargs: dict, label: str) -> dict:
    results = []
    total_time = 0
    total_tokens = 0

    print(f"\n  Running {label} on {len(samples)} samples...")
    for i, sample in enumerate(samples):
        prompt = build_prompt(tokenizer, system, sample["input"])
        output, wall_time, num_tokens = run_inference(model, tokenizer, prompt, gen_kwargs)

        metrics = evaluate_output(output, sample["expected"])
        metrics["input"] = sample["input"][:200]
        metrics["expected"] = sample["expected"][:200]
        metrics["predicted"] = output[:200]
        metrics["tokens"] = num_tokens
        metrics["wall_time"] = wall_time
        metrics["tps"] = num_tokens / wall_time if wall_time > 0 else 0
        results.append(metrics)

        total_time += wall_time
        total_tokens += num_tokens

        if (i + 1) % 10 == 0:
            print(f"    {i + 1}/{len(samples)} done ({(i + 1) / len(samples) * 100:.0f}%)")

    exact_matches = sum(1 for r in results if r["exact_match"])
    avg_char_f1 = sum(r["char_f1"] for r in results) / len(results)
    avg_recall = sum(r["redacted_recall"] for r in results) / len(results)
    avg_precision = sum(r["redacted_precision"] for r in results) / len(results)
    avg_tps = total_tokens / total_time if total_time > 0 else 0
    avg_wall = total_time / len(results)

    return {
        "label": label,
        "n": len(results),
        "exact_match_rate": exact_matches / len(results),
        "avg_char_f1": avg_char_f1,
        "avg_redacted_recall": avg_recall,
        "avg_redacted_precision": avg_precision,
        "avg_tps": avg_tps,
        "avg_wall_time": avg_wall,
        "total_time": total_time,
        "per_sample": results,
    }


def print_table(base: dict, trained: dict):
    w = 55
    print()
    print("=" * w)
    print(f"  BENCHMARK RESULTS  (N={base['n']})")
    print("=" * w)
    header = f"{'Metric':<22}{'Base':>14}{'Fine-Tuned':>14}"
    print(header)
    print("-" * w)

    rows = [
        ("Exact Match", fmt_pct(base["exact_match_rate"]), fmt_pct(trained["exact_match_rate"])),
        ("Char F1", fmt_float(base["avg_char_f1"]), fmt_float(trained["avg_char_f1"])),
        ("REDACTED Recall", fmt_float(base["avg_redacted_recall"]), fmt_float(trained["avg_redacted_recall"])),
        ("REDACTED Precision", fmt_float(base["avg_redacted_precision"]), fmt_float(trained["avg_redacted_precision"])),
        ("Avg Tokens/sec", fmt_speed(base["avg_tps"]), fmt_speed(trained["avg_tps"])),
        ("Avg Time (s)", fmt_float(base["avg_wall_time"]), fmt_float(trained["avg_wall_time"])),
    ]

    for label, bv, tv in rows:
        print(f"  {label:<20}{bv:>14}{tv:>14}")

    print("-" * w)
    print(f"  {'Total Time (s)':<20}{base['total_time']:>14.1f}{trained['total_time']:>14.1f}")
    print("=" * w)
    print()


def print_failure_analysis(results: list[dict], label: str, n: int = 3):
    scored = [(r["char_f1"], r) for r in results]
    scored.sort(key=lambda x: x[0])
    worst = scored[:n]

    print(f"\n  Worst {n} samples for {label}:")
    print("  " + "-" * 51)
    for rank, (score, r) in enumerate(worst, 1):
        print(f"  #{rank}  Char F1: {score:.3f}  |  REDACTED R/P: {r['redacted_recall']:.2f}/{r['redacted_precision']:.2f}")
        print(f"      Input:    {r['input'][:120]}...")
        print(f"      Expected: {r['expected'][:120]}...")
        print(f"      Got:      {r['predicted'][:120]}...")
        print()


def main():
    parser = argparse.ArgumentParser(description="Benchmark base vs fine-tuned PHI redaction model")
    parser.add_argument("--samples", type=int, default=100, help="Number of test samples (default: 100)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sample selection")
    parser.add_argument("--output", type=str, default=None, help="Save detailed results to JSON file")
    parser.add_argument("--failures", type=int, default=3, help="Number of worst samples to show (default: 3)")
    args = parser.parse_args()

    if not DATA_PATH.exists():
        print(f"Error: test data not found at {DATA_PATH}", file=sys.stderr)
        print("Run `npm run prepare:data` first.", file=sys.stderr)
        sys.exit(1)

    if not MERGED_PATH.exists():
        print(f"Error: merged model not found at {MERGED_PATH}", file=sys.stderr)
        print("Run `npm run export:model` first.", file=sys.stderr)
        sys.exit(1)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"  Device: {device}")
    print(f"  Samples: {args.samples}  |  Seed: {args.seed}")

    samples = load_samples(DATA_PATH, args.samples, args.seed)
    print(f"  Loaded {len(samples)} samples from {DATA_PATH.name}")

    print("\n  Loading base model...")
    base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, trust_remote_code=True, torch_dtype=torch.float32,
    ).to(device)
    base_model.eval()
    print(f"  Base model loaded ({sum(p.numel() for p in base_model.parameters()) / 1e6:.1f}M params)")

    print("  Loading fine-tuned model...")
    ft_tokenizer = AutoTokenizer.from_pretrained(str(MERGED_PATH), trust_remote_code=True)
    ft_model = AutoModelForCausalLM.from_pretrained(
        str(MERGED_PATH), trust_remote_code=True, torch_dtype=torch.float32,
    ).to(device)
    ft_model.eval()
    print(f"  Fine-tuned model loaded ({sum(p.numel() for p in ft_model.parameters()) / 1e6:.1f}M params)")

    base_results = run_benchmark(base_model, base_tokenizer, samples, BASE_SYSTEM, BASE_GEN_KWARGS, "Base Model")
    ft_results = run_benchmark(ft_model, ft_tokenizer, samples, TRAINED_SYSTEM, TRAINED_GEN_KWARGS, "Fine-Tuned Model")

    print_table(base_results, ft_results)

    if args.failures > 0:
        print("=" * 55)
        print("  FAILURE ANALYSIS")
        print("=" * 55)
        print_failure_analysis(ft_results["per_sample"], "Fine-Tuned", args.failures)
        print_failure_analysis(base_results["per_sample"], "Base", args.failures)

    if args.output:
        out_path = Path(args.output)
        with open(out_path, "w") as f:
            json.dump({"base": base_results, "trained": ft_results, "config": {"samples": args.samples, "seed": args.seed, "device": str(device)}}, f, indent=2)
        print(f"  Detailed results saved to {out_path}")

    del base_model
    del ft_model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
