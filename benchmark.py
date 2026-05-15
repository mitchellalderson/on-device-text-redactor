"""Modal-based benchmark: compare two fine-tuned model revisions on CUDA.

Downloads ONNX models from HuggingFace Hub, runs inference on GPU,
reports correctness & speed metrics.

Usage:
    modal run benchmark.py
    modal run benchmark.py --samples 20
    modal run benchmark.py --revisions main v2-50k
    modal run benchmark.py --output results.json
"""

import argparse
import json
import random
import re
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

import modal

HF_REPO = "aldersondev/phi-firewall-lfm2-350m-onnx"
SYSTEM_PROMPT = "Replace all names, SSNs, DOBs, phone numbers, emails, addresses, medical record numbers, and IDs with [REDACTED]. Output only the redacted text, nothing else."

CHATML_TEMPLATE = "{% for message in messages %}{% if message.role == 'system' %}<|im_start|>system\n{{ message.content }}<|im_end|>\n{% elif message.role == 'user' %}<|im_start|>user\n{{ message.content }}<|im_end|>\n{% elif message.role == 'assistant' %}<|im_start|>assistant\n{{ message.content }}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"

MAX_NEW_TOKENS = 1024
MAX_OUTPUT_MULTIPLIER = 2.5
TEMPERATURE = 0.1
TOP_K = 50
REPETITION_PENALTY = 1.05

app = modal.App("phi-firewall-benchmark")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-runtime-ubuntu22.04", add_python="3.12"
    )
    .pip_install(
        "onnxruntime-gpu==1.22.0",
        "transformers==4.52.4",
        "huggingface_hub",
        "numpy",
        "jinja2",
        "nvidia-cudnn-cu12==9.*",
    )
    .env(
        {
            "LD_LIBRARY_PATH": "/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib:/usr/local/cuda/lib64:/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib"
        }
    )
    .apt_install("wget")
    .add_local_file("training/data/test.jsonl", remote_path="/data/test.jsonl")
)


@app.function(image=image, gpu="A10G", timeout=4096)
def benchmark(
    revisions: list[str],
    samples: int,
    seed: int,
    failures: int,
    output_file: str | None,
):
    import numpy as np
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    def load_samples(n, seed_val):
        with open("/data/test.jsonl") as f:
            lines = f.readlines()
        random.seed(seed_val)
        random.shuffle(lines)
        out = []
        for line in lines[:n]:
            msgs = json.loads(line)["messages"]
            user_msg = assistant_msg = ""
            for m in msgs:
                if m["role"] == "user":
                    user_msg = m["content"]
                elif m["role"] == "assistant":
                    assistant_msg = m["content"]
            out.append({"input": user_msg, "expected": assistant_msg})
        return out

    def load_tokenizer(rev):
        from huggingface_hub import hf_hub_download

        tok_path = hf_hub_download(
            repo_id=HF_REPO, filename="tokenizer.json", revision=rev
        )
        config_path = hf_hub_download(
            repo_id=HF_REPO, filename="tokenizer_config.json", revision=rev
        )
        tok = PreTrainedTokenizerFast(tokenizer_file=tok_path)
        with open(config_path) as f:
            cfg = json.load(f)
        tok.eos_token_id = cfg.get("eos_token_id", 2)
        tok.chat_template = CHATML_TEMPLATE
        return tok

    def build_prompt(tok, text):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
        return tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

    def sample_token(logits, generated):
        logits = logits.copy().astype(np.float32)
        seen = set(generated)
        for tid in seen:
            if logits[tid] > 0:
                logits[tid] /= REPETITION_PENALTY
            else:
                logits[tid] *= REPETITION_PENALTY
        logits /= TEMPERATURE
        top_k_idx = np.argsort(logits)[-TOP_K:]
        top_k_logits = logits[top_k_idx]
        exps = np.exp(top_k_logits - top_k_logits.max())
        probs = exps / exps.sum()
        return int(top_k_idx[np.random.choice(len(top_k_idx), p=probs)])

    def run_inference(session, tok, input_text, cache_spec):
        cache = {}
        for name, shape, dtype in cache_spec:
            cache[name] = np.zeros(shape, dtype=dtype)

        prompt = build_prompt(tok, input_text)
        input_ids = tok.encode(prompt)
        generated = []
        all_ids = list(input_ids)

        t0 = time.perf_counter()
        for step in range(MAX_NEW_TOKENS):
            ids = all_ids if step == 0 else [generated[-1]]
            feeds = {
                "input_ids": np.array([ids], dtype=np.int64),
                "attention_mask": np.ones((1, len(all_ids)), dtype=np.int64),
            }
            for k, v in cache.items():
                feeds[k] = v

            raw_outputs = session.run(None, feeds)
            logits_out = raw_outputs[0]

            for idx, out_meta in enumerate(session.get_outputs()):
                name = out_meta.name
                if name == "logits":
                    continue
                cache_name = name.replace("present_conv", "past_conv").replace(
                    "present_key_values", "past_key_values"
                )
                if cache_name in cache:
                    cache[cache_name] = raw_outputs[idx]

            last_logits = logits_out[0, -1, :]
            next_token = sample_token(last_logits, generated)
            generated.append(next_token)

            if next_token == tok.eos_token_id:
                if len(generated) > 5:
                    break
                generated.pop()
                continue

            all_ids.append(next_token)
            if len(generated) > 10:
                decoded = tok.decode(generated, skip_special_tokens=True)
                if len(decoded) > len(input_text) * MAX_OUTPUT_MULTIPLIER:
                    break

        t1 = time.perf_counter()
        return tok.decode(generated, skip_special_tokens=True), t1 - t0, len(generated)

    def redacted_positions(text):
        positions = set()
        for m in re.finditer(r"\[REDACTED\]", text):
            for i in range(m.start(), m.end()):
                positions.add(i)
        return positions

    def char_f1(pred, ref):
        matcher = SequenceMatcher(None, pred, ref)
        matching = sum(b.size for b in matcher.get_matching_blocks())
        if matching == 0:
            return 0.0
        p = matching / len(pred) if pred else 0.0
        r = matching / len(ref) if ref else 0.0
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def redacted_recall(pred, ref):
        rp = redacted_positions(ref)
        if not rp:
            return 1.0
        pp = redacted_positions(pred)
        return len(rp & pp) / len(rp) if pp else 0.0

    def redacted_precision(pred, ref):
        pp = redacted_positions(pred)
        if not pp:
            return 1.0 if not redacted_positions(ref) else 0.0
        return len(pp & redacted_positions(ref)) / len(pp)

    def build_cache_spec(session):
        spec = []
        for inp in session.get_inputs():
            name = inp.name
            if name in ("input_ids", "attention_mask", "num_logits_to_keep"):
                continue
            shape = []
            for d in inp.shape:
                if isinstance(d, int):
                    shape.append(d)
                else:
                    shape.append(0)
            dtype = np.float16 if "float16" in str(inp.type) else np.float32
            spec.append((name, shape, dtype))
        return spec

    def run_bench(session, tok, data, label):
        cache_spec = build_cache_spec(session)
        results = []
        total_time = 0
        total_tokens = 0
        print(f"\n  Running {label} on {len(data)} samples...")
        for i, sample in enumerate(data):
            output, wall, ntok = run_inference(
                session, tok, sample["input"], cache_spec
            )
            metrics = {
                "exact_match": output.strip() == sample["expected"].strip(),
                "char_f1": char_f1(output, sample["expected"]),
                "redacted_recall": redacted_recall(output, sample["expected"]),
                "redacted_precision": redacted_precision(output, sample["expected"]),
            }
            metrics["input"] = sample["input"][:200]
            metrics["expected"] = sample["expected"][:200]
            metrics["predicted"] = output[:200]
            metrics["tokens"] = ntok
            metrics["wall_time"] = wall
            metrics["tps"] = ntok / wall if wall > 0 else 0
            results.append(metrics)
            total_time += wall
            total_tokens += ntok
            if (i + 1) % 10 == 0:
                print(
                    f"    {i + 1}/{len(data)} done ({(i + 1) / len(data) * 100:.0f}%)"
                )

        n = len(results)
        return {
            "label": label,
            "n": n,
            "exact_match_rate": sum(1 for r in results if r["exact_match"]) / n,
            "avg_char_f1": sum(r["char_f1"] for r in results) / n,
            "avg_redacted_recall": sum(r["redacted_recall"] for r in results) / n,
            "avg_redacted_precision": sum(r["redacted_precision"] for r in results) / n,
            "avg_tps": total_tokens / total_time if total_time > 0 else 0,
            "avg_wall_time": total_time / n,
            "total_time": total_time,
            "per_sample": results,
        }

    print(f"  GPU: A10G (CUDA)")
    print(f"  Revisions: {revisions[0]} vs {revisions[1]}")
    print(f"  Samples: {samples}  |  Seed: {seed}")

    np.random.seed(seed)

    print(f"\n  Loading test data...")
    data = load_samples(samples, seed)
    print(f"  Loaded {len(data)} samples")

    rev_a, rev_b = revisions

    print(f"\n  Downloading {rev_a}...")
    tok_a = load_tokenizer(rev_a)
    onnx_a = hf_hub_download(
        repo_id=HF_REPO, filename="model_fp16.onnx", revision=rev_a
    )
    sess_a = ort.InferenceSession(onnx_a, providers=["CUDAExecutionProvider"])
    print(f"    {rev_a} ready ({sess_a.get_providers()[0]})")

    print(f"  Downloading {rev_b}...")
    tok_b = load_tokenizer(rev_b)
    onnx_b = hf_hub_download(
        repo_id=HF_REPO, filename="model_fp16.onnx", revision=rev_b
    )
    sess_b = ort.InferenceSession(onnx_b, providers=["CUDAExecutionProvider"])
    print(f"    {rev_b} ready ({sess_b.get_providers()[0]})")

    results_a = run_bench(sess_a, tok_a, data, rev_a)
    del sess_a
    results_b = run_bench(sess_b, tok_b, data, rev_b)
    del sess_b

    w = 59
    col1 = rev_a[:14].center(14)
    col2 = rev_b[:14].center(14)
    print()
    print("=" * w)
    print(f"  BENCHMARK RESULTS  (N={results_a['n']})")
    print("=" * w)
    print(f"  {'Metric':<20}{col1:>18}{col2:>18}")
    print("-" * w)

    for label, v1, v2 in [
        (
            "Exact Match",
            f"{results_a['exact_match_rate'] * 100:.1f}%",
            f"{results_b['exact_match_rate'] * 100:.1f}%",
        ),
        (
            "Char F1",
            f"{results_a['avg_char_f1']:.3f}",
            f"{results_b['avg_char_f1']:.3f}",
        ),
        (
            "REDACTED Recall",
            f"{results_a['avg_redacted_recall']:.3f}",
            f"{results_b['avg_redacted_recall']:.3f}",
        ),
        (
            "REDACTED Precision",
            f"{results_a['avg_redacted_precision']:.3f}",
            f"{results_b['avg_redacted_precision']:.3f}",
        ),
        (
            "Avg Tokens/sec",
            f"{results_a['avg_tps']:.1f}",
            f"{results_b['avg_tps']:.1f}",
        ),
        (
            "Avg Time (s)",
            f"{results_a['avg_wall_time']:.2f}",
            f"{results_b['avg_wall_time']:.2f}",
        ),
    ]:
        print(f"  {label:<20}{v1:>18}{v2:>18}")

    print("-" * w)
    print(
        f"  {'Total Time (s)':<20}{results_a['total_time']:>18.1f}{results_b['total_time']:>18.1f}"
    )
    print("=" * w)

    if failures > 0:
        print()
        print("=" * w)
        print("  FAILURE ANALYSIS")
        print("=" * w)
        for res in [results_b, results_a]:
            worst = sorted(res["per_sample"], key=lambda r: r["char_f1"])[:failures]
            print(f"\n  Worst {failures} for {res['label']}:")
            print("  " + "-" * 55)
            for rank, r in enumerate(worst, 1):
                print(
                    f"  #{rank}  F1: {r['char_f1']:.3f}  R/P: {r['redacted_recall']:.2f}/{r['redacted_precision']:.2f}"
                )
                print(f"      Input:    {r['input'][:120]}...")
                print(f"      Expected: {r['expected'][:120]}...")
                print(f"      Got:      {r['predicted'][:120]}...")
                print()

    if output_file:
        with open(output_file, "w") as f:
            json.dump(
                {
                    "a": results_a,
                    "b": results_b,
                    "config": {
                        "samples": samples,
                        "seed": seed,
                        "revisions": revisions,
                    },
                },
                f,
                indent=2,
            )
        print(f"  Results saved to {output_file}")


@app.local_entrypoint()
def main(
    rev_a: str = "main",
    rev_b: str = "v2-50k",
    samples: int = 100,
    seed: int = 42,
    failures: int = 3,
    output: str = None,
):
    benchmark.remote(
        revisions=[rev_a, rev_b],
        samples=samples,
        seed=seed,
        failures=failures,
        output_file=output,
    )
