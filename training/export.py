"""Merge LoRA adapter into base model, export to ONNX, and push to HuggingFace.

Custom ONNX export for LFM2 hybrid architecture (conv + attention layers).

Two export modes:
1. Simple (default): input_ids -> logits (no KV cache, simpler export)
2. KV cache: input_ids + past states -> logits + present states (matches LiquidAI format)

Usage:
    # Merge + export simple FP16
    python export.py --lora-path output/lora-run --output-path output/merged --export-onnx

    # Merge + export + quantize
    python export.py --lora-path output/lora-run --output-path output/merged \\
        --export-onnx --quantize

    # Merge + export + push to HF (versioned)
    python export.py --lora-path output/lora-run --output-path output/merged \\
        --export-onnx --push-to-hub --tag v1-5k

    # Skip merge (already merged)
    python export.py --lora-path output/lora-run --output-path output/merged \\
        --export-onnx --skip-merge

Requirements:
    pip install peft transformers torch onnxscript onnxruntime
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

HF_MODEL_REPO = "aldersondev/phi-firewall-lfm2-350m-onnx"


def find_best_checkpoint(lora_path: Path) -> Path:
    if (lora_path / "adapter_config.json").exists():
        return lora_path

    candidates = [
        d
        for d in sorted(lora_path.iterdir())
        if d.is_dir()
        and (d / "adapter_config.json").exists()
        and (d / "trainer_state.json").exists()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoints with adapter_config.json found under {lora_path}"
        )

    best_path, best_loss = None, float("inf")
    print("Evaluating checkpoints:")
    for ckpt in candidates:
        with open(ckpt / "trainer_state.json") as f:
            state = json.load(f)
        eval_entries = [e for e in state["log_history"] if "eval_loss" in e]
        if not eval_entries:
            print(f"  {ckpt.name}: no eval_loss found, skipping")
            continue
        loss = eval_entries[-1]["eval_loss"]
        epoch = eval_entries[-1]["epoch"]
        print(f"  {ckpt.name}: eval_loss={loss:.4f} (epoch {epoch:.0f})")
        if loss < best_loss:
            best_loss = loss
            best_path = ckpt

    if best_path is None:
        print("No checkpoints with eval_loss found, using first candidate")
        best_path = candidates[0]
    else:
        print(f"\nSelected: {best_path.name} (eval_loss={best_loss:.4f})")
    return best_path


def merge_lora(lora_path: Path, output_path: Path) -> None:
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer

    print(f"\nLoading LoRA adapter from {lora_path}...")
    model = AutoPeftModelForCausalLM.from_pretrained(
        str(lora_path),
        device_map="cpu",
        torch_dtype="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(str(lora_path))

    print("Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {output_path}...")
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    print("Merge complete.")


class SimpleWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        return self.model(input_ids, use_cache=False).logits


class KVCacheWrapper(torch.nn.Module):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.num_layers = config["num_hidden_layers"]
        self.layer_types = config["layer_types"]
        self.hidden_size = config["hidden_size"]
        self.num_kv_heads = config["num_key_value_heads"]
        self.head_dim = config["hidden_size"] // config["num_attention_heads"]
        self.conv_L_cache = config.get("conv_L_cache", 3)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        position_ids: torch.LongTensor,
        *flat_caches,
    ) -> tuple:
        from transformers.models.lfm2.modeling_lfm2 import Lfm2HybridConvCache

        pkv = Lfm2HybridConvCache.__new__(Lfm2HybridConvCache)
        pkv.key_cache = []
        pkv.value_cache = []
        pkv.conv_cache = []
        pkv.max_batch_size = 1
        pkv.layer_types = self.layer_types
        pkv.first_attention_layer = self.layer_types.index("full_attention")
        pkv.conv_L_cache = self.conv_L_cache
        pkv._dtype = torch.float16

        flat_idx = 0
        for i in range(self.num_layers):
            if self.layer_types[i] == "full_attention":
                pkv.conv_cache.append(flat_caches[flat_idx])
                flat_idx += 1
                pkv.key_cache.append(flat_caches[flat_idx])
                flat_idx += 1
                pkv.value_cache.append(flat_caches[flat_idx])
                flat_idx += 1
            else:
                pkv.conv_cache.append(flat_caches[flat_idx])
                flat_idx += 1
                pkv.key_cache.append(torch.zeros(1, self.num_kv_heads, 0, self.head_dim, dtype=torch.float16))
                pkv.value_cache.append(torch.zeros(1, self.num_kv_heads, 0, self.head_dim, dtype=torch.float16))

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=pkv,
            use_cache=True,
        )

        result = [outputs.logits]
        for i in range(self.num_layers):
            result.append(outputs.past_key_values.conv_cache[i])
            if self.layer_types[i] == "full_attention":
                result.append(outputs.past_key_values.key_cache[i])
                result.append(outputs.past_key_values.value_cache[i])

        return tuple(result)


def export_onnx_simple(merged_path: Path, onnx_path: Path) -> None:
    import onnx
    from onnx.external_data_helper import load_external_data_for_model
    from transformers import AutoModelForCausalLM

    onnx_path.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading merged model from {merged_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        str(merged_path),
        trust_remote_code=True,
        local_files_only=True,
        device_map="cpu",
        torch_dtype=torch.float16,
    )
    model.eval()

    wrapper = SimpleWrapper(model)
    wrapper.eval()

    input_ids = torch.randint(0, 100, (1, 5), dtype=torch.long)

    print("\nExporting to ONNX (FP16, no KV cache)...")

    onnx_program = torch.onnx.export(
        wrapper,
        (input_ids,),
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"},
        },
        opset_version=18,
        dynamo=True,
        do_constant_folding=True,
        external_data=True,
    )

    if onnx_program is not None:
        tmp_path = onnx_path / "model_fp16_tmp.onnx"
        onnx_program.save(str(tmp_path))
        print(f"FP16 exported (with external data) to {tmp_path}")

        print("Inlining external data into standalone file...")
        onnx_model = onnx.load(str(tmp_path), load_external_data=True)
        standalone_path = onnx_path / "model_fp16.onnx"
        onnx.save_model(
            onnx_model,
            str(standalone_path),
            save_as_external_data=False,
        )
        print(f"Standalone FP16 ONNX: {standalone_path}")
        tmp_path.unlink()
        if (onnx_path / "model_fp16_tmp.onnx.data").exists():
            (onnx_path / "model_fp16_tmp.onnx.data").unlink()
    else:
        raise RuntimeError("ONNX export returned None")

    _print_dir_sizes(onnx_path)


def export_onnx_kv_cache(merged_path: Path, onnx_path: Path) -> None:
    import onnx
    from onnx.external_data_helper import load_external_data_for_model
    from transformers import AutoModelForCausalLM
    import json

    onnx_path.mkdir(parents=True, exist_ok=True)

    with open(merged_path / "config.json") as f:
        config = json.load(f)

    print(f"\nLoading merged model from {merged_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        str(merged_path),
        trust_remote_code=True,
        local_files_only=True,
        device_map="cpu",
        torch_dtype=torch.float16,
    )
    model.eval()

    wrapper = KVCacheWrapper(model, config)
    wrapper.eval()

    num_layers = config["num_hidden_layers"]
    layer_types = config["layer_types"]
    hidden_size = config["hidden_size"]
    num_kv_heads = config["num_key_value_heads"]
    head_dim = hidden_size // config["num_attention_heads"]
    conv_L_cache = config.get("conv_L_cache", 3)

    seq_len = 5

    dummy_inputs = {
        "input_ids": torch.randint(0, 100, (1, seq_len), dtype=torch.long),
        "attention_mask": torch.ones((1, seq_len), dtype=torch.long),
        "position_ids": torch.arange(seq_len, dtype=torch.long).unsqueeze(0),
    }

    input_names = ["input_ids", "attention_mask", "position_ids"]
    dummy_list = [dummy_inputs["input_ids"], dummy_inputs["attention_mask"], dummy_inputs["position_ids"]]
    for i in range(num_layers):
        if layer_types[i] == "full_attention":
            input_names.extend([f"past_conv_{i}", f"past_key_{i}", f"past_value_{i}"])
            dummy_list.extend([
                torch.zeros(1, hidden_size, conv_L_cache, dtype=torch.float16),
                torch.zeros(1, num_kv_heads, 1, head_dim, dtype=torch.float16),
                torch.zeros(1, num_kv_heads, 1, head_dim, dtype=torch.float16),
            ])
        else:
            input_names.append(f"past_conv_{i}")
            dummy_list.append(torch.zeros(1, hidden_size, conv_L_cache, dtype=torch.float16))
    output_names = ["logits"]
    for i in range(num_layers):
        output_names.append(f"present_conv_{i}")
        if layer_types[i] == "full_attention":
            output_names.append(f"present.{i}.key")
            output_names.append(f"present.{i}.value")

    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "total_seq_len"},
        "position_ids": {0: "batch_size", 1: "seq_len"},
        "logits": {0: "batch_size", 1: "seq_len"},
    }
    for i in range(num_layers):
        dynamic_axes[f"past_conv_{i}"] = {0: "batch_size"}
        dynamic_axes[f"present_conv_{i}"] = {0: "batch_size"}
        if layer_types[i] == "full_attention":
            dynamic_axes[f"past_key_{i}"] = {0: "batch_size", 2: "past_seq_len"}
            dynamic_axes[f"past_value_{i}"] = {0: "batch_size", 2: "past_seq_len"}
            dynamic_axes[f"present.{i}.key"] = {0: "batch_size", 2: "total_seq_len"}
            dynamic_axes[f"present.{i}.value"] = {0: "batch_size", 2: "total_seq_len"}

    input_order = tuple(dummy_list)

    print("\nExporting to ONNX (FP16, with KV cache)...")
    print(f"  {len(input_names)} inputs, {len(output_names)} outputs")

    from torch.onnx import register_custom_op_symbolic

    def copy_symbolic(g, self, src, non_blocking=False):
        return g.op("Identity", src)

    register_custom_op_symbolic("aten::copy", copy_symbolic, 18)

    tmp_path = onnx_path / "model_fp16_tmp.onnx"
    onnx_program = torch.onnx.export(
        wrapper,
        input_order,
        str(tmp_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=18,
        dynamo=False,
        do_constant_folding=True,
    )

    print(f"Traced and saved to {tmp_path}")
    print("Inlining external data into standalone file...")
    onnx_model = onnx.load(str(tmp_path), load_external_data=True)
    standalone_path = onnx_path / "model_fp16.onnx"
    onnx.save_model(
        onnx_model,
        str(standalone_path),
        save_as_external_data=False,
    )
    print(f"Standalone FP16 ONNX: {standalone_path}")
    tmp_path.unlink()
    for ext in [".data", "_data"]:
        p = onnx_path / f"model_fp16_tmp.onnx{ext}"
        if p.exists():
            p.unlink()

    _print_dir_sizes(onnx_path)


def quantize_onnx(onnx_path: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    src = onnx_path / "model_fp16.onnx"
    if not src.exists():
        for f in onnx_path.glob("*.onnx"):
            src = f
            break

    if not src.exists():
        print("No ONNX files found to quantize.")
        return

    print(f"\nQuantizing {src.name} to INT8...")

    q8_path = onnx_path / "model_q8.onnx"
    quantize_dynamic(
        model_input=str(src),
        model_output=str(q8_path),
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    print(f"Q8 model saved to {q8_path}")

    _print_dir_sizes(onnx_path)

    _quantize_int4(onnx_path, src)


def _quantize_int4(onnx_path: Path, src: Path) -> None:
    import onnx
    from onnx import numpy_helper

    print("\nQuantizing to INT4 (manual weight-only)...")

    model = onnx.load(str(src), load_external_data=True)

    q4_path = onnx_path / "model_q4.onnx"

    initializers = {init.name: init for init in model.graph.initializer}
    scale_map = {}
    zero_point_map = {}

    new_initializers = {}
    removed = set()

    for node in model.graph.node:
        if node.op_type != "MatMul":
            continue

        weight_name = node.input[1]
        if weight_name not in initializers:
            continue

        weight_init = initializers[weight_name]
        weight = numpy_helper.to_array(weight_init)

        if weight.dtype not in (np.float32, np.float16):
            continue

        w = weight.astype(np.float32)
        axis = 0
        group_size = 128

        orig_shape = w.shape
        if len(orig_shape) != 2:
            continue

        out_dim, in_dim = orig_shape
        if in_dim % group_size != 0:
            group_size = 32
            if in_dim % group_size != 0:
                continue

        n_groups = in_dim // group_size
        w_grouped = w.reshape(out_dim, n_groups, group_size)

        w_max = np.max(w_grouped, axis=-1, keepdims=True)
        w_min = np.min(w_grouped, axis=-1, keepdims=True)

        scale = (w_max - w_min) / 15.0
        scale = np.where(scale == 0, 1.0, scale)
        zero_point = np.round(-w_min / scale).astype(np.uint8)
        zero_point = np.clip(zero_point, 0, 15)

        w_q = np.round((w_grouped - w_min) / scale).astype(np.uint8)
        w_q = np.clip(w_q, 0, 15)

        w_packed = (w_q[..., 0::2] + w_q[..., 1::2] * 16).astype(np.uint8)

        packed_shape = list(w_packed.shape)
        packed_init = onnx.helper.make_tensor(
            name=f"{weight_name}_q4",
            data_type=onnx.TensorProto.UINT8,
            dims=packed_shape,
            vals=w_packed.tobytes(),
            raw=True,
        )

        scale_shape = list(scale.shape)
        scale_init = onnx.helper.make_tensor(
            name=f"{weight_name}_scale",
            data_type=onnx.TensorProto.FLOAT,
            dims=scale_shape,
            vals=scale.astype(np.float32).tobytes(),
            raw=True,
        )

        zp_shape = list(zero_point.shape)
        zp_init = onnx.helper.make_tensor(
            name=f"{weight_name}_zp",
            data_type=onnx.TensorProto.UINT8,
            dims=zp_shape,
            vals=zero_point.tobytes(),
            raw=True,
        )

        new_initializers[f"{weight_name}_q4"] = packed_init
        new_initializers[f"{weight_name}_scale"] = scale_init
        new_initializers[f"{weight_name}_zp"] = zp_init

        removed.add(weight_name)

    if not new_initializers:
        print("No weights quantized. Saving as-is.")
        model.Save(str(q4_path))
        return

    kept = [init for init in model.graph.initializer if init.name not in removed]
    model.graph.ClearField("initializer")
    for init in kept:
        model.graph.initializer.append(init)
    for init in new_initializers.values():
        model.graph.initializer.append(init)

    try:
        onnx.save(
            model,
            str(q4_path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="model_q4.onnx_data",
            size_threshold=0,
        )
        print(f"Q4 model saved to {q4_path}")
    except Exception as e:
        print(f"Q4 save failed: {e}")
        print("Falling back to FP16-only export.")

    _print_dir_sizes(onnx_path)


def _print_dir_sizes(path: Path) -> None:
    if path.exists():
        for f in sorted(path.iterdir()):
            if f.is_file():
                size = f.stat().st_size
                print(f"  {f.name}: {size / 1e6:.1f} MB")


def push_to_hub(
    model_path: Path,
    repo_id: str,
    tokenizer_path: Path | None = None,
    tag: str | None = None,
) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)

    if tokenizer_path:
        for f in [
            "tokenizer.json",
            "tokenizer_config.json",
            "tokenizer.model",
            "special_tokens_map.json",
            "config.json",
            "generation_config.json",
        ]:
            src = tokenizer_path / f
            if src.exists():
                dst = model_path / f
                if not dst.exists():
                    shutil.copy2(src, dst)

    print(f"\nUploading to HuggingFace Hub: {repo_id}...")
    api.upload_folder(
        folder_path=str(model_path),
        repo_id=repo_id,
        repo_type="model",
    )

    if tag:
        print(f"Creating tag: {tag}")
        api.create_tag(repo_id=repo_id, tag=tag, repo_type="model")

    print(f"Done. Model at: https://huggingface.co/{repo_id}")
    if tag:
        print(f"Tagged as: {tag}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LoRA and export to ONNX")
    parser.add_argument(
        "--lora-path",
        type=Path,
        required=True,
        help="Path to LoRA checkpoint directory",
    )
    parser.add_argument(
        "--output-path", type=Path, required=True, help="Path to save merged model"
    )
    parser.add_argument(
        "--export-onnx", action="store_true", help="Export to ONNX FP16"
    )
    parser.add_argument(
        "--kv-cache",
        action="store_true",
        help="Export with KV cache inputs/outputs (matches LiquidAI ONNX format)",
    )
    parser.add_argument(
        "--quantize", action="store_true", help="Quantize ONNX (Q8 + Q4)"
    )
    parser.add_argument(
        "--push-to-hub", action="store_true", help="Push to HuggingFace Hub"
    )
    parser.add_argument(
        "--skip-merge", action="store_true", help="Skip merge step (already merged)"
    )
    parser.add_argument("--repo-id", type=str, default=HF_MODEL_REPO, help="HF repo ID")
    parser.add_argument(
        "--tag", type=str, default=None, help="Git tag for this version (e.g. v1-5k)"
    )
    args = parser.parse_args()

    if not args.skip_merge:
        best_ckpt = find_best_checkpoint(args.lora_path)
        merge_lora(best_ckpt, args.output_path)
    else:
        print(f"Skipping merge. Using existing merged model at {args.output_path}")

    push_path = args.output_path

    if args.export_onnx:
        onnx_path = args.output_path.parent / (args.output_path.name + "-onnx")
        if args.kv_cache:
            export_onnx_kv_cache(args.output_path, onnx_path)
        else:
            export_onnx_simple(args.output_path, onnx_path)
        push_path = onnx_path

        if args.quantize:
            quantize_onnx(onnx_path)

    if args.push_to_hub:
        push_to_hub(push_path, args.repo_id, tokenizer_path=args.output_path, tag=args.tag)

    print("\nAll done!")


if __name__ == "__main__":
    main()
