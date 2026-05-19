# Model Release Checklist

Use this checklist when publishing a new fine-tuned model revision for the browser app.

## Before Training

- Confirm `training/config.yaml` points at the intended base model, dataset, Modal app, GPU, and output volume.
- Confirm the dataset contains no accidental real PHI beyond the permitted source data.
- Decide the release tag, for example `v3-50k-kv`.
- Set your Hugging Face target repos:

```bash
export PHI_FIREWALL_HF_DATASET_REPO=your-org/phi-redaction-sft
export PHI_FIREWALL_HF_MODEL_REPO=your-org/phi-firewall-lfm2-350m-onnx
```

## Prepare Data

```bash
npm run prepare:data
```

This writes local JSONL files only. To push the dataset:

```bash
npm run prepare:data:push -- --sample 50000 --dataset-repo your-org/phi-redaction-sft
```

Verify that `training/data/train.jsonl` and `training/data/test.jsonl` were created and contain chat-style `messages` examples.

Update `training/config.yaml` so `dataset.path` points to the pushed dataset repo before training.

## Train

```bash
npm run train
```

Watch detached logs:

```bash
npm run train:logs
```

Download the selected LoRA checkpoint from the Modal volume:

```bash
modal volume ls phi-firewall-finetune
modal volume get phi-firewall-finetune /outputs/phi-firewall-redaction/<run-name> training/output/lora
```

## Export

Export a browser-ready KV-cache ONNX graph and push it to Hugging Face:

```bash
npm run export:model:push -- \
  --kv-cache \
  --repo-id your-org/phi-firewall-lfm2-350m-onnx \
  --tag <release-tag>
```

Confirm the Hugging Face model revision includes:

- `model_fp16.onnx`
- tokenizer files
- `config.json`
- `generation_config.json`, if available

## Benchmark

Run at least one benchmark against the previous public revision:

```bash
modal run benchmark.py --repo-id your-org/phi-firewall-lfm2-350m-onnx --rev-a <old-tag> --rev-b <release-tag> --samples 50
```

Save benchmark notes in the release description. Do not publish examples containing real PHI.

## Update The App

- Set `MODEL_REVISION` in `src/lib/trained-model.ts` to the new tag.
- Increment `CACHE_VERSION` in `src/lib/model-cache.ts` so existing browser caches refresh.
- Run `npm run check`.
- Run `npm run build`.
- Test the app in a WebGPU-capable desktop browser.
- Confirm both `Fine-Tuned` and `Base Model` variants initialize.

## Release

- Commit the app revision bump and documentation updates.
- Deploy with `npm run deploy`, if publishing the hosted app.
- Include redaction quality caveats in the release notes.
