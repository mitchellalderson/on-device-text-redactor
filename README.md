# PHI Firewall

PHI Firewall is a browser-based Protected Health Information (PHI) redaction demo that runs language-model inference on the user's device with WebGPU. Paste clinical or administrative text into the app, choose either the base model or the fine-tuned redaction model, and stream back a version with names, dates of birth, SSNs, phone numbers, emails, addresses, medical record numbers, insurance IDs, and similar identifiers replaced with `[REDACTED]`.

> **Important:** PHI Firewall is an experimental demo, not a compliance-certified redaction system. Do not use it as the only control for HIPAA, privacy, legal, or production de-identification workflows without independent validation and review.

The app is built with Svelte, Vite, TypeScript, ONNX Runtime Web, and Hugging Face Transformers.js. Model files are fetched from Hugging Face on first use and cached in the browser Cache Storage for future sessions.

## What This Repo Contains

- A Svelte/Vite frontend for interactive PHI redaction.
- A WebGPU inference wrapper around LiquidAI LFM2.5-350M ONNX models.
- Two model variants:
  - **Fine-Tuned**: `aldersondev/phi-firewall-lfm2-350m-onnx` at revision `v2-50k-kv`.
  - **Base Model**: `LiquidAI/LFM2.5-350M-ONNX`.
- A Modal deployment entrypoint that serves the built static app.
- A training pipeline for preparing PHI redaction SFT data, fine-tuning with LoRA on Modal, exporting to ONNX, and pushing model artifacts to Hugging Face.
- A Modal benchmark script for comparing model revisions on held-out examples.

## Privacy Model

Inference is designed to run locally in the browser through WebGPU. Input text is not sent to this repository's backend for redaction.

There are still important network behaviors to understand:

- On first load, the browser downloads tokenizer and ONNX model artifacts from Hugging Face.
- The app caches those model files locally in browser Cache Storage.
- If you deploy this app, the hosting layer serves only the static app bundle; redaction itself happens in the client.

Do not treat this as a compliance-ready system without your own review. The model can miss PHI, over-redact non-PHI, or generate imperfect text. Human review and domain-specific validation are still required for production use.

## Known Limitations

- Redaction is probabilistic and can miss sensitive text.
- The model may over-redact non-sensitive text or alter wording around redactions.
- The app requires desktop WebGPU and does not provide a CPU fallback.
- First load can be slow because model artifacts are downloaded from Hugging Face.
- Browser Cache Storage may keep older model artifacts until `CACHE_VERSION` changes.
- The training and benchmark pipelines depend on external services: Hugging Face and Modal.
- Public issues, screenshots, logs, and examples must use synthetic data only.

## Requirements

For the browser app:

- Node.js 20+ is recommended.
- npm.
- A desktop browser with WebGPU support:
  - Chrome 113+
  - Edge 113+
  - Safari 26+
- A desktop GPU. The app intentionally blocks mobile browsers.

For training, exporting, and deployment:

- Python dependencies are installed inside `training/leap-finetune/.venv`.
- `uv`.
- A Modal account with billing configured.
- A Hugging Face account and token with write access.
- Enough local disk space for downloaded datasets, checkpoints, and ONNX exports.

## Quick Start

Install JavaScript dependencies:

```bash
npm install
```

Start the local development server:

```bash
npm run dev
```

Open the Vite URL shown in the terminal, usually:

```text
http://localhost:5173
```

On first load, the app downloads model files from Hugging Face. This can take a few minutes depending on network speed and the selected model. Subsequent loads use the browser cache until the cache version changes.

## Using The App

1. Open the app in a WebGPU-capable desktop browser.
2. Wait for the model status to reach `Ready`.
3. Choose a model variant:
   - **Fine-Tuned** for the redaction-specialized model.
   - **Base Model** to compare behavior against the upstream model.
4. Paste text that may contain PHI.
5. Click **Redact PHI**.
6. Watch the redacted output stream into the output panel.

The footer shows model-loading status and the detected GPU label. During generation, the output panel shows approximate tokens per second.

## Browser And GPU Notes

The app checks `navigator.gpu` before model initialization. If WebGPU is unavailable, the UI shows an error instead of silently falling back to CPU.

If Chrome reports that WebGPU is unavailable, check:

```text
chrome://gpu
```

For local experimentation, you may also need:

```text
chrome://flags/#enable-unsafe-webgpu
```

The fine-tuned model uses FP16 when the adapter reports `shader-f16` support and falls back to FP32 model loading otherwise.

## Project Structure

```text
.
|-- src/
|   |-- App.svelte              # Main PHI Firewall UI
|   |-- app.css                 # App styling
|   |-- main.ts                 # Svelte entrypoint
|   `-- lib/
|       |-- gpu.ts              # Model-variant selection and WebGPU checks
|       |-- trained-model.ts    # Fine-tuned model inference path
|       |-- base-model.ts       # Base model inference path
|       `-- model-cache.ts      # Browser Cache Storage helper
|-- training/
|   |-- README.md               # Fine-tuning pipeline details
|   |-- config.yaml             # Modal/leap-finetune configuration
|   |-- prepare_data.py         # Nemotron-PII -> SFT JSONL conversion
|   |-- export.py               # LoRA merge and ONNX export
|   `-- data/                   # Generated JSONL files (gitignored)
|-- benchmark.py                # Modal benchmark for ONNX model revisions
|-- serve.py                    # Modal ASGI app for serving dist/
|-- vite.config.ts              # Vite build config and vendor chunking
`-- package.json                # npm scripts and dependencies
```

## Available Scripts

### Frontend

```bash
npm run dev
```

Runs the Vite development server.

```bash
npm run build
```

Builds the production app into `dist/`.

```bash
npm run preview
```

Serves the production build locally with Vite.

```bash
npm run check
```

Runs `svelte-check` and TypeScript checks.

### Deployment

```bash
npm run deploy
```

Builds the frontend and deploys `serve.py` to Modal. The Modal app serves `dist/index.html` and static assets through Starlette.

### Training And Model Artifacts

```bash
npm run prepare:data
```

Downloads and converts the Nvidia Nemotron-PII dataset into supervised fine-tuning JSONL files under `training/data/`. This is local-only and does not push to Hugging Face.

```bash
npm run prepare:data:push
```

Runs the same data preparation step and uploads to the Hugging Face dataset repo provided with `--dataset-repo` or `PHI_FIREWALL_HF_DATASET_REPO`.

```bash
npm run train
```

Runs LoRA SFT with `training/leap-finetune` using `training/config.yaml`.

```bash
npm run train:logs
```

Streams logs for the Modal fine-tuning app.

```bash
npm run export:model
```

Merges LoRA weights and exports ONNX artifacts locally.

```bash
npm run export:model:push
```

Exports ONNX artifacts and uploads them to the Hugging Face model repo provided with `--repo-id` or `PHI_FIREWALL_HF_MODEL_REPO`.

```bash
npm run benchmark
```

Runs the Modal benchmark defined in `benchmark.py`.

## Model Loading Details

The runtime model switch lives in `src/lib/gpu.ts`.

The fine-tuned path in `src/lib/trained-model.ts` loads:

```text
aldersondev/phi-firewall-lfm2-350m-onnx
revision: v2-50k-kv
```

The base comparison path in `src/lib/base-model.ts` loads:

```text
LiquidAI/LFM2.5-350M-ONNX
```

Both inference paths:

- Load the tokenizer with `@huggingface/transformers`.
- Create an ONNX Runtime Web session with the `webgpu` execution provider.
- Maintain LFM2 hybrid conv/attention cache tensors between generation steps.
- Stream decoded tokens to the Svelte UI.
- Stop after EOS, repeated-token safeguards, token limits, or output-length limits.

Downloaded model responses are cached by `src/lib/model-cache.ts` in:

```text
phi-firewall-models-v5
```

If you change model files or revisions and need to force a fresh download, increment `CACHE_VERSION` in `src/lib/model-cache.ts`.

## Training Pipeline

The detailed training guide lives in [training/README.md](training/README.md), and model publication steps are tracked in [docs/MODEL_RELEASE_CHECKLIST.md](docs/MODEL_RELEASE_CHECKLIST.md). The short version is:

1. Clone and install `Liquid4All/leap-finetune` under `training/leap-finetune`.
2. Install Python dependencies in that virtual environment.
3. Authenticate with Hugging Face and Modal.
4. Run `npm run prepare:data`.
5. Run `npm run train`.
6. Download the selected LoRA checkpoint from the Modal volume.
7. Run `npm run export:model` locally, or `npm run export:model:push` with your Hugging Face model repo and tag.
8. Update the frontend model revision if you publish a new model tag.

The default training config uses:

- Base model: `LFM2.5-350M`.
- Dataset: set `dataset.path` in `training/config.yaml` to your pushed dataset repo, for example `your-org/phi-redaction-sft`.
- Training type: SFT.
- Adapter: LoRA.
- Epochs: 3.
- Batch size: 8.
- Learning rate: `2e-4`.
- GPU: `H100:1` on Modal.
- Modal app: `phi-firewall-finetune`.
- Modal volume: `phi-firewall-finetune`.

Generated training data, checkpoints, and model files are gitignored.

## Preparing Data

`training/prepare_data.py` converts the Nvidia Nemotron-PII dataset into chat-style SFT examples:

```json
{
  "messages": [
    { "role": "system", "content": "..." },
    { "role": "user", "content": "Original text with PHI" },
    { "role": "assistant", "content": "Redacted text" }
  ]
}
```

It filters rows without usable spans, drops very long examples, stratifies by domain, creates an 80/20 train/test split, writes JSONL files locally, and can push the dataset to Hugging Face only when explicitly requested.

Useful examples:

```bash
npm run prepare:data
npm run prepare:data -- --sample 50000
npm run prepare:data:push -- --dataset-repo your-org/phi-redaction-sft
```

## Exporting Models

`training/export.py` can:

- Select the best LoRA checkpoint by evaluation loss.
- Merge the LoRA adapter into the base model.
- Export a standalone FP16 ONNX model.
- Export a KV-cache ONNX graph for browser token-by-token inference.
- Quantize ONNX weights experimentally.
- Push exported artifacts to Hugging Face and create a tag.

Example:

```bash
npm run export:model:push -- \
  --kv-cache \
  --repo-id your-org/phi-firewall-lfm2-350m-onnx \
  --tag v3
```

After publishing a new model tag, update `MODEL_REVISION` in `src/lib/trained-model.ts`.

## Benchmarking

`benchmark.py` runs ONNX inference on Modal with an A10G GPU and compares one or more Hugging Face model revisions against `training/data/test.jsonl`.

It reports:

- Exact match rate.
- Character-level F1.
- Redaction recall.
- Redaction precision.
- Tokens per second.
- Wall-clock generation time.

Examples:

```bash
modal run benchmark.py
modal run benchmark.py --samples 20
modal run benchmark.py --repo-id your-org/phi-firewall-lfm2-350m-onnx --rev-a main --rev-b v2-50k-kv
modal run benchmark.py --output results.json
```

## Deployment

The deployment path is intentionally simple:

1. `npm run build` emits static assets into `dist/`.
2. `serve.py` mounts that directory into a Modal image.
3. Starlette serves `index.html` at `/` and static assets below `/`.

Deploy with:

```bash
npm run deploy
```

## Troubleshooting

### WebGPU Not Available

Use a desktop WebGPU-capable browser and inspect:

```text
chrome://gpu
```

If needed for local testing, enable:

```text
chrome://flags/#enable-unsafe-webgpu
```

### First Load Is Slow

The browser has to download model and tokenizer artifacts from Hugging Face. Leave the tab open until loading completes. Later loads should use Cache Storage.

### Model Keeps Loading An Old Revision

Increment `CACHE_VERSION` in `src/lib/model-cache.ts`, rebuild, and reload the app.

### Fine-Tuned Model Fails But Base Model Works

Check that `MODEL_ID` and `MODEL_REVISION` in `src/lib/trained-model.ts` point to a Hugging Face revision that contains the expected tokenizer files and ONNX file.

### Benchmark Cannot Find Test Data

Run:

```bash
npm run prepare:data
```

The Modal benchmark adds `training/data/test.jsonl` to the remote image.

### Hugging Face Upload Fails

Authenticate inside the Python environment used by the training pipeline:

```bash
training/leap-finetune/.venv/bin/python -c "from huggingface_hub import login; login()"
```

### Modal Commands Fail

Run:

```bash
modal setup
```

Then confirm that your account has billing enabled and access to the requested GPU type.

## Development Notes

- The app is a plain Vite/Svelte app, not SvelteKit.
- `vite.config.ts` excludes ONNX Runtime Web and Transformers.js from dependency pre-bundling and splits them into separate vendor chunks.
- `BaseModel` and `TrainedModel` intentionally duplicate some generation logic because their model artifacts differ in how files are loaded.
- The app stores the selected model variant in `localStorage` under `phi-model-variant`.
- Switching model variants reloads the page to reset the runtime cleanly.

## Contributing And Security

- Contribution guidance is in [CONTRIBUTING.md](CONTRIBUTING.md).
- Security and privacy reporting guidance is in [SECURITY.md](SECURITY.md).
- Example local environment variables are documented in [.env.example](.env.example).
- Release notes are tracked in [CHANGELOG.md](CHANGELOG.md).
- Do not include real PHI in issues, pull requests, fixtures, screenshots, logs, benchmark outputs, or documentation.

## License

This project is licensed under the [MIT License](LICENSE).
