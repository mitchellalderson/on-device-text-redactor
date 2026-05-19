# Contributing

Thanks for taking a look at PHI Firewall. This project touches health-related text, so contributions should be careful about privacy, model behavior, and reproducibility.

## Development Setup

```bash
npm install
npm run check
npm run build
```

For browser testing, use a desktop browser with WebGPU support. First model load can take several minutes because ONNX artifacts are downloaded from Hugging Face and cached by the browser.

## Pull Request Checklist

- Keep changes focused and describe the user-visible impact.
- Run `npm run check` and `npm run build`.
- If model-loading behavior changes, test both `Fine-Tuned` and `Base Model` variants.
- If the selected Hugging Face revision changes, bump `CACHE_VERSION` in `src/lib/model-cache.ts`.
- If training or export behavior changes, update `training/README.md` and `docs/MODEL_RELEASE_CHECKLIST.md`.
- Do not commit generated artifacts such as `dist/`, `training/data/*.jsonl`, `training/output/`, ONNX files, checkpoints, or credentials.

## Safety Expectations

Do not claim the system is HIPAA-compliant or production-ready unless that has been independently reviewed. Model outputs can miss PHI, over-redact harmless text, or generate incorrect text.

When reporting redaction quality issues, use synthetic examples whenever possible. Do not include real PHI in GitHub issues, pull requests, logs, screenshots, or benchmark artifacts.
