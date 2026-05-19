# Security Policy

## Reporting Security Or Privacy Issues

Please do not open public GitHub issues that contain real PHI, secrets, credentials, private model artifacts, or exploitable security details.

If this repository has GitHub private vulnerability reporting enabled, use that channel. Otherwise, contact the maintainer directly through the repository owner profile and include only the minimum synthetic detail needed to reproduce the issue.

## Data Handling

PHI Firewall is designed so redaction inference runs in the browser with WebGPU. The app still downloads model and tokenizer artifacts from Hugging Face, and any deployed copy serves the static application bundle from its hosting provider.

Do not paste real PHI into development builds, bug reports, screenshots, benchmark files, or logs unless your organization has explicitly approved that workflow.

## Supported Versions

This project is an experimental demo. Security fixes are handled on the default branch.
