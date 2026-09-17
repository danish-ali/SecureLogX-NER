# AGENTS.md

## Project Context

This project is SecureLogX, a Java-based secure logging framework focused on compliance-ready logging, sensitive data masking, and production-grade integration.

SecureLogX supports:
- SECURE log level for AI-based masking
- Configurable masking policies
- Entity detection for SSN, NPI, name, address, DOB, email, phone, card data, CVV, ZIP, and similar sensitive fields
- Java-based ONNX inference
- Hugging Face-compatible tokenizer integration
- Async and parallel logging modes
- Order-preserving log sequence behavior
- Runtime-configurable masking rules

## Design Principles

- Do not expose raw sensitive data in logs.
- Prefer configuration-driven masking over hard-coded entity behavior.
- Keep Java integration production-ready and modular.
- Avoid Python runtime dependency in the production Java path.
- Keep ONNX model, tokenizer.json, labels.json, and model metadata clearly separated.
- Prioritize correctness, compliance-readiness, and maintainability over micro-benchmark performance.
- GPU acceleration is optional and should not complicate the default CPU path.

## Coding Rules

- Use clean Java 17+ compatible code unless otherwise stated.
- Keep public APIs stable and backward-compatible.
- Add unit tests for masking behavior, policy configuration, and entity handling.
- Do not introduce dependencies unless necessary.
- Avoid logging secrets, tokens, keys, credentials, raw PII, or raw NPI.

## Before Making Changes

- Read the relevant source files first.
- Explain the intended change briefly.
- Prefer small, reviewable changes.
- Run or suggest relevant tests after modifying code.
- Preserve existing architecture unless there is a clear reason to refactor.

## Important Files to Check

- README.md
- docs/
- src/main/java/
- src/test/java/
- model metadata files
- tokenizer and ONNX integration files