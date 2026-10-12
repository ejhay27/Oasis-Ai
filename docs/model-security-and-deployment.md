# Model categories, security, and Studio boundary

## Categories

### External

A third party created the model. Record creator, official source/weight origin, license/version, usage restrictions, and attribution. OASIS integration, caching, serving, or evaluating does not transfer ownership.

### OASIS-adapted

A permitted base model has been changed by fine-tuning, adapters, distillation, or similar work. Record the base ID/license, dataset manifest and rights, method, inherited and new weights, attribution, and redistribution constraints. Do not call it “trained from scratch.”

### Original OASIS

An OASIS-controlled architecture and pipeline produced learned weights from a documented training run. Sidecars record random initialization and pretrained component provenance. If pretrained components/weights are added later, name their source, creator, license, role, and inherited parameter scope. Original architecture does not establish training-data rights or prove all weights are original.

The catalog schema and docs enforce this vocabulary. The catalog begins empty. The current tiny architecture has no pretrained component, but no actual OASIS training checkpoint has been created here.

## Checkpoint handling

- Use safetensors for inference weights and optimizer/RNG tensor state; JSON contains scalar metadata.
- Do not import/execute code from a model package or untrusted checkpoint.
- Verify SHA-256 and require an expected architecture ID/config before inference. Hash checking detects file changes relative to its sidecar; it is not a digital signature or an independent chain-of-custody audit.
- Keep checkpoint files, training data, and catalogs in controlled storage. Do not accept arbitrary Python model code or automatically download weights.
- Before distributing a checkpoint, independently inspect all dataset/model component licenses, attribution, privacy, safety, security, and weight origin. Catalog registration remains experimental and `not_cleared` by default.
- Treat model weights as sensitive assets. Validate source/sidecar paths and limits before loading; do not expose file import endpoints without access controls and resource limits.

## Desktop integration boundary

This checkout has no Windows Studio UI/backend, so no desktop integration is claimed complete. The Python `load_verified_model()` / `generate_video()` adapter is a candidate Research/Studio boundary for the fixed OASIS architecture. It accepts a verified checkpoint, uses a code-defined architecture, and returns a PNG-frame array. MP4 encoding, project management, Windows packaging, UI, progress/error display, model capability cards, compatibility checks, and export flows remain to be built and tested.

Future Studio Mode should load only compatible released checkpoints and display model version, hash, capabilities, hardware requirements, license, and limitations. Research Mode should expose registered versions, reports, evaluations, and explicitly requested local tests. Training must not run when the app opens. Research dependencies should remain separable from the lightweight UI. Any future remote compute flow must display provider, region/instance, estimate and uncertainties, data movement, and explicit authorization before submission.
