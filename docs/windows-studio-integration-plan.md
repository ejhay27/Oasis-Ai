# Windows Studio / Research Mode integration plan

**Current status:** not implemented. This checkout contains no desktop UI/runtime to integrate with. The current Python adapter and CLI demonstrate a backend contract only.

## Proposed boundary

Keep the Windows application independent from training dependencies. A future lightweight inference service/library should expose a fixed interface similar to:

```text
list_registered_models() -> model IDs, category, status, checkpoint hash, shape, capabilities, requirements
verify_import(path) -> compatibility report + provenance/model-card warnings
load_for_inference(model_id) -> restricted, code-defined inference adapter
generate(prompt, seed, project settings) -> progress, frames/video result, resource/error report
```

`load_verified_model()` currently accepts the OASIS v1 safetensors/checkpoint sidecar, verifies hashes, instantiates only the hard-coded `oasis.tiny_video_diffusion.v1` class, and returns a model for research inference. It does not enforce a public release status; the current CLI is explicitly a research utility. **Do not expose this unfiltered function as Studio Mode.** A production Studio loader must require a catalog entry with a reviewed/released status, a cleared distribution state, a compatible architecture/runtime, and an approved model card.

## Studio Mode

- List only verified, compatible, cleared model versions.
- Display creator/category, model version, checkpoint SHA-256, tasks actually validated, input/output dimensions/duration/FPS, supported hardware, license, training provenance, limitations, and release status.
- Generate/preview/export without importing training code or starting training.
- Use relative/local backend IPC; do not make browser/UI code call `localhost` services from another machine/browser context.
- Handle a missing model/backend with a clear message; never silently download external models or data.

## Research Mode

- Require an explicit user action to inspect a run, register/import a checkpoint, evaluate a held-out dataset, or launch local training.
- Show immutable config/registry/manifest hashes, checkpoint identity, hardware and resource logs, progress, costs (if remote is ever added), and errors.
- Validate checkpoint compatibility and safe serialization before registration; reject untrusted code.
- Keep research dependencies and dataset access out of the normal Studio startup path.

## Windows packaging and tests still required

1. Decide app technology and IPC/process boundary from the actual Studio codebase (not present here).
2. Package inference separately from training; test on supported Windows versions and CPU/NVIDIA GPU configurations.
3. Define signed release manifests and trusted checkpoint origins; SHA-256 in an unsigned sidecar alone does not establish authenticity.
4. Implement MP4 encoding, project/asset management, export behavior, cancellation, memory reporting, and device compatibility UI.
5. Test that app open/close never triggers training, downloads, or remote jobs.
6. Only enable public Studio models after rights/legal review, security/safety review, evaluation, and model-card sign-off.
