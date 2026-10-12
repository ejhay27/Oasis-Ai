# Training and reproducibility guide

## Experiment contract

Training is a separate Python research operation, not a desktop-app startup behavior. A local run begins only from an explicit `oasis-research train <config.toml>` command. There is no remote-compute integration.

Every run keeps:

- copied versioned TOML and resolved config;
- config, prepared manifest, rights registry, and split-sidecar SHA-256 references;
- training/validation example counts, architecture ID/config, framework versions, hardware/backend and precision;
- append-only `metrics.jsonl` records with actual train/validation loss, learning rate, elapsed step time, throughput, and resource fields (unavailable metrics are `null`);
- fixed validation sampling seeds, checkpoint files, training progress, optimizer state, random states, and weight provenance;
- `run.json` status and error/interruption summary.

`metrics.jsonl` is local logging, not a dashboard and not a source of invented or interpolated values. Do not hand-edit run records. A failed run is evidence and should be retained.

## Config and data validation

Start from `configs/tiny-video-diffusion.example.toml`, set real paths and experiment parameters, and check it with:

```bash
oasis-research validate-config path/to/experiment.toml
oasis-research validate-manifest path/to/splits.jsonl \
  --registry path/to/datasets.json --prepared --for-training
```

The sample TOML is intentionally not runnable while the registry is empty. Training checks approved dataset ID/version, human-reviewed captions, split assignments, prepared dimensions/frame count, paths, and frame hashes. Train and validation splits must both contain examples. Split groups are source-aware, so very small source sets may not populate all splits.

The training config fixes input shape, architecture width, diffusion schedule, optimizer, learning rate, batch size, gradient accumulation, precision, step cadence, determinism, and device request. Any experiment variation belongs in a new config; the trainer does not auto-tune parameters after an OOM.

## Training lifecycle

The training implementation performs:

1. local environment/config/rights/manifest validation;
2. random initialization and a named OASIS model instantiation;
3. clip/caption batching from the prepared, approved split;
4. forward diffusion noise addition and epsilon-prediction loss;
5. backpropagation, optional explicit gradient accumulation, and AdamW updates;
6. deterministic held-out validation noise/loss at the configured cadence;
7. actual metrics/resource logging;
8. periodic and final safe checkpoints;
9. controlled interrupt/failure reporting and no silent settings changes.

Determinism is best-effort and depends on the same source artifacts, PyTorch version, device, kernels, and hardware. `torch.use_deterministic_algorithms(..., warn_only=True)` is recorded in the configuration but cannot guarantee bitwise equality on every backend.

## Checkpoint format, verification, and resume

Each `checkpoints/step-XXXXXXXX` checkpoint consists of:

- `.safetensors`: model tensors;
- `.state.safetensors`: optimizer tensors, CPU/CUDA/NumPy/batch-order RNG tensors;
- `.json`: architecture, model config, step/progress, optimizer scalar state, scaler state, source/config/registry hashes, versions, hardware, and weight provenance.

Inference loads only the code-defined `oasis.tiny_video_diffusion.v1` architecture and safetensors tensors; checkpoint files do not provide Python code. `verify-checkpoint` checks referenced hashes/metadata and is not an independent authorship, legality, safety, or quality audit.

To resume an interrupted run, use the checkpoint sidecar and same config/data/registry/split hashes and selected device:

```bash
oasis-research train path/to/experiment.toml --resume \
  runs/<run-id>/checkpoints/step-00000100.json
```

The resume path restores model, AdamW state, mixed-precision scaler, sampler order/cursor, and saved random state. It refuses to resume with mismatched immutable inputs or selected device. A completed run at `max_steps` cannot be extended by changing the old config; start and document a new experiment for additional steps. Same-device/framework kernel nondeterminism can still limit exact replay.

## OOM and failures

The last successfully written checkpoint remains available after later errors. A CUDA OOM is logged with its step and guidance; the pipeline does not retry at a smaller shape/batch behind the user's back. Change explicit settings in a new config and preserve the failed run. A partially completed dataset/preprocessing operation also should not be presented as a complete manifest.

## Model versioning

Training creates experimental checkpoint metadata, but does not automatically register a release. `register-checkpoint` adds a catalog entry in category `original`, status `experimental`, and distribution status `not_cleared`. The entry records training-run/config/data hashes, pretrained component list, and checkpoint hashes. There are no current registered checkpoints. Model-card, license, rights, and release criteria must be completed before any public deployment.
