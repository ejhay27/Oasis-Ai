# Implementation status and evidence

**Status date:** 2026-10-12.

## Implemented in this checkout

- Python package/CLI with config and registry validators.
- Dataset rights registry contract with an empty starting registry and a training-eligibility gate.
- Source/prepared JSONL manifest validation, FFmpeg probing/extraction implementation, per-source/frame SHA-256, and preprocessing sidecar metadata.
- Stable source-hash group splits plus conservative 64-bit average-hash near-duplicate candidate grouping and audit metadata.
- A small OASIS-authored 3D-convolutional conditional diffusion architecture and local PyTorch training loop.
- Append-only local run metrics, validation, safe safetensors model/optimizer/RNG/scaler checkpoint files, integrity sidecars, and resume logic.
- Experimental model catalog with separate `external`, `adapted`, and `original` categories; an empty initial catalog.
- Fixed-seed PNG-frame generation, held-out epsilon-loss diagnostic, basic pixel diagnostics, and a human review CSV template.
- Research/architecture/data/compute/security/evaluation/Windows integration documentation and a model-card template.

## Tests actually performed

1. **Eight pure-Python unit tests** passed for empty/eligible rights registries, permission/version gates, TOML config, manifest/frame checks, path containment, source/near-duplicate split leakage grouping, hash verification, and model category validation.
2. **Model-forward smoke test** using PyTorch 2.14.1+cu130 on CPU: default architecture returned finite `[2,3,8,16,16]` output and instantiated **604,131 parameters** for `base_channels=32`, `condition_dim=128`.
3. **End-to-end synthetic smoke test** (`python scripts/smoke_test_synthetic.py`) passed on CPU. It generated 20 procedural random RGB clips (8 frames, 16×16), group-split them 12/4/4, trained two tiny optimizer steps, forced an interrupt after step 1, resumed to step 2, wrote and verified safe checkpoints, registered a temporary experimental catalog entry, sampled 8 PNG frames, and ran a one-sample held-out evaluation.
4. The smoke test used **only temporary procedural noise/patterns**. Its temporary registry entry was for test plumbing; it is not a dataset-rights review or approved real training source. A temporary checkpoint/catalog/evaluation were deleted at process exit. No OASIS weights or run records remain in the repository.

Reproduce the local software smoke test after installing `[training,preprocess]`:

```bash
python scripts/smoke_test_synthetic.py
```

The script prints an explicit limitation that the result establishes no real-data training or video quality.

## Not yet tested or delivered

- No actual licensed/contributed footage has been acquired, reviewed, prepared, or approved. The tracked dataset registry has zero entries.
- No FFmpeg/ffprobe binary was installed in the inspected environment, so actual source-video decoding/preprocessing is untested.
- No training run on rights-reviewed real videos, no scientific convergence claim, and no useful/competitive video-quality claim.
- No retained or released OASIS video model checkpoint; the tracked model catalog is empty.
- No CUDA GPU was available to PyTorch, and no `nvidia-smi` was available. Mixed precision, GPU memory/utilization, distributed/multi-GPU training, and accelerator-specific resume remain untested.
- No Windows application/UI exists in this checkout. Windows packaging, Studio/Research Mode, MP4 encoding/export, model download policy, and real desktop inference integration are not implemented.
- No public-release legal review, model safety assessment, or external-baseline comparison has been completed.

The project should therefore be described as **research infrastructure with a successful synthetic software smoke test**, not as a trained OASIS video-generation model.
