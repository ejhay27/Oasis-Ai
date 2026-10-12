# OASIS AI Video Model Research Foundations

This repository now contains the first **research-infrastructure increment** for the OASIS original video-model program: rights-gated dataset manifests, FFmpeg preprocessing, source/near-duplicate-aware splits, a compact OASIS-authored research architecture, a real PyTorch training loop, safe checkpoint/resume plumbing, a local experiment log, evaluation/review tools, and a three-category model catalog.

## Current state — read before using

- **No OASIS model has been trained on rights-reviewed real video or released.** A temporary synthetic end-to-end smoke run exercised the training/checkpoint/resume/evaluation code, but its checkpoint was deleted; it does not establish a real-data model or generation quality. The architecture and pipeline code alone do not establish a model.
- `data/registry/datasets.json` is empty. **No dataset is approved for training.** Do not add material unless its specific rights have been reviewed and documented.
- `models/catalog.json` is empty. There are no registered OASIS checkpoints.
- This checkout contains no Windows desktop application. The Python inference adapter is a backend boundary that a future Studio/Research UI can call; it does not implement a desktop UI.
- No training job, cloud job, or data download is started automatically. The only training entry point is an explicit `oasis-research train ...` command.
- The first model direction is intentionally experimental: pixel-space, text-conditioned DDPM; RGB 8-frame clips at 16×16; a small 3D-convolutional denoiser and trainable UTF-8-byte text embedding; no pretrained encoder, VAE, teacher, or inherited weights. It is not expected to compete with commercial systems.

See the [research roadmap](docs/research-roadmap.md), [targeted research landscape review](docs/research-landscape.md), [architecture decision](docs/architecture-decision-record-0001.md), [dataset policy](docs/dataset-policy.md), and [compute assessment](docs/compute-assessment.md) before planning a run.

## Requirements

- Python 3.11 or newer.
- FFmpeg and `ffprobe` on `PATH` for video probing/preprocessing. They are external tools, not bundled here.
- Pillow, NumPy, PyTorch, and safetensors for preprocessing/training/inference. Select a PyTorch build appropriate for the target Windows/Linux accelerator before installing the project extras.

```powershell
# Windows PowerShell, from the repository root
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# Install the PyTorch build appropriate for this machine first (CPU/CUDA/MPS as applicable).
python -m pip install -e ".[training,preprocess,test]"
```

On Linux/macOS, use the equivalent `python -m venv .venv`, activation, and `pip install -e '.[training,preprocess,test]'` commands. Training/inference imports remain optional, so registry and configuration checks can run without PyTorch.

To exercise the training/checkpoint/resume/inference/evaluation plumbing without source footage, run the explicitly synthetic, temporary smoke test:

```bash
python scripts/smoke_test_synthetic.py
```

It uses random 16×16 RGB patterns, deletes all artifacts on exit, and does **not** validate video quality or licensed-data training. See [implementation status and test evidence](docs/implementation-status.md).

## Local research workflow

### 1. Inspect the machine and validate the empty registry

```bash
oasis-research hardware --storage-path .
oasis-research validate-registry data/registry/datasets.json
oasis-research validate-catalog models/catalog.json
```

The hardware report leaves unavailable measurements as `null`/`unknown`; it does not infer a GPU from the operating system. See [compute assessment](docs/compute-assessment.md).

### 2. Admit a dataset only after rights review

Each `datasets[]` record in `data/registry/datasets.json` tracks source, version, acquisition date, rights holder, license and license text hash, training/commercial/redistribution permissions, attribution, privacy restrictions, known limitations, integrity data, and review evidence. An entry must be `status: "approved"` and have explicit `permissions.training: "allowed"` before it can appear in a training/preprocessing manifest. `unknown`, `restricted`, `pending`, `rejected`, and `takedown` do not pass training validation. Commercial use and redistribution are separate decisions.

Registry approval is a project workflow gate, **not legal advice or an independent legal opinion**. Keep a reviewed copy of the applicable license/evidence; do not set `approved` based only on a public download link. See [dataset policy and takedown process](docs/dataset-policy.md).

A source manifest is JSON Lines, one item per original video, with fields such as:

```json
{"item_id":"source-0001","dataset_id":"your-reviewed-dataset","dataset_version":"1.0","source_path":"videos/clip.mp4","caption":"A red kite moves across a cloudy sky.","caption_reviewed":true}
```

Paths in source manifests are relative to the manifest file. The caption must be human-reviewed before training; a web title or automatically generated label is not assumed accurate. Source bytes are hashed during preprocessing. The preprocessor never fetches videos and strips audio from extracted frames.

### 3. Preprocess and split

```bash
oasis-research validate-manifest data/source.jsonl --registry data/registry/datasets.json
oasis-research inspect-video data/videos/clip.mp4
oasis-research preprocess \
  --manifest data/source.jsonl \
  --registry data/registry/datasets.json \
  --output-dir data/prepared \
  --frames 8 --width 16 --height 16 --fps 4
oasis-research make-splits \
  --manifest data/prepared/prepared.jsonl \
  --registry data/registry/datasets.json \
  --output data/prepared/splits.jsonl \
  --seed 2026 --train-ratio 0.90 --validation-ratio 0.05 --test-ratio 0.05
```

The splitter keeps all clips from an exact source video together and can conservatively group visual near-duplicate candidates using 64-bit average-hash fingerprints. The hash is a screening heuristic, not semantic duplicate proof; inspect `splits.jsonl.meta.json` and review candidate groups. A small number of source videos can leave validation/test empty because splitting is by source group, not clip. Do not weaken leakage controls simply to fill a split.

### 4. Validate and launch a small experiment explicitly

`configs/tiny-video-diffusion.example.toml` is a **template**, not a runnable dataset declaration. Update its manifest/registry paths and set a suitable short experiment before use.

```bash
oasis-research validate-config configs/tiny-video-diffusion.example.toml
oasis-research validate-manifest data/prepared/splits.jsonl \
  --registry data/registry/datasets.json --prepared --for-training
# Training starts only after this explicit command:
oasis-research train configs/tiny-video-diffusion.example.toml
```

Runs write `run.json`, a resolved config, and append-only `metrics.jsonl` records with actual training/validation loss, throughput, elapsed time, and available resource measurements. The trainer never silently changes settings after an OOM. It records failure and suggests settings to test in a new, versioned config. Checkpoints use safetensors for model, optimizer, and random-state tensors, plus a JSON provenance sidecar; resuming requires matching config, registry, manifests, and device.

```bash
oasis-research verify-checkpoint runs/<run-id>/checkpoints/step-00001000.safetensors
oasis-research register-checkpoint \
  --checkpoint runs/<run-id>/checkpoints/step-00001000.json \
  --catalog models/catalog.json
oasis-research generate \
  --checkpoint runs/<run-id>/checkpoints/step-00001000.safetensors \
  --prompt "A red kite moves across a cloudy sky." \
  --output-dir runs/sample-0001 --seed 1234
oasis-research evaluate \
  --checkpoint runs/<run-id>/checkpoints/step-00001000.safetensors \
  --manifest data/prepared/splits.jsonl \
  --registry data/registry/datasets.json \
  --output-dir runs/evaluation-0001 --samples 8 --seed 271828
```

Generation currently writes numbered PNG frames rather than an MP4. Evaluation creates a fixed held-out test report, sample frame sequences, and `human-review.csv`. Epsilon-prediction loss and basic pixel diagnostics are not video-quality or prompt-adherence scores. Complete human review and add valid, documented benchmarks before making quality claims.

## Model categories and safety

OASIS records three distinct categories: **external** (created by others), **adapted** (fine-tuned/derived, with base license and inherited weights recorded), and **original** (OASIS architecture and training run, with weight provenance and any pretrained dependencies recorded). The current architecture uses random initialization and no pretrained components; a checkpoint is only produced after an explicit training run. Registering a checkpoint marks it experimental and **not cleared for redistribution**; it does not prove legal ownership, dataset rights, quality, safety, or production readiness.

Checkpoints are never loaded as Python code. Inference verifies SHA-256 and reads safetensors into the fixed architecture. This is an integrity/compatibility check, not an independent security audit. Keep `runs/`, raw datasets, processed data, and weights out of Git unless a separately reviewed license and storage decision requires otherwise.

## Documentation

- [Implementation status and test evidence](docs/implementation-status.md)
- [Research roadmap and gates](docs/research-roadmap.md)
- [Targeted research landscape review](docs/research-landscape.md)
- [ADR 0001 — first experimental architecture](docs/architecture-decision-record-0001.md)
- [Dataset rights, provenance, filtering, and takedown](docs/dataset-policy.md)
- [Compute and cost assessment](docs/compute-assessment.md)
- [Training, checkpoint, and reproducibility guide](docs/training-and-reproducibility.md)
- [Evaluation protocol](docs/evaluation.md)
- [Model categories, security, and deployment boundary](docs/model-security-and-deployment.md)
- [Windows Studio / Research Mode integration plan](docs/windows-studio-integration-plan.md)
- [Model card template](docs/model-card-template.md)
