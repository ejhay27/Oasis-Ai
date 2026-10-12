# Compute and cost assessment

**Observed environment:** the Arena development sandbox on 2026-10-12. This is an observation of the coding environment only; it is not a measurement of the eventual Windows user's PC or a dedicated training machine.

## Inventory captured during this change

| Resource | Observation | Confidence/limits |
|---|---|---|
| OS / architecture | Linux x86_64 sandbox | Not the Windows desktop target. |
| Python | 3.11.2 | Confirmed. |
| CPU | Intel(R) Xeon(R) Processor @ 2.60GHz; 2 logical CPUs reported | Host is shared/virtualized; not a dedicated workstation benchmark. |
| System RAM | 4,131,278,848 bytes (about 3.85 GiB) | Host/container accounting may differ from production. |
| Storage | At the repository path: 21,834,924,032 bytes total; 20,696,428,544 bytes were free before the temporary local ML dependencies, and 11,860,897,792 bytes were free after they were installed in the ignored sandbox `.venv`. | Space changes over time; no dataset storage provisioned. The venv is excluded from repository snapshots. |
| GPU / accelerator | No `nvidia-smi` executable detected. The temporary project venv has PyTorch 2.14.1+cu130, but `torch.cuda.is_available()` is false; no CUDA device is available to the smoke run. Physical GPU model/memory remains **unknown**. | Absence of `nvidia-smi` alone does not prove there is no physical GPU. No accelerator training compatibility can be verified here. |
| Media tooling | `ffmpeg` and `ffprobe` not found on `PATH` | Actual source-video preprocessing cannot run here until installed. |
| Research dependencies | The base system Python initially lacked PyTorch, NumPy, Pillow, and safetensors. A local ignored `.venv` was provisioned with PyTorch 2.14.1+cu130, NumPy 2.4.6, Pillow 12.3.0, safetensors 0.8.0, and psutil 7.2.2 for a CPU-only synthetic smoke test. | This validates small software plumbing only. TensorBoard is not required; metrics use local JSONL. |

Run `oasis-research hardware --storage-path <target-path>` again before any training run. The current CLI records unknowns rather than filling them with guesses. The repository was not able to benchmark representative model throughput or peak GPU memory. A temporary two-step CPU run on generated random patterns verified training, interruption/resume, checkpoint registration, sampling, and evaluation code; all outputs were deleted and this is not evidence of real video learning. See [implementation evidence](implementation-status.md).

## Feasibility

This sandbox is suitable for code review, manifest/config unit tests, synthetic fixtures, and small CPU tests. With 3.8 GiB RAM and no verified accelerator/runtime, it is not a credible environment for developing a competitive general-purpose video model from scratch. A smaller `base_channels=8` configuration completed a two-step CPU software smoke test. That does not establish convergence on video, memory requirements of the default `base_channels=32` config, or any generation quality.

For any workstation or rented infrastructure, first measure one short, rights-approved benchmark on the exact config. Record GPU model and memory, system RAM, storage, framework/CUDA versions, throughput, peak allocation, and failure behavior. Estimate training duration from observed steady-state steps/sec with uncertainty and include validation/checkpoint overhead. Do not extrapolate from parameter count alone.

## Cost position

No cloud provider, region, instance type, storage plan, or current rate source was selected. **A monetary training estimate is therefore unknown**, not zero. This repository does not launch remote jobs or call a cloud API. Before a remote run, the owner must document:

- chosen GPU/instance and number of GPUs;
- measured GPU-hours and expected wall duration;
- actual dated hourly price source and whether discounts/taxes apply;
- storage, snapshots, logs, and data egress/transfer costs;
- total estimate and uncertainty/range;
- run shutdown/cleanup plan and explicit approval.

Never begin a costly remote training operation without explicit user authorization. Distributed training is not justified for the current small feasibility experiment.

## Memory and OOM behavior

The trainer records parameter count and a lower-bound estimate for parameter, gradient, and Adam state storage (the 16-bytes-per-parameter FP32 approximation); this deliberately excludes activations, temporary tensors, allocator overhead, data, and runtime workspaces. It is not a fit guarantee. Actual resource metrics are emitted only when the relevant backend/API is available.

CUDA OOM is recorded without silently changing the experiment. For a new run, possible explicit changes include lower spatial resolution, shorter clips, smaller batches plus gradient accumulation, supported mixed precision, gradient checkpointing (not implemented yet), or a smaller architecture. Record every changed setting in a new version-controlled config.
